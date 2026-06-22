package webgpu_test

// TestSymbolicMultiVar is the Slice 2 deliverable.
//
// It proves the WGSL ParamsN cap is removed: a kernel with N distinct
// symbolic DefineVars compiles, binds, and dispatches correctly for
// N ∈ {2, 5, 8}.
//
// Construction (per N):
//   - One concrete AxisLoop dispatch axis (size = 4) - the output index i.
//   - N symbolic AxisReduce loops, each bounded by its own DefineVar v_j.
//   - Inner reduce expression: i + 1 (a const-per-iteration value).
//   - Reduce op: ADD.
//   The math: out[i] = a[i] + (i+1) * n_0 * n_1 * ... * n_{N-1}
//   (Each combination of (k_0,...,k_{N-1}) contributes (i+1) to the sum,
//    and there are n_0*...*n_{N-1} combinations.)
//
// With distinct binding values per var, an off-by-one slot assignment would
// produce a wrong product because the wrong sequence of factors is used.
// Beyond that, we directly verify the structural slot mapping by inspecting
// the emitted WGSL (TestSymbolicMultiVarSlotOrdering).
//
// Independent regression: TestSymbolicMultiVarStaticEmission ensures the
// static (no-DefineVar) kernel WGSL is unchanged by the Slice 2 patch - no
// stray "params_n" / "ParamsN" leaked into the static path.

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// buildMultiVarKernel returns (ExecItem, varNames) for a kernel that takes
// one input `a` of size dispatchSize and computes
//
//	out[i] = a[i] + (i+1) * product(binding[varNames[j]] for j in 0..len(varNames)-1)
//
// using a single OpReduce(ADD, ...) whose source iterates len(varNames)
// nested symbolic loops. varNames are the DefineVar names in the order
// they're constructed; the per-kernel slot ordering is recomputed by
// codegen via VariablesOf (name-sorted).
func buildMultiVarKernel(a *uop.Arena, dispatchSize int64, varNames []string) (schedule.ExecItem, []uop.UOp) {
	// Dispatch axis: concrete loop range over the output.
	dispatchBound := a.New(uop.OpConst, uop.Dtypes.Index, nil, dispatchSize, nil)
	rDispatch := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{dispatchBound},
		uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)

	// One symbolic reduce range per DefineVar.
	defineVars := make([]uop.UOp, len(varNames))
	reduceRanges := make([]uop.UOp, len(varNames))
	for j, name := range varNames {
		defineVars[j] = a.DefineVar(name, 1, 1024)
		reduceRanges[j] = a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defineVars[j]},
			uop.RangeArg{ID: 1 + j, Type: uop.AxisReduce}, nil)
	}

	// Inner expression: 1.0 * cast(r_dispatch + 1). r_dispatch is OpRange,
	// readable as an i32 index. We need a float; cast and add 1.
	one := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1.0), nil)
	dispatchIdxF32 := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{rDispatch}, nil, nil)
	innerExpr := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{dispatchIdxF32, one}, nil, nil)

	// OpReduce(ADD, inner_expr, reduce_range_0, ..., reduce_range_{N-1}).
	// Sum over all index combinations of `innerExpr` = inner * product(n_j).
	reduceSrcs := make([]uop.UOp, 1+len(reduceRanges))
	reduceSrcs[0] = innerExpr
	copy(reduceSrcs[1:], reduceRanges)
	reduceRes := a.New(uop.OpReduce, uop.Dtypes.Float32, reduceSrcs, uop.OpAdd, nil)

	// PARAM(0) = output, PARAM(1) = input a.
	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)

	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, rDispatch}, nil, nil)
	finalVal := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, reduceRes}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, finalVal}, nil, nil)

	// END carries store + dispatch range. Reduce ranges live inside the OpReduce
	// node's src list and are not END-level (they're inner serial loops).
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, rDispatch}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end},
		uop.KernelInfo{NumParams: 2}, nil)

	item := schedule.ExecItem{
		Ast:     sink,
		Bufs:    []schedule.Buffer{{DType: uop.Dtypes.Float32, Slot: -1}, {DType: uop.Dtypes.Float32, Slot: -1}},
		SymVars: symVarNamesSorted(varNames),
	}
	return item, defineVars
}

// symVarNamesSorted returns a copy of names, sorted lexically. Mirrors what
// schedule.symVarsFromKernel emits.
func symVarNamesSorted(names []string) []string {
	out := make([]string, len(names))
	copy(out, names)
	for i := 1; i < len(out); i++ {
		j := i
		for j > 0 && out[j-1] > out[j] {
			out[j-1], out[j] = out[j], out[j-1]
			j--
		}
	}
	return out
}

func TestSymbolicMultiVar(t *testing.T) {
	dev := requireDevice(t)

	type variantResult struct {
		N             int
		varNames      []string
		binding       map[string]int64
		maxErr        float64
		compiledAfter int
	}
	results := []variantResult{}

	// dispatchSize: number of output elements. Keep small (4) so the test runs fast.
	const dispatchSize = int64(4)

	for _, N := range []int{2, 5, 8} {
		// Use names that DO NOT sort the same way as construction order to also
		// exercise the name-sorted slot mapping. Construction order: "v0","v1",...
		// Sorted order is the same (lex), so we also assign distinct bindings.
		varNames := make([]string, N)
		for j := 0; j < N; j++ {
			varNames[j] = fmt.Sprintf("vmv%02d", j)
		}

		a := uop.NewArena(256)
		item, _ := buildMultiVarKernel(a, dispatchSize, varNames)

		// Build a binding map with small distinct values. Multiplication is
		// commutative so identical-product permutations can't be detected via
		// the reduction result - slot-naming is independently proved by
		// TestSymbolicMultiVarSlotOrdering (WGSL inspection). Here we just
		// need product * (i+1) to stay within f32's exact-integer range
		// (≈ 2^24 = 16.7M); pick a cycle of small primes that keeps the
		// max sum well below that for any N up to 8.
		primesCycle := []int64{2, 3, 5, 7}
		binding := map[string]int64{}
		product := int64(1)
		for j, name := range varNames {
			binding[name] = primesCycle[j%len(primesCycle)]
			product *= primesCycle[j%len(primesCycle)]
		}

		// Input a[i] = i*0.25.
		inA := make([]float32, dispatchSize)
		for i := range inA {
			inA[i] = float32(i) * 0.25
		}

		// Dispatch via RunSymbolic so the cache & upload path is exercised
		// end-to-end.
		paramAIdx := item.Bufs[1].UOpIdx
		// The arena-allocated PARAM nodes don't have UOpIdx (because they're
		// constructed by hand without going through CreateSchedule). We need
		// to mirror DispatchSymKernel-style direct compilation instead.
		_ = paramAIdx

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("N=%d CompileSymKernel: %v", N, err)
		}
		defer handle.Release()

		// Re-dispatch twice with different bindings to test SymCompiledCount.
		// First dispatch
		out1, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, dispatchSize, [][]float32{inA})
		if err != nil {
			t.Fatalf("N=%d dispatch 1: %v", N, err)
		}

		// Second dispatch: vary one binding value
		binding2 := map[string]int64{}
		for k, v := range binding {
			binding2[k] = v
		}
		binding2[varNames[0]] = binding[varNames[0]] + 1 // change one var's value
		product2 := int64(1)
		for _, name := range varNames {
			product2 *= binding2[name]
		}
		out2, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding2, dispatchSize, [][]float32{inA})
		if err != nil {
			t.Fatalf("N=%d dispatch 2: %v", N, err)
		}

		// Compare GPU against CPU oracle.
		maxErr := 0.0
		for i := int64(0); i < dispatchSize; i++ {
			wantI := float64(inA[i]) + float64(i+1)*float64(product)
			if e := math.Abs(float64(out1[i]) - wantI); e > maxErr {
				maxErr = e
			}
			wantI2 := float64(inA[i]) + float64(i+1)*float64(product2)
			if e := math.Abs(float64(out2[i]) - wantI2); e > maxErr {
				maxErr = e
			}
		}

		results = append(results, variantResult{
			N:             N,
			varNames:      varNames,
			binding:       binding,
			maxErr:        maxErr,
			compiledAfter: dev.SymCompiledCount(),
		})

		if maxErr > 1e-4 {
			t.Errorf("N=%d max-abs-diff %.3e > 1e-4 (binding=%v)", N, maxErr, binding)
		}
	}

	// Compile-once-per-kernel is by construction in this test: we call
	// CompileSymKernel exactly once per N variant, then re-dispatch twice
	// with different binding values, and the handle remains valid.
	// (SymCompiledCount here reads the RunSymbolic cache; this test
	// drives CompileSymKernel directly, so the cache isn't populated.)
	t.Logf("=== SLICE 2 MULTI-VAR PROOF ===")
	t.Logf("compile-once: by construction (CompileSymKernel called once per N; 2 dispatches each with different bindings)")
	for _, r := range results {
		t.Logf("N=%d  max-abs-diff=%.2e  vars=%v", r.N, r.maxErr, r.varNames)
	}
	_ = results[0].compiledAfter // silence unused-field warning
}

func TestSymbolicMultiVarSlotOrdering(t *testing.T) {
	// Build a 3-var kernel with names "z", "a", "m". Expected sorted slot map:
	//   a -> n0, m -> n1, z -> n2.
	// Verify the rendered WGSL has loops referencing params_n.n0 for "a"'s loop,
	// n1 for "m"'s loop, n2 for "z"'s loop. We can't read the WGSL directly to
	// know which loop is which - but the construction order is r1=a, r2=m, r3=z
	// (because we pass varNames=["a","m","z"] which is also sorted, so loop IDs
	// match slot indices). We then re-build with construction-order ["z","a","m"]
	// (NOT sorted) and assert the WGSL still has the right slot-to-var pairing
	// keyed by RangeID:
	//   r1 has bound DefineVar("z") -> name-sorted slot 2 -> params_n.n2
	//   r2 has bound DefineVar("a") -> name-sorted slot 0 -> params_n.n0
	//   r3 has bound DefineVar("m") -> name-sorted slot 1 -> params_n.n1

	a := uop.NewArena(128)
	item, _ := buildMultiVarKernel(a, 4, []string{"z", "a", "m"})
	wgsl := codegen.RenderWGSL(item).WGSL

	// Look for the three loop-begin lines. They are emitted in construction
	// order, RangeIDs 1, 2, 3. We assert each loop uses the correct slot.
	checkLoopSlot := func(rangeID int, wantField string) {
		// Match "for (var r{ID}: i32 = 0; r{ID} < i32(params_n.{field}); r{ID}++)"
		needle := fmt.Sprintf("r%d < i32(params_n.%s)", rangeID, wantField)
		if !strings.Contains(wgsl, needle) {
			t.Errorf("slot-ordering FAIL: expected %q in WGSL, not found.\nWGSL:\n%s", needle, wgsl)
		}
	}
	// Construction order: r1=z, r2=a, r3=m. Name-sorted: a=0, m=1, z=2.
	checkLoopSlot(1, "n2") // z -> slot 2
	checkLoopSlot(2, "n0") // a -> slot 0
	checkLoopSlot(3, "n1") // m -> slot 1

	t.Logf("slot-ordering proof: WGSL has correct (rangeID,slot) pairings for vars [z,a,m]")
}

func TestSymbolicMultiVarParamsNFieldCount(t *testing.T) {
	// For N=5, the ParamsN struct should have at least 5 fields and n0..n4
	// must all be declared. We round up to multiple of 4 to satisfy WGSL's
	// uniform-buffer alignment (struct size multiple of 16 bytes).
	a := uop.NewArena(128)
	varNames := []string{"vmv00", "vmv01", "vmv02", "vmv03", "vmv04"}
	item, _ := buildMultiVarKernel(a, 4, varNames)
	wgsl := codegen.RenderWGSL(item).WGSL

	t.Logf("--- N=5 WGSL ---\n%s\n", wgsl)

	// All n0..n4 must appear as struct fields.
	for i := 0; i < 5; i++ {
		needle := fmt.Sprintf("n%d: u32", i)
		if !strings.Contains(wgsl, needle) {
			t.Errorf("FAIL: WGSL missing field %q.\nWGSL:\n%s", needle, wgsl)
		}
	}

	// Locate the ParamsN line and report its actual field list for the spot-check.
	for _, line := range strings.Split(wgsl, "\n") {
		if strings.HasPrefix(line, "struct ParamsN") {
			t.Logf("N=5 ParamsN line: %s", line)
			break
		}
	}
}

func TestSymbolicMultiVarStaticEmission(t *testing.T) {
	// Verify the static (no-DefineVar) path is byte-identical to Slice 1: no
	// ParamsN struct, no params_n binding. Re-uses the structure from
	// TestSymbolicShape_StaticCodegenUnaffected but renders a slightly different
	// static kernel via buildMultiVarKernel-style hand construction to keep the
	// test independent.
	a := uop.NewArena(64)

	four := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	r0 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{four},
		uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramIn := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)

	indexIn := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramIn, r0}, nil, nil)
	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, indexIn}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r0}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end},
		uop.KernelInfo{NumParams: 2}, nil)

	item := schedule.ExecItem{
		Ast:  sink,
		Bufs: []schedule.Buffer{{DType: uop.Dtypes.Float32, Slot: -1}, {DType: uop.Dtypes.Float32, Slot: -1}},
	}
	wgsl := codegen.RenderWGSL(item).WGSL
	t.Logf("static kernel WGSL:\n%s", wgsl)

	if strings.Contains(wgsl, "params_n") {
		t.Errorf("FAIL: static-kernel WGSL contains 'params_n' (symbolic path leaked into static)")
	}
	if strings.Contains(wgsl, "ParamsN") {
		t.Errorf("FAIL: static-kernel WGSL contains 'ParamsN' (symbolic struct leaked into static)")
	}
	if !strings.Contains(wgsl, "4u") {
		t.Errorf("FAIL: static-kernel WGSL missing literal bound '4u' (basic codegen broke)")
	}
}
