package webgpu_test

// Slice 7b — C1/C2/C3/C4 value-oracle tests.
//
// These tests hand-build symbolic kernels at the arena level (mirroring
// buildMultiVarKernel in symbolic_multivar_test.go) so they exercise the
// lowerer's globalStrides + trailingProduct + per-(dim,level) stride math
// for symbolic dims at non-outermost positions and for two-sym-dim outputs.
//
// They run on the real GPU via DispatchSymKernelWithBinding and compare
// against a CPU oracle (max-abs-diff = 0 target on integer-arithmetic
// inputs that stay within f32's exact-integer range).

import (
	"fmt"
	"math"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ────────────────────────────────────────────────────────────────

// flatIndex2 builds a UOp expression for row*stride + col over the given range
// uops, where stride is a UOp (concrete OpConst or symbolic OpDefineVar).
func flatIndex2(a *uop.Arena, row, stride, col uop.UOp) uop.UOp {
	m := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{row, stride}, nil, nil)
	return a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{m, col}, nil, nil)
}

// constU returns an OpConst(int64) node in Dtypes.Index.
func constU(a *uop.Arena, v int64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil)
}

// twoSymElemBuf returns an output Buffer whose size resolves to mul0*var0 *
// mul1*var1 at runtime via SymDimMul/SymDimVar (parallel to Shape's sym
// positions). bufIdx is a stable per-test arena index slot.
func twoSymElemBuf(bufIdx uint32, dimVars [2]string, dimMuls [2]int64) schedule.Buffer {
	return schedule.Buffer{
		UOpIdx:    bufIdx,
		Shape:     []int64{0, 0},
		DType:     uop.Dtypes.Float32,
		Slot:      -1,
		SymDimMul: []int64{dimMuls[0], dimMuls[1]},
		SymDimVar: []string{dimVars[0], dimVars[1]},
	}
}

func mixedSymElemBuf(bufIdx uint32, shape []int64, symVar string, symMul int64) schedule.Buffer {
	muls := []int64{}
	vars := []string{}
	for _, s := range shape {
		if s == 0 {
			muls = append(muls, symMul)
			vars = append(vars, symVar)
		}
	}
	return schedule.Buffer{
		UOpIdx:    bufIdx,
		Shape:     shape,
		DType:     uop.Dtypes.Float32,
		Slot:      -1,
		SymDimMul: muls,
		SymDimVar: vars,
	}
}

// ── C1: [n, m] elementwise ────────────────────────────────────────────────

// buildC1Kernel returns (item, varNames, defVars) for c[i,j] = a[i,j] + b[i,j]
// where c, a, b are all shape [n, m] (both symbolic). loopRanges = [rN, rM].
// Inputs and output are accessed via hand-flattened single-dim OpIndex
// expressions so emitIndex (multi-dim stride math) is not exercised — the
// test isolates globalStrides + trailingProduct + per-(dim,level) strides.
func buildC1Kernel(a *uop.Arena, varN, varM string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	defM := a.DefineVar(varM, 1, 100)
	rN := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	rM := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defM}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	flatIdx := flatIndex2(a, rN, defM, rM)
	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, flatIdx}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, flatIdx}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, rN, rM}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	// SymVars sorted (alphabetical by var name).
	syms := []string{varN, varM}
	if syms[0] > syms[1] {
		syms[0], syms[1] = syms[1], syms[0]
	}

	return schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			twoSymElemBuf(paramOut.Index(), [2]string{varN, varM}, [2]int64{1, 1}),
			twoSymElemBuf(paramA.Index(), [2]string{varN, varM}, [2]int64{1, 1}),
			twoSymElemBuf(paramB.Index(), [2]string{varN, varM}, [2]int64{1, 1}),
		},
		SymVars: syms,
	}
}

func TestC1Elementwise(t *testing.T) {
	dev := requireDevice(t)

	cases := []struct{ n, m int64 }{{3, 5}, {5, 3}, {7, 11}}
	type result struct{ n, m int64; maxErr float64; bound string }
	var results []result

	for _, c := range cases {
		a := uop.NewArena(128)
		item := buildC1Kernel(a, "cn", "cm")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("C1 n=%d m=%d CompileSymKernel: %v", c.n, c.m, err)
		}
		defer handle.Release()

		// Inputs filled with distinct row+col patterns.
		inA := make([]float32, c.n*c.m)
		inB := make([]float32, c.n*c.m)
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < c.m; j++ {
				inA[i*c.m+j] = float32(i*10 + j)
				inB[i*c.m+j] = float32(i*100 + j*7)
			}
		}

		binding := map[string]int64{"cn": c.n, "cm": c.m}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, c.n*c.m, [][]float32{inA, inB})
		if err != nil {
			t.Fatalf("C1 n=%d m=%d dispatch: %v", c.n, c.m, err)
		}

		// Row-by-row check: assert each cell distinctly. If STOP-2 regressed,
		// rows would collapse and the (i*m+j) discriminator would catch it.
		maxErr := 0.0
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < c.m; j++ {
				want := inA[i*c.m+j] + inB[i*c.m+j]
				got := out[i*c.m+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}
		// WGSL spot-check on the first case: print the bounds-check line.
		wgsl := codegen.RenderWGSL(item).WGSL
		boundLine := ""
		for _, ln := range splitLines(wgsl) {
			if containsAll(ln, []string{"gid_x >="}) {
				boundLine = ln
				break
			}
		}
		t.Logf("C1 n=%d m=%d  max-abs-diff=%.3e  bound=%s", c.n, c.m, maxErr, boundLine)
		if maxErr != 0 {
			// Row collapse diagnostic: print first 2 rows for first failure.
			for i := int64(0); i < min64(2, c.n); i++ {
				t.Logf("  row %d got: %v", i, out[i*c.m:(i+1)*c.m])
			}
			t.Errorf("C1 n=%d m=%d max-abs-diff=%g (expect 0)", c.n, c.m, maxErr)
		}
		results = append(results, result{c.n, c.m, maxErr, boundLine})
	}

	t.Logf("=== SLICE 7b C1 PROOF ===")
	for _, r := range results {
		t.Logf("[n=%d, m=%d] max-abs-diff=%.3e  bound=%s", r.n, r.m, r.maxErr, r.bound)
	}
}

// ── C2: [4, n] elementwise (sym non-outermost) ─────────────────────────────

// buildC2Kernel returns a kernel for c[i,j] = a[i,j] + b[i,j] where shape is
// [4, n] — n is the inner symbolic dim. loopRanges = [r_4 concrete, rN sym].
// This exercises the STOP-2 regression case: globalStrides[i] must NOT be 0
// for the dim left of a sym dim; the stride for r_4 must be sym n.
func buildC2Kernel(a *uop.Arena, varN string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	rNConst := constU(a, 4)
	r_4 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{rNConst}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	rN := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	flatIdx := flatIndex2(a, r_4, defN, rN)
	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, flatIdx}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, flatIdx}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r_4, rN}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	return schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			mixedSymElemBuf(paramOut.Index(), []int64{4, 0}, varN, 1),
			mixedSymElemBuf(paramA.Index(), []int64{4, 0}, varN, 1),
			mixedSymElemBuf(paramB.Index(), []int64{4, 0}, varN, 1),
		},
		SymVars: []string{varN},
	}
}

func TestC2ElementwiseSymInner(t *testing.T) {
	dev := requireDevice(t)

	cases := []int64{3, 5, 7, 11}
	type result struct{ n int64; maxErr float64; bound string; rows [][]float32 }
	var results []result

	for _, n := range cases {
		a := uop.NewArena(128)
		item := buildC2Kernel(a, "cn2")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("C2 n=%d CompileSymKernel: %v", n, err)
		}
		defer handle.Release()

		inA := make([]float32, 4*n)
		inB := make([]float32, 4*n)
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				// Row-discriminating values: row*1000 + col makes row-collapse
				// readily detectable. STOP-2 regression would produce all rows
				// identical.
				inA[i*n+j] = float32(i*1000 + j)
				inB[i*n+j] = float32(i*7 + j*13)
			}
		}

		binding := map[string]int64{"cn2": n}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, 4*n, [][]float32{inA, inB})
		if err != nil {
			t.Fatalf("C2 n=%d dispatch: %v", n, err)
		}

		maxErr := 0.0
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				want := inA[i*n+j] + inB[i*n+j]
				got := out[i*n+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}

		wgsl := codegen.RenderWGSL(item).WGSL
		boundLine := ""
		for _, ln := range splitLines(wgsl) {
			if containsAll(ln, []string{"gid_x >="}) {
				boundLine = ln
				break
			}
		}

		// Capture row-by-row outputs for the report.
		rows := make([][]float32, 4)
		for i := int64(0); i < 4; i++ {
			row := make([]float32, n)
			copy(row, out[i*n:(i+1)*n])
			rows[i] = row
		}

		t.Logf("C2 n=%d  max-abs-diff=%.3e  bound=%s", n, maxErr, boundLine)
		if maxErr != 0 {
			t.Errorf("C2 n=%d max-abs-diff=%g (expect 0)", n, maxErr)
			for i := range rows {
				t.Logf("  row %d: %v", i, rows[i])
			}
		}

		results = append(results, result{n, maxErr, boundLine, rows})
	}

	t.Logf("=== SLICE 7b C2 PROOF (row-discriminating; STOP-2 regression guard) ===")
	for _, r := range results {
		t.Logf("[4, n=%d] max-abs-diff=%.3e", r.n, r.maxErr)
	}
	// Row-by-row paste of the first case so the report can show non-collapsed rows.
	if len(results) > 0 {
		t.Logf("First case row-by-row (n=%d):", results[0].n)
		for i, row := range results[0].rows {
			t.Logf("  row %d: %v", i, row)
		}
	}
}

// ── C3: pad inner axis [n, 4] → [n, 4+k] ───────────────────────────────────

// buildC3Kernel returns a kernel for c[i,j] = (j < 4) ? a[i, j] : 0 over a
// [n, 4+k] output. n and k are symbolic. loopRanges = [rN sym, r_inner sym=4+k].
// The pad fills out-of-source values with 0.
//
// This is Slice 5's deferred test (B): pad on inner axis of [n, 4] → [n, 4+k].
// Pre-Slice-7b the bounds check `gid_x >= n` (positional first-sym pick)
// truncated 4+k-wide rows; Slice 7b's trailingProduct produces `n * (4+k)`.
func buildC3Kernel(a *uop.Arena, varN, varK string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	defK := a.DefineVar(varK, 0, 100)

	// Inner dim bound = 4 + k (an ALU bound). The Affine encoding on the
	// output buffer captures dim-1 = 1*k + 4.
	four := constU(a, 4)
	innerBound := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{defK, four}, nil, nil)

	rN := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	rInner := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{innerBound}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)

	// Load a[i, j] for j < 4 only. Source flat index = i*4 + j (j masked).
	// We compute source idx = i * 4 + min(j, 3) and then mask with a where j>=4.
	// Simpler: load a[i*4 + j] for any j < (4+k) but only USE it where j < 4.
	// The shader will compute a[i*4 + j] for j in [0, 4+k); we mask the store
	// result with a Where: (j < 4) ? a[i*4+j] : 0.
	// But for j >= 4 the address i*4 + j ≥ i*4 + 4 — past the row end. We must
	// either clamp the address or do the mask before reading. Tinygrad clamps
	// via select() inside the body. Use select(0, a[i*4 + min(j,3)], j<4) so the
	// read is always in-bounds.
	// Clamp source j via Mod 4 so the read is in-bounds even when j>=4
	// (Where masks the result to 0 in that case; WGSL evaluates both Where
	// branches so the address must always be safe).
	clampJ := a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{rInner, four}, nil, nil)
	srcFlat := flatIndex2(a, rN, four, clampJ)
	loadA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, srcFlat}, nil, nil)

	// where(j < 4, loadA, 0)
	zeroF := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0), nil)
	cmpLt4 := a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{rInner, four}, nil, nil)
	selected := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cmpLt4, loadA, zeroF}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, selected}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, rN, rInner}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 2}, nil)

	// Output buffer dim 0 = n, dim 1 = 4+k. Encode via SymDimAffine for dim 1.
	outBuf := schedule.Buffer{
		UOpIdx: paramOut.Index(),
		Shape:  []int64{0, 0},
		DType:  uop.Dtypes.Float32,
		Slot:   -1,
		SymDimAffine: []schedule.SymDimAffineEntry{
			{Terms: []uop.AffineTerm{{Mul: 1, VarName: varN}}, Offset: 0},
			{Terms: []uop.AffineTerm{{Mul: 1, VarName: varK}}, Offset: 4},
		},
	}
	inBuf := schedule.Buffer{
		UOpIdx:    paramA.Index(),
		Shape:     []int64{0, 4},
		DType:     uop.Dtypes.Float32,
		Slot:      -1,
		SymDimMul: []int64{1},
		SymDimVar: []string{varN},
	}

	// SymVars sorted by name.
	syms := []string{varN, varK}
	if syms[0] > syms[1] {
		syms[0], syms[1] = syms[1], syms[0]
	}

	return schedule.ExecItem{
		Ast:     sink,
		Bufs:    []schedule.Buffer{outBuf, inBuf},
		SymVars: syms,
	}
}

func TestC3PadInner(t *testing.T) {
	dev := requireDevice(t)

	cases := []struct{ n, k int64 }{{3, 2}, {5, 1}, {8, 4}}
	type result struct{ n, k int64; maxErr float64; bound string }
	var results []result

	for _, c := range cases {
		a := uop.NewArena(256)
		item := buildC3Kernel(a, "c3n", "c3k")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("C3 n=%d k=%d CompileSymKernel: %v", c.n, c.k, err)
		}
		defer handle.Release()

		// Source: [n, 4] filled with row*100 + col so each cell is distinct.
		inA := make([]float32, c.n*4)
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < 4; j++ {
				inA[i*4+j] = float32(i*100 + j)
			}
		}

		binding := map[string]int64{"c3n": c.n, "c3k": c.k}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, c.n*(4+c.k), [][]float32{inA})
		if err != nil {
			t.Fatalf("C3 n=%d k=%d dispatch: %v", c.n, c.k, err)
		}

		maxErr := 0.0
		W := int64(4 + c.k)
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < W; j++ {
				var want float32
				if j < 4 {
					want = inA[i*4+j]
				} else {
					want = 0
				}
				got := out[i*W+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}

		wgsl := codegen.RenderWGSL(item).WGSL
		boundLine := ""
		for _, ln := range splitLines(wgsl) {
			if containsAll(ln, []string{"gid_x >="}) {
				boundLine = ln
				break
			}
		}
		t.Logf("C3 n=%d k=%d  max-abs-diff=%.3e  bound=%s", c.n, c.k, maxErr, boundLine)
		if maxErr != 0 {
			t.Errorf("C3 n=%d k=%d max-abs-diff=%g (expect 0)", c.n, c.k, maxErr)
		}
		results = append(results, result{c.n, c.k, maxErr, boundLine})
	}

	t.Logf("=== SLICE 7b C3 PROOF (Slice 5 deferred test B) ===")
	for _, r := range results {
		t.Logf("[n=%d, 4+k=%d] max-abs-diff=%.3e", r.n, 4+r.k, r.maxErr)
	}
}

// ── C4: permute then op ([n, 4] → [4, n]) ──────────────────────────────────

// buildC4Kernel computes c[i, j] = a[j, i] + b[i, j] where:
//   - a is shape [n, 4]
//   - b is shape [4, n]
//   - c is shape [4, n]
//
// The permute lands a's symbolic dim non-outermost in the output addressing.
// loopRanges = [r_4 concrete, rN sym] (output dim order). Per-thread:
//   r_4 = i, rN = j
//   load a at a-flat = i + j*4    (i.e. a[j, i] = a[j*4 + i] but the load uses
//                                   j as inner and i as outer for the source)
//   load b at b-flat = i*n + j
//   store at out-flat = i*n + j   (same as b's layout)
func buildC4Kernel(a *uop.Arena, varN string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	r_4 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{constU(a, 4)}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	rN := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	// a[j, i] = a_flat[j*4 + i] where a has shape [n, 4]
	aIdx := flatIndex2(a, rN, constU(a, 4), r_4)
	loadA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, aIdx}, nil, nil)

	// b[i, j] = b_flat[i*n + j] where b has shape [4, n]
	bIdx := flatIndex2(a, r_4, defN, rN)
	loadB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, bIdx}, nil, nil)

	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{loadA, loadB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r_4, rN}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	return schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			mixedSymElemBuf(paramOut.Index(), []int64{4, 0}, varN, 1),
			mixedSymElemBuf(paramA.Index(), []int64{0, 4}, varN, 1),
			mixedSymElemBuf(paramB.Index(), []int64{4, 0}, varN, 1),
		},
		SymVars: []string{varN},
	}
}

func TestC4PermutePlus(t *testing.T) {
	dev := requireDevice(t)

	for _, n := range []int64{3, 5} {
		a := uop.NewArena(128)
		item := buildC4Kernel(a, "c4n")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("C4 n=%d CompileSymKernel: %v", n, err)
		}
		defer handle.Release()

		// a[n, 4]: a[r,c] = r*10 + c
		inA := make([]float32, n*4)
		for r := int64(0); r < n; r++ {
			for c := int64(0); c < 4; c++ {
				inA[r*4+c] = float32(r*10 + c)
			}
		}
		// b[4, n]: b[r,c] = r*100 + c
		inB := make([]float32, 4*n)
		for r := int64(0); r < 4; r++ {
			for c := int64(0); c < n; c++ {
				inB[r*n+c] = float32(r*100 + c)
			}
		}

		binding := map[string]int64{"c4n": n}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, 4*n, [][]float32{inA, inB})
		if err != nil {
			t.Fatalf("C4 n=%d dispatch: %v", n, err)
		}

		maxErr := 0.0
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				// want = a[j, i] + b[i, j]
				want := inA[j*4+i] + inB[i*n+j]
				got := out[i*n+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}

		wgsl := codegen.RenderWGSL(item).WGSL
		boundLine := ""
		for _, ln := range splitLines(wgsl) {
			if containsAll(ln, []string{"gid_x >="}) {
				boundLine = ln
				break
			}
		}
		t.Logf("C4 n=%d  max-abs-diff=%.3e  bound=%s", n, maxErr, boundLine)
		if maxErr != 0 {
			t.Errorf("C4 n=%d max-abs-diff=%g (expect 0)", n, maxErr)
		}
	}
}

// ── helpers ────────────────────────────────────────────────────────────────

func splitLines(s string) []string {
	out := []string{}
	cur := ""
	for _, c := range s {
		if c == '\n' {
			out = append(out, cur)
			cur = ""
		} else {
			cur += string(c)
		}
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

func containsAll(s string, subs []string) bool {
	for _, sub := range subs {
		found := false
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// Avoid unused-import nags in the rare config where fmt isn't used directly.
var _ = fmt.Sprintf
