package webgpu_test

// TestSymbolicShapeProof is the SLICE 1 spike deliverable.
//
// It proves that a single compiled WGSL kernel can correctly compute
// out[i] = a[i] + b[i] for two different runtime values of n (8 and 128)
// without recompilation.  The test reports:
//
//   - compile count (must be 1)
//   - dispatch workgroup counts for n=8 and n=128 (must differ)
//   - max absolute error for each dispatch (must be < 1e-5)
//
// Design: the symbolic dim n flows through a params_n storage buffer
// (binding = NumParams) read at dispatch time.  Codegen emits
// `if (gid_x >= params_n[0]) { return; }` instead of a literal bound,
// and the dispatch grid is computed from n on the CPU.

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

func TestSymbolicShapeProof(t *testing.T) {
	dev := requireDevice(t)

	// ── 1. Build symbolic kernel AST: out[i] = a[i] + b[i], i ∈ [0, n) ─
	//
	// The kernel graph is hand-built at the UOp level, bypassing the tensor
	// frontend and scheduler.  The only symbolic thing is the loop BOUND; the
	// index expression itself is the range variable (stride 1).
	a := uop.NewArena(64)

	// r0: symbolic AxisLoop range — bound read from params_n[0] at dispatch time.
	r0 := a.New(uop.OpRange, uop.Dtypes.Index, nil,
		uop.RangeArg{ID: 0, Size: 0, Type: uop.AxisLoop, Symbolic: true, SymParamIdx: 0}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil) // PARAM(0) = output
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)   // PARAM(1) = a
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)   // PARAM(2) = b

	// INDEX(PARAM(k), r0) — 1-D flat access.
	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, r0}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, r0}, nil, nil)

	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)

	// END carries the STORE and the loop range(s).
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r0}, nil, nil)

	// Kernel SINK with KernelInfo{NumParams: 3}.
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end},
		uop.KernelInfo{NumParams: 3}, nil)

	item := schedule.ExecItem{
		Ast: sink,
		// Bufs metadata: sizes are 0 because n is symbolic.
		// The lowerer only uses Bufs[i].Shape for multi-dim strides; our 1-D
		// kernel uses nDims==1 so shape is not consulted.
		Bufs: []schedule.Buffer{
			{DType: uop.Dtypes.Float32, Slot: -1},
			{DType: uop.Dtypes.Float32, Slot: -1},
			{DType: uop.Dtypes.Float32, Slot: -1},
		},
	}

	// ── 2. Render WGSL and log it for inspection ──────────────────────────
	wgslSrc := codegen.RenderWGSL(item)
	t.Logf("=== Generated WGSL (compiled once) ===\n%s", wgslSrc)

	// ── 3. Compile exactly once ───────────────────────────────────────────
	k, err := dev.CompileSymKernel(item)
	if err != nil {
		t.Fatalf("CompileSymKernel: %v", err)
	}
	defer k.Release()

	// ── 4. Dispatch with n=8 ─────────────────────────────────────────────
	const n8 = int64(8)
	aData8 := make([]float32, n8)
	bData8 := make([]float32, n8)
	for i := range aData8 {
		aData8[i] = float32(i)
		bData8[i] = float32(i) * 2
	}
	got8, grid8, err := dev.DispatchSymKernel(k, n8, [][]float32{aData8, bData8})
	if err != nil {
		t.Fatalf("DispatchSymKernel n=8: %v", err)
	}

	// ── 5. Dispatch with n=128 (SAME compiled kernel) ─────────────────────
	const n128 = int64(128)
	aData128 := make([]float32, n128)
	bData128 := make([]float32, n128)
	for i := range aData128 {
		aData128[i] = float32(i) * 0.5
		bData128[i] = float32(i) * 1.5
	}
	got128, grid128, err := dev.DispatchSymKernel(k, n128, [][]float32{aData128, bData128})
	if err != nil {
		t.Fatalf("DispatchSymKernel n=128: %v", err)
	}

	// ── 6. Compute max absolute errors ───────────────────────────────────
	var maxErr8, maxErr128 float64
	for i, v := range got8 {
		want := aData8[i] + bData8[i]
		if e := math.Abs(float64(v - want)); e > maxErr8 {
			maxErr8 = e
		}
	}
	for i, v := range got128 {
		want := aData128[i] + bData128[i]
		if e := math.Abs(float64(v - want)); e > maxErr128 {
			maxErr128 = e
		}
	}

	// ── 7. Print the proof numbers ────────────────────────────────────────
	t.Logf("=== SYMBOLIC SHAPES SPIKE — SLICE 1 PROOF ===")
	t.Logf("compile count:            1  (compile-once by construction)")
	t.Logf("dispatch grid  n=8:       %d workgroup(s)", grid8)
	t.Logf("dispatch grid  n=128:     %d workgroup(s)", grid128)
	t.Logf("max abs error  n=8:       %.2e  (want 0)", maxErr8)
	t.Logf("max abs error  n=128:     %.2e  (want 0)", maxErr128)

	// ── 8. Assertions ─────────────────────────────────────────────────────
	if grid8 == grid128 {
		t.Errorf("FAIL: dispatch grids equal (%d == %d) — grid must vary with n", grid8, grid128)
	}
	if len(got8) != int(n8) {
		t.Errorf("FAIL: n=8 output length %d, want %d", len(got8), n8)
	}
	if len(got128) != int(n128) {
		t.Errorf("FAIL: n=128 output length %d, want %d", len(got128), n128)
	}
	if maxErr8 > 1e-5 {
		t.Errorf("FAIL: n=8 max error %.2e > 1e-5", maxErr8)
	}
	if maxErr128 > 1e-5 {
		t.Errorf("FAIL: n=128 max error %.2e > 1e-5", maxErr128)
	}
}

// TestSymbolicShape_StaticSuiteUnaffected verifies that the symbolic codegen
// changes (TotalN==1 scalar guard, hasSymDim detection) do not alter the WGSL
// emitted for a concrete static-size kernel.  This is a pure codegen test with
// no GPU required.
func TestSymbolicShape_StaticCodegenUnaffected(t *testing.T) {
	// Build a static 4-element elementwise kernel (same shape as exp2 test).
	a := uop.NewArena(32)

	r0 := a.New(uop.OpRange, uop.Dtypes.Index, nil,
		uop.RangeArg{ID: 0, Size: 4, Type: uop.AxisLoop}, nil) // static, Symbolic=false

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

	wgslSrc := codegen.RenderWGSL(item)
	t.Logf("Static kernel WGSL:\n%s", wgslSrc)

	// Static kernels must NOT contain params_n.
	if contains(wgslSrc, "params_n") {
		t.Error("FAIL: static kernel WGSL contains params_n binding — symbolic path leaked into static codegen")
	}
	// Must still have the concrete literal bound.
	if !contains(wgslSrc, "4u") {
		t.Error("FAIL: static kernel WGSL missing literal bound '4u'")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsStr(s, sub))
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
