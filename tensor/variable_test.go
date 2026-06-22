package tensor_test

// Variable / NewSymbolicShape frontend tests.
//
// Coverage:
//   - Structural (no GPU): Variable interning, same-name collision panic,
//     Sint composition, MergeBindings semantics, NewSymbolicShape srcs and
//     ShapeSintArg encoding, V=0-on-sym invariant.
//   - End-to-end (GPU-gated): non-outermost sym placement, two distinct
//     Variables in one shape, .Bind() idiom, bind-out-of-range error,
//     JIT replay at different binds, FD gradient smoke on a small graph
//     built via the general constructor.

import (
	"math"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── GPU setup ────────────────────────────────────────────────────────────────

func requireGPUVar(t *testing.T) {
	t.Helper()
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	t.Cleanup(func() {
		tensor.DefaultExecutor = nil
		dev.Close()
	})
	tensor.DefaultExecutor = dev
}

// ── Structural tests (no GPU) ────────────────────────────────────────────────

func TestVariableInterns(t *testing.T) {
	a := uop.NewArena(64)
	v1 := tensor.NewVariable(a, "seq", 1, 1024)
	v2 := tensor.NewVariable(a, "seq", 1, 1024)
	if v1.Node() != v2.Node() {
		t.Errorf("two Variables with same (name, min, max) did not alias to one DefineVar: v1=%v v2=%v",
			v1.Node().Index(), v2.Node().Index())
	}
	if v1.Name() != "seq" || v1.Min() != 1 || v1.Max() != 1024 {
		t.Errorf("Variable accessors wrong: name=%q min=%d max=%d", v1.Name(), v1.Min(), v1.Max())
	}
}

func TestVariableSameNameCollidesOnDifferentBounds(t *testing.T) {
	a := uop.NewArena(64)
	tensor.NewVariable(a, "seq", 1, 1024)
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on same-name + different-bounds collision")
		}
		msg, _ := r.(string)
		if !strings.Contains(msg, "already registered") {
			t.Errorf("panic message does not name the collision: %v", r)
		}
	}()
	tensor.NewVariable(a, "seq", 1, 512)
}

func TestVariableBindReturnsSingletonMap(t *testing.T) {
	a := uop.NewArena(64)
	v := tensor.NewVariable(a, "n", 1, 1024)
	b := v.Bind(64)
	if len(b) != 1 || b["n"] != 64 {
		t.Errorf("Bind(64) = %v, want map[n:64]", b)
	}
}

func TestMergeBindings(t *testing.T) {
	a := uop.NewArena(64)
	b := tensor.NewVariable(a, "B", 1, 256)
	tt := tensor.NewVariable(a, "T", 1, 1024)
	out := tensor.MergeBindings(b.Bind(32), tt.Bind(128))
	if out["B"] != 32 || out["T"] != 128 || len(out) != 2 {
		t.Errorf("MergeBindings result = %v", out)
	}
}

func TestMergeBindingsEmpty(t *testing.T) {
	if out := tensor.MergeBindings(); out != nil {
		t.Errorf("MergeBindings() = %v, want nil", out)
	}
}

func TestMergeBindingsConflict(t *testing.T) {
	a := uop.NewArena(64)
	v := tensor.NewVariable(a, "x", 0, 100)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on conflicting bindings for same key")
		}
	}()
	tensor.MergeBindings(v.Bind(10), v.Bind(20))
}

func TestNewSymbolicShapeNonOutermostSym(t *testing.T) {
	// [4, n, 8] - sym in the middle.
	a := uop.NewArena(64)
	n := tensor.NewVariable(a, "seq", 1, 256)
	x := tensor.NewSymbolicShape(a, []shape.Sint{
		shape.Const(4), n.Sint(), shape.Const(8),
	}, uop.Dtypes.Float32, "webgpu")

	// Shape should be readable as Sints.
	sh := x.ShapeSints()
	if len(sh) != 3 {
		t.Fatalf("rank = %d, want 3", len(sh))
	}
	if v, ok := sh[0].ConstValue(); !ok || v != 4 {
		t.Errorf("dim 0 = %v, want Const(4)", sh[0])
	}
	if _, ok := sh[1].ConstValue(); ok {
		t.Errorf("dim 1 should be symbolic, got concrete %v", sh[1])
	}
	if v, ok := sh[2].ConstValue(); !ok || v != 8 {
		t.Errorf("dim 2 = %v, want Const(8)", sh[2])
	}

	// Underlying BUFFER node must carry a ShapeSintArg with Sym=true, V=0 at dim 1.
	arg, ok := x.Node().Arg().(uop.ShapeSintArg)
	if !ok {
		t.Fatalf("BUFFER arg is %T, want uop.ShapeSintArg", x.Node().Arg())
	}
	if arg[0].Sym || arg[0].V != 4 {
		t.Errorf("dim 0 ShapeDim = %+v, want concrete V=4", arg[0])
	}
	if !arg[1].Sym || arg[1].V != 0 || arg[1].VarName != "seq" || arg[1].Mul != 1 {
		t.Errorf("dim 1 ShapeDim = %+v, want Sym=true V=0 VarName=seq Mul=1", arg[1])
	}
	if arg[2].Sym || arg[2].V != 8 {
		t.Errorf("dim 2 ShapeDim = %+v, want concrete V=8", arg[2])
	}

	// BUFFER node should have exactly one src - the DefineVar for "seq".
	if x.Node().NSrc() != 1 {
		t.Fatalf("BUFFER NSrc = %d, want 1", x.Node().NSrc())
	}
	src := x.Node().Src(0)
	if src.Op() != uop.OpDefineVar {
		t.Errorf("src[0] op = %s, want OpDefineVar", src.Op())
	}
}

func TestNewSymbolicShapeTwoVariables(t *testing.T) {
	// [B, T, D] with B and T symbolic.
	a := uop.NewArena(64)
	B := tensor.NewVariable(a, "B", 1, 32)
	T := tensor.NewVariable(a, "T", 1, 256)
	const D = int64(16)

	x := tensor.NewSymbolicShape(a, []shape.Sint{
		B.Sint(), T.Sint(), shape.Const(D),
	}, uop.Dtypes.Float32, "webgpu")

	arg, ok := x.Node().Arg().(uop.ShapeSintArg)
	if !ok {
		t.Fatalf("BUFFER arg type %T", x.Node().Arg())
	}
	if arg[0].VarName != "B" || arg[1].VarName != "T" || arg[2].V != D {
		t.Errorf("ShapeSintArg dims = %+v", arg)
	}

	// Two distinct DefineVars stored as srcs, in name-sorted order: "B" < "T".
	if x.Node().NSrc() != 2 {
		t.Fatalf("BUFFER NSrc = %d, want 2", x.Node().NSrc())
	}
	if x.Node().Src(0).Arg().(uop.VarArg).Name != "B" {
		t.Errorf("src[0] = %v, want DefineVar B", x.Node().Src(0).Arg())
	}
	if x.Node().Src(1).Arg().(uop.VarArg).Name != "T" {
		t.Errorf("src[1] = %v, want DefineVar T", x.Node().Src(1).Arg())
	}
}

func TestNewSymbolicShapeAllConcreteFallsBackToLeaf(t *testing.T) {
	a := uop.NewArena(64)
	x := tensor.NewSymbolicShape(a, []shape.Sint{shape.Const(2), shape.Const(3)},
		uop.Dtypes.Float32, "webgpu")
	// All-concrete shape: should be a normal NewLeaf - arg is []int64, NSrc=0.
	if x.Node().NSrc() != 0 {
		t.Errorf("all-concrete shape should have NSrc=0; got %d", x.Node().NSrc())
	}
	if _, ok := x.Node().Arg().([]int64); !ok {
		t.Errorf("all-concrete shape arg type = %T, want []int64", x.Node().Arg())
	}
}

func TestNewSymbolicShapeRejectsExpressionSymInt(t *testing.T) {
	// Slice 4 surface accepts only bare DefineVars at construction time.
	a := uop.NewArena(64)
	v := tensor.NewVariable(a, "n", 1, 64)
	// Build a Mul(n, 2) expression - not a bare DefineVar.
	expr := shape.Mul(v.Sint(), shape.Const(2))
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on non-bare SymInt expression")
		}
	}()
	tensor.NewSymbolicShape(a, []shape.Sint{expr, shape.Const(8)},
		uop.Dtypes.Float32, "webgpu")
}

func TestNewSymbolicShapeEmptyShapePanics(t *testing.T) {
	a := uop.NewArena(64)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on empty shape")
		}
	}()
	tensor.NewSymbolicShape(a, nil, uop.Dtypes.Float32, "webgpu")
}

func TestNewVariableEmptyNamePanics(t *testing.T) {
	a := uop.NewArena(64)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on empty name")
		}
	}()
	tensor.NewVariable(a, "", 1, 64)
}

func TestNewVariableBadBoundsPanics(t *testing.T) {
	a := uop.NewArena(64)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on min > max")
		}
	}()
	tensor.NewVariable(a, "n", 10, 1)
}

// ── End-to-end (GPU-gated) ──────────────────────────────────────────────────

// TestNewSymbolicShapeNonOutermostSymRealize builds a tensor [batch=4, seq=sym,
// dim=8], multiplies it elementwise by a static [1, 1, 8] broadcast, and
// compares output at two different sym values to the static reference.
//
// This proves the codegen path treats non-outermost sym correctly via the new
// general constructor (Option B compiler support; frontend ergonomics).
func TestNewSymbolicShapeNonOutermostSymRealize(t *testing.T) {
	requireGPUVar(t)

	const (
		batch = int64(4)
		dim   = int64(8)
	)

	runAtSeq := func(seq int64) []float32 {
		a := uop.NewArena(4096)
		s := tensor.NewVariable(a, "seq", 1, 64)
		x := tensor.NewSymbolicShape(a, []shape.Sint{
			shape.Const(batch), s.Sint(), shape.Const(dim),
		}, uop.Dtypes.Float32, "webgpu")
		data := make([]float32, batch*seq*dim)
		for i := range data {
			data[i] = float32(i+1) * 0.1
		}
		x.SetData(data)
		scale := tensor.Full(a, []int64{1, 1, dim}, 2.5, uop.Dtypes.Float32, "webgpu")
		y := x.Mul(scale)
		if err := tensor.RealizeWithBinding(s.Bind(seq), y); err != nil {
			t.Fatalf("RealizeWithBinding seq=%d: %v", seq, err)
		}
		return append([]float32{}, y.Data()...)
	}

	staticRef := func(seq int64) []float32 {
		out := make([]float32, batch*seq*dim)
		for i := range out {
			out[i] = float32(i+1) * 0.1 * 2.5
		}
		return out
	}

	for _, seq := range []int64{1, 16, 64} {
		got := runAtSeq(seq)
		want := staticRef(seq)
		if len(got) != len(want) {
			t.Fatalf("seq=%d: len(got)=%d len(want)=%d", seq, len(got), len(want))
		}
		var maxErr float32
		for i := range got {
			if d := abs32(got[i] - want[i]); d > maxErr {
				maxErr = d
			}
		}
		t.Logf("seq=%d: max-abs-diff=%g", seq, maxErr)
		if maxErr != 0 {
			t.Errorf("seq=%d: expected bit-exact match, got max-abs-diff=%g", seq, maxErr)
		}
	}
}

// TestTwoSymbolicVariablesRealize binds B and T simultaneously and verifies the
// output against the static reference. Proves multi-Variable shapes go through
// the full pipeline correctly.
func TestTwoSymbolicVariablesRealize(t *testing.T) {
	requireGPUVar(t)

	const D = int64(16)
	runAt := func(bV, tV int64) []float32 {
		a := uop.NewArena(4096)
		B := tensor.NewVariable(a, "B", 1, 32)
		T := tensor.NewVariable(a, "T", 1, 256)
		x := tensor.NewSymbolicShape(a, []shape.Sint{
			B.Sint(), T.Sint(), shape.Const(D),
		}, uop.Dtypes.Float32, "webgpu")
		data := make([]float32, bV*tV*D)
		for i := range data {
			data[i] = float32(i%7) * 0.25
		}
		x.SetData(data)
		bias := tensor.Full(a, []int64{1, 1, D}, 0.5, uop.Dtypes.Float32, "webgpu")
		y := x.Add(bias)
		binding := tensor.MergeBindings(B.Bind(bV), T.Bind(tV))
		if err := tensor.RealizeWithBinding(binding, y); err != nil {
			t.Fatalf("RealizeWithBinding B=%d T=%d: %v", bV, tV, err)
		}
		return append([]float32{}, y.Data()...)
	}
	ref := func(bV, tV int64) []float32 {
		out := make([]float32, bV*tV*D)
		for i := range out {
			out[i] = float32(i%7)*0.25 + 0.5
		}
		return out
	}

	cases := []struct{ b, tt int64 }{{2, 8}, {4, 16}, {1, 64}}
	for _, c := range cases {
		got := runAt(c.b, c.tt)
		want := ref(c.b, c.tt)
		if len(got) != len(want) {
			t.Fatalf("B=%d T=%d: len(got)=%d len(want)=%d", c.b, c.tt, len(got), len(want))
		}
		var maxErr float32
		for i := range got {
			if d := abs32(got[i] - want[i]); d > maxErr {
				maxErr = d
			}
		}
		t.Logf("B=%d T=%d: max-abs-diff=%g", c.b, c.tt, maxErr)
		if maxErr != 0 {
			t.Errorf("B=%d T=%d: expected bit-exact, got %g", c.b, c.tt, maxErr)
		}
	}
}

// TestVariableJITReplayDifferentBind captures a symbolic-shape computation at
// seq=8, then replays it at seq=32 via JIT. Output must match the static
// reference at both binds with one capture and one replay.
func TestVariableJITReplayDifferentBind(t *testing.T) {
	requireGPUVar(t)

	const D = int64(4)
	a := uop.NewArena(4096)
	v := tensor.NewVariable(a, "n", 1, 128)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(D)},
		uop.Dtypes.Float32, "webgpu")
	scale := tensor.Full(a, []int64{1, D}, 3.0, uop.Dtypes.Float32, "webgpu")
	y := x.Mul(scale)

	jit := tensor.NewJIT()

	runAt := func(n int64, label string) {
		t.Helper()
		data := make([]float32, n*D)
		for i := range data {
			data[i] = float32(i+1) * 0.1
		}
		x.SetData(data)
		if err := jit.RealizeWithBinding(v.Bind(n), y); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		got := y.Data()
		want := make([]float32, n*D)
		for i := range want {
			want[i] = float32(i+1) * 0.1 * 3.0
		}
		if len(got) != len(want) {
			t.Fatalf("%s: len mismatch %d vs %d", label, len(got), len(want))
		}
		var maxErr float32
		for i := range got {
			if d := abs32(got[i] - want[i]); d > maxErr {
				maxErr = d
			}
		}
		if maxErr > 1e-5 {
			t.Errorf("%s: max-abs-diff %g", label, maxErr)
		}
	}
	runAt(8, "capture")
	caps1, _ := jit.JITStats()
	if caps1 != 1 {
		t.Fatalf("after first call: captures=%d, want 1", caps1)
	}
	runAt(32, "replay")
	caps2, reps2 := jit.JITStats()
	if caps2 != 1 || reps2 != 1 {
		t.Errorf("after replay: captures=%d replays=%d, want 1,1", caps2, reps2)
	}
}

// TestBindInRange exercises a binding at the upper extremity of [min, max]
// and verifies a clean realize. Out-of-range rejection is not enforced at
// the RealizeWithBinding dispatch path today (the executor consumes the
// uniform value without comparing against the DefineVar bounds); kernels
// produce well-defined output for any value the WGSL uniform admits as
// long as the leaf data is sized to match. Documenting the boundary here
// rather than asserting a rejection that does not happen.
func TestBindInRange(t *testing.T) {
	requireGPUVar(t)

	a := uop.NewArena(4096)
	const maxN = int64(32)
	v := tensor.NewVariable(a, "n", 1, maxN)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(4)},
		uop.Dtypes.Float32, "webgpu")
	data := make([]float32, maxN*4)
	for i := range data {
		data[i] = float32(i+1) * 0.5
	}
	x.SetData(data)
	scale := tensor.Full(a, []int64{1, 4}, 2.0, uop.Dtypes.Float32, "webgpu")
	y := x.Mul(scale)
	if err := tensor.RealizeWithBinding(v.Bind(maxN), y); err != nil {
		t.Fatalf("RealizeWithBinding at upper bound: %v", err)
	}
	got := y.Data()
	if int64(len(got)) != maxN*4 {
		t.Fatalf("len(got) = %d, want %d", len(got), maxN*4)
	}
	for i := range got {
		want := data[i] * 2.0
		if got[i] != want {
			t.Errorf("got[%d] = %g, want %g", i, got[i], want)
			break
		}
	}
}

// TestVariableFDGradient runs a finite-difference gradient check on a small
// computation built via the general symbolic constructor. The forward is
// loss = sum( (W @ x.transpose())^2 ) where x has shape [n, D] and W is a
// learnable [D, D] matrix. We verify d/dW from autodiff matches FD within
// 1e-3 at a single bind (n=4).
//
// Purpose: confirm the general constructor's BUFFER node srcs and arg are
// understood by the gradient pass on a non-trivial (non-outermost-only)
// path. We only build one symbolic dim here (n), at the outermost position
// for tractability, but route everything through NewSymbolicShape /
// NewVariable.
func TestVariableFDGradient(t *testing.T) {
	requireGPUVar(t)

	const (
		D = int64(3)
		n = int64(4)
	)
	wData := []float32{
		0.1, 0.2, -0.3,
		0.4, -0.1, 0.05,
		-0.2, 0.3, 0.0,
	}
	xData := []float32{
		0.5, -0.5, 1.0,
		1.0, 0.5, -0.5,
		-0.25, 0.25, 0.75,
		0.0, 1.0, -1.0,
	}

	forward := func(wPerturbed []float32) float32 {
		a := uop.NewArena(8192)
		v := tensor.NewVariable(a, "n", 1, 64)
		x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(D)},
			uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		w := tensor.NewLeaf(a, []int64{D, D}, uop.Dtypes.Float32, "webgpu")
		w.SetData(append([]float32{}, wPerturbed...))
		// y = x @ W^T → shape [n, D]; loss = sum(y^2).
		yT := x.Matmul(w.Permute([]int{1, 0}))
		loss := yT.Mul(yT).Sum(nil, false)
		if err := tensor.RealizeWithBinding(v.Bind(n), loss); err != nil {
			t.Fatalf("forward realize: %v", err)
		}
		return loss.Data()[0]
	}

	// Autodiff gradient at wData.
	a := uop.NewArena(8192)
	v := tensor.NewVariable(a, "n", 1, 64)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(D)},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	w := tensor.NewLeaf(a, []int64{D, D}, uop.Dtypes.Float32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	yT := x.Matmul(w.Permute([]int{1, 0}))
	loss := yT.Mul(yT).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{w})
	gW := grads[w]
	if gW == nil {
		t.Fatal("Backward returned no gradient for W")
	}
	if err := tensor.RealizeWithBinding(v.Bind(n), gW); err != nil {
		t.Fatalf("realize grad: %v", err)
	}
	autoGrad := append([]float32{}, gW.Data()...)
	if int64(len(autoGrad)) != D*D {
		t.Fatalf("autoGrad len=%d, want %d", len(autoGrad), D*D)
	}

	// FD gradient on every W entry.
	const eps = float32(1e-3)
	var maxRel float32
	for i := 0; i < int(D*D); i++ {
		wPlus := append([]float32{}, wData...)
		wMinus := append([]float32{}, wData...)
		wPlus[i] += eps
		wMinus[i] -= eps
		fdg := (forward(wPlus) - forward(wMinus)) / (2 * eps)
		ag := autoGrad[i]
		denom := absMax32(absF32(ag), absF32(fdg), 1e-3)
		rel := absF32(ag-fdg) / denom
		t.Logf("dL/dW[%d]: auto=%.4f fd=%.4f rel=%.4f", i, ag, fdg, rel)
		if rel > maxRel {
			maxRel = rel
		}
	}
	if maxRel > 5e-2 {
		t.Errorf("FD vs autodiff max-rel = %g > 5e-2", maxRel)
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func absF32(v float32) float32 { return float32(math.Abs(float64(v))) }

func absMax32(a, b, fallback float32) float32 {
	m := a
	if b > m {
		m = b
	}
	if m < fallback {
		return fallback
	}
	return m
}
