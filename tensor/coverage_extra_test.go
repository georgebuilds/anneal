package tensor_test

// Additional CPU-path (graph-construction) coverage for the tensor package.
// These tests assert real graph shapes/ops and panic messages without
// requiring a live GPU device; they exercise movement ops, reductions, the
// scatter-preproc registry, gradient ruleset metadata, and constructor edge
// cases that the GPU-realize suites do not reach.

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── tensor.go accessor / constructor edges ────────────────────────────────────

func TestST_ReturnsTracker(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	st := x.ST()
	shapeEq(t, st.Shape(), []int64{2, 3})
}

func TestIsRealized_FalseThenTrue(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	if x.IsRealized() {
		t.Fatal("fresh leaf should not be realized")
	}
	x.SetData([]float32{1, 2, 3, 4})
	if !x.IsRealized() {
		t.Fatal("leaf should be realized after SetData")
	}
	if got := x.Data(); len(got) != 4 || got[2] != 3 {
		t.Fatalf("Data mismatch: %v", got)
	}
}

func TestSetData_QuantizesF16(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2}, uop.Dtypes.Float16, "cpu")
	// A value that is not exactly representable in f16 should be quantized,
	// i.e. SetData must store a (possibly) altered copy, not the caller slice.
	in := []float32{1.0, 0.1}
	x.SetData(in)
	got := x.Data()
	if &got[0] == &in[0] {
		t.Fatal("f16 SetData must store a quantized copy, not alias the input")
	}
	// 1.0 is exactly representable; 0.1 is not.
	if got[0] != 1.0 {
		t.Fatalf("1.0 should round-trip exactly in f16, got %v", got[0])
	}
}

func TestSetData_Float32NoCopy(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	in := []float32{1, 2, 3}
	x.SetData(in)
	if &x.Data()[0] != &in[0] {
		t.Fatal("f32 SetData should retain the caller slice (no quantize copy)")
	}
}

func TestFullSints_SymbolicShape(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 16)
	sh := []shape.Sint{v.Sint(), shape.Const(4)}
	f := tensor.FullSints(a, sh, 2.5, uop.Dtypes.Float32, "cpu")
	// Symbolic FullSints builds CONST -> RESHAPE(ones) -> EXPAND(sym).
	assertOp(t, f, uop.OpExpand)
	got := f.ShapeSints()
	if len(got) != 2 {
		t.Fatalf("rank: got %d want 2", len(got))
	}
	if _, ok := got[0].ConstValue(); ok {
		t.Fatal("dim 0 should remain symbolic")
	}
	if cv, _ := got[1].ConstValue(); cv != 4 {
		t.Fatalf("dim 1: got %v want 4", cv)
	}
}

func TestFullSints_EmptySymbolicNotReached(t *testing.T) {
	// An empty Sint shape has no symbolic dims, so FullSints delegates to Full
	// and returns a scalar CONST.
	a := newArena()
	f := tensor.FullSints(a, []shape.Sint{}, 1.0, uop.Dtypes.Float32, "cpu")
	assertOp(t, f, uop.OpConst)
}

func TestConstScalar_IntDtype(t *testing.T) {
	a := newArena()
	s := tensor.ConstScalar(a, 7.9, uop.Dtypes.Int32, "cpu")
	assertOp(t, s, uop.OpConst)
	// Int dtype truncates the float64 arg to int64(7).
	if arg, ok := s.Node().Arg().(int64); !ok || arg != 7 {
		t.Fatalf("int const arg: got %v want int64(7)", s.Node().Arg())
	}
}

func TestRandnGraph_ShapeAndOp(t *testing.T) {
	a := newArena()
	r := tensor.RandnGraph(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	assertOp(t, r, uop.OpBuffer)
	shapeEq(t, r.Shape(), []int64{2, 3})
	if arg, ok := r.Node().Arg().(string); !ok || arg != "randn" {
		t.Fatalf("randn arg: got %v", r.Node().Arg())
	}
}

// ── ops.go ────────────────────────────────────────────────────────────────────

func TestAbs_BuildsMaxOfNeg(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	abs := x.Abs()
	// Abs == Maximum(x, -x) -> root op is OpMax.
	assertOp(t, abs, uop.OpMax)
	shapeEq(t, abs.Shape(), []int64{4})
}

func TestDiv_IntegerPath(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Int32, "cpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Int32, "cpu")
	q := x.Div(y)
	// Integer Div lowers to OpIDiv (not mul-reciprocal).
	assertOp(t, q, uop.OpIDiv)
}

func TestDiv_FloatPath(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	q := x.Div(y)
	// Float Div is x * recip(y) -> root op is OpMul.
	assertOp(t, q, uop.OpMul)
}

// ── reduce.go ─────────────────────────────────────────────────────────────────

func TestMax_AllAxesDefault(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	m := x.Max(nil, false)
	assertOp(t, m, uop.OpReduceAxis)
	ra := m.Node().Arg().(uop.ReduceArg)
	if ra.Op != uop.OpMax {
		t.Fatalf("reduce op: got %v want OpMax", ra.Op)
	}
	if len(ra.Axes) != 2 {
		t.Fatalf("default Max should reduce all axes, got %v", ra.Axes)
	}
	shapeEq(t, m.Shape(), []int64{})
}

func TestMin_KeepDim(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	m := x.Min([]int{1}, true)
	// keepdim is a dropped reduce plus an explicit reshape, so the reduce is the
	// reshape's source.
	red := m.Node().Src(0)
	ra := red.Arg().(uop.ReduceArg)
	if ra.Op != uop.OpMin {
		t.Fatalf("reduce op: got %v want OpMin", ra.Op)
	}
	shapeEq(t, m.Shape(), []int64{2, 1})
}

func TestMean_DefaultAllAxes(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 4}, uop.Dtypes.Float32, "cpu")
	m := x.Mean(nil, false)
	// Mean is sum/count -> root op OpMul (Div on float = mul-reciprocal).
	assertOp(t, m, uop.OpMul)
	shapeEq(t, m.Shape(), []int64{})
}

func TestMean_NegativeAxis(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 4}, uop.Dtypes.Float32, "cpu")
	m := x.Mean([]int{-1}, true)
	shapeEq(t, m.Shape(), []int64{2, 1})
}

func TestMean_SymbolicAxisPanics(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 8)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(4)}, uop.Dtypes.Float32, "cpu")
	defer func() {
		r := recover()
		if r == nil || !strings.Contains(toStr(r), "symbolic axis") {
			t.Fatalf("expected symbolic-axis panic, got %v", r)
		}
	}()
	x.Mean([]int{0}, false)
}

func TestReduce_AxisOutOfRangePanics(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected out-of-range axis panic")
		}
	}()
	x.Sum([]int{5}, false)
}

func TestMin_DefaultAllAxes(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	m := x.Min(nil, false)
	ra := m.Node().Arg().(uop.ReduceArg)
	if len(ra.Axes) != 2 {
		t.Fatalf("default Min should reduce all axes, got %v", ra.Axes)
	}
	shapeEq(t, m.Shape(), []int64{})
}

func TestMax_KeepDim(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	m := x.Max([]int{0}, true)
	shapeEq(t, m.Shape(), []int64{1, 3})
}

// ── broadcast paths ───────────────────────────────────────────────────────────

func TestBroadcastTo_RankTooHighPanics(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	defer mustPanic(t, "cannot broadcast")
	tensor.BroadcastTo(x, []int64{3})
}

func TestBroadcastTo_PrependsRankOne(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	b := tensor.BroadcastTo(x, []int64{2, 3})
	shapeEq(t, b.Shape(), []int64{2, 3})
}

func TestBroadcastToSints_RankTooHighPanics(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	defer mustPanic(t, "rank too high")
	tensor.BroadcastToSints(x, []shape.Sint{shape.Const(3)})
}

func TestBroadcastToSints_PrependsSymbolic(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 8)
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	b := tensor.BroadcastToSints(x, []shape.Sint{v.Sint(), shape.Const(3)})
	if b.Rank() != 2 {
		t.Fatalf("rank: got %d want 2", b.Rank())
	}
}

func TestAdd_SymbolicBroadcast(t *testing.T) {
	// Exercises broadcastShapesSints: a [n,3] tensor + a [3] tensor.
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 8)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(3)}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	out := x.Add(y)
	if out.Rank() != 2 {
		t.Fatalf("rank: got %d want 2", out.Rank())
	}
	if _, ok := out.ShapeSints()[0].ConstValue(); ok {
		t.Fatal("dim 0 should stay symbolic after broadcast-add")
	}
}

// ── variable.go ───────────────────────────────────────────────────────────────

func TestNewVariable_RealiasesExisting(t *testing.T) {
	a := newArena()
	v1 := tensor.NewVariable(a, "n", 1, 16)
	v2 := tensor.NewVariable(a, "n", 1, 16)
	if v1.Node() != v2.Node() {
		t.Fatal("same (name,bounds) should intern to the same DefineVar")
	}
	if v2.Min() != 1 || v2.Max() != 16 {
		t.Fatalf("realiased bounds: got [%d,%d]", v2.Min(), v2.Max())
	}
}

func TestVariableAccessors(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "seq", 2, 128)
	if v.Name() != "seq" {
		t.Fatalf("Name: got %q", v.Name())
	}
	if v.Min() != 2 || v.Max() != 128 {
		t.Fatalf("bounds: got [%d,%d]", v.Min(), v.Max())
	}
	if v.Node().Op() != uop.OpDefineVar {
		t.Fatalf("Node op: got %v", v.Node().Op())
	}
}

// ── matmul.go panic paths ─────────────────────────────────────────────────────

func TestMatmul_ScalarOperandPanics(t *testing.T) {
	a := newArena()
	scalar := tensor.ConstScalar(a, 1, uop.Dtypes.Float32, "cpu")
	other := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	defer mustPanic(t, "at least 1D")
	scalar.Matmul(other)
}

func TestMatmul_MatVecMismatchPanics(t *testing.T) {
	a := newArena()
	m := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	v := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu") // wrong len
	defer mustPanic(t, "vector dim mismatch")
	m.Matmul(v)
}

func TestMatmul_InnerDimMismatchPanics(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{4, 5}, uop.Dtypes.Float32, "cpu") // K=4 != 3
	defer mustPanic(t, "inner dim mismatch")
	x.Matmul(y)
}

func TestMatmul_MatVec(t *testing.T) {
	a := newArena()
	m := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	v := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	out := m.Matmul(v)
	shapeEq(t, out.Shape(), []int64{2})
}

// ── movement.go ───────────────────────────────────────────────────────────────

func TestFlattenFrom(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3, 4}, uop.Dtypes.Float32, "cpu")
	f := x.FlattenFrom(1)
	shapeEq(t, f.Shape(), []int64{2, 12})
}

func TestFlattenFrom_Negative(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3, 4}, uop.Dtypes.Float32, "cpu")
	f := x.FlattenFrom(-2)
	shapeEq(t, f.Shape(), []int64{2, 12})
}

func TestFlattenFrom_PastEndIsNoop(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	f := x.FlattenFrom(5)
	if f != x {
		t.Fatal("FlattenFrom past last dim should be a no-op returning the same tensor")
	}
}

func TestUnsqueeze_Middle(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	u := x.Unsqueeze(1)
	shapeEq(t, u.Shape(), []int64{2, 1, 3})
}

func TestUnsqueeze_NegativeAppends(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	u := x.Unsqueeze(-1)
	shapeEq(t, u.Shape(), []int64{2, 3, 1})
}

func TestTranspose_RankTooLowPanics(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	defer mustPanic(t, "at least 2 dimensions")
	x.Transpose()
}

func TestPermute_WrongLengthPanics(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	defer mustPanic(t, "order length")
	x.Permute([]int{0})
}

func TestPad_AllZeroIsNoop(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	p := x.Pad([][2]int64{{0, 0}, {0, 0}})
	if p != x {
		t.Fatal("all-zero Pad should be a no-op")
	}
}

func TestShrink_IdentityIsNoop(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	s := x.Shrink([][2]int64{{0, 2}, {0, 3}})
	if s != x {
		t.Fatal("identity Shrink should be a no-op")
	}
}

func TestPadSints_AllConcreteFallsBack(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	p := x.PadSints([][2]shape.Sint{{shape.Const(1), shape.Const(0)}, {shape.Const(0), shape.Const(0)}})
	assertOp(t, p, uop.OpPad)
	shapeEq(t, p.Shape(), []int64{3, 3})
}

func TestPadSints_AllZeroConcreteIsNoop(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	p := x.PadSints([][2]shape.Sint{{shape.Const(0), shape.Const(0)}, {shape.Const(0), shape.Const(0)}})
	if p != x {
		t.Fatal("all-zero concrete PadSints should be a no-op")
	}
}

func TestPadSints_Symbolic(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 8)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(3)}, uop.Dtypes.Float32, "cpu")
	// Symbolic pad amount on axis 0.
	p := x.PadSints([][2]shape.Sint{{v.Sint(), shape.Const(0)}, {shape.Const(0), shape.Const(0)}})
	assertOp(t, p, uop.OpPad)
	if p.Rank() != 2 {
		t.Fatalf("rank: got %d", p.Rank())
	}
}

func TestShrinkSints_AllConcreteFallsBack(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 3}, uop.Dtypes.Float32, "cpu")
	s := x.ShrinkSints([][2]shape.Sint{{shape.Const(1), shape.Const(3)}, {shape.Const(0), shape.Const(3)}})
	assertOp(t, s, uop.OpShrink)
	shapeEq(t, s.Shape(), []int64{2, 3})
}

func TestShrinkSints_Symbolic(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 8)
	x := tensor.NewSymbolicShape(a, []shape.Sint{v.Sint(), shape.Const(4)}, uop.Dtypes.Float32, "cpu")
	s := x.ShrinkSints([][2]shape.Sint{{shape.Const(0), v.Sint()}, {shape.Const(0), shape.Const(2)}})
	assertOp(t, s, uop.OpShrink)
}

func TestExpandSints_ConcreteFallsBack(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{1, 3}, uop.Dtypes.Float32, "cpu")
	e := x.ExpandSints([]shape.Sint{shape.Const(4), shape.Const(3)})
	assertOp(t, e, uop.OpExpand)
	shapeEq(t, e.Shape(), []int64{4, 3})
}

func TestExpandSints_SymbolicNoopWhenEqual(t *testing.T) {
	a := newArena()
	v := tensor.NewVariable(a, "n", 1, 8)
	sh := []shape.Sint{v.Sint(), shape.Const(3)}
	x := tensor.NewSymbolicShape(a, sh, uop.Dtypes.Float32, "cpu")
	e := x.ExpandSints(sh)
	if e != x {
		t.Fatal("ExpandSints to identical symbolic shape should be a no-op")
	}
}

// ── scatter_preproc.go ────────────────────────────────────────────────────────

func TestRunScatterPreprocessors_EmptyArenaNoop(t *testing.T) {
	a := newArena()
	// No scatter registered on this arena: must return without panic.
	tensor.RunScatterPreprocessors(a)
}

func TestScatterPreproc_MissingDataPanics(t *testing.T) {
	// Build an Embedding-style gather/backward so a scatter preprocessor gets
	// registered, then run it without giving the idx tensor data. The
	// preprocessor must panic with the "no data attached" message.
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "cpu")
	out := w.Gather(0, idx)
	loss := out.Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{w})
	if grads[w] == nil {
		t.Fatal("expected gradient w.r.t. embedding table")
	}
	// idx never had SetData -> the preprocessor reports missing data and panics.
	defer mustPanic(t, "no data attached")
	tensor.RunScatterPreprocessors(a)
}

func TestScatterPreproc_RunsWithData(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "cpu")
	out := w.Gather(0, idx)
	loss := out.Sum(nil, false)
	tensor.Backward(loss, []*tensor.Tensor{w})
	// Pack int32 indices as float32 bit patterns (the leaf encoding).
	idx.SetData(i32sAsF32Bits([]int32{2, 0, 5}))
	// Should run cleanly (no panic) because idx now has data.
	tensor.RunScatterPreprocessors(a)
}

// ── gradient_ruleset.go ───────────────────────────────────────────────────────

func TestRegisteredOps_SortedAndNonEmpty(t *testing.T) {
	ops := tensor.Gradient.RegisteredOps()
	if len(ops) == 0 {
		t.Fatal("expected at least one registered gradient op")
	}
	for i := 1; i < len(ops); i++ {
		if ops[i] < ops[i-1] {
			t.Fatalf("RegisteredOps not sorted at %d: %v then %v", i, ops[i-1], ops[i])
		}
	}
	// A couple of well-known ops must be present.
	want := map[uop.Op]bool{uop.OpAdd: false, uop.OpMul: false, uop.OpNeg: false}
	for _, op := range ops {
		if _, ok := want[op]; ok {
			want[op] = true
		}
	}
	for op, found := range want {
		if !found {
			t.Fatalf("expected %v in RegisteredOps", op)
		}
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return ""
}

func mustPanic(t *testing.T, substr string) {
	t.Helper()
	r := recover()
	if r == nil {
		t.Fatalf("expected panic containing %q, got none", substr)
	}
	msg := toStr(r)
	if !strings.Contains(msg, substr) {
		t.Fatalf("panic value %v does not contain %q", r, substr)
	}
}
