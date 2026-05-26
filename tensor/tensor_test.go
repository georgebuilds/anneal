package tensor_test

import (
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func shapeEq(t *testing.T, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape len %d != %d; got %v want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape[%d]: got %d want %d", i, got[i], want[i])
		}
	}
}

// assertOp asserts the root op of t's node.
func assertOp(t *testing.T, ten *tensor.Tensor, op uop.Op) {
	t.Helper()
	if ten.Node().Op() != op {
		t.Fatalf("expected op %s, got %s", op, ten.Node().Op())
	}
}

func newArena() *uop.Arena { return uop.NewArena(256) }

// ── leaf / constant construction ──────────────────────────────────────────────

func TestNewLeaf(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")

	assertOp(t, x, uop.OpBuffer)
	shapeEq(t, x.Shape(), []int64{2, 3})
	if x.DType() != uop.Dtypes.Float32 {
		t.Fatalf("dtype mismatch")
	}
	if !x.IsLeaf() {
		t.Fatal("NewLeaf should be a leaf")
	}
}

func TestConstScalar(t *testing.T) {
	a := newArena()
	s := tensor.ConstScalar(a, 3.14, uop.Dtypes.Float32, "cpu")
	assertOp(t, s, uop.OpConst)
	shapeEq(t, s.Shape(), []int64{}) // scalar
}

func TestFullGraph(t *testing.T) {
	a := newArena()
	f := tensor.Full(a, []int64{2, 3}, 0.0, uop.Dtypes.Float32, "cpu")
	// Full builds: CONST → RESHAPE([1,1]) → EXPAND([2,3])
	assertOp(t, f, uop.OpExpand)
	shapeEq(t, f.Shape(), []int64{2, 3})

	reshapeNode := f.Node().Src(0)
	if reshapeNode.Op() != uop.OpReshape {
		t.Fatalf("expected RESHAPE before EXPAND, got %s", reshapeNode.Op())
	}
	constNode := reshapeNode.Src(0)
	if constNode.Op() != uop.OpConst {
		t.Fatalf("expected CONST at root, got %s", constNode.Op())
	}
}

func TestZerosOnes(t *testing.T) {
	a := newArena()
	z := tensor.Zeros(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	o := tensor.Ones(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	shapeEq(t, z.Shape(), []int64{4})
	shapeEq(t, o.Shape(), []int64{4})
	if z.Node() == o.Node() {
		t.Fatal("zeros and ones should produce distinct graph nodes")
	}
}

func TestArange(t *testing.T) {
	a := newArena()
	r := tensor.Arange(a, 10, uop.Dtypes.Int32, "cpu")
	assertOp(t, r, uop.OpBuffer)
	shapeEq(t, r.Shape(), []int64{10})
}

// ── no-eager-compute invariant ────────────────────────────────────────────────

func TestNoEagerCompute(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	y := x.Add(x)

	// Graph was extended, nothing was realized.
	if a.Len() == 0 {
		t.Fatal("arena should have nodes")
	}
	assertOp(t, y, uop.OpAdd)
	if !y.Node().Valid() {
		t.Fatal("result node must be valid")
	}
}

// ── unary ops ─────────────────────────────────────────────────────────────────

func TestUnaryOps(t *testing.T) {
	cases := []struct {
		name string
		fn   func(*tensor.Tensor) *tensor.Tensor
		op   uop.Op
	}{
		{"Neg", (*tensor.Tensor).Neg, uop.OpNeg},
		{"Exp2", (*tensor.Tensor).Exp2, uop.OpExp2},
		{"Log2", (*tensor.Tensor).Log2, uop.OpLog2},
		{"Sin", (*tensor.Tensor).Sin, uop.OpSin},
		{"Sqrt", (*tensor.Tensor).Sqrt, uop.OpSqrt},
		{"Recip", (*tensor.Tensor).Recip, uop.OpReciprocal},
		{"Trunc", (*tensor.Tensor).Trunc, uop.OpTrunc},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := newArena()
			x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
			y := tc.fn(x)
			assertOp(t, y, tc.op)
			if y.Node().NSrc() != 1 {
				t.Fatalf("unary op should have 1 src, got %d", y.Node().NSrc())
			}
			shapeEq(t, y.Shape(), []int64{2, 3})
		})
	}
}

func TestDerivedUnary(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")

	// Exp = exp2(x / ln2) — root should be Exp2.
	exp := x.Exp()
	assertOp(t, exp, uop.OpExp2)

	// Log = log2(x) * ln2 — root should be Mul.
	log := x.Log()
	assertOp(t, log, uop.OpMul)
}

// ── binary ops ────────────────────────────────────────────────────────────────

func TestBinaryOps(t *testing.T) {
	cases := []struct {
		name string
		fn   func(*tensor.Tensor, *tensor.Tensor) *tensor.Tensor
		op   uop.Op
	}{
		{"Add", (*tensor.Tensor).Add, uop.OpAdd},
		{"Sub", (*tensor.Tensor).Sub, uop.OpSub},
		{"Mul", (*tensor.Tensor).Mul, uop.OpMul},
		{"Maximum", (*tensor.Tensor).Maximum, uop.OpMax},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := newArena()
			x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
			y := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
			z := tc.fn(x, y)
			assertOp(t, z, tc.op)
			if z.Node().NSrc() != 2 {
				t.Fatalf("binary op should have 2 srcs, got %d", z.Node().NSrc())
			}
			shapeEq(t, z.Shape(), []int64{2, 3})
		})
	}
}

func TestComparisons(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")

	lt := x.CmpLt(y)
	if lt.DType() != uop.Dtypes.Bool {
		t.Fatalf("CmpLt should produce bool dtype")
	}
	assertOp(t, lt, uop.OpCmpLt)

	eq := x.CmpEq(y)
	if eq.DType() != uop.Dtypes.Bool {
		t.Fatalf("CmpEq should produce bool dtype")
	}
}

func TestWhere(t *testing.T) {
	a := newArena()
	cond := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Bool, "cpu")
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	w := tensor.Where(cond, x, y)
	assertOp(t, w, uop.OpWhere)
	if w.Node().NSrc() != 3 {
		t.Fatalf("Where should have 3 srcs, got %d", w.Node().NSrc())
	}
}

// ── dtype promotion ───────────────────────────────────────────────────────────

func TestDTypePromotion(t *testing.T) {
	cases := []struct {
		a, b, want *uop.DType
	}{
		{uop.Dtypes.Float32, uop.Dtypes.Float32, uop.Dtypes.Float32},
		{uop.Dtypes.Float16, uop.Dtypes.Float32, uop.Dtypes.Float32},
		{uop.Dtypes.Float32, uop.Dtypes.Float16, uop.Dtypes.Float32},
		{uop.Dtypes.Int32, uop.Dtypes.Float32, uop.Dtypes.Float32},
	}
	for _, tc := range cases {
		a := newArena()
		x := tensor.NewLeaf(a, []int64{3}, tc.a, "cpu")
		y := tensor.NewLeaf(a, []int64{3}, tc.b, "cpu")
		z := x.Add(y)
		if z.DType() != tc.want {
			t.Errorf("Add(%s, %s): got dtype %s, want %s", tc.a, tc.b, z.DType(), tc.want)
		}
	}
}

// ── broadcasting ──────────────────────────────────────────────────────────────

func TestBroadcastSameShape(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	z := x.Add(y)
	shapeEq(t, z.Shape(), []int64{2, 3})
}

func TestBroadcastExpand(t *testing.T) {
	a := newArena()
	// [2,1,3] + [1,4,3] → [2,4,3]
	x := tensor.NewLeaf(a, []int64{2, 1, 3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{1, 4, 3}, uop.Dtypes.Float32, "cpu")
	z := x.Add(y)
	shapeEq(t, z.Shape(), []int64{2, 4, 3})
}

func TestBroadcastRankAlign(t *testing.T) {
	a := newArena()
	// [3] + [2,3] → [2,3]
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	z := x.Add(y)
	shapeEq(t, z.Shape(), []int64{2, 3})
}

func TestBroadcastScalar(t *testing.T) {
	a := newArena()
	// [2,3] + scalar_full → [2,3]
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	s := tensor.Full(a, []int64{1, 1}, 1.0, uop.Dtypes.Float32, "cpu")
	z := x.Add(s)
	shapeEq(t, z.Shape(), []int64{2, 3})
}

// ── movement ops ──────────────────────────────────────────────────────────────

func TestReshape(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	r := x.Reshape([]int64{6})
	assertOp(t, r, uop.OpReshape)
	shapeEq(t, r.Shape(), []int64{6})
}

func TestExpand(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{1, 3}, uop.Dtypes.Float32, "cpu")
	e := x.Expand([]int64{4, 3})
	assertOp(t, e, uop.OpExpand)
	shapeEq(t, e.Shape(), []int64{4, 3})
}

func TestPermute(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3, 4}, uop.Dtypes.Float32, "cpu")
	p := x.Permute([]int{2, 0, 1})
	assertOp(t, p, uop.OpPermute)
	shapeEq(t, p.Shape(), []int64{4, 2, 3})
}

func TestPad(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	p := x.Pad([][2]int64{{1, 1}, {0, 2}})
	assertOp(t, p, uop.OpPad)
	shapeEq(t, p.Shape(), []int64{4, 5})
}

func TestShrink(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 6}, uop.Dtypes.Float32, "cpu")
	s := x.Shrink([][2]int64{{1, 3}, {2, 5}})
	assertOp(t, s, uop.OpShrink)
	shapeEq(t, s.Shape(), []int64{2, 3})
}

func TestFlip(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	f := x.Flip([]bool{true, false})
	assertOp(t, f, uop.OpFlip)
	shapeEq(t, f.Shape(), []int64{3, 4})
}

func TestTranspose(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	tr := x.T()
	assertOp(t, tr, uop.OpPermute)
	shapeEq(t, tr.Shape(), []int64{3, 2})
}

func TestFlatten(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3, 4}, uop.Dtypes.Float32, "cpu")
	f := x.Flatten()
	shapeEq(t, f.Shape(), []int64{24})
}

// ── reduce ops ────────────────────────────────────────────────────────────────

func TestSum(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3, 4}, uop.Dtypes.Float32, "cpu")

	// Reduce over axis 1.
	s := x.Sum([]int{1}, false)
	assertOp(t, s, uop.OpReduceAxis)
	shapeEq(t, s.Shape(), []int64{2, 4})

	// keepdim.
	sk := x.Sum([]int{1}, true)
	shapeEq(t, sk.Shape(), []int64{2, 1, 4})
}

func TestSumAllAxes(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	s := x.Sum(nil, false)
	shapeEq(t, s.Shape(), []int64{})
}

func TestMaxReduce(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	m := x.Max([]int{0}, false)
	assertOp(t, m, uop.OpReduceAxis)
	shapeEq(t, m.Shape(), []int64{3})
}

func TestMean(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 5}, uop.Dtypes.Float32, "cpu")
	m := x.Mean([]int{1}, false)
	// Mean builds Sum → Div — root is Div (or Mul(Recip)), not REDUCE_AXIS directly.
	shapeEq(t, m.Shape(), []int64{4})
	if !m.Node().Valid() {
		t.Fatal("mean node should be valid")
	}
}

// ── matmul ────────────────────────────────────────────────────────────────────

func TestMatmul2D(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	z := x.Matmul(y)
	shapeEq(t, z.Shape(), []int64{2, 4})
	// Root is REDUCE_AXIS (the K-dim sum).
	assertOp(t, z, uop.OpReduceAxis)
}

func TestMatmulBatched(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{5, 2, 3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{5, 3, 4}, uop.Dtypes.Float32, "cpu")
	z := x.Matmul(y)
	shapeEq(t, z.Shape(), []int64{5, 2, 4})
	assertOp(t, z, uop.OpReduceAxis)
}

func TestMatmulVec(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	z := x.Matmul(y)
	shapeEq(t, z.Shape(), []int64{3})
	assertOp(t, z, uop.OpReduceAxis)
}

// ── cast ─────────────────────────────────────────────────────────────────────

func TestCast(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float16, "cpu")
	y := x.Cast(uop.Dtypes.Float32)
	assertOp(t, y, uop.OpCast)
	if y.DType() != uop.Dtypes.Float32 {
		t.Fatalf("expected float32, got %s", y.DType())
	}
	// No-op cast.
	z := x.Cast(uop.Dtypes.Float16)
	if z != x {
		t.Fatal("no-op cast should return same tensor")
	}
}

// ── realize ───────────────────────────────────────────────────────────────────

func TestRealize(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	y := x.Add(x)

	lenBefore := a.Len()
	err := tensor.Realize(y)
	if err == nil {
		t.Fatal("Realize without DefaultExecutor should return an error")
	}
	// SINK node was added to arena.
	if a.Len() <= lenBefore {
		t.Fatal("Realize should add a SINK node to the arena")
	}
	sinkNode := a.At(uint32(a.Len() - 1))
	if sinkNode.Op() != uop.OpSink {
		t.Fatalf("last arena node should be OpSink, got %s", sinkNode.Op())
	}
}

func TestRealizeEmpty(t *testing.T) {
	err := tensor.Realize()
	if err != nil {
		t.Fatalf("Realize() with no args should return nil, got %v", err)
	}
}

// TestLeafRegistry_ArenaIsolation confirms that leaf data is scoped to its arena
// and never leaks to a separate arena even when both arenas have leaves at the
// same local indices. This is the structural guarantee that eliminates the
// address-reuse aliasing hazard present in a global registry keyed by uintptr.
func TestLeafRegistry_ArenaIsolation(t *testing.T) {
	a1 := uop.NewArena(64)
	x1 := tensor.NewLeaf(a1, []int64{2}, uop.Dtypes.Float32, "cpu")
	x1.SetData([]float32{1, 2})

	// x2 is the first node in a2 — same local index as x1 in a1.
	a2 := uop.NewArena(64)
	x2 := tensor.NewLeaf(a2, []int64{2}, uop.Dtypes.Float32, "cpu")
	// SetData NOT called on x2.

	if x1.Node().Index() != x2.Node().Index() {
		t.Skip("local indices differ; cannot test same-index isolation")
	}

	// a2 must not see x1's data.
	if _, ok := a2.Leaf(x2.Node().Index()); ok {
		t.Fatal("new arena has stale leaf data — aliasing from prior arena")
	}

	// a1 must still have x1's data.
	d, ok := a1.Leaf(x1.Node().Index())
	if !ok {
		t.Fatal("x1 leaf data missing from a1")
	}
	if d[0] != 1 || d[1] != 2 {
		t.Fatalf("wrong leaf data: got %v", d)
	}
}

// ── interning / dedup ─────────────────────────────────────────────────────────

func TestInterning(t *testing.T) {
	// Two const-zero tensors of the same shape and dtype should share the CONST node.
	a := newArena()
	z1 := tensor.Zeros(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	z2 := tensor.Zeros(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")

	// Walk to the root CONST.
	c1 := z1.Node().Src(0).Src(0) // EXPAND→RESHAPE→CONST
	c2 := z2.Node().Src(0).Src(0)
	if c1 != c2 {
		t.Fatal("same constant should be interned to the same arena node")
	}
}
