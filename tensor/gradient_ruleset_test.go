package tensor

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// buildTestShapeCache computes a shapeCache for all nodes reachable from root.
// Uses the same topoSortUOp and shapeOfNode helpers as runBackward.
func buildTestShapeCache(root uop.UOp) map[uint32][]shape.Sint {
	cache := make(map[uint32][]shape.Sint)
	topo := topoSortUOp(root)
	for _, n := range topo {
		shapeOfNode(n, cache)
	}
	return cache
}

// assertEquivGrad checks that two applyGradRule / Dispatch results are bit-identical:
// same length, same nil slots, and where non-nil, the same arena UOp index.
func assertEquivGrad(t *testing.T, op string, got1, got2 []*Tensor) {
	t.Helper()
	if len(got1) != len(got2) {
		t.Fatalf("%s: length mismatch: applyGradRule=%d Dispatch=%d", op, len(got1), len(got2))
		return
	}
	for i := range got1 {
		g1, g2 := got1[i], got2[i]
		if (g1 == nil) != (g2 == nil) {
			t.Errorf("%s[%d]: nil mismatch: applyGradRule=%v Dispatch=%v", op, i, g1 == nil, g2 == nil)
			continue
		}
		if g1 != nil && g1.node.Index() != g2.node.Index() {
			t.Errorf("%s[%d]: node index mismatch: applyGradRule=%d (op=%s) Dispatch=%d (op=%s)",
				op, i,
				g1.node.Index(), g1.node.Op(),
				g2.node.Index(), g2.node.Op())
		}
	}
}

// TestGradientRulesetEquivalence asserts that Gradient.Dispatch produces
// bit-identical UOp graph structure to applyGradRule for every op category.
// Both paths run on the same arena, so interning makes structurally-identical
// expressions share the same index — divergence means structural difference.
func TestGradientRulesetEquivalence(t *testing.T) {
	type gradCase struct {
		name     string
		makeCase func() (u uop.UOp, nodeT, adj *Tensor, sc map[uint32][]shape.Sint, device string)
	}

	cases := []gradCase{

		// ── Unary ALU ─────────────────────────────────────────────────────────

		{name: "OpNeg", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Neg()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpExp2", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Exp2()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpLog2", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Log2()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpSin", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Sin()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpSqrt", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Sqrt()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpReciprocal", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Recip()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpTrunc", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Trunc()
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		// ── Cast ──────────────────────────────────────────────────────────────

		// f32 → bf16 narrowing: adj is bf16, LeastUpperDType(bf16,f32)=f32, backward is adj.Cast(f32)
		{name: "OpCast_f32_bf16", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Cast(uop.Dtypes.BFloat16)
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.BFloat16, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		// OpBitcast with float src: same rule as OpCast
		{name: "OpBitcast_f32", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
			bitcastNode := a.New(uop.OpBitcast, uop.Dtypes.BFloat16, []uop.UOp{x.node}, nil, nil)
			sc := buildTestShapeCache(bitcastNode)
			nodeT := wrapGradTensor(bitcastNode, sc[bitcastNode.Index()], uop.Dtypes.BFloat16, "cpu")
			adj := NewLeaf(a, []int64{4}, uop.Dtypes.BFloat16, "cpu")
			return bitcastNode, nodeT, adj, sc, "cpu"
		}},

		// ── Binary ALU ────────────────────────────────────────────────────────

		{name: "OpAdd", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := x.Add(y)
			u := z.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpSub", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := x.Sub(y)
			u := z.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpMul", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := x.Mul(y)
			u := z.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpFDiv", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			// Div on floats uses mul-reciprocal, not OpFDiv directly.
			// Build OpFDiv directly to exercise the rule.
			z := a.New(uop.OpFDiv, uop.Dtypes.Float32, []uop.UOp{x.node, y.node}, nil, nil)
			sc := buildTestShapeCache(z)
			nodeT := wrapGradTensor(z, sc[z.Index()], uop.Dtypes.Float32, "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return z, nodeT, adj, sc, "cpu"
		}},

		{name: "OpMax", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := x.Maximum(y)
			u := z.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpMulAcc", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			mulaccNode := a.New(uop.OpMulAcc, uop.Dtypes.Float32, []uop.UOp{x.node, y.node, z.node}, nil, nil)
			sc := buildTestShapeCache(mulaccNode)
			nodeT := wrapGradTensor(mulaccNode, sc[mulaccNode.Index()], uop.Dtypes.Float32, "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return mulaccNode, nodeT, adj, sc, "cpu"
		}},

		// ── Non-differentiable ────────────────────────────────────────────────

		{name: "OpCmpLt_nil", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := x.CmpLt(y)
			u := z.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Bool, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		// ── Ternary ───────────────────────────────────────────────────────────

		{name: "OpWhere", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			cond := NewLeaf(a, []int64{3}, uop.Dtypes.Bool, "cpu")
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			z := Where(cond, x, y)
			u := z.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		// ── Movement ops ──────────────────────────────────────────────────────

		{name: "OpReshape", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
			y := x.Reshape([]int64{12})
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{12}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpExpand", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			// axis 0 is broadcast (size 1 → 3)
			x := NewLeaf(a, []int64{1, 4}, uop.Dtypes.Float32, "cpu")
			y := x.Expand([]int64{3, 4})
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpPermute", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
			y := x.Permute([]int{1, 0}) // transpose → [3, 2]
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpPad", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			y := x.Pad([][2]int64{{1, 2}}) // [3] → [6]
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{6}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpShrink", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{6}, uop.Dtypes.Float32, "cpu")
			y := x.Shrink([][2]int64{{1, 4}}) // [6] → [3]
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpFlip", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
			y := x.Flip([]bool{true})
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		// ── Reduce ────────────────────────────────────────────────────────────

		{name: "OpReduceAxis_Add", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(512)
			x := NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
			y := x.Sum([]int{1}, false) // [3, 4] → [3]
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		{name: "OpReduceAxis_Max", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(512)
			x := NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
			y := x.Max([]int{1}, false) // [3, 4] → [3]
			u := y.node
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},

		// ── Leaf (no gradient) ────────────────────────────────────────────────

		{name: "OpBuffer_nil", makeCase: func() (uop.UOp, *Tensor, *Tensor, map[uint32][]shape.Sint, string) {
			a := uop.NewArena(256)
			x := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			u := x.node // OpBuffer leaf
			sc := buildTestShapeCache(u)
			nodeT := wrapGradTensor(u, sc[u.Index()], u.DType(), "cpu")
			adj := NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
			return u, nodeT, adj, sc, "cpu"
		}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			u, nodeT, adj, sc, device := tc.makeCase()

			got1 := applyGradRule(u, nodeT, adj, sc, device)
			got2 := Gradient.Dispatch(u, nodeT, adj, sc, device)

			assertEquivGrad(t, tc.name, got1, got2)
		})
	}

	t.Logf("TestGradientRulesetEquivalence: %d cases all bit-exact", len(cases))
}

// TestGradientRulesetSentinel exercises the Dispatch sentinel: unregistered ops
// (e.g. OpConst) return nil just like applyGradRule's default.
func TestGradientRulesetSentinel(t *testing.T) {
	a := uop.NewArena(64)
	constT := ConstScalar(a, 1.0, uop.Dtypes.Float32, "cpu")
	constNode := constT.node
	sc := buildTestShapeCache(constNode)
	nodeT := wrapGradTensor(constNode, sc[constNode.Index()], uop.Dtypes.Float32, "cpu")
	adj := NewLeaf(a, []int64{}, uop.Dtypes.Float32, "cpu")

	got1 := applyGradRule(constNode, nodeT, adj, sc, "cpu")
	got2 := Gradient.Dispatch(constNode, nodeT, adj, sc, "cpu")

	if got1 != nil || got2 != nil {
		t.Errorf("OpConst: expected nil from both; got applyGradRule=%v Dispatch=%v",
			formatSlice(got1), formatSlice(got2))
	}
}

func formatSlice(ts []*Tensor) string {
	if ts == nil {
		return "<nil>"
	}
	return fmt.Sprintf("len=%d", len(ts))
}
