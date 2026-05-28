package tensor

import (
	"math"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// gradient_oracle_test.go
//
// applyGradRule is the original typed-switch gradient dispatch, retained as the
// DIFFERENTIAL EQUIVALENCE ORACLE for tensor.Gradient (the production ruleset
// in gradient_ruleset.go). TestGradientRulesetEquivalence enforces bit-exact
// equivalence between this oracle and the ruleset on every test run, so any
// future ruleset edit is caught the moment it diverges from this reference.
//
// This file is part of the package for tests but excluded from production
// builds (the _test.go suffix). Do not call applyGradRule from production code.

// applyGradRule returns adjoint contributions for each src of u.
// Index i in the returned slice is the contribution to u.Src(i).
// Nil elements and a nil slice both mean "no gradient for this op/src".
func applyGradRule(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
	a := u.Arena()

	// Helper: wrap src i as a Tensor (safe to call in rules that need src values).
	src := func(i int) *Tensor {
		s := u.Src(i)
		sh := shapeCache[s.Index()]
		return wrapGradTensor(s, sh, s.DType(), device)
	}
	// Helper: scalar constant broadcast to adj's shape.
	k := func(v float64) *Tensor {
		return FullSints(a, adj.ShapeSints(), v, adj.dtype, device)
	}
	// Helper: zero tensor with adj's shape.
	zeros := func() *Tensor {
		return FullSints(a, adj.ShapeSints(), 0.0, adj.dtype, device)
	}

	switch u.Op() {

	// ── Unary ALU ─────────────────────────────────────────────────────────────

	case uop.OpNeg:
		return []*Tensor{adj.Neg()}

	case uop.OpExp2:
		// d/dx 2^x = 2^x · ln2  (node IS 2^x)
		return []*Tensor{adj.Mul(nodeT).Mul(k(ln2))}

	case uop.OpLog2:
		// d/dx log₂(x) = 1 / (x · ln2)
		return []*Tensor{adj.Div(src(0).Mul(k(ln2)))}

	case uop.OpSin:
		// d/dx sin(x) = cos(x) = sin(x + π/2)
		return []*Tensor{adj.Mul(src(0).Add(k(math.Pi / 2)).Sin())}

	case uop.OpSqrt:
		// d/dx √x = 1 / (2√x) = 1 / (2 · node)
		return []*Tensor{adj.Div(k(2.0).Mul(nodeT))}

	case uop.OpReciprocal:
		// d/dx 1/x = -1/x² = -node²
		return []*Tensor{adj.Neg().Mul(nodeT).Mul(nodeT)}

	case uop.OpTrunc:
		return nil // non-differentiable

	case uop.OpCast, uop.OpBitcast:
		if u.Src(0).DType().IsFloat() {
			dtype := uop.LeastUpperDType(adj.dtype, u.Src(0).DType())
			return []*Tensor{adj.Cast(dtype)}
		}
		return nil

	// ── Binary ALU ────────────────────────────────────────────────────────────

	case uop.OpAdd:
		return []*Tensor{adj, adj}

	case uop.OpSub:
		return []*Tensor{adj, adj.Neg()}

	case uop.OpMul:
		return []*Tensor{adj.Mul(src(1)), adj.Mul(src(0))}

	case uop.OpFDiv:
		// d/da a/b = 1/b; d/db a/b = -a/b²
		s1 := src(1)
		return []*Tensor{
			adj.Div(s1),
			adj.Neg().Mul(src(0)).Mul(s1.Recip()).Mul(s1.Recip()),
		}

	case uop.OpMax:
		// At ties, src[0] wins
		z := zeros()
		s0, s1 := src(0), src(1)
		// g0 = adj where s0 == max-output
		g0 := Where(s0.CmpEq(nodeT), adj, z)
		// g1 = adj where s1 == max-output AND s0 is not also max (tie-break)
		notTied := Where(s0.CmpEq(nodeT), z, adj)
		g1 := Where(s1.CmpEq(nodeT), notTied, z)
		return []*Tensor{g0, g1}

	case uop.OpMulAcc:
		// mulacc(a,b,c) = a*b+c; grad_a=adj*b, grad_b=adj*a, grad_c=adj
		return []*Tensor{adj.Mul(src(1)), adj.Mul(src(0)), adj}

	case uop.OpCmpLt, uop.OpCmpNe, uop.OpCmpEq,
		uop.OpIDiv, uop.OpMod, uop.OpShl, uop.OpShr,
		uop.OpXor, uop.OpOr, uop.OpAnd, uop.OpPow:
		return nil // non-differentiable

	// ── Ternary ───────────────────────────────────────────────────────────────

	case uop.OpWhere:
		// where(cond, x, y): grad_cond=nil, grad_x=where(cond,adj,0), grad_y=where(cond,0,adj)
		cond := src(0)
		z := zeros()
		return []*Tensor{
			nil,
			Where(cond, adj, z),
			Where(cond, z, adj),
		}

	// ── Movement ops ──────────────────────────────────────────────────────────

	case uop.OpReshape:
		srcSints := shapeCache[u.Src(0).Index()]
		if srcSints == nil {
			srcSints = []shape.Sint{}
		}
		return []*Tensor{adj.ReshapeSints(srcSints)}

	case uop.OpExpand:
		// The inverse of expand is sum-reduce over broadcast axes.
		srcSints := shapeCache[u.Src(0).Index()]
		expandedSints := shapeCache[u.Index()]
		rank := len(expandedSints)

		// srcSints always has the same rank as expandedSints (Reshape prepends 1s before Expand).
		var sumAxes []int
		for i := 0; i < rank; i++ {
			var sv int64
			if i < len(srcSints) {
				if v, ok := srcSints[i].ConstValue(); ok {
					sv = v
				} else {
					continue // symbolic src dim — never 1, not broadcast
				}
			}
			if sv == 1 {
				sumAxes = append(sumAxes, i)
			}
		}

		g := adj
		if len(sumAxes) > 0 {
			g = g.Sum(sumAxes, true) // keepdim preserves rank
		}
		g = g.ReshapeSints(srcSints) // restore src shape
		return []*Tensor{g}

	case uop.OpPermute:
		permArg := u.Arg().([]int64)
		perm := make([]int, len(permArg))
		for i, p := range permArg {
			perm[i] = int(p)
		}
		// Inverse permutation.
		inv := make([]int, len(perm))
		for i, p := range perm {
			inv[p] = i
		}
		return []*Tensor{adj.Permute(inv)}

	case uop.OpPad:
		// Pad → Shrink: remove the padding from the adjoint.
		// Pad/Shrink only operate on concrete dims (scope guard).
		padding := u.Arg().([][2]int64)
		srcSints := shapeCache[u.Src(0).Index()]
		srcSh := shape.AsInts(srcSints)
		shrinkArg := make([][2]int64, len(padding))
		for i, p := range padding {
			shrinkArg[i] = [2]int64{p[0], p[0] + srcSh[i]}
		}
		return []*Tensor{adj.Shrink(shrinkArg)}

	case uop.OpShrink:
		// Shrink → Pad: restore the missing elements as zero.
		shrinkArg := u.Arg().([][2]int64)
		srcSints := shapeCache[u.Src(0).Index()]
		srcSh := shape.AsInts(srcSints)
		padArg := make([][2]int64, len(shrinkArg))
		for i, p := range shrinkArg {
			padArg[i] = [2]int64{p[0], srcSh[i] - p[1]}
		}
		return []*Tensor{adj.Pad(padArg)}

	case uop.OpFlip:
		flipArg := u.Arg().([]int64)
		axes := make([]bool, len(flipArg))
		for i, f := range flipArg {
			axes[i] = f != 0
		}
		return []*Tensor{adj.Flip(axes)}

	// ── Reduce ────────────────────────────────────────────────────────────────

	case uop.OpReduceAxis:
		ra := u.Arg().(uop.ReduceArg)
		srcSints := shapeCache[u.Src(0).Index()]
		if srcSints == nil {
			return nil
		}
		rank := len(srcSints)

		// Build keepdim Sint shape: same as srcSints but with reduced axes set to Const(1).
		keepSints := make([]shape.Sint, rank)
		copy(keepSints, srcSints)
		axSet := make(map[int]bool, len(ra.Axes))
		for _, ax := range ra.Axes {
			axSet[ax] = true
			keepSints[ax] = shape.Const(1)
		}

		switch ra.Op {
		case uop.OpAdd:
			// Sum backward: broadcast adjoint back to src shape.
			return []*Tensor{adj.ReshapeSints(keepSints).ExpandSints(srcSints)}

		case uop.OpMax:
			// Max backward: route adjoint to argmax positions; split ties equally.
			nodeExp := nodeT.ReshapeSints(keepSints).ExpandSints(srcSints)
			adjExp := adj.ReshapeSints(keepSints).ExpandSints(srcSints)
			s0 := src(0)
			mask := s0.CmpEq(nodeExp)
			maskFloat := mask.Cast(adj.dtype)
			tieCount := maskFloat.Sum(ra.Axes, true).ExpandSints(srcSints)
			zSrc := FullSints(a, srcSints, 0.0, adj.dtype, device)
			return []*Tensor{Where(mask, adjExp.Div(tieCount), zSrc)}
		}
	}

	return nil
}
