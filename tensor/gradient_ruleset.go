package tensor

import (
	"math"
	"sort"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// GradRule computes per-source adjoint contributions for a single forward op.
// Mirrors the per-case logic in applyGradRule, factored into a per-op closure.
// Index i of the returned slice is the contribution to u.Src(i); nil means no gradient.
type GradRule func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor

// GradRuleset is a dispatch table mapping op codes to gradient rules.
type GradRuleset struct {
	rules map[uop.Op]GradRule
}

// Dispatch looks up the rule for u.Op() and calls it.
// Returns nil for unregistered ops, matching applyGradRule's default behaviour.
func (rs GradRuleset) Dispatch(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
	if rule, ok := rs.rules[u.Op()]; ok {
		return rule(u, nodeT, adj, shapeCache, device)
	}
	return nil
}

// RegisteredOps returns the set of forward Ops with a registered gradient rule.
// Used by explain's drift check to enforce curated-prose ↔ ruleset correspondence.
// The returned slice is sorted (by uop.Op's underlying integer ordering) for
// deterministic iteration.
func (rs GradRuleset) RegisteredOps() []uop.Op {
	ops := make([]uop.Op, 0, len(rs.rules))
	for op := range rs.rules {
		ops = append(ops, op)
	}
	sort.Slice(ops, func(i, j int) bool { return ops[i] < ops[j] })
	return ops
}

// Gradient is the parallel gradient ruleset, equivalent to applyGradRule.
var Gradient = buildGradient()

// gradHelpers creates the per-invocation helper closures used by gradient rules.
// Mirrors the local helpers inside applyGradRule exactly.
func gradHelpers(u uop.UOp, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) (
	src func(int) *Tensor,
	k func(float64) *Tensor,
	zeros func() *Tensor,
) {
	a := u.Arena()
	src = func(i int) *Tensor {
		s := u.Src(i)
		sh := shapeCache[s.Index()]
		return wrapGradTensor(s, sh, s.DType(), device)
	}
	k = func(v float64) *Tensor {
		return FullSints(a, adj.ShapeSints(), v, adj.dtype, device)
	}
	zeros = func() *Tensor {
		return FullSints(a, adj.ShapeSints(), 0.0, adj.dtype, device)
	}
	return
}

func buildGradient() GradRuleset {
	m := map[uop.Op]GradRule{}

	// ── Unary ALU ─────────────────────────────────────────────────────────────

	// d/dx (-x) = -1
	m[uop.OpNeg] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		return []*Tensor{adj.Neg()}
	}

	// d/dx 2^x = 2^x · ln2  (node IS 2^x)
	m[uop.OpExp2] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		_, k, _ := gradHelpers(u, adj, shapeCache, device)
		return []*Tensor{adj.Mul(nodeT).Mul(k(ln2))}
	}

	// d/dx log₂(x) = 1 / (x · ln2)
	m[uop.OpLog2] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, k, _ := gradHelpers(u, adj, shapeCache, device)
		return []*Tensor{adj.Div(src(0).Mul(k(ln2)))}
	}

	// d/dx sin(x) = cos(x) = sin(x + π/2)
	m[uop.OpSin] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, k, _ := gradHelpers(u, adj, shapeCache, device)
		return []*Tensor{adj.Mul(src(0).Add(k(math.Pi / 2)).Sin())}
	}

	// d/dx √x = 1 / (2√x) = 1 / (2 · node)
	m[uop.OpSqrt] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		_, k, _ := gradHelpers(u, adj, shapeCache, device)
		return []*Tensor{adj.Div(k(2.0).Mul(nodeT))}
	}

	// d/dx 1/x = -1/x² = -node²
	m[uop.OpReciprocal] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		return []*Tensor{adj.Neg().Mul(nodeT).Mul(nodeT)}
	}

	// non-differentiable
	m[uop.OpTrunc] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		return nil
	}

	// ── Cast ──────────────────────────────────────────────────────────────────

	castRule := GradRule(func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		if u.Src(0).DType().IsFloat() {
			dtype := uop.LeastUpperDType(adj.dtype, u.Src(0).DType())
			return []*Tensor{adj.Cast(dtype)}
		}
		return nil
	})
	m[uop.OpCast] = castRule
	m[uop.OpBitcast] = castRule

	// ── Binary ALU ────────────────────────────────────────────────────────────

	// d/da (a+b) = 1; d/db (a+b) = 1
	m[uop.OpAdd] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		return []*Tensor{adj, adj}
	}

	// d/da (a-b) = 1; d/db (a-b) = -1
	m[uop.OpSub] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		return []*Tensor{adj, adj.Neg()}
	}

	// d/da (a*b) = b; d/db (a*b) = a
	m[uop.OpMul] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, _, _ := gradHelpers(u, adj, shapeCache, device)
		return []*Tensor{adj.Mul(src(1)), adj.Mul(src(0))}
	}

	// d/da a/b = 1/b; d/db a/b = -a/b²
	m[uop.OpFDiv] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, _, _ := gradHelpers(u, adj, shapeCache, device)
		s1 := src(1)
		return []*Tensor{
			adj.Div(s1),
			adj.Neg().Mul(src(0)).Mul(s1.Recip()).Mul(s1.Recip()),
		}
	}

	// max(a,b): at ties, src[0] wins
	m[uop.OpMax] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, _, zeros := gradHelpers(u, adj, shapeCache, device)
		z := zeros()
		s0, s1 := src(0), src(1)
		g0 := Where(s0.CmpEq(nodeT), adj, z)
		notTied := Where(s0.CmpEq(nodeT), z, adj)
		g1 := Where(s1.CmpEq(nodeT), notTied, z)
		return []*Tensor{g0, g1}
	}

	// mulacc(a,b,c) = a*b+c; grad_a=adj*b, grad_b=adj*a, grad_c=adj
	m[uop.OpMulAcc] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, _, _ := gradHelpers(u, adj, shapeCache, device)
		return []*Tensor{adj.Mul(src(1)), adj.Mul(src(0)), adj}
	}

	// non-differentiable integer/comparison ops
	nilRule := GradRule(func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		return nil
	})
	for _, op := range []uop.Op{
		uop.OpCmpLt, uop.OpCmpNe, uop.OpCmpEq,
		uop.OpIDiv, uop.OpMod, uop.OpShl, uop.OpShr,
		uop.OpXor, uop.OpOr, uop.OpAnd, uop.OpPow,
	} {
		m[op] = nilRule
	}

	// ── Ternary ───────────────────────────────────────────────────────────────

	// where(cond, x, y): grad_cond=nil, grad_x=where(cond,adj,0), grad_y=where(cond,0,adj)
	m[uop.OpWhere] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, _, zeros := gradHelpers(u, adj, shapeCache, device)
		cond := src(0)
		z := zeros()
		return []*Tensor{
			nil,
			Where(cond, adj, z),
			Where(cond, z, adj),
		}
	}

	// ── Movement ops ──────────────────────────────────────────────────────────

	// OpReshape backward: reshape adjoint back to src shape
	m[uop.OpReshape] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		srcSints := shapeCache[u.Src(0).Index()]
		if srcSints == nil {
			srcSints = []shape.Sint{}
		}
		return []*Tensor{adj.ReshapeSints(srcSints)}
	}

	// OpExpand backward: sum-reduce over broadcast axes
	m[uop.OpExpand] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		srcSints := shapeCache[u.Src(0).Index()]
		expandedSints := shapeCache[u.Index()]
		rank := len(expandedSints)

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
	}

	// OpPermute backward: apply inverse permutation
	m[uop.OpPermute] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		permArg := u.Arg().([]int64)
		perm := make([]int, len(permArg))
		for i, p := range permArg {
			perm[i] = int(p)
		}
		inv := make([]int, len(perm))
		for i, p := range perm {
			inv[p] = i
		}
		return []*Tensor{adj.Permute(inv)}
	}

	// OpPad backward: shrink to remove the padding from the adjoint
	m[uop.OpPad] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		padding := u.Arg().([][2]int64)
		srcSints := shapeCache[u.Src(0).Index()]
		srcSh := shape.AsInts(srcSints)
		shrinkArg := make([][2]int64, len(padding))
		for i, p := range padding {
			shrinkArg[i] = [2]int64{p[0], p[0] + srcSh[i]}
		}
		return []*Tensor{adj.Shrink(shrinkArg)}
	}

	// OpShrink backward: pad to restore the missing elements as zero
	m[uop.OpShrink] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		shrinkArg := u.Arg().([][2]int64)
		srcSints := shapeCache[u.Src(0).Index()]
		srcSh := shape.AsInts(srcSints)
		padArg := make([][2]int64, len(shrinkArg))
		for i, p := range shrinkArg {
			padArg[i] = [2]int64{p[0], srcSh[i] - p[1]}
		}
		return []*Tensor{adj.Pad(padArg)}
	}

	// OpFlip backward: flip the same axes
	m[uop.OpFlip] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		flipArg := u.Arg().([]int64)
		axes := make([]bool, len(flipArg))
		for i, f := range flipArg {
			axes[i] = f != 0
		}
		return []*Tensor{adj.Flip(axes)}
	}

	// ── Reduce ────────────────────────────────────────────────────────────────

	m[uop.OpReduceAxis] = func(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]shape.Sint, device string) []*Tensor {
		src, _, _ := gradHelpers(u, adj, shapeCache, device)
		a := u.Arena()
		ra := u.Arg().(uop.ReduceArg)
		srcSints := shapeCache[u.Src(0).Index()]
		if srcSints == nil {
			return nil
		}
		rank := len(srcSints)

		keepSints := make([]shape.Sint, rank)
		copy(keepSints, srcSints)
		axSet := make(map[int]bool, len(ra.Axes))
		for _, ax := range ra.Axes {
			axSet[ax] = true
			keepSints[ax] = shape.Const(1)
		}

		switch ra.Op {
		case uop.OpAdd:
			// Sum backward: broadcast adjoint back to src shape
			return []*Tensor{adj.ReshapeSints(keepSints).ExpandSints(srcSints)}
		case uop.OpMax:
			// Max backward: route adjoint to argmax positions; split ties equally.
			// Materialize adj before use so that deep adjoint chains (e.g. the FC
			// backward in a convnet) do not inline their leaf buffers into the
			// pool-backward kernel body, which would exceed WebGPU's 8-buffer limit.
			adjMat := adj.Contiguous()
			nodeExp := nodeT.ReshapeSints(keepSints).ExpandSints(srcSints)
			adjExp := adjMat.ReshapeSints(keepSints).ExpandSints(srcSints)
			s0 := src(0)
			mask := s0.CmpEq(nodeExp)
			maskFloat := mask.Cast(adjMat.dtype)
			tieCount := maskFloat.Sum(ra.Axes, true).ExpandSints(srcSints)
			zSrc := FullSints(a, srcSints, 0.0, adjMat.dtype, device)
			return []*Tensor{Where(mask, adjExp.Div(tieCount), zSrc)}
		}
		return nil
	}

	return GradRuleset{rules: m}
}
