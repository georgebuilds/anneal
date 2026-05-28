package tensor

import (
	"math"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// Backward computes reverse-mode gradients of loss w.r.t. params.
// Each param must be an OpBuffer leaf created by NewLeaf.
// Returns a map from each param to its gradient tensor (same shape/dtype).
// Params not connected to loss are absent from the result.
func Backward(loss *Tensor, params []*Tensor) map[*Tensor]*Tensor {
	if loss == nil || len(params) == 0 {
		return nil
	}

	targets := make(map[uint32]bool, len(params))
	for _, p := range params {
		targets[p.node.Index()] = true
	}

	adjMap := runBackward(loss, targets)

	result := make(map[*Tensor]*Tensor, len(params))
	for _, p := range params {
		if g, ok := adjMap[p.node.Index()]; ok {
			result[p] = g
		}
	}
	return result
}

// ── Backward driver ───────────────────────────────────────────────────────────

func runBackward(loss *Tensor, targets map[uint32]bool) map[uint32]*Tensor {
	device := loss.device
	a := loss.arena()
	prev := a.SetPhase(uop.PhaseBackward)
	defer a.SetPhase(prev)

	// Forward topological order (sources before consumers).
	topo := topoSortUOp(loss.node)

	// Compute Sint shapes for all nodes in forward order.
	shapeCache := make(map[uint32][]shape.Sint, len(topo))
	shapeCache[loss.node.Index()] = loss.ShapeSints()
	for _, u := range topo {
		shapeOfNode(u, shapeCache)
	}

	// Seed: adjoint of loss is ones of the same shape.
	lossSints := shapeCache[loss.node.Index()]
	if lossSints == nil {
		lossSints = []shape.Sint{}
	}
	adjMap := make(map[uint32]*Tensor, len(topo))
	adjMap[loss.node.Index()] = FullSints(a, lossSints, 1.0, loss.dtype, device)

	// Reverse traversal: accumulate adjoints.
	for i := len(topo) - 1; i >= 0; i-- {
		u := topo[i]
		adj, ok := adjMap[u.Index()]
		if !ok {
			continue
		}

		// Reconstruct a Tensor handle for this node so gradient rules can use
		// tensor operations to express derivative UOps.
		nodeSints := shapeCache[u.Index()]
		nodeT := wrapGradTensor(u, nodeSints, u.DType(), device)

		contribs := applyGradRule(u, nodeT, adj, shapeCache, device)
		if contribs == nil {
			continue
		}
		for j, g := range contribs {
			if j >= u.NSrc() || g == nil {
				continue
			}
			src := u.Src(j)
			// Only accumulate for float-dtype sources (integer/bool are not differentiable).
			if !src.DType().IsFloat() {
				continue
			}
			if prev, exists := adjMap[src.Index()]; exists {
				adjMap[src.Index()] = prev.Add(g)
			} else {
				adjMap[src.Index()] = g
			}
		}
	}

	return adjMap
}

// ── Per-op gradient rules ─────────────────────────────────────────────────────

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

// ── Graph utilities ───────────────────────────────────────────────────────────

// topoSortUOp returns the nodes reachable from root in forward topological
// order (each node appears after all its sources).
func topoSortUOp(root uop.UOp) []uop.UOp {
	seen := make(map[uint32]bool)
	var order []uop.UOp

	type frame struct {
		u       uop.UOp
		nextSrc int
	}
	stack := []frame{{root, 0}}

	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u

		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}

		// Push the next unvisited source to process first.
		pushed := false
		for f.nextSrc < u.NSrc() {
			child := u.Src(f.nextSrc)
			f.nextSrc++
			if !seen[child.Index()] {
				stack = append(stack, frame{child, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			order = append(order, u)
			stack = stack[:len(stack)-1]
		}
	}

	return order
}

// shapeOfNode computes and caches the output Sint shape of u.
// All of u's sources must already be in cache (guaranteed when called in topo order).
func shapeOfNode(u uop.UOp, cache map[uint32][]shape.Sint) {
	if _, ok := cache[u.Index()]; ok {
		return
	}

	var sh []shape.Sint
	switch u.Op() {
	case uop.OpConst:
		sh = []shape.Sint{} // scalar

	case uop.OpBuffer:
		// NewLeaf stores []int64; Arange stores int64; NewSymbolicBatchInput stores ShapeSintArg.
		switch v := u.Arg().(type) {
		case []int64:
			sh = intsToSints(v)
		case int64:
			sh = []shape.Sint{shape.Const(v)}
		case uop.ShapeSintArg:
			sh = shapeSintArgToSintsGrad(u.Arena(), v)
		}
		// "randn" string or nil: shape unknown without external context.

	case uop.OpReshape, uop.OpExpand:
		switch v := u.Arg().(type) {
		case []int64:
			sh = intsToSints(v)
		case uop.ShapeSintArg:
			sh = shapeSintArgToSintsGrad(u.Arena(), v)
		}

	case uop.OpPermute:
		srcSh := cache[u.Src(0).Index()]
		perm := u.Arg().([]int64)
		sh = make([]shape.Sint, len(perm))
		for i, p := range perm {
			sh[i] = srcSh[p]
		}

	case uop.OpPad:
		srcSh := cache[u.Src(0).Index()]
		padding := u.Arg().([][2]int64)
		sh = make([]shape.Sint, len(srcSh))
		for i, s := range srcSh {
			if v, ok := s.ConstValue(); ok {
				sh[i] = shape.Const(v + padding[i][0] + padding[i][1])
			} else {
				sh[i] = s // symbolic — pad amount must be 0 (scope guard)
			}
		}

	case uop.OpShrink:
		arg := u.Arg().([][2]int64)
		sh = make([]shape.Sint, len(arg))
		for i, p := range arg {
			sh[i] = shape.Const(p[1] - p[0])
		}

	case uop.OpFlip, uop.OpCast, uop.OpBitcast:
		if u.NSrc() > 0 {
			sh = cache[u.Src(0).Index()]
		}

	case uop.OpReduceAxis:
		srcSh := cache[u.Src(0).Index()]
		ra := u.Arg().(uop.ReduceArg)
		axSet := make(map[int]bool, len(ra.Axes))
		for _, ax := range ra.Axes {
			axSet[ax] = true
		}
		for i, s := range srcSh {
			if !axSet[i] {
				sh = append(sh, s)
			}
		}
		if sh == nil {
			sh = []shape.Sint{} // all axes reduced → scalar
		}

	default:
		// ALU and other ops: same shape as src[0].
		if u.NSrc() > 0 {
			sh = cache[u.Src(0).Index()]
		} else {
			sh = []shape.Sint{}
		}
	}

	cache[u.Index()] = sh
}

// wrapGradTensor creates a Tensor handle from a UOp with an externally provided Sint shape.
// Used only within the backward pass to apply tensor ops when building gradient UOps.
func wrapGradTensor(u uop.UOp, sh []shape.Sint, dtype *uop.DType, device string) *Tensor {
	if sh == nil {
		sh = []shape.Sint{}
	}
	return fromNode(u, shape.NewShapeTrackerSints(sh), dtype, device)
}

// ── Sint helpers ──────────────────────────────────────────────────────────────

// intsToSints converts a concrete []int64 to []shape.Sint.
func intsToSints(ints []int64) []shape.Sint {
	sh := make([]shape.Sint, len(ints))
	for i, v := range ints {
		sh[i] = shape.Const(v)
	}
	return sh
}

// shapeSintArgToSintsGrad converts a ShapeSintArg to []shape.Sint by looking
// up DefineVar nodes by arena index. Mirror of schedule.shapeSintArgToSints.
func shapeSintArgToSintsGrad(a *uop.Arena, arg uop.ShapeSintArg) []shape.Sint {
	sh := make([]shape.Sint, len(arg))
	for i, d := range arg {
		if d.Sym {
			sh[i] = shape.SymInt{Node: a.At(d.VarIdx)}
		} else {
			sh[i] = shape.Const(d.V)
		}
	}
	return sh
}
