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

	// Compute shapes for all nodes in forward order.
	shapeCache := make(map[uint32][]int64, len(topo))
	shapeCache[loss.node.Index()] = loss.Shape()
	for _, u := range topo {
		shapeOfNode(u, shapeCache)
	}

	// Seed: adjoint of loss is ones of the same shape.
	lossShape := shapeCache[loss.node.Index()]
	if lossShape == nil {
		lossShape = []int64{}
	}
	adjMap := make(map[uint32]*Tensor, len(topo))
	adjMap[loss.node.Index()] = Ones(a, lossShape, loss.dtype, device)

	// Reverse traversal: accumulate adjoints.
	for i := len(topo) - 1; i >= 0; i-- {
		u := topo[i]
		adj, ok := adjMap[u.Index()]
		if !ok {
			continue
		}

		// Reconstruct a Tensor handle for this node so gradient rules can use
		// tensor operations to express derivative UOps.
		nodeSh := shapeCache[u.Index()]
		nodeT := wrapGradTensor(u, nodeSh, u.DType(), device)

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
func applyGradRule(u uop.UOp, nodeT *Tensor, adj *Tensor, shapeCache map[uint32][]int64, device string) []*Tensor {
	a := u.Arena()

	// Helper: wrap src i as a Tensor (safe to call in rules that need src values).
	src := func(i int) *Tensor {
		s := u.Src(i)
		sh := shapeCache[s.Index()]
		return wrapGradTensor(s, sh, s.DType(), device)
	}
	// Helper: scalar constant broadcast to adj's shape.
	k := func(v float64) *Tensor {
		return Full(a, adj.Shape(), v, adj.dtype, device)
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
			return []*Tensor{adj.Cast(u.Src(0).DType())}
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
		zeros := Zeros(a, adj.Shape(), adj.dtype, device)
		s0, s1 := src(0), src(1)
		// g0 = adj where s0 == max-output
		g0 := Where(s0.CmpEq(nodeT), adj, zeros)
		// g1 = adj where s1 == max-output AND s0 is not also max (tie-break)
		notTied := Where(s0.CmpEq(nodeT), zeros, adj)
		g1 := Where(s1.CmpEq(nodeT), notTied, zeros)
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
		zeros := Zeros(a, adj.Shape(), adj.dtype, device)
		return []*Tensor{
			nil,
			Where(cond, adj, zeros),
			Where(cond, zeros, adj),
		}

	// ── Movement ops ──────────────────────────────────────────────────────────

	case uop.OpReshape:
		srcSh := shapeCache[u.Src(0).Index()]
		if srcSh == nil {
			srcSh = []int64{}
		}
		return []*Tensor{adj.Reshape(srcSh)}

	case uop.OpExpand:
		// The inverse of expand is sum-reduce over broadcast axes.
		srcSh := shapeCache[u.Src(0).Index()]
		expandedSh := shapeCache[u.Index()]
		rank := len(expandedSh)

		// srcSh always has the same rank as expandedSh (Reshape prepends 1s before Expand).
		var sumAxes []int
		for i := 0; i < rank; i++ {
			si := int64(0)
			if i < len(srcSh) {
				si = srcSh[i]
			}
			if si == 1 && expandedSh[i] > 1 {
				sumAxes = append(sumAxes, i)
			}
		}

		g := adj
		if len(sumAxes) > 0 {
			g = g.Sum(sumAxes, true) // keepdim preserves rank
		}
		g = g.Reshape(srcSh) // drop the kept size-1 dims if srcSh is smaller
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
		padding := u.Arg().([][2]int64)
		srcSh := shapeCache[u.Src(0).Index()]
		shrinkArg := make([][2]int64, len(padding))
		for i, p := range padding {
			shrinkArg[i] = [2]int64{p[0], p[0] + srcSh[i]}
		}
		return []*Tensor{adj.Shrink(shrinkArg)}

	case uop.OpShrink:
		// Shrink → Pad: restore the missing elements as zero.
		shrinkArg := u.Arg().([][2]int64)
		srcSh := shapeCache[u.Src(0).Index()]
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
		srcSh := shapeCache[u.Src(0).Index()]
		if srcSh == nil {
			return nil
		}
		rank := len(srcSh)

		// Build keepdim shape: same as srcSh but with reduced axes set to 1.
		keepSh := make([]int64, rank)
		copy(keepSh, srcSh)
		axSet := make(map[int]bool, len(ra.Axes))
		for _, ax := range ra.Axes {
			axSet[ax] = true
			keepSh[ax] = 1
		}

		switch ra.Op {
		case uop.OpAdd:
			// Sum backward: broadcast adjoint back to src shape.
			return []*Tensor{adj.Reshape(keepSh).Expand(srcSh)}

		case uop.OpMax:
			// Max backward: route adjoint to argmax positions; split ties equally.
			nodeExp := nodeT.Reshape(keepSh).Expand(srcSh)
			adjExp := adj.Reshape(keepSh).Expand(srcSh)
			s0 := src(0)
			mask := s0.CmpEq(nodeExp)
			maskFloat := mask.Cast(adj.dtype)
			tieCount := maskFloat.Sum(ra.Axes, true).Expand(srcSh)
			zeros := Zeros(a, srcSh, adj.dtype, device)
			return []*Tensor{Where(mask, adjExp.Div(tieCount), zeros)}
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

// shapeOfNode computes and caches the output shape of u.
// All of u's sources must already be in cache (guaranteed when called in topo order).
func shapeOfNode(u uop.UOp, cache map[uint32][]int64) {
	if _, ok := cache[u.Index()]; ok {
		return
	}

	var sh []int64
	switch u.Op() {
	case uop.OpConst:
		sh = []int64{} // scalar

	case uop.OpBuffer:
		// NewLeaf stores []int64 shape in arg; Arange stores int64 n.
		switch v := u.Arg().(type) {
		case []int64:
			sh = cloneShape(v)
		case int64:
			sh = []int64{v}
		}
		// "randn" string or nil: shape unknown without external context.

	case uop.OpReshape, uop.OpExpand:
		sh = cloneShape(u.Arg().([]int64))

	case uop.OpPermute:
		srcSh := cache[u.Src(0).Index()]
		perm := u.Arg().([]int64)
		sh = make([]int64, len(perm))
		for i, p := range perm {
			sh[i] = srcSh[p]
		}

	case uop.OpPad:
		srcSh := cache[u.Src(0).Index()]
		padding := u.Arg().([][2]int64)
		sh = make([]int64, len(srcSh))
		for i, s := range srcSh {
			sh[i] = s + padding[i][0] + padding[i][1]
		}

	case uop.OpShrink:
		arg := u.Arg().([][2]int64)
		sh = make([]int64, len(arg))
		for i, p := range arg {
			sh[i] = p[1] - p[0]
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
			sh = []int64{} // all axes reduced → scalar
		}

	default:
		// ALU and other ops: same shape as src[0].
		if u.NSrc() > 0 {
			sh = cache[u.Src(0).Index()]
		} else {
			sh = []int64{}
		}
	}

	cache[u.Index()] = sh
}

// wrapGradTensor creates a Tensor handle from a UOp with an externally provided shape.
// Used only within the backward pass to apply tensor ops when building gradient UOps.
func wrapGradTensor(u uop.UOp, sh []int64, dtype *uop.DType, device string) *Tensor {
	if sh == nil {
		sh = []int64{}
	}
	return fromNode(u, shape.NewShapeTracker(sh), dtype, device)
}
