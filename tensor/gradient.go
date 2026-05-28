package tensor

import (
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// Backward computes reverse-mode gradients of loss w.r.t. params.
// Each param must be an OpBuffer leaf created by NewLeaf.
// Returns a map from each param to its gradient tensor (same shape/dtype).
// Params not connected to loss are absent from the result.
func Backward(loss *Tensor, params []*Tensor) map[*Tensor]*Tensor {
	grads, _ := BackwardWithTrace(loss, params)
	return grads
}

// BackwardWithTrace runs the gradient pass identically to Backward and additionally
// captures a GradTrace recording which gradient rule fired on each forward node in
// reverse-topological order. Use this for visualization and debugging; prefer Backward
// for production hot paths.
func BackwardWithTrace(loss *Tensor, leaves []*Tensor) (map[*Tensor]*Tensor, *GradTrace) {
	if loss == nil || len(leaves) == 0 {
		return nil, nil
	}

	targets := make(map[uint32]bool, len(leaves))
	for _, p := range leaves {
		targets[p.node.Index()] = true
	}

	trace := &GradTrace{}
	adjMap := runBackward(loss, targets, trace)

	result := make(map[*Tensor]*Tensor, len(leaves))
	for _, p := range leaves {
		if g, ok := adjMap[p.node.Index()]; ok {
			result[p] = g
		}
	}
	return result, trace
}

// ── Backward driver ───────────────────────────────────────────────────────────

func runBackward(loss *Tensor, targets map[uint32]bool, trace *GradTrace) map[uint32]*Tensor {
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

		contribs := Gradient.Dispatch(u, nodeT, adj, shapeCache, device)
		if trace != nil {
			ev := GradTraceEvent{
				Seq:            len(trace.Events),
				ForwardNodeIdx: u.Index(),
				ForwardOp:      u.Op(),
				AdjointIdx:     adj.node.Index(),
				ProducedIdx:    make([]uint32, len(contribs)),
			}
			for j, c := range contribs {
				if c == nil {
					ev.ProducedIdx[j] = TraceSentinel
				} else {
					ev.ProducedIdx[j] = c.node.Index()
				}
			}
			trace.Events = append(trace.Events, ev)
		}
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
