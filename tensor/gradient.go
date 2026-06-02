package tensor

import (
	"fmt"

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
	topo := uop.TopoSort(loss.node)

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
		default:
			// NewSymbolicInput: arg=nil, src[0]=DefineVar, shape is [varNode].
			// Mirror of schedule/rangeify.go shapeOfNode's 1-D symbolic case so
			// the gradient pass agrees on rank with the scheduler. Without this,
			// a symbolic-batch backward path produces a rank-0 cache entry and
			// downstream rules (e.g. OpReduceAxis on a [n,D] product) panic on
			// axis-bounds lookup.
			if u.NSrc() > 0 && u.Src(0).Op() == uop.OpDefineVar {
				sh = []shape.Sint{shape.SymInt{Node: u.Src(0)}}
			}
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
		sh = make([]shape.Sint, len(srcSh))
		switch padding := u.Arg().(type) {
		case [][2]int64:
			// Concrete pad: shape[i] grows by lo+hi. If srcSh[i] is symbolic
			// (Option-A bare batch dim with concrete pad amounts on it), wrap
			// the symbolic dim in shape.Add to carry the pad-amount delta —
			// pre-Slice-5 this branch returned srcSh[i] unchanged with a
			// "pad amount must be 0 (scope guard)" comment, which silently
			// dropped the delta. Slice 5 surface now allows pad on a symbolic
			// axis, so propagate properly.
			for i, s := range srcSh {
				delta := padding[i][0] + padding[i][1]
				if v, ok := s.ConstValue(); ok {
					sh[i] = shape.Const(v + delta)
				} else if delta == 0 {
					sh[i] = s
				} else {
					sh[i] = shape.Add(s, shape.Const(delta))
				}
			}
		case uop.PadSintArg:
			// Symbolic pad: shape[i] = srcSh[i] + lo + hi via Sint arithmetic.
			a := u.Arena()
			for i, s := range srcSh {
				lo := shape.SintFromShapeDim(a, padding[i][0])
				hi := shape.SintFromShapeDim(a, padding[i][1])
				sh[i] = shape.Add(shape.Add(s, lo), hi)
			}
		default:
			panic(fmt.Sprintf("tensor/gradient: OpPad: unexpected arg type %T", u.Arg()))
		}

	case uop.OpShrink:
		switch arg := u.Arg().(type) {
		case [][2]int64:
			sh = make([]shape.Sint, len(arg))
			for i, p := range arg {
				sh[i] = shape.Const(p[1] - p[0])
			}
		case uop.ShrinkSintArg:
			a := u.Arena()
			sh = make([]shape.Sint, len(arg))
			for i, p := range arg {
				lo := shape.SintFromShapeDim(a, p[0])
				hi := shape.SintFromShapeDim(a, p[1])
				sh[i] = shape.Sub(hi, lo)
			}
		default:
			panic(fmt.Sprintf("tensor/gradient: OpShrink: unexpected arg type %T", u.Arg()))
		}

	case uop.OpFlip, uop.OpCast, uop.OpBitcast:
		if u.NSrc() > 0 {
			sh = cache[u.Src(0).Index()]
		}

	case uop.OpGather:
		// Torch-gather: output shape = data shape with dim replaced by index shape.
		// src[0] = data, src[1] = index, arg = int64 dim (already normalized
		// by the tensor frontend to be non-negative).
		dataSh := cache[u.Src(0).Index()]
		idxSh := cache[u.Src(1).Index()]
		dim := int(u.Arg().(int64))
		sh = make([]shape.Sint, 0, len(dataSh)-1+len(idxSh))
		sh = append(sh, dataSh[:dim]...)
		sh = append(sh, idxSh...)
		sh = append(sh, dataSh[dim+1:]...)

	case uop.OpScatterAdd:
		// ScatterAdd produces a tensor with the data-template's shape. By
		// convention src[0] is the data-template (zeros-like) carrying the
		// destination shape; downstream srcs are gradient/index/permutation
		// inputs added in Slice D.
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

// shapeSintArgToSintsGrad converts a ShapeSintArg to []shape.Sint via
// shape.SintFromShapeDim. Mirror of schedule.shapeSintArgToSints; shares the
// single source of truth for (VarName, Mul) -> bound-UOp reconstruction in
// uop.RebuildSymBound, so the gradient pass and the scheduler agree on the
// interned bound node for identical ShapeDims.
func shapeSintArgToSintsGrad(a *uop.Arena, arg uop.ShapeSintArg) []shape.Sint {
	sh := make([]shape.Sint, len(arg))
	for i, d := range arg {
		sh[i] = shape.SintFromShapeDim(a, d)
	}
	return sh
}
