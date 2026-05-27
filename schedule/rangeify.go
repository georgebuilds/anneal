package schedule

import (
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// ── Topological sort ──────────────────────────────────────────────────────────

// topoSort returns all nodes reachable from root in forward topological order
// (each node appears after all its sources). Iterative post-order DFS.
func topoSort(root uop.UOp) []uop.UOp {
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

// ── Shape map ─────────────────────────────────────────────────────────────────

// buildShapeMap computes the output shape for every node in topo (forward order).
func buildShapeMap(topo []uop.UOp) map[uint32][]shape.Sint {
	cache := make(map[uint32][]shape.Sint, len(topo))
	for _, u := range topo {
		shapeOfNode(u, cache)
	}
	return cache
}

func shapeOfNode(u uop.UOp, cache map[uint32][]shape.Sint) {
	if _, ok := cache[u.Index()]; ok {
		return
	}
	var sh []shape.Sint
	switch u.Op() {
	case uop.OpConst:
		sh = []shape.Sint{} // scalar

	case uop.OpBuffer:
		switch v := u.Arg().(type) {
		case uop.ShapeSintArg:
			// Multi-dim symbolic input ([symbolic, d0, d1, ...]).
			sh = shapeSintArgToSints(u.Arena(), v)
		case []int64:
			sh = cloneShape(shape.AsSints(v))
		case int64:
			sh = []shape.Sint{shape.Const(v)}
		default:
			// 1D symbolic: src[0]=DefineVar, arg=nil.
			if u.NSrc() > 0 && u.Src(0).Op() == uop.OpDefineVar {
				sh = []shape.Sint{shape.SymInt{Node: u.Src(0)}}
			}
		}

	case uop.OpReshape, uop.OpExpand:
		switch v := u.Arg().(type) {
		case []int64:
			sh = cloneShape(shape.AsSints(v))
		case uop.ShapeSintArg:
			sh = shapeSintArgToSints(u.Arena(), v)
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
			sh[i] = shape.Add(s, shape.Const(padding[i][0]+padding[i][1]))
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
			sh = []shape.Sint{}
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

// shapeSintArgToSints converts a ShapeSintArg to []shape.Sint.
// Symbolic dims are reconstructed as SymInt by looking up the DefineVar node
// by arena index in a.
func shapeSintArgToSints(a *uop.Arena, arg uop.ShapeSintArg) []shape.Sint {
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

func cloneShape(s []shape.Sint) []shape.Sint {
	c := make([]shape.Sint, len(s))
	copy(c, s)
	return c
}

// ── Consumer map ──────────────────────────────────────────────────────────────

// buildConsumerMap returns a map from each node's index to the set of unique
// parent node indices that directly reference it.
func buildConsumerMap(topo []uop.UOp) map[uint32]map[uint32]struct{} {
	result := make(map[uint32]map[uint32]struct{}, len(topo))
	for _, u := range topo {
		dedupe := make(map[uint32]bool)
		for i := 0; i < u.NSrc(); i++ {
			srcIdx := u.Src(i).Index()
			if dedupe[srcIdx] {
				continue
			}
			dedupe[srcIdx] = true
			if result[srcIdx] == nil {
				result[srcIdx] = make(map[uint32]struct{})
			}
			result[srcIdx][u.Index()] = struct{}{}
		}
	}
	return result
}

// ── Realize map ───────────────────────────────────────────────────────────────

// hardRealizeOps are ALWAYS_CONTIGUOUS: they must force a kernel boundary.
// REDUCE_AXIS is included conservatively — it changes the iteration space.
var hardRealizeOps = map[uop.Op]bool{
	uop.OpContiguous: true,
	uop.OpAssign:     true,
	uop.OpBufferView: true,
	uop.OpEncDec:     true,
	uop.OpReduceAxis: true,
}

// buildRealizeMap marks nodes that must produce a materialised buffer.
func buildRealizeMap(sink uop.UOp, topo []uop.UOp) map[uint32]bool {
	realize := make(map[uint32]bool)
	for i := 0; i < sink.NSrc(); i++ {
		realize[sink.Src(i).Index()] = true
	}
	for _, u := range topo {
		if hardRealizeOps[u.Op()] {
			realize[u.Index()] = true
		}
	}
	return realize
}

// ── Range context ─────────────────────────────────────────────────────────────

type rangeCtx struct {
	a             *uop.Arena
	nextID        int
	kernelRanges  []uop.UOp // all RANGE nodes created for the current kernel
	symParamCount int        // symbolic params allocated for the current kernel
}

func newRangeCtx(a *uop.Arena) *rangeCtx {
	return &rangeCtx{a: a}
}

// startKernel resets the per-kernel range accumulators.
func (rc *rangeCtx) startKernel() {
	rc.kernelRanges = rc.kernelRanges[:0]
	rc.symParamCount = 0
}

func (rc *rangeCtx) newRange(size int64, t uop.AxisType) uop.UOp {
	id := rc.nextID
	rc.nextID++
	// size-1 dimensions iterate once; a constant 0 avoids a degenerate loop.
	if size == 1 {
		return rc.a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	}
	r := rc.a.New(uop.OpRange, uop.Dtypes.Index, nil, uop.RangeArg{ID: id, Size: size, Type: t}, nil)
	rc.kernelRanges = append(rc.kernelRanges, r)
	return r
}

// newSymRange creates a symbolic RANGE node whose bound is provided at dispatch
// via the params_n buffer at slot SymParamIdx. VarName records the DefineVar name
// so RunSymbolic can look it up in the binding map.
func (rc *rangeCtx) newSymRange(varName string, t uop.AxisType) uop.UOp {
	id := rc.nextID
	rc.nextID++
	symIdx := rc.symParamCount
	rc.symParamCount++
	r := rc.a.New(uop.OpRange, uop.Dtypes.Index, nil, uop.RangeArg{
		ID: id, Size: 0, Type: t, Symbolic: true, SymParamIdx: symIdx, VarName: varName,
	}, nil)
	rc.kernelRanges = append(rc.kernelRanges, r)
	return r
}

func (rc *rangeCtx) freshRanges(sh []shape.Sint, t uop.AxisType) []uop.UOp {
	ranges := make([]uop.UOp, len(sh))
	for i, s := range sh {
		if v, ok := s.ConstValue(); ok {
			ranges[i] = rc.newRange(v, t)
		} else {
			sym := s.(shape.SymInt)
			varName := sym.Node.Arg().(uop.VarArg).Name
			ranges[i] = rc.newSymRange(varName, t)
		}
	}
	return ranges
}

// ── runRangeify: passes 2–4 (realize map + range threading + BUFFERIZE) ──────

// runRangeify computes the realize map, propagates range indices through every
// kernel subgraph via indexExprNode, and wraps each realize point in BUFFERIZE.
//
// The BUFFERIZE produced here carries:
//
//	src[0]   = fully-indexed kernel body (movement ops dissolved, INDEX at leaves)
//	src[1..] = all RANGE nodes for the kernel (AxisLoop first, then AxisReduce)
func runRangeify(sink uop.UOp) uop.UOp {
	a := sink.Arena()
	topo := topoSort(sink)
	shapeMap := buildShapeMap(topo)
	realizeMap := buildRealizeMap(sink, topo)
	rc := newRangeCtx(a)

	// rebuild maps old node index → new node index (upstream BUFFERIZE-wrapped nodes
	// appear as upstream boundaries that indexExprNode treats as leaf accesses).
	rebuild := make(map[uint32]uint32, len(topo))

	for _, u := range topo {
		// Rebuild this node with any already-wrapped upstream children.
		srcs := make([]uop.UOp, u.NSrc())
		childChanged := false
		for i := 0; i < u.NSrc(); i++ {
			ch := u.Src(i)
			if newIdx, ok := rebuild[ch.Index()]; ok {
				srcs[i] = a.At(newIdx)
				if newIdx != ch.Index() {
					childChanged = true
				}
			} else {
				srcs[i] = ch
			}
		}
		var node uop.UOp
		if childChanged {
			node = a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag())
		} else {
			node = u
		}

		if !realizeMap[u.Index()] {
			// Propagate shape to the rebuilt node so that indexExprNode can
			// look it up when this node appears as a source in a downstream kernel.
			if node.Index() != u.Index() {
				shapeMap[node.Index()] = shapeMap[u.Index()]
			}
			rebuild[u.Index()] = node.Index()
			continue
		}

		// Realize point: create output ranges and thread them through the kernel body.
		outShape := shapeMap[u.Index()]
		if outShape == nil {
			outShape = []shape.Sint{}
		}

		rc.startKernel()
		outRanges := rc.freshRanges(outShape, uop.AxisLoop)
		indexedBody := indexExprNode(a, node, outRanges, shapeMap, rc, 0)
		allRanges := append([]uop.UOp(nil), rc.kernelRanges...) // snapshot

		// BUFFERIZE(indexed_body, *all_ranges, arg=BufferizeArg{Removable:false})
		bfzSrcs := make([]uop.UOp, 1+len(allRanges))
		bfzSrcs[0] = indexedBody
		copy(bfzSrcs[1:], allRanges)
		bfz := a.New(uop.OpBufferize, u.DType(), bfzSrcs, uop.BufferizeArg{Removable: false}, nil)
		// Record the BUFFERIZE shape so downstream Reshape operations can compute
		// correct flat→per-dim index decomposition via unflatIndex. Without this,
		// shapeMap[bfz.Index()] would be nil and any Reshape(BUFFERIZE, ...) would
		// degenerate to INDEX(BUFFERIZE) with no arguments, reading element 0 always.
		shapeMap[bfz.Index()] = outShape
		rebuild[u.Index()] = bfz.Index()
	}

	if newIdx, ok := rebuild[sink.Index()]; ok {
		return a.At(newIdx)
	}
	return sink
}
