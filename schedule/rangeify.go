package schedule

import (
	"fmt"

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
		switch padding := u.Arg().(type) {
		case [][2]int64:
			sh = make([]shape.Sint, len(srcSh))
			for i, s := range srcSh {
				sh[i] = shape.Add(s, shape.Const(padding[i][0]+padding[i][1]))
			}
		case uop.PadSintArg:
			a := u.Arena()
			sh = make([]shape.Sint, len(srcSh))
			for i, s := range srcSh {
				lo := schedShapeDimToSint(a, padding[i][0])
				hi := schedShapeDimToSint(a, padding[i][1])
				sh[i] = shape.Add(shape.Add(s, lo), hi)
			}
		default:
			panic(fmt.Sprintf("schedule/rangeify: OpPad: unexpected arg type %T", u.Arg()))
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
				lo := schedShapeDimToSint(a, p[0])
				hi := schedShapeDimToSint(a, p[1])
				sh[i] = shape.Sub(hi, lo)
			}
		default:
			panic(fmt.Sprintf("schedule/rangeify: OpShrink: unexpected arg type %T", u.Arg()))
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
// Symbolic dims carry (VarName, Mul) structurally (Option B Slice 4); the
// bound expression UOp is rebuilt in a via name lookup + intern-stable Mul
// construction. For Mul>1 the reconstructed bound is OpMul(DefineVar, Const)
// in (var, const) canonical orientation — matches shape.Mul's natural order.
func shapeSintArgToSints(a *uop.Arena, arg uop.ShapeSintArg) []shape.Sint {
	sh := make([]shape.Sint, len(arg))
	for i, d := range arg {
		if d.Sym {
			sh[i] = shape.SymInt{Node: rebuildSymBound(a, d)}
		} else {
			sh[i] = shape.Const(d.V)
		}
	}
	return sh
}

// rebuildSymBound reconstructs the UOp bound expression for a symbolic ShapeDim
// from its (VarName, Mul) encoding. Interning ensures the rebuilt node aliases
// the original whenever the original was constructed in canonical orientation.
func rebuildSymBound(a *uop.Arena, d uop.ShapeDim) uop.UOp {
	defVar, ok := a.FindDefineVar(d.VarName)
	if !ok {
		panic(fmt.Sprintf("schedule: shapeSintArgToSints: DefineVar %q not found in arena", d.VarName))
	}
	if d.Mul <= 1 {
		return defVar
	}
	mulConst := a.New(uop.OpConst, uop.Dtypes.Index, nil, d.Mul, nil)
	return a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{defVar, mulConst}, nil, nil)
}

func cloneShape(s []shape.Sint) []shape.Sint {
	c := make([]shape.Sint, len(s))
	copy(c, s)
	return c
}

// schedShapeDimToSint converts a uop.ShapeDim into a shape.Sint by rebuilding
// the symbolic bound UOp from its (VarName, Mul) encoding. Used by the
// rangeify shape-cache OpPad/OpShrink branches when the pad amount is symbolic.
func schedShapeDimToSint(a *uop.Arena, d uop.ShapeDim) shape.Sint {
	if !d.Sym {
		return shape.Const(d.V)
	}
	return shape.SymInt{Node: rebuildSymBound(a, d)}
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
	a            *uop.Arena
	nextID       int
	kernelRanges []uop.UOp // all RANGE nodes created for the current kernel
}

func newRangeCtx(a *uop.Arena) *rangeCtx {
	return &rangeCtx{a: a}
}

// startKernel resets the per-kernel range accumulators.
func (rc *rangeCtx) startKernel() {
	rc.kernelRanges = rc.kernelRanges[:0]
}

func (rc *rangeCtx) newRange(size int64, t uop.AxisType) uop.UOp {
	id := rc.nextID
	rc.nextID++
	// size-1 dimensions iterate once; a constant 0 avoids a degenerate loop.
	if size == 1 {
		return rc.a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	}
	boundC := rc.a.New(uop.OpConst, uop.Dtypes.Index, nil, size, nil)
	r := rc.a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{boundC}, uop.RangeArg{ID: id, Type: t}, nil)
	rc.kernelRanges = append(rc.kernelRanges, r)
	return r
}

// newSymRange creates a symbolic RANGE node whose bound is the given UOp
// (a DefineVar or an expression over DefineVars). At codegen time the lowerer
// walks the kernel, collects DefineVars via uop.VariablesOf, and assigns
// params_n slots sorted by variable name.
func (rc *rangeCtx) newSymRange(bound uop.UOp, t uop.AxisType) uop.UOp {
	id := rc.nextID
	rc.nextID++
	r := rc.a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{bound}, uop.RangeArg{ID: id, Type: t}, nil)
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
			ranges[i] = rc.newSymRange(sym.Node, t)
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

		// outRanges contains all loop ranges (including OpConst(0) for size-1 dims).
		// rc.kernelRanges contains all OpRange nodes (Loop and Reduce).
		// We want: outRanges + any AxisReduce ranges from rc.kernelRanges.
		var redRanges []uop.UOp
		for _, r := range rc.kernelRanges {
			if r.Op() == uop.OpRange && r.Arg().(uop.RangeArg).Type == uop.AxisReduce {
				redRanges = append(redRanges, r)
			}
		}
		allSrcRanges := append([]uop.UOp(nil), outRanges...)
		allSrcRanges = append(allSrcRanges, redRanges...)

		// BUFFERIZE(indexed_body, *allSrcRanges, arg=BufferizeArg{Removable:true/false})
		removable := false
		if node.Op() == uop.OpReduceAxis {
			removable = true
		}
		bfzSrcs := make([]uop.UOp, 1+len(allSrcRanges))
		bfzSrcs[0] = indexedBody
		copy(bfzSrcs[1:], allSrcRanges)
		bfz := a.New(uop.OpBufferize, u.DType(), bfzSrcs, uop.BufferizeArg{Removable: removable}, nil)
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
