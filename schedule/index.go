package schedule

import (
	"math"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// indexExprNode propagates multi-dimensional indices through a kernel subgraph,
// dissolving all six movement ops into range arithmetic and returning a fully-
// indexed kernel body.
//
// The returned expression contains:
//   - INDEX(BUFFER, *arithmetic_indices) at every leaf buffer access
//   - INDEX(BUFFERIZE, *indices) at every upstream-kernel boundary
//   - REDUCE(acc_op, elem_expr, *reduce_ranges) for each accumulation
//   - No remaining movement ops (OpReshape, OpPermute, OpExpand, OpPad, OpShrink, OpFlip)
//
// Any AxisReduce RANGE variables created for ReduceAxis nodes are registered in
// rc.kernelRanges so the caller can include them in the enclosing END loop nest.
//
// fillOp is the enclosing reduce op (e.g. OpAdd, OpMax) if this call is
// evaluating the source of a ReduceAxis — used by OpPad to substitute the
// reduce identity element instead of 0 for out-of-bounds positions.
// Op(0) means "no reduce context; use 0 as the pad fill."
// Movement ops propagate fillOp unchanged; elementwise/ALU ops reset it to 0.
//
// This is the Go analogue of tinygrad's pm_mops PatternMatcher + apply_movement_op.
func indexExprNode(a *uop.Arena, expr uop.UOp, indices []uop.UOp, shapeMap map[uint32][]shape.Sint, rc *rangeCtx, fillOp uop.Op) uop.UOp {
	switch expr.Op() {

	// ── leaf accesses ─────────────────────────────────────────────────────
	case uop.OpBuffer, uop.OpBufferize:
		// Leaf or upstream-kernel boundary: INDEX(leaf, *indices)
		srcs := make([]uop.UOp, 1+len(indices))
		srcs[0] = expr
		copy(srcs[1:], indices)
		return a.New(uop.OpIndex, expr.DType(), srcs, nil, nil)

	case uop.OpConst, uop.OpRange, uop.OpLUnique, uop.OpDevice, uop.OpDefineVar:
		// Scalar/meta nodes — not position-dependent
		return expr

	case uop.OpGatherIdx:
		// Scalar (Index-dtype) indirect-index expression. Its subtree (src[0],
		// an OpIndex over the index BUFFER) has already been fully indexed at
		// the time it was constructed by the OpGather rewrite below; the
		// positional carriers in src[1:] exist only to mark this node as
		// position-dependent so it is not hoisted by intern-driven CSE. Pass
		// it through unchanged; the codegen lowerer handles emission.
		return expr

	// ── movement ops — dissolve into index arithmetic ─────────────────────

	case uop.OpReshape:
		// Flat index from output (new) shape, then decompose into source (old) shape.
		srcSints := shapeMap[expr.Src(0).Index()]
		switch v := expr.Arg().(type) {
		case []int64:
			flat := flatIndex(a, indices, v)
			srcIndices := unflatIndex(a, flat, shape.AsInts(srcSints))
			return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		case uop.ShapeSintArg:
			dstSints := shapeSintArgToSints(expr.Arena(), v)
			flat := flatIndexSints(a, indices, dstSints)
			srcIndices := unflatIndexSints(a, flat, srcSints)
			return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		}
		panic("schedule/index: OpReshape: unexpected arg type")

	case uop.OpPermute:
		// perm[i] = j means "output dim i comes from source dim j".
		// Source dim j is accessed by the output index for the position k where perm[k]=j.
		switch perm := expr.Arg().(type) {
		case []int64:
			srcIndices := make([]uop.UOp, len(perm))
			for i, p := range perm {
				srcIndices[p] = indices[i]
			}
			return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		}
		panic("schedule/index: OpPermute: unexpected arg type")

	case uop.OpExpand:
		// Broadcast: source dims that were size 1 map to index 0.
		// Use ConstValue() to avoid panicking on symbolic dims (SymInt is never size 1).
		srcSints := shapeMap[expr.Src(0).Index()]
		srcIndices := make([]uop.UOp, len(srcSints))
		for i, s := range srcSints {
			if cv, ok := s.ConstValue(); ok && cv == 1 {
				srcIndices[i] = a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
			} else {
				srcIndices[i] = indices[i]
			}
		}
		return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)

	case uop.OpShrink:
		// Slice: offset each index by its lower bound. Concrete arg keeps the
		// pre-Slice-5 fast path (byte-identical static behaviour). Symbolic arg
		// (ShrinkSintArg) builds an OpAdd whose offset operand may itself be a
		// SymInt expression rebuilt from (VarName, Mul) via padOffsetUOp.
		switch bounds := expr.Arg().(type) {
		case [][2]int64:
			srcIndices := make([]uop.UOp, len(bounds))
			for i, b := range bounds {
				if b[0] == 0 {
					srcIndices[i] = indices[i]
				} else {
					off := a.New(uop.OpConst, uop.Dtypes.Index, nil, b[0], nil)
					srcIndices[i] = a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{indices[i], off}, nil, nil)
				}
			}
			return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		case uop.ShrinkSintArg:
			srcIndices := make([]uop.UOp, len(bounds))
			for i, b := range bounds {
				off := dimToUOp(a, shape.SintFromShapeDim(a, b[0]))
				if isZeroConst(off) {
					srcIndices[i] = indices[i]
				} else {
					srcIndices[i] = a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{indices[i], off}, nil, nil)
				}
			}
			return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		}
		panic("schedule/index: OpShrink: unexpected arg type")

	case uop.OpPad:
		// Pad: validity guard + source index = r - lo; out-of-bounds → fill.
		// Concrete and symbolic args share the same predicate shape; the
		// difference is that lo/hi/srcShape[i] become UOp expressions instead
		// of compile-time constants when symbolic. Concrete args carrying a
		// symbolic source shape (e.g. concrete Pad on the static axis of an
		// [n,4] tensor) flow through the Sint path too so srcShape never hits
		// AsInts.
		srcSints := shapeMap[expr.Src(0).Index()]
		var padLo []uop.UOp // per-axis offset (== lo); nil when lo == 0
		var padHi []uop.UOp // per-axis upper-bound expression (lo + srcShape[i])
		var nonZeroLo []bool
		var nonZeroHi []bool
		// Normalise both arg variants into a slice of (loSint, hiSint) so the
		// per-axis emission logic below is shared.
		var loSints []shape.Sint
		var hiSints []shape.Sint
		switch padding := expr.Arg().(type) {
		case [][2]int64:
			loSints = make([]shape.Sint, len(padding))
			hiSints = make([]shape.Sint, len(padding))
			for i, p := range padding {
				loSints[i] = shape.Const(p[0])
				hiSints[i] = shape.Const(p[1])
			}
		case uop.PadSintArg:
			loSints = make([]shape.Sint, len(padding))
			hiSints = make([]shape.Sint, len(padding))
			for i, p := range padding {
				loSints[i] = shape.SintFromShapeDim(a, p[0])
				hiSints[i] = shape.SintFromShapeDim(a, p[1])
			}
		default:
			panic("schedule/index: OpPad: unexpected arg type")
		}
		n := len(loSints)
		padLo = make([]uop.UOp, n)
		padHi = make([]uop.UOp, n)
		nonZeroLo = make([]bool, n)
		nonZeroHi = make([]bool, n)
		for i := 0; i < n; i++ {
			loV, loConcrete := loSints[i].ConstValue()
			hiV, hiConcrete := hiSints[i].ConstValue()
			loIsZero := loConcrete && loV == 0
			if !loIsZero {
				padLo[i] = dimToUOp(a, loSints[i])
				nonZeroLo[i] = true
			}
			// Upper bound r < lo + srcSize. Only emit when hi != 0; symbolic hi
			// is conservatively non-zero.
			hiIsZero := hiConcrete && hiV == 0
			if !hiIsZero {
				srcSz := dimToUOp(a, srcSints[i])
				var upper uop.UOp
				if loIsZero {
					upper = srcSz
				} else {
					upper = a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{padLo[i], srcSz}, nil, nil)
				}
				padHi[i] = upper
				nonZeroHi[i] = true
			}
		}

		srcIndices := make([]uop.UOp, len(padLo))
		var validConds []uop.UOp
		for i := range padLo {
			r := indices[i]
			if nonZeroLo[i] {
				srcIndices[i] = a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{r, padLo[i]}, nil, nil)
				// r >= lo  ⇔  (lo - 1) < r. Express as Sub(lo, Const(1)) so the
				// constant path is byte-identical (interning collapses
				// Sub(Const(L), Const(1)) → Const(L-1)) and the symbolic path
				// builds a real UOp expression that emits correctly via emitALU.
				one := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
				loMinus1 := a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{padLo[i], one}, nil, nil)
				validConds = append(validConds, a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{loMinus1, r}, nil, nil))
			} else {
				srcIndices[i] = r
			}
			if nonZeroHi[i] {
				validConds = append(validConds, a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{r, padHi[i]}, nil, nil))
			}
		}
		inner := indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		if len(validConds) == 0 {
			return inner
		}
		valid := validConds[0]
		for _, c := range validConds[1:] {
			valid = a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{valid, c}, nil, nil)
		}
		fill := identityConst(a, fillOp, expr.DType())
		return a.New(uop.OpWhere, expr.DType(), []uop.UOp{valid, inner, fill}, nil, nil)

	case uop.OpFlip:
		// Mirror: index r → (size-1) - r for flipped axes. Concrete shape uses
		// a baked-in Const(size-1); symbolic shape builds Sub(srcSize, Const(1))
		// so the operand can be a SymInt expression.
		switch axisFlags := expr.Arg().(type) {
		case []int64:
			srcSints := shapeMap[expr.Src(0).Index()]
			srcIndices := make([]uop.UOp, len(axisFlags))
			for i, f := range axisFlags {
				if f != 0 {
					if v, ok := srcSints[i].ConstValue(); ok {
						sm1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, v-1, nil)
						srcIndices[i] = a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{sm1, indices[i]}, nil, nil)
					} else {
						srcSz := dimToUOp(a, srcSints[i])
						one := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
						sm1 := a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{srcSz, one}, nil, nil)
						srcIndices[i] = a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{sm1, indices[i]}, nil, nil)
					}
				} else {
					srcIndices[i] = indices[i]
				}
			}
			return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)
		}
		panic("schedule/index: OpFlip: unexpected arg type")

	// ── indirect indexing (gather) ────────────────────────────────────────

	case uop.OpGather:
		// Dissolve OpGather(data, idx) into OpIndex(data, ..., OpGatherIdx(...), ...)
		// following design §5.
		//
		// Output index layout (positional carriers from the caller):
		//   indices[0..dim)              → data dims [0..dim)
		//   indices[dim..dim+idxRank)    → idx dims  [0..idxRank)
		//   indices[dim+idxRank..)       → data dims [dim+1..]
		//
		// Inner load: OpIndex(IDX, indices[dim..dim+idxRank]). Built by
		// recursively indexing through expr.Src(1) (the index tensor subtree),
		// which terminates at the IDX OpBuffer leaf.
		//
		// OpGatherIdx wraps the inner load; its positional carriers are the
		// caller's full output indices so downstream uses see a fully
		// position-dependent scalar (no CSE-hoist out of the loop).
		//
		// Outer load: OpIndex(data, ..., OpGatherIdx, ...) built by recursing
		// into expr.Src(0) (the data tensor subtree) with the rewritten
		// positional indices.
		dim := int(expr.Arg().(int64))
		idxSrcShape := shapeMap[expr.Src(1).Index()]
		idxRank := len(idxSrcShape)

		// Inner load indices: positions corresponding to the index tensor's shape.
		innerIndices := make([]uop.UOp, idxRank)
		copy(innerIndices, indices[dim:dim+idxRank])
		innerLoad := indexExprNode(a, expr.Src(1), innerIndices, shapeMap, rc, 0)

		// Wrap in OpGatherIdx; carriers in src[1:] are the full output indices.
		gatherSrcs := make([]uop.UOp, 1+len(indices))
		gatherSrcs[0] = innerLoad
		copy(gatherSrcs[1:], indices)
		gIdx := a.New(uop.OpGatherIdx, uop.Dtypes.Index, gatherSrcs, nil, nil)

		// Build the data-side indices: pre-dim from indices[:dim], the
		// gather scalar at slot dim, post-dim from indices[dim+idxRank:].
		dataSrcShape := shapeMap[expr.Src(0).Index()]
		dataRank := len(dataSrcShape)
		dataIndices := make([]uop.UOp, dataRank)
		copy(dataIndices[:dim], indices[:dim])
		dataIndices[dim] = gIdx
		copy(dataIndices[dim+1:], indices[dim+idxRank:])

		return indexExprNode(a, expr.Src(0), dataIndices, shapeMap, rc, fillOp)

	case uop.OpScatterAdd:
		// Dissolve OpScatterAdd(zeros, grad, sortedIdx, perm) into a per-
		// output-position reduce body following design §6 dispatch geometry
		// (a). Output shape equals zeros' shape == grad's shape with dim 0
		// replaced by V. Slice D v1 restricts dim to 0 and idx to rank 1,
		// so:
		//
		//   outShape  = [V, *trailing]
		//   gradShape = [B, *trailing]
		//
		// For each output position (v, *t) the body computes:
		//
		//   sum_{b in [0, B)} where(
		//     sortedIdx[b] == v,
		//     grad[perm[b], *t],
		//     0.0,
		//   )
		//
		// Distinct (v, *t) write-positions touch disjoint output addresses,
		// so the resulting kernel is race-free without atomics. The reduce
		// is correctness-preserving for any execution order within the b
		// loop because addition is commutative and we accumulate into a
		// private register, not a shared cell.
		dim := int(expr.Arg().(int64))
		if dim != 0 {
			panic("schedule/index: OpScatterAdd: only dim=0 is supported in Slice D")
		}
		gradNode := expr.Src(1)
		sortedNode := expr.Src(2)
		permNode := expr.Src(3)

		gradShape := shapeMap[gradNode.Index()]
		if len(gradShape) == 0 {
			panic("schedule/index: OpScatterAdd: grad source has no shape")
		}
		// reduce range over B (== gradShape[0])
		var rB uop.UOp
		bDim := gradShape[0]
		if v, ok := bDim.ConstValue(); ok {
			rB = rc.newRange(v, uop.AxisReduce)
		} else {
			sym := bDim.(shape.SymInt)
			rB = rc.newSymRange(sym.Node, uop.AxisReduce)
		}

		// sortedIdx[b] : INDEX(sortedNode, rB) -> i32 (carried as Index dtype)
		sortedLoad := indexExprNode(a, sortedNode, []uop.UOp{rB}, shapeMap, rc, 0)
		// perm[b] : INDEX(permNode, rB) -> i32
		permLoad := indexExprNode(a, permNode, []uop.UOp{rB}, shapeMap, rc, 0)

		// Wrap permLoad in OpGatherIdx so the lowerer treats it as an
		// indirect scalar coordinate (same pattern as OpGather above).
		// Carriers in src[1:] are the full output indices.
		gatherSrcs := make([]uop.UOp, 1+len(indices))
		gatherSrcs[0] = permLoad
		copy(gatherSrcs[1:], indices)
		gIdxPerm := a.New(uop.OpGatherIdx, uop.Dtypes.Index, gatherSrcs, nil, nil)

		// grad[perm[b], *t] : INDEX(gradNode, gIdxPerm, t1, t2, ...)
		gradIndices := make([]uop.UOp, len(gradShape))
		gradIndices[0] = gIdxPerm
		copy(gradIndices[1:], indices[1:])
		gradLoad := indexExprNode(a, gradNode, gradIndices, shapeMap, rc, 0)

		// match = (sortedIdx[b] == v) where v == indices[0] (the V output range)
		v := indices[0]
		match := a.New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{sortedLoad, v}, nil, nil)

		// contrib = where(match, grad[perm[b], *t], 0)
		zero := identityConst(a, uop.OpAdd, expr.DType())
		contrib := a.New(uop.OpWhere, expr.DType(), []uop.UOp{match, gradLoad, zero}, nil, nil)

		// Reduce(contrib, OpAdd, rB)
		return a.New(uop.OpReduce, expr.DType(), []uop.UOp{contrib, rB}, uop.OpAdd, nil)

	// ── materialization hint ──────────────────────────────────────────────

	case uop.OpContiguous:
		// Contiguous is a transparent materialization hint: dissolve it at
		// index-expression time exactly like a movement op, passing the indices
		// straight through to the source.
		return indexExprNode(a, expr.Src(0), indices, shapeMap, rc, fillOp)

	// ── reduce ────────────────────────────────────────────────────────────

	case uop.OpReduceAxis:
		// Creates AxisReduce range vars for the reduced axes, then indexes through
		// the source. Returns a kernel-level REDUCE(acc_op, elem_expr, *reduce_ranges).
		switch ra := expr.Arg().(type) {
		case uop.ReduceArg:
			srcSints := shapeMap[expr.Src(0).Index()]

			// Reduce ranges, one per reduced axis.
			// Symbolic axes use a symbolic RANGE so the WGSL loop reads params_n at runtime.
			reduceRanges := make([]uop.UOp, len(ra.Axes))
			reducedAt := make(map[int]uop.UOp, len(ra.Axes))
			for i, ax := range ra.Axes {
				s := srcSints[ax]
				var rr uop.UOp
				if v, ok := s.ConstValue(); ok {
					rr = rc.newRange(v, uop.AxisReduce)
				} else {
					sym := s.(shape.SymInt)
					rr = rc.newSymRange(sym.Node, uop.AxisReduce)
				}
				reduceRanges[i] = rr
				reducedAt[ax] = rr
			}

			// Build full source index: reduced dims → reduce range; others → output index.
			fullIndices := make([]uop.UOp, len(srcSints))
			outIdx := 0
			for i := range srcSints {
				if rr, ok := reducedAt[i]; ok {
					fullIndices[i] = rr
				} else {
					fullIndices[i] = indices[outIdx]
					outIdx++
				}
			}

			// Pass ra.Op as fillOp so any Pad in the source uses the correct
			// reduce identity element (not 0) for out-of-bounds positions.
			indexedSrc := indexExprNode(a, expr.Src(0), fullIndices, shapeMap, rc, ra.Op)

			// REDUCE(acc_op, elem_expr, *reduce_ranges)
			reduceSrcs := make([]uop.UOp, 1+len(reduceRanges))
			reduceSrcs[0] = indexedSrc
			copy(reduceSrcs[1:], reduceRanges)
			return a.New(uop.OpReduce, expr.DType(), reduceSrcs, ra.Op, nil)
		}
		panic("schedule/index: OpReduceAxis: unexpected arg type")

	// ── elementwise / ALU — distribute index through all sources ──────────

	default:
		// Elementwise ops break the reduce context: a Pad behind an ALU should
		// still use 0 as its fill (ALU(identity) ≠ identity in general).
		newSrcs := make([]uop.UOp, expr.NSrc())
		for i := 0; i < expr.NSrc(); i++ {
			src := expr.Src(i)
			switch src.Op() {
			case uop.OpConst, uop.OpRange, uop.OpLUnique:
				// Scalar — no indexing
				newSrcs[i] = src
			default:
				newSrcs[i] = indexExprNode(a, src, indices, shapeMap, rc, 0)
			}
		}
		return a.New(expr.Op(), expr.DType(), newSrcs, expr.Arg(), expr.Tag())
	}
}

// ── index arithmetic helpers ──────────────────────────────────────────────────

// rowMajorStrides returns row-major strides for a static int64 shape:
// strides[i] = prod(shape[i+1:]). Last dim is 1. Shared by flatIndex and
// unflatIndex so the int64-stride accumulation lives in one place.
func rowMajorStrides(shape []int64) []int64 {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int64, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// flatIndex computes the row-major flat index from multi-dim indices and shape.
// flatIndex([r0, r1], [n0, n1]) = r0*n1 + r1
func flatIndex(a *uop.Arena, indices []uop.UOp, shape []int64) uop.UOp {
	if len(indices) == 0 {
		return a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	}
	if len(indices) == 1 {
		return indices[0]
	}
	strides := rowMajorStrides(shape)
	var result uop.UOp
	for i, r := range indices {
		s := strides[i]
		var term uop.UOp
		if s == 1 {
			term = r
		} else {
			sc := a.New(uop.OpConst, uop.Dtypes.Index, nil, s, nil)
			term = a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{r, sc}, nil, nil)
		}
		if !result.Valid() {
			result = term
		} else {
			result = a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{result, term}, nil, nil)
		}
	}
	return result
}

// unflatIndex decomposes a flat row-major index into per-dim indices for shape.
// unflatIndex(r_flat, [n0, n1]) = [r_flat/n1, r_flat%n1]
func unflatIndex(a *uop.Arena, flat uop.UOp, shape []int64) []uop.UOp {
	if len(shape) == 0 {
		return nil
	}
	if len(shape) == 1 {
		return []uop.UOp{flat}
	}
	strides := rowMajorStrides(shape)
	out := make([]uop.UOp, len(shape))
	for i, s := range shape {
		stride := strides[i]
		var divided uop.UOp
		if stride == 1 {
			divided = flat
		} else {
			sc := a.New(uop.OpConst, uop.Dtypes.Index, nil, stride, nil)
			divided = a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{flat, sc}, nil, nil)
		}
		// Always take modulo: isolates this dim even when stride==1 (last dim),
		// preventing the flat index from leaking into the per-dim value.
		szc := a.New(uop.OpConst, uop.Dtypes.Index, nil, s, nil)
		out[i] = a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{divided, szc}, nil, nil)
	}
	return out
}

// sintStrides computes row-major strides from a Sint shape slice.
//
// Each stride is itself a shape.Sint: concrete when every dim to its right
// is concrete, symbolic otherwise. The accumulator uses shape.Mul, which
// preserves the concrete fast-path (Mul(1, x) = x; ConstInt × ConstInt
// folds) and builds symbolic UOp nodes when a symbolic dim enters the
// product. The position of the symbolic dim is recovered structurally via
// s.ConstValue() — strides for dims to the left of a symbolic dim now
// carry the symbolic factor.
func sintStrides(sh []shape.Sint) []shape.Sint {
	n := len(sh)
	strides := make([]shape.Sint, n)
	acc := shape.Const(1)
	for i := n - 1; i >= 0; i-- {
		strides[i] = acc
		acc = shape.Mul(acc, sh[i])
	}
	return strides
}

// flatIndexSints computes a row-major flat index from multi-dim indices and a
// Sint shape. Strides are extracted via sintStrides; symbolic factors become
// real UOp operands via dimToUOp.
func flatIndexSints(a *uop.Arena, indices []uop.UOp, sh []shape.Sint) uop.UOp {
	if len(indices) == 0 {
		return a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	}
	if len(indices) == 1 {
		return indices[0]
	}
	strides := sintStrides(sh)
	var result uop.UOp
	for i, r := range indices {
		s := strides[i]
		var term uop.UOp
		if v, ok := s.ConstValue(); ok && v == 1 {
			term = r
		} else {
			sUOp := dimToUOp(a, s)
			term = a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{r, sUOp}, nil, nil)
		}
		if !result.Valid() {
			result = term
		} else {
			result = a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{result, term}, nil, nil)
		}
	}
	return result
}

// unflatIndexSints decomposes a flat index into per-dim indices for a Sint
// shape. Strides come from sintStrides (Sint) so non-outermost symbolic dims
// produce correct divisors.
//
// Mod is applied per-dim with one exception: when the outermost dim (i==0)
// is symbolic, the quotient is returned directly. Rationale: a Mod on the
// symbolic outermost dim would force a runtime read of the symbolic bound
// inside the inner loop. The quotient is exact for valid flat indices
// (flat < total = sh[0] * strides[0]). For symbolic dims at i>0 the Mod is
// required for correctness — flat / stride is not bounded by sh[i].
func unflatIndexSints(a *uop.Arena, flat uop.UOp, sh []shape.Sint) []uop.UOp {
	if len(sh) == 0 {
		return nil
	}
	if len(sh) == 1 {
		return []uop.UOp{flat}
	}
	strides := sintStrides(sh)
	out := make([]uop.UOp, len(sh))
	for i, s := range sh {
		stride := strides[i]
		var divided uop.UOp
		if v, ok := stride.ConstValue(); ok && v == 1 {
			divided = flat
		} else {
			strideUOp := dimToUOp(a, stride)
			divided = a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{flat, strideUOp}, nil, nil)
		}
		_, dimConcrete := s.ConstValue()
		if i == 0 && !dimConcrete {
			// Outermost symbolic dim: skip Mod to avoid a runtime bound fetch
			// in the inner loop. Quotient is exact for valid flat indices.
			out[i] = divided
		} else {
			sizeUOp := dimToUOp(a, s)
			out[i] = a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{divided, sizeUOp}, nil, nil)
		}
	}
	return out
}

// dimToUOp materialises a shape.Sint as an Index-dtype UOp in arena a.
// Concrete dims become OpConst(V); symbolic dims return the backing UOp node.
// Used by OpPad/OpShrink/OpFlip symbolic paths where pad/shrink amounts and
// source-shape dims may carry SymInts.
func dimToUOp(a *uop.Arena, s shape.Sint) uop.UOp {
	if cv, ok := s.ConstValue(); ok {
		return a.New(uop.OpConst, uop.Dtypes.Index, nil, cv, nil)
	}
	sym := s.(shape.SymInt)
	return sym.Node
}

// isZeroConst reports whether u is a literal int64 OpConst with value 0.
func isZeroConst(u uop.UOp) bool {
	if !u.Valid() || u.Op() != uop.OpConst {
		return false
	}
	v, ok := u.Arg().(int64)
	return ok && v == 0
}

// identityConst returns the identity element for reduceOp over dtype as a
// Const UOp.  When reduceOp is 0 (sentinel: no reduce context) it falls back
// to 0, matching the previous zeroConst behavior for elementwise pad.
//
// Identity table (mirrors tinygrad's dtypes.min / pm_mops at 9d9151a2):
//
//	OpAdd  → 0        (additive identity; float 0.0, int 0)
//	OpMul  → 1        (multiplicative identity; float 1.0, int 1)
//	OpMax  → −∞ / min (float −Inf; signed int min-value; unsigned 0)
//	other  → 0        (safe fallback; only OpAdd/OpMax arise in practice)
func identityConst(a *uop.Arena, reduceOp uop.Op, dtype *uop.DType) uop.UOp {
	var arg any
	switch reduceOp {
	case uop.OpMul:
		switch {
		case dtype.IsFloat():
			arg = float64(1)
		case dtype.IsBool():
			arg = true
		default:
			arg = int64(1)
		}
	case uop.OpMax:
		switch {
		case dtype.IsFloat():
			arg = math.Inf(-1)
		case dtype.IsBool():
			arg = false // false < true; false is the Max identity
		case dtype.IsUnsigned():
			arg = int64(0) // unsigned min is 0
		default:
			// Signed integer: use the dtype-width minimum value.
			switch dtype.Scalar().BitSize() {
			case 8:
				arg = int64(math.MinInt8)
			case 16:
				arg = int64(math.MinInt16)
			case 32:
				arg = int64(math.MinInt32)
			default: // 64-bit or unknown
				arg = int64(math.MinInt64)
			}
		}
	default:
		// OpAdd, unknown ops, and the zero-sentinel (no reduce context): use 0.
		switch {
		case dtype.IsFloat():
			arg = float64(0)
		case dtype.IsBool():
			arg = false
		default:
			arg = int64(0)
		}
	}
	return a.New(uop.OpConst, dtype, nil, arg, nil)
}
