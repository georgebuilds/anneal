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
		perm := expr.Arg().([]int64)
		srcIndices := make([]uop.UOp, len(perm))
		for i, p := range perm {
			srcIndices[p] = indices[i]
		}
		return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)

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
		// Slice: offset each index by its lower bound.
		bounds := expr.Arg().([][2]int64)
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

	case uop.OpPad:
		// Pad: validity guard + source index = r - lo; out-of-bounds → zero.
		padding := expr.Arg().([][2]int64)
		srcShape := shape.AsInts(shapeMap[expr.Src(0).Index()])
		srcIndices := make([]uop.UOp, len(padding))
		var validConds []uop.UOp
		for i, p := range padding {
			lo, hi := p[0], p[1]
			r := indices[i]
			if lo != 0 {
				loConst := a.New(uop.OpConst, uop.Dtypes.Index, nil, lo, nil)
				srcIndices[i] = a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{r, loConst}, nil, nil)
				// r >= lo: (lo-1) < r
				loMinus1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, lo-1, nil)
				validConds = append(validConds, a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{loMinus1, r}, nil, nil))
			} else {
				srcIndices[i] = r
			}
			if hi != 0 {
				// r < lo + srcShape[i]
				bound := a.New(uop.OpConst, uop.Dtypes.Index, nil, lo+srcShape[i], nil)
				validConds = append(validConds, a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{r, bound}, nil, nil))
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
		// Mirror: index r → (size-1) - r for flipped axes.
		axisFlags := expr.Arg().([]int64)
		srcShape := shape.AsInts(shapeMap[expr.Src(0).Index()])
		srcIndices := make([]uop.UOp, len(axisFlags))
		for i, f := range axisFlags {
			if f != 0 {
				sm1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, srcShape[i]-1, nil)
				srcIndices[i] = a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{sm1, indices[i]}, nil, nil)
			} else {
				srcIndices[i] = indices[i]
			}
		}
		return indexExprNode(a, expr.Src(0), srcIndices, shapeMap, rc, fillOp)

	// ── reduce ────────────────────────────────────────────────────────────

	case uop.OpReduceAxis:
		// Creates AxisReduce range vars for the reduced axes, then indexes through
		// the source. Returns a kernel-level REDUCE(acc_op, elem_expr, *reduce_ranges).
		ra := expr.Arg().(uop.ReduceArg)
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
				varName := sym.Node.Arg().(uop.VarArg).Name
				rr = rc.newSymRange(varName, uop.AxisReduce)
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

// flatIndex computes the row-major flat index from multi-dim indices and shape.
// flatIndex([r0, r1], [n0, n1]) = r0*n1 + r1
func flatIndex(a *uop.Arena, indices []uop.UOp, shape []int64) uop.UOp {
	if len(indices) == 0 {
		return a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	}
	if len(indices) == 1 {
		return indices[0]
	}
	// strides[i] = prod(shape[i+1:])
	strides := make([]int64, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
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
	strides := make([]int64, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
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

// sintStrides computes concrete row-major strides from a Sint shape slice.
// For Option A (symbolic dim is always outermost), all strides are concrete:
// the symbolic dim's stride = product of the concrete trailing dims.
func sintStrides(sh []shape.Sint) []int64 {
	n := len(sh)
	strides := make([]int64, n)
	acc := int64(1)
	for i := n - 1; i >= 0; i-- {
		strides[i] = acc
		if v, ok := sh[i].ConstValue(); ok {
			acc *= v
		}
		// For symbolic dim: acc is not updated (it will only be used by dims to the
		// left, which don't exist in Option A since symbolic is always outermost).
	}
	return strides
}

// flatIndexSints computes a row-major flat index from multi-dim indices and a
// Sint shape. Strides are extracted as concrete int64 values via sintStrides.
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

// unflatIndexSints decomposes a flat index into per-dim indices for a Sint shape.
// For concrete dims: applies the usual div+mod. For the symbolic outermost dim:
// only applies div (no mod needed — quotient is exact for valid flat indices).
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
		if stride == 1 {
			divided = flat
		} else {
			sc := a.New(uop.OpConst, uop.Dtypes.Index, nil, stride, nil)
			divided = a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{flat, sc}, nil, nil)
		}
		if _, ok := s.ConstValue(); ok {
			// Concrete dim: modulo isolates this dimension.
			sv, _ := s.ConstValue()
			szc := a.New(uop.OpConst, uop.Dtypes.Index, nil, sv, nil)
			out[i] = a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{divided, szc}, nil, nil)
		} else {
			// Symbolic outermost dim: quotient is the exact index (no modulo needed).
			out[i] = divided
		}
	}
	return out
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

// zeroConst returns a zero constant of the given dtype.
// Used for pad fill in elementwise (non-reduce) contexts.
func zeroConst(a *uop.Arena, dtype *uop.DType) uop.UOp {
	return identityConst(a, 0, dtype)
}
