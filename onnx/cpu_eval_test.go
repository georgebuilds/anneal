package onnx

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// cpuEval evaluates a *tensor.Tensor graph on the CPU and returns the result
// as []float32 along with its shape. It supports the subset of UOps the
// device-tier handlers emit: leaf BUFFER (with data), Const, all unary /
// binary ALU, Where, Cast (no-op at f32 host level), Reshape/Permute/Expand/
// Pad/Shrink/Flip, ReduceAxis.
//
// This is the value oracle for handler tests under CGO_ENABLED=0 — there is
// no GPU executor available, so we materialise the graph here on the CPU and
// compare against hand-computed expected values.
//
// The interpreter is intentionally minimal: it handles the small graphs the
// importer's Phase 1.B handlers build, not arbitrary tensor expressions.
func cpuEval(t *tensor.Tensor) ([]float32, []int64, error) {
	state := &cpuEvalState{cache: make(map[uint32]cpuArr)}
	out, err := state.eval(t.Node(), t.ShapeSints())
	if err != nil {
		return nil, nil, err
	}
	// Translate ShapeSints to []int64.
	sh := make([]int64, len(out.shape))
	for i, s := range out.shape {
		v, ok := s.ConstValue()
		if !ok {
			return nil, nil, fmt.Errorf("cpuEval: result has symbolic shape at axis %d", i)
		}
		sh[i] = v
	}
	return out.data, sh, nil
}

type cpuArr struct {
	data  []float32
	shape []shape.Sint
}

type cpuEvalState struct {
	cache map[uint32]cpuArr
}

func (s *cpuEvalState) eval(u uop.UOp, sh []shape.Sint) (cpuArr, error) {
	if c, ok := s.cache[u.Index()]; ok {
		return c, nil
	}
	r, err := s.evalNoCache(u, sh)
	if err != nil {
		return cpuArr{}, err
	}
	s.cache[u.Index()] = r
	return r, nil
}

func (s *cpuEvalState) evalNoCache(u uop.UOp, sh []shape.Sint) (cpuArr, error) {
	switch u.Op() {
	case uop.OpBuffer:
		data, ok := u.Arena().Leaf(u.Index())
		if !ok {
			return cpuArr{}, fmt.Errorf("cpuEval: leaf %d has no data", u.Index())
		}
		// Buffer shape: from arg (cloneShape was stored as []int64) or, for
		// arange/randn, from len(data).
		var bSh []shape.Sint
		if arg, ok := u.Arg().([]int64); ok {
			bSh = make([]shape.Sint, len(arg))
			for i, v := range arg {
				bSh[i] = shape.Const(v)
			}
		} else {
			// Fall back to len(data) as a 1-D shape.
			bSh = []shape.Sint{shape.Const(int64(len(data)))}
		}
		return cpuArr{data: append([]float32{}, data...), shape: bSh}, nil
	case uop.OpConst:
		var v float32
		switch a := u.Arg().(type) {
		case float64:
			v = float32(a)
		case int64:
			v = float32(a)
		case bool:
			if a {
				v = 1
			}
		default:
			return cpuArr{}, fmt.Errorf("cpuEval: unsupported Const arg type %T", u.Arg())
		}
		return cpuArr{data: []float32{v}, shape: []shape.Sint{}}, nil
	case uop.OpCast:
		// f32 host treats all numeric types identically.
		src, err := s.eval(u.Src(0), nil)
		if err != nil {
			return cpuArr{}, err
		}
		return src, nil
	case uop.OpContiguous:
		return s.eval(u.Src(0), sh)
	case uop.OpGather:
		return s.evalGather(u)
	}
	// Unary ALU
	if u.NSrc() == 1 {
		src, err := s.eval(u.Src(0), nil)
		if err != nil {
			return cpuArr{}, err
		}
		out := make([]float32, len(src.data))
		switch u.Op() {
		case uop.OpNeg:
			for i, v := range src.data {
				out[i] = -v
			}
		case uop.OpReciprocal:
			for i, v := range src.data {
				out[i] = 1.0 / v
			}
		case uop.OpSqrt:
			for i, v := range src.data {
				out[i] = float32(math.Sqrt(float64(v)))
			}
		case uop.OpExp2:
			for i, v := range src.data {
				out[i] = float32(math.Exp2(float64(v)))
			}
		case uop.OpLog2:
			for i, v := range src.data {
				out[i] = float32(math.Log2(float64(v)))
			}
		case uop.OpSin:
			for i, v := range src.data {
				out[i] = float32(math.Sin(float64(v)))
			}
		case uop.OpTrunc:
			for i, v := range src.data {
				out[i] = float32(math.Trunc(float64(v)))
			}
		case uop.OpErf:
			for i, v := range src.data {
				out[i] = float32(math.Erf(float64(v)))
			}
		case uop.OpReshape:
			// shape stored in arg.
			return cpuArr{data: src.data, shape: argShape(u, src.data)}, nil
		case uop.OpPermute:
			return cpuPermute(src, u)
		case uop.OpExpand:
			return cpuExpand(src, u)
		case uop.OpPad:
			return cpuPad(src, u)
		case uop.OpShrink:
			return cpuShrink(src, u)
		case uop.OpFlip:
			return cpuFlip(src, u)
		case uop.OpReduceAxis:
			return cpuReduceAxis(src, u)
		default:
			return cpuArr{}, fmt.Errorf("cpuEval: unhandled unary op %s", u.Op())
		}
		return cpuArr{data: out, shape: src.shape}, nil
	}
	// Binary ALU (broadcasted at the graph level — operands already same shape).
	if u.NSrc() == 2 {
		a, err := s.eval(u.Src(0), nil)
		if err != nil {
			return cpuArr{}, err
		}
		b, err := s.eval(u.Src(1), nil)
		if err != nil {
			return cpuArr{}, err
		}
		if len(a.data) != len(b.data) {
			return cpuArr{}, fmt.Errorf("cpuEval: binary op %s: operand length mismatch %d vs %d (broadcast expected at graph level)", u.Op(), len(a.data), len(b.data))
		}
		out := make([]float32, len(a.data))
		switch u.Op() {
		case uop.OpAdd:
			for i := range out {
				out[i] = a.data[i] + b.data[i]
			}
		case uop.OpSub:
			for i := range out {
				out[i] = a.data[i] - b.data[i]
			}
		case uop.OpMul:
			for i := range out {
				out[i] = a.data[i] * b.data[i]
			}
		case uop.OpFDiv, uop.OpIDiv:
			for i := range out {
				out[i] = a.data[i] / b.data[i]
			}
		case uop.OpMax:
			for i := range out {
				if a.data[i] > b.data[i] {
					out[i] = a.data[i]
				} else {
					out[i] = b.data[i]
				}
			}
		case uop.OpMin:
			for i := range out {
				if a.data[i] < b.data[i] {
					out[i] = a.data[i]
				} else {
					out[i] = b.data[i]
				}
			}
		case uop.OpCmpLt:
			for i := range out {
				if a.data[i] < b.data[i] {
					out[i] = 1
				}
			}
		case uop.OpCmpEq:
			for i := range out {
				if a.data[i] == b.data[i] {
					out[i] = 1
				}
			}
		case uop.OpPow:
			for i := range out {
				out[i] = float32(math.Pow(float64(a.data[i]), float64(b.data[i])))
			}
		default:
			return cpuArr{}, fmt.Errorf("cpuEval: unhandled binary op %s", u.Op())
		}
		return cpuArr{data: out, shape: a.shape}, nil
	}
	if u.NSrc() == 3 && u.Op() == uop.OpWhere {
		c, _ := s.eval(u.Src(0), nil)
		x, _ := s.eval(u.Src(1), nil)
		y, _ := s.eval(u.Src(2), nil)
		out := make([]float32, len(x.data))
		for i := range out {
			if c.data[i] != 0 {
				out[i] = x.data[i]
			} else {
				out[i] = y.data[i]
			}
		}
		return cpuArr{data: out, shape: x.shape}, nil
	}
	return cpuArr{}, fmt.Errorf("cpuEval: unhandled op %s with %d srcs", u.Op(), u.NSrc())
}

// argShape extracts the target shape from a Reshape/Expand UOp's arg field.
// Supports []int64 (concrete) and uop.ShapeSintArg (symbolic).
func argShape(u uop.UOp, data []float32) []shape.Sint {
	switch a := u.Arg().(type) {
	case []int64:
		out := make([]shape.Sint, len(a))
		for i, v := range a {
			out[i] = shape.Const(v)
		}
		return out
	case uop.ShapeSintArg:
		out := make([]shape.Sint, len(a))
		for i, d := range a {
			out[i] = shape.SintFromShapeDim(u.Arena(), d)
		}
		return out
	}
	// Fall back to data length.
	return []shape.Sint{shape.Const(int64(len(data)))}
}

func cpuPermute(src cpuArr, u uop.UOp) (cpuArr, error) {
	perm, ok := u.Arg().([]int64)
	if !ok {
		return cpuArr{}, fmt.Errorf("cpuPermute: arg not []int64")
	}
	rank := len(perm)
	if len(src.shape) != rank {
		return cpuArr{}, fmt.Errorf("cpuPermute: rank mismatch")
	}
	srcDims := make([]int64, rank)
	for i, s := range src.shape {
		v, ok := s.ConstValue()
		if !ok {
			return cpuArr{}, fmt.Errorf("cpuPermute: symbolic shape")
		}
		srcDims[i] = v
	}
	outDims := make([]int64, rank)
	for i, p := range perm {
		outDims[i] = srcDims[p]
	}
	// Strides of source row-major.
	srcStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcDims[i+1]
	}
	total := int64(1)
	for _, d := range outDims {
		total *= d
	}
	out := make([]float32, total)
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		// Decode k into out-shape coords.
		t := k
		for r := rank - 1; r >= 0; r-- {
			idx[r] = t % outDims[r]
			t /= outDims[r]
		}
		// Map to source.
		var srcK int64
		for r := 0; r < rank; r++ {
			srcK += idx[r] * srcStrides[perm[r]]
		}
		out[k] = src.data[srcK]
	}
	outSh := make([]shape.Sint, rank)
	for i, d := range outDims {
		outSh[i] = shape.Const(d)
	}
	return cpuArr{data: out, shape: outSh}, nil
}

func cpuExpand(src cpuArr, u uop.UOp) (cpuArr, error) {
	target := argShape(u, src.data)
	srcDims, _ := shapeSintsAsInts(src.shape)
	tgtDims, ok := shapeSintsAsInts(target)
	if !ok {
		return cpuArr{}, fmt.Errorf("cpuExpand: symbolic target shape")
	}
	rank := len(tgtDims)
	if len(srcDims) != rank {
		return cpuArr{}, fmt.Errorf("cpuExpand: rank mismatch %d vs %d", len(srcDims), rank)
	}
	srcStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcDims[i+1]
	}
	// For broadcast dims (srcDims[i]==1, tgtDims[i]>1), stride is 0.
	for i, sd := range srcDims {
		if sd == 1 && tgtDims[i] > 1 {
			srcStrides[i] = 0
		}
	}
	total := int64(1)
	for _, d := range tgtDims {
		total *= d
	}
	out := make([]float32, total)
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		t := k
		for r := rank - 1; r >= 0; r-- {
			idx[r] = t % tgtDims[r]
			t /= tgtDims[r]
		}
		var srcK int64
		for r := 0; r < rank; r++ {
			srcK += idx[r] * srcStrides[r]
		}
		out[k] = src.data[srcK]
	}
	outSh := make([]shape.Sint, rank)
	for i, d := range tgtDims {
		outSh[i] = shape.Const(d)
	}
	return cpuArr{data: out, shape: outSh}, nil
}

func cpuPad(src cpuArr, u uop.UOp) (cpuArr, error) {
	srcDims, _ := shapeSintsAsInts(src.shape)
	rank := len(srcDims)
	// Pad arg: [][2]int64 or uop.PadSintArg
	var pads [][2]int64
	switch a := u.Arg().(type) {
	case [][2]int64:
		pads = a
	case uop.PadSintArg:
		pads = make([][2]int64, len(a))
		for i, p := range a {
			lo := shape.SintFromShapeDim(u.Arena(), p[0])
			hi := shape.SintFromShapeDim(u.Arena(), p[1])
			lv, _ := lo.ConstValue()
			hv, _ := hi.ConstValue()
			pads[i] = [2]int64{lv, hv}
		}
	default:
		return cpuArr{}, fmt.Errorf("cpuPad: unsupported arg %T", u.Arg())
	}
	outDims := make([]int64, rank)
	for i := range outDims {
		outDims[i] = srcDims[i] + pads[i][0] + pads[i][1]
	}
	total := int64(1)
	for _, d := range outDims {
		total *= d
	}
	srcStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcDims[i+1]
	}
	out := make([]float32, total)
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		t := k
		for r := rank - 1; r >= 0; r-- {
			idx[r] = t % outDims[r]
			t /= outDims[r]
		}
		inBounds := true
		var srcK int64
		for r := 0; r < rank; r++ {
			j := idx[r] - pads[r][0]
			if j < 0 || j >= srcDims[r] {
				inBounds = false
				break
			}
			srcK += j * srcStrides[r]
		}
		if inBounds {
			out[k] = src.data[srcK]
		}
	}
	outSh := make([]shape.Sint, rank)
	for i, d := range outDims {
		outSh[i] = shape.Const(d)
	}
	return cpuArr{data: out, shape: outSh}, nil
}

func cpuShrink(src cpuArr, u uop.UOp) (cpuArr, error) {
	srcDims, _ := shapeSintsAsInts(src.shape)
	rank := len(srcDims)
	var bounds [][2]int64
	switch a := u.Arg().(type) {
	case [][2]int64:
		bounds = a
	case uop.ShrinkSintArg:
		bounds = make([][2]int64, len(a))
		for i, p := range a {
			lo := shape.SintFromShapeDim(u.Arena(), p[0])
			hi := shape.SintFromShapeDim(u.Arena(), p[1])
			lv, _ := lo.ConstValue()
			hv, _ := hi.ConstValue()
			bounds[i] = [2]int64{lv, hv}
		}
	default:
		return cpuArr{}, fmt.Errorf("cpuShrink: unsupported arg %T", u.Arg())
	}
	outDims := make([]int64, rank)
	for i := range outDims {
		outDims[i] = bounds[i][1] - bounds[i][0]
	}
	total := int64(1)
	for _, d := range outDims {
		total *= d
	}
	srcStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcDims[i+1]
	}
	out := make([]float32, total)
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		t := k
		for r := rank - 1; r >= 0; r-- {
			idx[r] = t % outDims[r]
			t /= outDims[r]
		}
		var srcK int64
		for r := 0; r < rank; r++ {
			srcK += (idx[r] + bounds[r][0]) * srcStrides[r]
		}
		out[k] = src.data[srcK]
	}
	outSh := make([]shape.Sint, rank)
	for i, d := range outDims {
		outSh[i] = shape.Const(d)
	}
	return cpuArr{data: out, shape: outSh}, nil
}

func cpuFlip(src cpuArr, u uop.UOp) (cpuArr, error) {
	srcDims, _ := shapeSintsAsInts(src.shape)
	rank := len(srcDims)
	flipArg, ok := u.Arg().([]int64)
	if !ok {
		return cpuArr{}, fmt.Errorf("cpuFlip: arg not []int64")
	}
	srcStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcDims[i+1]
	}
	total := int64(1)
	for _, d := range srcDims {
		total *= d
	}
	out := make([]float32, total)
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		t := k
		for r := rank - 1; r >= 0; r-- {
			idx[r] = t % srcDims[r]
			t /= srcDims[r]
		}
		var srcK int64
		for r := 0; r < rank; r++ {
			j := idx[r]
			if r < len(flipArg) && flipArg[r] != 0 {
				j = srcDims[r] - 1 - j
			}
			srcK += j * srcStrides[r]
		}
		out[k] = src.data[srcK]
	}
	return cpuArr{data: out, shape: src.shape}, nil
}

func cpuReduceAxis(src cpuArr, u uop.UOp) (cpuArr, error) {
	arg, ok := u.Arg().(uop.ReduceArg)
	if !ok {
		return cpuArr{}, fmt.Errorf("cpuReduceAxis: unsupported arg %T", u.Arg())
	}
	srcDims, _ := shapeSintsAsInts(src.shape)
	rank := len(srcDims)
	axesSet := make(map[int]bool)
	for _, a := range arg.Axes {
		axesSet[a] = true
	}
	// Output dims: copy non-reduced, drop reduced (no keepdim here — the
	// frontend `reduce` already handled keepdim by inserting size-1 dims
	// into the ShapeTracker; the reduce node itself drops reduced axes).
	var outDims []int64
	for i, d := range srcDims {
		if axesSet[i] {
			continue
		}
		outDims = append(outDims, d)
	}
	if len(outDims) == 0 {
		outDims = []int64{1}
	}
	srcStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcDims[i+1]
	}
	outTotal := int64(1)
	for _, d := range outDims {
		outTotal *= d
	}
	out := make([]float32, outTotal)
	var init float32
	switch arg.Op {
	case uop.OpAdd:
		init = 0
	case uop.OpMax:
		init = float32(math.Inf(-1))
	case uop.OpMin:
		init = float32(math.Inf(1))
	case uop.OpMul:
		init = 1
	default:
		return cpuArr{}, fmt.Errorf("cpuReduceAxis: unsupported reduce op %s", arg.Op)
	}
	for i := range out {
		out[i] = init
	}
	// Build out strides over outDims indexing.
	outStrides := make([]int64, len(outDims))
	if len(outDims) > 0 {
		outStrides[len(outDims)-1] = 1
		for i := len(outDims) - 2; i >= 0; i-- {
			outStrides[i] = outStrides[i+1] * outDims[i+1]
		}
	}
	// Iterate every input position; map to output position.
	idx := make([]int64, rank)
	total := int64(1)
	for _, d := range srcDims {
		total *= d
	}
	for k := int64(0); k < total; k++ {
		t := k
		for r := rank - 1; r >= 0; r-- {
			idx[r] = t % srcDims[r]
			t /= srcDims[r]
		}
		// Compute output offset.
		var outK int64
		oi := 0
		for r := 0; r < rank; r++ {
			if axesSet[r] {
				continue
			}
			outK += idx[r] * outStrides[oi]
			oi++
		}
		v := src.data[k]
		switch arg.Op {
		case uop.OpAdd:
			out[outK] += v
		case uop.OpMax:
			if v > out[outK] {
				out[outK] = v
			}
		case uop.OpMin:
			if v < out[outK] {
				out[outK] = v
			}
		case uop.OpMul:
			out[outK] *= v
		}
	}
	outSh := make([]shape.Sint, len(outDims))
	for i, d := range outDims {
		outSh[i] = shape.Const(d)
	}
	return cpuArr{data: out, shape: outSh}, nil
}

// evalGather implements OpGather per tensor/gather.go semantics:
//
//	data.shape  = [d0, ..., d_{dim-1}, V, d_{dim+1}, ..., d_{R-1}]
//	idx.shape   = idxShape
//	out.shape   = [d0, ..., d_{dim-1}, idxShape..., d_{dim+1}, ..., d_{R-1}]
//	out[a, j_idx, c] = data[a, idx[j_idx], c]
//
// Negative indices wrap modulo V (matches ONNX Gather semantics).
func (s *cpuEvalState) evalGather(u uop.UOp) (cpuArr, error) {
	dim, ok := u.Arg().(int64)
	if !ok {
		return cpuArr{}, fmt.Errorf("evalGather: arg not int64, got %T", u.Arg())
	}
	data, err := s.eval(u.Src(0), nil)
	if err != nil {
		return cpuArr{}, err
	}
	idx, err := s.eval(u.Src(1), nil)
	if err != nil {
		return cpuArr{}, err
	}
	dataDims, ok := shapeSintsAsInts(data.shape)
	if !ok {
		return cpuArr{}, fmt.Errorf("evalGather: symbolic data shape")
	}
	idxDims, ok := shapeSintsAsInts(idx.shape)
	if !ok {
		return cpuArr{}, fmt.Errorf("evalGather: symbolic index shape")
	}
	if int(dim) < 0 || int(dim) >= len(dataDims) {
		return cpuArr{}, fmt.Errorf("evalGather: dim %d out of range %d", dim, len(dataDims))
	}
	V := dataDims[dim]

	// Out shape: dataDims[:dim] ++ idxDims ++ dataDims[dim+1:]
	outDims := make([]int64, 0, len(dataDims)-1+len(idxDims))
	outDims = append(outDims, dataDims[:dim]...)
	outDims = append(outDims, idxDims...)
	outDims = append(outDims, dataDims[dim+1:]...)

	// Compute strides for data.
	dataRank := len(dataDims)
	dataStrides := make([]int64, dataRank)
	dataStrides[dataRank-1] = 1
	for i := dataRank - 2; i >= 0; i-- {
		dataStrides[i] = dataStrides[i+1] * dataDims[i+1]
	}

	// Iterate over output positions; for each, decode into (outerCoords,
	// idxCoords, innerCoords), look up idx[idxCoords] to get V-coord,
	// then read data[outerCoords, V-coord, innerCoords].
	outerRank := int(dim)
	innerRank := dataRank - int(dim) - 1
	idxRank := len(idxDims)

	outTotal := int64(1)
	for _, d := range outDims {
		outTotal *= d
	}
	out := make([]float32, outTotal)
	outCoord := make([]int64, len(outDims))
	for k := int64(0); k < outTotal; k++ {
		t := k
		for r := len(outDims) - 1; r >= 0; r-- {
			outCoord[r] = t % outDims[r]
			t /= outDims[r]
		}
		// Outer coords: outCoord[0..outerRank), inner: outCoord[outerRank+idxRank ..)
		outer := outCoord[:outerRank]
		idxCoords := outCoord[outerRank : outerRank+idxRank]
		inner := outCoord[outerRank+idxRank:]

		// Decode idxCoords into flat idx offset.
		var idxOff int64
		idxStride := int64(1)
		for r := idxRank - 1; r >= 0; r-- {
			idxOff += idxCoords[r] * idxStride
			idxStride *= idxDims[r]
		}
		v := int64(idx.data[idxOff])
		if v < 0 {
			v += V
		}
		if v < 0 || v >= V {
			return cpuArr{}, fmt.Errorf("evalGather: index %d out of range [0,%d)", v, V)
		}
		// Read data[outer..., v, inner...]
		var dataOff int64
		for r := 0; r < outerRank; r++ {
			dataOff += outer[r] * dataStrides[r]
		}
		dataOff += v * dataStrides[outerRank]
		for r := 0; r < innerRank; r++ {
			dataOff += inner[r] * dataStrides[outerRank+1+r]
		}
		out[k] = data.data[dataOff]
	}

	outSh := make([]shape.Sint, len(outDims))
	for i, d := range outDims {
		outSh[i] = shape.Const(d)
	}
	return cpuArr{data: out, shape: outSh}, nil
}

// allClose reports whether |a-b| ≤ atol for all elements.
func allClose(a, b []float32, atol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > atol {
			return false
		}
	}
	return true
}
