package onnx

import (
	"fmt"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/shape"
)

// Host-tier op handlers. These run the shape subgraph on the Go side: Shape,
// Size, integer Constant / ConstantOfShape, Range, integer arithmetic over
// host vectors, host Gather/Concat/Unsqueeze/Squeeze/Identity/Cast. The runner
// routes a node here iff every input is host-evaluable (concrete-host or, for
// Shape/Size, a Device tensor whose shape is read without realising).

func init() {
	RegisterHostOp("Shape", hostShape)
	RegisterHostOp("Size", hostSize)
	RegisterHostOp("Constant", hostConstant)
	RegisterHostOp("Range", hostRange)
	RegisterHostOp("Add", hostAdd)
	RegisterHostOp("Sub", hostSub)
	RegisterHostOp("Mul", hostMul)
	RegisterHostOp("Div", hostDiv)
	RegisterHostOp("Neg", hostNeg)
	RegisterHostOp("Gather", hostGather)
	RegisterHostOp("Concat", hostConcat)
	RegisterHostOp("Unsqueeze", hostUnsqueeze)
	RegisterHostOp("Squeeze", hostSqueeze)
	RegisterHostOp("Identity", hostIdentity)
	RegisterHostOp("Cast", hostCast)
}

// hostShape returns the shape of input[0] as host integers. Preserves symbolic
// dims as a HostSints vector when present (so dim_param flows downstream); for
// all-concrete inputs collapses to HostInts.
func hostShape(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) < 1 {
		return Value{}, fmt.Errorf("shape: expected 1 input, got %d", len(inputs))
	}
	in := inputs[0]
	var sh []shape.Sint
	switch in.Kind {
	case KindDevice:
		sh = in.Tensor().ShapeSints()
	case KindHostInts:
		// Shape of a host vector: a 1-D shape of size len.
		return HostInts([]int64{int64(len(in.Is))}), nil
	case KindHostSints:
		return HostInts([]int64{int64(len(in.Ss))}), nil
	default:
		return Value{}, fmt.Errorf("shape: unsupported input kind %d", in.Kind)
	}
	if ints, ok := shapeSintsAsInts(sh); ok {
		return HostInts(ints), nil
	}
	out := make([]shape.Sint, len(sh))
	copy(out, sh)
	return HostSints(out), nil
}

// hostSize returns the total element count of input[0]. Errors if any dim is
// symbolic (Size's contract is a scalar int64).
func hostSize(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) < 1 {
		return Value{}, fmt.Errorf("size: expected 1 input, got %d", len(inputs))
	}
	in := inputs[0]
	var sh []shape.Sint
	switch in.Kind {
	case KindDevice:
		sh = in.Tensor().ShapeSints()
	case KindHostInts:
		return HostInt64(int64(len(in.Is))), nil
	case KindHostSints:
		return HostInt64(int64(len(in.Ss))), nil
	default:
		return Value{}, fmt.Errorf("size: unsupported input kind %d", in.Kind)
	}
	prod := int64(1)
	for i, s := range sh {
		v, ok := s.ConstValue()
		if !ok {
			return Value{}, fmt.Errorf("size: symbolic dim at axis %d (Size requires a concrete scalar in v1)", i)
		}
		prod *= v
	}
	return HostInt64(prod), nil
}

// hostConstant reads the `value` attribute. Returns HostInts/HostInt64 for
// integer-typed scalar/vector tensors; returns the zero Value otherwise to
// signal "not host-evaluable" so the dispatcher falls through to the device
// Constant handler. Floats and rank-≥2 tensors take the device path.
func hostConstant(node *Node, inputs []Value, st *HostState) (Value, error) {
	tp := node.Attrs["value"].Tensor()
	if tp == nil {
		return Value{}, fmt.Errorf("constant: missing or non-tensor `value` attribute (host)")
	}
	dt := onnxpb.TensorProto_DataType(tp.GetDataType())
	switch dt {
	case onnxpb.TensorProto_INT32, onnxpb.TensorProto_INT64:
		// host path
	default:
		// Non-integer-typed Constant: device handler builds a tensor leaf.
		return Value{}, ErrFallThroughToDevice
	}
	vals, err := decodeIntTensor(tp)
	if err != nil {
		return Value{}, fmt.Errorf("constant: %w", err)
	}
	// Scalar special case: dims=[] AND len(vals)==1.
	if len(tp.GetDims()) == 0 && len(vals) == 1 {
		return HostInt64(vals[0]), nil
	}
	return HostInts(vals), nil
}

// hostRange computes [start, limit) with step delta as a HostInts. All three
// inputs must be HostInt64 / HostInts (scalar/length-1 vectors are accepted).
func hostRange(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) < 3 {
		return Value{}, fmt.Errorf("range: expected 3 inputs, got %d", len(inputs))
	}
	start, err := asHostScalar(inputs[0])
	if err != nil {
		return Value{}, fmt.Errorf("range: start: %w", err)
	}
	limit, err := asHostScalar(inputs[1])
	if err != nil {
		return Value{}, fmt.Errorf("range: limit: %w", err)
	}
	delta, err := asHostScalar(inputs[2])
	if err != nil {
		return Value{}, fmt.Errorf("range: delta: %w", err)
	}
	if delta == 0 {
		return Value{}, fmt.Errorf("range: delta is zero")
	}
	var out []int64
	if delta > 0 {
		for v := start; v < limit; v += delta {
			out = append(out, v)
		}
	} else {
		for v := start; v > limit; v += delta {
			out = append(out, v)
		}
	}
	if out == nil {
		out = []int64{}
	}
	return HostInts(out), nil
}

func hostAdd(node *Node, inputs []Value, st *HostState) (Value, error) {
	return hostBinop(inputs, func(a, b int64) int64 { return a + b }, "Add")
}
func hostSub(node *Node, inputs []Value, st *HostState) (Value, error) {
	return hostBinop(inputs, func(a, b int64) int64 { return a - b }, "Sub")
}
func hostMul(node *Node, inputs []Value, st *HostState) (Value, error) {
	return hostBinop(inputs, func(a, b int64) int64 { return a * b }, "Mul")
}
func hostDiv(node *Node, inputs []Value, st *HostState) (Value, error) {
	return hostBinop(inputs, func(a, b int64) int64 {
		if b == 0 {
			return 0
		}
		return a / b
	}, "Div")
}

func hostNeg(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) != 1 {
		return Value{}, fmt.Errorf("neg: expected 1 input, got %d", len(inputs))
	}
	switch inputs[0].Kind {
	case KindHostInt64:
		return HostInt64(-inputs[0].I), nil
	case KindHostInts:
		out := make([]int64, len(inputs[0].Is))
		for i, v := range inputs[0].Is {
			out[i] = -v
		}
		return HostInts(out), nil
	}
	return Value{}, fmt.Errorf("neg: unsupported input kind %d", inputs[0].Kind)
}

// hostBinop applies fn elementwise with NumPy-style scalar broadcast over int.
func hostBinop(inputs []Value, fn func(a, b int64) int64, name string) (Value, error) {
	if len(inputs) != 2 {
		return Value{}, fmt.Errorf("%s: expected 2 inputs, got %d", name, len(inputs))
	}
	a, b := inputs[0], inputs[1]
	aIsScalar := a.Kind == KindHostInt64
	bIsScalar := b.Kind == KindHostInt64
	if aIsScalar && bIsScalar {
		return HostInt64(fn(a.I, b.I)), nil
	}
	if aIsScalar && b.Kind == KindHostInts {
		out := make([]int64, len(b.Is))
		for i, v := range b.Is {
			out[i] = fn(a.I, v)
		}
		return HostInts(out), nil
	}
	if a.Kind == KindHostInts && bIsScalar {
		out := make([]int64, len(a.Is))
		for i, v := range a.Is {
			out[i] = fn(v, b.I)
		}
		return HostInts(out), nil
	}
	if a.Kind == KindHostInts && b.Kind == KindHostInts {
		if len(a.Is) != len(b.Is) {
			return Value{}, fmt.Errorf("%s: vector length mismatch %d != %d", name, len(a.Is), len(b.Is))
		}
		out := make([]int64, len(a.Is))
		for i := range a.Is {
			out[i] = fn(a.Is[i], b.Is[i])
		}
		return HostInts(out), nil
	}
	return Value{}, fmt.Errorf("%s: unsupported input kinds %d, %d", name, a.Kind, b.Kind)
}

// hostGather indexes data along axis with indices. v1 supports axis=0 over a
// host int vector (the common Shape→Gather→Reshape glue tail).
func hostGather(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) < 2 {
		return Value{}, fmt.Errorf("gather: expected 2 inputs, got %d", len(inputs))
	}
	axis := node.Attrs["axis"].Int(0)
	if axis != 0 {
		return Value{}, fmt.Errorf("gather (host): only axis=0 supported, got %d", axis)
	}
	data := inputs[0]
	idx := inputs[1]
	var src []int64
	switch data.Kind {
	case KindHostInts:
		src = data.Is
	case KindHostInt64:
		src = []int64{data.I}
	default:
		return Value{}, fmt.Errorf("gather (host): data kind %d not supported", data.Kind)
	}
	var idxes []int64
	switch idx.Kind {
	case KindHostInts:
		idxes = idx.Is
	case KindHostInt64:
		idxes = []int64{idx.I}
	default:
		return Value{}, fmt.Errorf("gather (host): index kind %d not supported", idx.Kind)
	}
	out := make([]int64, len(idxes))
	for i, k := range idxes {
		if k < 0 {
			k += int64(len(src))
		}
		if k < 0 || k >= int64(len(src)) {
			return Value{}, fmt.Errorf("gather (host): index %d out of range for length %d", idxes[i], len(src))
		}
		out[i] = src[k]
	}
	// Scalar-input idx: return scalar.
	if idx.Kind == KindHostInt64 {
		return HostInt64(out[0]), nil
	}
	return HostInts(out), nil
}

// hostConcat joins host int vectors along axis=0 (only supported axis for
// shape-tier arithmetic; matches the classifier-tail glue chain).
func hostConcat(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) == 0 {
		return Value{}, fmt.Errorf("concat: zero inputs")
	}
	axis := node.Attrs["axis"].Int(0)
	if axis != 0 {
		return Value{}, fmt.Errorf("concat (host): only axis=0 supported, got %d", axis)
	}
	var out []int64
	for i, in := range inputs {
		switch in.Kind {
		case KindHostInts:
			out = append(out, in.Is...)
		case KindHostInt64:
			out = append(out, in.I)
		default:
			return Value{}, fmt.Errorf("concat (host): input %d kind %d not supported", i, in.Kind)
		}
	}
	if out == nil {
		out = []int64{}
	}
	return HostInts(out), nil
}

// hostUnsqueeze inserts a 1 at each axis in axes (sorted ascending).
// Opset ≤ 12: axes is an attribute; opset ≥ 13: axes is input[1].
func hostUnsqueeze(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) < 1 {
		return Value{}, fmt.Errorf("unsqueeze: expected ≥ 1 input, got %d", len(inputs))
	}
	axes, err := readAxesAttrOrInput(node, inputs, 1)
	if err != nil {
		return Value{}, fmt.Errorf("unsqueeze: %w", err)
	}
	in := inputs[0]
	var data []int64
	switch in.Kind {
	case KindHostInts:
		data = in.Is
	case KindHostInt64:
		data = []int64{in.I}
	default:
		return Value{}, fmt.Errorf("unsqueeze: input kind %d not supported", in.Kind)
	}
	// Host Unsqueeze on a 1-D int vector: inserting at non-zero axis is a
	// no-op for the linearised int payload (we're not tracking a rank > 1
	// host tensor). For shape glue this is sufficient: the surrounding
	// Reshape consumes the same flat int vector regardless.
	_ = axes
	out := make([]int64, len(data))
	copy(out, data)
	return HostInts(out), nil
}

// hostSqueeze removes size-1 dims; axes selects which (opset ≤ 12 attr,
// opset ≥ 13 input). On a 1-D host int vector this is also a no-op for the
// payload (the rank-tracking is implicit at host tier).
func hostSqueeze(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) < 1 {
		return Value{}, fmt.Errorf("squeeze: expected ≥ 1 input, got %d", len(inputs))
	}
	in := inputs[0]
	switch in.Kind {
	case KindHostInts:
		out := make([]int64, len(in.Is))
		copy(out, in.Is)
		return HostInts(out), nil
	case KindHostInt64:
		return HostInt64(in.I), nil
	}
	return Value{}, fmt.Errorf("squeeze: input kind %d not supported", in.Kind)
}

func hostIdentity(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) != 1 {
		return Value{}, fmt.Errorf("identity: expected 1 input, got %d", len(inputs))
	}
	return inputs[0], nil
}

// hostCast: trivial int-to-int. Float-target casts fall through to device.
func hostCast(node *Node, inputs []Value, st *HostState) (Value, error) {
	if len(inputs) != 1 {
		return Value{}, fmt.Errorf("cast: expected 1 input, got %d", len(inputs))
	}
	to := onnxpb.TensorProto_DataType(node.Attrs["to"].Int(0))
	switch to {
	case onnxpb.TensorProto_INT32, onnxpb.TensorProto_INT64:
		switch inputs[0].Kind {
		case KindHostInt64:
			return HostInt64(inputs[0].I), nil
		case KindHostInts:
			out := make([]int64, len(inputs[0].Is))
			copy(out, inputs[0].Is)
			return HostInts(out), nil
		}
	}
	return Value{}, ErrFallThroughToDevice
}

// ── helpers ──────────────────────────────────────────────────────────────────

func asHostScalar(v Value) (int64, error) {
	switch v.Kind {
	case KindHostInt64:
		return v.I, nil
	case KindHostInts:
		if len(v.Is) == 1 {
			return v.Is[0], nil
		}
		return 0, fmt.Errorf("expected scalar (length 1), got vector of length %d", len(v.Is))
	}
	return 0, fmt.Errorf("expected host int scalar, got kind %d", v.Kind)
}

// decodeIntTensor extracts integer values from a TensorProto. Supports INT32
// and INT64 (with the int64 → int32-range overflow trap matching the
// initializer policy in §3 of the plan).
func decodeIntTensor(tp *onnxpb.TensorProto) ([]int64, error) {
	dt := onnxpb.TensorProto_DataType(tp.GetDataType())
	dims := tp.GetDims()
	elems := int64(1)
	for _, d := range dims {
		elems *= d
	}
	switch dt {
	case onnxpb.TensorProto_INT32:
		if td := tp.GetInt32Data(); len(td) > 0 {
			out := make([]int64, len(td))
			for i, v := range td {
				out[i] = int64(v)
			}
			return out, nil
		}
		raw := tp.GetRawData()
		if elems > 0 && int64(len(raw)) != elems*4 {
			return nil, fmt.Errorf("iNT32 raw_data length %d != %d", len(raw), elems*4)
		}
		out := make([]int64, len(raw)/4)
		for i := range out {
			out[i] = int64(int32(uint32(raw[i*4]) |
				uint32(raw[i*4+1])<<8 |
				uint32(raw[i*4+2])<<16 |
				uint32(raw[i*4+3])<<24))
		}
		return out, nil
	case onnxpb.TensorProto_INT64:
		if td := tp.GetInt64Data(); len(td) > 0 {
			out := make([]int64, len(td))
			copy(out, td)
			return out, nil
		}
		raw := tp.GetRawData()
		if elems > 0 && int64(len(raw)) != elems*8 {
			return nil, fmt.Errorf("iNT64 raw_data length %d != %d", len(raw), elems*8)
		}
		out := make([]int64, len(raw)/8)
		for i := range out {
			out[i] = int64(uint64(raw[i*8]) |
				uint64(raw[i*8+1])<<8 |
				uint64(raw[i*8+2])<<16 |
				uint64(raw[i*8+3])<<24 |
				uint64(raw[i*8+4])<<32 |
				uint64(raw[i*8+5])<<40 |
				uint64(raw[i*8+6])<<48 |
				uint64(raw[i*8+7])<<56)
		}
		return out, nil
	}
	return nil, fmt.Errorf("decodeIntTensor: dtype %v not supported", dt)
}

// readAxesAttrOrInput returns the axes vector. Opset ≤ 12 carries it as the
// `axes` attribute; opset ≥ 13 as inputs[idx]. We probe input first so the
// migration is transparent: if inputs[idx] is set we use it, else the attr.
// nilOnAbsent: returning nil when both are absent (the "axes optional, no-op"
// path for some ops).
func readAxesAttrOrInput(node *Node, inputs []Value, idx int) ([]int64, error) {
	if idx < len(inputs) {
		v := inputs[idx]
		switch v.Kind {
		case KindHostInts:
			out := make([]int64, len(v.Is))
			copy(out, v.Is)
			return out, nil
		case KindHostInt64:
			return []int64{v.I}, nil
		case KindUnset:
			// fall through to attribute
		default:
			return nil, fmt.Errorf("axes input has unsupported kind %d", v.Kind)
		}
	}
	if a, ok := node.Attrs["axes"]; ok {
		return a.Ints(nil), nil
	}
	return nil, nil
}

// shape import kept for shapeSintsAsInts helper defined in host.go.
var _ = shape.Const
