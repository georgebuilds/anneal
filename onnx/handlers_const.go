package onnx

import (
	"fmt"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
)

// resolveShapeInput converts a Value carrying a shape vector into a
// []shape.Sint slice. Host-tier inputs go through ToSints; device-tier inputs
// (initializers were interned as device int leaves) read the leaf data and
// promote to ConstInt.
func resolveShapeInput(v Value) ([]shape.Sint, error) {
	if v.IsHost() {
		return v.ToSints(), nil
	}
	if v.IsDevice() {
		t := v.Tensor()
		if !t.DType().IsInt() {
			return nil, fmt.Errorf("shape input has non-integer dtype %v", t.DType())
		}
		data := t.Data()
		if data == nil {
			return nil, fmt.Errorf("shape input device tensor has no host-side data")
		}
		out := make([]shape.Sint, len(data))
		for i, f := range data {
			out[i] = shape.Const(int64(f))
		}
		return out, nil
	}
	return nil, fmt.Errorf("shape input has unsupported kind %d", v.Kind)
}

// handleConstant emits a device-tier leaf tensor from the Constant's `value`
// attribute (a TensorProto). The host-tier Constant handler intercepts integer
// dtypes; floats fall through to here.
func handleConstant(ctx *HandlerCtx) ([]Value, error) {
	tp := ctx.Node.Attrs["value"].Tensor()
	if tp == nil {
		return nil, fmt.Errorf("Constant: missing `value` attribute")
	}
	leaf, err := tensorFromProto(ctx.Arena, tp, ctx.Device)
	if err != nil {
		return nil, err
	}
	return []Value{Device(leaf)}, nil
}

// handleIdentity passes the input through unchanged.
func handleIdentity(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) != 1 {
		return nil, fmt.Errorf("Identity: expected 1 input, got %d", len(ctx.Inputs))
	}
	if !ctx.Inputs[0].IsDevice() {
		return nil, fmt.Errorf("Identity: input is not a device tensor (kind=%d)", ctx.Inputs[0].Kind)
	}
	return []Value{ctx.Inputs[0]}, nil
}

// handleConstantOfShape builds a device tensor of shape input[0] filled with
// the scalar from the `value` attribute (default 0.0 f32).
func handleConstantOfShape(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 1 {
		return nil, fmt.Errorf("ConstantOfShape: expected 1 input")
	}
	sh, err := resolveShapeInput(ctx.Inputs[0])
	if err != nil {
		return nil, fmt.Errorf("ConstantOfShape: %w", err)
	}
	val := 0.0
	dt := onnxpb.TensorProto_FLOAT
	if tp := ctx.Node.Attrs["value"].Tensor(); tp != nil {
		dt = onnxpb.TensorProto_DataType(tp.GetDataType())
		switch dt {
		case onnxpb.TensorProto_FLOAT:
			fd := tp.GetFloatData()
			if len(fd) > 0 {
				val = float64(fd[0])
			}
		case onnxpb.TensorProto_INT32, onnxpb.TensorProto_INT64:
			vals, err := decodeIntTensor(tp)
			if err != nil {
				return nil, err
			}
			if len(vals) > 0 {
				val = float64(vals[0])
			}
		default:
			return nil, fmt.Errorf("ConstantOfShape: dtype %v not supported", dt)
		}
	}
	annealDT, _, _, ok := onnxDType(int32(dt))
	if !ok {
		return nil, fmt.Errorf("ConstantOfShape: dtype %v has no anneal mapping", dt)
	}
	out := tensor.FullSints(ctx.Arena, sh, val, annealDT, ctx.Device)
	return []Value{Device(out)}, nil
}
