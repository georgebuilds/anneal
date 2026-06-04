package onnx

import (
	"fmt"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func handleAdd(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Add")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.Add(b))}, nil
}

func handleSub(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Sub")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.Sub(b))}, nil
}

func handleMul(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Mul")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.Mul(b))}, nil
}

func handleDiv(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Div")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.Div(b))}, nil
}

// handlePow uses primitive log2/exp2 to express x^y = exp2(y * log2(x)) for
// the float case. Integer exponents over float bases are supported by the
// same expansion.
func handlePow(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Pow")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.Log2().Mul(b).Exp2())}, nil
}

func handleSqrt(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Sqrt")
	if err != nil {
		return nil, err
	}
	return []Value{Device(x.Sqrt())}, nil
}

func handleNeg(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Neg")
	if err != nil {
		return nil, err
	}
	return []Value{Device(x.Neg())}, nil
}

// handleTanh composes tanh via nn primitive.
func handleTanh(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Tanh")
	if err != nil {
		return nil, err
	}
	// tanh(x) = 2*sigmoid(2x) - 1
	two := tensor.FullSints(ctx.Arena, x.ShapeSints(), 2.0, x.DType(), x.Device())
	one := tensor.FullSints(ctx.Arena, x.ShapeSints(), 1.0, x.DType(), x.Device())
	twoX := two.Mul(x)
	sig := sigmoid(ctx, twoX)
	return []Value{Device(two.Mul(sig).Sub(one))}, nil
}

// sigmoid(x) = 1 / (1 + exp(-x))
func sigmoid(ctx *HandlerCtx, x *tensor.Tensor) *tensor.Tensor {
	one := tensor.FullSints(ctx.Arena, x.ShapeSints(), 1.0, x.DType(), x.Device())
	return one.Div(one.Add(x.Neg().Exp()))
}

func handleSigmoid(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Sigmoid")
	if err != nil {
		return nil, err
	}
	return []Value{Device(sigmoid(ctx, x))}, nil
}

// handleRelu composes max(x, 0).
func handleRelu(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Relu")
	if err != nil {
		return nil, err
	}
	zero := tensor.FullSints(ctx.Arena, x.ShapeSints(), 0.0, x.DType(), x.Device())
	return []Value{Device(x.Maximum(zero))}, nil
}

// handleClip composes max(min(x, max_), min_). Opset ≥ 11 reads min/max as
// inputs; opset ≤ 10 reads them as attributes. Defaults: -inf, +inf.
func handleClip(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Clip")
	if err != nil {
		return nil, err
	}
	// Reasonable finite defaults for the unbounded-side cases. Pure
	// arithmetic identities (max with -inf, min with +inf) would also work
	// but we keep this simple.
	minVal := -1.0e38
	maxVal := 1.0e38
	if ctx.Opset >= 11 {
		// inputs[1] = min, inputs[2] = max (both optional / absent if "")
		if len(ctx.Inputs) >= 2 && ctx.Inputs[1].IsDevice() {
			if d := ctx.Inputs[1].Tensor().Data(); len(d) >= 1 {
				minVal = float64(d[0])
			}
		}
		if len(ctx.Inputs) >= 3 && ctx.Inputs[2].IsDevice() {
			if d := ctx.Inputs[2].Tensor().Data(); len(d) >= 1 {
				maxVal = float64(d[0])
			}
		}
	} else {
		minVal = ctx.Node.Attrs["min"].Float(minVal)
		maxVal = ctx.Node.Attrs["max"].Float(maxVal)
	}
	minT := tensor.FullSints(ctx.Arena, x.ShapeSints(), minVal, x.DType(), x.Device())
	maxT := tensor.FullSints(ctx.Arena, x.ShapeSints(), maxVal, x.DType(), x.Device())
	// min(max(x, minVal), maxVal) — use Maximum and a negated-Maximum-of-neg
	// for Minimum (anneal Tensor has no direct .Minimum).
	clipped := x.Maximum(minT)
	// min(a, b) = -max(-a, -b)
	clipped = clipped.Neg().Maximum(maxT.Neg()).Neg()
	return []Value{Device(clipped)}, nil
}

// handleCast emits a Cast graph node to the dtype named by `to`.
func handleCast(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Cast")
	if err != nil {
		return nil, err
	}
	to := onnxpb.TensorProto_DataType(ctx.Node.Attrs["to"].Int(int64(onnxpb.TensorProto_FLOAT)))
	dt, _, _, ok := onnxDType(int32(to))
	if !ok {
		return nil, fmt.Errorf("cast: unsupported `to` dtype %v", to)
	}
	return []Value{Device(x.Cast(dt))}, nil
}

// handleEqual emits an element-wise equality (CmpEq) producing a bool tensor.
func handleEqual(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Equal")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.CmpEq(b))}, nil
}

// ── helpers ──────────────────────────────────────────────────────────────────

func oneTensorInput(ctx *HandlerCtx, name string) (*tensor.Tensor, error) {
	if len(ctx.Inputs) < 1 {
		return nil, fmt.Errorf("%s: expected ≥ 1 input, got %d", name, len(ctx.Inputs))
	}
	if !ctx.Inputs[0].IsDevice() {
		return nil, fmt.Errorf("%s: input 0 is not a device tensor", name)
	}
	return ctx.Inputs[0].Tensor(), nil
}

func twoTensorInputs(ctx *HandlerCtx, name string) (*tensor.Tensor, *tensor.Tensor, error) {
	if len(ctx.Inputs) < 2 {
		return nil, nil, fmt.Errorf("%s: expected ≥ 2 inputs, got %d", name, len(ctx.Inputs))
	}
	if !ctx.Inputs[0].IsDevice() || !ctx.Inputs[1].IsDevice() {
		return nil, nil, fmt.Errorf("%s: inputs are not device tensors", name)
	}
	return ctx.Inputs[0].Tensor(), ctx.Inputs[1].Tensor(), nil
}

var _ = uop.Dtypes // keep uop import alive for future uses
