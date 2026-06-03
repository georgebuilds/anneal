package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
)

// handleBatchNormalization implements the inference path: y = (x - mean) /
// sqrt(var + epsilon) * scale + B. Training-mode output > 1 is rejected.
//
// Inputs: X [N, C, ...spatial], scale [C], B [C], mean [C], var [C].
// Attrs: epsilon (default 1e-5), momentum (ignored, training-only),
// training_mode (must be 0 or absent).
func handleBatchNormalization(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 5 {
		return nil, fmt.Errorf("BatchNormalization: expected 5 inputs (X, scale, B, mean, var), got %d", len(ctx.Inputs))
	}
	for i := 0; i < 5; i++ {
		if !ctx.Inputs[i].IsDevice() {
			return nil, fmt.Errorf("BatchNormalization: input %d is not a device tensor", i)
		}
	}
	if ctx.Node.Attrs["training_mode"].Int(0) != 0 {
		return nil, fmt.Errorf("BatchNormalization: training_mode != 0 not supported in v1")
	}
	if len(ctx.Node.Outputs) > 1 {
		return nil, fmt.Errorf("BatchNormalization: training-mode outputs (running mean/var) not supported in v1")
	}

	x := ctx.Inputs[0].Tensor()
	scale := ctx.Inputs[1].Tensor()
	bias := ctx.Inputs[2].Tensor()
	mean := ctx.Inputs[3].Tensor()
	variance := ctx.Inputs[4].Tensor()

	epsilon := ctx.Node.Attrs["epsilon"].Float(1e-5)

	xSh := x.ShapeSints()
	if len(xSh) < 2 {
		return nil, fmt.Errorf("BatchNormalization: X rank %d, want ≥ 2", len(xSh))
	}
	// Reshape per-channel vectors to [1, C, 1, 1, ...] for broadcast.
	bcShape := make([]shape.Sint, len(xSh))
	for i := range bcShape {
		bcShape[i] = shape.Const(1)
	}
	bcShape[1] = xSh[1]

	scale = scale.ReshapeSints(bcShape)
	bias = bias.ReshapeSints(bcShape)
	mean = mean.ReshapeSints(bcShape)
	variance = variance.ReshapeSints(bcShape)

	// y = (x - mean) * (1 / sqrt(var + eps)) * scale + B
	eps := tensor.FullSints(ctx.Arena, bcShape, epsilon, x.DType(), x.Device())
	denom := variance.Add(eps).Sqrt()
	norm := x.Sub(mean).Div(denom)
	out := norm.Mul(scale).Add(bias)
	return []Value{Device(out)}, nil
}
