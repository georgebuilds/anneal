package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// readReduceAxes implements the opset-11/12 (axes attr) vs opset-13+ (axes
// input for ReduceSum; opset-18+ for the rest) duck-typing. We probe input
// first when input count > 1, then attr.
func readReduceAxes(ctx *HandlerCtx) ([]int, error) {
	// Prefer input[1] when present (post opset 13/18 migration).
	if len(ctx.Inputs) >= 2 && ctx.Inputs[1].Kind != KindUnset {
		vs, err := asHostIntVec(ctx.Inputs[1])
		if err != nil {
			return nil, fmt.Errorf("axes input: %w", err)
		}
		out := make([]int, len(vs))
		for i, a := range vs {
			out[i] = int(a)
		}
		return out, nil
	}
	if a, ok := ctx.Node.Attrs["axes"]; ok && a.Kind == AttrInts {
		out := make([]int, len(a.Is))
		for i, v := range a.Is {
			out[i] = int(v)
		}
		return out, nil
	}
	return nil, nil
}

// reduceWithF32Accumulator runs `redFn(x_f32, axes, keepdim)` on a tensor
// up-cast to f32, then casts back to the original dtype. Matches the plan §6
// reduction-accumulator pattern: f16/bf16 reductions overflow to +inf
// without the up-cast.
func reduceWithF32Accumulator(x *tensor.Tensor, axes []int, keepdim bool,
	redFn func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor) *tensor.Tensor {
	origDT := x.DType()
	needsUpcast := origDT == uop.Dtypes.Float16 || origDT == uop.Dtypes.BFloat16
	if needsUpcast {
		x = x.Cast(uop.Dtypes.Float32)
	}
	y := redFn(x, axes, keepdim)
	if needsUpcast {
		y = y.Cast(origDT)
	}
	return y
}

func handleReduceSum(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "ReduceSum")
	if err != nil {
		return nil, err
	}
	axes, err := readReduceAxes(ctx)
	if err != nil {
		return nil, fmt.Errorf("ReduceSum: %w", err)
	}
	keepdim := ctx.Node.Attrs["keepdims"].Int(1) != 0
	out := reduceWithF32Accumulator(x, axes, keepdim,
		func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor {
			return t.Sum(axes, keepdim)
		})
	return []Value{Device(out)}, nil
}

func handleReduceMean(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "ReduceMean")
	if err != nil {
		return nil, err
	}
	axes, err := readReduceAxes(ctx)
	if err != nil {
		return nil, fmt.Errorf("ReduceMean: %w", err)
	}
	keepdim := ctx.Node.Attrs["keepdims"].Int(1) != 0
	out := reduceWithF32Accumulator(x, axes, keepdim,
		func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor {
			return t.Mean(axes, keepdim)
		})
	return []Value{Device(out)}, nil
}

func handleReduceMax(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "ReduceMax")
	if err != nil {
		return nil, err
	}
	axes, err := readReduceAxes(ctx)
	if err != nil {
		return nil, fmt.Errorf("ReduceMax: %w", err)
	}
	keepdim := ctx.Node.Attrs["keepdims"].Int(1) != 0
	out := reduceWithF32Accumulator(x, axes, keepdim,
		func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor {
			return t.Max(axes, keepdim)
		})
	return []Value{Device(out)}, nil
}
