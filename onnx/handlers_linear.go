package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
)

// handleGemm computes alpha * (transA?A^T:A) * (transB?B^T:B) + beta * C.
// C is optional; defaults to 0. alpha/beta default 1.
func handleGemm(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 2 {
		return nil, fmt.Errorf("Gemm: expected ≥ 2 inputs")
	}
	if !ctx.Inputs[0].IsDevice() || !ctx.Inputs[1].IsDevice() {
		return nil, fmt.Errorf("Gemm: A and B must be device tensors")
	}
	a := ctx.Inputs[0].Tensor()
	b := ctx.Inputs[1].Tensor()
	transA := ctx.Node.Attrs["transA"].Int(0) != 0
	transB := ctx.Node.Attrs["transB"].Int(0) != 0
	alpha := ctx.Node.Attrs["alpha"].Float(1.0)
	beta := ctx.Node.Attrs["beta"].Float(1.0)

	if transA {
		a = a.T()
	}
	if transB {
		b = b.T()
	}
	out := a.Matmul(b)
	if alpha != 1.0 {
		s := tensor.FullSints(ctx.Arena, out.ShapeSints(), alpha, out.DType(), out.Device())
		out = out.Mul(s)
	}
	if len(ctx.Inputs) >= 3 && ctx.Inputs[2].IsDevice() {
		c := ctx.Inputs[2].Tensor()
		c = tensor.BroadcastToSints(c, out.ShapeSints())
		if beta != 1.0 {
			bs := tensor.FullSints(ctx.Arena, c.ShapeSints(), beta, c.DType(), c.Device())
			c = c.Mul(bs)
		}
		out = out.Add(c)
	}
	return []Value{Device(out)}, nil
}

func handleMatMul(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "MatMul")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.Matmul(b))}, nil
}
