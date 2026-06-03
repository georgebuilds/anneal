package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor/nn"
)

// handleMaxPool calls tensor/nn.MaxPool2D with attrs from the node. v1
// supports kernel_shape (required), strides (default 1), pads (default 0).
// Rejects auto_pad, ceil_mode, dilations != 1, storage_order != 0.
func handleMaxPool(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "MaxPool")
	if err != nil {
		return nil, err
	}
	if x.Rank() != 4 {
		return nil, fmt.Errorf("MaxPool: input rank %d, want 4 (only 2-D pool in v1)", x.Rank())
	}
	autoPad := ctx.Node.Attrs["auto_pad"].String("NOTSET")
	if autoPad != "NOTSET" && autoPad != "" {
		return nil, fmt.Errorf("MaxPool: auto_pad %q not supported in v1", autoPad)
	}
	if ctx.Node.Attrs["ceil_mode"].Int(0) != 0 {
		return nil, fmt.Errorf("MaxPool: ceil_mode != 0 not supported in v1")
	}
	if ctx.Node.Attrs["storage_order"].Int(0) != 0 {
		return nil, fmt.Errorf("MaxPool: storage_order != 0 not supported in v1")
	}
	if dil := ctx.Node.Attrs["dilations"].Ints(nil); len(dil) > 0 {
		for _, d := range dil {
			if d != 1 {
				return nil, fmt.Errorf("MaxPool: dilations != 1 not supported in v1 (got %v)", dil)
			}
		}
	}

	ks := ctx.Node.Attrs["kernel_shape"].Ints(nil)
	if len(ks) != 2 {
		return nil, fmt.Errorf("MaxPool: kernel_shape length %d, want 2", len(ks))
	}
	kH, kW := ks[0], ks[1]
	sH := int64(1)
	sW := int64(1)
	if s := ctx.Node.Attrs["strides"].Ints(nil); len(s) == 2 {
		sH, sW = s[0], s[1]
	}
	// pads: [Hb, Wb, He, We]; v1 only supports symmetric and we emit explicit
	// Pad for asymmetric.
	pads := ctx.Node.Attrs["pads"].Ints(nil)
	x2 := x
	if len(pads) == 4 {
		Hb, Wb, He, We := pads[0], pads[1], pads[2], pads[3]
		if Hb != 0 || Wb != 0 || He != 0 || We != 0 {
			padArg := [][2]int64{{0, 0}, {0, 0}, {Hb, He}, {Wb, We}}
			x2 = x.Pad(padArg)
		}
	}
	out := nn.MaxPool2D(x2, kH, kW, sH, sW)
	return []Value{Device(out)}, nil
}

// handleGlobalAveragePool collapses spatial dims to 1 by averaging.
// Input [N, C, *spatial] → output [N, C, 1, 1, ...] (keepdim=true).
func handleGlobalAveragePool(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "GlobalAveragePool")
	if err != nil {
		return nil, err
	}
	rank := x.Rank()
	if rank < 3 {
		return nil, fmt.Errorf("GlobalAveragePool: rank %d, want ≥ 3", rank)
	}
	// Reduce axes 2..rank-1.
	axes := make([]int, 0, rank-2)
	for i := 2; i < rank; i++ {
		axes = append(axes, i)
	}
	out := x.Mean(axes, true)
	return []Value{Device(out)}, nil
}
