package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
)

// handleConv covers 1-D and 2-D conv with symmetric pads, stride / dilation
// per the plan §6 callouts.
//
// Inputs: X [N, Cin, H, W] (or [N, Cin, L] for 1-D), W [Cout, Cin/group, kH, kW],
// optional B [Cout]. Attrs: kernel_shape, strides, pads, dilations, group,
// auto_pad. We reject auto_pad and group > 1 (current Conv2d does not expose
// grouping) with descriptive errors.
//
// Asymmetric pad (lo != hi on any axis): emit explicit tensor.Pad then conv
// with symmetric zero pads.
func handleConv(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 2 {
		return nil, fmt.Errorf("conv: expected ≥ 2 inputs (X, W)")
	}
	if !ctx.Inputs[0].IsDevice() || !ctx.Inputs[1].IsDevice() {
		return nil, fmt.Errorf("conv: X and W must be device tensors")
	}
	x := ctx.Inputs[0].Tensor()
	w := ctx.Inputs[1].Tensor()
	var b *tensor.Tensor
	if len(ctx.Inputs) >= 3 && ctx.Inputs[2].IsDevice() {
		b = ctx.Inputs[2].Tensor()
	}

	autoPad := ctx.Node.Attrs["auto_pad"].String("NOTSET")
	if autoPad != "NOTSET" && autoPad != "" {
		return nil, fmt.Errorf("conv: auto_pad %q not supported in v1 (use explicit pads)", autoPad)
	}
	group := ctx.Node.Attrs["group"].Int(1)
	if group != 1 {
		return nil, fmt.Errorf("conv: group != 1 not supported in v1 (got %d); requires grouped-Conv2d primitive", group)
	}

	xRank := x.Rank()
	wRank := w.Rank()
	if xRank != wRank {
		return nil, fmt.Errorf("conv: X rank %d != W rank %d", xRank, wRank)
	}
	if xRank != 3 && xRank != 4 {
		return nil, fmt.Errorf("conv: only 1-D / 2-D conv supported in v1 (got rank %d)", xRank)
	}

	// 1-D conv lift: reshape to 4-D with W=1 singleton spatial dim.
	is1D := xRank == 3
	if is1D {
		xs := x.Shape()
		ws := w.Shape()
		x = x.Reshape([]int64{xs[0], xs[1], xs[2], 1})
		w = w.Reshape([]int64{ws[0], ws[1], ws[2], 1})
	}

	kAttr := ctx.Node.Attrs["kernel_shape"].Ints(nil)
	stridesAttr := ctx.Node.Attrs["strides"].Ints(nil)
	padsAttr := ctx.Node.Attrs["pads"].Ints(nil)
	dilAttr := ctx.Node.Attrs["dilations"].Ints(nil)

	// Resolve defaults from W shape for kernel_shape.
	ws := w.Shape()
	kH := ws[2]
	kW := ws[3]
	if len(kAttr) > 0 {
		if is1D {
			if len(kAttr) != 1 {
				return nil, fmt.Errorf("conv: 1-D kernel_shape length %d", len(kAttr))
			}
			kH = kAttr[0]
		} else {
			if len(kAttr) != 2 {
				return nil, fmt.Errorf("conv: 2-D kernel_shape length %d", len(kAttr))
			}
			kH, kW = kAttr[0], kAttr[1]
		}
	}
	_ = kH
	_ = kW

	sH := int64(1)
	sW := int64(1)
	if len(stridesAttr) > 0 {
		if is1D {
			sH = stridesAttr[0]
		} else {
			sH, sW = stridesAttr[0], stridesAttr[1]
		}
	}

	// dilations: only [1,1] supported (Conv2d does not expose dilation).
	if len(dilAttr) > 0 {
		for _, d := range dilAttr {
			if d != 1 {
				return nil, fmt.Errorf("conv: dilations != 1 not supported in v1 (got %v)", dilAttr)
			}
		}
	}

	// pads layout: [x1_begin, x2_begin, ..., x1_end, x2_end, ...].
	var pHbegin, pWbegin, pHend, pWend int64
	if len(padsAttr) == 0 {
		// default zeros
	} else if is1D {
		if len(padsAttr) != 2 {
			return nil, fmt.Errorf("conv: 1-D pads length %d", len(padsAttr))
		}
		pHbegin, pHend = padsAttr[0], padsAttr[1]
	} else {
		if len(padsAttr) != 4 {
			return nil, fmt.Errorf("conv: 2-D pads length %d", len(padsAttr))
		}
		pHbegin, pWbegin, pHend, pWend = padsAttr[0], padsAttr[1], padsAttr[2], padsAttr[3]
	}

	// Asymmetric pad? Emit explicit Pad-then-conv-with-zero-pads.
	asymmetric := pHbegin != pHend || pWbegin != pWend
	convPadH := pHbegin
	convPadW := pWbegin
	if asymmetric {
		padArg := [][2]int64{{0, 0}, {0, 0}, {pHbegin, pHend}, {pWbegin, pWend}}
		x = x.Pad(padArg)
		convPadH = 0
		convPadW = 0
	}

	// Build a Conv2d-equivalent. Cout = W.dim0.
	cout := ws[0]
	cin := ws[1]
	conv := &nn.Conv2d{
		Weight: &nn.Parameter{T: w},
		Stride: [2]int{int(sH), int(sW)},
		Pad:    [2]int{int(convPadH), int(convPadW)},
	}
	if b != nil {
		conv.Bias = &nn.Parameter{T: b}
	}
	_ = cout
	_ = cin

	out := conv.Forward(x)

	// 1-D rollback: drop the trailing singleton spatial dim.
	if is1D {
		osh := out.Shape()
		// shape should be [N, Cout, Ho, 1] - drop the last dim
		out = out.Reshape([]int64{osh[0], osh[1], osh[2]})
	}

	return []Value{Device(out)}, nil
}

var _ = shape.Const
