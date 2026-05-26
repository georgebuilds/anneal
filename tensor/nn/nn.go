package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Parameter ─────────────────────────────────────────────────────────────────

// Parameter is a trainable leaf tensor. Phase 6 differentiates w.r.t. the
// underlying BUFFER node; Phase 9 optimizers update it via gradient tensors.
// The seam is tensor.IsLeaf: any BUFFER-backed Tensor is a differentiation target.
//
// Ownership: Value is the canonical weight data and outlives any arena. T is
// the leaf tensor for the current step's graph — rebuilt each step via Load.
// No arena index or pointer is used as the stable identity for this parameter;
// Value is found via the Parameter object itself.
type Parameter struct {
	T     *tensor.Tensor // leaf tensor for the current step; update via Load
	Name  string
	Value []float32 // canonical persistent value; survives arena resets

	shape  []int64
	dtype  *uop.DType
	device string
}

// NewParameter allocates a trainable parameter of the given shape.
// Value is zero-initialized; set it before the first Load call.
func NewParameter(a *uop.Arena, sh []int64, dtype *uop.DType, device string) *Parameter {
	n := 1
	for _, s := range sh {
		n *= int(s)
	}
	shCopy := make([]int64, len(sh))
	copy(shCopy, sh)
	p := &Parameter{
		shape:  shCopy,
		dtype:  dtype,
		device: device,
		Value:  make([]float32, n),
	}
	p.T = tensor.NewLeaf(a, sh, dtype, device)
	return p
}

// Load creates a fresh BUFFER leaf in arena a, seeded with a copy of p.Value.
// Returns the leaf tensor for use in this step's computation graph.
// p.T is updated so that layer Forward() methods see the current-step leaf.
//
// The data path is: p.Value → copy → a.leaves[newLeaf.Index()].
// No index or pointer from any prior arena is used to locate p.Value.
func (p *Parameter) Load(a *uop.Arena) *tensor.Tensor {
	leaf := tensor.NewLeaf(a, p.shape, p.dtype, p.device)
	data := make([]float32, len(p.Value))
	copy(data, p.Value)
	leaf.SetData(data)
	p.T = leaf
	return leaf
}

// SGDStep applies one step of plain SGD in-place: p.Value[i] -= lr * grad[i].
// No arena reference is used; the update is purely on the Parameter-owned Value slice.
func (p *Parameter) SGDStep(grad []float32, lr float32) {
	if len(grad) != len(p.Value) {
		panic(fmt.Sprintf("nn: SGDStep: gradient length %d != parameter length %d", len(grad), len(p.Value)))
	}
	for i := range p.Value {
		p.Value[i] -= lr * grad[i]
	}
}

// ── SGD Optimizer ─────────────────────────────────────────────────────────────

// SGD is a plain stochastic gradient descent optimizer over a fixed parameter set.
// Future optimizers (Adam, AdaGrad) slot in alongside it.
type SGD struct {
	Params []*Parameter
	LR     float32
}

// NewSGD constructs an SGD optimizer over the given parameters.
func NewSGD(params []*Parameter, lr float32) *SGD {
	ps := make([]*Parameter, len(params))
	copy(ps, params)
	return &SGD{Params: ps, LR: lr}
}

// Step applies one SGD update to each parameter whose gradient is in grads.
// grads is the map returned by tensor.Backward, keyed by the step's leaf tensors
// (p.T after p.Load was called for this step). Every gradient tensor in grads must
// have been realized (tensor.Realize called) before Step is invoked.
func (opt *SGD) Step(grads map[*tensor.Tensor]*tensor.Tensor) {
	for _, p := range opt.Params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		p.SGDStep(g.Data(), opt.LR)
	}
}

// ── Activations ───────────────────────────────────────────────────────────────

// ReLU returns max(0, x) expressed via the Maximum primitive.
func ReLU(x *tensor.Tensor) *tensor.Tensor {
	zero := tensor.Full(x.Arena(), x.Shape(), 0.0, x.DType(), x.Device())
	return x.Maximum(zero)
}

// Sigmoid returns 1 / (1 + exp(-x)) via primitives.
func Sigmoid(x *tensor.Tensor) *tensor.Tensor {
	a := x.Arena()
	one := tensor.Full(a, x.Shape(), 1.0, x.DType(), x.Device())
	return one.Div(one.Add(x.Neg().Exp()))
}

// Tanh returns tanh(x) = 2*sigmoid(2x) - 1.
func Tanh(x *tensor.Tensor) *tensor.Tensor {
	a := x.Arena()
	two := tensor.Full(a, x.Shape(), 2.0, x.DType(), x.Device())
	one := tensor.Full(a, x.Shape(), 1.0, x.DType(), x.Device())
	return two.Mul(Sigmoid(two.Mul(x))).Sub(one)
}

// ── Linear ────────────────────────────────────────────────────────────────────

// Linear is a fully-connected layer: y = x @ weight.T + bias.
// Weight shape: [OutFeatures, InFeatures]; bias shape: [OutFeatures].
type Linear struct {
	Weight *Parameter
	Bias   *Parameter // nil when useBias=false
}

// NewLinear constructs a Linear layer. Weights are uninitialised BUFFER leaves;
// the caller or an initialiser fills them before realize().
func NewLinear(a *uop.Arena, inFeatures, outFeatures int64, bias bool, dtype *uop.DType, device string) *Linear {
	l := &Linear{
		Weight: NewParameter(a, []int64{outFeatures, inFeatures}, dtype, device),
	}
	if bias {
		l.Bias = NewParameter(a, []int64{outFeatures}, dtype, device)
	}
	return l
}

// Forward computes x @ weight.T [+ bias].
// x shape: [..., InFeatures]; output shape: [..., OutFeatures].
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	// weight: [OutFeatures, InFeatures] → transpose to [InFeatures, OutFeatures]
	out := x.Matmul(l.Weight.T.Permute([]int{1, 0}))
	if l.Bias != nil {
		// bias: [OutFeatures] → broadcast to [..., OutFeatures]
		b := tensor.BroadcastTo(l.Bias.T, out.Shape())
		out = out.Add(b)
	}
	return out
}

// Params returns all trainable parameters.
func (l *Linear) Params() []*Parameter {
	if l.Bias != nil {
		return []*Parameter{l.Weight, l.Bias}
	}
	return []*Parameter{l.Weight}
}

// ── Conv2d ────────────────────────────────────────────────────────────────────

// Conv2d is a 2-D convolution layer.
// Weight shape: [OutChannels, InChannels, KH, KW].
// The forward pass decomposes into kH*kW matmul accumulations (im2col-style),
// expressed entirely via the primitive movement and reduce ops so Phase 6
// autodiff can differentiate the resulting graph.
//
// Stride=1, dilation=1 only in v1 (strided sampling requires the Phase 7 pool
// primitive for a non-materialized sliding-window view).
type Conv2d struct {
	Weight  *Parameter
	Bias    *Parameter // nil if useBias=false
	Stride  [2]int
	Pad     [2]int
}

// NewConv2d constructs a Conv2d layer.
// kernelSize is [KH, KW].
func NewConv2d(a *uop.Arena, inChannels, outChannels int64, kernelSize [2]int64, stride, pad [2]int, bias bool, dtype *uop.DType, device string) *Conv2d {
	c := &Conv2d{
		Weight: NewParameter(a, []int64{outChannels, inChannels, kernelSize[0], kernelSize[1]}, dtype, device),
		Stride: stride,
		Pad:    pad,
	}
	if bias {
		c.Bias = NewParameter(a, []int64{outChannels}, dtype, device)
	}
	return c
}

// Forward computes the 2-D convolution of x (shape [N, Cin, H, W]).
// Returns [N, Cout, Ho, Wo].
//
// Implementation: im2col + single matmul. Each kernel position (c_k, kh, kw)
// contributes column k = c_k*kH*kW + kh*kW + kw of the im2col matrix. The column
// is assembled via Pad-to-position: a [N, Ho*Wo, 1] slice placed at column k in a
// [N, Ho*Wo, K] tensor with zeros elsewhere. Summing K such tensors gives im2col
// where im2col[n, hw, k] = padded[n, c_k, h_o+kh, w_o+kw] at column k.
//
// The subsequent single batched matmul [N, Ho*Wo, K] @ [K, Cout] → [N, Ho*Wo, Cout]
// is one ReduceAxis, making the conv output ONE materialized buffer. This keeps
// the backward's per-kernel buffer count within WebGPU's 8-buffer limit; the
// prior 9-loop-accumulation approach left conv_out un-materialized, forcing the
// backward's ReLU-mask kernel to reference all 9 intermediate matmul buffers at once.
func (c *Conv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if c.Stride[0] != 1 || c.Stride[1] != 1 {
		panic("tensor/nn: Conv2d with stride != 1 requires the Phase 7 pool primitive")
	}

	xShape := x.Shape()
	N, Cin, H, W := xShape[0], xShape[1], xShape[2], xShape[3]

	wShape := c.Weight.T.Shape()
	Cout, _, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]

	pH, pW := int64(c.Pad[0]), int64(c.Pad[1])

	if H+2*pH < kH || W+2*pW < kW {
		panic(fmt.Sprintf("tensor/nn: Conv2d: kernel %dx%d larger than padded input %dx%d",
			kH, kW, H+2*pH, W+2*pW))
	}

	Ho := H + 2*pH - kH + 1
	Wo := W + 2*pW - kW + 1

	// Pad input if needed.
	var padded *tensor.Tensor
	if pH > 0 || pW > 0 {
		padded = x.Pad([][2]int64{{0, 0}, {0, 0}, {pH, pH}, {pW, pW}})
	} else {
		padded = x
	}

	K := Cin * kH * kW   // total kernel elements; im2col column count
	HoWo := Ho * Wo      // output spatial positions; im2col row count per sample

	// Assemble im2col [N, Ho*Wo, K] by placing each patch at its column via Pad.
	// im2col[n, ho*Wo+wo, k] = padded[n, c_k, ho+kh, wo+kw]
	// where k = c_k*kH*kW + kh*kW + kw.
	//
	// Each patch_k is [N, Ho*Wo, 1] padded to [N, Ho*Wo, K] with zeros outside
	// column k. Summing K such tensors yields the complete im2col without any
	// concat primitive and without extra materialized buffers — all patches share
	// the same padded input leaf, accessed at different index offsets.
	var im2col *tensor.Tensor
	for ck := int64(0); ck < Cin; ck++ {
		for kh := int64(0); kh < kH; kh++ {
			for kw := int64(0); kw < kW; kw++ {
				k := ck*kH*kW + kh*kW + kw
				// Extract single-channel patch: [N, 1, Ho, Wo]
				patch := padded.Shrink([][2]int64{
					{0, N}, {ck, ck + 1}, {kh, kh + Ho}, {kw, kw + Wo},
				})
				// Reshape to column vector: [N, Ho*Wo, 1]
				col := patch.Reshape([]int64{N, HoWo, 1})
				// Pad to full width at position k: [N, Ho*Wo, K]
				col = col.Pad([][2]int64{{0, 0}, {0, 0}, {k, K - 1 - k}})
				if im2col == nil {
					im2col = col
				} else {
					im2col = im2col.Add(col)
				}
			}
		}
	}
	// im2col: [N, HoWo, K]

	// Flatten weight: [Cout, Cin, kH, kW] → [Cout, K] → transpose to [K, Cout]
	wFlat := c.Weight.T.Reshape([]int64{Cout, K}).Permute([]int{1, 0})
	// wFlat: [K, Cout]

	// Single batched matmul: [N, HoWo, K] @ [K, Cout] → [N, HoWo, Cout]
	// One ReduceAxis over K materialises conv_out as a single buffer.
	matout := im2col.Matmul(wFlat)
	// matout: [N, HoWo, Cout]

	// Reorder to standard output: [N, Cout, Ho, Wo]
	out := matout.Permute([]int{0, 2, 1}).Reshape([]int64{N, Cout, Ho, Wo})

	if c.Bias != nil {
		b := c.Bias.T.Reshape([]int64{1, Cout, 1, 1}).Expand([]int64{N, Cout, Ho, Wo})
		out = out.Add(b)
	}

	return out
}

// Params returns all trainable parameters.
func (c *Conv2d) Params() []*Parameter {
	if c.Bias != nil {
		return []*Parameter{c.Weight, c.Bias}
	}
	return []*Parameter{c.Weight}
}
