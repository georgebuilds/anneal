package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── ResNet-9 ─────────────────────────────────────────────────────────────────

// ResNet9 is the David Page / fast.ai CIFAR-10 architecture. Five named blocks
// over a 32x32x3 input lowering to a 10-way classification head:
//
//	prep    : Conv(3   → 64,  3x3, pad=1) → BN → ReLU                     [32x32]
//	layer1  : Conv(64  → 128, 3x3, pad=1) → BN → ReLU → MaxPool(2)         [16x16]
//	          + residual: Conv(128→128) → BN → ReLU → Conv(128→128) → BN
//	            output = ReLU(layer1 + res)
//	layer2  : Conv(128 → 256, 3x3, pad=1) → BN → ReLU → MaxPool(2)          [8x8]
//	layer3  : Conv(256 → 512, 3x3, pad=1) → BN → ReLU → MaxPool(2)          [4x4]
//	          + residual: Conv(512→512) → BN → ReLU → Conv(512→512) → BN
//	            output = ReLU(layer3 + res)
//	pool    : MaxPool(4) → [1x1]
//	flat    : Reshape [B, 512]
//	head    : Linear(512 → 10)
//
// Param count: ~6.57M, all f32. Trained end-to-end with Adam + per-step arena
// reset. BatchNorm running stats are updated by walking PostStep over all BN
// submodules after each training step's Realize.
//
// The forward uses Conv2d's existing im2col-as-matmul lowering, which
// materialises each conv output as a single buffer (8-buffer cap safe). The
// residual Add reuses the OpAdd gradient rule; BatchNorm gradient flows
// through normalize + affine via primitive rules. col2im in backward is the
// Shrink/Pad dual via gradient_ruleset.
type ResNet9 struct {
	PrepConv *Conv2d
	PrepBN   *BatchNorm2d

	L1Conv *Conv2d
	L1BN   *BatchNorm2d

	R1Conv1 *Conv2d
	R1BN1   *BatchNorm2d
	R1Conv2 *Conv2d
	R1BN2   *BatchNorm2d

	L2Conv *Conv2d
	L2BN   *BatchNorm2d

	L3Conv *Conv2d
	L3BN   *BatchNorm2d

	R3Conv1 *Conv2d
	R3BN1   *BatchNorm2d
	R3Conv2 *Conv2d
	R3BN2   *BatchNorm2d

	Head *Linear
}

// NewResNet9 constructs the standard ResNet-9 with 64/128/256/512 channels.
// All Conv layers use 3x3 kernels with pad=1 and bias=false (BN follows).
//
// For unit tests and quick smoke runs, use NewResNet9Scaled to dial channels
// down: the im2col lowering in Conv2d generates O(Cin*kH*kW) graph nodes per
// conv, so at Cin=512 each conv contributes 4608 nodes and the full network
// graph approaches ~10K UOps — fine for the headline 90% training run on a
// downloaded CIFAR-10, but too heavy for an in-process unit test.
func NewResNet9(a *uop.Arena, numClasses int64, dtype *uop.DType, device string) *ResNet9 {
	return NewResNet9Scaled(a, [4]int64{64, 128, 256, 512}, numClasses, dtype, device)
}

// NewResNet9Scaled constructs a ResNet-9 with the four channel counts the
// architecture expands through. channels[0] is the prep output (was 64);
// channels[1..3] are the per-stage output widths (was 128, 256, 512). The
// residual blocks reuse channels[1] (after layer1) and channels[3] (after
// layer3). All Conv layers use 3x3 kernels with pad=1 and bias=false.
func NewResNet9Scaled(a *uop.Arena, channels [4]int64, numClasses int64, dtype *uop.DType, device string) *ResNet9 {
	for i, c := range channels {
		if c <= 0 {
			panic(fmt.Sprintf("nn: NewResNet9Scaled: channels[%d] must be positive, got %d", i, c))
		}
	}
	c0, c1, c2, c3 := channels[0], channels[1], channels[2], channels[3]

	mk := func(in, out int64) *Conv2d {
		return NewConv2d(a, in, out, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, false, dtype, device)
	}
	mkBN := func(c int64) *BatchNorm2d {
		return NewBatchNorm2d(a, c, 1e-5, 0.1, dtype, device)
	}
	return &ResNet9{
		PrepConv: mk(3, c0),
		PrepBN:   mkBN(c0),

		L1Conv: mk(c0, c1),
		L1BN:   mkBN(c1),

		R1Conv1: mk(c1, c1),
		R1BN1:   mkBN(c1),
		R1Conv2: mk(c1, c1),
		R1BN2:   mkBN(c1),

		L2Conv: mk(c1, c2),
		L2BN:   mkBN(c2),

		L3Conv: mk(c2, c3),
		L3BN:   mkBN(c3),

		R3Conv1: mk(c3, c3),
		R3BN1:   mkBN(c3),
		R3Conv2: mk(c3, c3),
		R3BN2:   mkBN(c3),

		Head: NewLinear(a, c3, numClasses, true, dtype, device),
	}
}

// Forward runs the network on x of shape [B, 3, 32, 32]. Returns logits
// of shape [B, numClasses].
func (m *ResNet9) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x.Rank() != 4 {
		panic(fmt.Sprintf("nn: ResNet9.Forward: expected 4-D input, got rank %d", x.Rank()))
	}
	xShape := x.Shape()
	if xShape[1] != 3 || xShape[2] != 32 || xShape[3] != 32 {
		panic(fmt.Sprintf("nn: ResNet9.Forward: expected [B,3,32,32], got %v", xShape))
	}
	B := xShape[0]

	// prep: Conv→BN→ReLU.
	h := ReLU(m.PrepBN.Forward(m.PrepConv.Forward(x)))

	// layer1: Conv→BN→ReLU→MaxPool.
	h = ReLU(m.L1BN.Forward(m.L1Conv.Forward(h)))
	h = MaxPool2D(h, 2, 2, 2, 2)
	// residual1: Conv→BN→ReLU→Conv→BN.
	r := ReLU(m.R1BN1.Forward(m.R1Conv1.Forward(h)))
	r = m.R1BN2.Forward(m.R1Conv2.Forward(r))
	h = ReLU(h.Add(r))

	// layer2: Conv→BN→ReLU→MaxPool.
	h = ReLU(m.L2BN.Forward(m.L2Conv.Forward(h)))
	h = MaxPool2D(h, 2, 2, 2, 2)

	// layer3: Conv→BN→ReLU→MaxPool.
	h = ReLU(m.L3BN.Forward(m.L3Conv.Forward(h)))
	h = MaxPool2D(h, 2, 2, 2, 2)
	// residual3: Conv→BN→ReLU→Conv→BN.
	r = ReLU(m.R3BN1.Forward(m.R3Conv1.Forward(h)))
	r = m.R3BN2.Forward(m.R3Conv2.Forward(r))
	h = ReLU(h.Add(r))

	// pool 4x4 → 1x1; flatten; head.
	h = MaxPool2D(h, 4, 4, 4, 4)
	// Recover the final-stage channel count from the head's weight shape
	// so the network is correct under any channel-scale config.
	c3 := m.Head.Weight.T.Shape()[1]
	h = h.Reshape([]int64{B, c3})
	return m.Head.Forward(h)
}

// Convs returns all Conv2d submodules in topological order.
func (m *ResNet9) Convs() []*Conv2d {
	return []*Conv2d{
		m.PrepConv,
		m.L1Conv,
		m.R1Conv1, m.R1Conv2,
		m.L2Conv,
		m.L3Conv,
		m.R3Conv1, m.R3Conv2,
	}
}

// BNs returns all BatchNorm2d submodules in topological order. Use this to
// walk PostStep over all BN layers after a training step's Realize.
func (m *ResNet9) BNs() []*BatchNorm2d {
	return []*BatchNorm2d{
		m.PrepBN,
		m.L1BN,
		m.R1BN1, m.R1BN2,
		m.L2BN,
		m.L3BN,
		m.R3BN1, m.R3BN2,
	}
}

// PostStep updates running statistics on all BatchNorm2d submodules. Must be
// called AFTER the training step's Realize completes.
func (m *ResNet9) PostStep() error {
	for i, bn := range m.BNs() {
		if err := bn.PostStep(); err != nil {
			return fmt.Errorf("nn: ResNet9.PostStep: BN[%d]: %w", i, err)
		}
	}
	return nil
}

// Train switches all BatchNorm2d submodules into training mode.
func (m *ResNet9) Train() {
	for _, bn := range m.BNs() {
		bn.Train()
	}
}

// Eval switches all BatchNorm2d submodules into evaluation mode.
func (m *ResNet9) Eval() {
	for _, bn := range m.BNs() {
		bn.Eval()
	}
}

// Params returns all trainable parameters in deterministic topological order.
// Conv biases are skipped (constructor sets bias=false on every Conv); BN
// running stats are NOT trainable and are not included.
func (m *ResNet9) Params() []*Parameter {
	var ps []*Parameter
	for _, c := range m.Convs() {
		ps = append(ps, c.Weight)
		if c.Bias != nil {
			ps = append(ps, c.Bias)
		}
	}
	for _, bn := range m.BNs() {
		ps = append(ps, bn.Params()...)
	}
	ps = append(ps, m.Head.Params()...)
	return ps
}

// Load reloads every trainable parameter onto arena a's leaf tensors. Call
// this once at the start of each training step, before building the forward
// graph in a freshly-reset arena.
func (m *ResNet9) Load(a *uop.Arena) {
	for _, p := range m.Params() {
		p.Load(a)
	}
}

// ParamCount returns the total number of trainable scalar parameters. Useful
// for sanity-checking against the reference (~6.57M for the standard config).
func (m *ResNet9) ParamCount() int64 {
	var n int64
	for _, p := range m.Params() {
		var sz int64 = 1
		for _, d := range p.shape {
			sz *= d
		}
		n += sz
	}
	return n
}
