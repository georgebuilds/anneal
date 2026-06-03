package examples

import (
	"fmt"
	"math/rand"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "resnet9",
		Summary: "ResNet-9 on CIFAR-10 (David Page architecture, 6.57M params, conv compiler surface)",
		Build:   buildResNet9,
		Train:   trainResNet9,
	})
}

// ── ResNet-9 example config ─────────────────────────────────────────────────
//
// Standard David Page / fast.ai ResNet-9 for CIFAR-10. The Build/forward path
// works today; the Train path is currently gated on a WGSL codegen
// scaling bug (see notes/resnet9_progress.md). Build is exercised by
// `anneal run resnet9` / `anneal graph resnet9` / `anneal kernels resnet9`
// and the per-submodule FD tests in tensor/nn are the load-bearing
// correctness gate. Train returns a documented error until the codegen
// fix lands.
const (
	resnet9Batch     = int64(2) // tiny by default — scalar-WGSL ceiling bites hard
	resnet9AdamLR    = float32(1e-3)
	resnet9InitScale = float32(0.05)
)

// resnet9Channels is the canonical David Page channel schedule. Tests in
// tensor/nn use a 4-tuple scale via NewResNet9Scaled; the example uses the
// canonical schedule directly.
var resnet9Channels = [4]int64{64, 128, 256, 512}

// buildResNet9 constructs the forward graph for ResNet-9, returning a
// BuildResult. Used by `anneal run resnet9` / `anneal graph resnet9` /
// `anneal kernels resnet9`. Forward path only.
func buildResNet9(device string) (*BuildResult, error) {
	// Seed arena for parameter shapes + init.
	seedArena := uop.NewArena(1 << 14)
	seed := nn.NewResNet9Scaled(seedArena, resnet9Channels, 10, uop.Dtypes.Float32, device)
	initRNG := rand.New(rand.NewSource(42))
	initResNet9Small(seed, resnet9InitScale, initRNG)

	// Compute arena.
	a := uop.NewArena(1 << 22)
	model := nn.NewResNet9Scaled(a, resnet9Channels, 10, uop.Dtypes.Float32, device)

	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return nil, fmt.Errorf("resnet9: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	leaves := make([]*tensor.Tensor, 0, len(dstParams))
	for _, p := range dstParams {
		p.Load(a)
		leaves = append(leaves, p.T)
	}

	// Synthetic batch matching the dataset shape so the graph realizes
	// without any external dependency.
	xData := make([]float32, resnet9Batch*3*32*32)
	rng := rand.New(rand.NewSource(43))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.3
	}
	x := tensor.NewLeaf(a, []int64{resnet9Batch, 3, 32, 32}, uop.Dtypes.Float32, device)
	x.SetData(xData)

	out := model.Forward(x)

	return &BuildResult{
		Arena:  a,
		Output: out,
		Device: device,
		Leaves: leaves,
	}, nil
}

// trainResNet9 currently returns a descriptive error: the full ResNet-9
// backward graph triggers a WGSL codegen scaling bug (unresolved-identifier
// during shader compilation). The loop body is intentionally NOT in the
// repository so coverage tracks reality — when the codegen fix lands, the
// loop is restored in the same commit. The loop structure follows trainViT:
// fresh-arena-per-step, Adam optimizer, cross-entropy via
// resnet9CrossEntropy, periodic eval via resnet9EvalLoss + dataset.Batch.
//
// See notes/resnet9_progress.md for the codegen gate.
func trainResNet9(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	_ = device
	_ = cfg
	_ = logFn
	return fmt.Errorf("resnet9: training is currently gated on a WGSL codegen bug on the full backward graph; see notes/resnet9_progress.md")
}

// resnet9CrossEntropy is the same mean-NLL-via-log-softmax recipe as
// vitCrossEntropy, specialised for [B, 10]. Builds the loss graph; the
// caller is responsible for tensor.Realize. Exposed at package scope so
// the loss can be tested against a synthetic forward graph without going
// through the gated training loop.
func resnet9CrossEntropy(logits, oneHot *tensor.Tensor, B, numCls int64) *tensor.Tensor {
	a := logits.Arena()
	device := logits.Device()
	dtype := logits.DType()

	expv := logits.Exp()
	sumC := expv.Sum([]int{1}, false)
	sumKD := sumC.Reshape([]int64{B, 1})
	logSum := sumKD.Log()
	logSumB := logSum.Expand([]int64{B, numCls})
	logSoftmax := logits.Sub(logSumB)
	nllPerEl := oneHot.Mul(logSoftmax)
	totalNLL := nllPerEl.Sum(nil, false)
	scale := tensor.ConstScalar(a, -1.0/float64(B), dtype, device)
	return totalNLL.Mul(scale)
}

// initResNet9Small fills every ResNet-9 parameter with small-normal samples.
// BatchNorm Weight gets 1.0 + perturbation, Bias zero-centered. Conv and
// Linear weights are zero-mean normal at the given scale.
func initResNet9Small(m *nn.ResNet9, scale float32, rng *rand.Rand) {
	for _, c := range m.Convs() {
		for i := range c.Weight.Value {
			c.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
	}
	for _, bn := range m.BNs() {
		for i := range bn.Weight.Value {
			bn.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale*0.1
		}
		for i := range bn.Bias.Value {
			bn.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.1
		}
	}
	for i := range m.Head.Weight.Value {
		m.Head.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	if m.Head.Bias != nil {
		for i := range m.Head.Bias.Value {
			m.Head.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
}
