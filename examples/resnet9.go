package examples

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	resnet9data "github.com/georgebuilds/anneal/examples/resnet9"
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
// Standard David Page / fast.ai ResNet-9 for CIFAR-10. Build is exercised by
// `anneal run resnet9` / `anneal graph resnet9` / `anneal kernels resnet9`;
// Train runs the full Forward + Backward + Adam + PostStep loop against a
// freshly-streamed CIFAR-10 tarball. Per-submodule FD tests in tensor/nn
// remain the load-bearing correctness gate; see notes/resnet9_progress.md
// for the workstream status.
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

// trainResNet9 runs the full ResNet-9 training loop on CIFAR-10. Pattern
// mirrors trainViT: fresh-arena-per-step, Adam optimizer, cross-entropy via
// resnet9CrossEntropy, periodic eval via resnet9EvalLoss + dataset.Batch.
// On completion the total wall-clock time is emitted via cfg.LogText. The
// canonical 64/128/256/512 channel schedule is used; tests can call
// runResNet9 with a scaled-down config + an in-memory CIFAR10 fixture to
// skip the asset download.
func trainResNet9(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	// CIFAR-10 host pipeline: streaming gzip+tar via the asset registry.
	ds, err := resnet9data.Load()
	if err != nil {
		return fmt.Errorf("resnet9: load CIFAR-10: %w", err)
	}
	return runResNet9(device, cfg, logFn, ds, resnet9Channels, 42)
}

// runResNet9 is the shared trainer used by both the production entry point
// (CIFAR-10 + canonical channels) and the smoke tests (in-memory fixture +
// scaled-down channels). Splitting trainResNet9 this way lets tests skip the
// 170 MB CIFAR-10 download while still exercising the full pipeline.
func runResNet9(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *resnet9data.CIFAR10,
	channels [4]int64,
	seed int64,
) error {
	// cmd_train.go's `--lr` default is SGD-tuned (0.05) and explodes Adam
	// on a conv net. When we see the sentinel (or zero), swap to the
	// canonical ResNet-9 Adam lr; any other LR is respected verbatim.
	lr := cfg.LR
	if lr == cmdTrainSGDDefaultLR || lr == 0 {
		lr = resnet9AdamLR
	}

	batch := cfg.Batch
	if batch <= 0 {
		batch = resnet9Batch
	}

	// Seed arena: allocate parameter shapes, apply small-normal init.
	seedArena := uop.NewArena(1 << 14)
	seedModel := nn.NewResNet9Scaled(seedArena, channels, 10, uop.Dtypes.Float32, device)
	initRNG := rand.New(rand.NewSource(seed))
	initResNet9Small(seedModel, resnet9InitScale, initRNG)

	// Persistent model (values survive arena resets via p.Value).
	a0 := uop.NewArena(1 << 14)
	model := nn.NewResNet9Scaled(a0, channels, 10, uop.Dtypes.Float32, device)
	srcParams := seedModel.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return fmt.Errorf("resnet9: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	params := model.Params()
	opt := nn.NewAdam(params, lr)
	model.Train()

	// Per-step batch-sampling RNG. Independent from the init RNG so a fresh
	// init can be paired with any sampling seed without affecting weights.
	sampleRNG := rand.New(rand.NewSource(seed + 1))

	// Reusable host-side batch buffers; the train step uploads them into a
	// fresh-per-step input leaf.
	const C = int64(3)
	const H = int64(32)
	const W = int64(32)
	const numCls = int64(10)
	xHost := make([]float32, batch*C*H*W)
	yHost := make([]int32, batch)
	ohHost := make([]float32, batch*numCls)

	// Initial-loss probe.
	if cfg.LogEvery > 0 {
		l0 := resnet9EvalLoss(model, params, ds, sampleRNG, batch, device)
		logFn(0, l0)
	}

	start := time.Now()

	for step := 1; step <= cfg.Steps; step++ {
		ds.Batch(sampleRNG, int(batch), xHost, yHost)
		resnet9data.OneHot(yHost, ohHost)

		a := uop.NewArena(1 << 22)
		for _, p := range params {
			p.Load(a)
		}

		x := tensor.NewLeaf(a, []int64{batch, C, H, W}, uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, xHost...))

		oh := tensor.NewLeaf(a, []int64{batch, numCls}, uop.Dtypes.Float32, device)
		oh.SetData(append([]float32{}, ohHost...))

		logits := model.Forward(x) // [B, 10]
		loss := resnet9CrossEntropy(logits, oh, batch, numCls)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		if err := tensor.Realize(loss); err != nil {
			return fmt.Errorf("resnet9: realize loss at step %d: %w", step, err)
		}
		for _, p := range params {
			gr, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(gr); err != nil {
				return fmt.Errorf("resnet9: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if err := model.PostStep(); err != nil {
			return fmt.Errorf("resnet9: PostStep at step %d: %w", step, err)
		}

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := resnet9EvalLoss(model, params, ds, sampleRNG, batch, device)
			logFn(step, lp)
		}
	}

	elapsed := time.Since(start)
	line := fmt.Sprintf("resnet9: training complete in %s (%d steps)\n", elapsed.Round(time.Millisecond), cfg.Steps)
	if cfg.LogText != nil {
		cfg.LogText(line)
	}

	return nil
}

// resnet9EvalLoss recomputes the classification loss for one freshly-sampled
// batch in a fresh arena. Mirrors evalViTLoss; used to log loss without
// keeping the training arena alive across steps.
func resnet9EvalLoss(
	m *nn.ResNet9,
	params []*nn.Parameter,
	ds *resnet9data.CIFAR10,
	rng *rand.Rand,
	B int64,
	device string,
) float32 {
	const C = int64(3)
	const H = int64(32)
	const W = int64(32)
	const numCls = int64(10)

	xHost := make([]float32, B*C*H*W)
	yHost := make([]int32, B)
	ohHost := make([]float32, B*numCls)
	ds.Batch(rng, int(B), xHost, yHost)
	resnet9data.OneHot(yHost, ohHost)

	a := uop.NewArena(1 << 22)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, device)
	x.SetData(xHost)
	oh := tensor.NewLeaf(a, []int64{B, numCls}, uop.Dtypes.Float32, device)
	oh.SetData(ohHost)
	loss := resnet9CrossEntropy(m.Forward(x), oh, B, numCls)
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
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
