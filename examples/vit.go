package examples

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "vit",
		Summary: "vision transformer (patch embed + 2-block encoder + mean-pool head); synthetic 32x32 RGB classification",
		Build:   buildViT,
		Train:   trainViT,
	})
}

// ── ViT example config ──────────────────────────────────────────────────────
//
// ViT-tiny on a synthetic CIFAR-shaped task: 32x32 RGB images, 10 classes,
// patch size 4 (so 64 patch tokens per image), embedding 64, 2 layers,
// 4 heads. Smaller than the textbook ViT-Tiny (192 dim, 12 layers); we cut
// every dim aggressively so the example trains in seconds and the FD test
// in the nn package runs under the standard per-test budget.
//
// The model exists to demonstrate the patch-embed + encoder-stack +
// classification-head wiring end to end. The compiler-surface story
// (forward/backward fusion through Linear, LayerNorm, attention softmax,
// GELU) is identical to nanoGPT. New surface exercised: PatchEmbed's
// reshape/permute/reshape chain into Linear (verified zero-copy via
// rangeify), and the mean-pool over patch tokens at the head.
const (
	vitImageH     = int64(32)
	vitImageW     = int64(32)
	vitPatch      = int64(4)
	vitInCh       = int64(3)
	vitEmbedDim   = int64(64)
	vitNLayer     = 2
	vitNHead      = 4
	vitNumClasses = int64(10)
	vitBatch      = int64(8)
	vitAdamLR     = float32(3e-4)
	vitInitScale  = float32(0.02)
)

// buildViT constructs the forward graph for ViT, returning a BuildResult.
// Used by `anneal run vit` / `anneal graph vit` / `anneal kernels vit`.
func buildViT(device string) (*BuildResult, error) {
	// Seed arena: allocate parameter shapes, apply small-normal init.
	seedArena := uop.NewArena(1 << 14)
	seed := nn.NewViT(seedArena, vitImageH, vitImageW, vitPatch, vitInCh,
		vitEmbedDim, vitNLayer, vitNHead, vitNumClasses)
	initRNG := rand.New(rand.NewSource(42))
	initViTSmall(seed, vitInitScale, initRNG)

	// Compute arena.
	a := uop.NewArena(1 << 20)
	model := nn.NewViT(a, vitImageH, vitImageW, vitPatch, vitInCh,
		vitEmbedDim, vitNLayer, vitNHead, vitNumClasses)

	// Copy initialised values position-by-position from the seed model.
	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return nil, fmt.Errorf("vit: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	// Load params into compute arena and collect leaf tensors.
	leaves := make([]*tensor.Tensor, 0, len(dstParams))
	for _, p := range dstParams {
		p.Load(a)
		leaves = append(leaves, p.T)
	}

	// Input tensor with synthetic batch.
	images, _ := vitDataset(vitBatch, vitInCh, vitImageH, vitImageW,
		vitNumClasses, rand.New(rand.NewSource(43)))
	x := tensor.NewLeaf(a, []int64{vitBatch, vitInCh, vitImageH, vitImageW},
		uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, images...))

	out := model.Forward(x)

	return &BuildResult{
		Arena:  a,
		Output: out,
		Device: device,
		Leaves: leaves,
	}, nil
}

// trainViT runs the ViT training loop.
func trainViT(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	// Use Adam by default. cmd_train's --lr default is SGD-tuned (0.05); when
	// the caller passes that sentinel through, switch to vitAdamLR. Any other
	// value is respected verbatim. Mirrors the nanogpt example's pattern.
	lr := cfg.LR
	if lr == cmdTrainSGDDefaultLR || lr == 0 {
		lr = vitAdamLR
	}

	batch := cfg.Batch
	if batch <= 0 {
		batch = vitBatch
	}

	// Seed arena.
	seedArena := uop.NewArena(1 << 14)
	seed := nn.NewViT(seedArena, vitImageH, vitImageW, vitPatch, vitInCh,
		vitEmbedDim, vitNLayer, vitNHead, vitNumClasses)
	initRNG := rand.New(rand.NewSource(42))
	initViTSmall(seed, vitInitScale, initRNG)

	// Persistent model (values survive arena resets).
	a0 := uop.NewArena(1 << 14)
	model := nn.NewViT(a0, vitImageH, vitImageW, vitPatch, vitInCh,
		vitEmbedDim, vitNLayer, vitNHead, vitNumClasses)
	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return fmt.Errorf("vit: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	params := model.Params()
	opt := nn.NewAdam(params, lr)

	// Synthetic dataset: fixed batch of B images with K class-conditional
	// patterns. The classifier learns to separate the K patterns from each
	// other; loss starts around log(K) and drops as the model fits.
	dataRNG := rand.New(rand.NewSource(43))
	images, labels := vitDataset(batch, vitInCh, vitImageH, vitImageW,
		vitNumClasses, dataRNG)

	// Initial-loss probe.
	if cfg.LogEvery > 0 {
		l0 := evalViTLoss(model, params, images, labels, batch, device)
		logFn(0, l0)
	}

	for step := 1; step <= cfg.Steps; step++ {
		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}

		x := tensor.NewLeaf(a, []int64{batch, vitInCh, vitImageH, vitImageW},
			uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, images...))

		// One-hot targets [B, numClasses].
		oh := tensor.NewLeaf(a, []int64{batch, vitNumClasses}, uop.Dtypes.Float32, device)
		oh.SetData(oneHotBitsViT(labels, int(vitNumClasses)))

		logits := model.Forward(x) // [B, numClasses]
		loss := vitCrossEntropy(logits, oh, batch, vitNumClasses)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		for _, p := range params {
			gr, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(gr); err != nil {
				return fmt.Errorf("vit: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := evalViTLoss(model, params, images, labels, batch, device)
			logFn(step, lp)
		}
	}

	return nil
}

// vitCrossEntropy computes mean per-sample cross-entropy over a batch of
// classification logits. Same numerical recipe as crossEntropyLoss (no max
// shift; init scale keeps exp in float32 range), specialised to the rank-2
// [B, numClasses] tensor shape that the ViT head produces.
func vitCrossEntropy(logits, oneHot *tensor.Tensor, B, numCls int64) *tensor.Tensor {
	a := logits.Arena()
	device := logits.Device()
	dtype := logits.DType()

	expv := logits.Exp()                 // [B, C]
	sumC := expv.Sum([]int{1}, false)    // [B]
	sumKD := sumC.Reshape([]int64{B, 1}) // [B, 1]
	logSum := sumKD.Log()                // [B, 1]
	logSumB := logSum.Expand([]int64{B, numCls})
	logSoftmax := logits.Sub(logSumB)    // [B, C]
	nllPerEl := oneHot.Mul(logSoftmax)   // [B, C] (only correct-class entry nonzero)
	totalNLL := nllPerEl.Sum(nil, false) // scalar
	scale := tensor.ConstScalar(a, -1.0/float64(B), dtype, device)
	return totalNLL.Mul(scale)
}

// evalViTLoss recomputes the classification loss for one batch in a fresh
// arena. Mirrors evalNanoGPTLoss; used to log loss without keeping the
// training arena alive across steps.
func evalViTLoss(
	v *nn.ViT,
	params []*nn.Parameter,
	images []float32,
	labels []int32,
	B int64,
	device string,
) float32 {
	a := uop.NewArena(1 << 20)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, vitInCh, vitImageH, vitImageW},
		uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, images...))
	oh := tensor.NewLeaf(a, []int64{B, vitNumClasses}, uop.Dtypes.Float32, device)
	oh.SetData(oneHotBitsViT(labels, int(vitNumClasses)))
	loss := vitCrossEntropy(v.Forward(x), oh, B, vitNumClasses)
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

// initViTSmall fills every ViT parameter with small-normal samples, mirroring
// vitInitSmall in vit_test.go. LayerNorm Weight gets 1.0 + perturbation,
// Bias zero-centered. Patch/Pos/Block/Head weights are zero-mean normal at
// the given scale.
func initViTSmall(v *nn.ViT, scale float32, rng *rand.Rand) {
	for i := range v.Patch.Proj.Weight.Value {
		v.Patch.Proj.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	if v.Patch.Proj.Bias != nil {
		for i := range v.Patch.Proj.Bias.Value {
			v.Patch.Proj.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
	for i := range v.PosEmb.Value {
		v.PosEmb.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for _, b := range v.Blocks {
		for i := range b.LN1.Weight.Value {
			b.LN1.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
		for i := range b.LN1.Bias.Value {
			b.LN1.Bias.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.LN2.Weight.Value {
			b.LN2.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
		for i := range b.LN2.Bias.Value {
			b.LN2.Bias.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.Attn.QKV.Weight.Value {
			b.Attn.QKV.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.Attn.Proj.Weight.Value {
			b.Attn.Proj.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if b.Attn.QKV.Bias != nil {
			for i := range b.Attn.QKV.Bias.Value {
				b.Attn.QKV.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
		if b.Attn.Proj.Bias != nil {
			for i := range b.Attn.Proj.Bias.Value {
				b.Attn.Proj.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
		for i := range b.MLP.FC1.Weight.Value {
			b.MLP.FC1.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.MLP.FC2.Weight.Value {
			b.MLP.FC2.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if b.MLP.FC1.Bias != nil {
			for i := range b.MLP.FC1.Bias.Value {
				b.MLP.FC1.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
		if b.MLP.FC2.Bias != nil {
			for i := range b.MLP.FC2.Bias.Value {
				b.MLP.FC2.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
	}
	for i := range v.LNf.Weight.Value {
		v.LNf.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
	}
	for i := range v.LNf.Bias.Value {
		v.LNf.Bias.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for i := range v.Head.Weight.Value {
		v.Head.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	if v.Head.Bias != nil {
		for i := range v.Head.Bias.Value {
			v.Head.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
}

// vitDataset generates a fixed synthetic image-classification batch.
// Each image is filled with a class-conditional pattern (per-channel mean +
// low-amplitude noise) so a sufficiently expressive model can fit the
// pattern-to-label mapping by extracting per-channel intensity, attending
// across patches, and predicting the right class. We pick a smooth pattern
// (constant offset per channel) because a Vision Transformer with no
// inductive bias has to learn its convolution-equivalent through positional
// embeddings; that takes many steps on truly textured data, more than the
// CLI demo budget. The synthetic task is calibrated so loss decreases
// monotonically over ~50 steps from the random init.
func vitDataset(B, C, H, W, numCls int64, rng *rand.Rand) (images []float32, labels []int32) {
	images = make([]float32, B*C*H*W)
	labels = make([]int32, B)

	// Per-class per-channel mean intensity. Spread across the [-1, 1] range
	// so different classes are linearly separable on the channel means.
	classCMeans := make([][]float32, numCls)
	for k := int64(0); k < numCls; k++ {
		classCMeans[k] = make([]float32, C)
		for c := int64(0); c < C; c++ {
			classCMeans[k][c] = -1.0 + 2.0*float32(k)/float32(numCls-1) +
				0.15*float32(c-C/2)
		}
	}

	for n := int64(0); n < B; n++ {
		k := int64(rng.Intn(int(numCls)))
		labels[n] = int32(k)
		base := n * C * H * W
		for c := int64(0); c < C; c++ {
			mean := classCMeans[k][c]
			for hh := int64(0); hh < H; hh++ {
				for ww := int64(0); ww < W; ww++ {
					idx := base + c*H*W + hh*W + ww
					// Small additive Gaussian noise on top of the class mean.
					images[idx] = mean + float32(rng.NormFloat64())*0.05
				}
			}
		}
	}
	return
}

// oneHotBitsViT converts a length-B []int32 of labels in [0, numCls) into a
// [B*numCls] float32 buffer suitable for a tensor.NewLeaf SetData call: 1.0
// at (b, labels[b]), 0.0 elsewhere.
func oneHotBitsViT(labels []int32, numCls int) []float32 {
	out := make([]float32, len(labels)*numCls)
	for b, k := range labels {
		if k < 0 || int(k) >= numCls {
			panic(fmt.Sprintf("vit: oneHotBitsViT: label %d out of range [0,%d)", k, numCls))
		}
		out[b*numCls+int(k)] = 1.0
	}
	return out
}
