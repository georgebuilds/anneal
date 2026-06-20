package examples

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// GPT-2 fine-tuning example.
//
// Fine-tunes the canonical tied-head GPT-2-small (HuggingFace weights, loaded
// bit-exact by examples/gpt2) on a BPE-tokenized corpus (tinyshakespeare). This
// is the training counterpart to `anneal gpt2 sample`, which is forward-only.
//
// Three things make this distinct from nanoGPT's trainer:
//
//   1. Tied weights. The LM head shares Wte.Weight; the shared leaf accumulates
//      gradient from both the embedding gather and the LM-head matmul. This is
//      sound since the OpExpand-backward fix (see tensor/gradient_ruleset.go and
//      tensor/nn/gpt_tied_grad_test.go); it was forward-only before.
//
//   2. Numerically stable cross-entropy. Pretrained GPT-2 logits are large
//      enough that the bare exp(logits) used by nanoGPT's loss overflows f32.
//      gpt2StableCrossEntropy subtracts the per-row max before exp. The
//      max-subtract is an exact identity on log-softmax, so the gradient is
//      unchanged (softmax - one_hot); no detach is required.
//
//   3. Fine-tune hygiene: global-norm gradient clipping and a small Adam LR.
//
// The production entry (trainGPT2Finetune) downloads the ~548 MB checkpoint and
// fine-tunes on Shakespeare. The shared core (runGPT2Finetune) is driven by a
// tiny-config CPU test with a synthetic corpus so the full pipeline is exercised
// inside the CI budget without the asset download.

func init() {
	Register(&Example{
		Name:    "gpt2",
		Summary: "fine-tune GPT-2-small (HuggingFace weights) on tinyshakespeare",
		Build:   buildGPT2Finetune,
		Train:   trainGPT2Finetune,
	})
}

const (
	// Fine-tune defaults. The sequence length and batch are deliberately small
	// relative to GPT-2's 1024 block size to keep per-step GPU memory (attention
	// scores scale as B*nHead*T*T) and the V=50257 one-hot target tractable.
	gpt2FinetuneSeqLen   = int64(64)
	gpt2FinetuneBatch    = int64(2)
	gpt2FinetuneLR       = float32(3e-5)
	gpt2FinetuneGradClip = float32(1.0)
	gpt2FinetuneSampleN  = 60
	gpt2FinetunePrompt   = "ROMEO:"
	// cmd_train.go's --lr default (0.05, tuned for the SGD MLP/Conv examples)
	// explodes a GPT-2 Adam update; treat that exact value as "user did not set
	// --lr" and swap to the canonical fine-tune LR. Any other value is honored.
	gpt2CmdTrainDefaultLR = float32(0.05)
)

// gpt2FinetuneConfig groups the model shape + training horizon the build, train,
// and test paths share. The production path fills it from the canonical GPT-2
// constants; the test path plugs in a tiny shape.
type gpt2FinetuneConfig struct {
	Vocab     int
	NLayer    int
	NHead     int
	NEmbd     int
	BlockSize int
	SeqLen    int64
	SampleN   int
}

// ── Train ────────────────────────────────────────────────────────────────────
//
// The production entrypoints (buildGPT2Finetune, trainGPT2Finetune) and the
// corpus loader live in gpt2_finetune_load.go; they depend on the ~548 MB
// checkpoint and are excluded from coverage. The shared core below is measured.

// runGPT2Finetune is the shared fine-tune loop, driven by both the production
// entry (real weights + Shakespeare) and the tiny-config CPU convergence test
// (synthetic corpus). g must be a tied-head GPT (LMHead.Weight == Wte.Weight)
// already seeded; tokens is the full BPE-encoded corpus to sample windows from.
func runGPT2Finetune(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	g *nn.GPT,
	tokens []int32,
	gptCfg gpt2FinetuneConfig,
	encode func(string) []int32,
	decode func([]int32) string,
	seed int64,
) error {
	batch := cfg.Batch
	if batch <= 0 {
		batch = gpt2FinetuneBatch
	}
	lr := cfg.LR
	if lr <= 0 || lr == gpt2CmdTrainDefaultLR {
		lr = gpt2FinetuneLR
	}

	params := g.Params()
	for _, p := range params {
		p.Load(uop.NewArena(1 << 12)) // ensure p.T exists before optimizer captures it
	}
	opt := nn.NewAdam(params, lr)

	T := gptCfg.SeqLen
	V := int64(gptCfg.Vocab)
	sampleRNG := rand.New(rand.NewSource(seed))

	// Step-0 loss probe for the logger.
	if cfg.LogEvery > 0 {
		xs0, ys0 := sampleTokenBatch(rand.New(rand.NewSource(seed+101)), tokens, int(batch), T)
		logFn(0, evalGPT2Loss(g, params, xs0, ys0, batch, T, V, device))
	}

	for step := 1; step <= cfg.Steps; step++ {
		xs, ys := sampleTokenBatch(sampleRNG, tokens, int(batch), T)
		if xs == nil {
			return fmt.Errorf("gpt2 finetune: corpus too small to sample a batch")
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}
		idx := tensor.NewLeaf(a, []int64{batch, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(xs))
		oh := tensor.NewLeaf(a, []int64{batch, T, V}, uop.Dtypes.Float32, device)
		oh.SetData(oneHotBits(ys, gptCfg.Vocab))

		logits := g.Forward(idx)
		loss := gpt2StableCrossEntropy(logits, oh, batch, T, V)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)
		for _, p := range params {
			if gr, ok := grads[p.T]; ok {
				if err := tensor.Realize(gr); err != nil {
					return fmt.Errorf("gpt2 finetune: realize grad %q at step %d: %w", p.Name, step, err)
				}
			}
		}
		clipGradsByGlobalNorm(grads, params, gpt2FinetuneGradClip)
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}
		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := evalGPT2Loss(g, params, xs, ys, batch, T, V, device)
			logFn(step, lp)
		}
	}

	// Final greedy sample from the fine-tuned model.
	if gptCfg.SampleN > 0 && encode != nil && decode != nil {
		sample, err := greedySampleGPT2(g, params, gptCfg, encode, decode, gpt2FinetunePrompt, gptCfg.SampleN, device)
		if err != nil {
			return fmt.Errorf("gpt2 finetune: sample: %w", err)
		}
		emitSample(cfg, sample)
	}
	return nil
}

// ── Numerically stable cross-entropy ─────────────────────────────────────────

// gpt2StableCrossEntropy computes mean per-token cross-entropy with a per-row
// max-subtract for numerical stability:
//
//	m          = max_v(logits)                       (detach-free; gradient cancels)
//	logsoftmax = (logits - m) - log(sum_v exp(logits - m))
//	loss       = -mean_{b,t} sum_v one_hot * logsoftmax
//
// Unlike nanoGPT's crossEntropyLoss this subtracts m before exp, so exp stays in
// (0, 1] and never overflows on pretrained GPT-2 logit magnitudes. The
// max-subtract is an exact identity on log-softmax; autodiff over the identical
// expression yields the identical gradient (softmax - one_hot), so no
// stop-gradient on m is needed.
func gpt2StableCrossEntropy(logits, oneHot *tensor.Tensor, B, T, V int64) *tensor.Tensor {
	a := logits.Arena()
	device := logits.Device()
	dtype := logits.DType()

	maxV := logits.Max([]int{2}, false).Reshape([]int64{B, T, 1}) // [B,T,1]
	shifted := logits.Sub(maxV.Expand([]int64{B, T, V}))          // [B,T,V] <= 0
	expv := shifted.Exp()
	sumV := expv.Sum([]int{2}, false).Reshape([]int64{B, T, 1})
	logSum := sumV.Log()
	logSoftmax := shifted.Sub(logSum.Expand([]int64{B, T, V}))
	nllPerEl := oneHot.Mul(logSoftmax)
	totalNLL := nllPerEl.Sum(nil, false)
	scale := tensor.ConstScalar(a, -1.0/float64(B*T), dtype, device)
	// Contiguous() materializes the full-reduce result before the scalar scale.
	// Without it, the scheduler epilogue-fuses the scale Mul into the large
	// reduce kernel (6.4M elements at V=50257), which miscompiles to 0 on the
	// WebGPU backend (the bare reduce alone is correct; only the fused
	// reduce+scalar-epilogue at scale fails). Contiguous is gradient-transparent
	// (see tensor OpContiguous gradient rule). Underlying scheduler bug tracked
	// in notes/gpt2_train_preflight.md; small vocab (nanoGPT/ViT) never hits it.
	return totalNLL.Contiguous().Mul(scale)
}

// ── Gradient clipping ────────────────────────────────────────────────────────

// clipGradsByGlobalNorm rescales realized gradients in place so the global L2
// norm across all parameters is at most maxNorm. Standard fine-tune hygiene
// (Pascanu et al.); without it the first few steps on pretrained weights can
// take destructive Adam steps. Operates on the host-side gradient buffers
// (grads are already realized via tensor.Realize before this call), so it is a
// plain float32 pass — no graph ops. maxNorm <= 0 disables clipping.
func clipGradsByGlobalNorm(grads map[*tensor.Tensor]*tensor.Tensor, params []*nn.Parameter, maxNorm float32) {
	if maxNorm <= 0 {
		return
	}
	var sumSq float64
	for _, p := range params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		for _, v := range g.Data() {
			sumSq += float64(v) * float64(v)
		}
	}
	norm := float32(math.Sqrt(sumSq))
	if norm <= maxNorm || norm == 0 {
		return
	}
	scale := maxNorm / norm
	for _, p := range params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		d := g.Data()
		for i := range d {
			d[i] *= scale
		}
	}
}

// ── Loss eval + sampling ─────────────────────────────────────────────────────

// evalGPT2Loss recomputes the stable cross-entropy for one batch in a fresh
// arena, independent of the training graph, for logging.
func evalGPT2Loss(g *nn.GPT, params []*nn.Parameter, xs, ys []int32, B, T, V int64, device string) float32 {
	a := uop.NewArena(1 << 20)
	for _, p := range params {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(xs))
	oh := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, device)
	oh.SetData(oneHotBits(ys, int(V)))
	loss := gpt2StableCrossEntropy(g.Forward(idx), oh, B, T, V)
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

// greedySampleGPT2 generates nGen tokens greedily (argmax) from the fine-tuned
// model, using a fixed seqLen rolling window. Mirrors generateNanoGPT; greedy
// decode keeps the convergence demo deterministic.
func greedySampleGPT2(
	g *nn.GPT,
	params []*nn.Parameter,
	gptCfg gpt2FinetuneConfig,
	encode func(string) []int32,
	decode func([]int32) string,
	prompt string,
	nGen int,
	device string,
) (string, error) {
	T := gptCfg.SeqLen
	V := gptCfg.Vocab

	encoded := encode(prompt)
	ctx := make([]int32, T)
	if int64(len(encoded)) >= T {
		copy(ctx, encoded[int64(len(encoded))-T:])
	} else {
		copy(ctx[int(T)-len(encoded):], encoded)
	}
	produced := append([]int32{}, encoded...)

	for k := 0; k < nGen; k++ {
		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}
		idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(ctx))
		logits := g.Forward(idx)
		if err := tensor.Realize(logits); err != nil {
			return "", err
		}
		data := logits.Data()
		base := (int(T) - 1) * V
		best, bestVal := int32(0), float32(math.Inf(-1))
		for j := 0; j < V; j++ {
			if data[base+j] > bestVal {
				bestVal = data[base+j]
				best = int32(j)
			}
		}
		produced = append(produced, best)
		copy(ctx, ctx[1:])
		ctx[T-1] = best
	}
	return decode(produced), nil
}

// ── Batch sampling ───────────────────────────────────────────────────────────

// sampleTokenBatch draws `batch` random length-T windows from tokens, returning
// the inputs xs (flattened [batch*T]) and next-token targets ys (same shape,
// each shifted one position). Returns nil if the corpus is too short.
func sampleTokenBatch(rng *rand.Rand, tokens []int32, batch int, T int64) (xs, ys []int32) {
	n := int64(len(tokens))
	if n < T+1 {
		return nil, nil
	}
	xs = make([]int32, int64(batch)*T)
	ys = make([]int32, int64(batch)*T)
	for b := 0; b < batch; b++ {
		start := int64(rng.Intn(int(n - T)))
		for j := int64(0); j < T; j++ {
			xs[int64(b)*T+j] = tokens[start+j]
			ys[int64(b)*T+j] = tokens[start+j+1]
		}
	}
	return xs, ys
}

// emitSample (nanogpt.go) and int32sAsBits / oneHotBits (nanogpt_data.go) are
// shared across examples.
