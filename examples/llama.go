package examples

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// Llama is the modern decoder-only example: a from-scratch char-level language
// model with the Llama/Qwen/Gemma primitive stack — RMSNorm, grouped-query
// attention with RoPE, SwiGLU feed-forward, and tied embeddings. It is the
// "level-up" companion to nanoGPT (which uses LayerNorm, learned absolute
// positions, vanilla multi-head attention, and a GELU MLP), and it reuses
// nanoGPT's tinyshakespeare char dataset so the example stays self-contained.

func init() {
	Register(&Example{
		Name:    "llama",
		Summary: "Llama-style decoder (RoPE + RMSNorm + SwiGLU + GQA, tied embeddings); trains on tinyshakespeare",
		Build:   buildLlama,
		Train:   trainLlama,
	})
}

// ── llama default config ──────────────────────────────────────────────────────
//
// A "tiny but Llama-shaped" config that runs end to end in reasonable wall time
// while exercising every modern primitive: 2 pre-RMSNorm blocks, grouped-query
// attention (4 query heads sharing 2 KV heads, group size 2) with RoPE, and a
// SwiGLU FFN. vocab is derived from the corpus (tinyshakespeare ~65 chars).
const (
	llamaNLayer         = 2
	llamaNHead          = 4
	llamaNKVHead        = 2 // group size = NHead/NKVHead = 2
	llamaNEmbd          = 64
	llamaBlockSize      = 32
	llamaSwiGLUMultiple = 32
	llamaBatch          = int64(4)
	llamaAdamLR         = float32(3e-4)
	llamaInitScale      = float32(0.02)
	llamaRoPEBase       = float64(10000)
	llamaSamplePrompt   = "ROMEO:"
	llamaSampleTokens   = 100
)

// llamaConfig groups the knobs the build / train / sample paths share. Exists so
// the convergence smoke test can plug in a tiny config without re-deriving every
// shape constant.
type llamaConfig struct {
	Vocab     int
	NLayer    int
	NHead     int
	NKVHead   int
	NEmbd     int
	Hidden    int
	BlockSize int
	// SampleTokens is how many tokens to generate after the final step. Zero
	// falls back to llamaSampleTokens.
	SampleTokens int
}

func defaultLlamaConfig(vocab int) llamaConfig {
	return llamaConfig{
		Vocab:        vocab,
		NLayer:       llamaNLayer,
		NHead:        llamaNHead,
		NKVHead:      llamaNKVHead,
		NEmbd:        llamaNEmbd,
		Hidden:       nn.SwiGLUHidden(llamaNEmbd, llamaSwiGLUMultiple),
		BlockSize:    llamaBlockSize,
		SampleTokens: llamaSampleTokens,
	}
}

func newLlamaModel(a *uop.Arena, cfg llamaConfig) *nn.Llama {
	return nn.NewLlama(a, cfg.Vocab, cfg.NLayer, cfg.NHead, cfg.NKVHead,
		cfg.NEmbd, cfg.Hidden, cfg.BlockSize, llamaRoPEBase)
}

// ── Build: registry pre-flight ───────────────────────────────────────────────
//
// buildLlama constructs a Llama forward graph with a fixed prompt-shaped input
// so the run / graph / kernels commands have something to inspect without
// running a full training loop. The corpus is loaded so the vocab matches what
// train would use (and to prime the asset cache).
func buildLlama(device string) (*BuildResult, error) {
	ds, err := loadDataset()
	if err != nil {
		return nil, err
	}
	cfg := defaultLlamaConfig(ds.VocabSize())
	const B = int64(1)
	T := int64(cfg.BlockSize)

	a := uop.NewArena(1 << 20)
	m := newLlamaModel(a, cfg)
	initLlamaSmall(m, llamaInitScale, rand.New(rand.NewSource(42)))

	for _, p := range m.Params() {
		p.Load(a)
	}

	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = ds.Data[int64(i)%int64(len(ds.Data))]
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(idxVals))

	logits := m.Forward(idx)

	leaves := make([]*tensor.Tensor, 0, len(m.Params()))
	for _, p := range m.Params() {
		leaves = append(leaves, p.T)
	}
	return &BuildResult{
		Arena:  a,
		Output: logits,
		Device: device,
		Leaves: leaves,
	}, nil
}

// ── Train ────────────────────────────────────────────────────────────────────

// trainLlama runs the char-level Llama training loop with Adam, then emits a
// generated sample. The corpus and default config are resolved here; runLlama
// is the shared body the convergence smoke test drives with a tiny in-memory
// fixture.
func trainLlama(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	ds, err := loadDataset()
	if err != nil {
		return err
	}
	return runLlama(device, cfg, logFn, ds, defaultLlamaConfig(ds.VocabSize()), 42)
}

// runLlama is the shared trainer used by the production entry point and the
// convergence smoke test. seed controls both the parameter-init RNG and the
// per-step batch-sampling RNG so identical seeds produce identical trajectories.
func runLlama(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *charDataset,
	llCfg llamaConfig,
	seed int64,
) error {
	batch := cfg.Batch
	if batch <= 0 {
		batch = llamaBatch
	}
	lr := cfg.LR
	// cmd_train.go's --lr default (0.05) is tuned for the SGD MLP/Conv examples
	// and explodes Adam on a transformer; swap to the canonical Adam lr when we
	// see that exact sentinel, respecting any other LR the caller passes.
	if lr <= 0 || lr == cmdTrainSGDDefaultLR {
		lr = llamaAdamLR
	}

	a0 := uop.NewArena(1 << 14)
	m := newLlamaModel(a0, llCfg)
	initLlamaSmall(m, llamaInitScale, rand.New(rand.NewSource(seed)))

	params := m.Params()
	opt := nn.NewAdam(params, lr)

	sampleRNG := rand.New(rand.NewSource(seed + 1))

	if len(ds.Data) < llCfg.BlockSize+1 {
		return fmt.Errorf("llama: corpus length %d < block_size+1 = %d", len(ds.Data), llCfg.BlockSize+1)
	}

	if cfg.LogEvery > 0 {
		xs0, ys0 := ds.SampleBatch(rand.New(rand.NewSource(seed+101)), int(batch), llCfg.BlockSize)
		l0 := evalLlamaLoss(m, params, xs0, ys0, batch, int64(llCfg.BlockSize), llCfg.Vocab, device)
		logFn(0, l0)
	}

	T := int64(llCfg.BlockSize)
	V := int64(llCfg.Vocab)

	for step := 1; step <= cfg.Steps; step++ {
		xs, ys := ds.SampleBatch(sampleRNG, int(batch), llCfg.BlockSize)
		if xs == nil {
			return fmt.Errorf("llama: failed to sample batch (corpus too small)")
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}

		idx := tensor.NewLeaf(a, []int64{batch, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(xs))

		oh := tensor.NewLeaf(a, []int64{batch, T, V}, uop.Dtypes.Float32, device)
		oh.SetData(oneHotBits(ys, llCfg.Vocab))

		logits := m.Forward(idx)
		loss := crossEntropyLoss(logits, oh, batch, T, V)

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
				return fmt.Errorf("llama: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := evalLlamaLoss(m, params, xs, ys, batch, T, llCfg.Vocab, device)
			logFn(step, lp)
		}
	}

	nGen := llCfg.SampleTokens
	if nGen <= 0 {
		nGen = llamaSampleTokens
	}
	sample, err := generateLlama(m, params, llCfg, ds, llamaSamplePrompt, nGen, device)
	if err != nil {
		return fmt.Errorf("llama: generation: %w", err)
	}
	emitLlamaSample(cfg, sample)

	return nil
}

// emitLlamaSample prints the generated sample via cfg.LogText if set, else stdout.
func emitLlamaSample(cfg TrainConfig, sample string) {
	line := "\nsample (" + llamaSamplePrompt + " ...):\n" + sample + "\n"
	if cfg.LogText != nil {
		cfg.LogText(line)
		return
	}
	//nolint:errcheck // best-effort write
	fmt.Fprint(os.Stdout, line)
}

// evalLlamaLoss recomputes the loss for one batch in a fresh arena, independent
// of the training forward / backward graph (used for honest loss logging).
func evalLlamaLoss(
	m *nn.Llama,
	params []*nn.Parameter,
	xs, ys []int32,
	B, T int64,
	V int,
	device string,
) float32 {
	a := uop.NewArena(1 << 20)
	for _, p := range params {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(xs))
	oh := tensor.NewLeaf(a, []int64{B, T, int64(V)}, uop.Dtypes.Float32, device)
	oh.SetData(oneHotBits(ys, V))
	loss := crossEntropyLoss(m.Forward(idx), oh, B, T, int64(V))
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

// ── Generation ───────────────────────────────────────────────────────────────
//
// generateLlama produces nGen tokens by greedy (argmax) decode over a rolling
// block-size window, mirroring generateNanoGPT. The model always sees an
// exactly block-size context; the prompt is left-padded with token 0 if shorter.
func generateLlama(
	m *nn.Llama,
	params []*nn.Parameter,
	cfg llamaConfig,
	ds *charDataset,
	prompt string,
	nGen int,
	device string,
) (string, error) {
	T := int64(cfg.BlockSize)
	V := cfg.Vocab

	encoded := ds.Encode(prompt)
	ctx := make([]int32, T)
	if int64(len(encoded)) >= T {
		copy(ctx, encoded[int64(len(encoded))-T:])
	} else {
		pad := int(T) - len(encoded)
		copy(ctx[pad:], encoded)
	}
	produced := make([]int32, 0, len(encoded)+nGen)
	produced = append(produced, encoded...)

	for k := 0; k < nGen; k++ {
		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}
		idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(ctx))

		logits := m.Forward(idx)
		if err := tensor.Realize(logits); err != nil {
			return "", err
		}
		data := logits.Data()
		base := (int(T) - 1) * V
		var bestID int32
		bestVal := float32(math.Inf(-1))
		for j := 0; j < V; j++ {
			v := data[base+j]
			if v > bestVal {
				bestVal = v
				bestID = int32(j)
			}
		}
		produced = append(produced, bestID)
		copy(ctx, ctx[1:])
		ctx[T-1] = bestID
	}

	return ds.Decode(produced), nil
}

// ── Initialisation ───────────────────────────────────────────────────────────
//
// initLlamaSmall seeds every learnable parameter with small normal samples.
// RMSNorm Weight stays near 1.0; all projection / embedding matrices are sampled
// from N(0, scale²). The Llama stack is bias-free, so there are no bias buffers
// to seed, and LMHead.Weight is tied to Tok.Weight (seeded once via Tok).
func initLlamaSmall(m *nn.Llama, scale float32, rng *rand.Rand) {
	fillNormal := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rng.NormFloat64()) * scale
		}
	}
	fillRMS := func(weight []float32) {
		for i := range weight {
			weight[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
	}

	fillNormal(m.Tok.Weight.Value)

	for _, b := range m.Blocks {
		fillRMS(b.Norm1.Weight.Value)
		fillNormal(b.Attn.Q.Weight.Value)
		fillNormal(b.Attn.K.Weight.Value)
		fillNormal(b.Attn.V.Weight.Value)
		fillNormal(b.Attn.Proj.Weight.Value)
		fillRMS(b.Norm2.Weight.Value)
		fillNormal(b.MLP.Gate.Weight.Value)
		fillNormal(b.MLP.Up.Weight.Value)
		fillNormal(b.MLP.Down.Weight.Value)
	}

	fillRMS(m.NormF.Weight.Value)
	// LMHead.Weight is tied to Tok.Weight (already filled); no bias to seed.
}
