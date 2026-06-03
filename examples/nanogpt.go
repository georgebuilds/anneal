package examples

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// NanoGPTStreamToken is one record emitted by NanoGPTGenerateStream — a
// per-token callback payload for the studio's /sse/generate handler. The
// shape mirrors gpt2.StreamToken so the SSE wire format stays uniform.
type NanoGPTStreamToken struct {
	Step         int     // 0-based token index within this generation
	ID           int32   // sampled token id (argmax for the current path)
	Text         string  // decoded text fragment for this id alone
	Argmax       int32   // argmax id at this step
	LogitMax     float32 // value of the max logit
	LogitMaxIdx  int     // index where the max logit lives
	LogitSummary string  // pre-formatted "max=X.YZ at idx N"
}

// NanoGPTGenerateStream is the streaming entry point used by the studio's
// /sse/generate handler. It constructs a fresh nanoGPT (seeded
// deterministically so successive runs over the same prompt diverge only
// on context), encodes the prompt against the tinyshakespeare vocabulary,
// and runs nGen autoregressive steps. Each emitted token is reported via
// onTok with the per-step record; the function returns the decoded text
// after the loop completes.
//
// The model is initialized fresh (no training); the generation is a
// compiler-correctness demo, not a language-modelling demo. The
// per-token kernel pulse and the WGSL click-through are the value
// proposition.
//
// The context is checked at the top of every step so a client disconnect
// (caught by the SSE handler) cleanly aborts.
func NanoGPTGenerateStream(
	ctx context.Context,
	device string,
	prompt string,
	nGen int,
	onTok func(NanoGPTStreamToken),
) (string, error) {
	if onTok == nil {
		return "", fmt.Errorf("nanogpt: NanoGPTGenerateStream: onTok callback must be non-nil")
	}
	if nGen <= 0 {
		return "", fmt.Errorf("nanogpt: NanoGPTGenerateStream: nGen must be > 0, got %d", nGen)
	}

	ds, err := loadShakespeareDataset()
	if err != nil {
		return "", err
	}
	cfg := defaultNanoGPTConfig(ds.VocabSize())

	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPT(a0, cfg.Vocab, cfg.NLayer, cfg.NHead, cfg.NEmbd, cfg.BlockSize)
	initGPTSmall(g, nanoGPTInitScale, rand.New(rand.NewSource(42)))
	params := g.Params()

	T := int64(cfg.BlockSize)
	V := cfg.Vocab

	encoded := ds.Encode(prompt)
	ctxBuf := make([]int32, T)
	if int64(len(encoded)) >= T {
		copy(ctxBuf, encoded[int64(len(encoded))-T:])
	} else {
		pad := int(T) - len(encoded)
		copy(ctxBuf[pad:], encoded)
	}
	produced := make([]int32, 0, len(encoded)+nGen)
	produced = append(produced, encoded...)

	for k := 0; k < nGen; k++ {
		if err := ctx.Err(); err != nil {
			return ds.Decode(produced), err
		}
		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}
		idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(ctxBuf))

		logits := g.Forward(idx)
		if err := tensor.Realize(logits); err != nil {
			return ds.Decode(produced), fmt.Errorf("nanogpt: NanoGPTGenerateStream: realize at step %d: %w", k, err)
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
		copy(ctxBuf, ctxBuf[1:])
		ctxBuf[T-1] = bestID

		fragment := ds.Decode([]int32{bestID})
		onTok(NanoGPTStreamToken{
			Step:         k,
			ID:           bestID,
			Text:         fragment,
			Argmax:       bestID,
			LogitMax:     bestVal,
			LogitMaxIdx:  int(bestID),
			LogitSummary: fmt.Sprintf("max=%.3f at idx %d", bestVal, bestID),
		})
	}
	return ds.Decode(produced), nil
}

func init() {
	Register(&Example{
		Name:    "nanogpt",
		Summary: "char-level transformer; trains on tinyshakespeare and emits a sample",
		Build:   buildNanoGPT,
		Train:   trainNanoGPT,
	})
}

// ── nanoGPT default config ──────────────────────────────────────────────────
//
// These constants describe a "tiny but transformer-shaped" config that runs
// end to end on the current anneal compiler in reasonable wall time. The
// karpathy/nanoGPT canonical char-level Shakespeare config (4 layers, 4
// heads, n_embd=128, block_size=64) is the target shape but lives outside
// the compile-throughput budget of this build (per-step compile+dispatch is
// minutes at canonical size; see slice-N report). We default to a smaller
// shape so `anneal train nanogpt --steps=100` finishes during a coffee
// break; the model is still a multi-block pre-LN transformer with causal
// self-attention, MLP, embeddings, and an LM head, so the demo exercises
// every Wave 2 module.
//
//	n_layer    = 2
//	n_head     = 2
//	n_embd     = 64
//	block_size = 32
//	batch      = 4  (overridable via cfg.Batch)
//	lr        ~= 3e-4 (Adam; auto-selected when cmd_train's SGD-tuned
//	                   --lr default is left in place)
//
// vocab_size is derived from the corpus (tinyshakespeare yields ~65 chars).
const (
	nanoGPTNLayer    = 2
	nanoGPTNHead     = 2
	nanoGPTNEmbd     = 64
	nanoGPTBlockSize = 32
	nanoGPTBatch     = int64(4)
	nanoGPTAdamLR    = float32(3e-4)
	nanoGPTInitScale = float32(0.02)
	// nanoGPTSamplePrompt is the seed text used to kick off generation
	// after training. The decoded characters are filtered against the
	// dataset vocabulary, so any chars not in tinyshakespeare are dropped
	// silently. "ROMEO:" is in the corpus.
	nanoGPTSamplePrompt = "ROMEO:"
	// nanoGPTSampleTokens is the number of tokens to generate after
	// training when running via the train CLI.
	nanoGPTSampleTokens = 100
	// cmdTrainSGDDefaultLR is cmd_train.go's `--lr` default (tuned for the
	// pre-existing MLP/Conv SGD examples). We treat the value 0.05 as the
	// "user did not set --lr" sentinel and silently swap to nanoGPTAdamLR;
	// any other value the caller passes through is respected verbatim.
	cmdTrainSGDDefaultLR = float32(0.05)
)

// nanoGPTConfig groups the knobs the build / train / sample paths share.
// Exists so the convergence smoke test can plug in a tiny config without
// re-deriving every shape constant.
type nanoGPTConfig struct {
	Vocab     int
	NLayer    int
	NHead     int
	NEmbd     int
	BlockSize int
	// SampleTokens is how many tokens to generate after the final training
	// step. Zero falls back to nanoGPTSampleTokens.
	SampleTokens int
}

func defaultNanoGPTConfig(vocab int) nanoGPTConfig {
	return nanoGPTConfig{
		Vocab:        vocab,
		NLayer:       nanoGPTNLayer,
		NHead:        nanoGPTNHead,
		NEmbd:        nanoGPTNEmbd,
		BlockSize:    nanoGPTBlockSize,
		SampleTokens: nanoGPTSampleTokens,
	}
}

// ── Build: registry pre-flight ───────────────────────────────────────────────
//
// Build constructs a GPT forward graph with a fixed prompt-shaped input so
// the run / graph / kernels commands have something to inspect without
// running a full training loop. The corpus is loaded so the vocab size
// matches what train would use; this also primes the asset cache.
//
// When ANNEAL_OFFLINE=1 is set and the corpus is not cached, Build returns
// a clear error so the CLI surfaces the offline state to the user.
func buildNanoGPT(device string) (*BuildResult, error) {
	ds, err := loadShakespeareDataset()
	if err != nil {
		return nil, err
	}
	cfg := defaultNanoGPTConfig(ds.VocabSize())
	// Use the configured block_size and a batch of 1 for the pre-flight.
	const B = int64(1)
	T := int64(cfg.BlockSize)

	a := uop.NewArena(1 << 20)
	g := nn.NewGPT(a, cfg.Vocab, cfg.NLayer, cfg.NHead, cfg.NEmbd, cfg.BlockSize)

	rng := rand.New(rand.NewSource(42))
	initGPTSmall(g, nanoGPTInitScale, rng)

	for _, p := range g.Params() {
		p.Load(a)
	}

	// Seed the input with the first T tokens of the corpus so the kernels
	// dispatch path always sees valid indices in [0, vocab).
	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = ds.Data[int64(i)%int64(len(ds.Data))]
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(idxVals))

	logits := g.Forward(idx)

	leaves := make([]*tensor.Tensor, 0, len(g.Params()))
	for _, p := range g.Params() {
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

// trainNanoGPT runs the char-level transformer training loop with Adam.
//
// Per cfg.Steps:
//   - Fresh arena per step (matches the cmd_train.go / MLP / Conv convention).
//   - Random sequential window from the corpus, batched to cfg.Batch.
//   - Forward through GPT -> logits [B, T, V].
//   - Loss = mean cross-entropy via one-hot @ log_softmax (sidesteps gather
//     on the last axis; Phase 2 plan, notes/roadmap.md).
//   - Backward + Adam.Step.
//
// After the last step, the function prints a sample of nanoGPTSampleTokens
// characters seeded by nanoGPTSamplePrompt to whatever sink cfg.LogText
// targets (defaults to stdout). Loss values flow through cfg.LogFn / logFn
// as usual.
func trainNanoGPT(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	ds, err := loadShakespeareDataset()
	if err != nil {
		return err
	}
	return runNanoGPT(device, cfg, logFn, ds, defaultNanoGPTConfig(ds.VocabSize()), 42)
}

// runNanoGPT is the shared trainer used by both the production entry point
// (Shakespeare corpus, default config) and the convergence smoke test
// (in-memory fixture, tiny config). Splitting Train this way lets the test
// skip the 30 MB asset download while still exercising the full pipeline.
//
// The seed argument controls both the parameter initialisation RNG and the
// per-step batch-sampling RNG, so identical seeds produce identical
// trajectories (subject to GPU determinism, which is exercised separately).
func runNanoGPT(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *charDataset,
	gptCfg nanoGPTConfig,
	seed int64,
) error {
	batch := cfg.Batch
	if batch <= 0 {
		batch = nanoGPTBatch
	}
	lr := cfg.LR
	// The cmd_train.go --lr default (0.05) is tuned for the existing
	// MLP/Conv SGD examples and explodes Adam updates on a transformer.
	// When we see that exact value we silently swap to the canonical
	// nanoGPT Adam lr (3e-4); any other LR the caller passes is respected.
	if lr <= 0 || lr == cmdTrainSGDDefaultLR {
		lr = nanoGPTAdamLR
	}

	// Persistent model. Parameters survive arena resets via p.Value.
	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPT(a0, gptCfg.Vocab, gptCfg.NLayer, gptCfg.NHead, gptCfg.NEmbd, gptCfg.BlockSize)
	initRNG := rand.New(rand.NewSource(seed))
	initGPTSmall(g, nanoGPTInitScale, initRNG)

	params := g.Params()
	opt := nn.NewAdam(params, lr)

	// Sampling RNG is independent from the init RNG so a fresh init can be
	// paired with any sampling seed without affecting weight initialisation.
	sampleRNG := rand.New(rand.NewSource(seed + 1))

	// Pre-flight: validate block-size fits in corpus.
	if len(ds.Data) < gptCfg.BlockSize+1 {
		return fmt.Errorf("nanogpt: corpus length %d < block_size+1 = %d", len(ds.Data), gptCfg.BlockSize+1)
	}

	// Initial-loss probe (one fixed batch reused only for logging the step-0 line).
	if cfg.LogEvery > 0 {
		xs0, ys0 := ds.SampleBatch(rand.New(rand.NewSource(seed+101)), int(batch), gptCfg.BlockSize)
		l0 := evalNanoGPTLoss(g, params, xs0, ys0, batch, int64(gptCfg.BlockSize), gptCfg.Vocab, device)
		logFn(0, l0)
	}

	T := int64(gptCfg.BlockSize)
	V := int64(gptCfg.Vocab)

	for step := 1; step <= cfg.Steps; step++ {
		xs, ys := ds.SampleBatch(sampleRNG, int(batch), gptCfg.BlockSize)
		if xs == nil {
			return fmt.Errorf("nanogpt: failed to sample batch (corpus too small)")
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}

		idx := tensor.NewLeaf(a, []int64{batch, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(xs))

		// One-hot targets shape [B, T, V].
		oh := tensor.NewLeaf(a, []int64{batch, T, V}, uop.Dtypes.Float32, device)
		oh.SetData(oneHotBits(ys, gptCfg.Vocab))

		logits := g.Forward(idx)
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
				return fmt.Errorf("nanogpt: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			// Realize loss for logging. The forward graph already references
			// the realized params; we need a fresh build for an honest probe,
			// reusing the just-trained weights via Load on a new arena.
			lp := evalNanoGPTLoss(g, params, xs, ys, batch, T, gptCfg.Vocab, device)
			logFn(step, lp)
		}
	}

	// Final sample. The decoded text is printed via cfg.LogText (if set) or
	// directly to stdout otherwise. Tests stream this into a bytes.Buffer.
	nGen := gptCfg.SampleTokens
	if nGen <= 0 {
		nGen = nanoGPTSampleTokens
	}
	sample, err := generateNanoGPT(g, params, gptCfg, ds, nanoGPTSamplePrompt, nGen, sampleRNG, device)
	if err != nil {
		return fmt.Errorf("nanogpt: generation: %w", err)
	}
	emitSample(cfg, sample)

	return nil
}

// emitSample prints the generated sample using cfg.LogText if set, falling
// back to stdout. Wrapped so the TUI / plain / test paths share one sink.
func emitSample(cfg TrainConfig, sample string) {
	line := "\nsample (" + nanoGPTSamplePrompt + " ...):\n" + sample + "\n"
	if cfg.LogText != nil {
		cfg.LogText(line)
		return
	}
	//nolint:errcheck // best-effort write
	fmt.Fprint(os.Stdout, line)
}

// ── Loss: cross-entropy via one-hot @ log_softmax ─────────────────────────────
//
// crossEntropyLoss computes mean per-token cross-entropy:
//
//	log_softmax(logits) = logits - log(sum_v(exp(logits)))
//	per_token_nll       = -sum_v(one_hot * log_softmax)
//	loss                = mean over (b, t) of per_token_nll
//
// The one-hot tensor is constructed host-side from the targets and uploaded
// as a fresh [B, T, V] float32 leaf. The reduction order matches the
// "sidesteps gather" plan in notes/roadmap.md (Phase 2): a single
// elementwise multiply followed by Sum over all axes yields the scalar loss.
//
// Numerical-stability note: we omit the standard max-subtract because the
// OpMax backward inflates the per-kernel buffer count and would hit the
// WebGPU 8-buffer cap. Initialisation at scale nanoGPTInitScale (~0.02)
// keeps exp(logits) in float32 range for the training horizons we target.
// Matches the precedent used in tensor/nn/attention.go.
func crossEntropyLoss(logits, oneHot *tensor.Tensor, B, T, V int64) *tensor.Tensor {
	a := logits.Arena()
	device := logits.Device()
	dtype := logits.DType()

	expv := logits.Exp()                       // [B, T, V]
	sumV := expv.Sum([]int{2}, false)          // [B, T]
	sumKD := sumV.Reshape([]int64{B, T, 1})    // [B, T, 1]
	logSum := sumKD.Log()                      // [B, T, 1]
	logSumB := logSum.Expand([]int64{B, T, V}) // [B, T, V] (broadcast)
	logSoftmax := logits.Sub(logSumB)          // [B, T, V]
	nllPerEl := oneHot.Mul(logSoftmax)         // [B, T, V] (only y-col nonzero)
	totalNLL := nllPerEl.Sum(nil, false)       // scalar
	scale := tensor.ConstScalar(a, -1.0/float64(B*T), dtype, device)
	return totalNLL.Mul(scale)
}

// evalNanoGPTLoss recomputes the loss for one batch in a fresh arena,
// independently of the training forward / backward graph. Used to log loss
// values without keeping the autodiff arena alive across steps.
func evalNanoGPTLoss(
	g *nn.GPT,
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
	loss := crossEntropyLoss(g.Forward(idx), oh, B, T, int64(V))
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

// ── Generation ───────────────────────────────────────────────────────────────
//
// generateNanoGPT produces nGen new tokens starting from prompt. The model
// always sees an exactly block-size window [1, T=blockSize]: we left-pad the
// initial prompt with token 0 if shorter, and slide the window forward by
// one as we append new tokens. The argmax of the last-position logits is
// sampled (greedy decode); this is sufficient for the "is the loss going
// down well enough for output to be Shakespeare-ish" smoke test and keeps
// the path deterministic.
//
// rng is currently unused (we use argmax) but is plumbed so a future
// stochastic-sampling switch does not change the function signature.
func generateNanoGPT(
	g *nn.GPT,
	params []*nn.Parameter,
	cfg nanoGPTConfig,
	ds *charDataset,
	prompt string,
	nGen int,
	rng *rand.Rand,
	device string,
) (string, error) {
	_ = rng
	T := int64(cfg.BlockSize)
	V := cfg.Vocab

	// Encode the prompt; drop any characters not in the corpus vocabulary.
	encoded := ds.Encode(prompt)
	// Build the rolling context window of length T. We left-pad with token 0
	// when the prompt is shorter than T.
	ctx := make([]int32, T)
	if int64(len(encoded)) >= T {
		copy(ctx, encoded[int64(len(encoded))-T:])
	} else {
		// Left-pad with token 0; place the prompt at the tail.
		pad := int(T) - len(encoded)
		copy(ctx[pad:], encoded)
	}
	// Track only the generated suffix (not the padding/prompt) for the
	// returned string. Keep the full encoded prompt as a prefix so the
	// caller sees what the model conditioned on.
	produced := make([]int32, 0, len(encoded)+nGen)
	produced = append(produced, encoded...)

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
		// Last-position logits live at the end of the [1, T, V] buffer.
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
		// Slide window: drop first token, append bestID.
		copy(ctx, ctx[1:])
		ctx[T-1] = bestID
	}

	return ds.Decode(produced), nil
}

// ── Initialisation ───────────────────────────────────────────────────────────
//
// initGPTSmall seeds every learnable parameter in the GPT with small normal
// samples. LayerNorm Weight stays near 1.0, Bias near 0.0; all other matrices
// are sampled from N(0, scale²). Mirrors gptInitSmall in tensor/nn/gpt_test.go
// (we keep our own copy here because the test helper is package-internal).
func initGPTSmall(g *nn.GPT, scale float32, rng *rand.Rand) {
	fillNormal := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rng.NormFloat64()) * scale
		}
	}
	fillLN := func(weight, bias []float32) {
		for i := range weight {
			weight[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
		for i := range bias {
			bias[i] = float32(rng.NormFloat64()) * scale
		}
	}

	fillNormal(g.Wte.Weight.Value)
	fillNormal(g.Wpe.Weight.Value)

	for _, b := range g.Blocks {
		fillLN(b.LN1.Weight.Value, b.LN1.Bias.Value)
		fillLN(b.LN2.Weight.Value, b.LN2.Bias.Value)
		fillNormal(b.Attn.QKV.Weight.Value)
		fillNormal(b.Attn.Proj.Weight.Value)
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
		fillNormal(b.MLP.FC1.Weight.Value)
		fillNormal(b.MLP.FC2.Weight.Value)
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

	fillLN(g.LNf.Weight.Value, g.LNf.Bias.Value)
	fillNormal(g.LMHead.Weight.Value)
	if g.LMHead.Bias != nil {
		for i := range g.LMHead.Bias.Value {
			g.LMHead.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
}
