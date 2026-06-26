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

// BERT is the encoder-only example: a from-scratch char-level bidirectional
// transformer trained with masked language modeling (MLM). Where nanoGPT and
// Llama are causal decoders (each token attends only to the past and predicts
// the next token), BERT attends in BOTH directions and predicts a random
// subset of input tokens that have been replaced by a [MASK] sentinel. It is
// the "other half" of the transformer zoo and reuses nanoGPT's tinyshakespeare
// char dataset so the example stays self-contained.
//
// Architecturally BERT is pure composition over the existing kit (see
// tensor/nn/bert.go): a token nn.Embedding plus a learned position Parameter,
// a stack of non-causal pre-LN nn.ViTBlock encoder blocks (the same block ViT
// uses; its NewSelfAttention has an all-ones mask, so attention is genuinely
// bidirectional), a final LayerNorm, and a separate nn.Linear masked-LM head.
// The only net-new code is example-level: the masked cross-entropy objective,
// the host-side mask sampler, and this train loop.

func init() {
	Register(&Example{
		Name:    "bert",
		Summary: "BERT encoder (bidirectional attention + masked-LM); trains on tinyshakespeare",
		Build:   buildBERT,
		Train:   trainBERT,
	})
}

// ── BERT default config ───────────────────────────────────────────────────────
//
// A "tiny but BERT-shaped" config that runs end to end on the current anneal
// compiler in reasonable wall time while exercising the encoder stack: 2
// non-causal pre-LN blocks, 2 heads, nEmbd=64, sequence length 32, masking 15%
// of positions per BERT's recipe. vocab is derived from the corpus
// (tinyshakespeare ~65 chars) plus one row for the [MASK] sentinel.
const (
	bertNLayer    = 2
	bertNHead     = 2
	bertNEmbd     = 64
	bertBlockSize = 32
	bertBatch     = int64(4)
	bertSteps     = 50
	bertAdamLR    = float32(3e-4)
	bertInitScale = float32(0.02)
	bertMaskProb  = 0.15
)

// bertConfig groups the knobs the build / train paths share. Exists so the
// convergence smoke test and the CPU full-loop test can plug in a tiny config
// without re-deriving every shape constant.
//
// Vocab is the MODEL vocabulary (BaseVocab + 1); the extra row is the [MASK]
// sentinel whose id is exactly BaseVocab. Targets are always real corpus ids in
// [0, BaseVocab), so the [MASK] column never receives a positive label.
type bertConfig struct {
	Vocab     int     // model vocab = BaseVocab + 1 (extra row is [MASK])
	BaseVocab int     // distinct corpus tokens; [MASK] sentinel id == BaseVocab
	NLayer    int     // number of encoder blocks
	NHead     int     // attention heads per block
	NEmbd     int     // embedding / residual width
	BlockSize int     // sequence length T
	Batch     int64   // default batch; TrainConfig.Batch overrides when > 0
	MaskProb  float64 // fraction of positions masked per sequence
	LR        float32 // default Adam lr; TrainConfig.LR overrides unless the SGD sentinel
	Steps     int     // default step count; TrainConfig.Steps overrides when > 0
}

// MaskID is the [MASK] sentinel token id: the synthetic vocab row appended past
// the real corpus tokens.
func (c bertConfig) MaskID() int32 { return int32(c.BaseVocab) }

func defaultBERTConfig(baseVocab int) bertConfig {
	return bertConfig{
		Vocab:     baseVocab + 1,
		BaseVocab: baseVocab,
		NLayer:    bertNLayer,
		NHead:     bertNHead,
		NEmbd:     bertNEmbd,
		BlockSize: bertBlockSize,
		Batch:     bertBatch,
		MaskProb:  bertMaskProb,
		LR:        bertAdamLR,
		Steps:     bertSteps,
	}
}

// tinyBERTConfig is a minimal BERT config for CPU tests: 1 block, 2 heads,
// nEmbd=16 (headDim 8), sequence length 8, batch 2, masking 15%. Small enough
// that a CPU forward+backward+Adam step fits the CI budget while still
// exercising the whole encoder stack and the masked-CE objective.
func tinyBERTConfig(baseVocab int) bertConfig {
	return bertConfig{
		Vocab:     baseVocab + 1,
		BaseVocab: baseVocab,
		NLayer:    1,
		NHead:     2,
		NEmbd:     16,
		BlockSize: 8,
		Batch:     2,
		MaskProb:  bertMaskProb,
		LR:        bertAdamLR,
		Steps:     1,
	}
}

func newBERTModel(a *uop.Arena, cfg bertConfig) *nn.BERT {
	return nn.NewBERT(a, cfg.Vocab, cfg.NLayer, cfg.NHead, cfg.NEmbd, cfg.BlockSize)
}

// ── Build: registry pre-flight ───────────────────────────────────────────────
//
// buildBERT constructs a BERT forward graph with a fixed sequence-shaped input
// so the run / graph / kernels commands have something to inspect without
// running a full training loop. The corpus is loaded so the vocab matches what
// train would use (and to prime the asset cache).
func buildBERT(device string) (*BuildResult, error) {
	ds, err := loadDataset()
	if err != nil {
		return nil, err
	}
	cfg := defaultBERTConfig(ds.VocabSize())
	const B = int64(1)
	T := int64(cfg.BlockSize)

	a := uop.NewArena(1 << 20)
	m := newBERTModel(a, cfg)
	initBERTSmall(m, bertInitScale, rand.New(rand.NewSource(42)))

	for _, p := range m.Params() {
		p.Load(a)
	}

	// Seed the input with the first T corpus tokens (valid ids in [0, BaseVocab)),
	// then mask the middle position with the [MASK] sentinel so the pre-flight
	// graph exercises the sentinel embedding row too.
	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = ds.Data[int64(i)%int64(len(ds.Data))]
	}
	idxVals[T/2] = cfg.MaskID()
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

// trainBERT runs the char-level masked-LM training loop with Adam, then emits a
// masked-token reconstruction sample. The corpus and default config are
// resolved here; runBERT is the shared body the convergence smoke test and the
// CPU full-loop test drive with a tiny in-memory fixture.
func trainBERT(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	ds, err := loadDataset()
	if err != nil {
		return err
	}
	return runBERT(device, cfg, logFn, ds, defaultBERTConfig(ds.VocabSize()), 42)
}

// runBERT is the shared trainer used by the production entry point and the
// tests. seed controls both the parameter-init RNG and the per-step
// mask-sampling RNG so identical seeds produce identical trajectories.
func runBERT(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *charDataset,
	beCfg bertConfig,
	seed int64,
) error {
	batch := cfg.Batch
	if batch <= 0 {
		batch = beCfg.Batch
	}
	lr := cfg.LR
	// cmd_train.go's --lr default (0.05) is tuned for the SGD MLP/Conv examples
	// and explodes Adam on a transformer; swap to the config's Adam lr when we
	// see that exact sentinel (or a non-positive lr), respecting any other LR.
	if lr <= 0 || lr == cmdTrainSGDDefaultLR {
		lr = beCfg.LR
	}
	steps := cfg.Steps
	if steps <= 0 {
		steps = beCfg.Steps
	}

	a0 := uop.NewArena(1 << 14)
	m := newBERTModel(a0, beCfg)
	initBERTSmall(m, bertInitScale, rand.New(rand.NewSource(seed)))

	params := m.Params()
	opt := nn.NewAdam(params, lr)

	sampleRNG := rand.New(rand.NewSource(seed + 1))

	if len(ds.Data) < beCfg.BlockSize+1 {
		return fmt.Errorf("bert: corpus length %d < block_size+1 = %d", len(ds.Data), beCfg.BlockSize+1)
	}

	T := int64(beCfg.BlockSize)
	V := int64(beCfg.Vocab)

	// Held-out eval batch, sampled once. Every logged loss (the step-0 probe and
	// the periodic probes) is measured on these SAME masked positions, so the
	// logFn stream is a clean learning curve rather than a per-step-batch-noisy
	// one. The training steps below draw fresh batches from sampleRNG.
	var evIn, evTg []int32
	var evMk []bool
	var evN int
	if cfg.LogEvery > 0 {
		evIn, evTg, evMk, evN = sampleMLMBatch(rand.New(rand.NewSource(seed+101)), int(batch), int(T), ds, beCfg.MaskID(), beCfg.MaskProb)
		l0 := evalBERTLoss(m, params, evIn, evTg, evMk, evN, batch, T, V, device)
		logFn(0, l0)
	}

	for step := 1; step <= steps; step++ {
		inputs, targets, mask, numMasked := sampleMLMBatch(sampleRNG, int(batch), int(T), ds, beCfg.MaskID(), beCfg.MaskProb)
		if inputs == nil {
			return fmt.Errorf("bert: failed to sample batch (corpus too small)")
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}

		idx := tensor.NewLeaf(a, []int64{batch, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(inputs))

		// Masked one-hot: zero rows at non-masked positions (ignore_index trick).
		oh := tensor.NewLeaf(a, []int64{batch, T, V}, uop.Dtypes.Float32, device)
		oh.SetData(maskedOneHotBits(targets, mask, beCfg.Vocab))

		logits := m.Forward(idx)
		loss := maskedCrossEntropyLoss(logits, oh, batch, T, V, numMasked)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		// Realize each grad on its own (one Realize per leaf), exactly as llama /
		// nanogpt / vit do. Batching several same-shape grad outputs of one shared
		// backward graph into a single variadic Realize is not proven to attribute
		// each output buffer to the right leaf: BERT has an untied head and token
		// embedding both shaped [vocab, nEmbd] plus a scatter-add embedding grad,
		// and batching empirically left a PosEmb grad empty. A single-output
		// Realize has no attribution ambiguity, so per-grad realize is correct.
		for _, p := range params {
			gr, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(gr); err != nil {
				return fmt.Errorf("bert: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := evalBERTLoss(m, params, evIn, evTg, evMk, evN, batch, T, V, device)
			logFn(step, lp)
		}
	}

	// Post-train artifact: a masked-token reconstruction over one corpus window.
	sample, err := reconstructBERT(m, params, beCfg, ds, device)
	if err != nil {
		return fmt.Errorf("bert: reconstruction: %w", err)
	}
	emitBERTSample(cfg, sample)

	return nil
}

// emitBERTSample prints the masked-LM reconstruction via cfg.LogText if set,
// else stdout. Mirrors emitLlamaSample / emitSample.
func emitBERTSample(cfg TrainConfig, sample string) {
	line := "\nbert masked-LM reconstruction:\n" + sample + "\n"
	if cfg.LogText != nil {
		cfg.LogText(line)
		return
	}
	//nolint:errcheck // best-effort write
	fmt.Fprint(os.Stdout, line)
}

// ── Masked cross-entropy loss ─────────────────────────────────────────────────
//
// maskedCrossEntropyLoss is the nanoGPT crossEntropyLoss op-chain with two
// changes: (1) the host-built one-hot has all-zero rows at non-masked positions
// (the PyTorch ignore_index=-100 equivalent), so only masked positions
// contribute to the sum; (2) the divisor is 1/numMasked rather than 1/(B*T), so
// the loss is the mean NLL over the predicted (masked) positions only.
//
// Every op is a pre-existing primitive (Exp/Sum/Reshape/Log/Expand/Sub/Mul/
// ConstScalar); this is an example-level function, not a new op.
func maskedCrossEntropyLoss(logits, oneHot *tensor.Tensor, B, T, V int64, numMasked int) *tensor.Tensor {
	a := logits.Arena()
	device := logits.Device()
	dtype := logits.DType()

	expv := logits.Exp()                       // [B, T, V]
	sumV := expv.Sum([]int{2}, false)          // [B, T]
	sumKD := sumV.Reshape([]int64{B, T, 1})    // [B, T, 1]
	logSum := sumKD.Log()                      // [B, T, 1]
	logSumB := logSum.Expand([]int64{B, T, V}) // [B, T, V] (broadcast)
	logSoftmax := logits.Sub(logSumB)          // [B, T, V]
	nllPerEl := oneHot.Mul(logSoftmax)         // [B, T, V] (nonzero only at masked positions' target col)
	totalNLL := nllPerEl.Sum(nil, false)       // scalar
	scale := tensor.ConstScalar(a, -1.0/float64(numMasked), dtype, device)
	return totalNLL.Mul(scale)
}

// evalBERTLoss recomputes the masked-LM loss for one batch in a fresh arena,
// independently of the training forward / backward graph (for honest logging).
func evalBERTLoss(
	m *nn.BERT,
	params []*nn.Parameter,
	inputs, targets []int32,
	mask []bool,
	numMasked int,
	B, T, V int64,
	device string,
) float32 {
	a := uop.NewArena(1 << 20)
	for _, p := range params {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(inputs))
	oh := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, device)
	oh.SetData(maskedOneHotBits(targets, mask, int(V)))
	loss := maskedCrossEntropyLoss(m.Forward(idx), oh, B, T, V, numMasked)
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

// ── Reconstruction (post-train artifact) ──────────────────────────────────────
//
// reconstructBERT masks the middle position of a single corpus window, runs the
// trained encoder, and reports the model's top prediction for that masked
// position against the true character. It is the encoder analogue of the
// decoder examples' text generation: a forward-only demonstration that the MLM
// objective taught the model to fill in masked tokens from bidirectional
// context. The argmax ranges over real corpus ids only ([0, BaseVocab)), so the
// [MASK] sentinel column is never predicted.
func reconstructBERT(
	m *nn.BERT,
	params []*nn.Parameter,
	beCfg bertConfig,
	ds *charDataset,
	device string,
) (string, error) {
	T := int64(beCfg.BlockSize)
	if int64(len(ds.Data)) < T {
		return "", fmt.Errorf("corpus length %d < block_size %d", len(ds.Data), T)
	}

	window := make([]int32, T)
	copy(window, ds.Data[:T])
	maskPos := T / 2
	trueID := window[maskPos]

	inputs := make([]int32, T)
	copy(inputs, window)
	inputs[maskPos] = beCfg.MaskID()

	a := uop.NewArena(1 << 20)
	for _, p := range params {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(inputs))

	logits := m.Forward(idx)
	if err := tensor.Realize(logits); err != nil {
		return "", err
	}
	data := logits.Data()

	// Logits for the masked position live at row maskPos of the [1, T, V] buffer.
	base := int(maskPos) * beCfg.Vocab
	var bestID int32
	bestVal := float32(math.Inf(-1))
	for j := 0; j < beCfg.BaseVocab; j++ { // real tokens only; skip the [MASK] column
		v := data[base+j]
		if v > bestVal {
			bestVal = v
			bestID = int32(j)
		}
	}

	predChar := ds.Decode([]int32{bestID})
	trueChar := ds.Decode([]int32{trueID})
	ok := "MISS"
	if bestID == trueID {
		ok = "HIT"
	}
	return fmt.Sprintf("  position %d: predicted %q  (true %q)  [%s]",
		maskPos, predChar, trueChar, ok), nil
}

// ── Host-side mask sampling ────────────────────────────────────────────────────
//
// sampleMLMBatch draws a char window per BERT's masking recipe. It samples a
// contiguous window per sequence (reusing charDataset.SampleBatch and ignoring
// the next-token shift), then masks ~MaskProb of positions: each masked input
// token is replaced by the [MASK] sentinel id, and its original id is recorded
// as the target. Non-masked positions get mask=false (their one-hot row is
// zeroed, so they are ignored by the loss). It guarantees numMasked >= 1 by
// force-masking one random position when the Bernoulli draw selects none.
//
// This is the simplified "mask-only" variant of BERT's 80/10/10 recipe (80% ->
// [MASK], 10% -> random token, 10% -> unchanged); mask-only is an accepted demo
// simplification and keeps the host prep trivial. Returns nil slices when the
// corpus is too small for a single window.
func sampleMLMBatch(
	rng *rand.Rand,
	batch, T int,
	ds *charDataset,
	maskID int32,
	p float64,
) (inputs, targets []int32, mask []bool, numMasked int) {
	xs, _ := ds.SampleBatch(rng, batch, T)
	if xs == nil {
		return nil, nil, nil, 0
	}
	n := len(xs)
	inputs = make([]int32, n)
	targets = make([]int32, n)
	mask = make([]bool, n)
	copy(inputs, xs)
	copy(targets, xs)

	for i := 0; i < n; i++ {
		if rng.Float64() < p {
			inputs[i] = maskID
			mask[i] = true
			numMasked++
		}
	}
	if numMasked == 0 {
		// Force-mask one random position so the loss divisor is never zero.
		j := rng.Intn(n)
		inputs[j] = maskID
		mask[j] = true
		numMasked = 1
	}
	return inputs, targets, mask, numMasked
}

// maskedOneHotBits returns a [n, vocab] flat row-major float32 buffer where row
// i is the one-hot of targets[i] when mask[i] is set, and all-zero otherwise.
// The zero rows implement the ignore_index trick: non-masked positions
// contribute nothing to the masked cross-entropy. Mirrors oneHotBits gated by
// the mask.
func maskedOneHotBits(targets []int32, mask []bool, vocab int) []float32 {
	out := make([]float32, len(targets)*vocab)
	for i, id := range targets {
		if !mask[i] {
			continue // zero row at non-masked positions (ignored by the loss)
		}
		if id < 0 || int(id) >= vocab {
			continue
		}
		out[i*vocab+int(id)] = 1.0
	}
	return out
}

// ── Initialisation ───────────────────────────────────────────────────────────
//
// initBERTSmall seeds every learnable parameter with small normal samples.
// LayerNorm Weight stays near 1.0, Bias near 0.0; all projection / embedding /
// position matrices are sampled from N(0, scale^2). Biases use half the scale,
// matching the GPT/ViT initialisers in this package.
func initBERTSmall(m *nn.BERT, scale float32, rng *rand.Rand) {
	fillNormal := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rng.NormFloat64()) * scale
		}
	}
	fillBias := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rng.NormFloat64()) * scale * 0.5
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

	fillNormal(m.Wte.Weight.Value)
	fillNormal(m.PosEmb.Value)

	for _, b := range m.Blocks {
		fillLN(b.LN1.Weight.Value, b.LN1.Bias.Value)
		fillNormal(b.Attn.QKV.Weight.Value)
		fillNormal(b.Attn.Proj.Weight.Value)
		if b.Attn.QKV.Bias != nil {
			fillBias(b.Attn.QKV.Bias.Value)
		}
		if b.Attn.Proj.Bias != nil {
			fillBias(b.Attn.Proj.Bias.Value)
		}
		fillLN(b.LN2.Weight.Value, b.LN2.Bias.Value)
		fillNormal(b.MLP.FC1.Weight.Value)
		fillNormal(b.MLP.FC2.Weight.Value)
		if b.MLP.FC1.Bias != nil {
			fillBias(b.MLP.FC1.Bias.Value)
		}
		if b.MLP.FC2.Bias != nil {
			fillBias(b.MLP.FC2.Bias.Value)
		}
	}

	fillLN(m.LNf.Weight.Value, m.LNf.Bias.Value)
	fillNormal(m.Head.Weight.Value)
	if m.Head.Bias != nil {
		fillBias(m.Head.Bias.Value)
	}
}
