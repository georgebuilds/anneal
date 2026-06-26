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

// MoE is the sparse-capacity companion to nanoGPT: a decoder-only char-level
// transformer whose per-block feed-forward network is replaced by a
// Mixture-of-Experts layer. A small router (a Linear over the embedding) emits a
// softmax distribution over E expert FFNs; every expert runs on every token and
// the outputs are combined by the router gates (dense / soft routing). A
// load-balance auxiliary loss nudges the router toward using all experts so it
// does not collapse onto a single one.
//
// Everything is assembled at the example level out of existing nn modules
// (Embedding, LayerNorm, CausalSelfAttention, Linear, MLP, Adam). nn.Block
// hard-wires the dense MLP and nn.NewGPT hard-wires nn.Block, so swapping the
// block FFN for an MoE layer is done here rather than in tensor/nn. The diff
// versus nanoGPT is exactly "block FFN -> moeFFN", which is the pedagogical
// payload. The rest (token + position embedding, causal self-attention blocks,
// final norm, LM head) mirrors nn.GPT.Forward, and the tinyshakespeare char
// dataset / cross-entropy loss are shared with the nanoGPT example.
//
// Routing is pure-soft (all experts, full softmax weights): no sparse dispatch,
// no top-k mask. Sparse top-k dispatch would need a scatter-add back-combine
// (host-side sort, no f32 atomics) and ragged per-expert shapes; the dense form
// is the MoE idea in one line (router gates x experts, summed) and stays fully
// CPU-trainable so the train loop is covered by host tests.

func init() {
	Register(&Example{
		Name:    "moe",
		Summary: "Mixture-of-Experts LM (router + expert FFNs, soft routing + load-balance loss); trains on tinyshakespeare",
		Build:   buildMoE,
		Train:   trainMoE,
	})
}

// ── MoE default config ────────────────────────────────────────────────────────
//
// A "tiny but transformer-shaped" config that runs end to end in reasonable wall
// time while exercising the MoE machinery: pre-LayerNorm blocks with causal
// self-attention, a router over E experts, E small MLP experts, the dense gated
// combine, and the load-balance aux loss. vocab is derived from the corpus
// (tinyshakespeare ~65 chars). Each expert is a 4x-expansion FFN, matching the
// GPT-2 MLP ratio (ExpertHidden = 4*NEmbd).
const (
	moeNLayer    = 2
	moeNHead     = 2
	moeNEmbd     = 64
	moeBlockSize = 32
	moeBatch     = int64(4)
	moeNExperts  = 4
	moeAuxAlpha  = float32(0.01)
	moeAdamLR    = float32(3e-4)
	moeInitScale = float32(0.02)
	moeSteps     = 100
	// moeSamplePrompt seeds generation after training; "ROMEO:" is in the
	// tinyshakespeare corpus.
	moeSamplePrompt = "ROMEO:"
	moeSampleTokens = 100
)

// moeConfig groups the knobs the build / train / sample paths share. It exists so
// the convergence smoke test and the CPU coverage tests can plug a small config
// without re-deriving every shape constant.
//
// Batch, LR, and Steps are the model's recommended training defaults; at runtime
// the TrainConfig passed to Train (sourced from the CLI flags) takes precedence
// when set, and Batch/LR fall back to these. Steps is always taken from the
// TrainConfig so the zero-steps build/coverage paths stay honest.
type moeConfig struct {
	Vocab        int
	NLayer       int
	NHead        int
	NEmbd        int
	BlockSize    int
	Batch        int64
	NExperts     int     // E: number of expert FFNs
	ExpertHidden int     // hidden width of each expert MLP (4*NEmbd by convention)
	AuxAlpha     float32 // weight on the load-balance auxiliary loss
	LR           float32 // recommended Adam learning rate
	Steps        int     // recommended number of training steps
	// SampleTokens is how many tokens to generate after the final step. Zero
	// falls back to moeSampleTokens.
	SampleTokens int
}

// defaultMoEConfig is the GPU-scale default used by `anneal train moe`.
func defaultMoEConfig(vocab int) moeConfig {
	return moeConfig{
		Vocab:        vocab,
		NLayer:       moeNLayer,
		NHead:        moeNHead,
		NEmbd:        moeNEmbd,
		BlockSize:    moeBlockSize,
		Batch:        moeBatch,
		NExperts:     moeNExperts,
		ExpertHidden: 4 * moeNEmbd,
		AuxAlpha:     moeAuxAlpha,
		LR:           moeAdamLR,
		Steps:        moeSteps,
		SampleTokens: moeSampleTokens,
	}
}

// tinyMoEConfig is the minimal CPU-trainable config: 1 block, 2 heads, nEmbd=16,
// block_size=4, batch=2, 4 experts each with hidden 4*16=64. It keeps a full
// forward + backward + Adam step inside the CI budget while exercising every new
// line (router, all experts, gated combine, aux loss).
func tinyMoEConfig(vocab int) moeConfig {
	return moeConfig{
		Vocab:        vocab,
		NLayer:       1,
		NHead:        2,
		NEmbd:        16,
		BlockSize:    4,
		Batch:        2,
		NExperts:     4,
		ExpertHidden: 4 * 16,
		AuxAlpha:     0.01,
		LR:           0, // 0 -> the canonical Adam lr is selected in runMoE
		Steps:        1,
		SampleTokens: 2,
	}
}

// ── MoE layer ─────────────────────────────────────────────────────────────────

// moeFFN is the Mixture-of-Experts feed-forward layer that replaces the dense
// MLP in a transformer block. Router is a Linear(NEmbd, E) whose softmax over
// the last axis produces per-token gate weights; Experts are E independent MLP
// FFNs (each [..., NEmbd] -> [..., NEmbd]).
type moeFFN struct {
	Router   *nn.Linear // gate logits: NEmbd -> E
	Experts  []*nn.MLP  // E expert FFNs
	NExperts int
}

// newMoEFFN builds a router plus E experts. Each expert is an nn.MLP constructed
// directly from two Linear layers so the hidden width honours cfg.ExpertHidden
// (nn.NewMLP fixes hidden at 4*nEmbd); MLP.Forward still applies the same
// FC1 -> exact-GELU -> FC2 path as the dense transformer FFN.
func newMoEFFN(a *uop.Arena, cfg moeConfig, dtype *uop.DType, device string) *moeFFN {
	experts := make([]*nn.MLP, cfg.NExperts)
	for e := range experts {
		experts[e] = &nn.MLP{
			FC1: nn.NewLinear(a, int64(cfg.NEmbd), int64(cfg.ExpertHidden), true, dtype, device),
			FC2: nn.NewLinear(a, int64(cfg.ExpertHidden), int64(cfg.NEmbd), true, dtype, device),
		}
	}
	return &moeFFN{
		Router:   nn.NewLinear(a, int64(cfg.NEmbd), int64(cfg.NExperts), true, dtype, device),
		Experts:  experts,
		NExperts: cfg.NExperts,
	}
}

// gates computes the router softmax over the E experts (last axis) for input
// x [B, T, NEmbd], returning [B, T, E]. The softmax is composed inline as
// exp(logits) / sum_e exp(logits) exactly like the nanoGPT cross-entropy and the
// attention softmax: anneal has no native Softmax op. We deliberately omit the
// reduce-max subtraction (its OpMax backward is the unstable one); small router
// init keeps the logits in range, so exp does not overflow at E=4.
func (f *moeFFN) gates(x *tensor.Tensor) *tensor.Tensor {
	logits := f.Router.Forward(x) // [B, T, E]
	sh := logits.Shape()
	B, T := sh[0], sh[1]
	expv := logits.Exp()                                       // [B, T, E]
	den := expv.Sum([]int{2}, false).Reshape([]int64{B, T, 1}) // [B, T, 1]
	return expv.Div(den)                                       // [B, T, E] (broadcasts)
}

// Forward runs dense / soft Mixture-of-Experts routing on x [B, T, NEmbd].
//
//	gates = softmax_E( Router(x) )                 # [B, T, E]
//	out   = sum_e gates[..., e] * Expert_e(x)      # [B, T, NEmbd]
//
// Every expert runs on every token; the per-expert gate slice (a Shrink view of
// the gate buffer) broadcast-multiplies that expert's output, and the products
// are summed by an Add-tree (no Stack / Concat, which do not exist). It also
// returns the scalar load-balance auxiliary loss for this layer:
//
//	aux = E * sum_e mean_{B,T}( gates[..., e] )^2
//
// which is minimised (= 1) when the average gate mass is uniform across experts
// and grows toward E when the router collapses onto one expert.
//
// 8-buffer note: at E=4 the fused weighted-sum epilogue reads the E expert
// outputs plus the router/softmax buffers and stays within the WebGPU
// 8-storage-buffer per-kernel cap (verified on Metal). If E grows past the cap,
// .Contiguous() each expert output before the sum to force a kernel break.
func (f *moeFFN) Forward(x *tensor.Tensor) (out, aux *tensor.Tensor) {
	sh := x.Shape()
	B, T := sh[0], sh[1]
	a := x.Arena()

	g := f.gates(x) // [B, T, E]

	// Dense gated combine: out = sum_e Expert_e(x) * gate_e.
	for e := 0; e < f.NExperts; e++ {
		ge := g.Shrink([][2]int64{{0, B}, {0, T}, {int64(e), int64(e) + 1}}) // [B, T, 1]
		we := f.Experts[e].Forward(x).Mul(ge)                                // broadcast over NEmbd
		if out == nil {
			out = we
		} else {
			out = out.Add(we)
		}
	}

	// Load-balance aux loss: E * sum_e mean_{B,T}(gate_e)^2. B and T are concrete
	// here, so the Mean over axes [0,1] is a concrete reduce (no symbolic-axis
	// panic). Only Mean / Mul / Sum, all differentiable into the router.
	meanGate := g.Mean([]int{0, 1}, false)                                      // [E]
	scaleE := tensor.ConstScalar(a, float64(f.NExperts), x.DType(), x.Device()) // E
	aux = meanGate.Mul(meanGate).Sum(nil, false).Mul(scaleE)                    // scalar

	return out, aux
}

// Params returns the router parameters followed by each expert's parameters, in
// deterministic order (mirrors nn.Block.Params).
func (f *moeFFN) Params() []*nn.Parameter {
	ps := make([]*nn.Parameter, 0, len(f.Router.Params())+len(f.Experts)*4)
	ps = append(ps, f.Router.Params()...)
	for _, ex := range f.Experts {
		ps = append(ps, ex.Params()...)
	}
	return ps
}

// ── MoE transformer block ─────────────────────────────────────────────────────

// moeBlock is a pre-LayerNorm transformer block whose FFN is a moeFFN. It is the
// nn.Block pattern (x = x + Attn(LN1(x)); x = x + FFN(LN2(x))) with the dense MLP
// swapped for the mixture, and it threads the per-layer aux loss out.
type moeBlock struct {
	LN1  *nn.LayerNorm
	Attn *nn.CausalSelfAttention
	LN2  *nn.LayerNorm
	FFN  *moeFFN
}

func newMoEBlock(a *uop.Arena, cfg moeConfig, dtype *uop.DType, device string) *moeBlock {
	const lnEps = float32(1e-5)
	return &moeBlock{
		LN1:  nn.NewLayerNorm(a, int64(cfg.NEmbd), lnEps),
		Attn: nn.NewCausalSelfAttention(a, cfg.NEmbd, cfg.NHead, cfg.BlockSize),
		LN2:  nn.NewLayerNorm(a, int64(cfg.NEmbd), lnEps),
		FFN:  newMoEFFN(a, cfg, dtype, device),
	}
}

// Forward computes the two-residual block and returns the block aux loss.
func (b *moeBlock) Forward(x *tensor.Tensor) (out, aux *tensor.Tensor) {
	h := x.Add(b.Attn.Forward(b.LN1.Forward(x)))
	ffnOut, aux := b.FFN.Forward(b.LN2.Forward(h))
	return h.Add(ffnOut), aux
}

// Params returns LN1, Attn, LN2, FFN parameters in deterministic order.
func (b *moeBlock) Params() []*nn.Parameter {
	ps := make([]*nn.Parameter, 0, 12)
	ps = append(ps, b.LN1.Params()...)
	ps = append(ps, b.Attn.Params()...)
	ps = append(ps, b.LN2.Params()...)
	ps = append(ps, b.FFN.Params()...)
	return ps
}

// ── MoE-GPT container ─────────────────────────────────────────────────────────

// moeGPT is the full decoder-only stack: token + learned-position embedding, N
// moeBlocks, a final LayerNorm, and an LM head. It mirrors nn.GPT.Forward except
// each block's FFN is an MoE layer and Forward returns the summed aux loss across
// all blocks alongside the logits.
type moeGPT struct {
	Wte    *nn.Embedding // token embedding   [vocab, nEmbd]
	Wpe    *nn.Embedding // position embedding [blockSize, nEmbd]
	Blocks []*moeBlock
	LNf    *nn.LayerNorm
	LMHead *nn.Linear

	NEmbd     int
	BlockSize int
	Vocab     int
}

func newMoEModel(a *uop.Arena, cfg moeConfig) *moeGPT {
	const lnEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"

	blocks := make([]*moeBlock, cfg.NLayer)
	for i := range blocks {
		blocks[i] = newMoEBlock(a, cfg, dtype, device)
	}
	return &moeGPT{
		Wte:       nn.NewEmbedding(a, int64(cfg.Vocab), int64(cfg.NEmbd), dtype, device),
		Wpe:       nn.NewEmbedding(a, int64(cfg.BlockSize), int64(cfg.NEmbd), dtype, device),
		Blocks:    blocks,
		LNf:       nn.NewLayerNorm(a, int64(cfg.NEmbd), lnEps),
		LMHead:    nn.NewLinear(a, int64(cfg.NEmbd), int64(cfg.Vocab), true, dtype, device),
		NEmbd:     cfg.NEmbd,
		BlockSize: cfg.BlockSize,
		Vocab:     cfg.Vocab,
	}
}

// Forward runs the full MoE-GPT stack on idx [B, T] Int32 (T <= BlockSize,
// values in [0, Vocab)). It returns logits [B, T, Vocab] (no softmax) and the
// scalar aux loss summed over every block. The embedding flatten trick and the
// host-built positional-index leaf are copied verbatim from nn.GPT.Forward (the
// gather backward wants a 1-D index; anneal has no on-graph arange).
func (g *moeGPT) Forward(idx *tensor.Tensor) (logits, aux *tensor.Tensor) {
	if idx.Rank() != 2 {
		panic(fmt.Sprintf("examples: moeGPT.Forward: idx must be rank 2 [B, T], got rank %d", idx.Rank()))
	}
	if idx.DType() != uop.Dtypes.Int32 {
		panic(fmt.Sprintf("examples: moeGPT.Forward: idx dtype must be Int32, got %s", idx.DType()))
	}
	idxShape := idx.Shape()
	B, T := idxShape[0], idxShape[1]
	if T > int64(g.BlockSize) {
		panic(fmt.Sprintf("examples: moeGPT.Forward: T=%d exceeds blockSize=%d", T, g.BlockSize))
	}

	a := idx.Arena()
	device := "webgpu"

	// Token embedding via a flat [B*T] index leaf (gather backward wants 1-D).
	idxData := idx.Data()
	if idxData == nil {
		panic("examples: moeGPT.Forward: idx.Data() is nil; call idx.SetData(...) before Forward")
	}
	if int64(len(idxData)) != B*T {
		panic(fmt.Sprintf("examples: moeGPT.Forward: idx.Data() length %d != B*T=%d", len(idxData), B*T))
	}
	idxFlat := tensor.NewLeaf(a, []int64{B * T}, uop.Dtypes.Int32, device)
	flatBits := make([]float32, B*T)
	copy(flatBits, idxData)
	idxFlat.SetData(flatBits)

	tokEmbFlat := g.Wte.Forward(idxFlat)                        // [B*T, nEmbd]
	tokEmb := tokEmbFlat.Reshape([]int64{B, T, int64(g.NEmbd)}) // [B, T, nEmbd]

	// Positional embedding: host-precomputed [0, 1, ..., T-1] Int32 leaf.
	posBits := make([]float32, T)
	for i := int64(0); i < T; i++ {
		posBits[i] = math.Float32frombits(uint32(int32(i)))
	}
	positions := tensor.NewLeaf(a, []int64{T}, uop.Dtypes.Int32, device)
	positions.SetData(posBits)

	posEmb := g.Wpe.Forward(positions)                              // [T, nEmbd]
	posEmbB := tensor.BroadcastToSints(posEmb, tokEmb.ShapeSints()) // [B, T, nEmbd]
	x := tokEmb.Add(posEmbB)

	// Transformer blocks, accumulating the per-block aux loss.
	var auxTotal *tensor.Tensor
	for _, blk := range g.Blocks {
		var blkAux *tensor.Tensor
		x, blkAux = blk.Forward(x)
		if auxTotal == nil {
			auxTotal = blkAux
		} else {
			auxTotal = auxTotal.Add(blkAux)
		}
	}

	x = g.LNf.Forward(x)
	logits = g.LMHead.Forward(x) // [B, T, vocab]
	return logits, auxTotal
}

// Params returns all trainable parameters in deterministic order:
// Wte, Wpe, each block's params, LNf, LMHead.
func (g *moeGPT) Params() []*nn.Parameter {
	ps := make([]*nn.Parameter, 0)
	ps = append(ps, g.Wte.Params()...)
	ps = append(ps, g.Wpe.Params()...)
	for _, blk := range g.Blocks {
		ps = append(ps, blk.Params()...)
	}
	ps = append(ps, g.LNf.Params()...)
	ps = append(ps, g.LMHead.Params()...)
	return ps
}

// moeTotalLoss is the training objective: mean cross-entropy plus the weighted
// load-balance aux loss. logits / oneHot are [B, T, V]; aux is the scalar summed
// block aux loss; alpha weights it (~0.01).
func moeTotalLoss(logits, aux, oneHot *tensor.Tensor, B, T, V int64, alpha float32) *tensor.Tensor {
	ce := crossEntropyLoss(logits, oneHot, B, T, V)
	a := ce.Arena()
	alphaT := tensor.ConstScalar(a, float64(alpha), ce.DType(), ce.Device())
	return ce.Add(aux.Mul(alphaT))
}

// ── Build: registry pre-flight ───────────────────────────────────────────────
//
// buildMoE constructs an MoE-GPT forward graph with a fixed prompt-shaped input
// so the run / graph / kernels commands can inspect it without running a training
// loop. The corpus is loaded so the vocab matches what train would use.
func buildMoE(device string) (*BuildResult, error) {
	ds, err := loadDataset()
	if err != nil {
		return nil, err
	}
	cfg := defaultMoEConfig(ds.VocabSize())
	const B = int64(1)
	T := int64(cfg.BlockSize)

	a := uop.NewArena(1 << 20)
	m := newMoEModel(a, cfg)
	initMoESmall(m, moeInitScale, rand.New(rand.NewSource(42)))

	for _, p := range m.Params() {
		p.Load(a)
	}

	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = ds.Data[int64(i)%int64(len(ds.Data))]
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(idxVals))

	logits, _ := m.Forward(idx)

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

// trainMoE runs the char-level MoE-GPT training loop with Adam, then emits a
// generated sample. The corpus and default config are resolved here; runMoE is
// the shared body the convergence smoke test and CPU tests drive with a tiny
// in-memory fixture.
func trainMoE(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	ds, err := loadDataset()
	if err != nil {
		return err
	}
	return runMoE(device, cfg, logFn, ds, defaultMoEConfig(ds.VocabSize()), 42)
}

// runMoE is the shared trainer used by the production entry point and the tests.
// seed controls both the parameter-init RNG and the per-step batch-sampling RNG
// so identical seeds produce identical trajectories.
//
// Per step: fresh arena + p.Load(a) for every parameter (the per-step IR
// invariant), a random corpus window batched to `batch`, one Forward producing
// logits + aux, the total loss, then a single batched Realize over all grads
// (Realize is stateless, so loss + every grad realize in ONE pass per step).
func runMoE(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *charDataset,
	moeCfg moeConfig,
	seed int64,
) error {
	batch := cfg.Batch
	if batch <= 0 {
		batch = moeCfg.Batch
	}
	// cmd_train.go's --lr default (0.05) is tuned for the SGD MLP/Conv examples
	// and explodes Adam on a transformer; swap to the config / canonical Adam lr
	// when we see that exact sentinel or no lr, respecting any other LR passed.
	lr := cfg.LR
	if lr <= 0 || lr == cmdTrainSGDDefaultLR {
		lr = moeCfg.LR
	}
	if lr <= 0 {
		lr = moeAdamLR
	}

	a0 := uop.NewArena(1 << 14)
	m := newMoEModel(a0, moeCfg)
	initMoESmall(m, moeInitScale, rand.New(rand.NewSource(seed)))

	params := m.Params()
	opt := nn.NewAdam(params, lr)

	if len(ds.Data) < moeCfg.BlockSize+1 {
		return fmt.Errorf("moe: corpus length %d < block_size+1 = %d", len(ds.Data), moeCfg.BlockSize+1)
	}

	sampleRNG := rand.New(rand.NewSource(seed + 1))

	// One fixed held-out batch drives the loss curve: step 0 and every periodic
	// log evaluate the SAME batch, independent of the training batches, so the
	// reported trajectory is honest rather than a post-step training-batch probe.
	var evalXs, evalYs []int32
	if cfg.LogEvery > 0 {
		evalXs, evalYs = ds.SampleBatch(rand.New(rand.NewSource(seed+101)), int(batch), moeCfg.BlockSize)
		l0 := evalMoELoss(m, params, evalXs, evalYs, batch, int64(moeCfg.BlockSize), moeCfg.Vocab, moeCfg.AuxAlpha, device)
		logFn(0, l0)
	}

	T := int64(moeCfg.BlockSize)
	V := int64(moeCfg.Vocab)

	for step := 1; step <= cfg.Steps; step++ {
		xs, ys := ds.SampleBatch(sampleRNG, int(batch), moeCfg.BlockSize)
		if xs == nil {
			return fmt.Errorf("moe: failed to sample batch (corpus too small)")
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}

		idx := tensor.NewLeaf(a, []int64{batch, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsBits(xs))

		oh := tensor.NewLeaf(a, []int64{batch, T, V}, uop.Dtypes.Float32, device)
		oh.SetData(oneHotBits(ys, moeCfg.Vocab))

		logits, aux := m.Forward(idx)
		loss := moeTotalLoss(logits, aux, oh, batch, T, V, moeCfg.AuxAlpha)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		// Realize each grad on its own. A single-output Realize attributes its one
		// output buffer unambiguously; batching several same-shape grad outputs of
		// one shared backward graph (the E symmetric experts here, plus the
		// scatter-add Wte grad) is not proven safe, so every token-embedding
		// transformer in this repo (llama, nanogpt, vit, bert) realizes grads one
		// at a time. MoE follows suit: correctness over the extra graph re-runs.
		for _, p := range params {
			gr, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(gr); err != nil {
				return fmt.Errorf("moe: realize grad at step %d: %w", step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := evalMoELoss(m, params, evalXs, evalYs, batch, T, moeCfg.Vocab, moeCfg.AuxAlpha, device)
			logFn(step, lp)
		}
	}

	nGen := moeCfg.SampleTokens
	if nGen <= 0 {
		nGen = moeSampleTokens
	}
	sample, err := generateMoE(m, params, moeCfg, ds, moeSamplePrompt, nGen, device)
	if err != nil {
		return fmt.Errorf("moe: generation: %w", err)
	}
	emitMoESample(cfg, sample)

	return nil
}

// emitMoESample prints the generated sample via cfg.LogText if set, else stdout.
func emitMoESample(cfg TrainConfig, sample string) {
	line := "\nsample (" + moeSamplePrompt + " ...):\n" + sample + "\n"
	if cfg.LogText != nil {
		cfg.LogText(line)
		return
	}
	//nolint:errcheck // best-effort write
	fmt.Fprint(os.Stdout, line)
}

// evalMoELoss recomputes the total loss (cross-entropy + alpha*aux) for one batch
// in a fresh arena, independent of the training forward / backward graph (used
// for honest loss logging).
func evalMoELoss(
	m *moeGPT,
	params []*nn.Parameter,
	xs, ys []int32,
	B, T int64,
	V int,
	alpha float32,
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
	logits, aux := m.Forward(idx)
	loss := moeTotalLoss(logits, aux, oh, B, T, int64(V), alpha)
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

// ── Generation ───────────────────────────────────────────────────────────────
//
// generateMoE produces nGen tokens by greedy (argmax) decode over a rolling
// block-size window, mirroring generateNanoGPT. The aux loss is discarded during
// generation. The model always sees an exactly block-size context; the prompt is
// left-padded with token 0 when shorter than block_size.
func generateMoE(
	m *moeGPT,
	params []*nn.Parameter,
	cfg moeConfig,
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

		logits, _ := m.Forward(idx)
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
// initMoESmall seeds every learnable parameter with small normal samples.
// LayerNorm Weight stays near 1.0, Bias near 0.0; all other matrices are sampled
// from N(0, scale^2). The router is seeded at the same small scale so the initial
// gates are near-uniform (1/E), which gives the load-balance loss a stable start.
func initMoESmall(m *moeGPT, scale float32, rng *rand.Rand) {
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
	fillBias := func(p *nn.Parameter, s float32) {
		if p == nil {
			return
		}
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * s
		}
	}

	fillNormal(m.Wte.Weight.Value)
	fillNormal(m.Wpe.Weight.Value)

	for _, b := range m.Blocks {
		fillLN(b.LN1.Weight.Value, b.LN1.Bias.Value)
		fillNormal(b.Attn.QKV.Weight.Value)
		fillNormal(b.Attn.Proj.Weight.Value)
		fillBias(b.Attn.QKV.Bias, scale*0.5)
		fillBias(b.Attn.Proj.Bias, scale*0.5)
		fillLN(b.LN2.Weight.Value, b.LN2.Bias.Value)

		fillNormal(b.FFN.Router.Weight.Value)
		fillBias(b.FFN.Router.Bias, scale*0.5)
		for _, ex := range b.FFN.Experts {
			fillNormal(ex.FC1.Weight.Value)
			fillNormal(ex.FC2.Weight.Value)
			fillBias(ex.FC1.Bias, scale*0.5)
			fillBias(ex.FC2.Bias, scale*0.5)
		}
	}

	fillLN(m.LNf.Weight.Value, m.LNf.Bias.Value)
	fillNormal(m.LMHead.Weight.Value)
	fillBias(m.LMHead.Bias, scale*0.5)
}
