package examples

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// TestGPT2FinetuneSmokeCPU exercises the full GPT-2 fine-tune pipeline (tied
// model, stable cross-entropy, global-norm clip, Adam step, sampling) on the
// pure-Go CPU backend with a tiny tied-head model. The CPU interpreter is too
// slow for a multi-step convergence assertion (~seconds per transformer step),
// so this runs a couple of real steps and asserts the loop completes with finite
// losses. Gradient correctness is proven separately by the FD checks
// (TestGPT2StableCrossEntropyGradCheck here, TestGPTTiedHeadGradCheck in nn);
// end-to-end convergence is demonstrated by the real fine-tune run.
func TestGPT2FinetuneSmokeCPU(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const (
		vocab     = 16
		nLayer    = 1
		nHead     = 2
		nEmbd     = 8
		blockSize = 8
	)
	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a0, vocab, nLayer, nHead, nEmbd, blockSize)
	if g.LMHead.Weight != g.Wte.Weight {
		t.Fatalf("expected tied head")
	}
	initGPTSmall(g, 0.02, rand.New(rand.NewSource(1)))

	corpus := make([]int32, vocab*8)
	for i := range corpus {
		corpus[i] = int32(i % vocab)
	}

	gptCfg := gpt2FinetuneConfig{
		Vocab: vocab, NLayer: nLayer, NHead: nHead, NEmbd: nEmbd,
		BlockSize: blockSize, SeqLen: 4,
		SampleN: 2, // exercise the greedy-sample path too
	}
	cfg := TrainConfig{Steps: 2, LR: 5e-3, LogEvery: 1, Batch: 2}

	var losses []float32
	logFn := func(step int, loss float32) { losses = append(losses, loss) }
	encode := func(s string) []int32 { return []int32{0, 1, 2} }
	decode := func(ids []int32) string { return "" }

	if err := runGPT2Finetune("cpu", cfg, logFn, g, corpus, gptCfg, encode, decode, 7); err != nil {
		t.Fatalf("runGPT2Finetune: %v", err)
	}
	if len(losses) < 2 {
		t.Fatalf("expected loss logs, got %d", len(losses))
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("loss[%d] not finite: %v", i, l)
		}
	}
	t.Logf("gpt2 fine-tune CPU smoke: losses=%v", losses)
}

// TestClipGradsByGlobalNorm covers the three branches of the global-norm clip:
// disabled (maxNorm<=0), below threshold (no-op), and above threshold (rescale).
func TestClipGradsByGlobalNorm(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	a := uop.NewArena(1 << 12)
	lin := nn.NewLinear(a, 2, 2, false, uop.Dtypes.Float32, "cpu")
	p := lin.Weight
	p.Load(a)

	newGrad := func(vals []float32) map[*tensor.Tensor]*tensor.Tensor {
		g := tensor.NewLeaf(a, []int64{2, 2}, uop.Dtypes.Float32, "cpu")
		g.SetData(append([]float32(nil), vals...))
		return map[*tensor.Tensor]*tensor.Tensor{p.T: g}
	}

	// Above threshold: grad norm = 10, maxNorm 1 -> scale 0.1.
	grads := newGrad([]float32{10, 0, 0, 0})
	clipGradsByGlobalNorm(grads, []*nn.Parameter{p}, 1.0)
	if got := grads[p.T].Data()[0]; math.Abs(float64(got-1.0)) > 1e-5 {
		t.Errorf("clip above threshold: got %.5f, want 1.0", got)
	}

	// Below threshold: grad norm = 0.5 < maxNorm 1 -> unchanged.
	grads = newGrad([]float32{0.5, 0, 0, 0})
	clipGradsByGlobalNorm(grads, []*nn.Parameter{p}, 1.0)
	if got := grads[p.T].Data()[0]; math.Abs(float64(got-0.5)) > 1e-6 {
		t.Errorf("below threshold should be unchanged: got %.5f", got)
	}

	// Disabled: maxNorm <= 0 -> no-op even with a large grad.
	grads = newGrad([]float32{100, 0, 0, 0})
	clipGradsByGlobalNorm(grads, []*nn.Parameter{p}, 0)
	if got := grads[p.T].Data()[0]; math.Abs(float64(got-100.0)) > 1e-4 {
		t.Errorf("disabled clip should be no-op: got %.5f", got)
	}
}

// TestSampleTokenBatchTooShort covers the corpus-too-short guard.
func TestSampleTokenBatchTooShort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	if xs, ys := sampleTokenBatch(rng, []int32{1, 2, 3}, 2, 8); xs != nil || ys != nil {
		t.Errorf("expected nil for corpus shorter than T+1, got xs=%v ys=%v", xs, ys)
	}
	xs, ys := sampleTokenBatch(rng, []int32{0, 1, 2, 3, 4, 5}, 2, 4)
	if len(xs) != 8 || len(ys) != 8 {
		t.Fatalf("expected 8-element xs/ys, got %d/%d", len(xs), len(ys))
	}
}

// TestRunGPT2FinetuneCorpusTooShort covers the runGPT2Finetune guard that
// returns an error when the corpus cannot yield a length-T window.
func TestRunGPT2FinetuneCorpusTooShort(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	a0 := uop.NewArena(1 << 12)
	g := nn.NewGPTWithTiedHead(a0, 8, 1, 2, 8, 8)
	initGPTSmall(g, 0.02, rand.New(rand.NewSource(1)))
	gptCfg := gpt2FinetuneConfig{Vocab: 8, NLayer: 1, NHead: 2, NEmbd: 8, BlockSize: 8, SeqLen: 8, SampleN: 0}
	// LogEvery=0 skips the step-0 probe; step 1 then samples nil and errors.
	cfg := TrainConfig{Steps: 1, LR: 5e-3, LogEvery: 0, Batch: 2}
	if err := runGPT2Finetune("cpu", cfg, func(int, float32) {}, g, []int32{0, 1, 2}, gptCfg, nil, nil, 1); err == nil {
		t.Error("expected error for too-short corpus")
	}
}

// TestGPT2FinetuneJITGPU exercises the JIT-wrapped training loop on the GPU
// backend with a small tied model: step 1 captures the schedule, steps 2-3
// replay it. Asserts the loop completes with finite losses (JIT replay zeroes
// the AST, so this would crash on the CPU interpreter — the trainer correctly
// uses JIT only on non-cpu devices). This is the small-scale guard for the path
// the real GPT-2 fine-tune drives.
func TestGPT2FinetuneJITGPU(t *testing.T) {
	requireGPUTest(t)

	const (
		vocab     = 24
		nLayer    = 1
		nHead     = 2
		nEmbd     = 16
		blockSize = 16
	)
	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a0, vocab, nLayer, nHead, nEmbd, blockSize)
	initGPTSmall(g, 0.02, rand.New(rand.NewSource(1)))
	corpus := make([]int32, vocab*8)
	for i := range corpus {
		corpus[i] = int32(i % vocab)
	}
	gptCfg := gpt2FinetuneConfig{
		Vocab: vocab, NLayer: nLayer, NHead: nHead, NEmbd: nEmbd,
		BlockSize: blockSize, SeqLen: 8, SampleN: 0,
	}
	cfg := TrainConfig{Steps: 3, LR: 5e-3, LogEvery: 1, Batch: 2}

	var losses []float32
	logFn := func(step int, loss float32) { losses = append(losses, loss) }
	if err := runGPT2Finetune("webgpu", cfg, logFn, g, corpus, gptCfg, nil, nil, 7); err != nil {
		t.Fatalf("runGPT2Finetune (GPU/JIT): %v", err)
	}
	if len(losses) < 3 {
		t.Fatalf("expected 3 loss logs (capture + 2 replays), got %d", len(losses))
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("loss[%d] not finite under JIT: %v", i, l)
		}
	}
	t.Logf("GPU/JIT train loop: losses=%v", losses)
}

// TestGPT2StableCrossEntropyNoOverflow confirms the stable loss stays finite on
// GPT-2-magnitude logits where the bare-exp loss (nanoGPT's) would overflow f32.
func TestGPT2StableCrossEntropyNoOverflow(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const B, T, V = int64(1), int64(2), int64(8)
	a := uop.NewArena(1 << 16)
	logits := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, "cpu")
	ld := make([]float32, int(B*T*V))
	for i := range ld {
		ld[i] = 120.0 // exp(120) overflows f32; stable CE must subtract the max first
	}
	ld[0] = 130.0
	logits.SetData(ld)
	oh := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, "cpu")
	ohd := make([]float32, int(B*T*V))
	ohd[0] = 1.0
	ohd[int(V)] = 1.0
	oh.SetData(ohd)

	loss := gpt2StableCrossEntropy(logits, oh, B, T, V)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("realize: %v", err)
	}
	got := loss.Data()[0]
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
		t.Fatalf("stable cross-entropy overflowed on large logits: %v", got)
	}
	t.Logf("stable CE on logits~120: loss=%.4f (finite)", got)
}

// TestGPT2StableCrossEntropyGradCheck verifies the stable cross-entropy gradient
// w.r.t. the logits matches central finite differences (the analytic gradient is
// softmax - one_hot, scaled). This is the load-bearing correctness proof for the
// loss; combined with correct tied-weight gradients, it makes fine-tuning sound.
func TestGPT2StableCrossEntropyGradCheck(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const B, T, V = int64(2), int64(3), int64(5)
	rng := rand.New(rand.NewSource(3))
	ld := make([]float32, int(B*T*V))
	for i := range ld {
		ld[i] = float32(rng.NormFloat64()) * 2.0 // moderate spread
	}
	ys := make([]int32, int(B*T))
	for i := range ys {
		ys[i] = int32(rng.Intn(int(V)))
	}
	ohd := oneHotBits(ys, int(V))

	build := func(data []float32) (*tensor.Tensor, *tensor.Tensor) {
		a := uop.NewArena(1 << 16)
		logits := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, "cpu")
		logits.SetData(data)
		oh := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, "cpu")
		oh.SetData(append([]float32(nil), ohd...))
		return logits, gpt2StableCrossEntropy(logits, oh, B, T, V)
	}

	lt, loss := build(ld)
	g := tensor.Backward(loss, []*tensor.Tensor{lt})[lt]
	if g == nil {
		t.Fatal("no gradient for logits")
	}
	if err := tensor.Realize(g); err != nil {
		t.Fatalf("realize grad: %v", err)
	}
	analytic := append([]float32(nil), g.Data()...)

	lossVal := func(data []float32) float32 {
		_, l := build(data)
		if err := tensor.Realize(l); err != nil {
			t.Fatalf("realize loss: %v", err)
		}
		return l.Data()[0]
	}
	const h = float32(1e-3)
	maxRel := 0.0
	for i := range ld {
		up := append([]float32(nil), ld...)
		up[i] += h
		dn := append([]float32(nil), ld...)
		dn[i] -= h
		fd := (lossVal(up) - lossVal(dn)) / (2 * h)
		diff := math.Abs(float64(analytic[i] - fd))
		// Scale floor 1.0 (as in the Block/GPT FD tests): loss-gradient entries
		// are < 1, and near-zero entries would otherwise inflate FD-truncation
		// noise into a large relative error. This makes the check effectively
		// absolute for sub-1 gradients.
		scale := math.Max(math.Max(math.Abs(float64(analytic[i])), math.Abs(float64(fd))), 1.0)
		if rel := diff / scale; rel > maxRel {
			maxRel = rel
		}
	}
	if maxRel > 1e-2 {
		t.Errorf("stable cross-entropy gradient drifts from FD: maxRel=%.3e", maxRel)
	}
	t.Logf("stable CE gradient vs FD: maxRel=%.3e", maxRel)
}
