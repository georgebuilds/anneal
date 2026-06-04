package nn_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// fillRandom seeds a parameter with N(0, 0.02) values, matching the canonical
// GPT-2 init scale used elsewhere in the test suite.
func fillRandom(p *nn.Parameter, rng *rand.Rand) {
	for i := range p.Value {
		p.Value[i] = float32(rng.NormFloat64()) * 0.02
	}
}

// TestCausalSelfAttention_ForwardKVStep_OracleAgainstForward verifies that a
// KV-cached single-token step produces the same attention output, within f32
// tolerance, as the full Forward at the corresponding position. This is the
// load-bearing correctness gate for the cache machinery: if the cache injects
// kNew and vNew at the wrong slot, or the lenMask leaks future positions in,
// the attn output at the last step diverges from Forward by orders of magnitude.
func TestCausalSelfAttention_ForwardKVStep_OracleAgainstForward(t *testing.T) {
	requireGPU(t)

	const (
		nEmbd     = 8
		nHead     = 2
		blockSize = 4
	)
	dHead := int64(nEmbd / nHead)

	rng := rand.New(rand.NewSource(7))

	// Build the attention module twice on the same arena so both runs share
	// the exact same weights via the Parameter.Value byte stream.
	a := uop.NewArena(1 << 20)
	attn := nn.NewCausalSelfAttention(a, nEmbd, nHead, blockSize)
	for _, p := range attn.Params() {
		fillRandom(p, rng)
	}
	for _, p := range attn.Params() {
		p.Load(a)
	}

	// Build the full input sequence and run Forward to get the oracle.
	xData := make([]float32, blockSize*nEmbd)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.1
	}
	xFull := tensor.NewLeaf(a, []int64{1, blockSize, nEmbd}, uop.Dtypes.Float32, "webgpu")
	xFull.SetData(xData)
	yFull := attn.Forward(xFull)
	if err := tensor.Realize(yFull); err != nil {
		t.Fatalf("Realize(yFull): %v", err)
	}
	yFullData := yFull.Data() // [1, blockSize, nEmbd]

	// Run the KV-step path one token at a time. After each step we read
	// kNew, vNew and store them in the cache, then advance Pos.
	cache := nn.NewKVCache(1, nHead, int(dHead), blockSize)

	var lastY []float32
	for step := 0; step < blockSize; step++ {
		stepArena := uop.NewArena(1 << 20)
		// Rebuild parameter leaves into the fresh arena.
		for _, p := range attn.Params() {
			p.Load(stepArena)
		}
		// One-token slice of x.
		xStepData := make([]float32, nEmbd)
		copy(xStepData, xData[step*nEmbd:(step+1)*nEmbd])
		xStep := tensor.NewLeaf(stepArena, []int64{1, 1, nEmbd}, uop.Dtypes.Float32, "webgpu")
		xStep.SetData(xStepData)

		kCache := cache.UploadKLeaf(stepArena, 0, uop.Dtypes.Float32, "webgpu")
		vCache := cache.UploadVLeaf(stepArena, 0, uop.Dtypes.Float32, "webgpu")
		posOH := cache.UploadPosOneHotLeaf(stepArena, uop.Dtypes.Float32, "webgpu")
		lenMask := cache.UploadLengthMaskLeaf(stepArena, uop.Dtypes.Float32, "webgpu")

		y, kNew, vNew := attn.ForwardKVStep(xStep, kCache, vCache, posOH, lenMask)
		if err := tensor.Realize(y); err != nil {
			t.Fatalf("step %d: Realize y: %v", step, err)
		}
		lastY = make([]float32, nEmbd)
		copy(lastY, y.Data())

		if err := tensor.Realize(kNew, vNew); err != nil {
			t.Fatalf("step %d: Realize kNew, vNew: %v", step, err)
		}
		cache.StoreLayerKV(0, kNew.Data(), vNew.Data())
		cache.Advance()
	}

	// Compare the last-step KV output to Forward's last-position output. The
	// tolerance is well above f32 epsilon: each step Realizes K_new in a
	// separate schedule whose matmul tile ordering differs from the inlined
	// view chain Forward uses, so per-step f32 sums drift by ~1e-4 and compound
	// over the prompt. Argmax greedy decoding is robust to this magnitude of
	// drift (verified in the gpt2 HF oracle suite).
	const tol = 1e-2
	wantY := yFullData[(blockSize-1)*nEmbd : blockSize*nEmbd]
	var maxDiff float32
	for i := 0; i < nEmbd; i++ {
		d := lastY[i] - wantY[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("lastY = %v", lastY)
	t.Logf("wantY = %v", wantY)
	if maxDiff > tol {
		t.Fatalf("ForwardKVStep last-token output max-abs-diff %g > tol %g", maxDiff, tol)
	}
	t.Logf("last-token max-abs-diff vs Forward: %g", maxDiff)
}

// TestGPT_ForwardKVStep_OracleAgainstForward repeats the oracle for the full
// GPT stack at a tiny scale that fits the 8-buffer cap and lavapipe.
func TestGPT_ForwardKVStep_OracleAgainstForward(t *testing.T) {
	requireGPU(t)

	const (
		vocab     = 16
		nLayer    = 2
		nHead     = 2
		nEmbd     = 8
		blockSize = 4
	)
	headDim := nEmbd / nHead

	rng := rand.New(rand.NewSource(31))

	a := uop.NewArena(1 << 21)
	g := nn.NewGPT(a, vocab, nLayer, nHead, nEmbd, blockSize)
	for _, p := range g.Params() {
		fillRandom(p, rng)
	}
	for _, p := range g.Params() {
		p.Load(a)
	}

	// Build a [1, blockSize] sequence of pseudo-random ids.
	ids := make([]int32, blockSize)
	for i := range ids {
		ids[i] = rng.Int31n(int32(vocab))
	}
	idxFull := tensor.NewLeaf(a, []int64{1, int64(blockSize)}, uop.Dtypes.Int32, "webgpu")
	idxBits := make([]float32, blockSize)
	for i, v := range ids {
		idxBits[i] = math.Float32frombits(uint32(v))
	}
	idxFull.SetData(idxBits)
	logitsFull := g.Forward(idxFull)
	if err := tensor.Realize(logitsFull); err != nil {
		t.Fatalf("Realize(logitsFull): %v", err)
	}
	logitsFullData := logitsFull.Data() // [1, blockSize, vocab]

	// KV path.
	cache := nn.NewKVCache(nLayer, nHead, headDim, blockSize)
	var lastLogits []float32
	for step := 0; step < blockSize; step++ {
		stepArena := uop.NewArena(1 << 21)
		for _, p := range g.Params() {
			p.Load(stepArena)
		}
		idxStep := tensor.NewLeaf(stepArena, []int64{1, 1}, uop.Dtypes.Int32, "webgpu")
		idxStep.SetData([]float32{math.Float32frombits(uint32(ids[step]))})

		lg, kNews, vNews := g.ForwardKVStep(idxStep, cache)
		if err := tensor.Realize(lg); err != nil {
			t.Fatalf("step %d: Realize logits: %v", step, err)
		}
		lastLogits = make([]float32, vocab)
		copy(lastLogits, lg.Data())
		kvOutputs := make([]*tensor.Tensor, 0, 2*nLayer)
		kvOutputs = append(kvOutputs, kNews...)
		kvOutputs = append(kvOutputs, vNews...)
		if err := tensor.Realize(kvOutputs...); err != nil {
			t.Fatalf("step %d: Realize kv outputs: %v", step, err)
		}
		for li := 0; li < nLayer; li++ {
			cache.StoreLayerKV(li, kNews[li].Data(), vNews[li].Data())
		}
		cache.Advance()
	}

	const tol = 5e-2 // attention drift compounds through every block of the stack
	wantLogits := logitsFullData[(blockSize-1)*vocab : blockSize*vocab]
	var maxDiff float32
	for i := 0; i < vocab; i++ {
		d := lastLogits[i] - wantLogits[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > tol {
		t.Fatalf("GPT.ForwardKVStep last-token logits max-abs-diff %g > tol %g", maxDiff, tol)
	}
	t.Logf("last-token logits max-abs-diff vs Forward: %g", maxDiff)
}

func TestGPT_ForwardKVStep_PanicsOnBadCache(t *testing.T) {
	a := uop.NewArena(1 << 18)
	g := nn.NewGPT(a, 8, 2, 2, 4, 4)
	// Cache geometry mismatch.
	bad := nn.NewKVCache(99, 2, 2, 4)
	idx := tensor.NewLeaf(a, []int64{1, 1}, uop.Dtypes.Int32, "webgpu")
	idx.SetData([]float32{0})
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic on bad cache geometry")
		}
	}()
	g.ForwardKVStep(idx, bad)
}

func TestGPT_ForwardKVStep_PanicsOnNilCache(t *testing.T) {
	a := uop.NewArena(1 << 18)
	g := nn.NewGPT(a, 8, 1, 2, 4, 4)
	idx := tensor.NewLeaf(a, []int64{1, 1}, uop.Dtypes.Int32, "webgpu")
	idx.SetData([]float32{0})
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic on nil cache")
		}
	}()
	g.ForwardKVStep(idx, nil)
}

func TestGPT_ForwardKVStep_PanicsOnBadIdxShape(t *testing.T) {
	a := uop.NewArena(1 << 18)
	g := nn.NewGPT(a, 8, 1, 2, 4, 4)
	cache := nn.NewKVCache(1, 2, 2, 4)
	idx := tensor.NewLeaf(a, []int64{2, 1}, uop.Dtypes.Int32, "webgpu")
	idx.SetData([]float32{0, 0})
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic on bad idx shape")
		}
	}()
	g.ForwardKVStep(idx, cache)
}
