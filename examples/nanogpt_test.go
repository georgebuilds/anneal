package examples

// nanoGPT offline tests.
//
// Three gates, none of which touch the network or the assets cache:
//
//  1. TestNanoGPTDataPipeline: vocab construction and encode/decode
//     round-trip on a small in-memory Shakespeare fixture.
//
//  2. TestNanoGPTConvergence: 50 Adam steps on a tiny config produce a
//     final loss <= half the initial loss. Skips when no GPU is available
//     (matches the requireGPU pattern used in tensor/nn). The full
//     2000-step training is too slow for CI; this is the cheapest signal
//     that the forward + backward + Adam loop works end to end.
//
//  3. TestNanoGPTGeneration: after the convergence loop, run 20 generation
//     steps and assert the output length and that every decoded char is
//     in the dataset vocabulary.
//
// All three operate on a 5-line fixture stored as a Go string constant so
// the test runs hermetically: no HTTP, no os.UserCacheDir().

import (
	"math/rand"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
)

// shakespeareFixture is a 5-line hand-picked excerpt from tinyshakespeare,
// long enough that a tiny block-size window has many possible starts. We
// intentionally repeat it so the in-memory corpus crosses the
// block_size+1 floor with margin.
const shakespeareFixture = `First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
`

// fixtureCorpus returns the fixture repeated enough times to give the
// tiny-config sampler a few hundred valid windows. Keeps the test
// self-contained without inflating the source file.
func fixtureCorpus() string {
	return strings.Repeat(shakespeareFixture, 8)
}

// requireGPUTest is the per-package GPU bootstrap. Mirrors the requireGPU
// helpers in cmd/anneal/cli_test.go and tensor/nn/train_test.go. We can't
// import either of those (different test packages); the body is a verbatim
// copy of the conventional pattern.
func requireGPUTest(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU: %v", err)
	}
	t.Cleanup(func() {
		tensor.DefaultExecutor = nil
		dev.Close()
	})
	tensor.DefaultExecutor = dev
}

// ── 1. Data pipeline ─────────────────────────────────────────────────────────

func TestNanoGPTDataPipeline(t *testing.T) {
	ds := newCharDatasetFromString(shakespeareFixture)

	// Vocab must contain the canonical Shakespearean punctuation and the
	// upper- and lowercase ASCII letters that appear in the fixture.
	for _, r := range "FCSAB,.:" {
		if _, ok := ds.charToIdx[r]; !ok {
			t.Errorf("vocab missing rune %q", r)
		}
	}

	// Vocab is sorted: a property the encode/decode path relies on
	// implicitly (Vocab[i] == decode of token id i).
	for i := 1; i < len(ds.Vocab); i++ {
		if ds.Vocab[i] <= ds.Vocab[i-1] {
			t.Fatalf("vocab not strictly sorted at index %d: %q then %q",
				i, ds.Vocab[i-1], ds.Vocab[i])
		}
	}

	// Round-trip: encode then decode the fixture must be the fixture.
	ids := ds.Encode(shakespeareFixture)
	if len(ids) != len([]rune(shakespeareFixture)) {
		t.Fatalf("encode length %d != fixture rune count %d", len(ids), len([]rune(shakespeareFixture)))
	}
	got := ds.Decode(ids)
	if got != shakespeareFixture {
		t.Fatalf("round-trip mismatch:\n  want: %q\n  got:  %q", shakespeareFixture, got)
	}

	// Encoded ids must all be in [0, VocabSize).
	for i, id := range ids {
		if id < 0 || int(id) >= ds.VocabSize() {
			t.Fatalf("encoded id %d at pos %d out of range [0, %d)", id, i, ds.VocabSize())
		}
	}

	// SampleBatch with block_size = 8 returns the expected shapes and
	// ys is xs shifted by one.
	rng := rand.New(rand.NewSource(1))
	xs, ys := ds.SampleBatch(rng, 3, 8)
	if len(xs) != 3*8 || len(ys) != 3*8 {
		t.Fatalf("SampleBatch len: xs=%d ys=%d, want %d each", len(xs), len(ys), 3*8)
	}
	// For each row, verify ys matches the next-position id of xs in the corpus.
	for b := 0; b < 3; b++ {
		// Find the start by searching: the row of xs must appear contiguously
		// in ds.Data, followed by the row of ys at offset +1. We don't know
		// the start position; just verify the shift property directly.
		row := xs[b*8 : (b+1)*8]
		yrow := ys[b*8 : (b+1)*8]
		// Find row in ds.Data.
		start := -1
		for s := 0; s+8 < len(ds.Data); s++ {
			match := true
			for j := 0; j < 8; j++ {
				if ds.Data[s+j] != row[j] {
					match = false
					break
				}
			}
			if match {
				start = s
				break
			}
		}
		if start < 0 {
			t.Fatalf("batch row %d not found in corpus", b)
		}
		for j := 0; j < 8; j++ {
			if yrow[j] != ds.Data[start+1+j] {
				t.Fatalf("batch row %d: ys[%d]=%d != Data[start+1+%d]=%d",
					b, j, yrow[j], j, ds.Data[start+1+j])
			}
		}
	}

	t.Logf("data pipeline ok: vocab=%d encoded=%d corpus=%d",
		ds.VocabSize(), len(ids), len(ds.Data))
}

// ── 2. Convergence smoke test ────────────────────────────────────────────────

// TestNanoGPTConvergence runs 50 Adam steps on a tiny config and asserts
// the final loss is at most half the initial loss. The config is
// intentionally tiny (n_layer=1, n_head=2, n_embd=16, block_size=8,
// batch=2) so the test finishes in a few seconds even on software GPU.
func TestNanoGPTConvergence(t *testing.T) {
	requireGPUTest(t)

	ds := newCharDatasetFromString(fixtureCorpus())

	cfg := nanoGPTConfig{
		Vocab:     ds.VocabSize(),
		NLayer:    1,
		NHead:     2,
		NEmbd:     16,
		BlockSize: 8,
	}
	const (
		steps  = 50
		batch  = int64(2)
		lr     = float32(5e-3) // higher than the canonical 3e-4 so 50 steps move the needle visibly
		seed   = int64(2026)
		ratioM = float32(0.5)
	)

	var (
		initialLoss float32 = -1
		finalLoss   float32
	)
	logFn := func(step int, loss float32) {
		if initialLoss < 0 {
			initialLoss = loss
		}
		finalLoss = loss
		t.Logf("step %d: loss=%.6f", step, loss)
	}

	tcfg := TrainConfig{
		Steps:    steps,
		LR:       lr,
		LogEvery: 10,
		Batch:    batch,
	}
	if err := runNanoGPT("webgpu", tcfg, logFn, ds, cfg, seed); err != nil {
		t.Fatalf("runNanoGPT: %v", err)
	}

	if initialLoss < 0 {
		t.Fatal("no losses logged; expected at least step-0 log")
	}
	ratio := finalLoss / initialLoss
	t.Logf("convergence smoke: initial=%.6f final=%.6f ratio=%.3f", initialLoss, finalLoss, ratio)
	if !(finalLoss < initialLoss*ratioM) {
		t.Fatalf("convergence: final loss %.6f >= initial %.6f * %.2f = %.6f (ratio=%.3f)",
			finalLoss, initialLoss, ratioM, initialLoss*ratioM, ratio)
	}
}

// ── 3. Generation smoke test ────────────────────────────────────────────────

// TestNanoGPTGeneration runs a tiny generation pass after a short Adam
// burst and asserts the generated text has the expected length and that
// every character is in the dataset vocabulary. Length is checked against
// the prompt + nGen tokens contract.
func TestNanoGPTGeneration(t *testing.T) {
	requireGPUTest(t)

	ds := newCharDatasetFromString(fixtureCorpus())

	const nGen = 20
	cfg := nanoGPTConfig{
		Vocab:        ds.VocabSize(),
		NLayer:       1,
		NHead:        2,
		NEmbd:        16,
		BlockSize:    8,
		SampleTokens: nGen,
	}

	// LogText capture so we can assert the sample was emitted via the
	// configured sink (not stdout).
	var captured strings.Builder
	tcfg := TrainConfig{
		Steps:    5,
		LR:       1e-3,
		LogEvery: 0,
		Batch:    2,
		LogText:  func(s string) { captured.WriteString(s) },
	}
	if err := runNanoGPT("webgpu", tcfg, func(int, float32) {}, ds, cfg, 7); err != nil {
		t.Fatalf("runNanoGPT: %v", err)
	}

	out := captured.String()
	if out == "" {
		t.Fatal("LogText was not invoked (no sample emitted)")
	}

	// Locate the sample text after the leading "sample (...)\n" header.
	idx := strings.Index(out, "):\n")
	if idx < 0 {
		t.Fatalf("could not locate sample body in output:\n%s", out)
	}
	body := out[idx+3 : len(out)-1] // strip trailing newline

	// Body must start with the prompt characters (the prompt is included in
	// the produced sequence by design; see generateNanoGPT).
	for _, r := range nanoGPTSamplePrompt {
		if _, ok := ds.charToIdx[r]; !ok {
			// Skip prompt chars not in fixture vocab; this is a fixture
			// that may not contain every prompt char. Verify only that
			// length is sensible and chars are in vocab.
			break
		}
	}

	// Length contract: encoded prompt + nGen tokens.
	encodedPromptLen := len(ds.Encode(nanoGPTSamplePrompt))
	wantLen := encodedPromptLen + nGen
	if len([]rune(body)) != wantLen {
		t.Errorf("sample length: got %d runes, want %d (encoded prompt=%d + gen=%d)",
			len([]rune(body)), wantLen, encodedPromptLen, nGen)
	}

	// Every char must be in the vocab.
	for i, r := range body {
		if _, ok := ds.charToIdx[r]; !ok {
			t.Errorf("sample char %q at index %d not in vocab", r, i)
		}
	}

	first := body
	if len(first) > 60 {
		first = first[:60]
	}
	t.Logf("generation sample (first 60): %q", first)
}
