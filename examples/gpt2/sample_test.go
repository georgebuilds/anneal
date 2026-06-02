package gpt2

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Test fixtures (embedded; computed at dispatch time) ──────────────────────

// hfOracle is the on-disk HuggingFace GPT-2 reference. The structure mirrors
// examples/gpt2/testdata/hf_oracle.json verbatim so the test can be updated
// in lockstep with the fixture file (regenerate with the Python snippet in
// the dispatch notes).
type hfOracle struct {
	Model    string `json:"model"`
	Fixtures []struct {
		Prompt             string  `json:"prompt"`
		PromptIDs          []int32 `json:"prompt_ids"`
		FirstArgmaxID      int32   `json:"first_argmax_id"`
		FirstArgmaxDecoded string  `json:"first_argmax_decoded"`
	} `json:"fixtures"`
	Greedy10 struct {
		Prompt       string  `json:"prompt"`
		DecodedText  string  `json:"decoded_text"`
		GeneratedIDs []int32 `json:"generated_ids"`
	} `json:"greedy_10_tokens"`
}

func loadHFOracle(t *testing.T) hfOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/hf_oracle.json")
	if err != nil {
		t.Fatalf("read testdata/hf_oracle.json: %v", err)
	}
	var o hfOracle
	if err := json.Unmarshal(data, &o); err != nil {
		t.Fatalf("parse hf_oracle.json: %v", err)
	}
	return o
}

// ── Sampling-helper unit tests (no GPU, no weights) ─────────────────────────

func TestArgmaxInt32(t *testing.T) {
	cases := []struct {
		in   []float32
		want int32
	}{
		{[]float32{0.1, 0.2, 0.3}, 2},
		{[]float32{3.0, 2.0, 1.0}, 0},
		// Tie at index 0 wins (stable: first occurrence).
		{[]float32{1.0, 1.0, 1.0}, 0},
		// +Inf still wins.
		{[]float32{0, float32(math.Inf(1)), 0}, 1},
	}
	for _, c := range cases {
		got := argmaxInt32(c.in)
		if got != c.want {
			t.Errorf("argmaxInt32(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

// TestSampleFromLogitsGreedyFallback verifies temperature<=0 collapses to
// argmax even when topK is unset.
func TestSampleFromLogitsGreedyFallback(t *testing.T) {
	logits := []float32{0.1, 5.0, 0.2}
	got := sampleFromLogits(logits, 0, 0, rand.New(rand.NewSource(7)))
	if got != 1 {
		t.Errorf("temperature=0 sample: got %d, want 1 (argmax)", got)
	}
}

// TestSampleFromLogitsTopK1IsArgmax exercises the top-k path: with topK=1
// only the argmax has nonzero probability, so the sampler must return it.
func TestSampleFromLogitsTopK1IsArgmax(t *testing.T) {
	logits := []float32{0.1, 5.0, 0.2, -1.0}
	rng := rand.New(rand.NewSource(123))
	got := sampleFromLogits(logits, 1.0, 1, rng)
	if got != 1 {
		t.Errorf("top-k=1 sample: got %d, want 1 (argmax)", got)
	}
}

// TestInt32sAsLeafBitsRoundtrip mirrors the helper across the project; the
// uint32 bit-pattern must round-trip through the float32 leaf encoding.
func TestInt32sAsLeafBitsRoundtrip(t *testing.T) {
	vs := []int32{0, 1, -1, 50256, -123456789}
	bits := int32sAsLeafBits(vs)
	for i, v := range vs {
		got := int32(math.Float32bits(bits[i]))
		if got != v {
			t.Errorf("roundtrip[%d]: got %d, want %d", i, got, v)
		}
	}
}

// TestSampleRejectsZeroMaxTokens documents the API contract for the
// CLI-facing entry: MaxTokens must be > 0.
func TestSampleRejectsZeroMaxTokens(t *testing.T) {
	if _, err := Sample(nil, nil, "x", 1, "webgpu", SampleOptions{MaxTokens: 0}); err == nil {
		t.Errorf("Sample(MaxTokens=0) should error, got nil")
	}
}

// ── Asset-presence helpers ──────────────────────────────────────────────────

// gpt2AssetsPresent reports whether vocab, merges, and safetensors are all
// cached. Tests use this to skip when the assets have not been fetched (CI
// runs offline). Sets ANNEAL_OFFLINE=1 for the duration of the test so the
// assets package does not try the network.
func gpt2AssetsPresent(t *testing.T) bool {
	t.Helper()
	return gpt2WeightsPath(t) != ""
}

// requireGPUForSample is the test-only GPU bring-up. We avoid importing the
// nn package's requireGPU (sub-package internal) and re-do the bring-up
// inline. Mirrors cmd/anneal/cli_test.requireGPU verbatim.
func requireGPUForSample(t *testing.T) {
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

// ── HF oracle integration (gated on cached weights + GPU) ───────────────────

// TestHFOracleFirstArgmax is the load-bearing value test for this slice:
// for each precomputed reference prompt, anneal's last-position argmax must
// equal HuggingFace GPT-2's last-position argmax. A failure here means the
// loader, the GPT module, or some op along the chain is computing the
// wrong forward.
//
// Skips when ANNEAL_OFFLINE=1 + weights uncached, or when no GPU is present.
// The cache check happens FIRST so CI (which runs offline and may also lack
// a GPU) skips cleanly with a single message.
func TestHFOracleFirstArgmax(t *testing.T) {
	if !gpt2AssetsPresent(t) {
		t.Skip("GPT-2 weights not in cache; run 'anneal gpt2 sample <prompt>' once to fetch (~550 MB), then re-run tests")
	}
	requireGPUForSample(t)

	oracle := loadHFOracle(t)
	// Clear ANNEAL_OFFLINE so LoadGPT2 can resolve via the cache normally
	// (its verify-then-fetch path treats a cache hit as offline-safe).
	t.Setenv("ANNEAL_OFFLINE", "")

	a := uop.NewArena(1 << 20)
	g, bpe, err := LoadGPT2(a, "webgpu")
	if err != nil {
		t.Fatalf("LoadGPT2: %v", err)
	}

	for _, f := range oracle.Fixtures {
		// Cross-check BPE first (cheap, isolates tokenizer drift from forward bugs).
		ids := bpe.Encode(f.Prompt)
		if !int32SliceEqual(ids, f.PromptIDs) {
			t.Fatalf("BPE.Encode(%q): got %v, want %v (HF reference)", f.Prompt, ids, f.PromptIDs)
		}

		gotID, err := SampleArgmaxFirst(g, bpe, f.Prompt, GPT2BlockSize, "webgpu")
		if err != nil {
			t.Fatalf("SampleArgmaxFirst(%q): %v", f.Prompt, err)
		}
		if gotID != f.FirstArgmaxID {
			t.Errorf("first argmax for %q: got %d (%q), want %d (%q)",
				f.Prompt, gotID, bpe.Decode([]int32{gotID}),
				f.FirstArgmaxID, f.FirstArgmaxDecoded)
		}
	}
}

// TestSampleShapeAndRange verifies the autoregressive Sample loop produces
// the requested number of tokens in vocab range and the prompt is preserved
// at the head. Cheap (1 token, greedy) so we keep it under the GPU smoke
// budget.
func TestSampleShapeAndRange(t *testing.T) {
	if !gpt2AssetsPresent(t) {
		t.Skip("GPT-2 weights not in cache; see TestHFOracleFirstArgmax skip message")
	}
	requireGPUForSample(t)
	t.Setenv("ANNEAL_OFFLINE", "")

	a := uop.NewArena(1 << 20)
	g, bpe, err := LoadGPT2(a, "webgpu")
	if err != nil {
		t.Fatalf("LoadGPT2: %v", err)
	}

	prompt := "Hello, world"
	const newTokens = 1
	text, err := Sample(g, bpe, prompt, GPT2BlockSize, "webgpu", SampleOptions{
		MaxTokens: newTokens,
		Greedy:    true,
	})
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	if !strings.HasPrefix(text, prompt) {
		t.Errorf("Sample output should start with prompt %q, got %q", prompt, text)
	}
	// HF oracle says step 0 from "Hello, world" is ".".
	wantSuffix := "."
	if !strings.HasSuffix(text, wantSuffix) {
		t.Errorf("Sample(greedy, 1 token) suffix: want %q (HF oracle), got tail %q",
			wantSuffix, text[len(prompt):])
	}
}

// TestOfflineMissingAssetsHint verifies the LoadGPT2 error path when
// ANNEAL_OFFLINE=1 + cache miss: the error must mention ANNEAL_OFFLINE so
// downstream code (cmd/anneal/cmd_gpt2.go) can switch on it to surface the
// extra "fetch manually" hint.
func TestOfflineMissingAssetsHint(t *testing.T) {
	// We must not run this when the cache already holds the assets (the
	// happy-path test path), so we skip. In CI the cache is empty.
	if gpt2AssetsPresent(t) {
		t.Skip("gpt2 assets already cached; offline-missing path is not exercisable here")
	}
	t.Setenv("ANNEAL_OFFLINE", "1")
	a := uop.NewArena(1 << 14)
	_, _, err := LoadGPT2(a, "webgpu")
	if err == nil {
		t.Fatalf("expected error when ANNEAL_OFFLINE=1 + missing cache, got nil")
	}
	if !strings.Contains(err.Error(), "ANNEAL_OFFLINE=1") {
		t.Errorf("error should mention ANNEAL_OFFLINE=1, got: %v", err)
	}
}

// ── helpers ───────────────────────────────────────────────────────────────

// int32SliceEqual compares two int32 slices elementwise.
func int32SliceEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
