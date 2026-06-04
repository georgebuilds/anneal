package gpt2

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestDefaultSampleOptions(t *testing.T) {
	o := DefaultSampleOptions()
	if o.MaxTokens != 20 {
		t.Errorf("MaxTokens = %d, want 20", o.MaxTokens)
	}
	if o.Temperature != 1.0 {
		t.Errorf("Temperature = %v, want 1.0", o.Temperature)
	}
	if o.TopK != 40 {
		t.Errorf("TopK = %d, want 40", o.TopK)
	}
	if o.Greedy {
		t.Errorf("Greedy = true, want false")
	}
	if o.Rng != nil {
		t.Errorf("Rng = %v, want nil", o.Rng)
	}
}

// TestTinyBPEEncodeDecode exercises Encode and Decode end-to-end against the
// tiny hand-crafted vocab. The full pipeline (pre-tokenize -> byte_encoder ->
// merge -> vocab lookup) runs without needing the canonical GPT-2 fixture.
func TestTinyBPEEncodeDecode(t *testing.T) {
	tok, err := NewBPE([]byte(tinyVocab), []byte(tinyMerges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	// Empty string short-circuits in Encode and Decode.
	if got := tok.Encode(""); got != nil {
		t.Errorf("Encode(\"\") = %v, want nil", got)
	}
	if got := tok.Decode(nil); got != "" {
		t.Errorf("Decode(nil) = %q, want \"\"", got)
	}
	// "hello" merges fully to id 7.
	ids := tok.Encode("hello")
	if !reflect.DeepEqual(ids, []int32{7}) {
		t.Fatalf("Encode(\"hello\") = %v, want [7]", ids)
	}
	if got := tok.Decode(ids); got != "hello" {
		t.Errorf("Decode(%v) = %q, want \"hello\"", ids, got)
	}
	// Unknown sub-tokens (here: lone 'h' produced by the merge loop when
	// the input ends mid-merge) are silently dropped to keep Encode total.
	// "hex" pre-tokenizes to ["hex"]; the merge loop yields h, e, x (merge
	// "h e" rank 1 -> he, then no further merges, so "he" + "x" = ids 5, 8).
	ids = tok.Encode("hex")
	if !reflect.DeepEqual(ids, []int32{5, 8}) {
		t.Errorf("Encode(\"hex\") = %v, want [5 8]", ids)
	}
}

// TestEncodeSkipsUnknownSubTokens covers the `if id, ok := b.encoder[sub]; ok`
// false branch in Encode: when bpe produces a sub-token that is not in the
// vocab, Encode silently skips it instead of crashing.
func TestEncodeSkipsUnknownSubTokens(t *testing.T) {
	// Vocab deliberately omits "x" so the bpe sub-token "x" is dropped.
	// Note: "h" is included so the pre-tokenized "x" alone tokenizes to "x"
	// (no merges) and is then skipped at the vocab lookup.
	const vocab = `{"h": 0, "y": 1}`
	const merges = `#version: 0.2
h y
`
	tok, err := NewBPE([]byte(vocab), []byte(merges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	// 'x' is not in vocab, so Encode("x") returns an empty slice.
	if got := tok.Encode("x"); len(got) != 0 {
		t.Errorf("Encode(\"x\") with missing vocab entry: got %v, want empty", got)
	}
}

// TestDecodeSkipsUnknownIdAndRune covers both fallthrough branches in Decode:
// an id that is not in the decoder map (skipped) and a rune in the
// concatenated token string that is not in the byteDecoder (silently
// dropped).
func TestDecodeSkipsUnknownIdAndRune(t *testing.T) {
	// "h" is byte 0x68, in the bijection -> maps back to itself.
	// "Ω" (U+03A9) is NOT in the byte->rune bijection image, so when it
	// appears in a token string Decode drops it.
	const vocab = `{"h": 0, "Ω": 1}`
	tok, err := NewBPE([]byte(vocab), []byte(tinyMerges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	// Unknown id 999 is skipped; id 0 decodes to "h".
	if got := tok.Decode([]int32{0, 999}); got != "h" {
		t.Errorf("Decode([0 999]) = %q, want \"h\"", got)
	}
	// id 1's token string contains Ω, which is dropped; result is empty.
	if got := tok.Decode([]int32{1}); got != "" {
		t.Errorf("Decode([1]) for Ω token = %q, want \"\" (rune outside bijection image)", got)
	}
}

// tinyGPTAndBPE builds a tiny tied-head GPT plus the tinyVocab BPE so the
// argument-validation paths in Sample/SampleStream/SampleArgmaxFirst can run
// without GPU or canonical weights.
func tinyGPTAndBPE(t *testing.T) (*nn.GPT, *BPE) {
	t.Helper()
	a := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a, 16, 1, 1, 8, 16)
	tok, err := NewBPE([]byte(tinyVocab), []byte(tinyMerges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	return g, tok
}

func TestSampleArgValidation(t *testing.T) {
	g, bpe := tinyGPTAndBPE(t)
	// ctxLen <= 0
	if _, err := Sample(g, bpe, "hello", 0, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}); err == nil {
		t.Errorf("Sample(ctxLen=0) should error")
	}
	// ctxLen > g.BlockSize
	if _, err := Sample(g, bpe, "hello", g.BlockSize+1, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}); err == nil {
		t.Errorf("Sample(ctxLen > BlockSize) should error")
	}
	// prompt encodes to zero tokens (empty string)
	_, err := Sample(g, bpe, "", g.BlockSize, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true})
	if err == nil || !strings.Contains(err.Error(), "zero tokens") {
		t.Errorf("Sample(\"\") want zero-tokens error, got %v", err)
	}
}

func TestSampleStreamArgValidation(t *testing.T) {
	g, bpe := tinyGPTAndBPE(t)
	cb := func(StreamToken) {}
	// onTok nil
	if _, err := SampleStream(context.Background(), g, bpe, "hello", 1, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}, nil); err == nil {
		t.Errorf("SampleStream(onTok=nil) should error")
	}
	// MaxTokens <= 0
	if _, err := SampleStream(context.Background(), g, bpe, "hello", 1, "webgpu", SampleOptions{MaxTokens: 0, Greedy: true}, cb); err == nil {
		t.Errorf("SampleStream(MaxTokens=0) should error")
	}
	// ctxLen <= 0
	if _, err := SampleStream(context.Background(), g, bpe, "hello", 0, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}, cb); err == nil {
		t.Errorf("SampleStream(ctxLen=0) should error")
	}
	// ctxLen > BlockSize
	if _, err := SampleStream(context.Background(), g, bpe, "hello", g.BlockSize+1, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}, cb); err == nil {
		t.Errorf("SampleStream(ctxLen > BlockSize) should error")
	}
	// prompt encodes to zero tokens
	_, err := SampleStream(context.Background(), g, bpe, "", g.BlockSize, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}, cb)
	if err == nil || !strings.Contains(err.Error(), "zero tokens") {
		t.Errorf("SampleStream(\"\") want zero-tokens error, got %v", err)
	}
	// Cancelled ctx aborts before forward; the encoded prompt is still
	// decoded and returned alongside ctx.Err().
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	text, err := SampleStream(ctx, g, bpe, "hello", g.BlockSize, "webgpu", SampleOptions{MaxTokens: 1, Greedy: true}, cb)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("SampleStream(cancelled ctx) want context.Canceled, got %v", err)
	}
	if text != "hello" {
		t.Errorf("SampleStream(cancelled ctx) text = %q, want \"hello\"", text)
	}
}

func TestSampleArgmaxFirstArgValidation(t *testing.T) {
	g, bpe := tinyGPTAndBPE(t)
	// ctxLen <= 0
	if _, err := SampleArgmaxFirst(g, bpe, "hello", 0, "webgpu"); err == nil {
		t.Errorf("SampleArgmaxFirst(ctxLen=0) should error")
	}
	// ctxLen > BlockSize
	if _, err := SampleArgmaxFirst(g, bpe, "hello", g.BlockSize+1, "webgpu"); err == nil {
		t.Errorf("SampleArgmaxFirst(ctxLen > BlockSize) should error")
	}
	// prompt encodes to zero tokens
	_, err := SampleArgmaxFirst(g, bpe, "", g.BlockSize, "webgpu")
	if err == nil || !strings.Contains(err.Error(), "zero tokens") {
		t.Errorf("SampleArgmaxFirst(\"\") want zero-tokens error, got %v", err)
	}
}

func TestLoadGPT2WeightsIntoValidation(t *testing.T) {
	a := uop.NewArena(1 << 14)
	// Non-tied GPT is rejected up front.
	gRef := nn.NewGPT(a, GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize)
	if err := LoadGPT2WeightsInto(gRef, "/dev/null"); err == nil || !strings.Contains(err.Error(), "tied-head") {
		t.Errorf("LoadGPT2WeightsInto(non-tied) want tied-head error, got %v", err)
	}
	// Tied-head but with non-canonical config is rejected by the shape gate.
	gSmall := nn.NewGPTWithTiedHead(a, 16, 2, 2, 8, 8)
	if err := LoadGPT2WeightsInto(gSmall, "/dev/null"); err == nil || !strings.Contains(err.Error(), "canonical GPT-2-small config") {
		t.Errorf("LoadGPT2WeightsInto(non-canonical config) want config error, got %v", err)
	}
	// Canonical config but invalid safetensors path -> parse error.
	gCanonical := nn.NewGPTWithTiedHead(a, GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize)
	bogus := filepath.Join(t.TempDir(), "missing.safetensors")
	if err := LoadGPT2WeightsInto(gCanonical, bogus); err == nil || !strings.Contains(err.Error(), "parse safetensors") {
		t.Errorf("LoadGPT2WeightsInto(missing file) want parse error, got %v", err)
	}
}

// TestLoadGPT2OfflineMissingAssets points ANNEAL_CACHE_DIR at a fresh temp
// dir with ANNEAL_OFFLINE=1, forcing assets.Get to fail on the very first
// fetch. This exercises the gpt2-vocab branch of LoadGPT2.
func TestLoadGPT2OfflineMissingAssets(t *testing.T) {
	t.Setenv("ANNEAL_CACHE_DIR", t.TempDir())
	t.Setenv("ANNEAL_OFFLINE", "1")
	a := uop.NewArena(1 << 14)
	_, _, err := LoadGPT2(a, "webgpu")
	if err == nil {
		t.Fatalf("LoadGPT2 should error when cache is empty + offline")
	}
	if !strings.Contains(err.Error(), "fetch vocab") {
		t.Errorf("LoadGPT2 error should mention vocab fetch, got %v", err)
	}
}

// TestRunSampleCLIOfflineMissingAssets verifies RunSampleCLI surfaces the
// LoadGPT2 error when assets are not cached and we are offline. Exercises
// the LoadGPT2 -> error -> return path in RunSampleCLI.
func TestRunSampleCLIOfflineMissingAssets(t *testing.T) {
	t.Setenv("ANNEAL_CACHE_DIR", t.TempDir())
	t.Setenv("ANNEAL_OFFLINE", "1")
	var buf strings.Builder
	err := RunSampleCLI(&buf, "webgpu", "hello", DefaultSampleOptions(), false)
	if err == nil {
		t.Fatalf("RunSampleCLI should error when LoadGPT2 cannot fetch assets")
	}
	if !strings.Contains(err.Error(), "fetch vocab") {
		t.Errorf("RunSampleCLI error should propagate LoadGPT2 fetch error, got %v", err)
	}
	if buf.Len() != 0 {
		t.Errorf("RunSampleCLI on error should not write output, got %q", buf.String())
	}
}
