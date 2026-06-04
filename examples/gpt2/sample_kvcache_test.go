package gpt2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestSampleGreedyKV_RejectsBadArgs(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPT(a, 8, 1, 2, 4, 4)
	bpe := buildStubRealBPE(t)

	cases := []struct {
		name          string
		maxContext    int
		maxNewTokens  int
		prompt        string
		wantSubstring string
	}{
		{"maxContext zero", 0, 1, "x", "maxContext"},
		{"maxContext beyond block", 100, 1, "x", "maxContext"},
		{"maxNewTokens zero", 4, 0, "x", "maxNewTokens"},
		{"prompt zero tokens", 4, 1, "", "zero tokens"},
		{"prompt plus new exceeds maxContext", 1, 1, "x", "maxContext"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := SampleGreedyKV(g, bpe, tc.prompt, tc.maxContext, tc.maxNewTokens, "webgpu")
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantSubstring) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSubstring)
			}
		})
	}
}

// TestSampleGreedyKV_HuggingFaceOracle is the load-bearing value gate. It runs
// the real GPT-2-small model loaded from the same HF safetensors blob that
// TestHFOracleFirstArgmax uses, then drives SampleGreedyKV to produce 10 new
// tokens from "Hello, world" and compares against the HF reference.
//
// Current status: at GPT-2-small scale (12 layers, 12 heads, 768 nEmbd, 1024
// blockSize), the KV step path produces NaN logits starting at prefill step 1
// regardless of cache MaxSeqLen. The small-scale oracle (TestGPT_ForwardKVStep
// _OracleAgainstForward, 2 layers / 2 heads / 8 nEmbd) passes within f32
// tolerance, so the per-layer math is correct in isolation. The compounding
// failure mode at full scale is suspected to involve the 8-buffer-per-kernel
// WebGPU cap interacting with the K_full = kCache + kNew * posOneHot scatter
// path; instrumentation of the attention kernel at full scale is the next
// slice. This test currently records the failure verbatim (first divergent
// position and the logit gap) per the verification-gate contract and t.Skip's
// when GPT-2 weights are unavailable.
func TestSampleGreedyKV_HuggingFaceOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: HF oracle test runs the full GPT-2 model")
	}
	cachePath := gpt2WeightsPath(t)
	if cachePath == "" {
		t.Skip("GPT-2 weights not in cache; run `anneal gpt2 sample <prompt>` once to populate")
	}
	if _, err := os.Stat(cachePath); err != nil {
		t.Skipf("GPT-2 weights stat at %s: %v", cachePath, err)
	}
	requireGPUForKV(t)

	a := uop.NewArena(1 << 14)
	g, bpe, err := LoadGPT2(a, "webgpu")
	if err != nil {
		t.Fatalf("LoadGPT2: %v", err)
	}

	f, err := os.Open(filepath.Join("testdata", "hf_oracle.json"))
	if err != nil {
		t.Fatalf("open oracle: %v", err)
	}
	defer func() { _ = f.Close() }()

	type greedy10 struct {
		Prompt       string  `json:"prompt"`
		DecodedText  string  `json:"decoded_text"`
		GeneratedIDs []int32 `json:"generated_ids"`
	}
	type oracle struct {
		Greedy10 greedy10 `json:"greedy_10_tokens"`
	}
	var orc oracle
	if err := json.NewDecoder(f).Decode(&orc); err != nil {
		t.Fatalf("decode oracle: %v", err)
	}

	firstArgmax, err := SampleArgmaxFirst(g, bpe, orc.Greedy10.Prompt, GPT2BlockSize, "webgpu")
	if err != nil {
		t.Fatalf("SampleArgmaxFirst sanity: %v", err)
	}
	t.Logf("sanity: SampleArgmaxFirst on %q = %d (HF expects %d)", orc.Greedy10.Prompt, firstArgmax, orc.Greedy10.GeneratedIDs[0])

	_, gotIDs, err := SampleGreedyKV(g, bpe, orc.Greedy10.Prompt, GPT2BlockSize, len(orc.Greedy10.GeneratedIDs), "webgpu")
	if err != nil {
		t.Fatalf("SampleGreedyKV: %v", err)
	}
	wantNew := orc.Greedy10.GeneratedIDs
	gotNew := gotIDs[len(gotIDs)-len(wantNew):]

	t.Logf("KV-cached greedy generated_ids: %v", gotNew)
	t.Logf("HF reference   generated_ids:  %v", wantNew)

	for i := range wantNew {
		if gotNew[i] != wantNew[i] {
			t.Fatalf("first divergence at position %d: got id %d (%q), want id %d (%q); full got=%v want=%v",
				i, gotNew[i], bpe.Decode([]int32{gotNew[i]}),
				wantNew[i], bpe.Decode([]int32{wantNew[i]}),
				gotNew, wantNew)
		}
	}
}

func requireGPUForKV(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU: %v", err)
	}
	tensor.DefaultExecutor = dev
}

// buildStubRealBPE returns a real *BPE built from a synthetic vocab so the
// argument-validation tests can construct one without touching disk. Its
// Encode of "" returns nil; non-empty prompts encode to at least one id.
func buildStubRealBPE(t *testing.T) *BPE {
	t.Helper()
	var vocab strings.Builder
	vocab.WriteByte('{')
	for i := 0; i < 256; i++ {
		if i > 0 {
			vocab.WriteByte(',')
		}
		runeC := rune('a' + i%26)
		vocab.WriteString(`"`)
		vocab.WriteRune(runeC)
		vocab.WriteString(`":`)
		vocab.WriteString(itoaInt(i))
	}
	vocab.WriteByte('}')

	// NewBPE needs at least one merge rule. We provide a single throwaway
	// merge over two synthetic tokens; the validation tests do not depend
	// on which subwords are produced.
	merges := "#version: 0.2\na b\n"
	bpe, err := NewBPE([]byte(vocab.String()), []byte(merges))
	if err != nil {
		t.Skipf("NewBPE rejected stub vocab (%v); arg-validation cases that need a non-empty prompt cannot run", err)
	}
	return bpe
}

func itoaInt(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [4]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
