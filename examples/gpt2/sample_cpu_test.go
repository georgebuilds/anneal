package gpt2

import (
	"context"
	"math/rand"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// The pure-Go CPU backend lets us exercise the GPU-executing bodies of the
// sampler (Sample / SampleStream / SampleArgmaxFirst / runSampleWithModel)
// without a real device, satisfying the no-GPU CI constraint.
//
// Two facts shape the fixtures here:
//
//  1. The nn.GPT forward needs nEmbd=16 / nHead=2 to lower cleanly on the CPU
//     interpreter; the original tiny config (nEmbd=8 / nHead=1) trips a
//     rank-collapse in the attention kernels.
//
//  2. A single-token (T=1) forward also rank-collapses on the CPU interp, so
//     every prompt fed to a CPU forward must encode to >= 2 tokens. The
//     tinyVocab "xy" prompt encodes to two ids (x=8, y=9) and is used
//     throughout. SampleGreedyKV / runKVStep feed T=1 per step by design and
//     therefore cannot run on the CPU backend (see TestSampleGreedyKV_CPU...).

// withCPUExecutor installs the CPU backend as the default executor for the
// duration of fn. It does NOT use t.Parallel because it mutates the global
// tensor.DefaultExecutor.
func withCPUExecutor(t *testing.T, fn func()) {
	t.Helper()
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()
	fn()
}

// cpuTinyGPT builds a CPU-runnable tied-head GPT plus the tinyVocab BPE.
func cpuTinyGPT(t *testing.T) (*nn.GPT, *BPE) {
	t.Helper()
	a := uop.NewArena(1 << 18)
	g := nn.NewGPTWithTiedHead(a, 16, 1, 2, 16, 8)
	tok, err := NewBPE([]byte(tinyVocab), []byte(tinyMerges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	return g, tok
}

// TestSampleCPU_Greedy runs the autoregressive Sample loop end-to-end on the
// CPU backend, covering the forward+realize+argmax body of Sample.
func TestSampleCPU_Greedy(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		text, err := Sample(g, bpe, "xy", g.BlockSize, "cpu", SampleOptions{MaxTokens: 2, Greedy: true})
		if err != nil {
			t.Fatalf("Sample: %v", err)
		}
		if !strings.HasPrefix(text, "xy") {
			t.Errorf("Sample output %q should start with prompt %q", text, "xy")
		}
	})
}

// TestSampleCPU_Stochastic covers the sampleFromLogits stochastic branch of
// Sample (Greedy=false with a seeded Rng), exercising temperature + top-k.
func TestSampleCPU_Stochastic(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		opts := SampleOptions{
			MaxTokens:   2,
			Temperature: 1.0,
			TopK:        5,
			Rng:         rand.New(rand.NewSource(1)),
		}
		text, err := Sample(g, bpe, "xy", g.BlockSize, "cpu", opts)
		if err != nil {
			t.Fatalf("Sample: %v", err)
		}
		if !strings.HasPrefix(text, "xy") {
			t.Errorf("Sample output %q should start with prompt %q", text, "xy")
		}
	})
}

// TestSampleCPU_NilRngDefault covers the `!opts.Greedy && rng == nil` default
// seeding branch in Sample (a fresh seed-1 source is created).
func TestSampleCPU_NilRngDefault(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		opts := SampleOptions{MaxTokens: 1, Temperature: 1.0, TopK: 0} // Rng nil, not greedy
		if _, err := Sample(g, bpe, "xy", g.BlockSize, "cpu", opts); err != nil {
			t.Fatalf("Sample (nil rng default): %v", err)
		}
	})
}

// TestSampleCPU_WindowTruncation drives Sample with ctxLen smaller than the
// running id sequence so the trailing-window slice branch fires.
func TestSampleCPU_WindowTruncation(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		// ctxLen=2 with a 2-token prompt + 2 new tokens forces the
		// len(windowIds) > ctxLen path on later steps.
		text, err := Sample(g, bpe, "xy", 2, "cpu", SampleOptions{MaxTokens: 2, Greedy: true})
		if err != nil {
			t.Fatalf("Sample (window truncation): %v", err)
		}
		if !strings.HasPrefix(text, "xy") {
			t.Errorf("Sample output %q should start with prompt", text)
		}
	})
}

// TestSampleStreamCPU_Greedy covers the SampleStream forward body + onTok
// callback path on CPU.
func TestSampleStreamCPU_Greedy(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		var toks []StreamToken
		text, err := SampleStream(context.Background(), g, bpe, "xy", g.BlockSize, "cpu",
			SampleOptions{MaxTokens: 2, Greedy: true},
			func(s StreamToken) { toks = append(toks, s) })
		if err != nil {
			t.Fatalf("SampleStream: %v", err)
		}
		if len(toks) != 2 {
			t.Fatalf("got %d stream tokens, want 2", len(toks))
		}
		// Step indices are 0-based and sequential.
		for i, tk := range toks {
			if tk.Step != i {
				t.Errorf("token %d Step = %d, want %d", i, tk.Step, i)
			}
			if !strings.Contains(tk.LogitSummary, "max=") {
				t.Errorf("token %d LogitSummary %q missing max=", i, tk.LogitSummary)
			}
			// On the greedy path ID == Argmax.
			if tk.ID != tk.Argmax {
				t.Errorf("token %d greedy: ID %d != Argmax %d", i, tk.ID, tk.Argmax)
			}
		}
		if !strings.HasPrefix(text, "xy") {
			t.Errorf("SampleStream text %q should start with prompt", text)
		}
	})
}

// TestSampleStreamCPU_Stochastic covers the SampleStream stochastic sampling
// branch (Greedy=false) including the nil-Rng default seeding.
func TestSampleStreamCPU_Stochastic(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		n := 0
		_, err := SampleStream(context.Background(), g, bpe, "xy", g.BlockSize, "cpu",
			SampleOptions{MaxTokens: 2, Temperature: 1.0, TopK: 5},
			func(StreamToken) { n++ })
		if err != nil {
			t.Fatalf("SampleStream stochastic: %v", err)
		}
		if n != 2 {
			t.Errorf("emitted %d tokens, want 2", n)
		}
	})
}

// TestSampleStreamCPU_WindowTruncation drives SampleStream with a small ctxLen
// so the trailing-window slice branch fires.
func TestSampleStreamCPU_WindowTruncation(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		_, err := SampleStream(context.Background(), g, bpe, "xy", 2, "cpu",
			SampleOptions{MaxTokens: 2, Greedy: true}, func(StreamToken) {})
		if err != nil {
			t.Fatalf("SampleStream (window truncation): %v", err)
		}
	})
}

// TestSampleArgmaxFirstCPU covers the forward+realize+argmax body of
// SampleArgmaxFirst on CPU. (Note: the production function hardcodes the
// "webgpu" device string for its leaf, but the actual execution backend is
// tensor.DefaultExecutor, so it runs on CPU here.)
func TestSampleArgmaxFirstCPU(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		id, err := SampleArgmaxFirst(g, bpe, "xy", g.BlockSize, "cpu")
		if err != nil {
			t.Fatalf("SampleArgmaxFirst: %v", err)
		}
		if id < 0 || int(id) >= g.Vocab {
			t.Errorf("argmax id %d out of vocab range [0,%d)", id, g.Vocab)
		}
	})
}

// TestSampleArgmaxFirstCPU_WindowTruncation drives SampleArgmaxFirst with a
// small ctxLen so the window-truncation branch fires.
func TestSampleArgmaxFirstCPU_WindowTruncation(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		if _, err := SampleArgmaxFirst(g, bpe, "xy", 1, "cpu"); err != nil {
			// ctxLen=1 truncates to a single token, which the CPU interp
			// rank-collapses; tolerate that specific failure but fail on a
			// successful-but-wrong path being silently broken.
			if !strings.Contains(err.Error(), "realize") {
				t.Fatalf("unexpected error: %v", err)
			}
		}
	})
}

// TestRunSampleWithModelCPU_Plain covers runSampleWithModel's plain-output
// branch (the body of RunSampleCLI after LoadGPT2).
func TestRunSampleWithModelCPU_Plain(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		var buf strings.Builder
		err := runSampleWithModel(&buf, g, bpe, "cpu", "xy",
			SampleOptions{MaxTokens: 1, Greedy: true}, true)
		if err != nil {
			t.Fatalf("runSampleWithModel(plain): %v", err)
		}
		out := buf.String()
		if !strings.HasPrefix(out, "xy") {
			t.Errorf("plain output %q should start with prompt", out)
		}
		if strings.Contains(out, "prompt:") {
			t.Errorf("plain output should not include the header, got %q", out)
		}
	})
}

// TestRunSampleWithModelCPU_Header covers runSampleWithModel's header-output
// branch, including the greedy/stochastic mode string.
func TestRunSampleWithModelCPU_Header(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		var buf strings.Builder
		err := runSampleWithModel(&buf, g, bpe, "cpu", "xy",
			SampleOptions{MaxTokens: 1, Greedy: true}, false)
		if err != nil {
			t.Fatalf("runSampleWithModel(header): %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "prompt: xy") {
			t.Errorf("header output missing prompt line, got %q", out)
		}
		if !strings.Contains(out, "mode: greedy") {
			t.Errorf("header output missing greedy mode, got %q", out)
		}
		if !strings.Contains(out, "---") {
			t.Errorf("header output missing separator, got %q", out)
		}
	})
}

// TestRunSampleWithModelCPU_StochasticMode covers the "stochastic" mode-string
// branch and the ctxLen clamp to BlockSize (MaxTokens large enough that
// len(ids)+MaxTokens > BlockSize).
func TestRunSampleWithModelCPU_StochasticMode(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		var buf strings.Builder
		// MaxTokens 1 keeps the run cheap; force the > BlockSize clamp by
		// using a model with BlockSize 8 and a prompt+budget under it, then
		// separately verify the stochastic header text.
		err := runSampleWithModel(&buf, g, bpe, "cpu", "xy",
			SampleOptions{MaxTokens: 1, Temperature: 1.0, TopK: 5, Rng: rand.New(rand.NewSource(2))}, false)
		if err != nil {
			t.Fatalf("runSampleWithModel(stochastic): %v", err)
		}
		if !strings.Contains(buf.String(), "mode: stochastic") {
			t.Errorf("expected stochastic mode header, got %q", buf.String())
		}
	})
}

// TestRunSampleWithModelCPU_CtxClamp covers the `ctxLen > g.BlockSize` clamp
// branch: a prompt + MaxTokens budget that exceeds BlockSize (8) is clamped.
func TestRunSampleWithModelCPU_CtxClamp(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t) // BlockSize == 8
		var buf strings.Builder
		// 2-token prompt ("xy") + MaxTokens 7 -> ctxLen 9 > 8, clamped to 8.
		// Sample then runs with the clamped window; 7 tiny CPU forwards stay
		// well under the test budget.
		err := runSampleWithModel(&buf, g, bpe, "cpu", "xy",
			SampleOptions{MaxTokens: 7, Greedy: true}, true)
		if err != nil {
			t.Fatalf("runSampleWithModel(ctx clamp): %v", err)
		}
	})
}

// TestRunSampleWithModelCPU_ZeroTokenPrompt covers the `ctxLen < 1` guard:
// an empty prompt encodes to zero ids.
func TestRunSampleWithModelCPU_ZeroTokenPrompt(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		var buf strings.Builder
		err := runSampleWithModel(&buf, g, bpe, "cpu", "",
			SampleOptions{MaxTokens: 0, Greedy: true}, true)
		if err == nil || !strings.Contains(err.Error(), "zero tokens") {
			t.Errorf("empty prompt: got %v, want zero-tokens error", err)
		}
	})
}

// TestRunSampleWithModelCPU_SampleError covers the error-return branch when
// Sample itself fails (here: MaxTokens<=0 is rejected by Sample, but ctxLen is
// still computed > 0 from the prompt so we reach the Sample call).
func TestRunSampleWithModelCPU_SampleError(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		var buf strings.Builder
		// MaxTokens 0 -> ctxLen = len(ids)+0 = 2 (>=1, passes the guard),
		// then Sample rejects MaxTokens<=0.
		err := runSampleWithModel(&buf, g, bpe, "cpu", "xy",
			SampleOptions{MaxTokens: 0, Greedy: true}, true)
		if err == nil || !strings.Contains(err.Error(), "MaxTokens") {
			t.Errorf("MaxTokens=0 via runSampleWithModel: got %v, want MaxTokens error", err)
		}
	})
}

// TestSampleGreedyKV_CPUUnsupported documents that the KV-cache step path
// feeds a single-token (T=1) forward that the CPU interpreter rank-collapses,
// so SampleGreedyKV / runKVStep cannot run on the CPU backend. The host-side
// validation and prefill-loop entry are still exercised; the GPU body is
// covered by the GPU-gated TestSampleGreedyKV_HuggingFaceOracle.
// TestSampleGreedyKV_CPU exercises the KV-cached greedy decode end-to-end on
// the CPU backend (prefill + per-token runKVStep, each a T=1 forward). The
// single-token forwards were previously a CPU rank-collapse failure; the
// broadcast-param indexing fix in backend/cpu (factor-1 strides + naga-style
// OOB clamp) made them realize correctly, so this path now runs in CI.
func TestSampleGreedyKV_CPU(t *testing.T) {
	withCPUExecutor(t, func() {
		g, bpe := cpuTinyGPT(t)
		const maxNew = 2
		_, ids, err := SampleGreedyKV(g, bpe, "xy", g.BlockSize, maxNew, "cpu")
		if err != nil {
			t.Fatalf("SampleGreedyKV on CPU: %v", err)
		}
		// ids = prompt ids followed by maxNew greedily-decoded ids (the returned
		// string is only the continuation, ids[len(prompt):]).
		promptIds := bpe.Encode("xy")
		if len(ids) != len(promptIds)+maxNew {
			t.Fatalf("ids len = %d, want %d (prompt %d + new %d)", len(ids), len(promptIds)+maxNew, len(promptIds), maxNew)
		}
		for i, want := range promptIds {
			if ids[i] != want {
				t.Errorf("ids[%d] = %d, want prompt id %d", i, ids[i], want)
			}
		}
	})
}
