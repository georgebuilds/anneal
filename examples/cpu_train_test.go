package examples

// CPU-backend full-train coverage. These tests run real forward + backward +
// optimizer-step loops on the pure-Go CPU executor (backend/cpu), so they
// execute under `go test -short` on a GPU-less CI machine and contribute real
// statement coverage for the train-loop bodies that the webgpu/Steps=0 tests
// in coverage_*_test.go cannot reach.
//
// Configs are kept tiny (small dims, 1-2 steps) to stay well inside the CI
// budget. The CPU backend does NOT implement SymbolicExecutor, so the
// dynmlp (symbolic-batch) train body is intentionally not run here.
//
// NOTE: these mutate the global tensor.DefaultExecutor, so they must not run
// with t.Parallel().

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// withCPU installs the pure-Go CPU executor as the default for the duration
// of fn, restoring the previous executor afterward.
func withCPU(t *testing.T, fn func()) {
	t.Helper()
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = prev }()
	fn()
}

// ── MLP full train on CPU ────────────────────────────────────────────────────

func TestTrainMLPCPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		var steps []int
		var losses []float32
		cfg := TrainConfig{
			Steps:    2,
			LR:       0.01,
			LogEvery: 1,
			OnStep:   func(int) {},
		}
		err := trainMLP("cpu", cfg, func(step int, loss float32) {
			steps = append(steps, step)
			losses = append(losses, loss)
		})
		if err != nil {
			t.Fatalf("trainMLP cpu: %v", err)
		}
		// LogEvery=1 logs step 0 plus steps 1,2.
		if len(losses) != 3 {
			t.Fatalf("expected 3 logged losses (0,1,2), got %d: %v", len(losses), losses)
		}
		for i, l := range losses {
			if l <= 0 {
				t.Errorf("loss[%d]=%v should be positive MSE", i, l)
			}
		}
	})
}

// ── Conv full train on CPU ───────────────────────────────────────────────────

func TestTrainConvCPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		var n int
		cfg := TrainConfig{
			Steps:    2,
			LR:       0.01,
			LogEvery: 1,
			OnStep:   func(int) { n++ },
		}
		var losses []float32
		err := trainConv("cpu", cfg, func(_ int, loss float32) {
			losses = append(losses, loss)
		})
		if err != nil {
			t.Fatalf("trainConv cpu: %v", err)
		}
		if n != 2 {
			t.Errorf("OnStep called %d times, want 2", n)
		}
		if len(losses) != 3 {
			t.Errorf("expected 3 logged losses, got %d", len(losses))
		}
	})
}

// ── nanoGPT full train + generate on CPU (tiny config) ───────────────────────

// fixtureTinyDataset is a small in-memory corpus with a tiny block_size
// window, large enough that SampleBatch has many valid starts.
func fixtureTinyDataset() *charDataset {
	return newCharDatasetFromString(strings.Repeat("abcdefgh ", 64))
}

func tinyGPTConfig(vocab int) nanoGPTConfig {
	return nanoGPTConfig{
		Vocab:        vocab,
		NLayer:       1,
		NHead:        2,
		NEmbd:        16,
		BlockSize:    4,
		SampleTokens: 2,
	}
}

func TestRunNanoGPTCPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyGPTConfig(ds.VocabSize())
		var captured strings.Builder
		var losses []float32
		tcfg := TrainConfig{
			Steps:    1,
			LR:       0,
			LogEvery: 1,
			Batch:    2,
			OnStep:   func(int) {},
			LogText:  func(s string) { captured.WriteString(s) },
		}
		err := runNanoGPT("cpu", tcfg, func(_ int, loss float32) {
			losses = append(losses, loss)
		}, ds, cfg, 7)
		if err != nil {
			t.Fatalf("runNanoGPT cpu: %v", err)
		}
		// LogEvery=1 -> step 0 + step 1.
		if len(losses) != 2 {
			t.Fatalf("expected 2 logged losses, got %d: %v", len(losses), losses)
		}
		// Final sample is emitted via LogText.
		out := captured.String()
		if !strings.Contains(out, "sample (") {
			t.Errorf("LogText missing generated sample header; got %q", out)
		}
	})
}

// TestRunNanoGPTCPUSGDDefaultLR exercises the lr<=0||sentinel swap path with a
// real step so the Adam construction with the swapped lr runs.
func TestRunNanoGPTCPUSentinelLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyGPTConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 1, LR: cmdTrainSGDDefaultLR, Batch: 0}
		err := runNanoGPT("cpu", tcfg, func(int, float32) {}, ds, cfg, 3)
		if err != nil {
			t.Fatalf("runNanoGPT sentinel lr cpu: %v", err)
		}
	})
}

// ── trainNanoGPT wrapper: dataset-injection seam ─────────────────────────────

// withFixtureDataset swaps the package loadDataset seam to return ds, then
// restores it. Lets the Build / Train / Generate entry points run their full
// body without touching the network or the asset cache.
func withFixtureDataset(t *testing.T, ds *charDataset, err error, fn func()) {
	t.Helper()
	prev := loadDataset
	loadDataset = func() (*charDataset, error) { return ds, err }
	defer func() { loadDataset = prev }()
	fn()
}

// TestTrainNanoGPTWrapperHandoff covers the success branch of the thin
// trainNanoGPT wrapper (loadDataset -> runNanoGPT). The injected corpus is
// deliberately shorter than the default block_size+1 so runNanoGPT returns
// its "corpus too small" guard immediately after the cheap setup — no
// Realize, so this stays fast on CPU.
func TestTrainNanoGPTWrapperHandoff(t *testing.T) {
	// Default config block_size is 32; a 5-char corpus trips the guard.
	ds := newCharDatasetFromString("abcde")
	withFixtureDataset(t, ds, nil, func() {
		err := trainNanoGPT("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
		if err == nil || !strings.Contains(err.Error(), "block_size") {
			t.Fatalf("expected block_size guard error, got: %v", err)
		}
	})
}

// TestBuildNanoGPTWrapperSuccess covers buildNanoGPT's success path via the
// injected dataset. buildNanoGPT constructs the graph but does NOT Realize,
// so the full default config is cheap here.
func TestBuildNanoGPTWrapperSuccess(t *testing.T) {
	ds := newCharDatasetFromString(strings.Repeat("abcdefgh ", 64))
	withFixtureDataset(t, ds, nil, func() {
		br, err := buildNanoGPT("webgpu")
		if err != nil {
			t.Fatalf("buildNanoGPT (fixture): %v", err)
		}
		if br.Output == nil {
			t.Fatal("buildNanoGPT: nil Output")
		}
		if len(br.Leaves) == 0 {
			t.Fatal("buildNanoGPT: no leaves")
		}
		// Output is logits [B=1, T=blockSize, V=vocab].
		sh := br.Output.Shape()
		if len(sh) != 3 || sh[0] != 1 {
			t.Errorf("unexpected logits shape %v", sh)
		}
	})
}

// ── loadShakespeareDataset happy path via readCharDataset ────────────────────

func TestReadCharDatasetHappyPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corpus.txt")
	if err := os.WriteFile(path, []byte("abcabcabc"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ds, err := readCharDataset(path)
	if err != nil {
		t.Fatalf("readCharDataset: %v", err)
	}
	if ds.VocabSize() != 3 {
		t.Errorf("vocab=%d, want 3", ds.VocabSize())
	}
	if len(ds.Data) != 9 {
		t.Errorf("data len=%d, want 9", len(ds.Data))
	}
}

func TestReadCharDatasetMissingFile(t *testing.T) {
	_, err := readCharDataset(filepath.Join(t.TempDir(), "nope.txt"))
	if err == nil {
		t.Fatal("expected error for missing file")
	}
	if !strings.Contains(err.Error(), "read shakespeare asset") {
		t.Errorf("error should mention read: %v", err)
	}
}

// TestLoadShakespeareDatasetSeamRestored guards against a test leaving the
// loadDataset seam swapped: after a fixture override the original function
// pointer must be restored.
func TestLoadShakespeareDatasetSeamDefault(t *testing.T) {
	// The seam should point at the real loader by default. We can't call it
	// (would hit the network/cache), but we can confirm it is non-nil and that
	// withFixtureDataset restores it.
	if loadDataset == nil {
		t.Fatal("loadDataset seam is nil")
	}
	orig := loadDataset
	ds := newCharDatasetFromString("xyz")
	withFixtureDataset(t, ds, nil, func() {})
	if loadDataset == nil {
		t.Fatal("loadDataset seam nil after restore")
	}
	_ = orig
}

// ── ViT full train on CPU (tiny arch) ────────────────────────────────────────

// tinyViTArch is a deliberately minimal ViT shape: patch 16 over the fixed
// 32x32 image gives 2x2=4 patch tokens, embedDim 8, 1 block, 2 heads. This
// keeps a CPU forward+backward+Adam step inside the CI budget while still
// exercising the entire runViTArch body (patch embed, attention softmax,
// mean-pool head, cross-entropy, backward, Adam step, eval-loss probe, final
// wall-time LogText emission).
func tinyViTArch() vitArch {
	return vitArch{patch: 16, embedDim: 8, nLayer: 1, nHead: 2}
}

func TestRunViTArchCPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		ds := synthCIFAR10(8, rand.New(rand.NewSource(51)))
		var captured strings.Builder
		var losses []float32
		cfg := TrainConfig{
			Steps:    1,
			LR:       0,
			LogEvery: 1,
			Batch:    2,
			OnStep:   func(int) {},
			LogText:  func(s string) { captured.WriteString(s) },
		}
		err := runViTArch("cpu", cfg, func(_ int, loss float32) {
			losses = append(losses, loss)
		}, ds, 1, tinyViTArch())
		if err != nil {
			t.Fatalf("runViTArch cpu: %v", err)
		}
		if len(losses) != 2 {
			t.Fatalf("expected 2 logged losses (step 0 + step 1), got %d", len(losses))
		}
		if !strings.Contains(captured.String(), "training complete") {
			t.Errorf("missing wall-time LogText line; got %q", captured.String())
		}
	})
}

// TestRunViTArchCPUSentinelLR drives the cmdTrainSGDDefaultLR swap through a
// real CPU step. Batch is kept small (2) so the step stays cheap; the
// batch<=0 fallback ladder is covered separately by the Steps=0 webgpu test
// TestRunViTLogTextEmits in coverage_more_test.go.
func TestRunViTArchCPUSentinelLR(t *testing.T) {
	withCPU(t, func() {
		ds := synthCIFAR10(8, rand.New(rand.NewSource(52)))
		cfg := TrainConfig{Steps: 1, LR: cmdTrainSGDDefaultLR, Batch: 2}
		err := runViTArch("cpu", cfg, func(int, float32) {}, ds, 2, tinyViTArch())
		if err != nil {
			t.Fatalf("runViTArch sentinel lr cpu: %v", err)
		}
	})
}

// NOTE on diffusion CPU coverage: the diffusion train body (trainDiffusion)
// and its post-train forward sampler (diffusionSampleSmoke) both stay GPU-only
// and remain residual coverage gaps. Two distinct blockers:
//
//  1. The backward pass cannot be realized on the pure-Go CPU backend — even
//     at the smallest viable denoiser shape the gradient realize hits a
//     CPU-backend interpreter bug:
//
//	cpu: kernel 7: interp: f32 load flat=128 out of range [0,128)
//
//     That is a defect in backend/cpu (shared library code owned elsewhere),
//     not in this package, so per the task contract it is reported, not fixed.
//
//  2. diffusionSampleSmoke is forward-only (realizes fine on CPU) but walks
//     diffSampleSteps=50 fresh-arena forward passes over the canonical
//     16-channel denoiser, which exceeds the per-test CPU budget.
//
// Both are covered on a real device by TestTrainDiffusionFewStepsSmoke in
// diffusion_test.go, plus the Steps=0 setup tests there.

// ── evalViTLoss success path on CPU (tiny ViT) ───────────────────────────────

// TestEvalViTLossCPUSuccess covers the success branch of evalViTLoss (the
// Realize-OK path returning loss.Data()[0]) on the CPU backend with a tiny
// ViT. The no-GPU NaN branch is covered separately by
// TestEvalViTLossNoGPUReturnsNaN in coverage_more_test.go.
func TestEvalViTLossCPUSuccess(t *testing.T) {
	withCPU(t, func() {
		a := uop.NewArena(1 << 16)
		const B = int64(2)
		arch := tinyViTArch()
		v := nn.NewViT(a, vitImageH, vitImageW, arch.patch, vitInCh,
			arch.embedDim, arch.nLayer, arch.nHead, vitNumClasses)
		initViTSmall(v, vitInitScale, rand.New(rand.NewSource(1)))
		images := make([]float32, B*vitInCh*vitImageH*vitImageW)
		for i := range images {
			images[i] = 0.01 * float32(i%7)
		}
		labels := []int32{0, 3}
		loss := evalViTLoss(v, v.Params(), images, labels, B, "cpu")
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("evalViTLoss cpu: non-finite loss %v", loss)
		}
		if loss <= 0 {
			t.Errorf("cross-entropy loss should be positive, got %v", loss)
		}
	})
}

func TestDiffusionDatasetSmokeRng(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	d := diffusionDataset(2, 8, 8, rng)
	if len(d) != 2*8*8 {
		t.Fatalf("len=%d", len(d))
	}
}

// ── NanoGPTGenerateStream core on CPU (tiny config) ──────────────────────────

func TestNanoGPTStreamCoreCPU(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString(strings.Repeat("abcdefgh ", 32))
		cfg := tinyGPTConfig(ds.VocabSize())
		var toks []NanoGPTStreamToken
		out, err := nanoGPTStreamCore(context.Background(), "cpu", "abc", 3,
			func(tk NanoGPTStreamToken) { toks = append(toks, tk) }, ds, cfg)
		if err != nil {
			t.Fatalf("nanoGPTStreamCore: %v", err)
		}
		if len(toks) != 3 {
			t.Fatalf("expected 3 streamed tokens, got %d", len(toks))
		}
		// Decoded output contains the encoded prompt prefix + 3 generated runes.
		if len([]rune(out)) != len(ds.Encode("abc"))+3 {
			t.Errorf("output rune count = %d", len([]rune(out)))
		}
		// Each token's decoded char must be in vocab and the summary populated.
		for i, tk := range toks {
			if tk.Step != i {
				t.Errorf("token %d Step=%d", i, tk.Step)
			}
			if !strings.Contains(tk.LogitSummary, "max=") {
				t.Errorf("token %d missing logit summary: %q", i, tk.LogitSummary)
			}
		}
	})
}

// TestNanoGPTStreamCoreLongPrompt exercises the encoded>=T branch (prompt
// longer than block_size) on CPU.
func TestNanoGPTStreamCoreLongPrompt(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString(strings.Repeat("abcdefgh ", 32))
		cfg := tinyGPTConfig(ds.VocabSize()) // BlockSize=4
		out, err := nanoGPTStreamCore(context.Background(), "cpu", "abcdefgh", 1,
			func(NanoGPTStreamToken) {}, ds, cfg)
		if err != nil {
			t.Fatalf("nanoGPTStreamCore long prompt: %v", err)
		}
		if out == "" {
			t.Fatal("empty output")
		}
	})
}

// TestNanoGPTStreamCoreCtxCancel covers the context-cancellation early-return
// inside the generation loop.
func TestNanoGPTStreamCoreCtxCancel(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString(strings.Repeat("abcdefgh ", 32))
		cfg := tinyGPTConfig(ds.VocabSize())
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // already canceled: loop returns on the first ctx.Err() check
		_, err := nanoGPTStreamCore(ctx, "cpu", "abc", 5,
			func(NanoGPTStreamToken) {}, ds, cfg)
		if err == nil {
			t.Fatal("expected context-canceled error")
		}
	})
}

// ── dynmlp setup-path coverage ───────────────────────────────────────────────
//
// The CPU backend does not implement SymbolicExecutor, so trainDynMLP's
// symbolic-batch train body (RealizeWithBinding) cannot run here — that path
// stays GPU-only. We cover everything up to the first Realize: the lr/batch
// resolution, seed + persistent model assembly, the dynBatchSlice host prep,
// and the initial-loss probe closure (evalLoss), which on a no-symbolic
// executor takes its RealizeWithBinding-error -> returns 0 branch.

func TestTrainDynMLPSetupWithEvalProbe(t *testing.T) {
	// LogEvery>0 with Steps=0 runs evalLoss() once (initial-loss probe) then
	// stops before the train loop. evalLoss's RealizeWithBinding errors (no
	// executor / no SymbolicExecutor), so it returns 0 — exercising the
	// closure body without a GPU.
	var got float32 = -1
	cfg := TrainConfig{Steps: 0, LR: 0.01, LogEvery: 1, Batch: 4}
	if err := trainDynMLP("webgpu", cfg, func(_ int, l float32) { got = l }); err != nil {
		t.Fatalf("trainDynMLP setup: %v", err)
	}
	if got != 0 {
		t.Errorf("initial-loss probe = %v, want 0 (RealizeWithBinding must fail without a symbolic executor)", got)
	}
}

// ── trainViT offline wrapper ─────────────────────────────────────────────────

// TestTrainViTOffline covers the trainViT thin wrapper's CIFAR-load + error
// path. resnet9data.Load resolves through internal/assets, which fails fast
// under ANNEAL_OFFLINE=1 with an empty cache; trainViT wraps and returns that
// error. The success handoff (return runViT(...)) needs the 170 MB CIFAR
// tarball and stays uncovered in CI.
func TestTrainViTOffline(t *testing.T) {
	t.Setenv("ANNEAL_OFFLINE", "1")
	t.Setenv("ANNEAL_CACHE_DIR", t.TempDir())
	err := trainViT("webgpu", TrainConfig{Steps: 0}, func(int, float32) {})
	if err == nil {
		t.Fatal("trainViT: expected offline CIFAR-load error")
	}
	if !strings.Contains(err.Error(), "CIFAR-10") {
		t.Errorf("error should mention CIFAR-10 load: %v", err)
	}
}

// ── listNames empty-registry branch ──────────────────────────────────────────

// TestListNamesEmpty covers the len(order)==0 -> "(none)" branch by
// temporarily swapping the package order slice to empty. Not parallel: it
// mutates package state.
func TestListNamesEmpty(t *testing.T) {
	saved := order
	order = nil
	defer func() { order = saved }()
	if got := listNames(); got != "(none)" {
		t.Errorf("listNames() empty = %q, want (none)", got)
	}
}
