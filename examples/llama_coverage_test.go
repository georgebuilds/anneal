package examples

// CPU-only branch-coverage for examples/llama.go. Mirrors the conventions in
// cpu_train_test.go (withCPU + fixtureTinyDataset) and coverage_*_test.go (the
// runNanoGPT zero-steps / sentinel-LR / short-corpus patterns), specialised for
// the Llama entry points. Every test here runs on the pure-Go CPU executor and
// stays inside the CI budget, so the whole file runs under `go test -short`
// without the GPU (lavapipe) burst. The GPU convergence proof lives in
// llama_test.go and is intentionally not duplicated.
//
// NOTE: these mutate global package state (tensor.DefaultExecutor via withCPU,
// and the loadDataset seam), so they must not run with t.Parallel().

import (
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── buildLlama: success path via injected dataset ────────────────────────────

// TestBuildLlamaWrapperSuccess covers buildLlama's full body (loadDataset seam
// -> defaultLlamaConfig -> newLlamaModel -> initLlamaSmall -> Forward). It only
// constructs the graph (no Realize), so the default config is cheap here.
func TestBuildLlamaWrapperSuccess(t *testing.T) {
	ds := fixtureTinyDataset()
	withFixtureDataset(t, ds, nil, func() {
		br, err := buildLlama("cpu")
		if err != nil {
			t.Fatalf("buildLlama (fixture): %v", err)
		}
		if br.Arena == nil {
			t.Error("buildLlama: nil Arena")
		}
		if br.Output == nil {
			t.Fatal("buildLlama: nil Output")
		}
		if br.Device != "cpu" {
			t.Errorf("buildLlama: Device = %q, want cpu", br.Device)
		}
		if len(br.Leaves) == 0 {
			t.Fatal("buildLlama: no leaves")
		}
		// Output is logits [B=1, T=blockSize, V=vocab].
		sh := br.Output.Shape()
		if len(sh) != 3 || sh[0] != 1 {
			t.Errorf("buildLlama: unexpected logits shape %v", sh)
		}
	})
}

// TestBuildLlamaDatasetError covers buildLlama's loadDataset-error early return
// via the seam (a non-nil error from the injected loader).
func TestBuildLlamaDatasetError(t *testing.T) {
	wantErr := errors.New("boom: no dataset")
	withFixtureDataset(t, nil, wantErr, func() {
		if _, err := buildLlama("cpu"); err == nil {
			t.Fatal("buildLlama: expected dataset-load error")
		}
	})
}

// ── trainLlama: wrapper handoff ──────────────────────────────────────────────

// TestTrainLlamaDatasetError covers trainLlama's loadDataset-error early return
// via the seam (a non-nil error from the injected loader).
func TestTrainLlamaDatasetError(t *testing.T) {
	wantErr := errors.New("boom: no dataset")
	withFixtureDataset(t, nil, wantErr, func() {
		err := trainLlama("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
		if err == nil {
			t.Fatal("trainLlama: expected dataset-load error")
		}
	})
}

// TestTrainLlamaWrapperHandoff covers the thin trainLlama wrapper (loadDataset
// seam -> defaultLlamaConfig -> runLlama). The injected corpus is deliberately
// shorter than the default block_size+1 so runLlama returns its corpus-length
// guard immediately after the cheap setup — no Realize and no 100-token greedy
// generation on the full default model, so this stays fast on CPU. The actual
// train-loop body (forward/backward/Adam/eval/generate) is covered by the
// tiny-config runLlama tests below and by TestRunLlamaCPUFullLoop. Mirrors
// TestTrainNanoGPTWrapperHandoff.
func TestTrainLlamaWrapperHandoff(t *testing.T) {
	withCPU(t, func() {
		// Default config block_size is 32; a 5-char corpus trips the guard.
		ds := newCharDatasetFromString("abcde")
		withFixtureDataset(t, ds, nil, func() {
			err := trainLlama("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
			if err == nil || !strings.Contains(err.Error(), "corpus length") {
				t.Fatalf("expected corpus length guard error, got: %v", err)
			}
		})
	})
}

// ── runLlama: zero-steps sample emission via LogText ──────────────────────────

// TestRunLlamaZeroStepsLogTextEmits drives runLlama with Steps=0 and LogEvery=0
// (no train loop, no eval probe) so the only Realize is inside generateLlama for
// the final sample. The sample is emitted through LogText, covering the
// emitLlamaSample LogText branch and generateLlama's short-prompt (left-pad)
// path on CPU.
func TestRunLlamaZeroStepsLogTextEmits(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize())
		cfg.SampleTokens = 2
		var captured strings.Builder
		tcfg := TrainConfig{
			Steps:    0,
			LR:       3e-4,
			LogEvery: 0,
			Batch:    2,
			LogText:  func(s string) { captured.WriteString(s) },
		}
		err := runLlama("cpu", tcfg, func(int, float32) {}, ds, cfg, 5)
		if err != nil {
			t.Fatalf("runLlama zero-steps cpu: %v", err)
		}
		if !strings.Contains(captured.String(), "sample (") {
			t.Errorf("LogText missing generated sample header; got %q", captured.String())
		}
	})
}

// ── runLlama: sentinel-LR swap ───────────────────────────────────────────────

// TestRunLlamaSentinelLR drives the lr == cmdTrainSGDDefaultLR swap branch with
// a real CPU step so the Adam construction with the swapped (canonical Adam) lr
// runs. SampleTokens kept small to keep generation cheap.
func TestRunLlamaSentinelLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 1, LR: cmdTrainSGDDefaultLR, Batch: 0}
		err := runLlama("cpu", tcfg, func(int, float32) {}, ds, cfg, 3)
		if err != nil {
			t.Fatalf("runLlama sentinel lr cpu: %v", err)
		}
	})
}

// TestRunLlamaZeroLR drives the lr <= 0 swap branch (same canonical-Adam swap as
// the sentinel), one real CPU step.
func TestRunLlamaZeroLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 1, LR: 0, Batch: 2}
		err := runLlama("cpu", tcfg, func(int, float32) {}, ds, cfg, 9)
		if err != nil {
			t.Fatalf("runLlama zero lr cpu: %v", err)
		}
	})
}

// ── runLlama: short-corpus guard ─────────────────────────────────────────────

// TestRunLlamaShortCorpusErrors covers the "corpus length < block_size+1" guard.
// A 2-char corpus against blockSize=4 trips it before any Realize.
func TestRunLlamaShortCorpusErrors(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString("ab")
		cfg := tinyLlamaConfig(ds.VocabSize()) // BlockSize=4
		tcfg := TrainConfig{Steps: 0, Batch: 1}
		err := runLlama("cpu", tcfg, func(int, float32) {}, ds, cfg, 1)
		if err == nil || !strings.Contains(err.Error(), "corpus length") {
			t.Fatalf("expected corpus length guard error, got: %v", err)
		}
	})
}

// ── emitLlamaSample: stdout fallback (LogText nil) ───────────────────────────

// TestRunLlamaStdoutSampleNoPanic drives runLlama with LogText:nil so
// emitLlamaSample takes its stdout-fallback branch. We only assert it runs
// without error/panic (the line goes to os.Stdout). Steps=0 keeps it cheap.
func TestRunLlamaStdoutSampleNoPanic(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 0, LR: 3e-4, LogEvery: 0, Batch: 2, LogText: nil}
		if err := runLlama("cpu", tcfg, func(int, float32) {}, ds, cfg, 13); err != nil {
			t.Fatalf("runLlama stdout sample cpu: %v", err)
		}
	})
}

// ── runLlama: SampleTokens=0 fallback ────────────────────────────────────────

// TestRunLlamaSampleTokensZeroFallback covers the nGen<=0 -> llamaSampleTokens
// fallback inside runLlama. Steps=0 so only the (defaulted) generation runs; the
// tiny vocab keeps even 100 greedy decode steps cheap on CPU.
func TestRunLlamaSampleTokensZeroFallback(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize())
		cfg.SampleTokens = 0 // exercises the nGen<=0 -> llamaSampleTokens swap
		var captured strings.Builder
		tcfg := TrainConfig{
			Steps:   0,
			Batch:   2,
			LogText: func(s string) { captured.WriteString(s) },
		}
		if err := runLlama("cpu", tcfg, func(int, float32) {}, ds, cfg, 17); err != nil {
			t.Fatalf("runLlama sample-tokens-zero cpu: %v", err)
		}
		if !strings.Contains(captured.String(), "sample (") {
			t.Errorf("LogText missing sample header; got %q", captured.String())
		}
	})
}

// ── generateLlama: prompt-longer-than-T copy branch ──────────────────────────

// TestGenerateLlamaLongPrompt exercises generateLlama's `len(encoded) >= T`
// branch (prompt longer than block_size) directly on CPU. tinyLlamaConfig has
// BlockSize=4; the 8-char prompt forces the tail-copy path. Called directly so
// we don't depend on the fixed llamaSamplePrompt length.
func TestGenerateLlamaLongPrompt(t *testing.T) {
	withCPU(t, func() {
		a := uop.NewArena(1 << 16)
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize()) // BlockSize=4
		m := newLlamaModel(a, cfg)
		initLlamaSmall(m, llamaInitScale, rand.New(rand.NewSource(1)))
		out, err := generateLlama(m, m.Params(), cfg, ds, "abcdefgh", 1, "cpu")
		if err != nil {
			t.Fatalf("generateLlama long prompt cpu: %v", err)
		}
		if out == "" {
			t.Fatal("generateLlama: empty output")
		}
	})
}
