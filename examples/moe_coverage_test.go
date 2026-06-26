package examples

// CPU-only branch-coverage for examples/moe.go. Mirrors the conventions in
// cpu_train_test.go (withCPU + fixtureTinyDataset) and llama_coverage_test.go
// (the build/train wrapper, zero-steps, sentinel-LR, short-corpus patterns),
// specialised for the MoE entry points. Every test here runs on the pure-Go CPU
// executor and stays inside the CI budget, so the whole file runs under
// `go test -short` without the GPU (lavapipe) burst. The GPU convergence proof
// lives in moe_test.go and is intentionally not duplicated.
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

// ── buildMoE: success path via injected dataset ──────────────────────────────

// TestBuildMoEWrapperSuccess covers buildMoE's full body (loadDataset seam ->
// defaultMoEConfig -> newMoEModel -> initMoESmall -> Forward). It only constructs
// the graph (no Realize), so the default config is cheap here.
func TestBuildMoEWrapperSuccess(t *testing.T) {
	ds := fixtureTinyDataset()
	withFixtureDataset(t, ds, nil, func() {
		br, err := buildMoE("cpu")
		if err != nil {
			t.Fatalf("buildMoE (fixture): %v", err)
		}
		if br.Arena == nil {
			t.Error("buildMoE: nil Arena")
		}
		if br.Output == nil {
			t.Fatal("buildMoE: nil Output")
		}
		if br.Device != "cpu" {
			t.Errorf("buildMoE: Device = %q, want cpu", br.Device)
		}
		if len(br.Leaves) == 0 {
			t.Fatal("buildMoE: no leaves")
		}
		// Output is logits [B=1, T=blockSize, V=vocab].
		sh := br.Output.Shape()
		if len(sh) != 3 || sh[0] != 1 {
			t.Errorf("buildMoE: unexpected logits shape %v", sh)
		}
	})
}

// TestBuildMoEDatasetError covers buildMoE's loadDataset-error early return via
// the seam (a non-nil error from the injected loader).
func TestBuildMoEDatasetError(t *testing.T) {
	wantErr := errors.New("boom: no dataset")
	withFixtureDataset(t, nil, wantErr, func() {
		if _, err := buildMoE("cpu"); err == nil {
			t.Fatal("buildMoE: expected dataset-load error")
		}
	})
}

// ── trainMoE: wrapper handoff ────────────────────────────────────────────────

// TestTrainMoEDatasetError covers trainMoE's loadDataset-error early return via
// the seam.
func TestTrainMoEDatasetError(t *testing.T) {
	wantErr := errors.New("boom: no dataset")
	withFixtureDataset(t, nil, wantErr, func() {
		err := trainMoE("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
		if err == nil {
			t.Fatal("trainMoE: expected dataset-load error")
		}
	})
}

// TestTrainMoEWrapperHandoff covers the thin trainMoE wrapper (loadDataset seam
// -> defaultMoEConfig -> runMoE). The injected corpus is deliberately shorter
// than the default block_size+1 so runMoE returns its corpus-length guard
// immediately after the cheap setup - no Realize and no full-model generation,
// so this stays fast on CPU. Mirrors TestTrainLlamaWrapperHandoff.
func TestTrainMoEWrapperHandoff(t *testing.T) {
	withCPU(t, func() {
		// Default config block_size is 32; a 5-char corpus trips the guard.
		ds := newCharDatasetFromString("abcde")
		withFixtureDataset(t, ds, nil, func() {
			err := trainMoE("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
			if err == nil || !strings.Contains(err.Error(), "corpus length") {
				t.Fatalf("expected corpus length guard error, got: %v", err)
			}
		})
	})
}

// ── runMoE: zero-steps sample emission via LogText ───────────────────────────

// TestRunMoEZeroStepsLogTextEmits drives runMoE with Steps=0 and LogEvery=0 (no
// train loop, no eval probe) so the only Realize is inside generateMoE for the
// final sample. The sample is emitted through LogText, covering the
// emitMoESample LogText branch and generateMoE's short-prompt (left-pad) path.
func TestRunMoEZeroStepsLogTextEmits(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 2
		var captured strings.Builder
		tcfg := TrainConfig{
			Steps:    0,
			LR:       3e-4,
			LogEvery: 0,
			Batch:    2,
			LogText:  func(s string) { captured.WriteString(s) },
		}
		err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 5)
		if err != nil {
			t.Fatalf("runMoE zero-steps cpu: %v", err)
		}
		if !strings.Contains(captured.String(), "sample (") {
			t.Errorf("LogText missing generated sample header; got %q", captured.String())
		}
	})
}

// ── runMoE: learning-rate resolution branches ────────────────────────────────

// TestRunMoESentinelLR drives the lr == cmdTrainSGDDefaultLR swap branch with a
// real CPU step (tiny config LR is 0, so the canonical Adam lr is selected).
func TestRunMoESentinelLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 1, LR: cmdTrainSGDDefaultLR, Batch: 0}
		err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 3)
		if err != nil {
			t.Fatalf("runMoE sentinel lr cpu: %v", err)
		}
	})
}

// TestRunMoEZeroLR drives the lr <= 0 swap with a config LR of 0, so lr falls all
// the way through to the canonical Adam lr (moeAdamLR).
func TestRunMoEZeroLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 1, LR: 0, Batch: 2}
		err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 9)
		if err != nil {
			t.Fatalf("runMoE zero lr cpu: %v", err)
		}
	})
}

// TestRunMoEConfigLR covers the branch where the TrainConfig lr is unset but the
// model config carries a positive LR (so lr = moeCfg.LR and the moeAdamLR
// fallback is NOT taken).
func TestRunMoEConfigLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		cfg.LR = 1e-3 // positive config default
		tcfg := TrainConfig{Steps: 1, LR: 0, Batch: 2}
		err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 21)
		if err != nil {
			t.Fatalf("runMoE config lr cpu: %v", err)
		}
	})
}

// TestRunMoECustomLR covers the branch where the caller passes an explicit
// non-sentinel positive lr, which is respected verbatim (both lr-swap ifs false).
func TestRunMoECustomLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 1, LR: 2e-3, Batch: 2}
		err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 23)
		if err != nil {
			t.Fatalf("runMoE custom lr cpu: %v", err)
		}
	})
}

// ── runMoE: short-corpus guard ───────────────────────────────────────────────

// TestRunMoEShortCorpusErrors covers the "corpus length < block_size+1" guard. A
// 2-char corpus against blockSize=4 trips it before any Realize.
func TestRunMoEShortCorpusErrors(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString("ab")
		cfg := tinyMoEConfig(ds.VocabSize()) // BlockSize=4
		tcfg := TrainConfig{Steps: 0, Batch: 1}
		err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 1)
		if err == nil || !strings.Contains(err.Error(), "corpus length") {
			t.Fatalf("expected corpus length guard error, got: %v", err)
		}
	})
}

// ── emitMoESample: stdout fallback (LogText nil) ─────────────────────────────

// TestRunMoEStdoutSampleNoPanic drives runMoE with LogText:nil so emitMoESample
// takes its stdout-fallback branch. Steps=0 keeps it cheap.
func TestRunMoEStdoutSampleNoPanic(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 1
		tcfg := TrainConfig{Steps: 0, LR: 3e-4, LogEvery: 0, Batch: 2, LogText: nil}
		if err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 13); err != nil {
			t.Fatalf("runMoE stdout sample cpu: %v", err)
		}
	})
}

// ── runMoE: SampleTokens=0 fallback ──────────────────────────────────────────

// TestRunMoESampleTokensZeroFallback covers the nGen<=0 -> moeSampleTokens
// fallback inside runMoE. Steps=0 so only the (defaulted) generation runs; the
// tiny vocab keeps even the default 100 greedy decode steps cheap on CPU.
func TestRunMoESampleTokensZeroFallback(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		cfg.SampleTokens = 0 // exercises the nGen<=0 -> moeSampleTokens swap
		var captured strings.Builder
		tcfg := TrainConfig{
			Steps:   0,
			Batch:   2,
			LogText: func(s string) { captured.WriteString(s) },
		}
		if err := runMoE("cpu", tcfg, func(int, float32) {}, ds, cfg, 17); err != nil {
			t.Fatalf("runMoE sample-tokens-zero cpu: %v", err)
		}
		if !strings.Contains(captured.String(), "sample (") {
			t.Errorf("LogText missing sample header; got %q", captured.String())
		}
	})
}

// ── generateMoE: prompt-longer-than-T copy branch ────────────────────────────

// TestGenerateMoELongPrompt exercises generateMoE's `len(encoded) >= T` branch
// (prompt longer than block_size) directly on CPU. tinyMoEConfig has BlockSize=4;
// the 8-char prompt forces the tail-copy path.
func TestGenerateMoELongPrompt(t *testing.T) {
	withCPU(t, func() {
		a := uop.NewArena(1 << 16)
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize()) // BlockSize=4
		m := newMoEModel(a, cfg)
		initMoESmall(m, moeInitScale, rand.New(rand.NewSource(1)))
		out, err := generateMoE(m, m.Params(), cfg, ds, "abcdefgh", 1, "cpu")
		if err != nil {
			t.Fatalf("generateMoE long prompt cpu: %v", err)
		}
		if out == "" {
			t.Fatal("generateMoE: empty output")
		}
	})
}
