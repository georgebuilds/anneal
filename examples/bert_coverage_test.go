package examples

// CPU-only branch-coverage for examples/bert.go. Mirrors the conventions in
// cpu_train_test.go (withCPU + fixtureTinyDataset) and llama_coverage_test.go
// (the runX zero-steps / sentinel-LR / short-corpus patterns), specialised for
// the BERT entry points. Every test runs on the pure-Go CPU executor and stays
// inside the CI budget, so the whole file runs under `go test -short` without
// the GPU burst. The GPU convergence proof lives in bert_test.go.
//
// NOTE: these mutate global package state (tensor.DefaultExecutor via withCPU,
// the loadDataset seam), so they must not run with t.Parallel().

import (
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── buildBERT: success + dataset-error ────────────────────────────────────────

func TestBuildBERTWrapperSuccess(t *testing.T) {
	ds := fixtureTinyDataset()
	withFixtureDataset(t, ds, nil, func() {
		br, err := buildBERT("cpu")
		if err != nil {
			t.Fatalf("buildBERT (fixture): %v", err)
		}
		if br.Arena == nil {
			t.Error("buildBERT: nil Arena")
		}
		if br.Output == nil {
			t.Fatal("buildBERT: nil Output")
		}
		if br.Device != "cpu" {
			t.Errorf("buildBERT: Device = %q, want cpu", br.Device)
		}
		if len(br.Leaves) == 0 {
			t.Fatal("buildBERT: no leaves")
		}
		// Output is logits [B=1, T=blockSize, V=vocab].
		sh := br.Output.Shape()
		if len(sh) != 3 || sh[0] != 1 {
			t.Errorf("buildBERT: unexpected logits shape %v", sh)
		}
		// Model vocab is baseVocab + 1 (the [MASK] sentinel row).
		if sh[2] != int64(ds.VocabSize()+1) {
			t.Errorf("buildBERT: vocab dim = %d, want %d (+1 for [MASK])", sh[2], ds.VocabSize()+1)
		}
	})
}

func TestBuildBERTDatasetError(t *testing.T) {
	wantErr := errors.New("boom: no dataset")
	withFixtureDataset(t, nil, wantErr, func() {
		if _, err := buildBERT("cpu"); err == nil {
			t.Fatal("buildBERT: expected dataset-load error")
		}
	})
}

// ── trainBERT: wrapper handoff + dataset-error ────────────────────────────────

func TestTrainBERTDatasetError(t *testing.T) {
	wantErr := errors.New("boom: no dataset")
	withFixtureDataset(t, nil, wantErr, func() {
		err := trainBERT("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
		if err == nil {
			t.Fatal("trainBERT: expected dataset-load error")
		}
	})
}

// TestTrainBERTWrapperHandoff covers the thin trainBERT wrapper (loadDataset
// seam -> defaultBERTConfig -> runBERT). The injected corpus is shorter than the
// default block_size+1 so runBERT returns its corpus-length guard immediately
// after the cheap setup - no Realize. Mirrors TestTrainLlamaWrapperHandoff.
func TestTrainBERTWrapperHandoff(t *testing.T) {
	withCPU(t, func() {
		// Default config block_size is 32; a 5-char corpus trips the guard.
		ds := newCharDatasetFromString("abcde")
		withFixtureDataset(t, ds, nil, func() {
			err := trainBERT("cpu", TrainConfig{Steps: 1}, func(int, float32) {})
			if err == nil || !strings.Contains(err.Error(), "corpus length") {
				t.Fatalf("expected corpus length guard error, got: %v", err)
			}
		})
	})
}

// ── runBERT: lr / batch / steps resolution ────────────────────────────────────

// TestRunBERTSentinelLR drives the lr == cmdTrainSGDDefaultLR swap and the
// batch <= 0 fallback (Batch: 0) through a real CPU step.
func TestRunBERTSentinelLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyBERTConfig(ds.VocabSize())
		tcfg := TrainConfig{Steps: 1, LR: cmdTrainSGDDefaultLR, Batch: 0}
		err := runBERT("cpu", tcfg, func(int, float32) {}, ds, cfg, 3)
		if err != nil {
			t.Fatalf("runBERT sentinel lr cpu: %v", err)
		}
	})
}

// TestRunBERTZeroLR drives the lr <= 0 swap branch (same canonical-Adam swap as
// the sentinel), one real CPU step.
func TestRunBERTZeroLR(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyBERTConfig(ds.VocabSize())
		tcfg := TrainConfig{Steps: 1, LR: 0, Batch: 2}
		err := runBERT("cpu", tcfg, func(int, float32) {}, ds, cfg, 9)
		if err != nil {
			t.Fatalf("runBERT zero lr cpu: %v", err)
		}
	})
}

// TestRunBERTStepsFallback covers the steps <= 0 -> beCfg.Steps fallback: with
// TrainConfig.Steps == 0 and tinyBERTConfig.Steps == 1, exactly one step runs.
func TestRunBERTStepsFallback(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyBERTConfig(ds.VocabSize()) // Steps: 1
		var nSteps int
		tcfg := TrainConfig{Steps: 0, LR: 1e-3, Batch: 2, OnStep: func(int) { nSteps++ }}
		err := runBERT("cpu", tcfg, func(int, float32) {}, ds, cfg, 5)
		if err != nil {
			t.Fatalf("runBERT steps-fallback cpu: %v", err)
		}
		if nSteps != 1 {
			t.Errorf("steps fallback ran %d steps, want 1 (beCfg.Steps)", nSteps)
		}
	})
}

// ── runBERT: short-corpus guard ───────────────────────────────────────────────

func TestRunBERTShortCorpusErrors(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString("ab")
		cfg := tinyBERTConfig(ds.VocabSize()) // BlockSize=8
		tcfg := TrainConfig{Steps: 1, Batch: 1}
		err := runBERT("cpu", tcfg, func(int, float32) {}, ds, cfg, 1)
		if err == nil || !strings.Contains(err.Error(), "corpus length") {
			t.Fatalf("expected corpus length guard error, got: %v", err)
		}
	})
}

// ── emitBERTSample: stdout fallback (LogText nil) ─────────────────────────────

// TestRunBERTStdoutSampleNoPanic drives runBERT with LogText:nil so
// emitBERTSample takes its stdout-fallback branch. We only assert it runs
// without error/panic (the line goes to os.Stdout). One CPU step keeps it cheap.
func TestRunBERTStdoutSampleNoPanic(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyBERTConfig(ds.VocabSize())
		tcfg := TrainConfig{Steps: 1, LR: 1e-3, LogEvery: 0, Batch: 2, LogText: nil}
		if err := runBERT("cpu", tcfg, func(int, float32) {}, ds, cfg, 13); err != nil {
			t.Fatalf("runBERT stdout sample cpu: %v", err)
		}
	})
}

// ── sampleMLMBatch: force-mask + nil branches ─────────────────────────────────

// TestSampleMLMBatchForceMask: with p=0 no Bernoulli draw masks anything, so the
// numMasked==0 force-mask-one branch must fire, yielding exactly one masked
// position and numMasked==1.
func TestSampleMLMBatchForceMask(t *testing.T) {
	ds := fixtureTinyDataset()
	const maskID = int32(9) // baseVocab for the 9-char fixture
	rng := rand.New(rand.NewSource(1))
	inputs, targets, mask, numMasked := sampleMLMBatch(rng, 2, 8, ds, maskID, 0.0)
	if inputs == nil {
		t.Fatal("sampleMLMBatch returned nil for a valid corpus")
	}
	if numMasked != 1 {
		t.Fatalf("p=0 should force exactly one mask, got numMasked=%d", numMasked)
	}
	nMaskTrue, nMaskID := 0, 0
	for i := range mask {
		if mask[i] {
			nMaskTrue++
		}
		if inputs[i] == maskID {
			nMaskID++
		}
	}
	if nMaskTrue != 1 || nMaskID != 1 {
		t.Errorf("force-mask: mask-true=%d maskID-count=%d, want 1 and 1", nMaskTrue, nMaskID)
	}
	// Targets are the unmodified original window (never the [MASK] sentinel).
	for i, tg := range targets {
		if tg == maskID {
			t.Errorf("target[%d] is the [MASK] sentinel; targets must be real ids", i)
		}
	}
}

// TestSampleMLMBatchProbabilistic covers the rng.Float64()<p masking branch (p=1
// masks every position).
func TestSampleMLMBatchAllMasked(t *testing.T) {
	ds := fixtureTinyDataset()
	const maskID = int32(9)
	rng := rand.New(rand.NewSource(2))
	inputs, _, mask, numMasked := sampleMLMBatch(rng, 2, 8, ds, maskID, 1.0)
	if numMasked != len(mask) {
		t.Fatalf("p=1 should mask every position; numMasked=%d want %d", numMasked, len(mask))
	}
	for i := range inputs {
		if inputs[i] != maskID {
			t.Errorf("p=1: input[%d]=%d not [MASK]", i, inputs[i])
		}
	}
}

// TestSampleMLMBatchNilCorpus covers the SampleBatch-returns-nil branch (corpus
// shorter than the window).
func TestSampleMLMBatchNilCorpus(t *testing.T) {
	ds := newCharDatasetFromString("abc") // length 3 < T+1
	rng := rand.New(rand.NewSource(3))
	inputs, targets, mask, numMasked := sampleMLMBatch(rng, 2, 8, ds, 3, 0.15)
	if inputs != nil || targets != nil || mask != nil || numMasked != 0 {
		t.Fatalf("expected all-nil/0 for too-small corpus, got inputs=%v numMasked=%d", inputs, numMasked)
	}
}

// ── maskedOneHotBits: masked rows, ignored rows, out-of-range skip ────────────

func TestMaskedOneHotBits(t *testing.T) {
	const vocab = 5
	targets := []int32{1, 2, 3, 7, -1}
	mask := []bool{true, false, true, true, true}
	out := maskedOneHotBits(targets, mask, vocab)
	if len(out) != len(targets)*vocab {
		t.Fatalf("len=%d, want %d", len(out), len(targets)*vocab)
	}
	// Row 0 masked -> one-hot at col 1.
	if out[0*vocab+1] != 1.0 {
		t.Errorf("row0: expected 1.0 at col1")
	}
	// Row 1 NOT masked -> all zero (ignore_index).
	for j := 0; j < vocab; j++ {
		if out[1*vocab+j] != 0 {
			t.Errorf("row1 must be all-zero (ignored), found nonzero at col %d", j)
		}
	}
	// Row 2 masked -> one-hot at col 3.
	if out[2*vocab+3] != 1.0 {
		t.Errorf("row2: expected 1.0 at col3")
	}
	// Rows 3 (id 7 >= vocab) and 4 (id -1 < 0) are masked but out-of-range -> skipped (all-zero).
	for _, row := range []int{3, 4} {
		for j := 0; j < vocab; j++ {
			if out[row*vocab+j] != 0 {
				t.Errorf("row%d out-of-range id must produce all-zero, nonzero at col %d", row, j)
			}
		}
	}
}

// ── maskedCrossEntropyLoss: numeric oracle on CPU ─────────────────────────────

// TestMaskedCrossEntropyLossNumeric pins the masked-CE value: with all-zero
// logits [1,1,V], a single masked position (numMasked=1) targeting class 0, the
// log-softmax of zeros is -log(V) uniformly, so the masked NLL is -1/1 *
// (1 * -log(V)) = log(V).
func TestMaskedCrossEntropyLossNumeric(t *testing.T) {
	withCPU(t, func() {
		const V = int64(4)
		a := uop.NewArena(1 << 14)
		logits := tensor.NewLeaf(a, []int64{1, 1, V}, uop.Dtypes.Float32, "cpu")
		logits.SetData([]float32{0, 0, 0, 0})
		oh := tensor.NewLeaf(a, []int64{1, 1, V}, uop.Dtypes.Float32, "cpu")
		oh.SetData([]float32{1, 0, 0, 0}) // masked position, target class 0
		loss := maskedCrossEntropyLoss(logits, oh, 1, 1, V, 1)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("realize masked-CE: %v", err)
		}
		got := loss.Data()[0]
		want := float32(math.Log(float64(V)))
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Errorf("masked-CE = %v, want log(%d) = %v", got, V, want)
		}
	})
}

// ── reconstructBERT: short-corpus guard (direct) ──────────────────────────────

// TestReconstructBERTShortCorpus covers reconstructBERT's defensive
// len(ds.Data) < T guard, reached by calling it directly with a corpus shorter
// than the block size (runBERT's own guard normally prevents this).
func TestReconstructBERTShortCorpus(t *testing.T) {
	withCPU(t, func() {
		ds := newCharDatasetFromString("abc")
		cfg := tinyBERTConfig(ds.VocabSize()) // BlockSize=8 > 3
		a := uop.NewArena(1 << 14)
		m := newBERTModel(a, cfg)
		initBERTSmall(m, bertInitScale, rand.New(rand.NewSource(1)))
		_, err := reconstructBERT(m, m.Params(), cfg, ds, "cpu")
		if err == nil || !strings.Contains(err.Error(), "block_size") {
			t.Fatalf("expected block_size guard error, got: %v", err)
		}
	})
}

// ── Realize-failure branches (no executor) ────────────────────────────────────
//
// With tensor.DefaultExecutor == nil every Realize fails fast, exercising the
// error/NaN paths that a working backend never hits. Mirrors the
// TestEvalViTLossNoGPUReturnsNaN precedent in coverage_more_test.go.

func TestEvalBERTLossNoExecutorNaN(t *testing.T) {
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = nil
	defer func() { tensor.DefaultExecutor = prev }()

	ds := fixtureTinyDataset()
	cfg := tinyBERTConfig(ds.VocabSize())
	a := uop.NewArena(1 << 14)
	m := newBERTModel(a, cfg)
	initBERTSmall(m, bertInitScale, rand.New(rand.NewSource(1)))
	in, tg, mk, n := sampleMLMBatch(rand.New(rand.NewSource(2)), 1, cfg.BlockSize, ds, cfg.MaskID(), cfg.MaskProb)
	loss := evalBERTLoss(m, m.Params(), in, tg, mk, n, 1, int64(cfg.BlockSize), int64(cfg.Vocab), "webgpu")
	if !math.IsNaN(float64(loss)) {
		t.Errorf("evalBERTLoss with no executor: expected NaN, got %v", loss)
	}
}

func TestReconstructBERTRealizeError(t *testing.T) {
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = nil
	defer func() { tensor.DefaultExecutor = prev }()

	ds := fixtureTinyDataset()
	cfg := tinyBERTConfig(ds.VocabSize())
	a := uop.NewArena(1 << 14)
	m := newBERTModel(a, cfg)
	initBERTSmall(m, bertInitScale, rand.New(rand.NewSource(1)))
	if _, err := reconstructBERT(m, m.Params(), cfg, ds, "webgpu"); err == nil {
		t.Fatal("reconstructBERT with no executor: expected Realize error")
	}
}

func TestRunBERTRealizeErrorNoExecutor(t *testing.T) {
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = nil
	defer func() { tensor.DefaultExecutor = prev }()

	ds := fixtureTinyDataset()
	cfg := tinyBERTConfig(ds.VocabSize())
	tcfg := TrainConfig{Steps: 1, LR: 1e-3, LogEvery: 0, Batch: 2}
	err := runBERT("webgpu", tcfg, func(int, float32) {}, ds, cfg, 1)
	if err == nil || !strings.Contains(err.Error(), "realize") {
		t.Fatalf("runBERT with no executor: expected realize error, got: %v", err)
	}
}

// ── MaskID accessor ───────────────────────────────────────────────────────────

func TestBERTConfigMaskID(t *testing.T) {
	cfg := defaultBERTConfig(65)
	if cfg.BaseVocab != 65 || cfg.Vocab != 66 {
		t.Fatalf("defaultBERTConfig(65): BaseVocab=%d Vocab=%d, want 65 and 66", cfg.BaseVocab, cfg.Vocab)
	}
	if cfg.MaskID() != 65 {
		t.Errorf("MaskID()=%d, want 65 (== BaseVocab)", cfg.MaskID())
	}
}
