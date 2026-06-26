package examples

import (
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
)

// requireGPUForBERTTest mirrors the per-test GPU bootstrap used by the other
// example tests (kept local so the file is self-contained).
func requireGPUForBERTTest(t *testing.T) {
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

// TestRunBERTGPUConvergence trains a tiny BERT encoder on the GPU and asserts
// the held-out masked-LM loss at least halves - the end-to-end GPU proof that
// the bidirectional encoder stack (non-causal attention softmax, learned
// position broadcast, scatter-add token-embedding backward, masked
// cross-entropy) trains on the real device. Guarded on -short because the GPU
// JIT/compile burst is slow on the CI software renderer (lavapipe); the CPU
// smoke + the tensor/nn FD check cover correctness in CI.
//
// The corpus is exactly block_size+1 characters, so there is a single training
// window and each masked position's target is fixed (learnable from the
// position embedding alone). The model overfits it quickly, which makes this a
// fast, reliable optimizer smoke rather than a hard generalization task: with a
// random-offset corpus the masked char depends only on neighbour context
// (induction), which needs far more than a smoke-test step budget to learn.
// NLayer is 1 so the per-grad backward realize stays within the GPU budget.
func TestRunBERTGPUConvergence(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: GPU BERT training burst is slow on the software renderer; CPU smoke + FD checks cover the logic")
	}
	requireGPUForBERTTest(t)
	ds := newCharDatasetFromString("abcdefghi") // 9 chars, seq 8 -> one fixed window
	cfg := tinyBERTConfig(ds.VocabSize())
	cfg.NLayer = 1
	var losses []float32
	tcfg := TrainConfig{
		Steps:    120,
		LR:       1e-2,
		LogEvery: 120,
		Batch:    4,
		OnStep:   func(int) {},
		LogText:  func(string) {},
	}
	err := runBERT("webgpu", tcfg, func(_ int, loss float32) {
		losses = append(losses, loss)
	}, ds, cfg, 11)
	if err != nil {
		t.Fatalf("runBERT webgpu: %v", err)
	}
	if len(losses) < 2 {
		t.Fatalf("expected an initial and a final loss, got %v", losses)
	}
	initial, final := losses[0], losses[len(losses)-1]
	if !(final < 0.5*initial) {
		t.Errorf("masked-LM loss did not halve: start=%v end=%v", initial, final)
	}
	t.Logf("BERT masked-LM loss %.4f -> %.4f over %d steps", initial, final, tcfg.Steps)
}

// TestRunBERTCPUFullLoop runs the full masked-LM train loop (sample mask ->
// forward -> masked cross-entropy -> backward -> Adam -> eval probe ->
// reconstruction emit) on the pure-Go CPU backend with the tiny config, so it
// executes under `go test -short` without a GPU. Mirrors TestRunLlamaCPUFullLoop.
func TestRunBERTCPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyBERTConfig(ds.VocabSize())
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
		err := runBERT("cpu", tcfg, func(_ int, loss float32) {
			losses = append(losses, loss)
		}, ds, cfg, 7)
		if err != nil {
			t.Fatalf("runBERT cpu: %v", err)
		}
		// LogEvery=1 -> step 0 + step 1.
		if len(losses) != 2 {
			t.Fatalf("expected 2 logged losses, got %d: %v", len(losses), losses)
		}
		for i, l := range losses {
			if l <= 0 {
				t.Errorf("loss[%d]=%v should be positive masked cross-entropy", i, l)
			}
		}
		if !strings.Contains(captured.String(), "reconstruction") {
			t.Errorf("LogText missing masked-LM reconstruction; got %q", captured.String())
		}
	})
}
