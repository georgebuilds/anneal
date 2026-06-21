package examples

import (
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
)

// requireGPUForLlamaTest mirrors the per-test GPU bootstrap used by the other
// example tests (kept local so the file is self-contained).
func requireGPUForLlamaTest(t *testing.T) {
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

// TestRunLlamaGPUConvergence trains the tiny Llama on the GPU for a handful of
// steps and asserts the loss decreases — the end-to-end GPU proof (RoPE concat,
// GQA expand, tied-weight backward, and the 8-buffer budget on the deeper Llama
// backward all exercised on the real device). Guarded on -short because the GPU
// JIT/compile burst is slow on the CI software renderer (lavapipe); CPU smoke +
// the tensor/nn FD checks cover correctness in CI.
func TestRunLlamaGPUConvergence(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: GPU Llama training burst is slow on the software renderer; CPU smoke + FD checks cover the logic")
	}
	requireGPUForLlamaTest(t)
	ds := fixtureTinyDataset()
	cfg := tinyLlamaConfig(ds.VocabSize())
	cfg.NLayer = 2
	cfg.SampleTokens = 4
	var losses []float32
	tcfg := TrainConfig{
		Steps:    30,
		LR:       3e-3,
		LogEvery: 30,
		Batch:    4,
		OnStep:   func(int) {},
		LogText:  func(string) {},
	}
	err := runLlama("webgpu", tcfg, func(_ int, loss float32) {
		losses = append(losses, loss)
	}, ds, cfg, 11)
	if err != nil {
		t.Fatalf("runLlama webgpu: %v", err)
	}
	if len(losses) < 2 {
		t.Fatalf("expected an initial and a final loss, got %v", losses)
	}
	if !(losses[len(losses)-1] < losses[0]) {
		t.Errorf("loss did not decrease: start=%v end=%v", losses[0], losses[len(losses)-1])
	}
}

// tinyLlamaConfig is a minimal Llama config for CPU tests: 1 block, 2 query
// heads sharing 1 KV head (group size 2), nEmbd=16 (headDim=8, even for RoPE).
func tinyLlamaConfig(vocab int) llamaConfig {
	return llamaConfig{
		Vocab:        vocab,
		NLayer:       1,
		NHead:        2,
		NKVHead:      1,
		NEmbd:        16,
		Hidden:       32,
		BlockSize:    4,
		SampleTokens: 2,
	}
}

func TestRunLlamaCPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyLlamaConfig(ds.VocabSize())
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
		err := runLlama("cpu", tcfg, func(_ int, loss float32) {
			losses = append(losses, loss)
		}, ds, cfg, 7)
		if err != nil {
			t.Fatalf("runLlama cpu: %v", err)
		}
		if len(losses) != 2 {
			t.Fatalf("expected 2 logged losses, got %d: %v", len(losses), losses)
		}
		for i, l := range losses {
			if l <= 0 {
				t.Errorf("loss[%d]=%v should be positive cross-entropy", i, l)
			}
		}
		if !strings.Contains(captured.String(), "sample (") {
			t.Errorf("LogText missing generated sample header; got %q", captured.String())
		}
	})
}
