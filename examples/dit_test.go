package examples

// DiT example (S3): registration, graph construction, and a GPU training smoke.
//
// The training loop runs on the GPU (webgpu), not the pure-Go CPU interpreter:
// the diffusion/DiT BACKWARD pass hits a known, documented backend/cpu defect
// ("f32 load flat=N out of range"; see the NOTE in cpu_train_test.go and
// notes/dit_meanflow_program.md). That is shared-library code reported, not
// fixed, by convention, so DiT training is GPU-only exactly like the diffusion
// example. The full CIFAR-10 convergence run is S4.

import (
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
)

// requireGPUForDiTTest mirrors the per-test GPU bootstrap used by the other
// example tests. Kept local so this file stays self-contained.
func requireGPUForDiTTest(t *testing.T) {
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

func TestDiTExampleRegistered(t *testing.T) {
	ex, err := Get("dit")
	if err != nil {
		t.Fatalf("Get(dit): %v", err)
	}
	if ex.Build == nil || ex.Train == nil {
		t.Fatal("dit example missing Build or Train")
	}
	if !strings.Contains(ex.Summary, "Diffusion Transformer") {
		t.Errorf("Summary should mention Diffusion Transformer: %q", ex.Summary)
	}
}

func TestBuildDiTConstructs(t *testing.T) {
	// Graph construction only (no Realize): exercises model assembly, seed copy,
	// param Load, and the full forward graph build used by `anneal run/graph/kernels`.
	br, err := buildDiT("webgpu")
	if err != nil {
		t.Fatalf("buildDiT: %v", err)
	}
	if br == nil || br.Arena == nil || br.Output == nil {
		t.Fatal("buildDiT returned nil arena/output")
	}
	if len(br.Leaves) == 0 {
		t.Fatal("buildDiT returned no parameter leaves")
	}
	dc := ditDefaultConfig()
	sh := br.Output.Shape()
	if len(sh) != 4 || sh[0] != ditBatch || sh[1] != dc.inCh || sh[2] != dc.imageH || sh[3] != dc.imageW {
		t.Fatalf("buildDiT output shape: got %v, want [%d, %d, %d, %d]",
			sh, ditBatch, dc.inCh, dc.imageH, dc.imageW)
	}
}

// TestRunDiTFewStepsSmoke runs the full DiT training loop (forward + backward +
// Adam) for a couple of steps on the GPU with a small config and an in-memory
// CIFAR-10 fixture (no 170MB download), then the classifier-free-guidance sample
// sweep. Mirrors TestTrainDiffusionFewStepsSmoke. Loss values are checked finite;
// convergence is the S4 GPU run.
func TestRunDiTFewStepsSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU DiT smoke")
	}
	requireGPUForDiTTest(t)

	ds := synthCIFAR10(4, rand.New(rand.NewSource(7)))
	dc := ditConfig{
		imageH: 32, imageW: 32, patch: 8, inCh: 3,
		embedDim: 32, condDim: 32, timeEmbedDim: 32, numClasses: 10,
		nLayer: 1, nHead: 2, T: 20,
		betaStart: 1e-4, betaEnd: 0.02, adamLR: 1e-3, initScale: 0.02,
		cfgDropProb: 0.1,
	}

	var captured strings.Builder
	var losses []float32
	cfg := TrainConfig{
		Steps:    2,
		LR:       0, // exercises the 0 -> dc.adamLR swap
		Batch:    2,
		LogEvery: 1,
		LogText:  func(s string) { captured.WriteString(s) },
	}
	if err := runDiT("webgpu", cfg, func(_ int, loss float32) {
		losses = append(losses, loss)
	}, ds, dc, 7); err != nil {
		t.Fatalf("runDiT: %v", err)
	}

	if len(losses) == 0 {
		t.Fatal("no losses logged")
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) || l < 0 {
			t.Fatalf("loss[%d] not a valid MSE: %v", i, l)
		}
	}
	if !strings.Contains(captured.String(), "training complete") {
		t.Errorf("LogText did not receive completion line; got %q", captured.String())
	}
	if !strings.Contains(captured.String(), "CFG sample mean=") {
		t.Errorf("LogText did not receive CFG sample line; got %q", captured.String())
	}
}

// TestRunDiTDefaultDimsGPU verifies the PRODUCTION config (ditDefaultConfig:
// embedDim 64, 64 patch tokens, 2 blocks) trains on the GPU, i.e. the full-size
// forward + backward realizes on Metal with no codegen-scaling ceiling (the risk
// that gates ResNet-9 training). Uses an in-memory CIFAR fixture (no download)
// and batch 2 / 1 step so it stays a smoke; the long convergence run is S4.
func TestRunDiTDefaultDimsGPU(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU DiT default-dims smoke")
	}
	requireGPUForDiTTest(t)

	ds := synthCIFAR10(4, rand.New(rand.NewSource(11)))
	dc := ditDefaultConfig()
	var losses []float32
	cfg := TrainConfig{Steps: 1, LR: 0, Batch: 2, LogEvery: 1}
	if err := runDiT("webgpu", cfg, func(_ int, l float32) {
		losses = append(losses, l)
	}, ds, dc, 11); err != nil {
		t.Fatalf("runDiT (default dims): %v", err)
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) || l < 0 {
			t.Fatalf("loss[%d] invalid: %v", i, l)
		}
	}
}
