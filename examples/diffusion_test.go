package examples

import (
	"math/rand"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
)

// requireGPUForDiffusionTest mirrors the per-test GPU bootstrap used by
// nanogpt / resnet9. Kept local so each example test file is self-contained.
func requireGPUForDiffusionTest(t *testing.T) {
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

func TestDiffusionExampleRegistered(t *testing.T) {
	ex, err := Get("diffusion")
	if err != nil {
		t.Fatalf("Get(diffusion): %v", err)
	}
	if ex.Build == nil || ex.Train == nil {
		t.Fatal("diffusion example missing Build or Train")
	}
	if !strings.Contains(ex.Summary, "DDPM") {
		t.Errorf("Summary should mention DDPM: %q", ex.Summary)
	}
}

func TestBuildDiffusionConstructsForward(t *testing.T) {
	br, err := buildDiffusion("webgpu")
	if err != nil {
		t.Fatalf("buildDiffusion: %v", err)
	}
	if br.Arena == nil {
		t.Fatal("BuildResult.Arena is nil")
	}
	if br.Output == nil {
		t.Fatal("BuildResult.Output is nil")
	}
	// 10 params: 3 conv W+B, 2 linear W+B.
	if got, want := len(br.Leaves), 10; got != want {
		t.Fatalf("BuildResult.Leaves: got %d, want %d", got, want)
	}
	sh := br.Output.Shape()
	if len(sh) != 4 || sh[0] != diffBatch || sh[1] != diffInCh || sh[2] != diffImageH || sh[3] != diffImageW {
		t.Fatalf("BuildResult.Output shape: got %v, want [%d, %d, %d, %d]",
			sh, diffBatch, diffInCh, diffImageH, diffImageW)
	}
}

// TestTrainDiffusionZeroSteps exercises trainDiffusion with Steps=0: the
// loop body is skipped, but lr/batch resolution, model assembly, and final
// wall-time LogText emission all execute. CPU-only - no GPU dispatch.
func TestTrainDiffusionZeroSteps(t *testing.T) {
	var captured strings.Builder
	cfg := TrainConfig{
		Steps:   0,
		LR:      0, // exercises the 0 -> diffAdamLR swap
		Batch:   0, // exercises the <=0 -> diffBatch swap
		LogText: func(s string) { captured.WriteString(s) },
	}
	err := trainDiffusion("webgpu", cfg, func(int, float32) {})
	if err != nil {
		t.Fatalf("trainDiffusion (default fallbacks): %v", err)
	}
	if !strings.Contains(captured.String(), "training complete") {
		t.Errorf("LogText did not receive wall-time line; got %q", captured.String())
	}
}

// TestTrainDiffusionSentinelLR covers the cmdTrainSGDDefaultLR sentinel
// branch in the lr-fallback ladder. Steps=0 keeps us out of the Forward
// path so this stays CPU-only.
func TestTrainDiffusionSentinelLR(t *testing.T) {
	err := trainDiffusion("webgpu",
		TrainConfig{Steps: 0, LR: cmdTrainSGDDefaultLR, Batch: 2},
		func(int, float32) {})
	if err != nil {
		t.Fatalf("trainDiffusion (sentinel LR): %v", err)
	}
}

// TestTrainDiffusionFewStepsSmoke runs trainDiffusion with Steps=2 (plus
// LogText set) on the GPU so the loop body and final sample-sweep both
// execute. Loss values are not asserted - wiring is the only check.
// Skipped when no GPU is available so the CPU-only run stays clean.
func TestTrainDiffusionFewStepsSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU diffusion smoke")
	}
	requireGPUForDiffusionTest(t)

	var captured strings.Builder
	var loggedSteps []int
	cfg := TrainConfig{
		Steps:    2,
		LR:       0,
		Batch:    2,
		LogEvery: 1,
		LogText:  func(s string) { captured.WriteString(s) },
	}
	err := trainDiffusion("webgpu", cfg, func(step int, _ float32) {
		loggedSteps = append(loggedSteps, step)
	})
	if err != nil {
		t.Fatalf("trainDiffusion: %v", err)
	}
	if len(loggedSteps) != 2 {
		t.Errorf("expected 2 logged steps, got %d", len(loggedSteps))
	}
	if !strings.Contains(captured.String(), "training complete") {
		t.Errorf("LogText did not receive wall-time line; got %q", captured.String())
	}
	// diffusionSampleSmoke runs only when cfg.LogText is set; assert it emitted.
	if !strings.Contains(captured.String(), "sample mean=") {
		t.Errorf("LogText did not receive sample-stats line; got %q", captured.String())
	}
}

func TestDiffusionDatasetShape(t *testing.T) {
	const (
		B = int64(4)
		H = int64(8)
		W = int64(8)
	)
	rng := rand.New(rand.NewSource(123))
	got := diffusionDataset(B, H, W, rng)
	if int64(len(got)) != B*H*W {
		t.Fatalf("diffusionDataset: len=%d, want %d", len(got), B*H*W)
	}
	// Sanity: not all zero.
	var nz int
	for _, v := range got {
		if v != 0 {
			nz++
		}
	}
	if nz == 0 {
		t.Fatal("diffusionDataset: all values zero")
	}
}
