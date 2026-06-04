package examples

// CPU-only tests for the resnet9 example. The heavy GPU path (forward
// realize on the full 64/128/256/512 network) is acknowledged but not
// exercised here — those concerns live in tensor/nn/resnet9_test.go
// using a scaled-down channel config. This file's load-bearing checks are:
//   - the example is correctly registered with Build + Train
//   - buildResNet9 constructs a graph (no Realize) without panic
//   - resnet9CrossEntropy composes a scalar loss graph
//   - initResNet9Small fills every Conv / BN / Head parameter and the
//     in-place mutation is observable on the Param.Value slices
//   - resnet9EvalLoss returns a finite scalar against a tiny synthetic
//     CIFAR-10 batch on the GPU (skipped when no device is available)

import (
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	resnet9data "github.com/georgebuilds/anneal/examples/resnet9"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestResNet9ExampleRegistered(t *testing.T) {
	ex, err := Get("resnet9")
	if err != nil {
		t.Fatalf("Get(resnet9): %v", err)
	}
	if ex.Build == nil || ex.Train == nil {
		t.Fatal("resnet9 example missing Build or Train")
	}
	if !strings.Contains(ex.Summary, "ResNet") {
		t.Errorf("Summary should mention ResNet: %q", ex.Summary)
	}
}

func TestBuildResNet9ConstructsForward(t *testing.T) {
	br, err := buildResNet9("webgpu")
	if err != nil {
		t.Fatalf("buildResNet9: %v", err)
	}
	if br.Arena == nil {
		t.Fatal("BuildResult.Arena is nil")
	}
	if br.Output == nil {
		t.Fatal("BuildResult.Output is nil")
	}
	if len(br.Leaves) != 26 {
		t.Fatalf("BuildResult.Leaves: got %d, want 26 (8 conv W + 16 BN + 2 head)", len(br.Leaves))
	}
	sh := br.Output.Shape()
	if len(sh) != 2 || sh[0] != resnet9Batch || sh[1] != 10 {
		t.Fatalf("BuildResult.Output shape: got %v, want [%d, 10]", sh, resnet9Batch)
	}
}

func TestResNet9CrossEntropyShape(t *testing.T) {
	// Build the cross-entropy loss graph against synthetic logits +
	// one-hot targets. No Realize — just confirm the graph composes and
	// returns a scalar tensor (shape []).
	const B = int64(2)
	const C = int64(10)
	a := uop.NewArena(1 << 16)
	logits := tensor.NewLeaf(a, []int64{B, C}, uop.Dtypes.Float32, "webgpu")
	oh := tensor.NewLeaf(a, []int64{B, C}, uop.Dtypes.Float32, "webgpu")
	loss := resnet9CrossEntropy(logits, oh, B, C)
	if loss == nil {
		t.Fatal("resnet9CrossEntropy returned nil")
	}
	if got := len(loss.Shape()); got != 0 {
		t.Fatalf("loss rank: got %d, want 0 (scalar); shape=%v", got, loss.Shape())
	}
}

func TestInitResNet9SmallTouchesEveryParam(t *testing.T) {
	a := uop.NewArena(64)
	m := nn.NewResNet9Scaled(a, [4]int64{4, 8, 16, 32}, 10, uop.Dtypes.Float32, "webgpu")
	rng := rand.New(rand.NewSource(7))
	initResNet9Small(m, 0.05, rng)

	// Sanity: convs should be ~zero-mean small-normal; BN Weight should be
	// near 1.0; Head Weight should be ~zero-mean. We don't assert exact
	// distributions, only that the buffers are NOT all zero.
	allZero := func(s []float32) bool {
		for _, v := range s {
			if v != 0 {
				return false
			}
		}
		return true
	}
	for i, c := range m.Convs() {
		if allZero(c.Weight.Value) {
			t.Errorf("Conv[%d].Weight all zero", i)
		}
	}
	for i, bn := range m.BNs() {
		// Weight init: 1.0 + small perturbation, so not zero.
		if allZero(bn.Weight.Value) {
			t.Errorf("BN[%d].Weight all zero", i)
		}
	}
	if allZero(m.Head.Weight.Value) {
		t.Error("Head.Weight all zero")
	}
}

// synthCIFAR10 builds a tiny on-host CIFAR-10 struct (no network, no asset
// cache) suitable for the resnet9EvalLoss smoke test below. The train set
// holds nSamples records of small-amplitude Gaussian noise paired with
// deterministic labels in [0, 10). Mirrors the layout that resnet9data.Batch
// expects: a [N, 3, 32, 32] flat row-major Train block and a length-N
// TrainLabels slice.
func synthCIFAR10(nSamples int, rng *rand.Rand) *resnet9data.CIFAR10 {
	const imagePixels = 3 * 32 * 32
	x := make([]float32, nSamples*imagePixels)
	y := make([]int32, nSamples)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 0.1
	}
	for i := range y {
		y[i] = int32(i % 10)
	}
	return &resnet9data.CIFAR10{
		Train:       x,
		TrainLabels: y,
	}
}

// requireGPUForResNet9Test is the per-package GPU bootstrap mirroring the
// one in nanogpt_test.go. We can't share it (Go forbids cross-test-file
// symbol leaks via t.Helper alone — there's no conflict, this is just a
// per-test isolation choice that keeps each smoke test self-contained).
func requireGPUForResNet9Test(t *testing.T) {
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

// TestResNet9EvalLossSmoke wires the eval helper end to end against a tiny
// synthetic dataset on the GPU. The check is intentionally weak (finite,
// non-NaN, non-Inf) — the loss value is not under test, only the wiring
// (Batch -> OneHot -> Load -> Forward -> Realize -> Data[0]). Skipped when
// no GPU is available so the CPU-only test run stays clean.
func TestResNet9EvalLossSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU-bound smoke test")
	}
	requireGPUForResNet9Test(t)

	// Tiny model (scaled channels) so the GPU graph is small enough to
	// realize quickly without exercising the full 6.57M-param config.
	a0 := uop.NewArena(1 << 14)
	m := nn.NewResNet9Scaled(a0, [4]int64{4, 8, 16, 32}, 10, uop.Dtypes.Float32, "webgpu")
	initResNet9Small(m, 0.05, rand.New(rand.NewSource(11)))

	ds := synthCIFAR10(8, rand.New(rand.NewSource(12)))
	rng := rand.New(rand.NewSource(13))

	loss := resnet9EvalLoss(m, m.Params(), ds, rng, int64(2), "webgpu")
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("resnet9EvalLoss: non-finite loss %v", loss)
	}
}

// TestRunResNet9LogTextEmits drives runResNet9 with zero steps against the
// tiny in-memory fixture; the loop body is skipped (Steps=0, LogEvery=0)
// but the lr/batch fallback resolution, model assembly, and final wall-time
// LogText emission all execute. CPU-only — no GPU dispatch, no Realize.
func TestRunResNet9LogTextEmits(t *testing.T) {
	ds := synthCIFAR10(4, rand.New(rand.NewSource(31)))

	var captured strings.Builder
	cfg := TrainConfig{
		Steps:   0,
		LR:      0, // exercises the zero -> resnet9AdamLR swap
		Batch:   0, // exercises the <=0 -> resnet9Batch swap
		LogText: func(s string) { captured.WriteString(s) },
	}
	err := runResNet9("webgpu", cfg, func(int, float32) {}, ds,
		[4]int64{2, 4, 8, 16}, 1)
	if err != nil {
		t.Fatalf("runResNet9 (default fallbacks): %v", err)
	}
	if !strings.Contains(captured.String(), "training complete") {
		t.Errorf("LogText did not receive wall-time line; got %q", captured.String())
	}
}

// TestRunResNet9SentinelLR covers the cmdTrainSGDDefaultLR sentinel branch
// in the lr-fallback ladder. Steps=0 keeps us out of the Forward path so
// this stays CPU-only.
func TestRunResNet9SentinelLR(t *testing.T) {
	ds := synthCIFAR10(4, rand.New(rand.NewSource(32)))
	err := runResNet9("webgpu",
		TrainConfig{Steps: 0, LR: cmdTrainSGDDefaultLR, Batch: 2},
		func(int, float32) {}, ds, [4]int64{2, 4, 8, 16}, 1)
	if err != nil {
		t.Fatalf("runResNet9 (sentinel LR): %v", err)
	}
}
