package examples

// CPU-only tests for the resnet9 example. The heavy GPU path (forward
// realize on the full 64/128/256/512 network) is acknowledged but not
// exercised here — those concerns live in tensor/nn/resnet9_test.go
// using a scaled-down channel config. This file's load-bearing checks are:
//   - the example is correctly registered with Build + Train
//   - buildResNet9 constructs a graph (no Realize) without panic
//   - trainResNet9 returns the documented gate error when the codegen
//     workstream has not enabled training
//   - initResNet9Small fills every Conv / BN / Head parameter and the
//     in-place mutation is observable on the Param.Value slices

import (
	"math/rand"
	"strings"
	"testing"

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

func TestTrainResNet9ReturnsGateError(t *testing.T) {
	// Training is gated on a WGSL codegen bug; the entry point returns a
	// descriptive error rather than running the broken backward path.
	err := trainResNet9("webgpu", TrainConfig{Steps: 1}, func(int, float32) {})
	if err == nil {
		t.Fatal("expected gating error from trainResNet9")
	}
	if !strings.Contains(err.Error(), "codegen") {
		t.Fatalf("gating error should mention codegen: %v", err)
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
