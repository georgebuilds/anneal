package examples

import (
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// requireGPUForMoETest mirrors the per-test GPU bootstrap used by the other
// example tests (kept local so the file is self-contained).
func requireGPUForMoETest(t *testing.T) {
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

// TestRunMoEGPUConvergence trains the tiny MoE-GPT on the GPU for ~50 steps and
// asserts the loss at least halves - the end-to-end GPU proof that the router
// softmax, the dense gated combine (E expert buffers + gate fused into one
// epilogue within the 8-buffer cap), the load-balance aux loss, and the whole
// backward realize on the real device. Guarded on -short because the GPU
// JIT/compile burst is slow on the CI software renderer (lavapipe); the CPU
// full-loop + aux/gate tests cover correctness in CI.
func TestRunMoEGPUConvergence(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: GPU MoE training burst is slow on the software renderer; CPU smoke + aux/gate tests cover the logic")
	}
	requireGPUForMoETest(t)
	ds := fixtureTinyDataset()
	cfg := tinyMoEConfig(ds.VocabSize())
	cfg.SampleTokens = 4
	var losses []float32
	tcfg := TrainConfig{
		Steps:    50,
		LR:       3e-3,
		LogEvery: 50,
		Batch:    4,
		OnStep:   func(int) {},
		LogText:  func(string) {},
	}
	err := runMoE("webgpu", tcfg, func(_ int, loss float32) {
		losses = append(losses, loss)
	}, ds, cfg, 11)
	if err != nil {
		t.Fatalf("runMoE webgpu: %v", err)
	}
	if len(losses) < 2 {
		t.Fatalf("expected an initial and a final loss, got %v", losses)
	}
	start, end := losses[0], losses[len(losses)-1]
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			t.Fatalf("loss[%d] non-finite: %v", i, l)
		}
	}
	if !(end < 0.5*start) {
		t.Errorf("loss did not halve: start=%v end=%v", start, end)
	}
}

// TestRunMoECPUFullLoop runs a real forward + backward + Adam loop on the pure-Go
// CPU executor with the tiny config (2 steps), exercising the whole runMoE body:
// router softmax, all 4 experts, the dense gated combine, the aux loss added to
// cross-entropy, the per-grad Realize loop, eval-loss probes, and the final
// generated sample emitted through LogText. It runs under `go test -short`.
func TestRunMoECPUFullLoop(t *testing.T) {
	withCPU(t, func() {
		ds := fixtureTinyDataset()
		cfg := tinyMoEConfig(ds.VocabSize())
		var captured strings.Builder
		var losses []float32
		tcfg := TrainConfig{
			Steps:    2,
			LR:       0,
			LogEvery: 1,
			Batch:    2,
			OnStep:   func(int) {},
			LogText:  func(s string) { captured.WriteString(s) },
		}
		err := runMoE("cpu", tcfg, func(_ int, loss float32) {
			losses = append(losses, loss)
		}, ds, cfg, 7)
		if err != nil {
			t.Fatalf("runMoE cpu: %v", err)
		}
		// LogEvery=1, Steps=2 -> step 0 + step 1 + step 2.
		if len(losses) != 3 {
			t.Fatalf("expected 3 logged losses, got %d: %v", len(losses), losses)
		}
		for i, l := range losses {
			if l <= 0 || math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				t.Errorf("loss[%d]=%v should be a positive finite total loss", i, l)
			}
		}
		if !strings.Contains(captured.String(), "sample (") {
			t.Errorf("LogText missing generated sample header; got %q", captured.String())
		}
	})
}

// TestMoEAuxAndGateMass exercises the moeFFN routing in isolation on CPU and
// asserts the load-balance contract: the aux loss is finite and within its
// analytic [1, E] band, every expert receives nonzero average gate mass, the mean
// gate masses form a distribution (sum to 1), and the mixed output is finite with
// the right shape. With near-uniform initial gates the aux loss sits at ~1.
func TestMoEAuxAndGateMass(t *testing.T) {
	withCPU(t, func() {
		a := uop.NewArena(1 << 16)
		cfg := tinyMoEConfig(9)
		dtype := uop.Dtypes.Float32
		f := newMoEFFN(a, cfg, dtype, "cpu")

		rng := rand.New(rand.NewSource(3))
		for _, p := range f.Params() {
			for i := range p.Value {
				p.Value[i] = float32(rng.NormFloat64()) * 0.02
			}
			p.Load(a)
		}

		const B, T = int64(2), int64(3)
		x := tensor.NewLeaf(a, []int64{B, T, int64(cfg.NEmbd)}, dtype, "cpu")
		xd := make([]float32, B*T*int64(cfg.NEmbd))
		for i := range xd {
			xd[i] = float32(rng.NormFloat64()) * 0.5
		}
		x.SetData(xd)

		g := f.gates(x)
		meanGate := g.Mean([]int{0, 1}, false) // [E]
		out, aux := f.Forward(x)

		if err := tensor.Realize(out, aux, meanGate); err != nil {
			t.Fatalf("realize moeFFN: %v", err)
		}

		av := aux.Data()[0]
		if math.IsNaN(float64(av)) || math.IsInf(float64(av), 0) {
			t.Fatalf("aux loss non-finite: %v", av)
		}
		// aux = E * sum_e mean(gate_e)^2 in [1, E]: 1 at uniform, E at collapse.
		if av < 0.99 || av > float32(cfg.NExperts)+0.01 {
			t.Errorf("aux=%v outside analytic [1, %d] band", av, cfg.NExperts)
		}

		mg := meanGate.Data()
		if len(mg) != cfg.NExperts {
			t.Fatalf("meanGate length %d, want %d", len(mg), cfg.NExperts)
		}
		var sum float32
		for e, v := range mg {
			if !(v > 0) {
				t.Errorf("expert %d received zero gate mass (%v)", e, v)
			}
			sum += v
		}
		if math.Abs(float64(sum)-1.0) > 1e-4 {
			t.Errorf("mean gate masses sum to %v, want 1 (a distribution)", sum)
		}

		osh := out.Shape()
		if len(osh) != 3 || osh[0] != B || osh[1] != T || osh[2] != int64(cfg.NEmbd) {
			t.Errorf("mixed output shape %v, want [%d %d %d]", osh, B, T, cfg.NEmbd)
		}
		for i, v := range out.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("mixed output[%d] non-finite: %v", i, v)
			}
		}
	})
}
