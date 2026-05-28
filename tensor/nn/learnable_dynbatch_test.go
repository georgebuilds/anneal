package nn_test

// TestDynBatchLearnableConvergence confirms parity between the dynamic-batch
// (symbolic-n) training path and the static-batch baseline on the learnable
// task y = x₁² + x₂².
//
// Identical setup to TestMLPConvergence: 2→8→1 MLP, lr=0.05, 2000 steps,
// He init seed 42, toyDataset (16 samples). The sole difference is inputs are
// NewSymbolicBatchInput and grads are realized via RealizeWithBinding.
//
// The 0.094 ratio from TestDynBatchTrainConvergence was a no-signal artifact
// (random Gaussian targets carry no learnable structure). On a learnable task
// the dyn-batch net should reach a ratio comparable to the static baseline.

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// evalLossSymbolic runs a forward-only pass with symbolic batch input and
// returns the MSE-sum loss (Σ(pred−tgt)²), matching the scale of evalLossGPU.
func evalLossSymbolic(t *testing.T, model *mlp, xData, yData []float32, batch int64) float32 {
	t.Helper()
	a := uop.NewArena(65536)
	for _, p := range model.mlpParams() {
		p.Load(a)
	}
	x := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))
	diff := model.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)
	binding := map[string]int64{"n": batch}
	if err := tensor.RealizeWithBinding(binding, loss); err != nil {
		t.Fatalf("evalLossSymbolic: %v", err)
	}
	return loss.Data()[0]
}

// dynTrainStep performs one forward → MSE-mean loss → backward → SGD step
// with symbolic batch input. Mirrors trainStep but uses RealizeWithBinding.
func dynTrainStep(t *testing.T, model *mlp, opt *nn.SGD, xData, yData []float32, batch int64) {
	t.Helper()
	a := uop.NewArena(65536)
	for _, p := range opt.Params {
		p.Load(a)
	}
	x := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))

	pred := model.Forward(x)
	diff := pred.Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(batch), uop.Dtypes.Float32, "webgpu")
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

	leaves := make([]*tensor.Tensor, len(opt.Params))
	for i, p := range opt.Params {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	binding := map[string]int64{"n": batch}
	for _, p := range opt.Params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		if err := tensor.RealizeWithBinding(binding, g); err != nil {
			t.Fatalf("dynTrainStep RealizeWithBinding grad: %v", err)
		}
	}
	opt.Step(grads)
}

// TestDynBatchLearnableConvergence trains a symbolic-batch 2→8→1 MLP on
// y = x₁²+x₂² with the same hyperparameters as TestMLPConvergence and
// reports dyn-batch vs static loss ratios side by side.
func TestDynBatchLearnableConvergence(t *testing.T) {
	requireGPU(t)

	const (
		lr       = float32(0.05)
		nSteps   = 2000
		logEvery = 100
	)

	xData, yData := toyDataset()

	// ── dyn-batch training ────────────────────────────────────────────────────
	a0 := uop.NewArena(64)
	dynModel := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float32)
	rng := rand.New(rand.NewSource(42))
	heInit(dynModel.l1.Weight, 2, rng)
	heInit(dynModel.l2.Weight, int(mlpHidden), rng)

	dynOpt := nn.NewSGD(dynModel.mlpParams(), lr)

	dynLoss0 := evalLossSymbolic(t, dynModel, xData, yData, mlpBatch)
	t.Logf("dyn-batch step %5d: MSE-sum=%.6f", 0, dynLoss0)

	var dynFinal float32
	for step := 1; step <= nSteps; step++ {
		dynTrainStep(t, dynModel, dynOpt, xData, yData, mlpBatch)
		if step%logEvery == 0 {
			l := evalLossSymbolic(t, dynModel, xData, yData, mlpBatch)
			t.Logf("dyn-batch step %5d: MSE-sum=%.6f", step, l)
			if step == nSteps {
				dynFinal = l
			}
		}
	}
	dynRatio := dynFinal / dynLoss0

	// ── static baseline (same init, same task) ────────────────────────────────
	a1 := uop.NewArena(64)
	staticModel := newMLP(a1, 2, mlpHidden, 1, uop.Dtypes.Float32)
	rng2 := rand.New(rand.NewSource(42))
	heInit(staticModel.l1.Weight, 2, rng2)
	heInit(staticModel.l2.Weight, int(mlpHidden), rng2)

	staticOpt := nn.NewSGD(staticModel.mlpParams(), lr)

	staticLoss0 := evalLossGPU(t, staticModel, xData, yData)
	var staticFinal float32
	for step := 1; step <= nSteps; step++ {
		trainStep(t, staticModel, staticOpt, xData, yData)
		if step == nSteps {
			staticFinal = evalLossGPU(t, staticModel, xData, yData)
		}
	}
	staticRatio := staticFinal / staticLoss0

	// ── side-by-side report ───────────────────────────────────────────────────
	t.Logf("=== PARITY REPORT ===")
	t.Logf("  static  MLP: initial=%.6f  final=%.6f  ratio=%.4f", staticLoss0, staticFinal, staticRatio)
	t.Logf("  dyn-batch:   initial=%.6f  final=%.6f  ratio=%.4f", dynLoss0, dynFinal, dynRatio)
	t.Logf("  The 0.094 from TestDynBatchTrainConvergence was a no-signal (random Gaussian) artifact.")

	if dynRatio >= 0.05 {
		t.Errorf("FAIL: dyn-batch ratio=%.4f >= 0.05 (static=%.4f)", dynRatio, staticRatio)
	} else {
		t.Logf("dyn-batch learnable convergence ✓ (ratio=%.4f, static=%.4f)", dynRatio, staticRatio)
	}
}
