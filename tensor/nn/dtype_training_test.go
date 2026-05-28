package nn_test

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func checkShaderF16(t *testing.T) {
	t.Helper()
	dev, ok := tensor.DefaultExecutor.(*webgpu.Device)
	if !ok {
		t.Skip("DefaultExecutor is not webgpu.Device")
	}
	if !dev.HasShaderF16 {
		t.Skip("shader-f16 not supported")
	}
}

func combinedEvalPoints() (inputs [][2]float32, trueVals []float32) {
	xTrain, yTrain := toyDataset()
	inputs = sliceToPairs(xTrain)
	trueVals = yTrain
	// Add the same 2 held-out points as TestMLPConvergence.
	inputs = append(inputs, [2]float32{0.5, 0.5}, [2]float32{0.0, 0.0})
	trueVals = append(trueVals, 0.5, 0.0) // 0.5² + 0.5² = 0.5; 0² + 0² = 0.0
	return
}

// TestF16ForwardMatchesF32 builds two identical MLPs (f32 and f16) and verifies
// that their forward passes match within literature tolerances.
func TestF16ForwardMatchesF32(t *testing.T) {
	requireGPU(t)
	checkShaderF16(t)

	rng := rand.New(rand.NewSource(42))
	inputs, _ := combinedEvalPoints()

	a0 := uop.NewArena(128)
	mlp32 := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float32)
	heInit(mlp32.l1.Weight, 2, rng)
	heInit(mlp32.l2.Weight, int(mlpHidden), rng)
	if mlp32.l1.Bias != nil {
		for i := range mlp32.l1.Bias.Value {
			mlp32.l1.Bias.Value[i] = rng.Float32()*0.2 - 0.1
		}
	}
	if mlp32.l2.Bias != nil {
		for i := range mlp32.l2.Bias.Value {
			mlp32.l2.Bias.Value[i] = rng.Float32()*0.2 - 0.1
		}
	}

	mlp16 := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float16)
	// Copy weights from f32 MLP to f16 MLP.
	copy(mlp16.l1.Weight.Value, mlp32.l1.Weight.Value)
	copy(mlp16.l1.Bias.Value, mlp32.l1.Bias.Value)
	copy(mlp16.l2.Weight.Value, mlp32.l2.Weight.Value)
	copy(mlp16.l2.Bias.Value, mlp32.l2.Bias.Value)

	pred32 := evalPredGPU(t, mlp32, inputs)
	pred16 := evalPredGPU(t, mlp16, inputs)

	// Tolerance: atol=1e-2, rtol=1e-3 (TVM single-conv defaults).
	// Ref: TVM test_to_mixed_precision single-conv defaults.
	const atol, rtol = 1e-2, 1e-3
	var maxAbs, maxRel float64
	nPass := 0
	for i := range pred32 {
		diff := float64(absF32(pred32[i] - pred16[i]))
		rel := diff / (float64(absF32(pred32[i])) + 1e-8)
		if diff > maxAbs {
			maxAbs = diff
		}
		if rel > maxRel {
			maxRel = rel
		}
		if diff <= atol || rel <= rtol {
			nPass++
		}
	}

	t.Logf("f16 vs f32 forward: max-abs=%.2e  max-rel=%.2e  pass=%d/%d (atol=%.0e, rtol=%.0e)",
		maxAbs, maxRel, nPass, len(pred32), atol, rtol)
	if nPass < len(pred32) {
		t.Errorf("f16 forward exceeds tolerance: %d/%d pass", nPass, len(pred32))
	}
}

func sliceToPairs(data []float32) [][2]float32 {
	out := make([][2]float32, len(data)/2)
	for i := range out {
		out[i] = [2]float32{data[2*i], data[2*i+1]}
	}
	return out
}

func TestF16Convergence(t *testing.T) {
	requireGPU(t)
	checkShaderF16(t)

	const (
		lr       = float32(0.05)
		nSteps   = 2000
		logEvery = 500
	)
	xData, yData := toyDataset()

	a0 := uop.NewArena(64)
	model := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float16)
	rng := rand.New(rand.NewSource(42))
	heInit(model.l1.Weight, 2, rng)
	heInit(model.l2.Weight, int(mlpHidden), rng)

	opt := nn.NewSGD(model.mlpParams(), lr)

	loss0 := evalLossGPU(t, model, xData, yData)
	t.Logf("f16 convergence: step 0: MSE-sum=%.6f", loss0)

	var lossFinal float32
	for step := 1; step <= nSteps; step++ {
		trainStep(t, model, opt, xData, yData)
		if step%logEvery == 0 || step == nSteps {
			l := evalLossGPU(t, model, xData, yData)
			if step%logEvery == 0 {
				t.Logf("f16 convergence: step %d: MSE-sum=%.6f", step, l)
			}
			lossFinal = l
		}
	}

	ratio := lossFinal / loss0
	// Evaluation set matches TestMLPConvergence: 16 training + 2 held-out points,
	// so the Pearson can be compared directly to the f32 baseline's 0.9735.
	inputs, trueVals := combinedEvalPoints()
	preds := evalPredGPU(t, model, inputs)
	r := pearsonF32(trueVals, preds)

	t.Logf("f16 convergence: ratio=%.4f  Pearson=%.4f", ratio, r)
	// Ref: tinygrad openpilot fp16 work; expecting Pearson > 0.9.
	if r < 0.9 {
		t.Errorf("f16 training Pearson=%.4f < 0.9", r)
	}
}

func TestBF16Convergence(t *testing.T) {
	requireGPU(t)

	const (
		lr       = float32(0.05)
		nSteps   = 2000
		logEvery = 500
	)
	xData, yData := toyDataset()

	a0 := uop.NewArena(64)
	model := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.BFloat16)
	rng := rand.New(rand.NewSource(42))
	heInit(model.l1.Weight, 2, rng)
	heInit(model.l2.Weight, int(mlpHidden), rng)

	opt := nn.NewSGD(model.mlpParams(), lr)

	loss0 := evalLossGPU(t, model, xData, yData)
	t.Logf("bf16-storage convergence: step 0: MSE-sum=%.6f", loss0)

	var lossFinal float32
	for step := 1; step <= nSteps; step++ {
		trainStep(t, model, opt, xData, yData)
		if step%logEvery == 0 || step == nSteps {
			l := evalLossGPU(t, model, xData, yData)
			if step%logEvery == 0 {
				t.Logf("bf16-storage convergence: step %d: MSE-sum=%.6f", step, l)
			}
			lossFinal = l
		}
	}

	ratio := lossFinal / loss0
	// Evaluation set matches TestMLPConvergence: 16 training + 2 held-out points,
	// so the Pearson can be compared directly to the f32 baseline's 0.9735.
	inputs, trueVals := combinedEvalPoints()
	preds := evalPredGPU(t, model, inputs)
	r := pearsonF32(trueVals, preds)

	t.Logf("bf16-storage convergence: ratio=%.4f  Pearson=%.4f", ratio, r)
	// bf16-storage should be close to f32 quality (Pearson > 0.95).
	if r < 0.95 {
		t.Errorf("bf16-storage training Pearson=%.4f < 0.95", r)
	}
}

func TestQuantization(t *testing.T) {
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Float16, "cpu")
	// 1.0001 is not exactly representable in f16 (next is ~1.00098).
	x.SetData([]float32{1.0001})
	if x.Data()[0] != 1.0 {
		t.Errorf("f16 quantization failed: got %v, want 1.0", x.Data()[0])
	}

	y := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.BFloat16, "cpu")
	// bf16 has 7 bits of mantissa. 1.0 + 2^-7 is representable. 1.0 + 2^-8 is not.
	v := float32(1.0 + 1.0/128.0 + 1.0/256.0)
	y.SetData([]float32{v})
	want := float32(1.0 + 1.0/128.0)
	if y.Data()[0] != want {
		t.Errorf("bf16 quantization failed: got %v, want %v", y.Data()[0], want)
	}
}
