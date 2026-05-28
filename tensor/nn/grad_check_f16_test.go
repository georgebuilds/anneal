package nn_test

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// TestFP16GradCheck verifies GPU backward against finite differences for a
// 2-layer f16 MLP.
//
// f16 FD gradient check. NOT a tight equivalence oracle —
// f16's ~1e-3 precision floor + FD's epsilon/h rounding error
// make tight gradient checks impossible. This test catches
// catastrophic regression (e.g. a gradient bug that produces
// wrong-direction or NaN gradients), not subtle f16-induced drift.
// For tight gradient correctness, run TestMLPGradientCheck
// in f32 instead.
func TestFP16GradCheck(t *testing.T) {
	requireGPU(t)
	checkShaderF16(t)

	const (
		h    = float32(1e-2)
		atol = float32(1e-1)
		rtol = float32(1e-1)
	)

	xData, yData := toyDataset()

	a0 := uop.NewArena(64)
	model := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float16)
	rng := rand.New(rand.NewSource(42))
	heInit(model.l1.Weight, 2, rng)
	heInit(model.l2.Weight, int(mlpHidden), rng)

	// ── Analytic gradient via GPU backward ────────────────────────────────────
	a := uop.NewArena(65536)
	for _, p := range model.mlpParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float16, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float16, "webgpu")
	tgt.SetData(append([]float32{}, yData...))

	diff := model.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	leaves := make([]*tensor.Tensor, len(model.mlpParams()))
	for i, p := range model.mlpParams() {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	for _, p := range model.mlpParams() {
		if g, ok := grads[p.T]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("f16 grad check: Realize: %v", err)
			}
		}
	}

	l1WGrad := append([]float32{}, grads[model.l1.Weight.T].Data()...)
	l2WGrad := append([]float32{}, grads[model.l2.Weight.T].Data()...)

	fdGrad := func(p *nn.Parameter, idx int) float32 {
		orig := p.Value[idx]
		p.Value[idx] = orig + h
		lp := evalLossGPU(t, model, xData, yData)
		p.Value[idx] = orig - h
		lm := evalLossGPU(t, model, xData, yData)
		p.Value[idx] = orig
		return (lp - lm) / (2 * h)
	}

	checkParam := func(p *nn.Parameter, analytic []float32, label string, nCheck int) {
		t.Helper()
		for i := 0; i < nCheck && i < len(analytic); i++ {
			fd := fdGrad(p, i)
			ag := analytic[i]
			absDiff := absF32(ag - fd)
			relDiff := absDiff / (absF32(fd) + 1e-7)

			t.Logf("%s[%d]: analytic=%.6f  fd=%.6f  absDiff=%.2e relDiff=%.2e", label, i, ag, fd, absDiff, relDiff)
			if absDiff > atol && relDiff > rtol {
				t.Errorf("%s[%d]: analytic=%.6f fd=%.6f absDiff=%.2e > atol=%.2e && relDiff=%.2e > rtol=%.2e",
					label, i, ag, fd, absDiff, atol, relDiff, rtol)
			}
		}
	}

	checkParam(model.l1.Weight, l1WGrad, "l1.Weight", 4)
	checkParam(model.l2.Weight, l2WGrad, "l2.Weight", 4)
}

// TestBF16GradCheck verifies GPU backward against finite differences for a
// 2-layer bf16 MLP. Since bf16 compute is f32 internally, the analytic
// gradient is high-precision f32, but it is checked against a loss function
// whose weights are quantized to bf16.
//
// bf16-storage FD gradient check. Since compute is f32, we expect
// tight equivalence (similar to f32 test) IF h is large enough to
// overcome quantization noise.
//
// Tolerance calculation:
// bf16 has a 7-bit mantissa → relative quantization step ≈ 1/128 ≈ 0.78%.
// For weight w with FD perturbations w±h at h=1e-2:
//   - The perturbed weight w±h gets quantized to bf16 storage.
//   - The quantization noise relative to 2h is O(epsilon·w / h) ≈ (0.0078·w) / 0.02 ≈ 0.4·w.
//   - Empirically, the worst observed relDiff is ~15%.
//
// We use rtol=0.3 as a catastrophic-regression guard. This is ~60x looser than
// the f32 oracle (5e-3) but calibrated to catch real gradient bugs (sign flips,
// dimension errors, NaN) while tolerating bf16's inherent quantization noise.
func TestBF16GradCheck(t *testing.T) {
	requireGPU(t)

	const (
		h    = float32(1e-2)
		atol = float32(0.3)
		rtol = float32(0.3)
	)

	xData, yData := toyDataset()

	a0 := uop.NewArena(64)
	model := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.BFloat16)
	rng := rand.New(rand.NewSource(42))
	heInit(model.l1.Weight, 2, rng)
	heInit(model.l2.Weight, int(mlpHidden), rng)

	a := uop.NewArena(65536)
	for _, p := range model.mlpParams() {
		p.Load(a)
	}
	// Use Float32 for x and tgt to minimize noise, matching evalLossGPU.
	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))

	diff := model.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	leaves := make([]*tensor.Tensor, len(model.mlpParams()))
	for i, p := range model.mlpParams() {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	for _, p := range model.mlpParams() {
		if g, ok := grads[p.T]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("bf16 grad check: Realize: %v", err)
			}
		}
	}

	l1WGrad := append([]float32{}, grads[model.l1.Weight.T].Data()...)
	l2WGrad := append([]float32{}, grads[model.l2.Weight.T].Data()...)

	fdGrad := func(p *nn.Parameter, idx int) float32 {
		orig := p.Value[idx]
		p.Value[idx] = orig + h
		lp := evalLossGPU(t, model, xData, yData)
		p.Value[idx] = orig - h
		lm := evalLossGPU(t, model, xData, yData)
		p.Value[idx] = orig
		return (lp - lm) / (2 * h)
	}

	checkParam := func(p *nn.Parameter, analytic []float32, label string, nCheck int) {
		t.Helper()
		for i := 0; i < nCheck && i < len(analytic); i++ {
			fd := fdGrad(p, i)
			ag := analytic[i]
			absDiff := absF32(ag - fd)
			relDiff := absDiff / (absF32(fd) + 1e-7)

			t.Logf("%s[%d]: analytic=%.6f  fd=%.6f  absDiff=%.2e relDiff=%.2e", label, i, ag, fd, absDiff, relDiff)
			if absDiff > atol && relDiff > rtol {
				t.Errorf("%s[%d]: analytic=%.6f fd=%.6f absDiff=%.2e > atol=%.2e && relDiff=%.2e > rtol=%.2e",
					label, i, ag, fd, absDiff, atol, relDiff, rtol)
			}
		}
	}

	checkParam(model.l1.Weight, l1WGrad, "l1.Weight", 4)
	checkParam(model.l2.Weight, l2WGrad, "l2.Weight", 4)
}
