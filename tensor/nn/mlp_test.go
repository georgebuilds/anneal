package nn_test

// Phase 9b: SGD optimizer + multi-parameter training loop on a toy MLP.
//
// TestMLPGradientCheck  verifies GPU backward against finite differences on a
//                       2-layer MLP, proving multi-param Backward is numerically
//                       correct (not just monotonically decreasing).
// TestMLPConvergence    trains 2→8→1 MLP on y = x1²+x2² for 300 SGD steps and
//                       verifies that the MSE loss falls to < 50 % of its initial
//                       value on the Metal device.
//
// Design rationale — convergence tuning:
//   Loss:  MSE mean = (1/N)·Σ(pred−tgt)² — scale-independent of batch size.
//   LR:    0.02 (effective step on MSE-sum gradient = lr/N = 0.00125; safe for
//          He-initialised ReLU nets; prevents divergence at N=16).
//   Init:  He (std=√(2/fanIn)): sets pre-activation variance to 1, preventing
//          dead-ReLU starts and gradient vanishing in the first layer.
//   Steps: 300 — gives ~98 % loss reduction for a linear model at this lr/N;
//          the MLP converges similarly for this smooth quadratic target.
//
// Gradient check uses MSE-sum loss (no 1/N) so the analytic gradient and FD
// perturbations share the same scale. Training uses MSE-mean to keep lr stable.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── MLP model ─────────────────────────────────────────────────────────────────

type mlp struct {
	l1 *nn.Linear
	l2 *nn.Linear
}

func newMLP(a *uop.Arena, inSize, hiddenSize, outSize int64, dtype *uop.DType) *mlp {
	return &mlp{
		l1: nn.NewLinear(a, inSize, hiddenSize, true, dtype, "webgpu"),
		l2: nn.NewLinear(a, hiddenSize, outSize, true, dtype, "webgpu"),
	}
}

// mlpParams returns all trainable parameters in a deterministic order:
// [l1.Weight, l1.Bias, l2.Weight, l2.Bias].
func (m *mlp) mlpParams() []*nn.Parameter {
	return append(m.l1.Params(), m.l2.Params()...)
}

// Forward computes x → L1 → ReLU → L2.
// All parameters must be loaded into the current-step arena via p.Load(a) first.
func (m *mlp) Forward(x *tensor.Tensor) *tensor.Tensor {
	return m.l2.Forward(nn.ReLU(m.l1.Forward(x)))
}

// ── Weight initialisation ──────────────────────────────────────────────────────

func heInit(p *nn.Parameter, fanIn int, rng *rand.Rand) {
	std := float32(math.Sqrt(2.0 / float64(fanIn)))
	for i := range p.Value {
		p.Value[i] = float32(rng.NormFloat64()) * std
	}
}

// ── Synthetic dataset ─────────────────────────────────────────────────────────

// toyDataset returns 16 fixed samples for the task y = x1² + x2².
// x is drawn from a 4×4 grid over {−0.75, −0.25, 0.25, 0.75}².
// Targets lie in [0.125, 1.125] with mean 0.625.
func toyDataset() (xData, yData []float32) {
	pts := []float32{-0.75, -0.25, 0.25, 0.75}
	for _, x1 := range pts {
		for _, x2 := range pts {
			xData = append(xData, x1, x2)
			yData = append(yData, x1*x1+x2*x2)
		}
	}
	return // 16 samples; xData len=32, yData len=16
}

// ── GPU helpers ───────────────────────────────────────────────────────────────

const (
	mlpBatch  = int64(16)
	mlpHidden = int64(8)
)

// evalLossGPU runs a forward-only GPU pass and returns the MSE-sum loss
// (Σ(pred−tgt)²). Used for logging and FD gradient checks.
func evalLossGPU(t *testing.T, model *mlp, xData, yData []float32) float32 {
	t.Helper()
	a := uop.NewArena(65536)
	for _, p := range model.mlpParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))
	diff := model.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("evalLossGPU: %v", err)
	}
	return loss.Data()[0]
}

// trainStep performs one forward → MSE-mean loss → backward → SGD update cycle.
// A fresh arena is created each step; parameter values persist via Parameter.Value
// (the 9a design). Each gradient is realized in a separate Realize call to avoid
// multi-output buffer assignment ambiguity in the current scheduler.
func trainStep(t *testing.T, model *mlp, opt *nn.SGD, xData, yData []float32) {
	t.Helper()
	a := uop.NewArena(65536)

	for _, p := range opt.Params {
		p.Load(a)
	}

	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))

	pred := model.Forward(x)
	diff := pred.Sub(tgt)
	// MSE mean: 1/N · Σ(pred−tgt)² so gradient scale is independent of N.
	scale := tensor.ConstScalar(a, 1.0/float64(mlpBatch), uop.Dtypes.Float32, "webgpu")
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

	leaves := make([]*tensor.Tensor, len(opt.Params))
	for i, p := range opt.Params {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	// Realize each gradient in deterministic param order.
	// Separate Realize calls avoid multi-output buffer assignment ordering issues.
	for _, p := range opt.Params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("trainStep Realize grad: %v", err)
		}
	}

	opt.Step(grads)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// TestMLPGradientCheck verifies GPU backward against finite differences for a
// 2-layer MLP. Checks 4 elements each of l1.Weight and l2.Weight to prove that
// the multi-param backward is numerically correct across both layers.
func TestMLPGradientCheck(t *testing.T) {
	requireGPU(t)

	const (
		h   = float32(1e-3)
		tol = float32(5e-3)
	)

	xData, yData := toyDataset()

	a0 := uop.NewArena(64)
	model := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float32)
	rng := rand.New(rand.NewSource(7))
	heInit(model.l1.Weight, 2, rng)
	heInit(model.l2.Weight, int(mlpHidden), rng)

	// ── Analytic gradient via GPU backward ────────────────────────────────────
	a := uop.NewArena(65536)
	for _, p := range model.mlpParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))

	// MSE sum (no 1/N): FD and analytic gradient must use the same loss.
	diff := model.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	leaves := make([]*tensor.Tensor, len(model.mlpParams()))
	for i, p := range model.mlpParams() {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	// Realize each gradient separately.
	for _, p := range model.mlpParams() {
		if g, ok := grads[p.T]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("gradient check: Realize: %v", err)
			}
		}
	}

	// Capture gradient data before FD calls mutate p.T via p.Load.
	l1WGrad := append([]float32{}, grads[model.l1.Weight.T].Data()...)
	l2WGrad := append([]float32{}, grads[model.l2.Weight.T].Data()...)

	// FD helper: perturb p.Value[idx] by ±h, re-run GPU forward, central difference.
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
			d := ag - fd
			if d < 0 {
				d = -d
			}
			t.Logf("%s[%d]: analytic=%.6f  fd=%.6f  diff=%.2e", label, i, ag, fd, d)
			if d > tol {
				t.Fatalf("%s[%d]: analytic=%.6f fd=%.6f diff=%.2e > tol=%.2e",
					label, i, ag, fd, d, tol)
			}
		}
	}

	checkParam(model.l1.Weight, l1WGrad, "l1.Weight", 4)
	checkParam(model.l2.Weight, l2WGrad, "l2.Weight", 4)
	t.Logf("multi-param FD gradient check ✓  (8 elements checked, tol=%.0e)", tol)
}

// TestMLPConvergence trains a 2→8→1 MLP on y = x1²+x2² for 2000 SGD steps on
// the Metal device, verifying:
//   (a) loss reaches a plateau (< 3 % of initial, trajectory logged every 100 steps)
//   (b) per-input predictions track the true function on 4 training inputs and
//       2 held-out inputs (Pearson r > 0.97, table printed)
//
// Reference baseline for 9c conv-net training: lr=0.05, nSteps=2000, He init,
// MSE-mean loss (1/N·Σ(pred−tgt)²), effective per-sample step = lr/N = 0.003125.
func TestMLPConvergence(t *testing.T) {
	requireGPU(t)

	const (
		lr       = float32(0.05)
		nSteps   = 2000
		logEvery = 100
	)

	xData, yData := toyDataset()

	a0 := uop.NewArena(64)
	model := newMLP(a0, 2, mlpHidden, 1, uop.Dtypes.Float32)
	rng := rand.New(rand.NewSource(42))
	heInit(model.l1.Weight, 2, rng)
	heInit(model.l2.Weight, int(mlpHidden), rng)

	opt := nn.NewSGD(model.mlpParams(), lr)

	loss0 := evalLossGPU(t, model, xData, yData)
	t.Logf("step %5d: MSE-sum=%.6f", 0, loss0)

	var lossFinal float32
	for step := 1; step <= nSteps; step++ {
		trainStep(t, model, opt, xData, yData)
		if step%logEvery == 0 {
			l := evalLossGPU(t, model, xData, yData)
			t.Logf("step %5d: MSE-sum=%.6f", step, l)
			if step == nSteps {
				lossFinal = l
			}
		}
	}

	ratio := lossFinal / loss0
	t.Logf("convergence: initial=%.6f  final=%.6f  ratio=%.4f", loss0, lossFinal, ratio)
	if lossFinal >= loss0*0.03 {
		t.Fatalf("MLP did not converge: loss0=%.6f loss%d=%.6f ratio=%.4f (want ratio<0.03)",
			loss0, nSteps, lossFinal, ratio)
	}
	t.Logf("loss plateau reached ✓ (%.2f%% of initial)", ratio*100)

	// ── Per-input output table ─────────────────────────────────────────────────
	type probe struct {
		x1, x2  float32
		label   string
	}
	probes := []probe{
		{-0.75, -0.75, "train"},
		{-0.25, -0.25, "train"},
		{+0.75, +0.75, "train"},
		{+0.25, +0.75, "train"},
		{+0.50, +0.50, "held-out"},
		{+0.00, +0.00, "held-out"},
	}
	inputs := make([][2]float32, len(probes))
	trueVals := make([]float32, len(probes))
	for i, p := range probes {
		inputs[i] = [2]float32{p.x1, p.x2}
		trueVals[i] = p.x1*p.x1 + p.x2*p.x2
	}
	preds := evalPredGPU(t, model, inputs)

	const datasetMean = float32(0.625)
	t.Logf("%-22s  %7s  %7s  %7s  %7s", "input (type)", "true", "pred", "|err|", "mean|err|")
	for i, p := range probes {
		t.Logf("(%+.2f,%+.2f) %-8s  %7.4f  %7.4f  %7.4f  %7.4f",
			p.x1, p.x2, p.label, trueVals[i], preds[i],
			absF32(preds[i]-trueVals[i]), absF32(datasetMean-trueVals[i]))
	}

	r := pearsonF32(trueVals, preds)
	t.Logf("Pearson r(pred, true) = %.4f", r)
	if r < 0.97 {
		t.Fatalf("predictions do not track true function: Pearson r=%.4f < 0.97", r)
	}
	t.Logf("function fit confirmed ✓  (Pearson r=%.4f, predictions track y=x1²+x2²)", r)
}

// evalPredGPU runs a forward-only GPU pass on a batch of (x1,x2) inputs and
// returns one predicted scalar per input.  Batch size is len(inputs); the model
// handles any batch size because matmul is batch-agnostic.
func evalPredGPU(t *testing.T, model *mlp, inputs [][2]float32) []float32 {
	t.Helper()
	n := int64(len(inputs))
	xFlat := make([]float32, n*2)
	for i, inp := range inputs {
		xFlat[2*i], xFlat[2*i+1] = inp[0], inp[1]
	}
	a := uop.NewArena(65536)
	for _, p := range model.mlpParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{n, 2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(xFlat)
	pred := model.Forward(x)
	if err := tensor.Realize(pred); err != nil {
		t.Fatalf("evalPredGPU: %v", err)
	}
	// Output shape is [n, 1]; data is row-major so data[i] == pred[i,0].
	data := pred.Data()
	out := make([]float32, n)
	copy(out, data)
	return out
}

func pearsonF32(x, y []float32) float32 {
	n := float32(len(x))
	var sx, sy, sxx, syy, sxy float32
	for i := range x {
		sx += x[i]
		sy += y[i]
		sxx += x[i] * x[i]
		syy += y[i] * y[i]
		sxy += x[i] * y[i]
	}
	num := sxy - sx*sy/n
	den := float32(math.Sqrt(float64((sxx - sx*sx/n) * (syy - sy*sy/n))))
	if den == 0 {
		return 0
	}
	return num / den
}
