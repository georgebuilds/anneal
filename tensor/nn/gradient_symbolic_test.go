package nn_test

// TestDynBatchMLPGradient is the Slice 3b-3 deliverable test.
//
// It proves that autodiff works over a symbolic batch dimension. The MLP
// structure is [n, 4] → Linear(4→8) → ReLU → Linear(8→2) → MSE loss.
// The batch n is symbolic; the weight/feature dims are concrete.
//
// Expected failure (before fix): panic at loss.Shape() inside gradient.go
// (shapeCache is seeded from loss.Shape() which calls cv() on each Sint).

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

const (
	symBatch  = int64(4) // concrete binding for the symbolic batch dim
	symIn     = int64(4)
	symHidden = int64(8)
	symOut    = int64(2)
)

// symMLP holds a 2-layer MLP with a symbolic batch leading dimension.
type symMLP struct {
	l1 *nn.Linear
	l2 *nn.Linear
}

func newSymMLP(a *uop.Arena) *symMLP {
	return &symMLP{
		l1: nn.NewLinear(a, symIn, symHidden, true, uop.Dtypes.Float32, "webgpu"),
		l2: nn.NewLinear(a, symHidden, symOut, true, uop.Dtypes.Float32, "webgpu"),
	}
}

func (m *symMLP) params() []*nn.Parameter {
	return append(m.l1.Params(), m.l2.Params()...)
}

func (m *symMLP) forward(x *tensor.Tensor) *tensor.Tensor {
	return m.l2.Forward(nn.ReLU(m.l1.Forward(x)))
}

// initSymMLP fills the MLP weights with deterministic He-init values.
func initSymMLP(model *symMLP, rng *rand.Rand) {
	heInitP := func(p *nn.Parameter, fanIn int) {
		std := float32(math.Sqrt(2.0 / float64(fanIn)))
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * std
		}
	}
	heInitP(model.l1.Weight, int(symIn))
	heInitP(model.l2.Weight, int(symHidden))
	for i := range model.l1.Bias.Value {
		model.l1.Bias.Value[i] = float32(rng.NormFloat64()) * 0.01
	}
	for i := range model.l2.Bias.Value {
		model.l2.Bias.Value[i] = float32(rng.NormFloat64()) * 0.01
	}
}

// buildSymLoss builds a fresh graph: symbolic-batch input → forward → MSE-sum loss.
// Returns (loss tensor, x leaf, target leaf, map[param → leaf tensor]).
func buildSymLoss(a *uop.Arena, model *symMLP, xData, tgtData []float32) (loss, x, tgt *tensor.Tensor) {
	for _, p := range model.params() {
		p.Load(a)
	}
	x = tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{symIn}, uop.Dtypes.Float32, "webgpu")
	x.SetData(xData)
	tgt = tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{symOut}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(tgtData)

	pred := model.forward(x)
	diff := pred.Sub(tgt)
	loss = diff.Mul(diff).Sum(nil, false)
	return
}

// evalSymLoss evaluates the MSE-sum loss with a given binding.
func evalSymLoss(t *testing.T, model *symMLP, xData, tgtData []float32, batch int64) float32 {
	t.Helper()
	a := uop.NewArena(65536)
	loss, _, _ := buildSymLoss(a, model, xData, tgtData)
	binding := map[string]int64{"n": batch}
	if err := tensor.RealizeWithBinding(binding, loss); err != nil {
		t.Fatalf("evalSymLoss RealizeWithBinding: %v", err)
	}
	return loss.Data()[0]
}

// ── TestDynBatchMLPGradient ───────────────────────────────────────────────────

// TestDynBatchMLPGradient proves that Backward works over a symbolic batch dim.
// It runs an FD gradient check on the symbolic-batch MLP and verifies the
// gradients match those of an identically-initialized static-batch MLP.
func TestDynBatchMLPGradient(t *testing.T) {
	requireGPU(t)

	rng := rand.New(rand.NewSource(99))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.5
		}
		return s
	}

	a0 := uop.NewArena(64)
	model := newSymMLP(a0)
	initSymMLP(model, rng)

	batchI := int(symBatch)
	xData := randSlice(batchI * int(symIn))
	tgtData := randSlice(batchI * int(symOut))
	binding := map[string]int64{"n": symBatch}

	// ── 1. Analytic gradients via Backward ────────────────────────────────────
	a := uop.NewArena(65536)
	loss, _, _ := buildSymLoss(a, model, xData, tgtData)

	paramLeaves := make([]*tensor.Tensor, len(model.params()))
	for i, p := range model.params() {
		paramLeaves[i] = p.T
	}

	grads := tensor.Backward(loss, paramLeaves)

	// Realize each gradient with the binding.
	for _, p := range model.params() {
		g, ok := grads[p.T]
		if !ok {
			t.Fatalf("no gradient for param %s", p.Name)
		}
		if err := tensor.RealizeWithBinding(binding, g); err != nil {
			t.Fatalf("RealizeWithBinding gradient: %v", err)
		}
	}

	// Collect analytic gradient data in param order.
	analytic := make(map[*nn.Parameter][]float32)
	for _, p := range model.params() {
		g := grads[p.T]
		if g == nil || g.Data() == nil {
			t.Fatalf("grad data nil for param %s", p.Name)
		}
		analytic[p] = g.Data()
	}

	t.Logf("=== SLICE 3b-3 PROOF (symbolic batch autodiff) ===")
	t.Logf("analytic grads realized for all %d params ✓", len(model.params()))

	// ── 2. FD gradient check ──────────────────────────────────────────────────
	const (
		h   = float32(1e-3)
		tol = float32(5e-3)
	)

	checkParam := func(p *nn.Parameter, name string, indices []int) {
		t.Helper()
		origVal := make([]float32, len(p.Value))
		copy(origVal, p.Value)

		maxFDErr := float32(0)
		for _, idx := range indices {
			orig := p.Value[idx]

			p.Value[idx] = orig + h
			lossPlus := evalSymLoss(t, model, xData, tgtData, symBatch)

			p.Value[idx] = orig - h
			lossMinus := evalSymLoss(t, model, xData, tgtData, symBatch)

			p.Value[idx] = orig

			fd := (lossPlus - lossMinus) / (2 * h)
			an := analytic[p][idx]
			err := float32(math.Abs(float64(fd - an)))
			if err > maxFDErr {
				maxFDErr = err
			}
			t.Logf("  %s[%d]: FD=%.5f  analytic=%.5f  diff=%.2e", name, idx, fd, an, err)
		}
		if maxFDErr > tol {
			t.Errorf("FAIL: %s max FD error %.2e > tol %.2e", name, maxFDErr, tol)
		} else {
			t.Logf("  %s FD check PASS (max err %.2e) ✓", name, maxFDErr)
		}
		copy(p.Value, origVal) // restore
	}

	checkParam(model.l1.Weight, "l1.Weight", []int{0, 1, 4, 8, 12})
	checkParam(model.l1.Bias, "l1.Bias", []int{0, 2, 5})
	checkParam(model.l2.Weight, "l2.Weight", []int{0, 3, 7, 8})
	checkParam(model.l2.Bias, "l2.Bias", []int{0, 1})

	// ── 3. Gradients match static-batch ──────────────────────────────────────
	staticGrads := staticMLPGrads(t, model, xData, tgtData)

	maxDiff := float32(0)
	for _, p := range model.params() {
		sg := staticGrads[p]
		ag := analytic[p]
		if len(sg) != len(ag) {
			t.Errorf("grad len mismatch for %s: static=%d sym=%d", p.Name, len(sg), len(ag))
			continue
		}
		for i := range sg {
			if d := float32(math.Abs(float64(sg[i] - ag[i]))); d > maxDiff {
				maxDiff = d
			}
		}
	}
	t.Logf("symbolic vs static grad max abs diff: %.2e  (want < 1e-4)", maxDiff)
	if maxDiff > 1e-4 {
		t.Errorf("FAIL: symbolic and static grads differ by %.2e > 1e-4", maxDiff)
	} else {
		t.Logf("symbolic == static grads ✓")
	}
}

// staticMLPGrads computes gradients for the same MLP using fully-concrete (static)
// tensors. The MLP uses the same weights; only the input tensors differ in how the
// batch dimension is represented (concrete vs symbolic).
func staticMLPGrads(t *testing.T, model *symMLP, xData, tgtData []float32) map[*nn.Parameter][]float32 {
	t.Helper()
	a := uop.NewArena(65536)

	for _, p := range model.params() {
		p.Load(a)
	}

	x := tensor.NewLeaf(a, []int64{symBatch, symIn}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{symBatch, symOut}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, tgtData...))

	l1 := nn.NewLinear(a, symIn, symHidden, true, uop.Dtypes.Float32, "webgpu")
	// Match weights from model (already loaded into model.l1/l2 via p.Load(a) above)
	l1.Weight.T = model.l1.Weight.T
	l1.Bias.T = model.l1.Bias.T

	l2 := nn.NewLinear(a, symHidden, symOut, true, uop.Dtypes.Float32, "webgpu")
	l2.Weight.T = model.l2.Weight.T
	l2.Bias.T = model.l2.Bias.T

	pred := l2.Forward(nn.ReLU(l1.Forward(x)))
	diff := pred.Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	paramLeaves := []*tensor.Tensor{l1.Weight.T, l1.Bias.T, l2.Weight.T, l2.Bias.T}
	grads := tensor.Backward(loss, paramLeaves)

	for _, leaf := range paramLeaves {
		if g, ok := grads[leaf]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("staticMLPGrads Realize: %v", err)
			}
		}
	}

	result := make(map[*nn.Parameter][]float32)
	params := model.params()
	for i, leaf := range paramLeaves {
		if g, ok := grads[leaf]; ok {
			result[params[i]] = g.Data()
		}
	}
	return result
}

// ── TestDynBatchTrainConvergence ──────────────────────────────────────────────

// TestDynBatchTrainConvergence proves that training a symbolic-batch MLP
// converges — the loss decreases over 200 SGD steps.
func TestDynBatchTrainConvergence(t *testing.T) {
	requireGPU(t)

	rng := rand.New(rand.NewSource(77))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.5
		}
		return s
	}

	a0 := uop.NewArena(64)
	model := newSymMLP(a0)
	initSymMLP(model, rng)

	// Fixed data for the training loop.
	const trainBatch = int64(8)
	xData := randSlice(int(trainBatch) * int(symIn))
	tgtData := randSlice(int(trainBatch) * int(symOut))
	binding := map[string]int64{"n": trainBatch}

	params := model.params()
	opt := nn.NewSGD(params, 0.02)

	// Measure initial loss.
	initLoss := evalSymLoss(t, model, xData, tgtData, trainBatch)
	t.Logf("initial loss: %.4f", initLoss)

	const steps = 200
	var finalLoss float32
	for step := 0; step < steps; step++ {
		a := uop.NewArena(65536)
		for _, p := range opt.Params {
			p.Load(a)
		}
		x := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{symIn}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		tgt := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{symOut}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(append([]float32{}, tgtData...))

		pred := model.forward(x)
		diff := pred.Sub(tgt)
		// MSE mean: 1/trainBatch · Σ(pred−tgt)²
		scale := tensor.ConstScalar(a, 1.0/float64(trainBatch), uop.Dtypes.Float32, "webgpu")
		loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		for _, p := range params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.RealizeWithBinding(binding, g); err != nil {
				t.Fatalf("step %d RealizeWithBinding grad: %v", step, err)
			}
		}
		opt.Step(grads)

		if step == steps-1 {
			finalLoss = evalSymLoss(t, model, xData, tgtData, trainBatch)
		}
	}

	ratio := finalLoss / initLoss
	t.Logf("final loss: %.4f  ratio: %.4f  (want < 0.5)", finalLoss, ratio)

	if ratio >= 0.5 {
		t.Errorf("FAIL: loss did not converge (ratio=%.4f >= 0.5)", ratio)
	} else {
		t.Logf("convergence ✓ (ratio=%.4f)", ratio)
	}
}
