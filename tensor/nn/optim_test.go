package nn_test

// Wave 1 Slice F: Adam optimizer oracles.
//
// Three tests cover the Adam contract end-to-end:
//
//	TestAdamQuadratic_Converges:  sum((w-target)²) drops by >=10000x over 200
//	                              steps; proves the basic update converges.
//	TestAdamBiasCorrection_Step1: after exactly one Step at default betas, the
//	                              per-element update equals lr*g/(|g|+eps)
//	                              because m_hat=g and v_hat=g², so the
//	                              m_hat/sqrt(v_hat) factor collapses to sign(g).
//	TestAdamDeterminism_Sha256:   three independent runs with identical seed
//	                              and identical target produce bit-identical
//	                              final weight buffers (sha256 hash equality).

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// runAdamStep mirrors train_test.go's runStep but invokes Adam instead of SGD.
// Builds a fresh graph for the loss sum((p - target)²), realizes the gradient
// on the GPU, then applies one Adam update through opt.Step.
func runAdamStep(t *testing.T, opt *nn.Adam, p *nn.Parameter, targetVals []float32) []float32 {
	t.Helper()
	n := int64(len(p.Value))

	a := uop.NewArena(65536)
	pLeaf := p.Load(a)

	tgt := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, targetVals...))

	diff := pLeaf.Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{pLeaf})
	gradTensor, ok := grads[pLeaf]
	if !ok {
		t.Fatal("Backward: no gradient for pLeaf")
	}
	if err := tensor.Realize(gradTensor); err != nil {
		t.Fatalf("Realize(grad): %v", err)
	}
	gradData := gradTensor.Data()

	// Adam.Step expects a grad map keyed by the current-step leaf. Construct a
	// throwaway map mirroring what tensor.Backward returns at the outer scope.
	opt.Step(map[*tensor.Tensor]*tensor.Tensor{pLeaf: gradTensor})
	return gradData
}

// quadraticLoss returns sum((w-target)²) for the supplied f32 vectors.
func quadraticLoss(w, target []float32) float32 {
	var s float64
	for i := range w {
		d := float64(w[i]) - float64(target[i])
		s += d * d
	}
	return float32(s)
}

// TestAdamQuadratic_Converges drives Adam on sum((w-target)²) for 200 steps and
// asserts the final loss is at most 1e-4 times the initial loss. Convergence
// for this loss is monotone with Adam's defaults; failing this bound indicates
// a math bug in the update.
func TestAdamQuadratic_Converges(t *testing.T) {
	requireGPU(t)

	const (
		nElems     = int64(8)
		nSteps     = 200
		lr         = float32(0.1)
		ratioBound = float32(1e-4) // final/initial loss ratio must be <= this
	)

	target := []float32{1, -1, 2, -2, 0.5, -0.5, 3, -3}

	seedArena := uop.NewArena(16)
	p := nn.NewParameter(seedArena, []int64{nElems}, uop.Dtypes.Float32, "webgpu")
	// Deterministic init away from target so initial loss is non-trivial.
	rng := rand.New(rand.NewSource(0xA11CE))
	for i := range p.Value {
		p.Value[i] = float32(rng.NormFloat64()) * 2
	}

	initialLoss := quadraticLoss(p.Value, target)
	if initialLoss <= 0 {
		t.Fatalf("initial loss must be positive, got %g", initialLoss)
	}

	opt := nn.NewAdam([]*nn.Parameter{p}, lr)
	for step := 0; step < nSteps; step++ {
		_ = runAdamStep(t, opt, p, target)
	}

	finalLoss := quadraticLoss(p.Value, target)
	ratio := finalLoss / initialLoss
	t.Logf("Adam quadratic: initial=%.6g final=%.6g ratio=%.3e (bound %.0e)",
		initialLoss, finalLoss, ratio, ratioBound)

	if !(finalLoss < initialLoss) {
		t.Fatalf("Adam did not reduce loss: initial=%g final=%g", initialLoss, finalLoss)
	}
	if ratio > ratioBound {
		t.Fatalf("Adam convergence too slow: final/initial=%.3e > bound %.0e",
			ratio, ratioBound)
	}
}

// TestAdamBiasCorrection_Step1 checks the analytical step-1 identity.
//
// With m₀=v₀=0 and default betas, after exactly one Step:
//
//	m  = (1-β1) g
//	v  = (1-β2) g²
//	m̂  = m / (1-β1) = g
//	v̂  = v / (1-β2) = g²
//	Δw = lr * m̂ / (sqrt(v̂) + eps) = lr * g / (|g| + eps)
//
// So Δw should equal lr*g/(|g|+eps) per element, modulo float32 roundoff.
func TestAdamBiasCorrection_Step1(t *testing.T) {
	requireGPU(t)

	const (
		nElems = int64(8)
		lr     = float32(0.05)
		relTol = float64(1e-5)
	)

	// Target chosen so gradient g = 2*(w-target) takes mixed signs and magnitudes.
	target := []float32{0, 0, 0, 0, 0, 0, 0, 0}

	seedArena := uop.NewArena(16)
	p := nn.NewParameter(seedArena, []int64{nElems}, uop.Dtypes.Float32, "webgpu")
	initVals := []float32{1.0, -2.0, 0.25, -0.125, 7.5, -7.5, 0.001, -0.001}
	copy(p.Value, initVals)

	opt := nn.NewAdam([]*nn.Parameter{p}, lr)
	// Defaults must be exactly the paper values; the analytical identity uses them.
	if opt.Beta1 != 0.9 || opt.Beta2 != 0.999 || opt.Eps != 1e-8 {
		t.Fatalf("NewAdam defaults wrong: beta1=%v beta2=%v eps=%v",
			opt.Beta1, opt.Beta2, opt.Eps)
	}

	gradData := runAdamStep(t, opt, p, target)

	// Expected per-element update under the step-1 identity.
	for i := range initVals {
		g := float64(gradData[i])
		expectedDelta := -float64(lr) * g / (math.Abs(g) + float64(opt.Eps))
		expectedW := float64(initVals[i]) + expectedDelta
		gotW := float64(p.Value[i])

		// Use a relative tolerance referenced to expected magnitude, with a small
		// absolute floor for the cases where expectedW lies close to zero.
		denom := math.Abs(expectedW)
		if denom < 1e-3 {
			denom = 1e-3
		}
		relErr := math.Abs(gotW-expectedW) / denom
		if relErr > relTol {
			t.Fatalf("bias-correction step 1 element %d: got w=%.9f want w=%.9f (g=%.6f, relErr=%.3e > %.0e)",
				i, gotW, expectedW, g, relErr, relTol)
		}
	}
	t.Logf("Adam step 1 bias correction matches lr*g/(|g|+eps) within %.0e relative", relTol)
}

// TestAdamDeterminism_Sha256 runs three identical Adam training loops and
// asserts that the final []float32 weight buffer hashes to the same sha256
// digest across all three runs. This catches non-determinism in either the
// optimizer state machine or the upstream gradient pipeline.
func TestAdamDeterminism_Sha256(t *testing.T) {
	requireGPU(t)

	const (
		nElems = int64(8)
		nSteps = 50
		lr     = float32(0.05)
		seed   = int64(0xDE7E5)
	)

	target := []float32{1, -1, 2, -2, 0.5, -0.5, 3, -3}

	runOnce := func() string {
		seedArena := uop.NewArena(16)
		p := nn.NewParameter(seedArena, []int64{nElems}, uop.Dtypes.Float32, "webgpu")
		rng := rand.New(rand.NewSource(seed))
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64())
		}

		opt := nn.NewAdam([]*nn.Parameter{p}, lr)
		for step := 0; step < nSteps; step++ {
			_ = runAdamStep(t, opt, p, target)
		}

		// Hash the final weight buffer byte-for-byte. binary.LittleEndian matches
		// the in-memory layout of []float32 on all supported platforms (amd64,
		// arm64), so the digest is stable across hosts assuming bit-equality.
		buf := make([]byte, 4*len(p.Value))
		for i, v := range p.Value {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		sum := sha256.Sum256(buf)
		return hex.EncodeToString(sum[:])
	}

	h1 := runOnce()
	h2 := runOnce()
	h3 := runOnce()
	t.Logf("Adam determinism sha256: %s", h1)

	if h1 != h2 {
		t.Fatalf("determinism failure: run1 sha=%s run2 sha=%s", h1, h2)
	}
	if h2 != h3 {
		t.Fatalf("determinism failure: run2 sha=%s run3 sha=%s", h2, h3)
	}
}
