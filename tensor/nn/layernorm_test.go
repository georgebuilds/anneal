package nn_test

// Wave 2 Slice I: LayerNorm.
//
// Three oracles:
//
//   1. TestLayerNormForward      : output rows have mean ~0 and variance ~1
//                                  when Weight=1, Bias=0.
//   2. TestLayerNormGradCheck    : finite-difference vs analytic gradient on
//                                  Weight, Bias, and the input. Relative
//                                  tolerance 1e-3.
//   3. TestLayerNormDeterminism  : fixed seed produces bit-identical output
//                                  across 3 fresh runs (sha256).

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── 1. Forward correctness ───────────────────────────────────────────────────

// TestLayerNormForward sets up LayerNorm(d=8, eps=1e-5) with Weight=ones and
// Bias=zeros, then feeds a [4, 8] input with non-trivial row mean and variance.
// With Weight=1 and Bias=0 the output should have per-row mean ~0 and per-row
// variance ~1 (within 1e-5, allowing for float32 reduction noise).
func TestLayerNormForward(t *testing.T) {
	requireGPU(t)

	const (
		rows    = int64(4)
		d       = int64(8)
		eps     = float32(1e-5)
		meanTol = float32(1e-4)
		varTol  = float32(1e-3)
	)

	a0 := uop.NewArena(64)
	ln := nn.NewLayerNorm(a0, d, eps)
	// Weight defaults to ones, Bias to zeros; assert it before use.
	for i, w := range ln.Weight.Value {
		if w != 1.0 {
			t.Fatalf("Weight[%d] = %f, want 1.0", i, w)
		}
	}
	for i, b := range ln.Bias.Value {
		if b != 0.0 {
			t.Fatalf("Bias[%d] = %f, want 0.0", i, b)
		}
	}

	// Build a [4, 8] input where each row has a known (different) mean and scale.
	// Row r has values: base[r] + scale[r] * (i - 3.5), for i in [0, 8).
	// This gives per-row mean = base[r] and stable non-zero variance.
	rng := rand.New(rand.NewSource(11))
	xData := make([]float32, int(rows*d))
	for r := int64(0); r < rows; r++ {
		base := float32(rng.NormFloat64()) * 3.0
		scale := float32(0.5 + rng.Float64()*1.5)
		for i := int64(0); i < d; i++ {
			xData[r*d+i] = base + scale*(float32(i)-3.5)
		}
	}

	a := uop.NewArena(65536)
	for _, p := range ln.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{rows, d}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))

	y := ln.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}

	out := y.Data()
	if int64(len(out)) != rows*d {
		t.Fatalf("output length %d != %d", len(out), rows*d)
	}

	// Verify per-row mean ~0 and per-row variance ~1.
	for r := int64(0); r < rows; r++ {
		var sum float64
		for i := int64(0); i < d; i++ {
			sum += float64(out[r*d+i])
		}
		mean := sum / float64(d)
		var ssq float64
		for i := int64(0); i < d; i++ {
			diff := float64(out[r*d+i]) - mean
			ssq += diff * diff
		}
		variance := ssq / float64(d)

		t.Logf("row %d: mean=%.6e variance=%.6f", r, mean, variance)
		if math.Abs(mean) > float64(meanTol) {
			t.Errorf("row %d mean = %.3e, want |mean| < %.0e", r, mean, meanTol)
		}
		if math.Abs(variance-1.0) > float64(varTol) {
			t.Errorf("row %d variance = %.6f, want |var - 1| < %.0e", r, variance, varTol)
		}
	}
}

// ── 2. FD gradient check on Weight, Bias, and input ──────────────────────────

// TestLayerNormGradCheck runs central-difference finite differences vs the
// analytic gradient produced by tensor.Backward, for all three differentiable
// inputs to LayerNorm: Weight, Bias, and the input tensor x.
//
// Loss = sum(LayerNorm(x)). Tiny config (d=4 over [2,4]) to keep the check fast.
//
// Tolerance: 1e-3 relative (matches the slice spec). LayerNorm's gradient passes
// through sqrt and reciprocal which amplify FD rounding error, so we require
// either absDiff < atol OR relDiff < rtol (the standard MLP-grad-check pattern).
func TestLayerNormGradCheck(t *testing.T) {
	requireGPU(t)

	const (
		rows = int64(2)
		d    = int64(4)
		eps  = float32(1e-5)
		h    = float32(5e-3)
		atol = float32(1e-3)
		rtol = float32(1e-3)
	)

	// Build LayerNorm with non-trivial Weight, Bias so all three gradients are
	// non-trivial. Use a fixed seed for reproducibility.
	a0 := uop.NewArena(64)
	ln := nn.NewLayerNorm(a0, d, eps)
	rng := rand.New(rand.NewSource(7))
	for i := range ln.Weight.Value {
		ln.Weight.Value[i] = 0.75 + float32(rng.NormFloat64())*0.25
	}
	for i := range ln.Bias.Value {
		ln.Bias.Value[i] = float32(rng.NormFloat64()) * 0.1
	}

	// Input values are away from constant so variance is well-defined.
	xData := make([]float32, int(rows*d))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}

	// evalLoss computes sum(LayerNorm(xData)) on the GPU using the current
	// ln.Weight.Value and ln.Bias.Value. A fresh arena per call ensures each
	// evaluation starts from p.Value (no stale leaf reuse).
	evalLoss := func(xLocal []float32) float32 {
		a := uop.NewArena(65536)
		for _, p := range ln.Params() {
			p.Load(a)
		}
		xt := tensor.NewLeaf(a, []int64{rows, d}, uop.Dtypes.Float32, "webgpu")
		xt.SetData(append([]float32{}, xLocal...))
		y := ln.Forward(xt)
		loss := y.Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("evalLoss Realize: %v", err)
		}
		return loss.Data()[0]
	}

	// ── Analytic gradients ──────────────────────────────────────────────────
	a := uop.NewArena(65536)
	for _, p := range ln.Params() {
		p.Load(a)
	}
	xLeaf := tensor.NewLeaf(a, []int64{rows, d}, uop.Dtypes.Float32, "webgpu")
	xLeaf.SetData(append([]float32{}, xData...))

	y := ln.Forward(xLeaf)
	loss := y.Sum(nil, false)

	leaves := []*tensor.Tensor{ln.Weight.T, ln.Bias.T, xLeaf}
	grads := tensor.Backward(loss, leaves)
	for _, leaf := range leaves {
		g, ok := grads[leaf]
		if !ok {
			t.Fatalf("no gradient for leaf %p", leaf)
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("Realize grad: %v", err)
		}
	}

	wGrad := append([]float32{}, grads[ln.Weight.T].Data()...)
	bGrad := append([]float32{}, grads[ln.Bias.T].Data()...)
	xGrad := append([]float32{}, grads[xLeaf].Data()...)

	if len(wGrad) != int(d) || len(bGrad) != int(d) || len(xGrad) != int(rows*d) {
		t.Fatalf("grad shape mismatch: |wGrad|=%d |bGrad|=%d |xGrad|=%d",
			len(wGrad), len(bGrad), len(xGrad))
	}

	// FD helpers: perturb a single element, recompute loss, central diff.
	fdParam := func(p *nn.Parameter, idx int) float32 {
		orig := p.Value[idx]
		p.Value[idx] = orig + h
		lp := evalLoss(xData)
		p.Value[idx] = orig - h
		lm := evalLoss(xData)
		p.Value[idx] = orig
		return (lp - lm) / (2 * h)
	}
	fdInput := func(idx int) float32 {
		orig := xData[idx]
		xData[idx] = orig + h
		lp := evalLoss(xData)
		xData[idx] = orig - h
		lm := evalLoss(xData)
		xData[idx] = orig
		return (lp - lm) / (2 * h)
	}

	checkOne := func(label string, idx int, analytic, fd float32) {
		t.Helper()
		absDiff := absF32(analytic - fd)
		relDiff := absDiff / (absF32(fd) + 1e-7)
		t.Logf("%s[%d]: analytic=%.6f fd=%.6f absDiff=%.2e relDiff=%.2e",
			label, idx, analytic, fd, absDiff, relDiff)
		if absDiff > atol && relDiff > rtol {
			t.Errorf("%s[%d]: analytic=%.6f fd=%.6f absDiff=%.2e relDiff=%.2e (atol=%.0e rtol=%.0e)",
				label, idx, analytic, fd, absDiff, relDiff, atol, rtol)
		}
	}

	// Check every element (the tensor is tiny: d=4, x=8 elements).
	for i := 0; i < int(d); i++ {
		checkOne("Weight", i, wGrad[i], fdParam(ln.Weight, i))
	}
	for i := 0; i < int(d); i++ {
		checkOne("Bias", i, bGrad[i], fdParam(ln.Bias, i))
	}
	for i := 0; i < int(rows*d); i++ {
		checkOne("x", i, xGrad[i], fdInput(i))
	}
}

// ── 3. Determinism ───────────────────────────────────────────────────────────

// TestLayerNormDeterminism runs the same forward pass 3 times from scratch with
// a fixed seed and verifies the sha256 of the output bytes is identical on all
// three runs. This guards against non-determinism leaking in via op ordering,
// reduction order, or scheduler choices.
func TestLayerNormDeterminism(t *testing.T) {
	requireGPU(t)

	const (
		rows = int64(4)
		d    = int64(8)
		eps  = float32(1e-5)
	)

	run := func() [32]byte {
		t.Helper()
		a0 := uop.NewArena(64)
		ln := nn.NewLayerNorm(a0, d, eps)

		// Fixed seed → same Weight, Bias, input every run.
		rng := rand.New(rand.NewSource(123))
		for i := range ln.Weight.Value {
			ln.Weight.Value[i] = 0.5 + float32(rng.NormFloat64())*0.3
		}
		for i := range ln.Bias.Value {
			ln.Bias.Value[i] = float32(rng.NormFloat64()) * 0.2
		}
		xData := make([]float32, int(rows*d))
		for i := range xData {
			xData[i] = float32(rng.NormFloat64())
		}

		a := uop.NewArena(65536)
		for _, p := range ln.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{rows, d}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		y := ln.Forward(x)
		if err := tensor.Realize(y); err != nil {
			t.Fatalf("Realize: %v", err)
		}

		out := y.Data()
		buf := make([]byte, 4*len(out))
		for i, v := range out {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		return sha256.Sum256(buf)
	}

	h1 := run()
	h2 := run()
	h3 := run()

	t.Logf("run 1 sha256 = %x", h1)
	t.Logf("run 2 sha256 = %x", h2)
	t.Logf("run 3 sha256 = %x", h3)

	if h1 != h2 {
		t.Fatalf("run 1 vs run 2 differ: %x vs %x", h1, h2)
	}
	if h2 != h3 {
		t.Fatalf("run 2 vs run 3 differ: %x vs %x", h2, h3)
	}
}
