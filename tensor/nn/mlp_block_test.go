package nn_test

// Wave 2 Slice K: transformer FFN block (nn.MLP) with tanh-approximant GELU.
//
// The pre-existing mlp_test.go (Phase 9b) builds a toy 2-layer MLP for SGD
// convergence proof and is unrelated to the nn.MLP module added in this slice;
// the two files coexist without name collision because the 9b file uses a
// package-local lowercase mlp struct.
//
// Tests in this file:
//   TestMLPBlockShape       : shape correctness for [B,T,nEmbd] forward pass.
//   TestGELUTanhKnownPoints : gelu_tanh oracle vs reference values, 1e-3 tol.
//                             The tanh-approximant deviates from exact erf-
//                             GELU by O(1e-4); reference values below come
//                             from the analytic formula 0.5*x*(1+tanh(
//                             sqrt(2/pi)*(x + 0.044715*x^3))).
//   TestMLPBlockGradCheck   : central-difference FD vs analytic gradient on
//                             all four parameter tensors and the input, with
//                             relative tolerance 1e-3.
//   TestMLPBlockDeterminism : sha256 of forward output bit-identical across
//                             3 runs with same RNG seed.

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

// ── Shape correctness (CPU graph only, no realize) ───────────────────────────

func TestMLPBlockShape(t *testing.T) {
	a := uop.NewArena(2048)
	m := nn.NewMLP(a, 16, uop.Dtypes.Float32, "cpu")

	// FC1 expands to 4*nEmbd=64; FC2 contracts back to nEmbd=16.
	x := tensor.NewLeaf(a, []int64{2, 8, 16}, uop.Dtypes.Float32, "cpu")
	y := m.Forward(x)

	got := y.Shape()
	want := []int64{2, 8, 16}
	if len(got) != len(want) {
		t.Fatalf("rank mismatch: got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape[%d]: got %d want %d (full got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}

	if !y.Node().Valid() {
		t.Fatal("MLP output node must be valid")
	}

	// Parameter ordering / count: [FC1.W, FC1.B, FC2.W, FC2.B].
	ps := m.Params()
	if len(ps) != 4 {
		t.Fatalf("MLP.Params(): want 4 (FC1.W, FC1.B, FC2.W, FC2.B), got %d", len(ps))
	}
}

// ── GELU known-point oracle (GPU) ────────────────────────────────────────────

// geluErfRef computes the exact (erf-based) GELU in float64 reference math:
// 0.5 * x * (1 + erf(x / sqrt(2)))
func geluErfRef(x float64) float64 {
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
}

func TestGELUKnownPoints(t *testing.T) {
	requireGPU(t)

	// Probe inputs and analytic-formula references.
	probes := []struct {
		x    float32
		want float32 // analytic exact-GELU value
	}{
		{0.0, 0.0},
		{1.0, float32(geluErfRef(1.0))},   // ≈ 0.8413
		{-1.0, float32(geluErfRef(-1.0))}, // ≈ -0.1587
		{2.0, float32(geluErfRef(2.0))},   // ≈ 1.9545
	}

	// Build a single forward pass over all probes batched as a [N] vector.
	a := uop.NewArena(4096)
	xs := make([]float32, len(probes))
	for i, p := range probes {
		xs[i] = p.x
	}
	xT := tensor.NewLeaf(a, []int64{int64(len(probes))}, uop.Dtypes.Float32, "webgpu")
	xT.SetData(xs)

	// Exercise the same erf-GELU path used by MLP.Forward by building it inline
	// against the public API (Erf + Mul/Add), equivalent to nn.MLP's internal gelu.
	yT := callGELUErf(xT)
	if err := tensor.Realize(yT); err != nil {
		t.Fatalf("realize GELU: %v", err)
	}
	got := yT.Data()

	const tol = float32(1e-3) // anneal's OpErf is a polynomial approximation
	// (~1e-7 abs error); 1e-3 keeps headroom against that and float32 rounding.
	for i, p := range probes {
		diff := got[i] - p.want
		if diff < 0 {
			diff = -diff
		}
		t.Logf("gelu(%+.2f) = %+.6f  ref=%+.6f  diff=%.2e", p.x, got[i], p.want, diff)
		if diff > tol {
			t.Fatalf("gelu(%+.2f): got %.6f want %.6f diff=%.2e > tol=%.2e",
				p.x, got[i], p.want, diff, tol)
		}
	}
}

// callGELUErf rebuilds the exact erf-GELU using the public tensor API so the
// oracle test exercises the same primitive chain MLP uses.
func callGELUErf(x *tensor.Tensor) *tensor.Tensor {
	const invSqrt2 = 0.7071067811865476 // 1/sqrt(2)
	a := x.Arena()
	sh := x.ShapeSints()
	dt := x.DType()
	dev := x.Device()
	half := tensor.FullSints(a, sh, 0.5, dt, dev)
	one := tensor.FullSints(a, sh, 1.0, dt, dev)
	kInv := tensor.FullSints(a, sh, invSqrt2, dt, dev)
	return half.Mul(x).Mul(one.Add(x.Mul(kInv).Erf()))
}

// ── FD gradient check (GPU) ──────────────────────────────────────────────────

func TestMLPBlockGradCheck(t *testing.T) {
	requireGPU(t)

	const (
		nEmbd = 8
		B     = int64(2)
		h     = float32(1e-3)
		// Relative tolerance: cubic x^3 inside GELU amplifies float32 rounding
		// for moderate inputs; 1e-3 keeps fail-loud while accepting GPU/CPU
		// FD asymmetry in the ~5e-4 range. For small-magnitude gradients
		// (|g| < atol), the absolute floor takes over to absorb float32
		// cancellation noise in the central-difference subtraction.
		relTol = float32(1e-3)
		atol   = float32(1e-4)
	)

	// Build the MLP with deterministic small weights.
	a0 := uop.NewArena(64)
	m := nn.NewMLP(a0, nEmbd, uop.Dtypes.Float32, "webgpu")
	rng := rand.New(rand.NewSource(11))
	for _, p := range m.Params() {
		for i := range p.Value {
			// Small uniform values keep the activation in a non-saturating
			// regime where FD/analytic agreement is tightest.
			p.Value[i] = float32(rng.NormFloat64()) * 0.1
		}
	}

	// Slice spec: input shape [2, 8] = [B, nEmbd].
	xData := make([]float32, B*int64(nEmbd))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}

	// Loss = sum(MLP(x)) :  gradient of sum is constant 1, so FD perturbations
	// directly probe the elementwise output sensitivity.
	evalLoss := func() float32 {
		a := uop.NewArena(65536)
		for _, p := range m.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{B, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		out := m.Forward(x)
		loss := out.Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("evalLoss: %v", err)
		}
		return loss.Data()[0]
	}

	// Analytic gradients in one backward pass.
	a := uop.NewArena(65536)
	for _, p := range m.Params() {
		p.Load(a)
	}
	xLeaf := tensor.NewLeaf(a, []int64{B, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
	xLeaf.SetData(append([]float32{}, xData...))
	out := m.Forward(xLeaf)
	loss := out.Sum(nil, false)

	leaves := []*tensor.Tensor{xLeaf}
	for _, p := range m.Params() {
		leaves = append(leaves, p.T)
	}
	grads := tensor.Backward(loss, leaves)
	for _, l := range leaves {
		if g, ok := grads[l]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("realize grad: %v", err)
			}
		}
	}

	// Snapshot analytic gradients before later FD passes mutate p.T.
	type analytic struct {
		label string
		data  []float32
	}
	xGrad := append([]float32{}, grads[xLeaf].Data()...)
	pGrads := make([]analytic, 0, 4)
	pLabels := []string{"FC1.Weight", "FC1.Bias", "FC2.Weight", "FC2.Bias"}
	ps := m.Params()
	for i, p := range ps {
		pGrads = append(pGrads, analytic{
			label: pLabels[i],
			data:  append([]float32{}, grads[p.T].Data()...),
		})
	}

	// FD for parameters: perturb p.Value[idx] by ±h, re-run forward.
	fdParam := func(p *nn.Parameter, idx int) float32 {
		orig := p.Value[idx]
		p.Value[idx] = orig + h
		lp := evalLoss()
		p.Value[idx] = orig - h
		lm := evalLoss()
		p.Value[idx] = orig
		return (lp - lm) / (2 * h)
	}

	// FD for input: perturb xData[idx] by ±h, re-run forward.
	fdInput := func(idx int) float32 {
		orig := xData[idx]
		xData[idx] = orig + h
		lp := evalLoss()
		xData[idx] = orig - h
		lm := evalLoss()
		xData[idx] = orig
		return (lp - lm) / (2 * h)
	}

	// PyTorch-style hybrid tolerance: pass when |a-b| < atol + rtol*max(|a|,|b|).
	// Equivalent to relative tolerance 1e-3 once gradients are larger than ~atol;
	// the atol floor absorbs float32 cancellation in (lp-lm) for small gradients.
	checkClose := func(label string, ag, fd float32, idx int) {
		t.Helper()
		d := ag - fd
		if d < 0 {
			d = -d
		}
		denom := float32(math.Max(math.Abs(float64(ag)), math.Abs(float64(fd))))
		bound := atol + relTol*denom
		re := float32(0)
		if denom > 0 {
			re = d / denom
		}
		t.Logf("%s[%d]: analytic=%+.6f  fd=%+.6f  rel=%.2e  abs=%.2e", label, idx, ag, fd, re, d)
		if d > bound {
			t.Fatalf("%s[%d]: analytic=%.6f fd=%.6f abs=%.2e rel=%.2e > atol=%.0e + rtol=%.0e*|%.4f|",
				label, idx, ag, fd, d, re, atol, relTol, denom)
		}
	}

	// Check up to 4 elements per tensor (16 FD evaluations + 4 for input).
	const nCheck = 4
	for i, ag := range pGrads {
		p := ps[i]
		for idx := 0; idx < nCheck && idx < len(p.Value); idx++ {
			checkClose(ag.label, ag.data[idx], fdParam(p, idx), idx)
		}
	}
	for idx := 0; idx < nCheck && idx < len(xData); idx++ {
		checkClose("input", xGrad[idx], fdInput(idx), idx)
	}
	t.Logf("FD gradient check ok (4 params + input, atol=%.0e rtol=%.0e)", atol, relTol)
}

// ── Determinism: bit-identical output across 3 runs at same seed ─────────────

func TestMLPBlockDeterminism(t *testing.T) {
	requireGPU(t)

	const (
		nEmbd = 8
		B     = int64(2)
		T     = int64(4)
		seed  = int64(20260601)
	)

	runOnce := func() []float32 {
		a0 := uop.NewArena(64)
		m := nn.NewMLP(a0, nEmbd, uop.Dtypes.Float32, "webgpu")
		rng := rand.New(rand.NewSource(seed))
		for _, p := range m.Params() {
			for i := range p.Value {
				p.Value[i] = float32(rng.NormFloat64()) * 0.1
			}
		}
		xData := make([]float32, B*T*int64(nEmbd))
		for i := range xData {
			xData[i] = float32(rng.NormFloat64()) * 0.5
		}

		a := uop.NewArena(65536)
		for _, p := range m.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{B, T, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
		x.SetData(xData)
		out := m.Forward(x)
		if err := tensor.Realize(out); err != nil {
			t.Fatalf("realize: %v", err)
		}
		return append([]float32{}, out.Data()...)
	}

	hashFloats := func(xs []float32) string {
		h := sha256.New()
		buf := make([]byte, 4)
		for _, v := range xs {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
			h.Write(buf)
		}
		return hex.EncodeToString(h.Sum(nil))
	}

	hashes := make([]string, 3)
	for i := range hashes {
		hashes[i] = hashFloats(runOnce())
		t.Logf("run %d: sha256=%s", i+1, hashes[i])
	}
	if hashes[0] != hashes[1] || hashes[1] != hashes[2] {
		t.Fatalf("non-determinism across 3 runs:\n  run1=%s\n  run2=%s\n  run3=%s",
			hashes[0], hashes[1], hashes[2])
	}
	t.Logf("determinism ✓ (3 runs bit-identical: %s)", hashes[0])
}
