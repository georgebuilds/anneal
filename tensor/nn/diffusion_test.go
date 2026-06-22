package nn_test

// Slice 1: tiny DDPM denoiser.
//
// Gates:
//
//   1. TestSiLUFDGrad                    : SiLU FD-grad vs analytic Backward.
//   2. TestSinusoidalTimeEmbedShapeAndValues : numeric oracle vs host-side
//                                              reference for a few t and a small
//                                              embedDim.
//   3. TestMakeLinearBetasMonotonic      : linear schedule endpoints + monotonicity.
//   4. TestMakeAlphaBarsDecreasing       : ᾱ strictly decreasing from 1-β[0].
//   5. TestDDPMDenoiserForwardShape      : Forward returns the input shape.
//   6. TestDDPMDenoiserParamsCount       : Params() count = 10 (5 W + 5 B).
//   7. TestDDPMDenoiserFDGrad            : end-to-end FD-grad on Conv1.Weight
//                                          (tol 1e-2, lax - many composed ops).
//   8. TestDDPMLossDecreasesOverSteps    : Adam loop drives MSE loss to <= 0.5x
//                                          initial value within 200 steps.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── 1. SiLU FD-grad ───────────────────────────────────────────────────────────

func TestSiLUFDGrad(t *testing.T) {
	requireGPU(t)

	const (
		R      = int64(3)
		C      = int64(4)
		eps    = float32(1e-3)
		tol    = float32(1e-3)
		nCheck = 5
	)

	rng := rand.New(rand.NewSource(17))
	xData := make([]float32, R*C)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.7
	}

	lossFn := func(xs []float32) float32 {
		a := uop.NewArena(1 << 16)
		x := tensor.NewLeaf(a, []int64{R, C}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xs...))
		y := nn.SiLU(x)
		loss := y.Mul(y).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("Realize loss: %v", err)
		}
		return loss.Data()[0]
	}

	// Analytic gradient via Backward.
	a := uop.NewArena(1 << 16)
	x := tensor.NewLeaf(a, []int64{R, C}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	y := nn.SiLU(x)
	loss := y.Mul(y).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{x})
	if err := tensor.Realize(grads[x]); err != nil {
		t.Fatalf("Realize x grad: %v", err)
	}
	ag := grads[x].Data()

	t.Logf("SiLU FD-grad check  (tol=%.0e, eps=%.0e):", tol, eps)
	var maxDiff float32
	for i := 0; i < nCheck; i++ {
		orig := xData[i]
		xData[i] = orig + eps
		lp := lossFn(xData)
		xData[i] = orig - eps
		lm := lossFn(xData)
		xData[i] = orig

		fd := (lp - lm) / (2 * eps)
		diff := absF32(ag[i] - fd)
		if diff > maxDiff {
			maxDiff = diff
		}
		pass := diff <= tol
		mark := "PASS"
		if !pass {
			mark = "FAIL"
		}
		t.Logf("  x[%d]: analytic=%+.6f fd=%+.6f diff=%.2e %s", i, ag[i], fd, diff, mark)
		if !pass {
			t.Fatalf("FAIL x[%d]: analytic=%.6f fd=%.6f diff=%.2e > tol=%.2e", i, ag[i], fd, diff, tol)
		}
	}
	t.Logf("SiLU FD-grad ✓  max diff=%.2e (tol=%.0e)", maxDiff, tol)
}

// ── 2. Sinusoidal time embedding shape + values ──────────────────────────────

func TestSinusoidalTimeEmbedShapeAndValues(t *testing.T) {
	// CPU-only - purely host-side construction + SetData.
	const embedDim = int64(8)
	tVals := []int32{0, 1, 5}
	a := uop.NewArena(1 << 14)
	emb := nn.SinusoidalTimeEmbed(a, tVals, embedDim, uop.Dtypes.Float32, "webgpu")

	sh := emb.Shape()
	if len(sh) != 2 || sh[0] != int64(len(tVals)) || sh[1] != embedDim {
		t.Fatalf("emb shape: got %v, want [%d, %d]", sh, len(tVals), embedDim)
	}

	got := emb.Data()
	if int64(len(got)) != int64(len(tVals))*embedDim {
		t.Fatalf("emb data length: got %d, want %d", len(got), int64(len(tVals))*embedDim)
	}

	// Host-side oracle: identical formula to the implementation.
	half := embedDim / 2
	freqs := make([]float64, half)
	for i := int64(0); i < half; i++ {
		freqs[i] = math.Pow(10000.0, -float64(2*i)/float64(embedDim))
	}
	want := make([]float32, int64(len(tVals))*embedDim)
	for b, tv := range tVals {
		tt := float64(tv)
		for i := int64(0); i < half; i++ {
			ang := tt * freqs[i]
			want[int64(b)*embedDim+2*i] = float32(math.Sin(ang))
			want[int64(b)*embedDim+2*i+1] = float32(math.Cos(ang))
		}
	}

	const tol = float32(1e-6)
	for i := range got {
		if absF32(got[i]-want[i]) > tol {
			t.Fatalf("emb[%d]: got %.6f want %.6f", i, got[i], want[i])
		}
	}

	// Spot-check t=0: sin(0)=0, cos(0)=1. The pattern across embedDim/2 pairs
	// should be (0, 1, 0, 1, 0, 1, 0, 1).
	for i := int64(0); i < half; i++ {
		if got[2*i] != 0 || got[2*i+1] != 1 {
			t.Fatalf("t=0 row: pair %d got (%.3f, %.3f), want (0, 1)", i, got[2*i], got[2*i+1])
		}
	}
}

// ── 3. Linear beta schedule monotonicity ──────────────────────────────────────

func TestMakeLinearBetasMonotonic(t *testing.T) {
	const (
		T    = 200
		bMin = float32(1e-4)
		bMax = float32(0.02)
	)
	betas := nn.MakeLinearBetas(T, bMin, bMax)
	if len(betas) != T {
		t.Fatalf("len(betas)=%d want %d", len(betas), T)
	}
	if betas[0] != bMin {
		t.Fatalf("betas[0]=%v want %v", betas[0], bMin)
	}
	if betas[T-1] != bMax {
		t.Fatalf("betas[T-1]=%v want %v", betas[T-1], bMax)
	}
	for i := 1; i < T; i++ {
		if betas[i] <= betas[i-1] {
			t.Fatalf("betas not strictly increasing at i=%d: %v -> %v", i, betas[i-1], betas[i])
		}
	}
}

// ── 4. AlphaBars strictly decreasing ──────────────────────────────────────────

func TestMakeAlphaBarsDecreasing(t *testing.T) {
	const (
		T    = 200
		bMin = float32(1e-4)
		bMax = float32(0.02)
	)
	betas := nn.MakeLinearBetas(T, bMin, bMax)
	alphas := nn.MakeAlphas(betas)
	bars := nn.MakeAlphaBars(alphas)

	if len(bars) != T {
		t.Fatalf("len(bars)=%d want %d", len(bars), T)
	}
	want0 := 1.0 - bMin
	if absF32(bars[0]-want0) > 1e-7 {
		t.Fatalf("bars[0]=%v want %v", bars[0], want0)
	}
	for i := 1; i < T; i++ {
		if bars[i] >= bars[i-1] {
			t.Fatalf("alphaBars not strictly decreasing at i=%d: %v -> %v", i, bars[i-1], bars[i])
		}
	}
	// At T=200 with typical schedule, ᾱ_{T-1} is around 0.13, not "near zero"
	// in the sense of <1e-3. We just require it dropped substantially.
	if bars[T-1] >= bars[0]*0.5 {
		t.Fatalf("bars[T-1]=%v did not drop below half of bars[0]=%v", bars[T-1], bars[0])
	}
	t.Logf("alphaBars trajectory: bars[0]=%.6f bars[%d]=%.6f", bars[0], T-1, bars[T-1])
}

// ── 5. Forward shape ──────────────────────────────────────────────────────────

func TestDDPMDenoiserForwardShape(t *testing.T) {
	requireGPU(t)

	const (
		N            = int64(2)
		InCh         = int64(1)
		H            = int64(4)
		W            = int64(4)
		Channels     = int64(8)
		TimeEmbedDim = int64(16)
	)

	rng := rand.New(rand.NewSource(3))

	a0 := uop.NewArena(1 << 14)
	model := nn.NewDDPMDenoiser(a0, InCh, Channels, TimeEmbedDim, uop.Dtypes.Float32, "webgpu")
	seedDDPM(model, 0.1, rng)

	a := uop.NewArena(1 << 18)
	for _, p := range model.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, InCh, H, W}, uop.Dtypes.Float32, "webgpu")
	xData := make([]float32, N*InCh*H*W)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.3
	}
	x.SetData(xData)

	tVals := []int32{0, 1}
	tEmb := nn.SinusoidalTimeEmbed(a, tVals, TimeEmbedDim, uop.Dtypes.Float32, "webgpu")

	out := model.Forward(x, tEmb)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	sh := out.Shape()
	if len(sh) != 4 || sh[0] != N || sh[1] != InCh || sh[2] != H || sh[3] != W {
		t.Fatalf("output shape: got %v, want [%d, %d, %d, %d]", sh, N, InCh, H, W)
	}
}

// ── 6. Params count ───────────────────────────────────────────────────────────

func TestDDPMDenoiserParamsCount(t *testing.T) {
	a := uop.NewArena(1 << 14)
	model := nn.NewDDPMDenoiser(a, 1, 8, 16, uop.Dtypes.Float32, "webgpu")
	ps := model.Params()
	const want = 10 // 3 conv (W+B) + 2 linear (W+B)
	if len(ps) != want {
		t.Fatalf("Params(): got %d, want %d", len(ps), want)
	}
}

// ── 7. End-to-end FD gradient on Conv1.Weight ─────────────────────────────────

func TestDDPMDenoiserFDGrad(t *testing.T) {
	requireGPU(t)

	const (
		N            = int64(1)
		InCh         = int64(1)
		H            = int64(4)
		W            = int64(4)
		Channels     = int64(4)
		TimeEmbedDim = int64(8)

		eps    = float32(1e-3)
		tol    = float32(1e-2)
		nCheck = 5
	)

	rng := rand.New(rand.NewSource(29))

	// Master parameter values (persisted across loss evaluations).
	a0 := uop.NewArena(1 << 14)
	model := nn.NewDDPMDenoiser(a0, InCh, Channels, TimeEmbedDim, uop.Dtypes.Float32, "webgpu")
	seedDDPM(model, 0.1, rng)

	// Fixed input + target so the loss landscape is deterministic.
	xData := make([]float32, N*InCh*H*W)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.3
	}
	targetData := make([]float32, N*InCh*H*W)
	for i := range targetData {
		targetData[i] = float32(rng.NormFloat64()) * 0.3
	}
	tVals := []int32{2}

	buildLoss := func(returnGrads bool) (float32, []float32) {
		a := uop.NewArena(1 << 18)
		for _, p := range model.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{N, InCh, H, W}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		tgt := tensor.NewLeaf(a, []int64{N, InCh, H, W}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(append([]float32{}, targetData...))
		tEmb := nn.SinusoidalTimeEmbed(a, tVals, TimeEmbedDim, uop.Dtypes.Float32, "webgpu")

		pred := model.Forward(x, tEmb)
		diff := pred.Sub(tgt)
		loss := diff.Mul(diff).Mean(nil, false)
		if !returnGrads {
			if err := tensor.Realize(loss); err != nil {
				t.Fatalf("Realize loss: %v", err)
			}
			return loss.Data()[0], nil
		}
		grads := tensor.Backward(loss, []*tensor.Tensor{model.Conv1.Weight.T})
		g, ok := grads[model.Conv1.Weight.T]
		if !ok {
			t.Fatalf("no gradient for Conv1.Weight")
		}
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("Realize loss: %v", err)
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("Realize Conv1.Weight grad: %v", err)
		}
		out := make([]float32, len(g.Data()))
		copy(out, g.Data())
		return loss.Data()[0], out
	}

	_, ag := buildLoss(true)

	t.Logf("DDPMDenoiser FD-grad on Conv1.Weight  (tol=%.0e, eps=%.0e):", tol, eps)
	wvals := model.Conv1.Weight.Value
	// Pick a few distinct positions across the kernel.
	if len(wvals) < nCheck {
		t.Fatalf("Conv1.Weight too small for %d FD checks: %d entries", nCheck, len(wvals))
	}
	stride := len(wvals) / nCheck
	if stride < 1 {
		stride = 1
	}
	var maxDiff float32
	for k := 0; k < nCheck; k++ {
		i := k * stride
		if i >= len(wvals) {
			i = len(wvals) - 1
		}
		orig := wvals[i]
		wvals[i] = orig + eps
		lp, _ := buildLoss(false)
		wvals[i] = orig - eps
		lm, _ := buildLoss(false)
		wvals[i] = orig

		fd := (lp - lm) / (2 * eps)
		diff := absF32(ag[i] - fd)
		if diff > maxDiff {
			maxDiff = diff
		}
		pass := diff <= tol
		mark := "PASS"
		if !pass {
			mark = "FAIL"
		}
		t.Logf("  Conv1.W[%d]: analytic=%+.6f fd=%+.6f diff=%.2e %s",
			i, ag[i], fd, diff, mark)
		if !pass {
			t.Fatalf("FAIL Conv1.W[%d]: analytic=%.6f fd=%.6f diff=%.2e > tol=%.2e",
				i, ag[i], fd, diff, tol)
		}
	}
	t.Logf("DDPMDenoiser FD-grad ✓  max diff=%.2e (tol=%.0e)", maxDiff, tol)
}

// ── 8. Loss-decrease integration ──────────────────────────────────────────────

func TestDDPMLossDecreasesOverSteps(t *testing.T) {
	requireGPU(t)
	if testing.Short() {
		t.Skip("short mode: skipping multi-step DDPM training smoke")
	}

	const (
		B            = int64(2)
		InCh         = int64(1)
		H            = int64(4)
		W            = int64(4)
		Channels     = int64(8)
		TimeEmbedDim = int64(16)
		Tsteps       = 200
		nIter        = 200
		lr           = float32(1e-3)
		logEvery     = 50
		seedInit     = int64(101)
		seedTrain    = int64(202)
	)

	betas := nn.MakeLinearBetas(Tsteps, 1e-4, 0.02)
	alphas := nn.MakeAlphas(betas)
	alphaBars := nn.MakeAlphaBars(alphas)
	sqrtAlphaBar := make([]float32, Tsteps)
	sqrtOneMinusAlphaBar := make([]float32, Tsteps)
	for i := 0; i < Tsteps; i++ {
		sqrtAlphaBar[i] = float32(math.Sqrt(float64(alphaBars[i])))
		sqrtOneMinusAlphaBar[i] = float32(math.Sqrt(1.0 - float64(alphaBars[i])))
	}

	initRNG := rand.New(rand.NewSource(seedInit))
	a0 := uop.NewArena(1 << 14)
	model := nn.NewDDPMDenoiser(a0, InCh, Channels, TimeEmbedDim, uop.Dtypes.Float32, "webgpu")
	seedDDPM(model, 0.1, initRNG)

	// Fixed clean batch: per-sample distinct sinusoidal patterns.
	dataRNG := rand.New(rand.NewSource(seedInit + 1))
	x0 := make([]float32, B*InCh*H*W)
	for i := range x0 {
		x0[i] = float32(dataRNG.NormFloat64()) * 0.5
	}

	opt := nn.NewAdam(model.Params(), lr)
	stepRNG := rand.New(rand.NewSource(seedTrain))

	noiseBuf := make([]float32, B*InCh*H*W)
	xtBuf := make([]float32, B*InCh*H*W)
	tValsBuf := make([]int32, B)

	runStep := func(captureLoss bool) float32 {
		// Sample t per batch element.
		for b := int64(0); b < B; b++ {
			tValsBuf[b] = int32(stepRNG.Intn(Tsteps))
		}
		// Sample ε ~ N(0, I).
		for i := range noiseBuf {
			noiseBuf[i] = float32(stepRNG.NormFloat64())
		}
		// Build x_t host-side: √ᾱ_t · x_0 + √(1-ᾱ_t) · ε, per-sample t.
		perSample := InCh * H * W
		for b := int64(0); b < B; b++ {
			tIdx := int(tValsBuf[b])
			sa := sqrtAlphaBar[tIdx]
			so := sqrtOneMinusAlphaBar[tIdx]
			base := b * perSample
			for i := int64(0); i < perSample; i++ {
				xtBuf[base+i] = sa*x0[base+i] + so*noiseBuf[base+i]
			}
		}

		a := uop.NewArena(1 << 19)
		params := model.Params()
		for _, p := range params {
			p.Load(a)
		}
		xt := tensor.NewLeaf(a, []int64{B, InCh, H, W}, uop.Dtypes.Float32, "webgpu")
		xt.SetData(append([]float32{}, xtBuf...))
		eps := tensor.NewLeaf(a, []int64{B, InCh, H, W}, uop.Dtypes.Float32, "webgpu")
		eps.SetData(append([]float32{}, noiseBuf...))
		tEmb := nn.SinusoidalTimeEmbed(a, tValsBuf, TimeEmbedDim, uop.Dtypes.Float32, "webgpu")

		pred := model.Forward(xt, tEmb)
		diff := pred.Sub(eps)
		loss := diff.Mul(diff).Mean(nil, false)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("Realize loss: %v", err)
		}
		for _, p := range params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("Realize grad %q: %v", p.Name, err)
			}
		}
		opt.Step(grads)
		if captureLoss {
			return loss.Data()[0]
		}
		return loss.Data()[0]
	}

	loss0 := runStep(true)
	t.Logf("step %4d: loss=%.6f", 1, loss0)
	lossLast := loss0
	for step := 2; step <= nIter; step++ {
		l := runStep(true)
		lossLast = l
		if step%logEvery == 0 || step == nIter {
			t.Logf("step %4d: loss=%.6f  ratio=%.4f", step, l, l/loss0)
		}
	}

	ratio := lossLast / loss0
	if ratio > 0.5 {
		t.Fatalf("DDPM loss did not halve: loss0=%.6f loss%d=%.6f ratio=%.4f (want <= 0.5)",
			loss0, nIter, lossLast, ratio)
	}
	t.Logf("DDPM loss-decrease ✓  initial=%.6f  final=%.6f  ratio=%.4f  (<= 0.5)",
		loss0, lossLast, ratio)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// seedDDPM fills every parameter with small-normal samples. Conv weights use
// `scale`; bias values use scale*0.5.
func seedDDPM(m *nn.DDPMDenoiser, scale float32, rng *rand.Rand) {
	convs := []*nn.Conv2d{m.Conv1, m.Conv2, m.Conv3}
	for _, c := range convs {
		for i := range c.Weight.Value {
			c.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if c.Bias != nil {
			for i := range c.Bias.Value {
				c.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
	}
	lins := []*nn.Linear{m.TEmbProj1, m.TEmbProj2}
	for _, l := range lins {
		for i := range l.Weight.Value {
			l.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if l.Bias != nil {
			for i := range l.Bias.Value {
				l.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
	}
}

// ── Panic-branch coverage ────────────────────────────────────────────────────
//
// Cheap guards over the constructors / Forward fail-loud paths. CPU-only,
// no Realize, so they run under -short.

func TestMakeLinearBetasRejectsTinyT(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("MakeLinearBetas(T=1) should panic")
		}
	}()
	_ = nn.MakeLinearBetas(1, 1e-4, 0.02)
}

func TestSinusoidalTimeEmbedRejectsOddDim(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("SinusoidalTimeEmbed(embedDim=odd) should panic")
		}
	}()
	a := uop.NewArena(1 << 12)
	_ = nn.SinusoidalTimeEmbed(a, []int32{0}, 7, uop.Dtypes.Float32, "webgpu")
}

func TestSinusoidalTimeEmbedRejectsEmptyTValues(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("SinusoidalTimeEmbed(empty tValues) should panic")
		}
	}()
	a := uop.NewArena(1 << 12)
	_ = nn.SinusoidalTimeEmbed(a, nil, 8, uop.Dtypes.Float32, "webgpu")
}

func TestNewDDPMDenoiserRejectsNonPositiveDims(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("NewDDPMDenoiser(inCh=0) should panic")
		}
	}()
	a := uop.NewArena(1 << 12)
	_ = nn.NewDDPMDenoiser(a, 0, 16, 32, uop.Dtypes.Float32, "webgpu")
}

func TestDDPMDenoiserForwardRejectsRankMismatch(t *testing.T) {
	a := uop.NewArena(1 << 14)
	m := nn.NewDDPMDenoiser(a, 1, 8, 16, uop.Dtypes.Float32, "webgpu")
	xBad := tensor.NewLeaf(a, []int64{1, 1, 4}, uop.Dtypes.Float32, "webgpu") // rank 3
	tEmb := tensor.NewLeaf(a, []int64{1, 16}, uop.Dtypes.Float32, "webgpu")

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Forward(rank-3 x) should panic")
			}
		}()
		_ = m.Forward(xBad, tEmb)
	}()

	xGood := tensor.NewLeaf(a, []int64{1, 1, 4, 4}, uop.Dtypes.Float32, "webgpu")
	tBad := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu") // rank 1
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Forward(rank-1 tEmb) should panic")
			}
		}()
		_ = m.Forward(xGood, tBad)
	}()
}
