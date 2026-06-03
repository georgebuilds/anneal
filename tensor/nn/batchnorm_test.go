package nn_test

// Slice R1: BatchNorm2d.
//
// Five oracles:
//
//   1. TestBatchNorm2dForwardTraining : with Weight=1, Bias=0, per-channel
//      output should have mean ~0 and variance ~1 over (N, H, W).
//   2. TestBatchNorm2dForwardEval     : eval mode uses RunningMean/Var
//      directly; output normalisation matches the running stats, not the
//      batch stats.
//   3. TestBatchNorm2dPostStep        : EMA rule running = (1-m)*running + m*batch
//      is applied to both mean and var; lastBatch* refs cleared after consumption.
//   4. TestBatchNorm2dGradCheck       : FD vs analytic on Weight and Bias.
//      Relative tolerance 1e-3.
//   5. TestBatchNorm2dGradInput       : FD vs analytic on x (the input adjoint
//      is the load-bearing path for ResNet residual blocks).

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── 1. Forward correctness, training mode ────────────────────────────────────

func TestBatchNorm2dForwardTraining(t *testing.T) {
	requireGPU(t)

	const (
		N       = int64(4)
		C       = int64(3)
		H       = int64(4)
		W       = int64(4)
		eps     = float32(1e-5)
		meanTol = float32(1e-4)
		varTol  = float32(2e-3)
	)

	a0 := uop.NewArena(64)
	bn := nn.NewBatchNorm2d(a0, C, eps, 0.1, uop.Dtypes.Float32, "webgpu")

	// Sanity-check initial param values.
	for i, w := range bn.Weight.Value {
		if w != 1.0 {
			t.Fatalf("Weight[%d] = %f, want 1.0", i, w)
		}
	}
	for i, b := range bn.Bias.Value {
		if b != 0.0 {
			t.Fatalf("Bias[%d] = %f, want 0.0", i, b)
		}
	}
	for i, r := range bn.RunningMean {
		if r != 0.0 {
			t.Fatalf("RunningMean[%d] = %f, want 0.0", i, r)
		}
	}
	for i, r := range bn.RunningVar {
		if r != 1.0 {
			t.Fatalf("RunningVar[%d] = %f, want 1.0", i, r)
		}
	}

	// Build [N, C, H, W] input with per-channel different mean and scale.
	rng := rand.New(rand.NewSource(11))
	xData := make([]float32, int(N*C*H*W))
	for c := int64(0); c < C; c++ {
		base := float32(rng.NormFloat64()) * 3.0
		scale := float32(0.5 + rng.Float64()*1.5)
		for n := int64(0); n < N; n++ {
			for h := int64(0); h < H; h++ {
				for w := int64(0); w < W; w++ {
					idx := ((n*C+c)*H+h)*W + w
					noise := float32(rng.NormFloat64())
					xData[idx] = base + scale*noise
				}
			}
		}
	}

	a := uop.NewArena(131072)
	for _, p := range bn.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))

	y := bn.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}
	yData := y.Data()

	// Per-channel: mean ~0, variance ~1.
	perChan := int64(N * H * W)
	for c := int64(0); c < C; c++ {
		var sum, sq float64
		for n := int64(0); n < N; n++ {
			for h := int64(0); h < H; h++ {
				for w := int64(0); w < W; w++ {
					idx := ((n*C+c)*H+h)*W + w
					v := float64(yData[idx])
					sum += v
					sq += v * v
				}
			}
		}
		mean := float32(sum / float64(perChan))
		variance := float32(sq/float64(perChan) - (sum/float64(perChan))*(sum/float64(perChan)))
		if math.Abs(float64(mean)) > float64(meanTol) {
			t.Fatalf("channel %d: mean=%.6f, want ~0 (tol %.0e)", c, mean, meanTol)
		}
		// Population variance is biased low by 1/N; we expect ~1 - 1/N + eps_adjust.
		// Just bound to [1-2*varTol, 1+2*varTol].
		if math.Abs(float64(variance)-1.0) > float64(varTol) {
			t.Fatalf("channel %d: variance=%.6f, want ~1 (tol %.0e)", c, variance, varTol)
		}
		t.Logf("channel %d: mean=%+.4e var=%+.4e (PASS)", c, mean, variance)
	}
}

// ── 1b. Train / Eval toggle ──────────────────────────────────────────────────

func TestBatchNorm2dTrainEvalToggle(t *testing.T) {
	a := uop.NewArena(64)
	bn := nn.NewBatchNorm2d(a, 2, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")
	if !bn.Training {
		t.Fatalf("constructed BatchNorm2d should start in training mode")
	}
	bn.Eval()
	if bn.Training {
		t.Fatalf("after Eval(), Training should be false")
	}
	bn.Train()
	if !bn.Training {
		t.Fatalf("after Train(), Training should be true")
	}
}

// ── 2. Forward correctness, eval mode ────────────────────────────────────────

func TestBatchNorm2dForwardEval(t *testing.T) {
	requireGPU(t)

	const (
		N = int64(2)
		C = int64(2)
		H = int64(2)
		W = int64(2)
	)
	a0 := uop.NewArena(64)
	bn := nn.NewBatchNorm2d(a0, C, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")

	// Seed running stats that are NOT the batch stats: this distinguishes
	// eval-mode (uses running) from train-mode (uses batch).
	bn.RunningMean[0] = 1.0
	bn.RunningMean[1] = -2.0
	bn.RunningVar[0] = 4.0 // sqrt = 2
	bn.RunningVar[1] = 9.0 // sqrt = 3
	bn.Eval()

	// Build an input whose batch mean/var differ wildly from running stats.
	xData := []float32{
		// n=0
		3, 3, 3, 3, // c=0
		0, 0, 0, 0, // c=1
		// n=1
		3, 3, 3, 3, // c=0
		0, 0, 0, 0, // c=1
	}

	a := uop.NewArena(65536)
	for _, p := range bn.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))

	y := bn.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}
	yData := y.Data()

	// Expected: y[c=0] = (3 - 1) / sqrt(4 + eps) ≈ 1.0
	//           y[c=1] = (0 - (-2)) / sqrt(9 + eps) ≈ 0.6667
	const tol = float32(1e-4)
	for i, v := range yData {
		// Channel for index i: stride layout NCHW with C=2, HW=4 → channel = (i/4) & 1.
		c := (i / 4) & 1
		var want float32
		if c == 0 {
			want = 2.0 / float32(math.Sqrt(4.0+1e-5))
		} else {
			want = 2.0 / float32(math.Sqrt(9.0+1e-5))
		}
		if absF32(v-want) > tol {
			t.Fatalf("y[%d]: got %.6f, want %.6f", i, v, want)
		}
	}
	// Eval mode should NOT have stashed a batch-stat reference.
	if err := bn.PostStep(); err != nil {
		t.Fatalf("PostStep in eval mode should be a no-op, got error: %v", err)
	}
	for c := int64(0); c < C; c++ {
		if bn.RunningMean[c] != []float32{1.0, -2.0}[c] {
			t.Fatalf("eval-mode RunningMean[%d] mutated to %f", c, bn.RunningMean[c])
		}
	}
	t.Logf("Eval-mode forward PASS — running stats drove normalisation")
}

// ── 3. PostStep EMA rule ─────────────────────────────────────────────────────

func TestBatchNorm2dPostStep(t *testing.T) {
	requireGPU(t)

	const (
		N = int64(2)
		C = int64(2)
		H = int64(2)
		W = int64(2)
	)
	a0 := uop.NewArena(64)
	bn := nn.NewBatchNorm2d(a0, C, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")

	// Capture starting running stats.
	startMean := append([]float32{}, bn.RunningMean...)
	startVar := append([]float32{}, bn.RunningVar...)

	// Build an input with per-channel stats we can compute on host.
	// Channel 0: all 4's → batch mean=4, var=0. Channel 1: 1,2,3,4 across both n.
	xData := []float32{
		// n=0
		4, 4, 4, 4,
		1, 2, 3, 4,
		// n=1
		4, 4, 4, 4,
		1, 2, 3, 4,
	}

	a := uop.NewArena(65536)
	for _, p := range bn.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))

	y := bn.Forward(x) // training mode, default
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}

	// Host-side batch stats over (N, H, W) = 8 elements per channel.
	// c=0: mean=4, var=0.
	// c=1: values are [1,2,3,4] repeated twice → mean=2.5, var=(1.5²+0.5²+0.5²+1.5²)/4=1.25.
	batchMean := []float32{4.0, 2.5}
	batchVar := []float32{0.0, 1.25}

	if err := bn.PostStep(); err != nil {
		t.Fatalf("PostStep: %v", err)
	}

	const m = float32(0.1)
	const tol = float32(1e-5)
	for c := int64(0); c < C; c++ {
		wantMean := (1-m)*startMean[c] + m*batchMean[c]
		wantVar := (1-m)*startVar[c] + m*batchVar[c]
		if absF32(bn.RunningMean[c]-wantMean) > tol {
			t.Fatalf("RunningMean[%d]: got %.6f, want %.6f", c, bn.RunningMean[c], wantMean)
		}
		if absF32(bn.RunningVar[c]-wantVar) > tol {
			t.Fatalf("RunningVar[%d]: got %.6f, want %.6f", c, bn.RunningVar[c], wantVar)
		}
	}

	// Second PostStep without an intervening Forward should be a no-op.
	beforeMean := append([]float32{}, bn.RunningMean...)
	if err := bn.PostStep(); err != nil {
		t.Fatalf("PostStep idempotent: %v", err)
	}
	for c := int64(0); c < C; c++ {
		if bn.RunningMean[c] != beforeMean[c] {
			t.Fatalf("PostStep was not idempotent on stale state: RunningMean[%d] %.6f → %.6f",
				c, beforeMean[c], bn.RunningMean[c])
		}
	}
	t.Logf("PostStep EMA rule PASS (idempotent on stale state)")
}

// ── 4. FD gradient check on Weight, Bias ─────────────────────────────────────

func TestBatchNorm2dGradCheck(t *testing.T) {
	requireGPU(t)

	const (
		N      = int64(2)
		C      = int64(3)
		H      = int64(2)
		W      = int64(2)
		eps    = float32(1e-3)
		tol    = float32(2e-3)
		nCheck = 4
	)

	// Stable input with mean ~0 and small spread per channel — chosen so
	// the batch-norm statistics don't degenerate.
	rng := rand.New(rand.NewSource(7))
	xData := make([]float32, int(N*C*H*W))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}

	// Initial Weight, Bias values: perturb away from the identity (1, 0)
	// so the FD signal is non-trivial.
	wInit := []float32{1.1, 0.9, 1.05}
	bInit := []float32{0.0, 0.1, -0.05}

	lossFn := func(wOverride, bOverride []float32) float32 {
		a := uop.NewArena(131072)
		bn := nn.NewBatchNorm2d(a, C, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")
		copy(bn.Weight.Value, wOverride)
		copy(bn.Bias.Value, bOverride)
		for _, p := range bn.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		out := bn.Forward(x)
		loss := out.Mul(out).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("Realize loss: %v", err)
		}
		return loss.Data()[0]
	}

	// Analytic gradients via Backward.
	a := uop.NewArena(131072)
	bn := nn.NewBatchNorm2d(a, C, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")
	copy(bn.Weight.Value, wInit)
	copy(bn.Bias.Value, bInit)
	for _, p := range bn.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	out := bn.Forward(x)
	loss := out.Mul(out).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{bn.Weight.T, bn.Bias.T})
	if err := tensor.Realize(grads[bn.Weight.T]); err != nil {
		t.Fatalf("Realize Weight grad: %v", err)
	}
	if err := tensor.Realize(grads[bn.Bias.T]); err != nil {
		t.Fatalf("Realize Bias grad: %v", err)
	}
	gW := grads[bn.Weight.T].Data()
	gB := grads[bn.Bias.T].Data()

	t.Logf("BatchNorm2d FD gradient check  (tol=%.0e, eps=%.0e):", tol, eps)
	checkParam := func(name string, init []float32, ag []float32) {
		n := len(init)
		if n > nCheck {
			n = nCheck
		}
		for i := 0; i < n; i++ {
			orig := init[i]

			init[i] = orig + eps
			lp := lossFn(wInit, bInit)
			init[i] = orig - eps
			lm := lossFn(wInit, bInit)
			init[i] = orig

			fd := (lp - lm) / (2 * eps)
			diff := absF32(ag[i] - fd)
			pass := diff <= tol
			mark := "PASS"
			if !pass {
				mark = "FAIL"
			}
			t.Logf("  %s[%d]: analytic=%+.6f fd=%+.6f diff=%.2e %s",
				name, i, ag[i], fd, diff, mark)
			if !pass {
				t.Fatalf("FAIL %s[%d]: analytic=%.6f fd=%.6f diff=%.2e > tol=%.2e",
					name, i, ag[i], fd, diff, tol)
			}
		}
	}
	checkParam("Weight", wInit, gW)
	checkParam("Bias", bInit, gB)
}

// ── 5. FD gradient check on input ────────────────────────────────────────────

func TestBatchNorm2dGradInput(t *testing.T) {
	requireGPU(t)

	const (
		N      = int64(2)
		C      = int64(2)
		H      = int64(2)
		W      = int64(2)
		eps    = float32(1e-3)
		tol    = float32(5e-3)
		nCheck = 4
	)

	rng := rand.New(rand.NewSource(13))
	xData := make([]float32, int(N*C*H*W))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.7
	}

	lossFn := func(xOverride []float32) float32 {
		a := uop.NewArena(131072)
		bn := nn.NewBatchNorm2d(a, C, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")
		bn.Weight.Value[0] = 1.2
		bn.Weight.Value[1] = 0.8
		for _, p := range bn.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xOverride...))
		out := bn.Forward(x)
		loss := out.Mul(out).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("Realize loss: %v", err)
		}
		return loss.Data()[0]
	}

	a := uop.NewArena(131072)
	bn := nn.NewBatchNorm2d(a, C, 1e-5, 0.1, uop.Dtypes.Float32, "webgpu")
	bn.Weight.Value[0] = 1.2
	bn.Weight.Value[1] = 0.8
	for _, p := range bn.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	out := bn.Forward(x)
	loss := out.Mul(out).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{x})
	if err := tensor.Realize(grads[x]); err != nil {
		t.Fatalf("Realize x grad: %v", err)
	}
	ag := grads[x].Data()

	t.Logf("BatchNorm2d input-grad FD check  (tol=%.0e, eps=%.0e):", tol, eps)
	for i := 0; i < nCheck; i++ {
		orig := xData[i]
		xData[i] = orig + eps
		lp := lossFn(xData)
		xData[i] = orig - eps
		lm := lossFn(xData)
		xData[i] = orig

		fd := (lp - lm) / (2 * eps)
		diff := absF32(ag[i] - fd)
		pass := diff <= tol
		mark := "PASS"
		if !pass {
			mark = "FAIL"
		}
		t.Logf("  x[%d]: analytic=%+.6f fd=%+.6f diff=%.2e %s",
			i, ag[i], fd, diff, mark)
		if !pass {
			t.Fatalf("FAIL x[%d]: analytic=%.6f fd=%.6f diff=%.2e > tol=%.2e",
				i, ag[i], fd, diff, tol)
		}
	}
}
