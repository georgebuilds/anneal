package tensor_test

// JIT capture/replay tests — six proofs required by the spec.
//
// Proof 1 — Replay correctness vs non-JIT (TestJITReplayMatchesNonJIT):
//   Runs an MLP training loop in two parallel threads — one via JIT, one
//   via normal Realize — from identical initial weights.  Asserts max-abs-diff
//   == 0 between the gradient tensors on every step.  Proves that remapped
//   leaf data reaches the right buffer slots and that stale-weight replay is
//   absent.
//
// Proof 2 — Training converges under JIT (TestJITConvergence):
//   Trains the same 2→8→1 MLP via JIT for 2000 steps.  Reports loss
//   trajectory, ratio, and Pearson.  Requires ratio < 0.03 and Pearson > 0.97.
//
// Proof 3 — Replay skips scheduling (TestJITSkipsScheduling):
//   After the capture step, resets schedule cache stats, runs five replays, and
//   asserts hits == 0 AND misses == 0, proving JIT never calls CreateSchedule
//   on replay — in contrast to the schedule-cache-only path, which still calls
//   CreateSchedule (hitting the arena-local cache) every step.
//
// Proof 4 — Symbolic replay at a different bind (TestJITSymbolicReplayDifferentBatch):
//   Captures a dynamic-batch (symbolic) element-wise computation at batch=8,
//   replays at batch=32, and verifies the output against a CPU reference.
//
// Proof 5 — Mismatch falls back safely (TestJITMismatchFallback):
//   Captures a computation with leaf size L, then calls JIT with leaves of
//   different size.  Asserts that JIT re-captures (captures==2) and produces
//   the correct output rather than replaying the stale plan.
//
// Proof 6 — Value oracle unchanged: go build and full suite remain green
//   (checked externally; no test encodes this directly).

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── GPU setup ─────────────────────────────────────────────────────────────────

func requireGPUJIT(t *testing.T) {
	t.Helper()
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	t.Cleanup(func() {
		tensor.DefaultExecutor = nil
		dev.Close()
	})
	tensor.DefaultExecutor = dev
}

// ── Shared helpers ────────────────────────────────────────────────────────────

const (
	jitBatch  = int64(16)
	jitHidden = int64(8)
)

func jitToyDataset() (xData, yData []float32) {
	pts := []float32{-0.75, -0.25, 0.25, 0.75}
	for _, x1 := range pts {
		for _, x2 := range pts {
			xData = append(xData, x1, x2)
			yData = append(yData, x1*x1+x2*x2)
		}
	}
	return
}

func jitHeInit(p *nn.Parameter, fanIn int, rng *rand.Rand) {
	std := float32(math.Sqrt(2.0 / float64(fanIn)))
	for i := range p.Value {
		p.Value[i] = float32(rng.NormFloat64()) * std
	}
}

func absF32JIT(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// makeJITModel creates a fresh 2→8→1 MLP with He-initialised weights.
// The returned parameters live on seedArena; the caller should call p.Load(a)
// before each training step.
func makeJITModel(seed int64, device string) (params []*nn.Parameter, l1 *nn.Linear, l2 *nn.Linear) {
	a0 := uop.NewArena(64)
	rng := rand.New(rand.NewSource(seed))
	l1 = nn.NewLinear(a0, 2, jitHidden, true, uop.Dtypes.Float32, device)
	l2 = nn.NewLinear(a0, jitHidden, 1, true, uop.Dtypes.Float32, device)
	jitHeInit(l1.Weight, 2, rng)
	jitHeInit(l2.Weight, int(jitHidden), rng)
	params = append(l1.Params(), l2.Params()...)
	return
}

// copyParams copies the Value slice of each src param into the corresponding dst param.
func copyParams(dst, src []*nn.Parameter) {
	for i := range dst {
		copy(dst[i].Value, src[i].Value)
	}
}

// jitRunStep builds a fresh graph on a new arena, runs backward, and realizes
// each gradient using the provided realize function.  Returns the realized
// gradient data per param in parameter order.
func jitRunStep(
	t *testing.T,
	params []*nn.Parameter,
	l1 *nn.Linear, l2 *nn.Linear,
	xData, yData []float32,
	device string,
	realizeGrad func(g *tensor.Tensor) error,
) [][]float32 {
	t.Helper()
	a := uop.NewArena(65536)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{jitBatch, 2}, uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{jitBatch, 1}, uop.Dtypes.Float32, device)
	tgt.SetData(append([]float32{}, yData...))

	h := nn.ReLU(l1.Forward(x))
	pred := l2.Forward(h)
	diff := pred.Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(jitBatch), uop.Dtypes.Float32, device)
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

	leaves := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	gradData := make([][]float32, len(params))
	for i, p := range params {
		g, ok := grads[p.T]
		if !ok {
			t.Fatalf("jitRunStep: no gradient for param %d", i)
		}
		if err := realizeGrad(g); err != nil {
			t.Fatalf("jitRunStep: realize grad %d: %v", i, err)
		}
		gradData[i] = append([]float32{}, g.Data()...)
	}
	return gradData
}

// ── Proof 1: replay correctness vs non-JIT ────────────────────────────────────

// TestJITReplayMatchesNonJIT runs an MLP training loop for 10 steps.
// Each step computes the same gradient via two paths:
//   (a) JIT path (one JIT handle per parameter gradient)
//   (b) Normal tensor.Realize path
//
// Both models start from identical weights.  After each step the gradient data
// from both paths must be bit-identical (max-abs-diff == 0).  Identical
// gradients → identical SGD updates → identical weights for the next step,
// so correctness is load-bearing across all 10 steps.
//
// The crux: from step 2 onwards the JIT replays (fresh arena, new leaf
// indices).  Any wrong-buffer remap would produce non-zero diff here.
func TestJITReplayMatchesNonJIT(t *testing.T) {
	requireGPUJIT(t)

	const (
		nSteps = 10
		lr     = float32(0.02)
		device = "webgpu"
	)

	xData, yData := jitToyDataset()

	// Two models, same seed → identical initial weights.
	jitParams, jitL1, jitL2 := makeJITModel(42, device)
	refParams, refL1, refL2 := makeJITModel(42, device)

	// One JIT handle per parameter gradient (same graph structure per param
	// means one handle per param is the natural decomposition).
	jits := make([]*tensor.JIT, len(jitParams))
	for i := range jits {
		jits[i] = tensor.NewJIT()
	}

	for step := 1; step <= nSteps; step++ {
		// ── JIT path ──
		jitGrads := jitRunStepIndexed(t, jitParams, jitL1, jitL2, xData, yData, device, jits)

		// ── Ref path ──
		refGrads := jitRunStep(t, refParams, refL1, refL2, xData, yData, device,
			func(g *tensor.Tensor) error {
				return tensor.Realize(g)
			})

		// ── Compare step-for-step ──
		for i, jg := range jitGrads {
			rg := refGrads[i]
			if len(jg) != len(rg) {
				t.Fatalf("step %d param %d: JIT grad len %d != ref len %d", step, i, len(jg), len(rg))
			}
			var maxDiff float32
			for k := range jg {
				if d := absF32JIT(jg[k] - rg[k]); d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff != 0 {
				t.Fatalf("step %d param %d: JIT grad != ref grad (max-abs-diff=%.2e); "+
					"this indicates a stale-buffer or wrong-remap replay", step, i, maxDiff)
			}
		}

		// ── Apply optimizer update to both models ──
		// Grads are identical, so after Step both models have identical weights.
		applyJITGrads(jitParams, jitGrads, lr)
		applyJITGrads(refParams, refGrads, lr)

		caps, reps := sumJITStats(jits)
		t.Logf("step %3d: captures=%d replays=%d", step, caps, reps)
	}

	// After nSteps*nParams calls, step 1 captures each, remaining are replays.
	nParams := len(jitParams)
	caps, reps := sumJITStats(jits)
	t.Logf("PROOF 1: %d params × %d steps: captures=%d replays=%d  (want caps=%d reps=%d)",
		nParams, nSteps, caps, reps, int64(nParams), int64((nSteps-1)*nParams))
	if caps != int64(nParams) {
		t.Errorf("expected %d total captures, got %d", nParams, caps)
	}
	if reps != int64((nSteps-1)*nParams) {
		t.Errorf("expected %d total replays, got %d", (nSteps-1)*nParams, reps)
	}
	t.Logf("PROOF 1: step-for-step max-abs-diff == 0 over %d steps ✓", nSteps)
}

// jitRunStepIndexed is like jitRunStep but uses the indexed JIT handles.
func jitRunStepIndexed(
	t *testing.T,
	params []*nn.Parameter,
	l1 *nn.Linear, l2 *nn.Linear,
	xData, yData []float32,
	device string,
	jits []*tensor.JIT,
) [][]float32 {
	t.Helper()
	a := uop.NewArena(65536)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{jitBatch, 2}, uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{jitBatch, 1}, uop.Dtypes.Float32, device)
	tgt.SetData(append([]float32{}, yData...))

	h := nn.ReLU(l1.Forward(x))
	pred := l2.Forward(h)
	diff := pred.Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(jitBatch), uop.Dtypes.Float32, device)
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

	leaves := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	gradData := make([][]float32, len(params))
	for i, p := range params {
		g, ok := grads[p.T]
		if !ok {
			t.Fatalf("jitRunStepIndexed: no gradient for param %d", i)
		}
		if err := jits[i].Realize(g); err != nil {
			t.Fatalf("jitRunStepIndexed: JIT.Realize grad %d: %v", i, err)
		}
		gradData[i] = append([]float32{}, g.Data()...)
	}
	return gradData
}

func applyJITGrads(params []*nn.Parameter, grads [][]float32, lr float32) {
	for i, p := range params {
		p.SGDStep(grads[i], lr)
	}
}

func sumJITStats(jits []*tensor.JIT) (caps, reps int64) {
	for _, j := range jits {
		c, r := j.JITStats()
		caps += c
		reps += r
	}
	return
}

// ── Proof 2: training converges under JIT ─────────────────────────────────────

// TestJITConvergence trains a 2→8→1 MLP on y=x₁²+x₂² for 2000 SGD steps via
// the JIT path (one JIT per parameter gradient) and verifies:
//   (a) loss ratio < 0.03 (same threshold as TestMLPConvergence)
//   (b) Pearson r(pred, true) > 0.97
//   (c) total captures == nParams (one capture per param); all other calls replay
//
// These metrics confirm that JIT replay correctly picks up weight updates at
// every step — a silent stale-weight bug would prevent convergence.
func TestJITConvergence(t *testing.T) {
	requireGPUJIT(t)

	const (
		lr       = float32(0.05)
		nSteps   = 2000
		logEvery = 200
		device   = "webgpu"
	)

	xData, yData := jitToyDataset()

	params, l1, l2 := makeJITModel(42, device)
	opt := nn.NewSGD(params, lr)

	jits := make([]*tensor.JIT, len(params))
	for i := range jits {
		jits[i] = tensor.NewJIT()
	}

	forward := func(x *tensor.Tensor) *tensor.Tensor {
		return l2.Forward(nn.ReLU(l1.Forward(x)))
	}

	evalLoss := func() float32 {
		a := uop.NewArena(65536)
		for _, p := range params {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{jitBatch, 2}, uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, xData...))
		tgt := tensor.NewLeaf(a, []int64{jitBatch, 1}, uop.Dtypes.Float32, device)
		tgt.SetData(append([]float32{}, yData...))
		diff := forward(x).Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(jitBatch), uop.Dtypes.Float32, device)
		loss := diff.Mul(diff).Sum(nil, false).Mul(scale)
		if err := tensor.Realize(loss); err != nil {
			return 0
		}
		return loss.Data()[0]
	}

	loss0 := evalLoss()
	t.Logf("step %5d: MSE-mean=%.6f", 0, loss0)

	var lossFinal float32
	for step := 1; step <= nSteps; step++ {
		a := uop.NewArena(65536)
		for _, p := range opt.Params {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{jitBatch, 2}, uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, xData...))
		tgt := tensor.NewLeaf(a, []int64{jitBatch, 1}, uop.Dtypes.Float32, device)
		tgt.SetData(append([]float32{}, yData...))

		pred := forward(x)
		diff := pred.Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(jitBatch), uop.Dtypes.Float32, device)
		loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

		leaves := make([]*tensor.Tensor, len(opt.Params))
		for i, p := range opt.Params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		for i, p := range opt.Params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := jits[i].Realize(g); err != nil {
				t.Fatalf("step %d JIT.Realize: %v", step, err)
			}
		}
		opt.Step(grads)

		if step%logEvery == 0 {
			l := evalLoss()
			t.Logf("step %5d: MSE-mean=%.6f", step, l)
			if step == nSteps {
				lossFinal = l
			}
		}
	}

	ratio := lossFinal / loss0
	t.Logf("PROOF 2 convergence: initial=%.6f  final=%.6f  ratio=%.4f", loss0, lossFinal, ratio)
	if lossFinal >= loss0*0.03 {
		t.Fatalf("JIT MLP did not converge: loss0=%.6f loss%d=%.6f ratio=%.4f (want ratio<0.03)",
			loss0, nSteps, lossFinal, ratio)
	}

	// Pearson correlation on a small probe set.
	probeInputs := [][2]float32{{-0.75, -0.75}, {-0.25, -0.25}, {0.25, 0.25}, {0.75, 0.75}, {0.5, 0.5}, {0.0, 0.0}}
	trueVals := make([]float32, len(probeInputs))
	for i, p := range probeInputs {
		trueVals[i] = p[0]*p[0] + p[1]*p[1]
	}
	preds := jitEvalPreds(t, params, l1, l2, probeInputs, device)
	r := pearsonJIT(trueVals, preds)
	t.Logf("PROOF 2 Pearson r(pred, true) = %.4f", r)
	if r < 0.97 {
		t.Fatalf("JIT training: predictions do not track true function: Pearson r=%.4f < 0.97", r)
	}

	// JIT stats: exactly nParams captures, rest are replays.
	caps, reps := sumJITStats(jits)
	nParams := int64(len(params))
	t.Logf("PROOF 2 JIT stats: captures=%d replays=%d  (want caps=%d)", caps, reps, nParams)
	if caps != nParams {
		t.Errorf("expected %d captures (one per param), got %d", nParams, caps)
	}
	t.Logf("PROOF 2: JIT convergence ✓ (ratio=%.4f, Pearson=%.4f)", ratio, r)
}

func jitEvalPreds(t *testing.T, params []*nn.Parameter, l1 *nn.Linear, l2 *nn.Linear, inputs [][2]float32, device string) []float32 {
	t.Helper()
	n := int64(len(inputs))
	xFlat := make([]float32, n*2)
	for i, inp := range inputs {
		xFlat[2*i], xFlat[2*i+1] = inp[0], inp[1]
	}
	a := uop.NewArena(65536)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{n, 2}, uop.Dtypes.Float32, device)
	x.SetData(xFlat)
	pred := l2.Forward(nn.ReLU(l1.Forward(x)))
	if err := tensor.Realize(pred); err != nil {
		t.Fatalf("jitEvalPreds: %v", err)
	}
	out := make([]float32, n)
	copy(out, pred.Data())
	return out
}

func pearsonJIT(x, y []float32) float32 {
	n := float32(len(x))
	var sx, sy, sxx, syy, sxy float32
	for i := range x {
		sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; syy += y[i] * y[i]; sxy += x[i] * y[i]
	}
	num := sxy - sx*sy/n
	den := float32(math.Sqrt(float64((sxx - sx*sx/n) * (syy - sy*sy/n))))
	if den == 0 {
		return 0
	}
	return num / den
}

// ── Proof 3: replay skips scheduling ─────────────────────────────────────────

// TestJITSkipsScheduling proves that JIT.Realize never calls CreateSchedule
// on replay.  The test resets the schedule cache counters immediately after the
// capture step, then runs five replays, and asserts that both hits and misses
// remain at zero.
//
// Contrast with the arena-cache-only path: calling Realize in a training loop
// with a fresh arena each step always triggers a cache miss (no cross-arena
// cache).  JIT skips CreateSchedule entirely — no misses, no hits.
func TestJITSkipsScheduling(t *testing.T) {
	requireGPUJIT(t)

	const (
		nReplays = 5
		device   = "webgpu"
	)

	xData, yData := jitToyDataset()

	params, l1, l2 := makeJITModel(7, device)
	jit := tensor.NewJIT()

	// Build and realize a single gradient via JIT — this is the capture step.
	runOneStep := func() {
		t.Helper()
		a := uop.NewArena(65536)
		for _, p := range params {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{jitBatch, 2}, uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, xData...))
		tgt := tensor.NewLeaf(a, []int64{jitBatch, 1}, uop.Dtypes.Float32, device)
		tgt.SetData(append([]float32{}, yData...))

		pred := l2.Forward(nn.ReLU(l1.Forward(x)))
		diff := pred.Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(jitBatch), uop.Dtypes.Float32, device)
		loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

		grads := tensor.Backward(loss, []*tensor.Tensor{params[0].T})
		g := grads[params[0].T]
		if err := jit.Realize(g); err != nil {
			t.Fatalf("JIT.Realize: %v", err)
		}
	}

	// Step 1: capture.
	runOneStep()

	caps, reps := jit.JITStats()
	if caps != 1 || reps != 0 {
		t.Fatalf("after capture: want (caps=1, reps=0), got (caps=%d, reps=%d)", caps, reps)
	}

	// Reset cache counters AFTER capture so we can distinguish capture-phase
	// scheduling from replay-phase scheduling.
	schedule.ResetScheduleCache()

	// Steps 2..N+1: replays — must not call CreateSchedule at all.
	for i := 0; i < nReplays; i++ {
		runOneStep()
	}

	hits, misses := schedule.ScheduleCacheStats()
	caps2, reps2 := jit.JITStats()

	t.Logf("PROOF 3: after %d replays: schedule hits=%d misses=%d  JIT caps=%d reps=%d",
		nReplays, hits, misses, caps2, reps2)
	t.Logf("PROOF 3: cache-only path would show misses=%d (new arena per step); JIT shows 0", nReplays)

	if hits != 0 || misses != 0 {
		t.Errorf("JIT called CreateSchedule during replay: hits=%d misses=%d (want 0, 0)", hits, misses)
	}
	if reps2 != int64(nReplays) {
		t.Errorf("expected %d replays, got %d", nReplays, reps2)
	}
	t.Logf("PROOF 3: zero scheduling on replay ✓ (hits=0, misses=0 over %d replays)", nReplays)
}

// ── Proof 4: symbolic replay at a different bind ──────────────────────────────

// TestJITSymbolicReplayDifferentBatch captures a symbolic element-wise scale
// computation at batch=8, then replays it at batch=32.  Both results are
// verified against a CPU reference (max-abs-diff < 1e-5).
//
// This proves that JIT.RealizeWithBinding can replay at a different binding
// than the one used for capture: the WGSL shader reads the batch size from a
// uniform buffer, so the same compiled schedule handles both dispatches.
func TestJITSymbolicReplayDifferentBatch(t *testing.T) {
	requireGPUJIT(t)

	const (
		symVar   = "n"
		symMin   = int64(1)
		symMax   = int64(256)
		trailDim = int64(4) // trailing non-symbolic dimension
		device   = "webgpu"
	)

	// Simple computation: y = x * 3.0  where x has shape [n, trailDim].
	a := uop.NewArena(4096)
	x := tensor.NewSymbolicBatchInput(a, symVar, symMin, symMax, []int64{trailDim}, uop.Dtypes.Float32, device)
	scale := tensor.Full(a, []int64{1, trailDim}, 3.0, uop.Dtypes.Float32, device)
	y := x.Mul(scale)

	jit := tensor.NewJIT()

	cpuRef := func(data []float32) []float32 {
		out := make([]float32, len(data))
		for i, v := range data {
			out[i] = v * 3.0
		}
		return out
	}

	runBatch := func(batch int64, label string) {
		t.Helper()
		data := make([]float32, batch*trailDim)
		for i := range data {
			data[i] = float32(i+1) * 0.1
		}
		x.SetData(data)
		if err := jit.RealizeWithBinding(map[string]int64{symVar: batch}, y); err != nil {
			t.Fatalf("%s: JIT.RealizeWithBinding(batch=%d): %v", label, batch, err)
		}
		got := y.Data()
		want := cpuRef(data)
		if len(got) != len(want) {
			t.Fatalf("%s: output length %d != expected %d", label, len(got), len(want))
		}
		var maxErr float32
		for i := range got {
			if d := absF32JIT(got[i] - want[i]); d > maxErr {
				maxErr = d
			}
		}
		caps, reps := jit.JITStats()
		t.Logf("PROOF 4 %s (batch=%d): max-abs-diff=%.2e  captures=%d replays=%d",
			label, batch, maxErr, caps, reps)
		if maxErr > 1e-5 {
			t.Errorf("%s: max-abs-diff=%.2e > 1e-5", label, maxErr)
		}
	}

	// Capture at batch=8.
	runBatch(8, "capture")
	caps1, _ := jit.JITStats()
	if caps1 != 1 {
		t.Fatalf("after capture: want 1 capture, got %d", caps1)
	}

	// Replay at batch=32 — different binding, same schedule.
	runBatch(32, "replay-batch32")
	caps2, reps2 := jit.JITStats()
	if caps2 != 1 {
		t.Fatalf("replay at batch=32: expected no re-capture, got caps=%d", caps2)
	}
	if reps2 != 1 {
		t.Fatalf("replay at batch=32: expected 1 replay, got reps=%d", reps2)
	}
	t.Logf("PROOF 4: symbolic replay at different bind ✓ (caps=1, reps=1)")
}

// ── Proof 5: mismatch falls back safely ───────────────────────────────────────

// TestJITMismatchFallback captures a static computation with leaf size L, then
// calls JIT.Realize with a different-sized leaf.  The match guard detects the
// mismatch, re-captures, and produces the correct output for the new graph.
// This proves that stale-plan replay is never executed on structural change.
func TestJITMismatchFallback(t *testing.T) {
	requireGPUJIT(t)

	const device = "webgpu"

	jit := tensor.NewJIT()

	// ── First capture: batch=16 ──
	{
		a := uop.NewArena(4096)
		n := int64(16)
		x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, device)
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i + 1)
		}
		x.SetData(data)
		y := x.Sum(nil, false) // sum of [1..16] = 136

		if err := jit.Realize(y); err != nil {
			t.Fatalf("first capture: %v", err)
		}
		caps1, reps1 := jit.JITStats()
		want1 := float32(136)
		got1 := y.Data()[0]
		t.Logf("PROOF 5 capture (n=16): sum=%.1f  want=%.1f  caps=%d reps=%d",
			got1, want1, caps1, reps1)
		if got1 != want1 {
			t.Fatalf("capture: sum=%.1f want=%.1f", got1, want1)
		}
		if caps1 != 1 || reps1 != 0 {
			t.Fatalf("capture: want caps=1 reps=0, got caps=%d reps=%d", caps1, reps1)
		}
	}

	// ── Mismatch: batch=32 (different leaf size) ──
	{
		a := uop.NewArena(4096)
		n := int64(32)
		x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, device)
		data := make([]float32, n)
		for i := range data {
			data[i] = 1.0 // sum of [1..32] all-ones = 32
		}
		x.SetData(data)
		y := x.Sum(nil, false) // sum = 32

		if err := jit.Realize(y); err != nil {
			t.Fatalf("mismatch: %v", err)
		}
		caps2, reps2 := jit.JITStats()
		want2 := float32(32)
		got2 := y.Data()[0]
		t.Logf("PROOF 5 mismatch (n=32): sum=%.1f  want=%.1f  caps=%d reps=%d",
			got2, want2, caps2, reps2)
		if got2 != want2 {
			t.Fatalf("mismatch: sum=%.1f want=%.1f (would be %.1f if stale plan replayed)",
				got2, want2, float32(136))
		}
		if caps2 != 2 {
			t.Fatalf("mismatch: expected 2 captures (re-captured), got %d", caps2)
		}
		if reps2 != 0 {
			t.Fatalf("mismatch: expected 0 replays (only re-captures), got %d", reps2)
		}
	}

	// ── Replay the new (n=32) plan ──
	{
		a := uop.NewArena(4096)
		n := int64(32)
		x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, device)
		data := make([]float32, n)
		for i := range data {
			data[i] = 2.0 // sum = 64
		}
		x.SetData(data)
		y := x.Sum(nil, false)

		if err := jit.Realize(y); err != nil {
			t.Fatalf("replay after mismatch: %v", err)
		}
		caps3, reps3 := jit.JITStats()
		want3 := float32(64)
		got3 := y.Data()[0]
		t.Logf("PROOF 5 replay (n=32, all-2s): sum=%.1f  want=%.1f  caps=%d reps=%d",
			got3, want3, caps3, reps3)
		if got3 != want3 {
			t.Fatalf("replay after mismatch: sum=%.1f want=%.1f", got3, want3)
		}
		if caps3 != 2 || reps3 != 1 {
			t.Fatalf("replay after mismatch: want caps=2 reps=1, got caps=%d reps=%d", caps3, reps3)
		}
	}

	t.Logf("PROOF 5: mismatch falls back safely ✓ (re-captured, correct output, then replays)")
}
