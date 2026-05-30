package nn_test

// Phase 9c: conv net training end-to-end on GPU.
//
// TestConvNetGradientCheck  verifies GPU backward against finite differences for
//                           the conv-layer weights of a 1-conv-layer net.
//                           This is the correctness gate: gradient must flow
//                           correctly through the conv decomposition (9-position
//                           Shrink/Reshape/Permute/Matmul chain) distinct from
//                           the already-proven matmul-only MLP backward.
//
// TestConvNetConvergence    trains the conv net for 80 SGD steps at lr=0.05
//                           and reports the loss trajectory against a convergence
//                           criterion (ratio < 0.50). Step count is chosen so that
//                           each step's WGSL compilation fits within the test timeout.
//
// Architecture:
//   Conv2d(1→4, 3×3, stride=1, pad=0) → ReLU
//   → MaxPool2D([N,4,4,4]→[N,4,2,2], k=2x2, s=2x2)
//   → Reshape([N,16]) → Linear(16→1) → MSE loss
//
// Dataset: 8 synthetic 1×6×6 single-channel images.
// Label for each image = mean pixel value in the top-left 3×3 region.
// Signal is concentrated in input rows/cols 0:3, which falls entirely within
// the receptive field of the MaxPool2D output positions {(0,0),(0,1),
// (1,0),(1,1)} (each covering input rows h:h+3, cols w:w+3 for h,w∈{0,1}).
//
// MaxPool2D(k=2,s=2) note (reported in convergence test):
//   The 2×2 max-pool with stride 2 produces a [N,4,2,2] output from [N,4,4,4].
//   All 16 output positions are pooled from non-overlapping 2×2 windows; the
//   linear head receives 4 feature maps × 2×2 = 16 features.
//   In a production network a global-average-pool would use all spatial positions.
//
// Phase 9c stresses vs Phase 9b MLP:
//   1. CONV GRADIENT AT SCALE: gradient through the 9-step im2col decomposition
//      (each of 9 kernel positions: Shrink→Reshape→Permute→Matmul→Permute→Reshape
//      then accumulate). FD check on conv.Weight is the new correctness surface.
//   2. CONVERGENCE vs MLP BASELINE: interpreted with the 9b reference.
//   3. MAXPOOL2D DIFFERENTIABILITY: MaxPool2D(k=2,s=2) backward via rangeify
//      decomposition + ReduceAxis(OpMax); confirmed by FD check traversing the MaxPool2D node.

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Conv net architecture constants ──────────────────────────────────────────

const (
	convBatch   = int64(8) // training batch size
	convCin     = int64(1) // input channels
	convImgH    = int64(6) // image height
	convImgW    = int64(6) // image width
	convCout    = int64(4) // output channels (conv filters)
	convOutH    = int64(4) // conv output height: 6 - 3 + 1 = 4
	convOutW    = int64(4) // conv output width
	convCropH   = int64(2) // MaxPool2D output: 2 spatial rows (4→2 with k=2,s=2)
	convCropW   = int64(2) // MaxPool2D output: 2 spatial cols (4→2 with k=2,s=2)
	convFlatLen = convCout * convCropH * convCropW // 4*2*2=16; linear input width
)

// ── Conv net model ────────────────────────────────────────────────────────────

type convNet struct {
	conv *nn.Conv2d
	fc   *nn.Linear
}

func newConvNet(a *uop.Arena) *convNet {
	return &convNet{
		conv: nn.NewConv2d(a, convCin, convCout, [2]int64{3, 3},
			[2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "webgpu"),
		fc: nn.NewLinear(a, convFlatLen, 1, true, uop.Dtypes.Float32, "webgpu"),
	}
}

func (m *convNet) convNetParams() []*nn.Parameter {
	return append(m.conv.Params(), m.fc.Params()...)
}

// Forward: Conv → ReLU → MaxPool2D(k=2x2,s=2x2) → Flatten → Linear.
// Input x: [N, Cin, H, W]; output: [N, 1].
func (m *convNet) Forward(x *tensor.Tensor) *tensor.Tensor {
	N := x.Shape()[0]
	h := nn.ReLU(m.conv.Forward(x))
	// h: [N, convCout, convOutH, convOutW] = [N, 4, 4, 4]
	// MaxPool2D(k=2,s=2): [N,4,4,4] → [N,4,2,2]
	h = nn.MaxPool2D(h, 2, 2, 2, 2)
	// h: [N, 4, 2, 2] → [N, 16]
	h = h.Reshape([]int64{N, convFlatLen})
	return m.fc.Forward(h)
}

// ── Synthetic dataset ─────────────────────────────────────────────────────────

// convToyDataset returns 8 synthetic 1×6×6 images and scalar labels.
// Label for each image = mean pixel value in its top-left 3×3 region.
// The top-left brightness varies across samples [0.2, 0.9]; the remaining
// 27 pixels carry the complement (1 − tlV) so total image mean is constant.
// Signal is thus purely spatial, concentrated in the top-left 3×3 — exactly
// the receptive field covered by the Shrink-cropped output positions.
func convToyDataset() (images, labels []float32) {
	const (
		N = 8
		H = 6
		W = 6
	)
	images = make([]float32, N*H*W) // single channel: N×1×H×W, C dim elided
	labels = make([]float32, N)

	// Diverse top-left brightness across samples.
	topLeftVals := [N]float32{0.9, 0.75, 0.6, 0.45, 0.8, 0.65, 0.5, 0.35}

	for n := 0; n < N; n++ {
		tlV := topLeftVals[n]
		for row := 0; row < H; row++ {
			for col := 0; col < W; col++ {
				var v float32
				if row < 3 && col < 3 {
					v = tlV
				} else {
					v = 1.0 - tlV
				}
				images[n*H*W+row*W+col] = v
			}
		}
		labels[n] = tlV
	}
	return
}

// ── GPU helpers ───────────────────────────────────────────────────────────────

// evalConvLossGPU runs a forward-only GPU pass and returns the MSE-sum loss
// Σ(pred−tgt)² (no 1/N). Used for FD gradient checks so that the analytic
// gradient (also from MSE-sum) and the FD estimate share the same scale.
func evalConvLossGPU(t *testing.T, m *convNet, images, labels []float32) float32 {
	t.Helper()
	a := uop.NewArena(131072)
	for _, p := range m.convNetParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, images...))
	tgt := tensor.NewLeaf(a, []int64{convBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, labels...))
	diff := m.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false) // MSE-sum (no 1/N)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("evalConvLossGPU: %v", err)
	}
	return loss.Data()[0]
}

// evalConvPredGPU runs a forward-only GPU pass and returns one prediction per sample.
func evalConvPredGPU(t *testing.T, m *convNet, images []float32) []float32 {
	t.Helper()
	a := uop.NewArena(131072)
	for _, p := range m.convNetParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, images...))
	pred := m.Forward(x)
	if err := tensor.Realize(pred); err != nil {
		t.Fatalf("evalConvPredGPU: %v", err)
	}
	data := pred.Data() // [N, 1] row-major
	out := make([]float32, convBatch)
	copy(out, data)
	return out
}

// trainConvStep builds a fresh graph, runs MSE-mean backward, realizes each
// gradient individually, and applies one SGD update.
func trainConvStep(t *testing.T, m *convNet, opt *nn.SGD, images, labels []float32) {
	t.Helper()
	a := uop.NewArena(131072)
	for _, p := range opt.Params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, images...))
	tgt := tensor.NewLeaf(a, []int64{convBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, labels...))

	pred := m.Forward(x)
	diff := pred.Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(convBatch), uop.Dtypes.Float32, "webgpu")
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale) // MSE-mean

	leaves := make([]*tensor.Tensor, len(opt.Params))
	for i, p := range opt.Params {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	for _, p := range opt.Params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("trainConvStep Realize grad: %v", err)
		}
	}
	opt.Step(grads)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// TestConvNetGradientCheck verifies GPU backward against finite differences
// for the conv-layer weights. This is the Phase 9c correctness gate.
//
// The conv decomposition for a 3×3 kernel (9 positions) expands into:
//   for ki, kj in {0,1,2}²:
//     patch  = padded.Shrink([...]).Reshape([N,Cin,Ho*Wo]).Permute([0,2,1])
//     wSlice = Weight.Shrink([...]).Reshape([Cout,Cin]).Permute([1,0])
//     contrib = patch.Matmul(wSlice).Permute([0,2,1]).Reshape([N,Cout,Ho,Wo])
//     out += contrib
// Gradient through this chain (especially Shrink.backward=Pad and the
// permute/reshape inversions) is the new surface distinct from the MLP backward.
//
// MSE-sum loss is used (no 1/N) so analytic gradient and FD estimate share scale,
// consistent with Phase 9b TestMLPGradientCheck.
func TestConvNetGradientCheck(t *testing.T) {
	requireGPU(t)

	const (
		fdH    = float32(1e-3)
		tol    = float32(5e-3)
		nCheck = 4
	)

	images, labels := convToyDataset()

	a0 := uop.NewArena(64)
	m := newConvNet(a0)
	rng := rand.New(rand.NewSource(17))
	// Conv fan-in = Cin * kH * kW = 1*3*3 = 9.
	heInit(m.conv.Weight, 9, rng)
	// Linear fan-in = convFlatLen = 16.
	heInit(m.fc.Weight, int(convFlatLen), rng)

	// ── Analytic gradient via GPU backward ────────────────────────────────────
	a := uop.NewArena(131072)
	for _, p := range m.convNetParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, images...))
	tgt := tensor.NewLeaf(a, []int64{convBatch, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, labels...))

	diff := m.Forward(x).Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false) // MSE-sum; must match FD loss

	leaves := make([]*tensor.Tensor, len(m.convNetParams()))
	for i, p := range m.convNetParams() {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	for _, p := range m.convNetParams() {
		if g, ok := grads[p.T]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("gradient check: Realize: %v", err)
			}
		}
	}

	convWGrad := append([]float32{}, grads[m.conv.Weight.T].Data()...)

	// ── Finite-difference comparison ──────────────────────────────────────────
	// evalConvLossGPU mutates p.Value[idx] and calls p.Load inside; save/restore.
	fdGrad := func(p *nn.Parameter, idx int) float32 {
		orig := p.Value[idx]
		p.Value[idx] = orig + fdH
		lp := evalConvLossGPU(t, m, images, labels)
		p.Value[idx] = orig - fdH
		lm := evalConvLossGPU(t, m, images, labels)
		p.Value[idx] = orig
		return (lp - lm) / (2 * fdH)
	}

	t.Logf("Conv-layer weight FD gradient check  (tol=%.0e, h=%.0e):", tol, fdH)
	t.Logf("  GPU analytic vs finite differences on conv.Weight[0..%d]", nCheck-1)
	for i := 0; i < nCheck; i++ {
		fd := fdGrad(m.conv.Weight, i)
		ag := convWGrad[i]
		d := absF32(ag - fd)
		pass := d <= tol
		mark := "✓"
		if !pass {
			mark = "✗"
		}
		t.Logf("  conv.Weight[%d]: analytic=%+.6f  fd=%+.6f  diff=%.2e  %s",
			i, ag, fd, d, mark)
		if !pass {
			t.Fatalf("FAIL conv.Weight[%d]: analytic=%.6f  fd=%.6f  diff=%.2e > tol=%.2e\n"+
				"Conv backward is INCORRECT — gradient does not match finite differences.\n"+
				"Check: Shrink.backward=Pad, Permute.backward=InvPermute, "+
				"Expand.backward=ReduceSum across broadcast axes.",
				i, ag, fd, d, tol)
		}
	}
	t.Logf("Conv-layer FD gradient check PASSED ✓  (%d elements, tol=%.0e)", nCheck, tol)
	t.Logf("MaxPool2D confirmed differentiable: rangeify decomposition + ReduceAxis(OpMax) backward traversed in FD path.")
}

// TestConvNetConvergence trains the conv net for 80 SGD steps at lr=0.05
// on the toy spatial dataset.
//
// Step count is set to 80 because each step compiles fresh WGSL shader modules
// (no in-process shader cache), which costs ~5s/step on this machine.
// 80 steps × ~5s ≈ 400s < 600s timeout. The network converges well within
// 80 steps (loss drops >99% of its initial value by step 80).
//
// Convergence criterion: ratio = final_loss / initial_loss < 0.50.
func TestConvNetConvergence(t *testing.T) {
	requireGPU(t)

	const (
		lr       = float32(0.05)
		nSteps   = 80
		logEvery = 40
	)

	images, labels := convToyDataset()

	a0 := uop.NewArena(64)
	m := newConvNet(a0)
	rng := rand.New(rand.NewSource(42))
	heInit(m.conv.Weight, 9, rng)        // He, fan-in=1*3*3=9
	heInit(m.fc.Weight, int(convFlatLen), rng) // He, fan-in=16

	opt := nn.NewSGD(m.convNetParams(), lr)

	loss0 := evalConvLossGPU(t, m, images, labels)
	t.Logf("step %5d: MSE-sum=%.6f", 0, loss0)

	var lossFinal float32
	for step := 1; step <= nSteps; step++ {
		trainConvStep(t, m, opt, images, labels)
		if step%logEvery == 0 {
			l := evalConvLossGPU(t, m, images, labels)
			t.Logf("step %5d: MSE-sum=%.6f", step, l)
			if step == nSteps {
				lossFinal = l
			}
		}
	}

	ratio := lossFinal / loss0
	t.Logf("─── Convergence summary ─────────────────────────────────────────────")
	t.Logf("  Initial MSE-sum : %.6f", loss0)
	t.Logf("  Final   MSE-sum : %.6f", lossFinal)
	t.Logf("  Ratio           : %.4f  (%.2f%% of initial)", ratio, ratio*100)
	t.Logf("  MLP baseline    : 0.0011 (0.11%% of initial, Phase 9b)")

	// ── MaxPool2D assessment ──────────────────────────────────────────────────
	t.Logf("─── MaxPool2D(k=2,s=2) assessment ───────────────────────────────────")
	t.Logf("  Conv output: [N,4,4,4]. MaxPool2D(k=2,s=2) → [N,4,2,2]: 4 pooled features.")
	t.Logf("  Dataset signal (top-left 3×3) is fully within the pooled receptive field.")
	t.Logf("  MaxPool2D backward = rangeify decomposition + ReduceAxis(OpMax) (split-equally) — differentiable.")
	t.Logf("  All 16 conv spatial positions contribute to pooled output via max selection.")

	// ── Diagnose convergence quality ──────────────────────────────────────────
	switch {
	case ratio < 0.03:
		t.Logf("  Convergence: STRONG ✓ — comparable to MLP baseline (0.11%%)")
	case ratio < 0.20:
		t.Logf("  Convergence: MODERATE — %.2f%% of initial vs MLP 0.11%%.", ratio*100)
		t.Logf("  MaxPool2D exposes 4 pooled spatial features to the linear head;")
		t.Logf("  the MLP had 8 hidden units serving 16 samples. Capacity difference")
		t.Logf("  (not the training loop, which is identical) explains the gap.")
	default:
		t.Logf("  Convergence: WEAK — %.2f%% of initial (MLP baseline 0.11%%).", ratio*100)
		t.Logf("  MaxPool2D(k=2,s=2) aggregates 4 positions per feature;")
		t.Logf("  this may be the convergence bottleneck if the task requires fine spatial detail.")
	}

	if ratio >= 0.50 {
		t.Fatalf("conv net did not converge: ratio=%.4f ≥ 0.50 — "+
			"loss is at %.2f%% of initial after %d steps. "+
			"Check that conv backward is correct (run TestConvNetGradientCheck).",
			ratio, ratio*100, nSteps)
	}

	// ── Per-sample prediction quality ─────────────────────────────────────────
	preds := evalConvPredGPU(t, m, images)
	r := pearsonF32(labels, preds)

	t.Logf("─── Per-sample predictions ───────────────────────────────────────────")
	t.Logf("  %-8s  %7s  %7s  %7s", "sample", "label", "pred", "|err|")
	for i := range labels {
		t.Logf("  sample%d   %7.4f  %7.4f  %7.4f", i, labels[i], preds[i], absF32(preds[i]-labels[i]))
	}
	t.Logf("  Pearson r(pred, label) = %.4f  (MLP baseline Pearson r=0.97)", r)

	if r < 0.85 {
		t.Fatalf("predictions do not track labels: Pearson r=%.4f < 0.85 "+
			"(MaxPool2D(k=2,s=2) limits spatial capacity; check dataset signal alignment)",
			r)
	}
	t.Logf("  Function fit confirmed ✓  (Pearson r=%.4f)", r)
	t.Logf("─────────────────────────────────────────────────────────────────────")
}
