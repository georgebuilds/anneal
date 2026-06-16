package nn_test

// Regression guard for the CPU interpreter's broadcast-param indexing.
//
// BatchNorm2d's Weight and Bias are per-channel [C] parameters that broadcast
// over a 4-D [N, C, H, W] activation. Their forward loads index a [C] buffer
// with a 4-D index expression (the broadcast dims carry index 0), and their
// backward gradient kernels compute offsets that run one past the buffer end
// for the broadcast dims. Two interp bugs lived here:
//
//   1. evalIntIndex errored when the index had more dims than the buffer
//      shape, instead of folding the missing dims to a stride factor of 1
//      (the codegen paramDimFactor convention).
//   2. evalIndexLoadFloat faulted on the resulting out-of-range offset,
//      instead of clamping like naga/WGSL storage-buffer robustness does on
//      the GPU.
//
// Both made the CPU backend diverge from the GPU for any 4-D conv + BatchNorm
// graph (resnet9, diffusion). This finite-difference check runs entirely on
// the pure-Go CPU backend (no GPU, ~0.02s) and fails if the analytic
// broadcast-param gradients drift from numerical ground truth.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestBatchNorm2dCPUGradCheck(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const (
		N, C, H, W = int64(2), int64(3), int64(2), int64(2)
		eps        = float32(1e-3)
		tol        = float64(4e-3)
	)
	rng := rand.New(rand.NewSource(7))
	xData := make([]float32, int(N*C*H*W))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}
	wInit := []float32{1.1, 0.9, 1.05}
	bInit := []float32{0.0, 0.1, -0.05}

	lossFn := func(w, b []float32) float32 {
		a := uop.NewArena(131072)
		bn := nn.NewBatchNorm2d(a, C, 1e-5, 0.1, uop.Dtypes.Float32, "cpu")
		copy(bn.Weight.Value, w)
		copy(bn.Bias.Value, b)
		for _, p := range bn.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "cpu")
		x.SetData(append([]float32{}, xData...))
		out := bn.Forward(x)
		loss := out.Mul(out).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("realize loss: %v", err)
		}
		return loss.Data()[0]
	}

	a := uop.NewArena(131072)
	bn := nn.NewBatchNorm2d(a, C, 1e-5, 0.1, uop.Dtypes.Float32, "cpu")
	copy(bn.Weight.Value, wInit)
	copy(bn.Bias.Value, bInit)
	for _, p := range bn.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "cpu")
	x.SetData(append([]float32{}, xData...))
	out := bn.Forward(x)
	loss := out.Mul(out).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{bn.Weight.T, bn.Bias.T})
	if err := tensor.Realize(grads[bn.Weight.T]); err != nil {
		t.Fatalf("realize Weight grad: %v", err)
	}
	if err := tensor.Realize(grads[bn.Bias.T]); err != nil {
		t.Fatalf("realize Bias grad: %v", err)
	}
	gW := grads[bn.Weight.T].Data()
	gB := grads[bn.Bias.T].Data()

	for c := 0; c < int(C); c++ {
		wp := append([]float32{}, wInit...)
		wp[c] += eps
		wm := append([]float32{}, wInit...)
		wm[c] -= eps
		fdW := (lossFn(wp, bInit) - lossFn(wm, bInit)) / (2 * eps)
		relW := math.Abs(float64(fdW-gW[c])) / (math.Abs(float64(fdW)) + 1e-4)
		if relW > tol {
			t.Errorf("Weight[%d]: analytic=%.6f fd=%.6f rel=%.2e > tol %.0e", c, gW[c], fdW, relW, tol)
		}

		bp := append([]float32{}, bInit...)
		bp[c] += eps
		bm := append([]float32{}, bInit...)
		bm[c] -= eps
		fdB := (lossFn(wInit, bp) - lossFn(wInit, bm)) / (2 * eps)
		relB := math.Abs(float64(fdB-gB[c])) / (math.Abs(float64(fdB)) + 1e-4)
		if relB > tol {
			t.Errorf("Bias[%d]: analytic=%.6f fd=%.6f rel=%.2e > tol %.0e", c, gB[c], fdB, relB, tol)
		}
	}
}
