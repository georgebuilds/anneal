package nn_test

// TestMaxPool2DForward  — table-driven forward value-identity check vs reference.
// TestMaxPool2DBackward — FD gradient check on a subset of shapes.
// TestMaxPool2DTieCoverage — assert winner-take-all tie semantics for binary Maximum.

import (
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// refMaxPool2D is a pure-Go reference implementation for correctness comparison.
func refMaxPool2D(data []float32, N, C, H, W, kH, kW, sH, sW int64) []float32 {
	oH := (H-kH)/sH + 1
	oW := (W-kW)/sW + 1
	out := make([]float32, N*C*oH*oW)
	for n := int64(0); n < N; n++ {
		for c := int64(0); c < C; c++ {
			for i := int64(0); i < oH; i++ {
				for j := int64(0); j < oW; j++ {
					best := float32(-1e38)
					for di := int64(0); di < kH; di++ {
						for dj := int64(0); dj < kW; dj++ {
							v := data[n*C*H*W+c*H*W+(i*sH+di)*W+(j*sW+dj)]
							if v > best {
								best = v
							}
						}
					}
					out[n*C*oH*oW+c*oH*oW+i*oW+j] = best
				}
			}
		}
	}
	return out
}

// TestMaxPool2DForward verifies forward values match the reference implementation
// for a variety of shapes, kernels, strides, and value patterns.
func TestMaxPool2DForward(t *testing.T) {
	requireGPU(t)

	type tc struct {
		name         string
		N, C, H, W   int64
		kH, kW       int64
		sH, sW       int64
		negativeData bool
	}
	cases := []tc{
		{"regular k2s2", 1, 1, 4, 4, 2, 2, 2, 2, false},
		{"overlapping k2s1", 1, 1, 4, 4, 2, 2, 1, 1, false},
		{"stride>1 k2s2 big", 1, 1, 8, 8, 2, 2, 2, 2, false},
		{"large kernel k4s4", 1, 1, 8, 8, 4, 4, 4, 4, false},
		{"asymmetric k3x2 s2x1", 1, 1, 8, 6, 3, 2, 2, 1, false},
		{"negative values", 1, 1, 4, 4, 2, 2, 2, 2, true},
		{"multi batch/channel", 2, 3, 8, 8, 2, 2, 2, 2, false},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			nElem := c.N * c.C * c.H * c.W
			data := make([]float32, nElem)
			if c.negativeData {
				for i := range data {
					data[i] = -float32(i + 1)
				}
			} else {
				for i := range data {
					data[i] = float32(i + 1)
				}
			}

			a := uop.NewArena(131072)
			x := tensor.NewLeaf(a, []int64{c.N, c.C, c.H, c.W}, uop.Dtypes.Float32, "webgpu")
			x.SetData(append([]float32{}, data...))

			out := nn.MaxPool2D(x, c.kH, c.kW, c.sH, c.sW)
			if err := tensor.Realize(out); err != nil {
				t.Fatalf("Realize: %v", err)
			}

			ref := refMaxPool2D(data, c.N, c.C, c.H, c.W, c.kH, c.kW, c.sH, c.sW)
			got := out.Data()

			if len(got) != len(ref) {
				t.Fatalf("output length mismatch: got %d want %d", len(got), len(ref))
			}

			maxDiff := float32(0)
			for i := range ref {
				d := absF32(got[i] - ref[i])
				if d > maxDiff {
					maxDiff = d
				}
			}

			oH := (c.H - c.kH) / c.sH + 1
			oW := (c.W - c.kW) / c.sW + 1
			t.Logf("shape [%d,%d,%d,%d] k=%dx%d s=%dx%d → [%d,%d,%d,%d] maxAbsDiff=%.2e",
				c.N, c.C, c.H, c.W, c.kH, c.kW, c.sH, c.sW, c.N, c.C, oH, oW, maxDiff)

			if maxDiff != 0 {
				t.Fatalf("forward mismatch: maxAbsDiff=%.2e (want 0)", maxDiff)
			}
		})
	}
}

// evalMaxPoolLoss builds a fresh graph with the given x data, computes
// L = sum(MaxPool2D(x, ...)^2) (MSE-sum vs zeros / sum of squares), realizes,
// and returns the scalar loss.
func evalMaxPoolLoss(t *testing.T, data []float32, N, C, H, W, kH, kW, sH, sW int64) float32 {
	t.Helper()
	a := uop.NewArena(131072)
	x := tensor.NewLeaf(a, []int64{N, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, data...))
	out := nn.MaxPool2D(x, kH, kW, sH, sW)
	loss := out.Mul(out).Sum(nil, false)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("evalMaxPoolLoss Realize: %v", err)
	}
	return loss.Data()[0]
}

// TestMaxPool2DBackward verifies gradients via finite differences.
func TestMaxPool2DBackward(t *testing.T) {
	requireGPU(t)

	type tc struct {
		name         string
		N, C, H, W   int64
		kH, kW       int64
		sH, sW       int64
	}
	cases := []tc{
		{"k2s2 [1,1,4,4]", 1, 1, 4, 4, 2, 2, 2, 2},
		{"k2s1 [1,1,4,4]", 1, 1, 4, 4, 2, 2, 1, 1},
		{"k3s3 [1,1,6,6]", 1, 1, 6, 6, 3, 3, 3, 3},
		{"k2s2 [8,4,4,4]", 8, 4, 4, 4, 2, 2, 2, 2}, // matches convnet MaxPool2D shape
	}

	const (
		eps    = float32(1e-3)
		tol    = float32(1e-3)
		nCheck = 4
	)

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			nElem := int(c.N * c.C * c.H * c.W)
			data := make([]float32, nElem)
			for i := range data {
				// Use non-tied values to get clean gradients.
				data[i] = float32(i+1) * 0.1
			}

			// Analytic gradient via backward.
			a := uop.NewArena(131072)
			x := tensor.NewLeaf(a, []int64{c.N, c.C, c.H, c.W}, uop.Dtypes.Float32, "webgpu")
			x.SetData(append([]float32{}, data...))
			out := nn.MaxPool2D(x, c.kH, c.kW, c.sH, c.sW)
			loss := out.Mul(out).Sum(nil, false)

			grads := tensor.Backward(loss, []*tensor.Tensor{x})
			gTensor, ok := grads[x]
			if !ok {
				t.Fatal("Backward: no gradient for x")
			}
			if err := tensor.Realize(gTensor); err != nil {
				t.Fatalf("Realize grad: %v", err)
			}
			analytic := gTensor.Data()

			// Finite differences.
			t.Logf("MaxPool2D FD gradient check  (tol=%.0e, eps=%.0e):", tol, eps)
			for i := 0; i < nCheck; i++ {
				orig := data[i]
				data[i] = orig + eps
				lp := evalMaxPoolLoss(t, data, c.N, c.C, c.H, c.W, c.kH, c.kW, c.sH, c.sW)
				data[i] = orig - eps
				lm := evalMaxPoolLoss(t, data, c.N, c.C, c.H, c.W, c.kH, c.kW, c.sH, c.sW)
				data[i] = orig
				fd := (lp - lm) / (2 * eps)

				ag := analytic[i]
				d := absF32(ag - fd)
				pass := d <= tol
				mark := "PASS"
				if !pass {
					mark = "FAIL"
				}
				t.Logf("  x[%d]: analytic=%+.6f  fd=%+.6f  diff=%.2e  %s", i, ag, fd, d, mark)
				if !pass {
					t.Fatalf("FAIL x[%d]: analytic=%.6f fd=%.6f diff=%.2e > tol=%.2e",
						i, ag, fd, d, tol)
				}
			}
			t.Logf("MaxPool2D backward PASSED ✓")
		})
	}
}

// TestMaxPool2DTieCoverage asserts split-equally tie semantics for MaxPool2D.
// Non-overlapping MaxPool2D (kH ≤ sH AND kW ≤ sW) uses the rangeify decomposition
// which delegates to ReduceAxis(OpMax). ReduceAxis(OpMax) backward splits the
// adjoint equally among all tied max positions (adj / tieCount per winner).
//
// split ties equally, matching ReduceAxis(OpMax) and tinygrad — SPEC §10 canonical policy.
//   - 2-way tie: both tied positions get adj/2 = 0.5; non-max positions get 0
//   - 3-way tie: all three tied positions get adj/3 ≈ 0.333...; all others get 0
func TestMaxPool2DTieCoverage(t *testing.T) {
	requireGPU(t)

	t.Run("2-way tie", func(t *testing.T) {
		// Input [1,1,2,2] = [3, 3, 1, 2], k=2x2, s=2x2 (non-overlapping, kH=sH=2).
		// Window covers all 4 elements. Max=3, tied at positions [0,0] and [0,1].
		// Rangeify decomposition → ReduceAxis(OpMax) backward: tieCount=2.
		// Expected: g[0]=0.5, g[1]=0.5, g[2]=0, g[3]=0.
		data := []float32{3, 3, 1, 2}

		a := uop.NewArena(131072)
		x := tensor.NewLeaf(a, []int64{1, 1, 2, 2}, uop.Dtypes.Float32, "webgpu")
		x.SetData(data)

		out := nn.MaxPool2D(x, 2, 2, 2, 2) // [1,1,1,1], value=3
		loss := out.Sum(nil, false)          // adj into pool = 1

		grads := tensor.Backward(loss, []*tensor.Tensor{x})
		gTensor, ok := grads[x]
		if !ok {
			t.Fatal("Backward: no gradient for x")
		}
		if err := tensor.Realize(gTensor); err != nil {
			t.Fatalf("Realize grad: %v", err)
		}
		g := gTensor.Data()

		t.Logf("2-way tie gradient values: x=[%.0f,%.0f,%.0f,%.0f] → g=[%.6f,%.6f,%.6f,%.6f]",
			data[0], data[1], data[2], data[3], g[0], g[1], g[2], g[3])

		// Both tied positions each receive adj/tieCount = 1/2 = 0.5.
		if absF32(g[0]-0.5) > 1e-5 {
			t.Errorf("tied position 0: want 0.5, got %.6f", g[0])
		}
		if absF32(g[1]-0.5) > 1e-5 {
			t.Errorf("tied position 1: want 0.5, got %.6f", g[1])
		}
		// Non-max positions get 0.
		for i := 2; i < 4; i++ {
			if absF32(g[i]) > 1e-5 {
				t.Errorf("non-max position %d: want 0, got %.6f", i, g[i])
			}
		}
		t.Logf("2-way tie PASSED ✓  split equally: both tied positions get 0.5")
	})

	t.Run("3-way tie", func(t *testing.T) {
		// Input [1,1,1,9], k=1x9, s=1x9 (non-overlapping, kH=1≤sH=9).
		// Values: [5, 5, 5, 0, 0, 0, 0, 0, 0]. Max=5, tied 3 ways at 0,1,2.
		// Rangeify decomposition → ReduceAxis(OpMax) backward: tieCount=3.
		// Expected: g[0]=g[1]=g[2]=1/3, g[3..8]=0.
		data := []float32{5, 5, 5, 0, 0, 0, 0, 0, 0}

		a := uop.NewArena(131072)
		x := tensor.NewLeaf(a, []int64{1, 1, 1, 9}, uop.Dtypes.Float32, "webgpu")
		x.SetData(data)

		out := nn.MaxPool2D(x, 1, 9, 1, 9) // [1,1,1,1], value=5
		loss := out.Sum(nil, false)          // adj into pool = 1

		grads := tensor.Backward(loss, []*tensor.Tensor{x})
		gTensor, ok := grads[x]
		if !ok {
			t.Fatal("Backward: no gradient for x")
		}
		if err := tensor.Realize(gTensor); err != nil {
			t.Fatalf("Realize grad: %v", err)
		}
		g := gTensor.Data()

		t.Logf("3-way tie gradient values:")
		for i, v := range g {
			t.Logf("  x[%d]=%g  g[%d]=%.8f", i, data[i], i, v)
		}

		// All three tied positions each receive adj/tieCount = 1/3.
		const want = float32(1.0) / 3.0
		for i := 0; i < 3; i++ {
			if absF32(g[i]-want) > 1e-5 {
				t.Errorf("tied position %d: want %.8f (1/3), got %.8f", i, want, g[i])
			}
		}
		// Non-max positions (3..8) get 0.
		for i := 3; i < 9; i++ {
			if absF32(g[i]) > 1e-5 {
				t.Errorf("non-max position %d: want 0, got %.8f", i, g[i])
			}
		}
		t.Logf("3-way tie PASSED ✓  split equally: all three tied positions get 1/3 ≈ %.8f", want)
	})
}
