package nn_test

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestGradientThroughCastBF16ToF32 verifies that gradients are computed in
// high precision (f32) even when the forward path involves a bf16 storage leaf
// cast to f32. This validates the fix in tensor/gradient.go that prevents
// adjoint narrowing during OpCast backward.
func TestGradientThroughCastBF16ToF32(t *testing.T) {
	requireGPU(t)

	const (
		h   = float32(1e-3)
		tol = float32(1e-3)
	)

	a := uop.NewArena(65536)
	rng := rand.New(rand.NewSource(42))

	// 1. Create a bf16 leaf.
	xVal := float32(rng.NormFloat64())
	x := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.BFloat16, "webgpu")
	x.SetData([]float32{xVal})
	
	// x.Data() returns the quantized value that actually went into the BUFFER.
	// For the FD reference, we must use this exact value as the center point,
	// otherwise the FD reference and the analytic gradient are differentiating
	// at different points.
	xQuantized := x.Data()[0]

	// 2. Build graph: y = (x_as_f32 * 2 + 1)
	// The cast from BFloat16 to Float32 is the node under test.
	xAsF32 := x.Cast(uop.Dtypes.Float32)
	two := tensor.ConstScalar(a, 2.0, uop.Dtypes.Float32, "webgpu")
	one := tensor.ConstScalar(a, 1.0, uop.Dtypes.Float32, "webgpu")
	y := xAsF32.Mul(two).Add(one)

	// 3. Backward
	grads := tensor.Backward(y, []*tensor.Tensor{x})
	g, ok := grads[x]
	if !ok {
		t.Fatal("no gradient for x")
	}
	if err := tensor.Realize(g); err != nil {
		t.Fatalf("Realize grad: %v", err)
	}
	analytic := g.Data()[0]

	// 4. Finite Differences Reference (computed in pure f32)
	// We want to prove that the analytic gradient is exactly 2.0 (the derivative 
	// of 2x+1), unaffected by any bf16-narrowing in the backward path.
	// Since x was quantized to bf16, the forward pass is effectively y = 2*bf16(x) + 1.
	// The FD check must also respect that x is quantized to bf16 in the model.
	
	evalLoss := func(val float32) float32 {
		// New arena for each eval to avoid index collisions.
		ae := uop.NewArena(1024)
		// FD reference is computed in pure f32 (no bf16 involved)
		xe := tensor.NewLeaf(ae, []int64{1}, uop.Dtypes.Float32, "webgpu")
		xe.SetData([]float32{val})
		
		twoE := tensor.ConstScalar(ae, 2.0, uop.Dtypes.Float32, "webgpu")
		oneE := tensor.ConstScalar(ae, 1.0, uop.Dtypes.Float32, "webgpu")
		
		ye := xe.Mul(twoE).Add(oneE)
		if err := tensor.Realize(ye); err != nil {
			t.Fatalf("FD Realize: %v", err)
		}
		return ye.Data()[0]
	}

	lp := evalLoss(xQuantized + h)
	lm := evalLoss(xQuantized - h)
	fd := (lp - lm) / (2 * h)

	diff := absF32(analytic - fd)
	t.Logf("xQuantized=%.6f analytic=%.6f fd=%.6f diff=%.2e", xQuantized, analytic, fd, diff)

	if diff > tol {
		t.Fatalf("gradient through cast narrowed! diff=%.2e > tol=%.2e", diff, tol)
	}
	
	// Also check that it's actually 2.0. If the fix were missing, the adjoint (1.0)
	// of the cast would be narrowed to bf16. But bf16 can represent 1.0 and 2.0 
	// exactly, so we need a multiplier that bf16 would lose precision on if 
	// narrowed.
	// bf16 has 7 bits. 1.0 + 1/256 is NOT representable in bf16 (quantizes to 1.0).
	
	a2 := uop.NewArena(1024)
	x2 := tensor.NewLeaf(a2, []int64{1}, uop.Dtypes.BFloat16, "webgpu")
	x2.SetData([]float32{xVal})
	
	// y = x * (1 + 1/256)
	// If the gradient (1 + 1/256) is narrowed to bf16, it becomes 1.0.
	multiplier := float32(1.0 + 1.0/256.0)
	m := tensor.ConstScalar(a2, float64(multiplier), uop.Dtypes.Float32, "webgpu")
	y2 := x2.Cast(uop.Dtypes.Float32).Mul(m)
	
	grads2 := tensor.Backward(y2, []*tensor.Tensor{x2})
	g2 := grads2[x2]
	tensor.Realize(g2)
	analytic2 := g2.Data()[0]
	
	if analytic2 == 1.0 && multiplier != 1.0 {
		t.Fatalf("gradient precision lost: analytic=%.6f, want %.6f (found 1.0, likely narrowed to bf16)", 
			analytic2, multiplier)
	}
	t.Logf("high-precision gradient confirmed: analytic=%.8f want=%.8f", analytic2, multiplier)
}
