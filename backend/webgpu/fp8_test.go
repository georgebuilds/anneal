package webgpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── fp8 GPU execution tests ───────────────────────────────────────────────────
//
// fp8 (e4m3fn / e5m2) is storage-only with f32 compute: the u32 storage word
// holds the fp8-quantized value's full f32 bit pattern (decoded storage, the
// same scheme as bf16), loads are a free bitcast<f32>, and RTNE narrowing to
// the fp8 grid happens at the store boundary. No device extension is needed -
// unlike the f16 tests, these run on any adapter.
//
// The host oracle uses uop.DType.Quantize, whose bit-twiddling the WGSL
// _fp8*_rtne_bits helpers mirror instruction for instruction, so every
// assertion here is bit-exact (max-abs-diff = 0), not tolerance-based.

// TestFP8_ElementwiseAdd runs z = x + y over fp8 buffers for both formats.
// The input range [-300, 300] makes many e4m3fn sums cross the ±448 finite
// ceiling, so the store helper's saturating path is exercised alongside the
// normal RTNE path; e5m2 stays in range and exercises plain RTNE.
func TestFP8_ElementwiseAdd(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype *uop.DType
	}{
		{"e4m3fn", uop.Dtypes.FP8E4M3},
		{"e5m2", uop.Dtypes.FP8E5M2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := requireDevice(t)

			rng := rand.New(rand.NewSource(23))
			const n = 512
			aVals := make([]float32, n)
			bVals := make([]float32, n)
			for i := range aVals {
				aVals[i] = float32(rng.Float64()*600 - 300) // [-300, 300]
				bVals[i] = float32(rng.Float64()*600 - 300)
			}

			a := uop.NewArena(4096)
			x := tensor.NewLeaf(a, []int64{n}, tc.dtype, "webgpu")
			y := tensor.NewLeaf(a, []int64{n}, tc.dtype, "webgpu")
			z := x.Add(y)

			items := makeSchedule(t, "webgpu", z)
			inputs := map[uint32][]float32{
				x.Node().Index(): aVals,
				y.Node().Index(): bVals,
			}
			outputs := runSchedule(t, dev, items, inputs)
			got := firstFinalOutput(t, items, outputs)

			q := tc.dtype.Quantize
			var maxDiff float64
			nFail := 0
			for i := range got {
				ref := q(q(aVals[i]) + q(bVals[i]))
				diff := math.Abs(float64(got[i]) - float64(ref))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff != 0 {
					nFail++
				}
			}
			t.Logf("fp8 %s elementwise add [%d]: max-abs-diff=%.6e, failures=%d/%d",
				tc.name, n, maxDiff, nFail, n)
			if nFail > 0 {
				t.Errorf("%d/%d elements differ from fp8 RTNE reference; max-abs-diff=%.6e",
					nFail, n, maxDiff)
			}
		})
	}
}

// TestFP8_RoundTrip verifies fp8 → f32 → fp8 is bit-identical for values
// exactly on each format's grid, including subnormals. RTNE is the identity
// on the grid, so this catches any divergence between upload encoding,
// in-shader bitcast load, store narrowing, and readback.
func TestFP8_RoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype *uop.DType
		vals  []float32
	}{
		{"e4m3fn", uop.Dtypes.FP8E4M3, []float32{
			0.0, 1.0, -1.0, 0.5, -3.5, 448, -448,
			0.001953125, // 2^-9, min subnormal
			0.013671875, // 7 * 2^-9, max subnormal
			0.015625,    // 2^-6, min normal
		}},
		{"e5m2", uop.Dtypes.FP8E5M2, []float32{
			0.0, 1.0, -1.0, 0.5, -3.5, 57344, -57344,
			0.0000152587890625, // 2^-16, min subnormal
			0.0000457763671875, // 3 * 2^-16, max subnormal
			0.00006103515625,   // 2^-14, min normal
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := requireDevice(t)

			a := uop.NewArena(1024)
			x := tensor.NewLeaf(a, []int64{int64(len(tc.vals))}, tc.dtype, "webgpu")
			xf32 := x.Cast(uop.Dtypes.Float32)
			xfp8 := xf32.Cast(tc.dtype)

			items := makeSchedule(t, "webgpu", xfp8)
			inputs := map[uint32][]float32{x.Node().Index(): tc.vals}
			outputs := runSchedule(t, dev, items, inputs)
			got := firstFinalOutput(t, items, outputs)

			for i, v := range got {
				want := tc.dtype.Quantize(tc.vals[i])
				if v != want {
					t.Errorf("fp8 %s round-trip[%d]: got %v, want %v", tc.name, i, v, want)
				}
			}
			t.Logf("fp8 %s round-trip: %d grid values checked (incl. subnormals)",
				tc.name, len(tc.vals))
		})
	}
}

// TestFP8_SumReduce verifies an fp8 sum-reduce accumulates in f32 and narrows
// once at the store. Inputs are small integers, so the f32 accumulation is
// exact regardless of reduction order and the reference is deterministic:
// quantize(sum of quantized inputs), bit-exact.
func TestFP8_SumReduce(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype *uop.DType
	}{
		{"e4m3fn", uop.Dtypes.FP8E4M3},
		{"e5m2", uop.Dtypes.FP8E5M2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := requireDevice(t)

			rng := rand.New(rand.NewSource(31))
			const n = 64
			vals := make([]float32, n)
			var sum float32
			for i := range vals {
				// Integers in [-4, 4]: exact on both fp8 grids (e5m2's coarsest
				// step below 8 is 1), and 64 of them sum within ±256 - exact in
				// f32 under any accumulation order.
				vals[i] = float32(rng.Intn(9) - 4)
				sum += vals[i]
			}

			a := uop.NewArena(4096)
			x := tensor.NewLeaf(a, []int64{n}, tc.dtype, "webgpu")
			y := x.Sum([]int{0}, false)

			items := makeSchedule(t, "webgpu", y)
			inputs := map[uint32][]float32{x.Node().Index(): vals}
			outputs := runSchedule(t, dev, items, inputs)
			got := firstFinalOutput(t, items, outputs)

			want := tc.dtype.Quantize(sum)
			if len(got) != 1 || got[0] != want {
				t.Errorf("fp8 %s sum-reduce: got %v, want [%v]", tc.name, got, want)
			}
			t.Logf("fp8 %s sum-reduce [%d]: got %v (exact)", tc.name, n, got)
		})
	}
}
