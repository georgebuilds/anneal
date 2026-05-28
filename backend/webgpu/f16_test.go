package webgpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── f16 GPU execution tests ───────────────────────────────────────────────────
//
// These tests require a GPU with the shader-f16 extension.
// Slice 2 requests the extension at device-open time; on hardware that exposes
// it, these tests pass. On hardware that does not expose it (e.g. some drivers
// that support f16 in practice but do not advertise the feature), they are
// skipped. All tests use requireDevice(t) which skips when no GPU is available.

// requireShaderF16 skips the test if the device was opened without the shader-f16
// extension. This happens when the adapter does not advertise FeatureShaderF16,
// even on hardware that physically supports half-precision arithmetic.
func requireShaderF16(t *testing.T, dev *webgpu.Device) {
	t.Helper()
	if !dev.HasShaderF16 {
		t.Skip("shader-f16 not exposed by this adapter; skipping f16 GPU test")
	}
}

// f16RoundTrip simulates Go-side what happens when a float32 value is uploaded
// to a GPU f16 buffer and read back: float32 → f16 (quantize) → float32 (widen).
// Used to compute accurate references that account for input quantization.
func f16RoundTrip(v float32) float32 {
	bits := math.Float32bits(v)
	sign := uint16(bits >> 31)
	exp := int32((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	var h uint16
	switch {
	case exp > 15:
		h = (sign << 15) | 0x7C00
	case exp < -24:
		h = sign << 15
	case exp < -14:
		frac |= 0x800000
		h = (sign << 15) | uint16(frac>>(uint32(-14-exp)+13))
	default:
		h = (sign << 15) | (uint16(exp+15) << 10) | uint16(frac>>13)
	}
	// Widen back to float32.
	sign32 := uint32(h>>15) << 31
	exp16 := uint32((h >> 10) & 0x1F)
	frac16 := uint32(h & 0x3FF)
	var bits32 uint32
	switch exp16 {
	case 0:
		if frac16 == 0 {
			bits32 = sign32
		} else {
			e := uint32(127 - 14)
			for frac16&0x400 == 0 {
				frac16 <<= 1
				e--
			}
			frac16 &= 0x3FF
			bits32 = sign32 | (e << 23) | (frac16 << 13)
		}
	case 31:
		bits32 = sign32 | 0x7F800000 | (frac16 << 13)
	default:
		bits32 = sign32 | ((exp16+112)<<23) | (frac16 << 13)
	}
	return math.Float32frombits(bits32)
}

// TestF16_ElementwiseAdd tests f16 elementwise addition vs a reference that
// accounts for f16 quantization of inputs during upload.
// Tolerance: atol=2e-2 — input values in [-10,10] quantize to f16 with ULP up to
// 2^-7 ≈ 0.0078 (for |x| ∈ [8,16]); two inputs + output rounding ≤ 3×0.0078 ≈ 0.023.
// Ref: IEEE 754-2008 §3.3 half-precision ULP table.
func TestF16_ElementwiseAdd(t *testing.T) {
	dev := requireDevice(t)
	requireShaderF16(t, dev)

	rng := rand.New(rand.NewSource(42))
	const n = 1024
	aVals := make([]float32, n)
	bVals := make([]float32, n)
	for i := range aVals {
		aVals[i] = float32(rng.Float64()*20 - 10) // [-10, 10]
		bVals[i] = float32(rng.Float64()*20 - 10)
	}

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float16, "webgpu")
	y := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float16, "webgpu")
	z := x.Add(y)

	items := makeSchedule(t, "webgpu", z)
	inputs := map[uint32][]float32{
		x.Node().Index(): aVals,
		y.Node().Index(): bVals,
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)

	// Reference: add the f16-quantized inputs (matching what the GPU actually sees).
	// This is tighter than comparing to float32 sums because upload quantizes to f16.
	const atol = 2e-2 // covers input + output f16 ULP for values in [-10,10]
	var maxDiff float64
	nFail := 0
	for i := range got {
		ref := f16RoundTrip(aVals[i]) + f16RoundTrip(bVals[i])
		diff := math.Abs(float64(got[i]) - float64(ref))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > atol {
			nFail++
		}
	}
	t.Logf("f16 elementwise add [1024]: max-abs-diff=%.6e, failures=%d/%d (atol=%.0e)",
		maxDiff, nFail, n, atol)
	if nFail > 0 {
		t.Errorf("%d/%d elements exceed atol=%.0e; max-abs-diff=%.6e", nFail, n, atol, maxDiff)
	}
}

// TestF16_MatmulF32Acc tests f16 64×64 matmul with f32 accumulator.
// Tolerance: atol=1e-2, rtol=1e-3.
// Ref: TVM test_to_mixed_precision single-conv defaults (atol=1e-2, rtol=1e-3).
func TestF16_MatmulF32Acc(t *testing.T) {
	dev := requireDevice(t)
	requireShaderF16(t, dev)

	const M, K, N = 64, 64, 64
	rng := rand.New(rand.NewSource(7))
	aVals := make([]float32, M*K)
	bVals := make([]float32, K*N)
	for i := range aVals {
		aVals[i] = float32(rng.Float64()*2 - 1) // [-1, 1]
	}
	for i := range bVals {
		bVals[i] = float32(rng.Float64()*2 - 1)
	}

	a := uop.NewArena(8192)
	A := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.Float16, "webgpu")
	B := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.Float16, "webgpu")
	C := A.Matmul(B)

	items := makeSchedule(t, "webgpu", C)
	inputs := map[uint32][]float32{
		A.Node().Index(): aVals,
		B.Node().Index(): bVals,
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)

	// Reference: f64 matmul on Go side.
	ref := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += float64(aVals[i*K+k]) * float64(bVals[k*N+j])
			}
			ref[i*N+j] = float32(sum)
		}
	}

	const atol, rtol = 1e-2, 1e-3
	var maxDiff float64
	nFail := 0
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(ref[i]))
		rel := diff / (math.Abs(float64(ref[i])) + 1e-8)
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > atol && rel > rtol {
			nFail++
		}
	}
	t.Logf("f16 64×64 matmul: max-abs-diff=%.6e, failures=%d/%d (atol=%.0e, rtol=%.0e)",
		maxDiff, nFail, M*N, atol, rtol)
	if nFail > 0 {
		t.Errorf("%d/%d elements exceed tolerances (atol=%.0e, rtol=%.0e)", nFail, M*N, atol, rtol)
	}
}

// TestF16_ChainedMatmul tests D = A @ B @ C with three 32×32 f16 matrices.
// Tolerance: atol=5e-2, rtol=1e-2.
// Ref: tinygrad PR #7973 found rtol=atol=2e-3 violated after 7 convs; 3 matmuls
// is similar accumulation depth, so a looser bound is expected and appropriate.
func TestF16_ChainedMatmul(t *testing.T) {
	dev := requireDevice(t)
	requireShaderF16(t, dev)

	const sz = 32
	rng := rand.New(rand.NewSource(13))
	mkMat := func() []float32 {
		v := make([]float32, sz*sz)
		for i := range v {
			v[i] = float32(rng.Float64()*2 - 1)
		}
		return v
	}
	aVals, bVals, cVals := mkMat(), mkMat(), mkMat()

	a := uop.NewArena(8192)
	A := tensor.NewLeaf(a, []int64{sz, sz}, uop.Dtypes.Float16, "webgpu")
	B := tensor.NewLeaf(a, []int64{sz, sz}, uop.Dtypes.Float16, "webgpu")
	C := tensor.NewLeaf(a, []int64{sz, sz}, uop.Dtypes.Float16, "webgpu")
	D := A.Matmul(B).Matmul(C)

	items := makeSchedule(t, "webgpu", D)
	inputs := map[uint32][]float32{
		A.Node().Index(): aVals,
		B.Node().Index(): bVals,
		C.Node().Index(): cVals,
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)

	// Reference: (A @ B) @ C in float64.
	f64mat := func(a, b []float32, m, k, n int) []float32 {
		r := make([]float32, m*n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float64
				for p := 0; p < k; p++ {
					s += float64(a[i*k+p]) * float64(b[p*n+j])
				}
				r[i*n+j] = float32(s)
			}
		}
		return r
	}
	ab := f64mat(aVals, bVals, sz, sz, sz)
	ref := f64mat(ab, cVals, sz, sz, sz)

	const atol, rtol = 5e-2, 1e-2
	var maxDiff float64
	nFail := 0
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(ref[i]))
		rel := diff / (math.Abs(float64(ref[i])) + 1e-8)
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > atol && rel > rtol {
			nFail++
		}
	}
	t.Logf("f16 chained 32×32 matmul (A@B@C): max-abs-diff=%.6e, failures=%d/%d (atol=%.0e, rtol=%.0e)",
		maxDiff, nFail, sz*sz, atol, rtol)
	if nFail > 0 {
		t.Errorf("%d/%d elements exceed tolerances (atol=%.0e, rtol=%.0e)", nFail, sz*sz, atol, rtol)
	}
}

// TestF16_CastRoundTrip tests that f16→f32→f16 produces bit-identical values for
// numbers that are exactly representable in f16. This is a sanity check on the
// cast lowering (explicit f32()/f16() WGSL conversions), not a precision test.
func TestF16_CastRoundTrip(t *testing.T) {
	dev := requireDevice(t)
	requireShaderF16(t, dev)

	// Values that are exact in f16 (representable without rounding).
	vals := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.5, 0.00390625}

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{int64(len(vals))}, uop.Dtypes.Float16, "webgpu")
	xf32 := x.Cast(uop.Dtypes.Float32)
	xf16 := xf32.Cast(uop.Dtypes.Float16)

	items := makeSchedule(t, "webgpu", xf16)
	inputs := map[uint32][]float32{x.Node().Index(): vals}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)

	for i, v := range got {
		if v != vals[i] {
			t.Errorf("cast round-trip[%d]: got %v, want %v (f16→f32→f16 must be bit-identical for representable values)", i, v, vals[i])
		}
	}
	t.Logf("f16 cast round-trip: bit-identical for %d representable values", len(vals))
}

// TestBF16_ElementwiseAdd tests bf16 elementwise addition on the GPU.
// bf16 is stored as u32 (high 16 bits = bf16 bits, low 16 zeroed). Arithmetic
// runs in f32 on the GPU; results are truncated back to bf16 at store time.
// Tolerance: atol=5e-3 — bf16 ULP for values in [-1,1] is ~7.8e-3/128 ≈ 6e-5,
// but two additions and the output truncation can accumulate to ~3e-3.
func TestBF16_ElementwiseAdd(t *testing.T) {
	dev := requireDevice(t)

	rng := rand.New(rand.NewSource(17))
	const n = 512
	aVals := make([]float32, n)
	bVals := make([]float32, n)
	for i := range aVals {
		aVals[i] = float32(rng.Float64()*2 - 1) // [-1, 1]
		bVals[i] = float32(rng.Float64()*2 - 1)
	}

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.BFloat16, "webgpu")
	y := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.BFloat16, "webgpu")
	z := x.Add(y)

	items := makeSchedule(t, "webgpu", z)
	inputs := map[uint32][]float32{
		x.Node().Index(): aVals,
		y.Node().Index(): bVals,
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)

	// Reference: truncate inputs to bf16, add in f32, truncate output to bf16.
	// This mirrors exactly what the GPU does: bitcast<f32>(u32_upload) → add → bitcast<u32>(&mask).
	bf16Trunc := func(v float32) float32 {
		return math.Float32frombits(math.Float32bits(v) & 0xFFFF0000)
	}
	const atol = 1e-6 // bit-identical: same truncation on both sides
	var maxDiff float64
	nFail := 0
	for i := range got {
		ref := bf16Trunc(bf16Trunc(aVals[i]) + bf16Trunc(bVals[i]))
		diff := math.Abs(float64(got[i]) - float64(ref))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > atol {
			nFail++
		}
	}
	t.Logf("bf16 elementwise add [%d]: max-abs-diff=%.6e, failures=%d/%d",
		n, maxDiff, nFail, n)
	if nFail > 0 {
		t.Errorf("%d/%d elements differ from bf16-exact reference; max-abs-diff=%.6e", nFail, n, maxDiff)
	}
}

// TestBF16_RoundTrip verifies that bf16→f32→bf16 is bit-identical for values
// exactly representable in bf16 (which is all values, since bf16 = f32 high 16 bits).
// This checks the upload (float32sToBF16U32Bytes) and readback paths agree with
// the in-shader bitcast<f32>/bitcast<u32> operations.
func TestBF16_RoundTrip(t *testing.T) {
	dev := requireDevice(t)

	vals := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.5, 0.015625}

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{int64(len(vals))}, uop.Dtypes.BFloat16, "webgpu")
	xf32 := x.Cast(uop.Dtypes.Float32)
	xbf16 := xf32.Cast(uop.Dtypes.BFloat16)

	items := makeSchedule(t, "webgpu", xbf16)
	inputs := map[uint32][]float32{x.Node().Index(): vals}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)

	for i, v := range got {
		// bf16 round-trip: f32 → truncate mantissa → f32; must match bf16RoundTrip.
		want := math.Float32frombits(math.Float32bits(vals[i]) & 0xFFFF0000)
		if v != want {
			t.Errorf("bf16 round-trip[%d]: got %v, want %v", i, v, want)
		}
	}
	t.Logf("bf16 round-trip: %d values checked", len(vals))
}

// TestF16_FailClosed verifies that when HasShaderF16 is false, executing an f16
// kernel returns an error — not a GPU crash or silent incorrect output.
// The fail-closed check is in runLocked before any GPU resource allocation.
func TestF16_FailClosed(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float16, "webgpu")
	y := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float16, "webgpu")
	z := x.Add(y)

	items := makeSchedule(t, "webgpu", z)
	inputs := map[uint32][]float32{
		x.Node().Index(): {1, 2, 3, 4, 5, 6, 7, 8},
		y.Node().Index(): {1, 2, 3, 4, 5, 6, 7, 8},
	}

	// Force fail-closed: pretend adapter does not expose shader-f16.
	dev.HasShaderF16 = false
	_, err := dev.Run(items, inputs)
	dev.HasShaderF16 = true // restore for any subsequent tests sharing this device

	if err == nil {
		t.Fatal("Run must fail when HasShaderF16=false and kernel uses f16 buffers")
	}
	t.Logf("fail-closed error (correct): %v", err)
}
