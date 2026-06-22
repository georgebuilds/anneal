package codegen_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── fp8 structural WGSL tests (no GPU required) ───────────────────────────────
//
// fp8 (e4m3fn / e5m2) is storage-only with f32 compute, on the bf16 decoded-
// storage scheme: the u32 storage word holds the quantized value's full f32
// bit pattern, so loads are a free bitcast<f32> and the RTNE narrowing to the
// fp8 grid happens once, at the store boundary, via the _fp8*_rtne_bits
// prelude helpers. No device extension is involved (unlike f16's shader-f16).

// fp8WGSLFor builds a single-kernel schedule for fn(arena) and returns its WGSL.
func fp8WGSLFor(t *testing.T, build func(a *uop.Arena) *tensor.Tensor) string {
	t.Helper()
	a := newArena()
	out := build(a)
	items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
	if len(items) == 0 {
		t.Fatal("no schedule items")
	}
	res, err := codegen.CompileWGSL(items[0])
	if err != nil {
		t.Fatalf("CompileWGSL: %v", err)
	}
	return res.WGSL
}

// TestFP8_WGSL verifies fp8 elementwise kernels generate correct WGSL for
// both formats:
// - buffers declared as array<u32> (decoded storage; WGSL has no fp8 type)
// - loads widen via bitcast<f32> (free - the storage word is f32 bits)
// - stores narrow via the format's _fp8*_rtne_bits helper
// - no "enable f16;" directive (fp8 needs no device extension)
func TestFP8_WGSL(t *testing.T) {
	for _, tc := range []struct {
		name   string
		dtype  *uop.DType
		helper string
		other  string
	}{
		{"e4m3fn", uop.Dtypes.FP8E4M3, "_fp8e4m3_rtne_bits(", "_fp8e5m2_rtne_bits("},
		{"e5m2", uop.Dtypes.FP8E5M2, "_fp8e5m2_rtne_bits(", "_fp8e4m3_rtne_bits("},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wgsl := fp8WGSLFor(t, func(a *uop.Arena) *tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{8}, tc.dtype, "webgpu")
				y := tensor.NewLeaf(a, []int64{8}, tc.dtype, "webgpu")
				return x.Add(y)
			})
			t.Logf("%s elementwise WGSL:\n%s", tc.name, wgsl)

			assertContains(t, wgsl, "array<u32>")
			assertNotContains(t, wgsl, "enable f16;")
			assertContains(t, wgsl, "bitcast<f32>(")
			assertContains(t, wgsl, tc.helper)
			// The prelude gate must emit only the output format's helper.
			assertNotContains(t, wgsl, tc.other)
		})
	}
}

// TestFP8_ReduceWGSL verifies an fp8 sum-reduce uses an f32 accumulator and
// narrows the result through the RTNE helper only at store time.
func TestFP8_ReduceWGSL(t *testing.T) {
	wgsl := fp8WGSLFor(t, func(a *uop.Arena) *tensor.Tensor {
		x := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.FP8E4M3, "webgpu")
		return x.Sum([]int{0}, false)
	})
	t.Logf("fp8 e4m3 reduce WGSL:\n%s", wgsl)

	assertContains(t, wgsl, "var acc0: f32")
	assertContains(t, wgsl, "fn _fp8e4m3_rtne_bits(", "_fp8e4m3_rtne_bits(acc0)")
	assertNotContains(t, wgsl, "enable f16;")
}

// TestFP8_F32KernelUnchanged guards the static path: a pure-f32 kernel must
// not pick up fp8 preludes or u32 buffer types from the fp8 changes.
func TestFP8_F32KernelUnchanged(t *testing.T) {
	wgsl := fp8WGSLFor(t, func(a *uop.Arena) *tensor.Tensor {
		x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "webgpu")
		return x.Exp2()
	})
	assertContains(t, wgsl, "array<f32>")
	if strings.Contains(wgsl, "_fp8e4m3_rtne_bits") || strings.Contains(wgsl, "_fp8e5m2_rtne_bits") {
		t.Errorf("f32 kernel must not contain fp8 preludes\nshader:\n%s", wgsl)
	}
}
