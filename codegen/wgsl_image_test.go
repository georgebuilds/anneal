package codegen_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// buildImageElementwiseSchedule builds a one-kernel schedule that copies an
// image-typed input to an image-typed output via a trivial elementwise op,
// chosen so the kernel exercises both the image-load and the image-store
// codegen paths. We use Mul by a scalar Const so the result tensor stays
// shaped like the input (no reduction, no broadcast) and the body is a
// single ALU node feeding the store.
func buildImageElementwiseSchedule(t *testing.T) (schedule.ExecItem, *tensor.Tensor) {
	t.Helper()
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.ImageFloat32, "webgpu")
	two := tensor.ConstScalar(a, 2.0, uop.Dtypes.ImageFloat32, "webgpu")
	out := x.Mul(two)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatalf("CreateSchedule returned 0 items for image elementwise")
	}
	return items[0], out
}

// TestWGSL_ImageDType_BufferBinding pins the storage-buffer binding format:
// an image-typed buffer must be rendered as `array<vec4<f32>>`, not
// `array<f32>`. The wgsl emitter routes through codegen.WGSLTypeInfoFor;
// this test guards that path against regressions that would silently fall
// back to the scalar storage layout.
func TestWGSL_ImageDType_BufferBinding(t *testing.T) {
	item, _ := buildImageElementwiseSchedule(t)
	wgsl := codegen.RenderWGSL(item).WGSL
	if !strings.Contains(wgsl, "array<vec4<f32>>") {
		t.Errorf("image kernel WGSL missing array<vec4<f32>> binding\nfull shader:\n%s", wgsl)
	}
	if strings.Contains(wgsl, "array<f32>") {
		t.Errorf("image kernel WGSL still contains scalar array<f32> binding (image fork missed)\nfull shader:\n%s", wgsl)
	}
	// Output and input bindings must both be image (the schedule has at least one
	// image input plus an image output).
	if !strings.Contains(wgsl, "var<storage, read_write> data0: array<vec4<f32>>") {
		t.Errorf("output binding is not array<vec4<f32>>\nfull shader:\n%s", wgsl)
	}
}

// TestWGSL_ImageDType_LoadStoreEmit pins the per-element addressing form:
// every load from an image-typed buffer must emit the vec4-packed
// `data{i}[u32(idx) / 4u].{x,y,z,w}` select chain, and the output side must
// address vec4 slots (the slot-dispatch store writes data0[gid_x] where
// gid_x is the slot index; the per-lane flat index carries the * 4u/% 4u
// packing). If these patterns regress to bare `data{i}[idx]` the kernel
// would index vec4 slots as if they were individual scalars, producing 4x
// out-of-range loads and wrong stores.
func TestWGSL_ImageDType_LoadStoreEmit(t *testing.T) {
	item, _ := buildImageElementwiseSchedule(t)
	wgsl := codegen.RenderWGSL(item).WGSL
	if !strings.Contains(wgsl, "/ 4u") || !strings.Contains(wgsl, "% 4u") {
		t.Errorf("image kernel WGSL missing vec4 packing (/ 4u or %% 4u)\nfull shader:\n%s", wgsl)
	}
	// At least one of the form data{i}[(...)/4u][(...) %4u] must appear for
	// the input load and the store; we check by substring count.
	if cnt := strings.Count(wgsl, "/ 4u"); cnt < 2 {
		t.Errorf("expected at least 2 vec4-packed accesses (one load + one store), got %d\nfull shader:\n%s",
			cnt, wgsl)
	}
}

// TestWGSL_ImageDType_NormalizeWGSL_Stable is the SPEC §10 cross-arena
// byte-identity proof for the image codegen path. Two arenas building the
// same image kernel must produce identical normalized WGSL so the BEAM disk
// cache keyed on FNV-64a(normalizeWGSL(...)) survives process restarts. The
// image fork adds the `data{i}[(...)/4u][(...) %4u]` form; this test catches
// any new per-arena-varying identifier introduced by that path.
func TestWGSL_ImageDType_NormalizeWGSL_Stable(t *testing.T) {
	build := func() string {
		item, _ := buildImageElementwiseSchedule(t)
		return codegen.RenderWGSL(item).WGSL
	}
	w1 := build()
	w2 := build()
	h1 := codegen.BeamWGSLHash(w1)
	h2 := codegen.BeamWGSLHash(w2)
	if h1 != h2 {
		t.Fatalf("normalizeWGSL not byte-stable across arenas for image kernel: "+
			"hash1=%s hash2=%s — SPEC §10 BEAM disk-cache invariant violated.\n"+
			"shader1:\n%s\nshader2:\n%s", h1, h2, w1, w2)
	}
}

// TestWGSL_ImageDType_LoadStoreEmit_NonImage_Bypass exercises the scalar
// path: a Float32 kernel must not emit any vec4 packing — the image fork
// is gated on dtype.IsImage and must stay off for scalar tensors.
func TestWGSL_ImageDType_LoadStoreEmit_NonImage_Bypass(t *testing.T) {
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	two := tensor.ConstScalar(a, 2.0, uop.Dtypes.Float32, "webgpu")
	out := x.Mul(two)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	wgsl := codegen.RenderWGSL(items[0]).WGSL
	if strings.Contains(wgsl, "/ 4u") || strings.Contains(wgsl, "vec4<f32>") {
		t.Errorf("scalar f32 kernel should not emit vec4 packing — image fork triggered on a non-image dtype.\nfull shader:\n%s", wgsl)
	}
}

// TestWGSL_ImageDType_BufferByteSize pins the per-dtype byte-size helper
// used by every GPU allocator site: image buffers consume ceil(n/4) * 16
// bytes, not n * 4. We check the tail-padding corner case (n=17 rounds up
// to 5 vec4 slots = 80 bytes) so a future regression that drops the
// ceiling would surface here.
func TestWGSL_ImageDType_BufferByteSize(t *testing.T) {
	cases := []struct {
		n    int64
		want uint64
	}{
		{0, 0},
		{1, 16},
		{4, 16},
		{5, 32},
		{16, 64},
		{17, 80}, // 5 vec4 slots; the load-bearing tail-padding case
	}
	for _, tc := range cases {
		got := codegen.BufferByteSize(tc.n, uop.Dtypes.ImageFloat32)
		if got != tc.want {
			t.Errorf("BufferByteSize(%d, ImageFloat32) = %d, want %d (vec4 packing: ceil(n/4)*16)",
				tc.n, got, tc.want)
		}
	}
	// Scalar float32 still bills per logical element.
	if got := codegen.BufferByteSize(17, uop.Dtypes.Float32); got != 68 {
		t.Errorf("BufferByteSize(17, Float32) = %d, want 68 (per-element 4 bytes)", got)
	}
}
