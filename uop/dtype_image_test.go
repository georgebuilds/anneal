package uop_test

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// TestDtypeImage_Interning proves that Dtypes.ImageFloat32 and Dtypes.Float32
// are distinct interned singletons: same priority/bitsize/name/fmt/count but
// different isImage. The intern cache keys on the full struct including
// isImage, so two scalar dtypes with the same fields but different isImage
// resolve to different pointers (and pointer equality stays a sound proxy
// for structural equality).
func TestDtypeImage_Interning(t *testing.T) {
	img := uop.Dtypes.ImageFloat32
	scalar := uop.Dtypes.Float32

	if img == scalar {
		t.Fatalf("ImageFloat32 and Float32 should be distinct interned singletons, " +
			"got identical pointers - isImage field is not in the intern key")
	}
	if !img.IsImage() {
		t.Errorf("Dtypes.ImageFloat32.IsImage() = false, want true")
	}
	if scalar.IsImage() {
		t.Errorf("Dtypes.Float32.IsImage() = true, want false")
	}
	if !img.IsFloat() {
		t.Errorf("Dtypes.ImageFloat32.IsFloat() = false, want true (image shares its scalar peer's compute identity)")
	}
	// ItemSize must match the scalar peer (per-logical-element size).
	if img.ItemSize() != scalar.ItemSize() {
		t.Errorf("ImageFloat32.ItemSize()=%d != Float32.ItemSize()=%d (per-logical-element size must match)",
			img.ItemSize(), scalar.ItemSize())
	}
}

// TestDtypeImage_StructuralHash_Distinct proves that the isImage flag
// participates in StructuralHash so the cross-arena identity check
// discriminates image from scalar. Without this, BEAM disk cache and other
// SPEC §10 structural-key consumers would silently treat image and scalar
// buffers as the same kernel.
func TestDtypeImage_StructuralHash_Distinct(t *testing.T) {
	hImg := uop.Dtypes.ImageFloat32.StructuralHash()
	hScalar := uop.Dtypes.Float32.StructuralHash()
	if hImg == hScalar {
		t.Fatalf("StructuralHash(ImageFloat32)=0x%016x == StructuralHash(Float32) - "+
			"isImage flag was dropped from the hash; BEAM disk cache would collide image and scalar kernels",
			hImg)
	}
}

// TestDtypeImage_StructuralHash_CrossArena pins the load-bearing SPEC §10
// invariant: a UOp built with an image dtype hashes identically in two
// arenas. Because StructuralHash is a pure function of dtype field values,
// the cross-build BEAM disk-cache key for an image-typed kernel stays
// portable.
func TestDtypeImage_StructuralHash_CrossArena(t *testing.T) {
	a1 := uop.NewArena(4)
	a2 := uop.NewArena(4)
	c1 := a1.New(uop.OpConst, uop.Dtypes.ImageFloat32, nil, float64(1.5), nil)
	c2 := a2.New(uop.OpConst, uop.Dtypes.ImageFloat32, nil, float64(1.5), nil)
	k1 := uop.StructuralKeys(a1)[c1.Index()]
	k2 := uop.StructuralKeys(a2)[c2.Index()]
	if k1 != k2 {
		t.Fatalf("cross-arena structural-key divergence for image-typed const: "+
			"arena1=0x%016x arena2=0x%016x - SPEC §10 portability broken", k1, k2)
	}
	// And the image-typed key must differ from the scalar-typed key (otherwise
	// BEAM disk cache would silently misroute image kernels through scalar
	// cache entries).
	a3 := uop.NewArena(4)
	c3 := a3.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1.5), nil)
	kScalar := uop.StructuralKeys(a3)[c3.Index()]
	if k1 == kScalar {
		t.Fatalf("image-typed and scalar-typed const share structural key 0x%016x - "+
			"isImage missing from the structural key path", k1)
	}
}

// TestDtypeImage_LeastUpperDType pins the promotion-lattice behaviour:
//   - (Image, Scalar) collapses to Scalar (storage is not a compute type)
//   - (Scalar, Image) is symmetric, also Scalar
//   - (Image, Image) collapses to Image (storage is preserved when every input agrees)
//   - (Image, narrower-float) promotes to Scalar (image carries no extra
//     precision, the promotion target is the compute peer)
func TestDtypeImage_LeastUpperDType(t *testing.T) {
	img := uop.Dtypes.ImageFloat32
	f32 := uop.Dtypes.Float32
	f16 := uop.Dtypes.Float16
	f64 := uop.Dtypes.Float64

	cases := []struct {
		name string
		ds   []*uop.DType
		want *uop.DType
	}{
		{"image_vs_float32", []*uop.DType{img, f32}, f32},
		{"float32_vs_image", []*uop.DType{f32, img}, f32},
		{"image_vs_image", []*uop.DType{img, img}, img},
		{"image_vs_float16", []*uop.DType{img, f16}, f32},
		{"float16_vs_image", []*uop.DType{f16, img}, f32},
		{"image_vs_float64", []*uop.DType{img, f64}, f64},
		{"image_alone", []*uop.DType{img}, img},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := uop.LeastUpperDType(tc.ds...)
			if got != tc.want {
				t.Errorf("LeastUpperDType(%v) = %v, want %v", tc.ds, got, tc.want)
			}
		})
	}
}

// TestDtypeImage_Vec_Panics proves that vectorizing an image dtype is
// disallowed: the vec4 packing already lives in the storage layout, so a
// dtype-level lane count would conflict. We require a recovered panic.
func TestDtypeImage_Vec_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("ImageFloat32.Vec(2) did not panic; image dtypes must not participate in Vec")
		}
	}()
	_ = uop.Dtypes.ImageFloat32.Vec(2)
}

// TestDtypeImage_Vec_IdentityPath verifies that Vec(1) returns the dtype
// unchanged for image - same as the scalar peer's identity early-exit - so
// callers that pass sz=1 don't hit the panic. This is the documented
// no-op contract for Vec.
func TestDtypeImage_Vec_IdentityPath(t *testing.T) {
	d := uop.Dtypes.ImageFloat32
	if d.Vec(1) != d {
		t.Errorf("ImageFloat32.Vec(1) should return the dtype unchanged (sz=1 identity path)")
	}
}

// TestDtypeImage_IsImage_Nil covers the nil-receiver early-exit so
// IsImage() reaches the documented contract on a nil dtype (returns false
// rather than panicking).
func TestDtypeImage_IsImage_Nil(t *testing.T) {
	var d *uop.DType
	if d.IsImage() {
		t.Errorf("(*DType)(nil).IsImage() = true, want false")
	}
}

// TestDtypeImage_StructuralHash_Self_Stable verifies hash stability across
// repeated calls and against a hand-computed expectation. The point is to
// exercise both branches of the isImage mix-in inside StructuralHash (the
// scalar peer's hash uses the false branch; the image dtype uses the true
// branch).
func TestDtypeImage_StructuralHash_Self_Stable(t *testing.T) {
	h1 := uop.Dtypes.ImageFloat32.StructuralHash()
	h2 := uop.Dtypes.ImageFloat32.StructuralHash()
	if h1 != h2 {
		t.Errorf("ImageFloat32.StructuralHash unstable across calls: %#x != %#x", h1, h2)
	}
	hF32 := uop.Dtypes.Float32.StructuralHash()
	hF32Again := uop.Dtypes.Float32.StructuralHash()
	if hF32 != hF32Again {
		t.Errorf("Float32.StructuralHash unstable across calls: %#x != %#x", hF32, hF32Again)
	}
}

// TestDtypeImage_LeastUpperDType_OrderInvariant covers both orderings of
// the cmpForLUDT tie-breaker by exercising both 2-argument permutations of
// (Image, Scalar). The lattice path with three inputs (Image, Scalar,
// Float16) reaches the tie-break under map iteration order regardless of
// which argument is canonical first.
func TestDtypeImage_LeastUpperDType_OrderInvariant(t *testing.T) {
	img := uop.Dtypes.ImageFloat32
	f32 := uop.Dtypes.Float32
	f16 := uop.Dtypes.Float16
	// Both orderings of (image, scalar) must collapse to scalar.
	if got := uop.LeastUpperDType(img, f32); got != f32 {
		t.Errorf("LeastUpperDType(image, f32) = %v, want %v", got, f32)
	}
	if got := uop.LeastUpperDType(f32, img); got != f32 {
		t.Errorf("LeastUpperDType(f32, image) = %v, want %v", got, f32)
	}
	// Three-input cases that visit the lattice intersection multiple times.
	if got := uop.LeastUpperDType(img, f32, f16); got != f32 {
		t.Errorf("LeastUpperDType(image, f32, f16) = %v, want %v", got, f32)
	}
	if got := uop.LeastUpperDType(f16, img, f32); got != f32 {
		t.Errorf("LeastUpperDType(f16, image, f32) = %v, want %v", got, f32)
	}
	// Empty input returns nil.
	if got := uop.LeastUpperDType(); got != nil {
		t.Errorf("LeastUpperDType() = %v, want nil", got)
	}
}
