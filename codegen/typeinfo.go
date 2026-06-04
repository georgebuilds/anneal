package codegen

import "github.com/georgebuilds/anneal/uop"

// WGSLTypeInfo carries per-dtype WGSL metadata in one place: the scalar type
// literal used in expression context, the storage buffer element type (which
// promotes bool and bf16 to u32), and the per-element byte size on the GPU.
//
// Single source of truth for the three previously split lookups (wgslDType,
// wgslBufferElemType, elemBytes). Backend code that needs only the byte size
// reads WGSLTypeInfoFor(d).SizeBytes; the renderer reads .Scalar and .BufferElem.
type WGSLTypeInfo struct {
	Scalar     string
	BufferElem string
	SizeBytes  uint64
}

// WGSLTypeInfoFor returns the WGSL metadata for a dtype. Pointer dtypes are
// unwrapped to their base. Nil and Void are treated as f32 (matches the
// pre-consolidation behaviour of wgslDType, wgslBufferElemType, and elemBytes).
func WGSLTypeInfoFor(d *uop.DType) WGSLTypeInfo {
	if d == nil || d == uop.Dtypes.Void {
		return WGSLTypeInfo{Scalar: "f32", BufferElem: "f32", SizeBytes: 4}
	}
	if d.IsPtr() {
		d = d.Base()
	}
	scalar := d.Scalar()

	bf16Storage := scalar == uop.Dtypes.BFloat16
	imageStorage := d.IsImage()

	var wgslName string
	var sizeBytes uint64 = 4
	switch scalar {
	case uop.Dtypes.Float32:
		wgslName = "f32"
	case uop.Dtypes.Float16:
		wgslName = "f16"
		sizeBytes = 2
	case uop.Dtypes.Int32:
		wgslName = "i32"
	case uop.Dtypes.UInt32:
		wgslName = "u32"
	case uop.Dtypes.Index:
		wgslName = "i32"
	case uop.Dtypes.Bool:
		wgslName = "bool"
	case uop.Dtypes.Int8, uop.Dtypes.Int16:
		wgslName = "i32"
	case uop.Dtypes.UInt8, uop.Dtypes.UInt16:
		wgslName = "u32"
	case uop.Dtypes.Int64:
		// WGSL has no i64; silently downgrade. The principled vmax-driven
		// decision (rules.IndexDtypeForBound) is honored at the per-loop
		// symbolic-bound emission site (InstrLoopBegin emits a comment when
		// the bound's vmax would have required i64). tinygrad PR #8268.
		wgslName = "i32"
	case uop.Dtypes.UInt64:
		wgslName = "u32"
	default:
		if d.IsFloat() {
			wgslName = "f32"
		} else {
			wgslName = "i32"
		}
	}

	buf := wgslName
	if bf16Storage || buf == "bool" {
		buf = "u32"
	}
	if imageStorage {
		// Image dtypes bind their storage buffer as array<vec4<f32>>; the
		// per-logical-element size stays 4 bytes (SizeBytes is per logical
		// f32 element). Byte-size callers (backend allocators) round up to
		// the vec4 packing via BufferByteSize so allocations cover
		// ceil(elems/4) vec4 slots.
		buf = "vec4<f32>"
	}

	return WGSLTypeInfo{Scalar: wgslName, BufferElem: buf, SizeBytes: sizeBytes}
}

// BufferByteSize returns the total GPU buffer byte size for a buffer of
// elems logical elements of dtype d. For image dtypes the buffer holds
// ceil(elems/4) vec4 slots (16 bytes each); for everything else it is the
// pre-image elems * SizeBytes formula every call site previously used.
func BufferByteSize(elems int64, d *uop.DType) uint64 {
	info := WGSLTypeInfoFor(d)
	if d != nil && d.IsImage() {
		// One vec4 slot covers 4 logical f32 elements; round up the slot
		// count via ceiling division and multiply by 16 bytes per slot.
		slots := uint64((elems + 3) / 4)
		return slots * 16
	}
	return uint64(elems) * info.SizeBytes
}
