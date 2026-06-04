package onnx

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// WarnFunc is the type of the package-level callback invoked on lossy or
// otherwise notable decisions during initializer decoding (e.g. DOUBLE
// downcast to f32). Tests overwrite Warn to assert that the warning fired.
// Default: log via the standard log package, prefixed "onnx:".
type WarnFunc func(format string, args ...any)

// Warn is the package-level warning callback. Default implementation forwards
// to the standard library log.Printf. Tests may swap it for capture.
var Warn WarnFunc = defaultWarn

func defaultWarn(format string, args ...any) {
	// Avoid importing log at top-level to keep the default cheap; we still
	// want to surface warnings visibly, so go through fmt to stderr.
	fmt.Printf("onnx: "+format+"\n", args...)
}

// onnxDType maps an ONNX TensorProto.DataType to anneal's *uop.DType, plus the
// decoded element byte width of the source. ok=false signals an unsupported
// dtype that the caller must reject.
//
// Per plan §3 dtype table:
//   - FLOAT (1) -> f32; FLOAT16 (10) -> f16; BFLOAT16 (16) -> bf16
//   - DOUBLE (11) -> f32 with a warning (WebGPU has no f64)
//   - INT8/16/32 -> i8/i16/i32; UINT8/16/32 -> u8/u16/u32
//   - INT64 (7) -> i32 with overflow trap at decode time
//   - BOOL (9) -> bool
//   - STRING / COMPLEX64/128 / FLOAT8* / UINT4 / INT4 / UNDEFINED -> unsupported
func onnxDType(dt int32) (annealDT *uop.DType, srcWidth int, downcast bool, ok bool) {
	switch onnxpb.TensorProto_DataType(dt) {
	case onnxpb.TensorProto_FLOAT:
		return uop.Dtypes.Float32, 4, false, true
	case onnxpb.TensorProto_FLOAT16:
		return uop.Dtypes.Float16, 2, false, true
	case onnxpb.TensorProto_BFLOAT16:
		return uop.Dtypes.BFloat16, 2, false, true
	case onnxpb.TensorProto_DOUBLE:
		// Decoded width is 8 (f64), but we materialise as f32 — warn the
		// caller so the downcast is visible.
		return uop.Dtypes.Float32, 8, true, true
	case onnxpb.TensorProto_INT8:
		return uop.Dtypes.Int8, 1, false, true
	case onnxpb.TensorProto_INT16:
		return uop.Dtypes.Int16, 2, false, true
	case onnxpb.TensorProto_INT32:
		return uop.Dtypes.Int32, 4, false, true
	case onnxpb.TensorProto_INT64:
		// Per plan §3 int64 policy: device-tier int64 doesn't exist on WGSL.
		// Decode as int32 with an overflow trap; the trap happens in
		// decodeRawData.
		return uop.Dtypes.Int32, 8, true, true
	case onnxpb.TensorProto_UINT8:
		return uop.Dtypes.UInt8, 1, false, true
	case onnxpb.TensorProto_UINT16:
		return uop.Dtypes.UInt16, 2, false, true
	case onnxpb.TensorProto_UINT32:
		return uop.Dtypes.UInt32, 4, false, true
	case onnxpb.TensorProto_UINT64:
		// WGSL has no u64. Mirror INT64: downcast to UInt32 with an overflow
		// trap in decodeRawData (any value > MaxUInt32 errors).
		return uop.Dtypes.UInt32, 8, true, true
	case onnxpb.TensorProto_BOOL:
		return uop.Dtypes.Bool, 1, false, true
	}
	return nil, 0, false, false
}

// tensorFromProto decodes an ONNX TensorProto initializer into an anneal leaf
// tensor on the given arena. The result is a *tensor.Tensor whose SetData has
// been called with the decoded float32 payload (anneal's host-side data
// representation; SetData handles f16/bf16 quantisation).
//
// Decoding rules (plan §3):
//   - All raw_data is little-endian per the ONNX spec; we decode via
//     encoding/binary.LittleEndian, which is correct on both LE and BE hosts.
//     (The optional unsafe.Slice fast path is deliberately NOT the default —
//     it requires both LE-host and natural-alignment checks; we defer it.)
//   - Each TensorProto must populate exactly one data field: raw_data, OR a
//     type-matched typed field (float_data, int32_data, int64_data,
//     double_data, uint64_data, string_data).
//   - STRING / COMPLEX / FLOAT8 / UINT4 / INT4 / UNDEFINED are rejected.
//   - DOUBLE input is downcast to f32 with a Warn() call.
//   - INT64 input is downcast to int32 with an overflow trap that errors
//     loudly when any element falls outside [math.MinInt32, math.MaxInt32].
//   - External-data initializers are rejected (out of v1 scope; plan §9).
//
// The leaf is NOT interned at this layer; the Runner's initializer interner
// hashes the (dtype, dims, raw bytes) tuple and reuses the same leaf when
// two TensorProtos are structurally identical.
func tensorFromProto(arena *uop.Arena, tp *onnxpb.TensorProto, device string) (*tensor.Tensor, error) {
	if tp == nil {
		return nil, fmt.Errorf("onnx: nil TensorProto")
	}
	if tp.GetDataLocation() == onnxpb.TensorProto_EXTERNAL {
		return nil, fmt.Errorf("onnx: external-data initializer %q not supported in v1", tp.GetName())
	}

	annealDT, srcWidth, downcast, ok := onnxDType(tp.GetDataType())
	if !ok {
		return nil, fmt.Errorf("onnx: unsupported dtype %d for initializer %q",
			tp.GetDataType(), tp.GetName())
	}

	if downcast && onnxpb.TensorProto_DataType(tp.GetDataType()) == onnxpb.TensorProto_DOUBLE {
		Warn("initializer %q DOUBLE downcast to f32 (WebGPU has no f64)", tp.GetName())
	}

	// Validate dims (per plan §1 device-tier requires concrete dims; symbolic
	// dims live only on inputs, never on initializers).
	dims := tp.GetDims()
	for _, d := range dims {
		if d < 0 {
			return nil, fmt.Errorf("onnx: initializer %q has negative dim %d", tp.GetName(), d)
		}
	}
	shape := make([]int64, len(dims))
	copy(shape, dims)

	// Compute element count.
	elems := int64(1)
	for _, d := range shape {
		elems *= d
	}
	data, err := decodeTensorData(tp, int(elems), srcWidth)
	if err != nil {
		return nil, fmt.Errorf("onnx: initializer %q: %w", tp.GetName(), err)
	}

	leaf := tensor.NewLeaf(arena, shape, annealDT, device)
	leaf.SetData(data)
	return leaf, nil
}

// structureOnlyLeafFromProto decodes the (dtype, dims) header of a TensorProto
// and returns a leaf tensor with the correct shape + dtype but no host-side
// payload. SetData is NOT called, so Data() returns nil and the arena leaf
// slot stays unset. This is the WithStructureOnly fast path: we still need
// the shape inference and graph topology to be correct, but we never look at
// the values. Subsequent Run() will fail loudly because payloads aren't
// there; the contract is documented at WithStructureOnly().
//
// Decoding cost: just the header (a handful of int64s + a small dtype switch).
// The raw_data bytes never get scanned, decoded, or copied. On models like
// ResNet-9 (~482 KB total) this is a small win; on multi-megabyte transformer
// weights the savings are the whole point.
func structureOnlyLeafFromProto(arena *uop.Arena, tp *onnxpb.TensorProto, device string) (*tensor.Tensor, error) {
	if tp == nil {
		return nil, fmt.Errorf("onnx: nil TensorProto")
	}
	if tp.GetDataLocation() == onnxpb.TensorProto_EXTERNAL {
		return nil, fmt.Errorf("onnx: external-data initializer %q not supported in v1", tp.GetName())
	}
	annealDT, _, _, ok := onnxDType(tp.GetDataType())
	if !ok {
		return nil, fmt.Errorf("onnx: unsupported dtype %d for initializer %q",
			tp.GetDataType(), tp.GetName())
	}
	dims := tp.GetDims()
	for _, d := range dims {
		if d < 0 {
			return nil, fmt.Errorf("onnx: initializer %q has negative dim %d", tp.GetName(), d)
		}
	}
	sh := make([]int64, len(dims))
	copy(sh, dims)
	// NewLeaf reserves an arena slot but does NOT set host data; Data() will
	// return nil. We deliberately skip SetData here.
	return tensor.NewLeaf(arena, sh, annealDT, device), nil
}

// decodeTensorData materialises the TensorProto payload as a []float32 that
// matches anneal's host representation (SetData accepts []float32 regardless
// of dtype). It dispatches on the source dtype and prefers the typed field
// when present, else falls back to raw_data.
func decodeTensorData(tp *onnxpb.TensorProto, elems, srcWidth int) ([]float32, error) {
	dt := onnxpb.TensorProto_DataType(tp.GetDataType())

	switch dt {
	case onnxpb.TensorProto_FLOAT:
		if len(tp.GetFloatData()) > 0 {
			return copyFloat32(tp.GetFloatData(), elems)
		}
		return decodeRawFloat32(tp.GetRawData(), elems)
	case onnxpb.TensorProto_DOUBLE:
		// DOUBLE: typed field is double_data; otherwise raw is 8-byte LE.
		if len(tp.GetDoubleData()) > 0 {
			out := make([]float32, len(tp.GetDoubleData()))
			for i, v := range tp.GetDoubleData() {
				out[i] = float32(v)
			}
			if elems > 0 && len(out) != elems {
				return nil, fmt.Errorf("double_data length %d != elem count %d", len(out), elems)
			}
			return out, nil
		}
		return decodeRawFloat64ToFloat32(tp.GetRawData(), elems)
	case onnxpb.TensorProto_FLOAT16:
		// float16 lives in int32_data (per spec) bit-packed in the low 16
		// bits, OR in raw_data as 2 bytes per element. We unpack to
		// float32 host values; SetData re-quantises to fit anneal's f16.
		if len(tp.GetInt32Data()) > 0 {
			out := make([]float32, len(tp.GetInt32Data()))
			for i, v := range tp.GetInt32Data() {
				out[i] = float16Bits(uint16(v))
			}
			if elems > 0 && len(out) != elems {
				return nil, fmt.Errorf("int32_data length %d != elem count %d", len(out), elems)
			}
			return out, nil
		}
		return decodeRawFloat16(tp.GetRawData(), elems)
	case onnxpb.TensorProto_BFLOAT16:
		if len(tp.GetInt32Data()) > 0 {
			out := make([]float32, len(tp.GetInt32Data()))
			for i, v := range tp.GetInt32Data() {
				out[i] = bfloat16Bits(uint16(v))
			}
			if elems > 0 && len(out) != elems {
				return nil, fmt.Errorf("int32_data length %d != elem count %d", len(out), elems)
			}
			return out, nil
		}
		return decodeRawBFloat16(tp.GetRawData(), elems)
	case onnxpb.TensorProto_INT8, onnxpb.TensorProto_INT16, onnxpb.TensorProto_INT32,
		onnxpb.TensorProto_UINT8, onnxpb.TensorProto_UINT16, onnxpb.TensorProto_UINT32,
		onnxpb.TensorProto_BOOL:
		// int32_data is the typed field for all of these; raw_data is the
		// fallback packing.
		if len(tp.GetInt32Data()) > 0 {
			out := make([]float32, len(tp.GetInt32Data()))
			for i, v := range tp.GetInt32Data() {
				out[i] = float32(v)
			}
			if elems > 0 && len(out) != elems {
				return nil, fmt.Errorf("int32_data length %d != elem count %d", len(out), elems)
			}
			return out, nil
		}
		return decodeRawIntWidth(tp.GetRawData(), elems, srcWidth, isSignedDType(dt))
	case onnxpb.TensorProto_UINT64:
		// Plan §3 int64 policy applied to uint64: downcast to uint32 at decode
		// time with an explicit overflow trap. Anything > MaxUint32 errors
		// rather than silently losing precision through the float32 host
		// representation.
		if len(tp.GetUint64Data()) > 0 {
			out := make([]float32, len(tp.GetUint64Data()))
			for i, v := range tp.GetUint64Data() {
				if v > math.MaxUint32 {
					return nil, fmt.Errorf("uint64 value %d at index %d overflows uint32 (v1 has no device uint64)", v, i)
				}
				out[i] = float32(v)
			}
			if elems > 0 && len(out) != elems {
				return nil, fmt.Errorf("uint64_data length %d != elem count %d", len(out), elems)
			}
			return out, nil
		}
		raw := tp.GetRawData()
		if elems > 0 && len(raw) != elems*8 {
			return nil, fmt.Errorf("uint64 raw_data length %d != %d", len(raw), elems*8)
		}
		out := make([]float32, len(raw)/8)
		for i := range out {
			v := binary.LittleEndian.Uint64(raw[i*8:])
			if v > math.MaxUint32 {
				return nil, fmt.Errorf("uint64 value %d at index %d overflows uint32 (v1 has no device uint64)", v, i)
			}
			out[i] = float32(v)
		}
		return out, nil
	case onnxpb.TensorProto_INT64:
		// Plan §3 int64 policy: downcast to i32 at decode time with an
		// explicit overflow trap.
		if len(tp.GetInt64Data()) > 0 {
			out := make([]float32, len(tp.GetInt64Data()))
			for i, v := range tp.GetInt64Data() {
				if v < math.MinInt32 || v > math.MaxInt32 {
					return nil, fmt.Errorf("int64 value %d at index %d overflows int32 (v1 has no device int64)", v, i)
				}
				out[i] = float32(v)
			}
			if elems > 0 && len(out) != elems {
				return nil, fmt.Errorf("int64_data length %d != elem count %d", len(out), elems)
			}
			return out, nil
		}
		raw := tp.GetRawData()
		if elems > 0 && len(raw) != elems*8 {
			return nil, fmt.Errorf("int64 raw_data length %d != %d", len(raw), elems*8)
		}
		out := make([]float32, len(raw)/8)
		for i := range out {
			v := int64(binary.LittleEndian.Uint64(raw[i*8:]))
			if v < math.MinInt32 || v > math.MaxInt32 {
				return nil, fmt.Errorf("int64 value %d at index %d overflows int32 (v1 has no device int64)", v, i)
			}
			out[i] = float32(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("onnx: unsupported dtype %d for raw decode", dt)
}

func copyFloat32(src []float32, elems int) ([]float32, error) {
	if elems > 0 && len(src) != elems {
		return nil, fmt.Errorf("float_data length %d != elem count %d", len(src), elems)
	}
	out := make([]float32, len(src))
	copy(out, src)
	return out, nil
}

func decodeRawFloat32(raw []byte, elems int) ([]float32, error) {
	if elems > 0 && len(raw) != elems*4 {
		return nil, fmt.Errorf("float raw_data length %d != %d", len(raw), elems*4)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

func decodeRawFloat64ToFloat32(raw []byte, elems int) ([]float32, error) {
	if elems > 0 && len(raw) != elems*8 {
		return nil, fmt.Errorf("double raw_data length %d != %d", len(raw), elems*8)
	}
	out := make([]float32, len(raw)/8)
	for i := range out {
		bits := binary.LittleEndian.Uint64(raw[i*8:])
		out[i] = float32(math.Float64frombits(bits))
	}
	return out, nil
}

func decodeRawFloat16(raw []byte, elems int) ([]float32, error) {
	if elems > 0 && len(raw) != elems*2 {
		return nil, fmt.Errorf("float16 raw_data length %d != %d", len(raw), elems*2)
	}
	out := make([]float32, len(raw)/2)
	for i := range out {
		bits := binary.LittleEndian.Uint16(raw[i*2:])
		out[i] = float16Bits(bits)
	}
	return out, nil
}

func decodeRawBFloat16(raw []byte, elems int) ([]float32, error) {
	if elems > 0 && len(raw) != elems*2 {
		return nil, fmt.Errorf("bfloat16 raw_data length %d != %d", len(raw), elems*2)
	}
	out := make([]float32, len(raw)/2)
	for i := range out {
		bits := binary.LittleEndian.Uint16(raw[i*2:])
		out[i] = bfloat16Bits(bits)
	}
	return out, nil
}

// decodeRawIntWidth handles INT8/INT16/INT32/UINT8/UINT16/UINT32/BOOL packed
// in raw_data. width is the source element width in bytes (1/2/4).
func decodeRawIntWidth(raw []byte, elems, width int, signed bool) ([]float32, error) {
	if width <= 0 {
		return nil, fmt.Errorf("invalid src width %d", width)
	}
	if elems > 0 && len(raw) != elems*width {
		return nil, fmt.Errorf("int%d raw_data length %d != %d", width*8, len(raw), elems*width)
	}
	out := make([]float32, len(raw)/width)
	switch width {
	case 1:
		for i := range out {
			b := raw[i]
			if signed {
				out[i] = float32(int8(b))
			} else {
				out[i] = float32(b)
			}
		}
	case 2:
		for i := range out {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			if signed {
				out[i] = float32(int16(u))
			} else {
				out[i] = float32(u)
			}
		}
	case 4:
		for i := range out {
			u := binary.LittleEndian.Uint32(raw[i*4:])
			if signed {
				out[i] = float32(int32(u))
			} else {
				out[i] = float32(u)
			}
		}
	default:
		return nil, fmt.Errorf("unhandled int width %d", width)
	}
	return out, nil
}

func isSignedDType(dt onnxpb.TensorProto_DataType) bool {
	switch dt {
	case onnxpb.TensorProto_INT8, onnxpb.TensorProto_INT16, onnxpb.TensorProto_INT32, onnxpb.TensorProto_INT64:
		return true
	}
	return false
}

// float16Bits decodes an IEEE-754 binary16 to its float32 value.
func float16Bits(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	var f uint32
	switch exp {
	case 0:
		if frac == 0 {
			f = sign << 31
		} else {
			// subnormal: renormalise
			e := uint32(-14 + 127)
			for (frac & 0x400) == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3ff
			f = (sign << 31) | (e << 23) | (frac << 13)
		}
	case 0x1f:
		// inf / NaN
		if frac == 0 {
			f = (sign << 31) | 0x7f800000
		} else {
			f = (sign << 31) | 0x7f800000 | (frac << 13)
		}
	default:
		f = (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13)
	}
	return math.Float32frombits(f)
}

// bfloat16Bits decodes a bf16 (sign+exp matches f32, 7-bit mantissa) to f32 by
// shifting into the upper 16 bits.
func bfloat16Bits(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// initializerHashKey returns a content hash for an initializer so identical
// (dtype, dims, raw bytes) TensorProtos resolve to the same arena leaf
// regardless of their order in the .onnx file. Plan invariant: intern by
// structural identity, not by load position.
func initializerHashKey(tp *onnxpb.TensorProto) [32]byte {
	h := sha256.New()
	// Dtype.
	var b [8]byte
	binary.LittleEndian.PutUint32(b[:4], uint32(tp.GetDataType()))
	h.Write(b[:4])
	// Dims.
	for _, d := range tp.GetDims() {
		binary.LittleEndian.PutUint64(b[:8], uint64(d))
		h.Write(b[:8])
	}
	// Raw payload. We hash every typed field too so that two equivalent
	// initializers expressed via different storage paths still alias.
	h.Write(tp.GetRawData())
	if fd := tp.GetFloatData(); len(fd) > 0 {
		for _, v := range fd {
			binary.LittleEndian.PutUint32(b[:4], math.Float32bits(v))
			h.Write(b[:4])
		}
	}
	if id := tp.GetInt32Data(); len(id) > 0 {
		for _, v := range id {
			binary.LittleEndian.PutUint32(b[:4], uint32(v))
			h.Write(b[:4])
		}
	}
	if id := tp.GetInt64Data(); len(id) > 0 {
		for _, v := range id {
			binary.LittleEndian.PutUint64(b[:8], uint64(v))
			h.Write(b[:8])
		}
	}
	if dd := tp.GetDoubleData(); len(dd) > 0 {
		for _, v := range dd {
			binary.LittleEndian.PutUint64(b[:8], math.Float64bits(v))
			h.Write(b[:8])
		}
	}
	if ud := tp.GetUint64Data(); len(ud) > 0 {
		for _, v := range ud {
			binary.LittleEndian.PutUint64(b[:8], v)
			h.Write(b[:8])
		}
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}
