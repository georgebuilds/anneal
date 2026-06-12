package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/uop"
)

// allocator is the WebGPU implementation of backend.Allocator. It owns the
// slot→buffer map for one orchestration round (one Run / RunSymbolic call)
// and batches dedup'd release at the end via Reset.
//
// Threading: every method touches *wgpu.* and must be called from the GPU-owner
// goroutine.
type allocator struct {
	dev       *Device
	slotBufs  map[int]*deviceBuffer // slot → shared buffer (Phase A intermediates)
	dedicated []*deviceBuffer       // Phase B + Phase C: dedicated outputs / leaves
}

func newAllocator(dev *Device) *allocator {
	return &allocator{
		dev:      dev,
		slotBufs: make(map[int]*deviceBuffer),
	}
}

// Alloc creates a fresh storage buffer sized to hold elems * elemBytes(dt)
// bytes (or ceil(elems/4) * 16 bytes for image dtypes — one vec4 slot per
// 4 logical elements). usage controls the wgpu.BufferUsage flag set (leaf
// inputs skip the CopySrc bit; everything else gets the IO triple).
func (a *allocator) Alloc(elems int64, dt *uop.DType, usage backend.BufferUsage, label string) (backend.DeviceBuffer, error) {
	flags := gputypes.BufferUsageStorage | gputypes.BufferUsageCopyDst
	if usage != backend.BufferUsageLeafInput {
		flags |= gputypes.BufferUsageCopySrc
	}
	buf, err := a.dev.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Usage: flags,
		Size:  bufferByteSize(elems, dt),
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: alloc %q: %w", label, err)
	}
	db := &deviceBuffer{dev: a.dev, buf: buf, elems: elems, dt: dt}
	a.dedicated = append(a.dedicated, db)
	return db, nil
}

// AllocSlot returns the slot-shared buffer for slot, creating it on first use.
// elems is the maximum count across all kernels writing this slot — the
// orchestrator computes this before allocation and passes it on the first
// call; subsequent calls for the same slot simply return the existing buffer
// (the maxElems on later calls is informational only and is not used to
// resize).
func (a *allocator) AllocSlot(slot int, elems int64, dt *uop.DType, label string) (backend.DeviceBuffer, error) {
	if existing, ok := a.slotBufs[slot]; ok {
		return existing, nil
	}
	// Slots historically allocate at 4 bytes per element regardless of dtype
	// because the static path used a uniform 4-byte slot stride. Preserve that
	// for byte-identical behaviour: the WGSL bitcast<f32>(u32) load path for
	// bf16 / int32 intermediates relies on the 4-byte stride.
	const slotElemBytes uint64 = 4
	buf, err := a.dev.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  uint64(elems) * slotElemBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: alloc slot %d (%q): %w", slot, label, err)
	}
	db := &deviceBuffer{dev: a.dev, buf: buf, elems: elems, dt: dt}
	a.slotBufs[slot] = db
	return db, nil
}

// Reset releases every buffer allocated through this Allocator since the last
// Reset, dedup'd by underlying *wgpu.Buffer so slot-shared buffers are not
// double-released (double-free → Metal autorelease corruption).
//
// Caller (the orchestrator) must have drained the GPU queue (e.g. via readback
// of all outputs) before calling Reset.
func (a *allocator) Reset() {
	released := make(map[*wgpu.Buffer]bool, len(a.dedicated)+len(a.slotBufs))
	for _, db := range a.dedicated {
		if db.buf != nil && !released[db.buf] {
			released[db.buf] = true
			db.buf.Release()
			db.buf = nil
		}
	}
	for _, db := range a.slotBufs {
		if db.buf != nil && !released[db.buf] {
			released[db.buf] = true
			db.buf.Release()
			db.buf = nil
		}
	}
	a.dedicated = nil
	a.slotBufs = make(map[int]*deviceBuffer)
}

// EncodeFloat32Input converts a float32 host slice to the right byte
// representation for the destination dtype's GPU layout: f16 narrows to
// IEEE 754 half, bf16 narrows (round-to-nearest-even) and lands in the upper
// 16 bits of each u32 slot (codegen/wgsl.go _bf16_rtne_bits expects this
// layout), fp8 quantizes to the e4m3fn / e5m2 grid and stores the quantized
// value's full f32 bit pattern per u32 slot (decoded storage; the
// _fp8*_rtne_bits store helpers produce the same layout), image dtypes pad
// the tail to a full vec4 slot (ceil(n/4)*4 f32 lanes; tail bytes are
// zero-filled), and everything else passes through as little-endian f32 bits.
//
// Used by the orchestrator when uploading leaf input data through the
// inputs map[uint32][]float32 contract of Executor/SymbolicExecutor.
func EncodeFloat32Input(data []float32, dt *uop.DType) []byte {
	switch {
	case dt != nil && dt.Scalar() == uop.Dtypes.Float16:
		return float32sToF16Bytes(data)
	case dt != nil && dt.Scalar() == uop.Dtypes.BFloat16:
		return float32sToBF16U32Bytes(data)
	case dt != nil && (dt.Scalar() == uop.Dtypes.FP8E4M3 || dt.Scalar() == uop.Dtypes.FP8E5M2):
		return float32sToFP8F32Bytes(data, dt)
	case dt != nil && dt.IsImage():
		return float32sToImageBytes(data)
	default:
		return float32sToBytes(data)
	}
}

// float32sToImageBytes encodes float32 values for an image-dtype storage
// buffer. The buffer holds ceil(n/4) vec4 slots (16 bytes each); the leading
// n*4 bytes are the f32 input values in order and the tail of the final
// vec4 (when n%4 != 0) is zero-padded. GPU loads at index i ≥ n are never
// emitted by codegen because the dispatch bound stays at n; the tail bytes
// only matter for write-path masking and Read truncation.
func float32sToImageBytes(data []float32) []byte {
	n := len(data)
	slots := (n + 3) / 4
	b := make([]byte, slots*16)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// float16ToFloat32 converts an IEEE 754 half-precision bit pattern to float32.
func float16ToFloat32(h uint16) float32 {
	return uop.Float16ToFloat32(h)
}

// float32ToFloat16 converts a float32 to its nearest IEEE 754 half-precision value.
func float32ToFloat16(f float32) uint16 {
	return uop.Float32ToFloat16(f)
}

// float32sToF16Bytes converts float32 values to packed f16 little-endian bytes.
func float32sToF16Bytes(data []float32) []byte {
	b := make([]byte, len(data)*2)
	for i, v := range data {
		binary.LittleEndian.PutUint16(b[i*2:], float32ToFloat16(v))
	}
	return b
}

// float32sToBF16U32Bytes encodes float32 values as bf16 packed in u32 slots.
// Narrowing uses round-to-nearest-even via uop.Float32ToBFloat16, matching the
// GPU store path (codegen/wgsl.go _bf16_rtne_bits) and the CPU oracle
// (uop.DType.Quantize). The bf16 bits land in the upper 16 of each u32 storage
// word so the GPU's bitcast<f32>(u32) load reconstructs the bf16-quantized f32
// directly.
func float32sToBF16U32Bytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		bf16u32 := uint32(uop.Float32ToBFloat16(v)) << 16
		binary.LittleEndian.PutUint32(b[i*4:], bf16u32)
	}
	return b
}

// float32sToFP8F32Bytes encodes float32 values as fp8-quantized f32 bit
// patterns, one per u32 slot (decoded storage). Quantization is RTNE via
// uop.DType.Quantize (e4m3fn saturating / e5m2 round-to-Inf), matching the
// GPU store path (codegen/wgsl.go _fp8e4m3_rtne_bits / _fp8e5m2_rtne_bits)
// bit for bit, so GPU-vs-host storage comparisons are exact. The stored word
// is a valid f32 encoding, which is what makes the GPU's bitcast<f32> load
// and the orchestrator's default f32 readback decode both work unchanged.
func float32sToFP8F32Bytes(data []float32, dt *uop.DType) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(dt.Quantize(v)))
	}
	return b
}

// float32sToBytes converts a float32 slice to its little-endian byte representation.
func float32sToBytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// bytesToFloat32s reinterprets a byte slice as float32 values (little-endian).
// Kept here (alongside the host-side encoders) for symmetry with the input
// path; currently unused at package level but the test file imports it.
func bytesToFloat32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// Compile-time assertion that *allocator satisfies backend.Allocator.
var _ backend.Allocator = (*allocator)(nil)
