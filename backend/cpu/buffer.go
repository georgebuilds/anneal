package cpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/uop"
)

// Buffer is the host-resident DeviceBuffer used by the CPU backend.
//
// Element storage is kept in the most convenient typed slice (f32 / i32);
// Write/Read at the backend boundary still speak raw little-endian bytes
// for symmetry with the WebGPU backend's transfer surface and so that the
// orchestrator's per-dtype encode/decode helpers stay reusable.
type Buffer struct {
	dt    *uop.DType
	elems int64

	// Exactly one of these is non-nil based on dt.
	f32 []float32
	i32 []int32
}

// Size returns the number of elements (not bytes) in the buffer.
func (b *Buffer) Size() int64 { return b.elems }

// DType returns the buffer's element dtype.
func (b *Buffer) DType() *uop.DType { return b.dt }

// Write copies raw little-endian bytes into the buffer. The layout matches
// the WebGPU encode side (f32: 4 bytes per elem; i32: 4 bytes per elem).
func (b *Buffer) Write(data []byte) error {
	switch {
	case b.f32 != nil:
		if int64(len(data)) < b.elems*4 {
			return fmt.Errorf("cpu.Buffer.Write: short write: have %d bytes, need %d", len(data), b.elems*4)
		}
		for i := int64(0); i < b.elems; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4:])
			b.f32[i] = math.Float32frombits(bits)
		}
	case b.i32 != nil:
		if int64(len(data)) < b.elems*4 {
			return fmt.Errorf("cpu.Buffer.Write: short write: have %d bytes, need %d", len(data), b.elems*4)
		}
		for i := int64(0); i < b.elems; i++ {
			b.i32[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
		}
	default:
		return fmt.Errorf("cpu.Buffer.Write: buffer has no storage")
	}
	return nil
}

// Read returns the buffer's contents as raw little-endian bytes.
func (b *Buffer) Read() ([]byte, error) {
	switch {
	case b.f32 != nil:
		out := make([]byte, b.elems*4)
		for i := int64(0); i < b.elems; i++ {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(b.f32[i]))
		}
		return out, nil
	case b.i32 != nil:
		out := make([]byte, b.elems*4)
		for i := int64(0); i < b.elems; i++ {
			binary.LittleEndian.PutUint32(out[i*4:], uint32(b.i32[i]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("cpu.Buffer.Read: buffer has no storage")
	}
}

// Release drops the host storage. After Release the Buffer is unusable.
func (b *Buffer) Release() {
	b.f32 = nil
	b.i32 = nil
	b.elems = 0
}

// asF32 returns the typed f32 slice; the interpreter holds the only callers
// and they all gate on dtype before calling. Returns nil if the buffer is
// not float-typed.
func (b *Buffer) asF32() []float32 { return b.f32 }

// asI32 returns the typed i32 slice; nil for non-int buffers.
func (b *Buffer) asI32() []int32 { return b.i32 }
