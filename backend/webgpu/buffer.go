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

// deviceBuffer wraps a *wgpu.Buffer with its element count and dtype, and owns
// the MapAsync + Poll(PollWait) readback path. It implements backend.DeviceBuffer.
//
// Threading: every method touches *wgpu.* and must be called from the device's
// GPU-owner goroutine (open.go gpuOwnerLoop). The orchestrator funnels all
// public calls through d.onGPU; collaborator methods like these assume they
// already run there.
type deviceBuffer struct {
	dev   *Device // for queue access and Poll on Read
	buf   *wgpu.Buffer
	elems int64
	dt    *uop.DType
}

// Size returns the element count this buffer was sized to hold.
func (b *deviceBuffer) Size() int64 { return b.elems }

// DType returns the element dtype of the buffer. May be nil for legacy f32
// buffers allocated through the symbolic-dispatch fast path.
func (b *deviceBuffer) DType() *uop.DType { return b.dt }

// Raw returns the underlying *wgpu.Buffer. Used by collaborators (Compiler,
// Program) that need to attach the buffer to a BindGroup. Not exposed via the
// backend.DeviceBuffer interface - backends are not required to expose a raw
// handle.
func (b *deviceBuffer) Raw() *wgpu.Buffer { return b.buf }

// Write uploads raw bytes to the GPU buffer. Caller is responsible for encoding
// the host data into the right per-element layout (see allocator.go's
// EncodeFloat32Input which handles f16 / bf16 narrowing for inputs sourced as
// float32 from tensor/realize.go).
func (b *deviceBuffer) Write(data []byte) error {
	if err := b.dev.queue.WriteBuffer(b.buf, 0, data); err != nil {
		return fmt.Errorf("webgpu: buffer write: %w", err)
	}
	return nil
}

// Read returns the buffer contents as raw bytes. Internally it copies through
// a staging buffer and resolves the mapping via MapAsync + Poll(PollWait) -
// NOT wgpu.Buffer.Map, which spawns an unpinned goroutine whose
// NSAutoreleasePool drain would race a different OS thread. See the readBuffer
// commentary in open.go and the original executor.go for the full rationale.
func (b *deviceBuffer) Read() ([]byte, error) {
	byteSize := bufferByteSize(b.elems, b.dt)

	staging, err := b.dev.device.CreateBuffer(&wgpu.BufferDescriptor{
		Usage: gputypes.BufferUsageCopyDst | gputypes.BufferUsageMapRead,
		Size:  byteSize,
	})
	if err != nil {
		return nil, fmt.Errorf("alloc staging: %w", err)
	}
	defer func() {
		staging.Unmap() //nolint:errcheck
		staging.Release()
	}()

	enc, err := b.dev.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("CreateCommandEncoder: %w", err)
	}
	enc.CopyBufferToBuffer(b.buf, 0, staging, 0, byteSize)
	cmd, err := enc.Finish()
	if err != nil {
		return nil, fmt.Errorf("CommandEncoder.Finish: %w", err)
	}
	if _, err := b.dev.queue.Submit(cmd); err != nil {
		return nil, fmt.Errorf("Queue.Submit: %w", err)
	}

	// MapAsync registers the pending map without spawning anything; a single
	// Poll(PollWait) issues a full GPU barrier (WaitIdle) and resolves all
	// pending maps. Because we run on the GPU-owner goroutine (locked to one
	// OS thread), the WaitIdle's NSAutoreleasePool is created and drained on
	// the same thread - no migration, no crash.
	pending, err := staging.MapAsync(wgpu.MapModeRead, 0, byteSize)
	if err != nil {
		return nil, fmt.Errorf("MapAsync: %w", err)
	}
	b.dev.device.Poll(wgpu.PollWait)
	ready, werr := pending.Status()
	for i := 0; i < 8 && !ready && werr == nil; i++ {
		b.dev.device.Poll(wgpu.PollWait)
		ready, werr = pending.Status()
	}
	if werr != nil {
		return nil, fmt.Errorf("map: %w", werr)
	}
	if !ready {
		return nil, fmt.Errorf("map: pending map did not resolve after PollWait")
	}
	rng, err := staging.MappedRange(0, byteSize)
	if err != nil {
		return nil, fmt.Errorf("MappedRange: %w", err)
	}
	raw := rng.Bytes()
	out := make([]byte, len(raw))
	copy(out, raw) // detach from the staging mapping before Unmap
	rng.Release()
	return out, nil
}

// Release frees the underlying GPU buffer. Must be called after the GPU has
// drained any pending work that references this buffer; the orchestrator
// guarantees this by syncing through readback before releasing.
func (b *deviceBuffer) Release() {
	if b.buf != nil {
		b.buf.Release()
		b.buf = nil
	}
}

// DecodeBytesToFloat32 converts raw bytes (as returned by DeviceBuffer.Read)
// to []float32 for the given dtype. For f16 it does the 2-byte → f32 expansion;
// for f32 / int32 / uint32 / bf16-packed-in-u32 it bit-reinterprets.
//
// Lives in this file because it is the inverse of buffer.Read and is only used
// by the orchestrator's output-decoding step.
func DecodeBytesToFloat32(raw []byte, nElems int64, dtype *uop.DType) []float32 {
	result := make([]float32, nElems)
	if dtype != nil && dtype.Scalar() == uop.Dtypes.Float16 {
		for i := int64(0); i < nElems; i++ {
			result[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return result
	}
	for i := int64(0); i < nElems; i++ {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return result
}

// Compile-time assertion that *deviceBuffer satisfies backend.DeviceBuffer.
var _ backend.DeviceBuffer = (*deviceBuffer)(nil)
