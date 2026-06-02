package backend

import "github.com/georgebuilds/anneal/uop"

// BufferUsage tags how the runtime intends to use a freshly-allocated buffer.
// Backends translate these to native usage flags (WebGPU's
// BufferUsage{Storage, CopySrc, CopyDst}, CUDA's allocation kind, etc.).
type BufferUsage uint8

const (
	// BufferUsageIO is the default: storage buffer used by kernels, with both
	// CPU→GPU (Write) and GPU→CPU (Read) transfer enabled. Used for
	// intermediate, output, and any buffer that may be read back to the host.
	BufferUsageIO BufferUsage = iota
	// BufferUsageLeafInput is for leaf input buffers that are only written
	// from the host and read by kernels — never read back. WebGPU allows the
	// implementation to drop the CopySrc bit for these.
	BufferUsageLeafInput
)

// Allocator allocates GPU buffers.
//
// Alloc creates a dedicated, fresh buffer sized to hold elems values of dtype.
// Used for leaf inputs and final outputs.
//
// AllocSlot is the slot-aware variant for intermediate buffers: multiple kernels
// inside one schedule may share the same physical buffer (the memory planner
// assigns positive Slot IDs). The Allocator owns the slot→Buffer table for the
// lifetime of a Run; callers identify which slot a kernel writes via slot, and
// the Allocator returns the (allocated-once, reused) DeviceBuffer for that slot.
//
// Two-phase contract: the FIRST AllocSlot call for a given slot fixes that
// slot's physical size; subsequent calls for the same slot return the existing
// buffer and ignore elems. The caller is responsible for computing the maximum
// elems across all consumers of the slot before calling AllocSlot, so that the
// first call gets the right size. The orchestrator in backend/webgpu/executor.go
// is the reference for this two-phase pattern.
//
// Reset frees all per-Run allocations created since the last Reset. Backends
// should batch-release buffers in safe order (after GPU sync) and deduplicate
// slot-shared buffers so the same native buffer is not released twice.
//
// Threading: every method touches native GPU state and must run on the backend
// GPU-owner goroutine (see DeviceBuffer doc).
type Allocator interface {
	Alloc(elems int64, dt *uop.DType, usage BufferUsage, label string) (DeviceBuffer, error)
	AllocSlot(slot int, elems int64, dt *uop.DType, label string) (DeviceBuffer, error)
	Reset()
}
