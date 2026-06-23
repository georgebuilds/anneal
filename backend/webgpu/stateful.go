package webgpu

import (
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/georgebuilds/anneal/uop"
)

// Stateful realize buffer cache.
//
// Realize is normally stateless: every call schedules the graph from leaves and
// runs every kernel, so realizing N outputs of a shared graph (e.g. a loss and
// its gradients) in separate calls re-executes the shared forward+backward N
// times. With stateful realize the executor caches each intermediate (slotted)
// buffer by its node UOpIdx, scoped by (scopeID, scopeGen); a later Run in the
// same scope reuses the cached buffers and SKIPS their producer kernels, so the
// shared graph executes once.
//
// Safety: in stateful mode intermediates are allocated persistently, one buffer
// per node with NO slot reuse, which removes the slot-aliasing hazard entirely.
// The cache is freed whenever the scope or generation changes (a fresh arena =
// new scopeID; any leaf SetData/Load = new generation), so a stale buffer can
// never be served and memory is bounded to one scope's intermediates.

// BeginRealizeScope arms the cache for the NEXT Run, scoped by (scopeID,
// scopeGen). The realize layer calls this just before Run when stateful realize
// is enabled. The real free-on-scope-change runs inside the Run on the GPU-owner
// goroutine (beginScopeLocked); here we only record intent. The jobs-channel send
// that Run performs is the happens-before barrier for these writes.
func (d *Device) BeginRealizeScope(scopeID, scopeGen uint64) {
	d.pendingScopeID = scopeID
	d.pendingScopeGen = scopeGen
	d.realizeOn = true
}

// beginScopeLocked, run at the start of a stateful Run on the GPU-owner
// goroutine, frees the cache when the scope or generation changed, then returns
// the set of node UOpIdx already cached by a prior Run in this scope (the run
// loop skips their producer kernels).
func (d *Device) beginScopeLocked() map[uint32]bool {
	if d.realizeCache == nil ||
		d.pendingScopeID != d.realizeScopeID ||
		d.pendingScopeGen != d.realizeScopeGen {
		d.freeRealizeCacheLocked()
		d.realizeCache = make(map[uint32]*deviceBuffer)
		d.realizeScopeID = d.pendingScopeID
		d.realizeScopeGen = d.pendingScopeGen
	}
	prebuilt := make(map[uint32]bool, len(d.realizeCache))
	for k := range d.realizeCache {
		prebuilt[k] = true
	}
	return prebuilt
}

// freeRealizeCacheLocked releases every cached buffer. GPU-owner goroutine only.
func (d *Device) freeRealizeCacheLocked() {
	for _, db := range d.realizeCache {
		if db != nil && db.buf != nil {
			db.buf.Release()
			db.buf = nil
		}
	}
	d.realizeCache = nil
}

// allocCachedIntermediateLocked creates a persistent storage buffer for a cached
// intermediate, using the same 4-byte-per-element stride the codegen assumes for
// slotted intermediates (the bf16 / int32 bitcast<f32> load path relies on it).
// The buffer is owned by the realize cache (not the per-Run allocator), so it
// survives alloc.Reset() and is freed on the next scope change.
func (d *Device) allocCachedIntermediateLocked(elems int64, dt *uop.DType, label string) (*deviceBuffer, error) {
	const slotElemBytes uint64 = 4
	buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  uint64(elems) * slotElemBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: alloc cached intermediate %q: %w", label, err)
	}
	return &deviceBuffer{dev: d, buf: buf, elems: elems, dt: dt}, nil
}
