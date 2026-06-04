package cpu

import (
	"fmt"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/uop"
)

// allocator implements backend.Allocator for the CPU backend. Slot-shared
// intermediates and dedicated alloc requests both end up as fresh Go slices
// owned by *Buffer values.
//
// The two-phase contract from backend/allocator.go is respected: the FIRST
// AllocSlot call for a slot fixes its size; subsequent calls with the same
// slot return the existing Buffer and ignore the elems / dtype arguments
// (matching the WebGPU reference orchestrator's pattern).
type allocator struct {
	slots map[int]*Buffer
	all   []*Buffer
}

func newAllocator() *allocator {
	return &allocator{slots: make(map[int]*Buffer)}
}

// Alloc creates a dedicated, fresh buffer sized to hold elems values of dt.
func (a *allocator) Alloc(elems int64, dt *uop.DType, _ backend.BufferUsage, _ string) (backend.DeviceBuffer, error) {
	buf, err := newBuffer(elems, dt)
	if err != nil {
		return nil, err
	}
	a.all = append(a.all, buf)
	return buf, nil
}

// AllocSlot is the slot-aware variant for intermediates. The first call for
// a slot fixes the physical size; subsequent calls return that same buffer.
func (a *allocator) AllocSlot(slot int, elems int64, dt *uop.DType, _ string) (backend.DeviceBuffer, error) {
	if buf, ok := a.slots[slot]; ok {
		return buf, nil
	}
	buf, err := newBuffer(elems, dt)
	if err != nil {
		return nil, err
	}
	a.slots[slot] = buf
	a.all = append(a.all, buf)
	return buf, nil
}

// Reset releases every per-Run allocation. Idempotent.
func (a *allocator) Reset() {
	for _, b := range a.all {
		b.Release()
	}
	a.all = nil
	a.slots = make(map[int]*Buffer)
}

// newBuffer creates a host-resident Buffer of elems × dt. Defaults to f32
// when dt is nil (a few schedule paths leave dt unset on AllocSlot first
// calls; the orchestrator then overrides via the writer kernel's dtype on
// the buffer-level dtype map — matching the WebGPU allocator's behavior).
func newBuffer(elems int64, dt *uop.DType) (*Buffer, error) {
	if elems < 0 {
		return nil, fmt.Errorf("cpu.allocator: negative elems %d", elems)
	}
	if elems == 0 {
		// Match WebGPU's "actualElems==0 → 1" floor used for symbolic leafs.
		elems = 1
	}
	b := &Buffer{dt: dt, elems: elems}
	switch {
	case dt == nil || dt.Scalar() == uop.Dtypes.Float32 || dt.IsImage():
		// Image-storage dtypes behave as their scalar peer on the CPU: the
		// vec4 packing is a GPU buffer concept, so host storage is a plain
		// flat f32 slice. Tensor.Realize round-trips through Write/Read
		// without the vec4 stride. (Currently only ImageFloat32 exists; its
		// scalar peer is Float32.)
		b.f32 = make([]float32, elems)
	case dt.Scalar() == uop.Dtypes.Int32, dt.Scalar() == uop.Dtypes.UInt32:
		b.i32 = make([]int32, elems)
	default:
		return nil, fmt.Errorf("cpu.allocator: unsupported dtype %s for CPU backend (slice 1 supports f32 and i32 only)", dt)
	}
	return b, nil
}
