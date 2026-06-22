package cpu

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// Device is the pure-Go CPU backend. It implements backend.Executor by
// interpreting each kernel's SINK-rooted UOp AST against host-side buffers.
//
// There is no GPU-owner goroutine, no native lib, no codegen step. All work
// happens on the caller's goroutine. Suitable as a value oracle for the
// WebGPU backend and as a zero-dependency fallback for environments without
// a working GPU stack.
type Device struct{}

// Open returns a ready-to-use CPU device. Always succeeds.
func Open() (*Device, error) {
	return &Device{}, nil
}

// Close is a no-op for the CPU backend; included to satisfy backend.Executor.
func (d *Device) Close() {}

// Run executes a static (non-symbolic) schedule on the CPU.
//
// inputs maps Buffer.UOpIdx → flat float32 data for leaf input buffers.
// Returns output data keyed by Buffer.UOpIdx for final outputs (buffers
// written by some kernel and read by none in this schedule).
func (d *Device) Run(items []schedule.ExecItem, inputs map[uint32][]float32) (map[uint32][]float32, error) {
	if len(items) == 0 {
		return nil, nil
	}

	alloc := newAllocator()
	defer alloc.Reset()

	bufs, err := d.allocateBuffers(items, alloc)
	if err != nil {
		return nil, err
	}

	// Upload leaf inputs.
	bufDType := buildBufDTypeMap(items)
	for uopIdx, data := range inputs {
		buf, ok := bufs[uopIdx]
		if !ok {
			continue
		}
		dt := bufDType[uopIdx]
		if dt == nil {
			dt = uop.Dtypes.Float32
		}
		switch {
		case dt.Scalar() == uop.Dtypes.Int32 || dt.Scalar() == uop.Dtypes.UInt32:
			// Int leaves arrive bits-in-f32 (the tensor layer's SetData
			// contract for Int32 data, mirroring the GPU's byte-level
			// passthrough in EncodeFloat32Input); reinterpret per element.
			dst := buf.asI32()
			if dst == nil {
				return nil, fmt.Errorf("cpu: leaf %d has no i32 storage", uopIdx)
			}
			n := int64(len(data))
			if n > int64(len(dst)) {
				n = int64(len(dst))
			}
			for i := int64(0); i < n; i++ {
				dst[i] = int32(math.Float32bits(data[i]))
			}
		case dt.Scalar() == uop.Dtypes.Float32 || dt.IsImage():
			// Image-storage dtypes share their scalar peer's compute identity;
			// CPU holds them as a flat f32 slice (vec4 packing is a GPU concept).
			dst := buf.asF32()
			if dst == nil {
				return nil, fmt.Errorf("cpu: leaf %d has no f32 storage", uopIdx)
			}
			n := int64(len(data))
			if n > int64(len(dst)) {
				n = int64(len(dst))
			}
			copy(dst[:n], data[:n])
		case dt.IsFloat():
			// Narrow float dtypes (f16/bf16/fp8): quantize on upload so the
			// buffer holds on-grid values, matching the GPU's
			// EncodeFloat32Input narrowing. Idempotent when the tensor layer
			// already quantized at SetData.
			dst := buf.asF32()
			if dst == nil {
				return nil, fmt.Errorf("cpu: leaf %d has no f32 storage", uopIdx)
			}
			n := int64(len(data))
			if n > int64(len(dst)) {
				n = int64(len(dst))
			}
			for i := int64(0); i < n; i++ {
				dst[i] = dt.Quantize(data[i])
			}
		default:
			return nil, fmt.Errorf("cpu: leaf %d has unsupported input dtype %s", uopIdx, dt)
		}
	}

	// Execute kernels in schedule order.
	for i, item := range items {
		if len(item.SymVars) > 0 {
			return nil, fmt.Errorf("cpu: kernel %d is symbolic (SymVars=%v); the CPU backend slice 1 only handles static kernels", i, item.SymVars)
		}
		if err := interpret(item, bufs); err != nil {
			return nil, fmt.Errorf("cpu: kernel %d: %w", i, err)
		}
	}

	// Collect final outputs.
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, b := range item.Bufs[1:] {
			readByAny[b.UOpIdx] = true
		}
	}
	outputs := make(map[uint32][]float32)
	for _, item := range items {
		out := item.Bufs[0]
		if readByAny[out.UOpIdx] {
			continue
		}
		buf := bufs[out.UOpIdx]
		if buf == nil {
			continue
		}
		if cp := outputAsF32(buf, out.Size); cp != nil {
			outputs[out.UOpIdx] = cp
		}
	}
	return outputs, nil
}

// outputAsF32 converts a kernel-output buffer to the orchestrator's
// []float32 contract: f32 storage copies verbatim; i32 storage
// bit-reinterprets per element - the GPU readback path
// (webgpu.DecodeBytesToFloat32) returns i32 outputs as bit patterns inside
// f32, matching the tensor layer's bits-in-f32 contract for Int32 data, and
// the CPU must mirror it exactly. Returns nil for a storage-less buffer.
func outputAsF32(buf *Buffer, size int64) []float32 {
	if f := buf.asF32(); f != nil {
		cp := make([]float32, size)
		copy(cp, f[:size])
		return cp
	}
	if src := buf.asI32(); src != nil {
		cp := make([]float32, size)
		for i := int64(0); i < size; i++ {
			cp[i] = math.Float32frombits(uint32(src[i]))
		}
		return cp
	}
	return nil
}

// allocateBuffers mirrors the WebGPU three-phase allocation pattern:
// slot-shared intermediates, dedicated outputs, then leaf inputs.
func (d *Device) allocateBuffers(items []schedule.ExecItem, alloc *allocator) (map[uint32]*Buffer, error) {
	writtenBy := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenBy[item.Bufs[0].UOpIdx] = i
	}

	out := make(map[uint32]*Buffer, len(items)*2)

	// Phase A: slot-shared intermediates.
	slotMaxElems := make(map[int]int64)
	slotDType := make(map[int]*uop.DType)
	for _, item := range items {
		o := item.Bufs[0]
		if o.Slot >= 0 {
			if o.Size > slotMaxElems[o.Slot] {
				slotMaxElems[o.Slot] = o.Size
			}
			if slotDType[o.Slot] == nil {
				slotDType[o.Slot] = o.DType
			}
		}
	}
	for slot, maxElems := range slotMaxElems {
		db, err := alloc.AllocSlot(slot, maxElems, slotDType[slot], "")
		if err != nil {
			return nil, err
		}
		_ = db
	}
	for _, item := range items {
		o := item.Bufs[0]
		if o.Slot >= 0 {
			db, err := alloc.AllocSlot(o.Slot, 0, nil, "")
			if err != nil {
				return nil, err
			}
			out[o.UOpIdx] = db.(*Buffer)
		}
	}

	// Phase B: dedicated final outputs.
	for _, item := range items {
		o := item.Bufs[0]
		if o.Slot < 0 {
			if _, ok := out[o.UOpIdx]; !ok {
				db, err := alloc.Alloc(o.Size, o.DType, backend.BufferUsageIO, "")
				if err != nil {
					return nil, err
				}
				out[o.UOpIdx] = db.(*Buffer)
			}
		}
	}

	// Phase C: leaf inputs.
	for _, item := range items {
		for _, b := range item.Bufs[1:] {
			if _, written := writtenBy[b.UOpIdx]; written {
				continue
			}
			if _, ok := out[b.UOpIdx]; ok {
				continue
			}
			db, err := alloc.Alloc(b.Size, b.DType, backend.BufferUsageLeafInput, "")
			if err != nil {
				return nil, err
			}
			out[b.UOpIdx] = db.(*Buffer)
		}
	}
	return out, nil
}

// buildBufDTypeMap collects a UOpIdx → dtype lookup over every buffer in
// the schedule. Used by Run to know how to decode host inputs.
func buildBufDTypeMap(items []schedule.ExecItem) map[uint32]*uop.DType {
	m := make(map[uint32]*uop.DType, len(items)*4)
	for _, item := range items {
		for _, b := range item.Bufs {
			m[b.UOpIdx] = b.DType
		}
	}
	return m
}
