package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

const workgroupSize = 64 // must match codegen/wgsl.go

// Run executes a compiled schedule on this device.
// inputs maps Buffer.UOpIdx → flat float32 data for leaf (external-input) buffers.
// Returns output data keyed by Buffer.UOpIdx for final output buffers (not read by
// any subsequent kernel in the schedule).
//
// All Metal work funnels onto the GPU-owner goroutine (see open.go threading model).
func (d *Device) Run(items []schedule.ExecItem, inputs map[uint32][]float32) (map[uint32][]float32, error) {
	if len(items) == 0 {
		return nil, nil
	}
	var outputs map[uint32][]float32
	err := d.onGPU(func() error {
		var rerr error
		outputs, rerr = d.runLocked(items, inputs)
		return rerr
	})
	return outputs, err
}

// runLocked is Run's body; it assumes it is already executing on the GPU-owner
// goroutine and must not call onGPU.
func (d *Device) runLocked(items []schedule.ExecItem, inputs map[uint32][]float32) (map[uint32][]float32, error) {
	if !d.HasShaderF16 {
		for _, item := range items {
			for _, buf := range item.Bufs {
				if buf.DType != nil && buf.DType.Scalar() == uop.Dtypes.Float16 {
					return nil, fmt.Errorf("webgpu: kernel requires shader-f16 but adapter does not support it — enable the extension at device open time or use f32")
				}
			}
		}
	}

	// ── Render all WGSL shaders before touching the GPU ───────────────────
	wgsls := make([]string, len(items))
	for i, item := range items {
		if item.WGSL != "" {
			wgsls[i] = item.WGSL // pre-rendered by cache; Ast may be zeroed
		} else {
			wgsls[i] = codegen.RenderWGSL(item)
		}
	}

	// ── Classify buffers ──────────────────────────────────────────────────
	// writtenBy: output buffer UOpIdx → kernel index that writes it
	writtenBy := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenBy[item.Bufs[0].UOpIdx] = i
	}
	// readByAny: buffer UOpIdx → true if any kernel reads it as input
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}

	// ── Allocate GPU buffers ──────────────────────────────────────────────
	gpuBufs := make(map[uint32]*wgpu.Buffer, len(items)*2)

	// Phase A: slot-shared intermediate buffers (Slot >= 0 on Bufs[0]).
	// Find the maximum element count for each slot so one allocation covers all.
	slotMaxElems := make(map[int]int64)
	for _, item := range items {
		out := item.Bufs[0]
		if out.Slot >= 0 {
			if out.Size > slotMaxElems[out.Slot] {
				slotMaxElems[out.Slot] = out.Size
			}
		}
	}
	slotGPUBuf := make(map[int]*wgpu.Buffer, len(slotMaxElems))
	for slot, maxElems := range slotMaxElems {
		buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("slot%d", slot),
			Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
			Size:  uint64(maxElems) * 4,
		})
		if err != nil {
			return nil, fmt.Errorf("webgpu: alloc slot %d: %w", slot, err)
		}
		slotGPUBuf[slot] = buf
	}
	// Map output buffer UOpIdx → GPU buffer via slot.
	for _, item := range items {
		out := item.Bufs[0]
		if out.Slot >= 0 {
			gpuBufs[out.UOpIdx] = slotGPUBuf[out.Slot]
		}
	}

	// Phase B: dedicated output buffers (Slot == -1, i.e. final outputs).
	for _, item := range items {
		out := item.Bufs[0]
		if out.Slot < 0 {
			if _, ok := gpuBufs[out.UOpIdx]; !ok {
				buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
					Label: fmt.Sprintf("out%d", out.UOpIdx),
					Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
					Size:  uint64(out.Size) * elemBytes(out.DType),
				})
				if err != nil {
					return nil, fmt.Errorf("webgpu: alloc output buf %d: %w", out.UOpIdx, err)
				}
				gpuBufs[out.UOpIdx] = buf
			}
		}
	}

	// Phase C: leaf input buffers (appear in Bufs[1..] but never in any Bufs[0]).
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			if _, written := writtenBy[buf.UOpIdx]; written {
				continue // produced by an upstream kernel, already allocated
			}
			if _, ok := gpuBufs[buf.UOpIdx]; ok {
				continue // already allocated
			}
			gpuBuf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: fmt.Sprintf("leaf%d", buf.UOpIdx),
				Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopyDst,
				Size:  uint64(buf.Size) * elemBytes(buf.DType),
			})
			if err != nil {
				return nil, fmt.Errorf("webgpu: alloc leaf buf %d: %w", buf.UOpIdx, err)
			}
			gpuBufs[buf.UOpIdx] = gpuBuf
		}
	}

	// ── Upload leaf input data ────────────────────────────────────────────
	// Build a lookup from UOpIdx → dtype so we can encode f16 inputs correctly.
	bufDType := make(map[uint32]*uop.DType, len(items)*4)
	for _, item := range items {
		for _, buf := range item.Bufs {
			bufDType[buf.UOpIdx] = buf.DType
		}
	}
	for uopIdx, data := range inputs {
		gpuBuf, ok := gpuBufs[uopIdx]
		if !ok {
			continue // caller provided data for a buffer not in this schedule — skip
		}
		var raw []byte
		dt := bufDType[uopIdx]
		switch {
		case dt != nil && dt.Scalar() == uop.Dtypes.Float16:
			raw = float32sToF16Bytes(data)
		case dt != nil && dt.Scalar() == uop.Dtypes.BFloat16:
			raw = float32sToBF16U32Bytes(data)
		default:
			raw = float32sToBytes(data)
		}
		if err := d.queue.WriteBuffer(gpuBuf, 0, raw); err != nil {
			return nil, fmt.Errorf("webgpu: upload buf %d: %w", uopIdx, err)
		}
	}

	// ── Execute kernels in schedule order ─────────────────────────────────
	// kernelRes holds per-kernel GPU objects that must outlive dispatch.
	// Released after GPU sync (after readbacks) to cancel GC finalizers and
	// prevent finalizer-goroutine Metal calls from racing with future dispatches.
	type kernelRes struct {
		shader         *wgpu.ShaderModule
		bgLayout       *wgpu.BindGroupLayout
		pipelineLayout *wgpu.PipelineLayout
		pipeline       *wgpu.ComputePipeline
		bg             *wgpu.BindGroup
	}
	kernelRess := make([]kernelRes, 0, len(items))

	for i, item := range items {
		res, err := d.runKernel(item, wgsls[i], gpuBufs)
		if err != nil {
			return nil, fmt.Errorf("webgpu: kernel %d: %w", i, err)
		}
		kernelRess = append(kernelRess, kernelRes(res))
	}

	// ── Read back final outputs ───────────────────────────────────────────
	// readBuffer blocks until the GPU has completed all prior work, so after
	// all readbacks the GPU is fully idle and we can safely release resources.
	outputs := make(map[uint32][]float32)
	for _, item := range items {
		out := item.Bufs[0]
		if readByAny[out.UOpIdx] {
			continue // consumed by a downstream kernel — not a final output
		}
		data, err := d.readBuffer(gpuBufs[out.UOpIdx], out.Size, out.DType)
		if err != nil {
			return nil, fmt.Errorf("webgpu: readback buf %d: %w", out.UOpIdx, err)
		}
		outputs[out.UOpIdx] = data
	}

	// GPU is now idle. Release all Metal resources synchronously so that GC
	// finalizers for these objects become no-ops, preventing finalizer-goroutine
	// Metal API calls from racing with Metal calls in subsequent Realize rounds.
	for i := len(kernelRess) - 1; i >= 0; i-- {
		r := kernelRess[i]
		r.bg.Release()
		r.pipeline.Release()
		r.pipelineLayout.Release()
		r.bgLayout.Release()
		r.shader.Release()
	}
	// Slot-shared intermediate buffers appear under multiple UOpIdx keys in gpuBufs.
	// Deduplicate before releasing to prevent double-free → Metal autorelease corruption.
	releasedBufs := make(map[*wgpu.Buffer]bool, len(gpuBufs))
	for _, buf := range gpuBufs {
		if !releasedBufs[buf] {
			releasedBufs[buf] = true
			buf.Release()
		}
	}

	return outputs, nil
}

// runKernelRes holds GPU objects created per kernel. Callers must release them
// after GPU completion (after all readbacks) to prevent GC finalizer races.
type runKernelRes struct {
	shader         *wgpu.ShaderModule
	bgLayout       *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
	bg             *wgpu.BindGroup
}

// runKernel compiles and dispatches one kernel. It returns the GPU objects
// allocated for this kernel; the caller must release them after GPU completion.
func (d *Device) runKernel(item schedule.ExecItem, wgsl string, gpuBufs map[uint32]*wgpu.Buffer) (runKernelRes, error) {
	var nParams int
	if item.Ast.Valid() {
		nParams = item.Ast.Arg().(uop.KernelInfo).NumParams
	} else {
		nParams = len(item.Bufs) // Ast zeroed by cache; NumParams == len(Bufs) invariant
	}

	// ── Compile shader ────────────────────────────────────────────────────
	shader, err := d.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: wgsl})
	if err != nil {
		return runKernelRes{}, fmt.Errorf("CreateShaderModule: %w\n--- WGSL ---\n%s", err, wgsl)
	}

	// ── Bind group layout ─────────────────────────────────────────────────
	layoutEntries := make([]gputypes.BindGroupLayoutEntry, nParams)
	for i := range layoutEntries {
		bt := gputypes.BufferBindingTypeReadOnlyStorage
		if i == 0 {
			bt = gputypes.BufferBindingTypeStorage // read_write for output
		}
		layoutEntries[i] = gputypes.BindGroupLayoutEntry{
			Binding:    uint32(i),
			Visibility: gputypes.ShaderStageCompute,
			Buffer:     &gputypes.BufferBindingLayout{Type: bt},
		}
	}
	bgLayout, err := d.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: layoutEntries,
	})
	if err != nil {
		shader.Release()
		return runKernelRes{}, fmt.Errorf("CreateBindGroupLayout: %w", err)
	}

	// ── Pipeline layout + compute pipeline ───────────────────────────────
	pipelineLayout, err := d.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("CreatePipelineLayout: %w", err)
	}

	pipeline, err := d.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "main",
	})
	if err != nil {
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("CreateComputePipeline: %w\n--- WGSL ---\n%s", err, wgsl)
	}

	// ── Bind group ────────────────────────────────────────────────────────
	entries := make([]wgpu.BindGroupEntry, nParams)
	for i, buf := range item.Bufs {
		gpuBuf := gpuBufs[buf.UOpIdx]
		if gpuBuf == nil {
			pipeline.Release()
			pipelineLayout.Release()
			bgLayout.Release()
			shader.Release()
			return runKernelRes{}, fmt.Errorf("missing GPU buffer for UOpIdx %d (param %d)", buf.UOpIdx, i)
		}
		entries[i] = wgpu.BindGroupEntry{
			Binding: uint32(i),
			Buffer:  gpuBuf,
			Size:    uint64(buf.Size) * elemBytes(buf.DType),
		}
	}
	bg, err := d.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  bgLayout,
		Entries: entries,
	})
	if err != nil {
		pipeline.Release()
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("CreateBindGroup: %w", err)
	}

	// ── Dispatch ──────────────────────────────────────────────────────────
	outElems := item.Bufs[0].Size
	workgroups := uint32((outElems + workgroupSize - 1) / workgroupSize)
	if workgroups == 0 {
		workgroups = 1
	}

	enc, err := d.device.CreateCommandEncoder(nil)
	if err != nil {
		bg.Release()
		pipeline.Release()
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("CreateCommandEncoder: %w", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		bg.Release()
		pipeline.Release()
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("BeginComputePass: %w", err)
	}
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(workgroups, 1, 1)
	if err := pass.End(); err != nil {
		bg.Release()
		pipeline.Release()
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("ComputePass.End: %w", err)
	}
	cmd, err := enc.Finish()
	if err != nil {
		bg.Release()
		pipeline.Release()
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("CommandEncoder.Finish: %w", err)
	}
	if _, err := d.queue.Submit(cmd); err != nil {
		bg.Release()
		pipeline.Release()
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return runKernelRes{}, fmt.Errorf("Queue.Submit: %w", err)
	}
	return runKernelRes{shader: shader, bgLayout: bgLayout, pipelineLayout: pipelineLayout, pipeline: pipeline, bg: bg}, nil
}

// readBuffer maps a GPU storage buffer and returns its contents as float32.
// For non-f32 dtypes the raw bytes are reinterpreted (int32/uint32 → float32 bitcast).
func (d *Device) readBuffer(buf *wgpu.Buffer, nElems int64, dtype *uop.DType) ([]float32, error) {
	byteSize := uint64(nElems) * elemBytes(dtype)

	staging, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
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

	enc, err := d.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("CreateCommandEncoder: %w", err)
	}
	enc.CopyBufferToBuffer(buf, 0, staging, 0, byteSize)
	cmd, err := enc.Finish()
	if err != nil {
		return nil, fmt.Errorf("CommandEncoder.Finish: %w", err)
	}
	if _, err := d.queue.Submit(cmd); err != nil {
		return nil, fmt.Errorf("Queue.Submit: %w", err)
	}

	// Resolve the map ourselves rather than calling staging.Map: Buffer.Map
	// spawns an internal, UNPINNED goroutine to run Poll(PollWait)→WaitIdle,
	// whose NSAutoreleasePool then drains on whatever OS thread the scheduler
	// migrated it to → SIGSEGV. MapAsync registers the pending map without
	// spawning anything; a single Poll(PollWait) issues a full GPU barrier
	// (WaitIdle) and resolves all pending maps with the "all completed"
	// sentinel. Because runLocked/readBuffer run on the GPU-owner goroutine
	// (locked to one OS thread), that WaitIdle's pool is created and drained
	// on the same thread — no migration, no crash.
	pending, err := staging.MapAsync(wgpu.MapModeRead, 0, byteSize)
	if err != nil {
		return nil, fmt.Errorf("MapAsync: %w", err)
	}
	d.device.Poll(wgpu.PollWait)
	ready, werr := pending.Status()
	for i := 0; i < 8 && !ready && werr == nil; i++ {
		d.device.Poll(wgpu.PollWait)
		ready, werr = pending.Status()
	}
	if werr != nil {
		return nil, fmt.Errorf("Map: %w", werr)
	}
	if !ready {
		return nil, fmt.Errorf("Map: pending map did not resolve after PollWait")
	}
	rng, err := staging.MappedRange(0, byteSize)
	if err != nil {
		return nil, fmt.Errorf("MappedRange: %w", err)
	}

	raw := rng.Bytes()
	result := make([]float32, nElems)
	if dtype != nil && dtype.Scalar() == uop.Dtypes.Float16 {
		for i := int64(0); i < nElems; i++ {
			result[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	} else {
		for i := int64(0); i < nElems; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}
	rng.Release()
	return result, nil
}

// elemBytes returns the GPU buffer element size in bytes for a dtype.
// f16 uses 2 bytes; all other dtypes (bool→u32, int8/16→i32, etc.) use 4 bytes,
// matching wgsl.go's wgslBufferElemType promotion rules.
func elemBytes(d *uop.DType) uint64 {
	if d != nil && d.Scalar() == uop.Dtypes.Float16 {
		return 2
	}
	return 4
}

// float16ToFloat32 converts an IEEE 754 half-precision bit pattern to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x3FF)
	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign // ±zero
		} else {
			// Subnormal f16: normalise into f32.
			exp32 := uint32(127 - 14)
			for frac&0x400 == 0 {
				frac <<= 1
				exp32--
			}
			frac &= 0x3FF
			bits = sign | (exp32 << 23) | (frac << 13)
		}
	case 31:
		// Inf or NaN.
		bits = sign | 0x7F800000 | (frac << 13)
	default:
		bits = sign | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}

// float32ToFloat16 converts a float32 to its nearest IEEE 754 half-precision value.
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits >> 31)
	exp := int32((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	switch {
	case exp > 15:
		// Overflow → ±Inf in f16.
		return (sign << 15) | 0x7C00
	case exp < -24:
		// Too small → ±zero.
		return sign << 15
	case exp < -14:
		// Subnormal f16.
		frac |= 0x800000
		shift := uint32(-14 - exp)
		frac >>= shift
		return (sign << 15) | uint16(frac>>13)
	default:
		return (sign << 15) | (uint16(exp+15) << 10) | uint16(frac>>13)
	}
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
// Each float32 is truncated to bf16 by zeroing the low 16 mantissa bits; the
// result is stored in a 4-byte u32 (little-endian). The GPU reads it back via
// bitcast<f32>(u32), which recovers the bf16-approximated f32 value.
func float32sToBF16U32Bytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		bf16u32 := math.Float32bits(v) & 0xFFFF0000
		binary.LittleEndian.PutUint32(b[i*4:], bf16u32)
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
func bytesToFloat32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// silence unused import warning for unsafe (used in test file, kept here for completeness)
var _ = unsafe.Pointer(nil)

// ── Symbolic dispatch (SLICE 3b-1) ───────────────────────────────────────────

// symElemCount returns the actual element count for buf in a symbolic schedule.
// For concrete buffers (Size>0) it returns Size directly. For symbolic buffers
// (Size==0), it multiplies the concrete dims in Shape by the binding values for
// the symbolic dims (marked as 0 in Shape). nil Shape falls back to binding[symVars[0]].
func symElemCount(buf schedule.Buffer, binding map[string]int64, symVars []string) int64 {
	if buf.Size > 0 {
		return buf.Size
	}
	if len(buf.Shape) == 0 {
		// 1D symbolic (arg=nil): size equals the single symbolic variable.
		if len(symVars) > 0 {
			if n, ok := binding[symVars[0]]; ok {
				return n
			}
		}
		return 0
	}
	// Multi-dim symbolic: product over dims, using binding for symbolic (0) dims.
	n := int64(1)
	symIdx := 0
	for _, s := range buf.Shape {
		if s == 0 {
			if symIdx < len(symVars) {
				if bv, ok := binding[symVars[symIdx]]; ok {
					n *= bv
				}
			}
			symIdx++
		} else {
			n *= s
		}
	}
	return n
}

// RunSymbolic executes a schedule that may contain symbolic kernels.
// binding maps DefineVar name → concrete int64 value for this dispatch.
// Symbolic kernels are compiled once (keyed by WGSL source) and reused across
// calls with the same kernel structure but different binding values.
// Schedules may contain a mix of symbolic and fully-concrete kernels; concrete
// kernels are dispatched via the static runKernel path.
func (d *Device) RunSymbolic(items []schedule.ExecItem, inputs map[uint32][]float32, binding map[string]int64) (map[uint32][]float32, error) {
	if len(items) == 0 {
		return nil, nil
	}
	var outputs map[uint32][]float32
	err := d.onGPU(func() error {
		var rerr error
		outputs, rerr = d.runSymbolicLocked(items, inputs, binding)
		return rerr
	})
	return outputs, err
}

// runSymbolicLocked is RunSymbolic's body; it assumes it is already executing on
// the GPU-owner goroutine and must not call onGPU.
func (d *Device) runSymbolicLocked(items []schedule.ExecItem, inputs map[uint32][]float32, binding map[string]int64) (map[uint32][]float32, error) {
	if !d.HasShaderF16 {
		for _, item := range items {
			for _, buf := range item.Bufs {
				if buf.DType != nil && buf.DType.Scalar() == uop.Dtypes.Float16 {
					return nil, fmt.Errorf("webgpu: kernel requires shader-f16 but adapter does not support it — enable the extension at device open time or use f32")
				}
			}
		}
	}

	// ── Compile all kernels (cached by WGSL source) ───────────────────────
	// Symbolic kernels → CompileSymKernel (cached handle); concrete → just WGSL.
	wgsls := make([]string, len(items))
	handles := make([]*SymKernelHandle, len(items)) // nil for concrete kernels
	for i, item := range items {
		if item.WGSL != "" {
			wgsls[i] = item.WGSL // pre-rendered by cache; Ast may be zeroed
		} else {
			wgsls[i] = codegen.RenderWGSL(item)
		}
		if len(item.SymVars) > 0 {
			handle, cached := d.symCache[wgsls[i]]
			if !cached {
				var err error
				handle, err = d.compileSymKernelLocked(item)
				if err != nil {
					return nil, fmt.Errorf("RunSymbolic compile kernel %d: %w", i, err)
				}
				d.symCache[wgsls[i]] = handle
			}
			handles[i] = handle
		}
	}

	// ── Classify buffers ──────────────────────────────────────────────────
	writtenBy := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenBy[item.Bufs[0].UOpIdx] = i
	}
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}

	// ── Allocate GPU buffers ──────────────────────────────────────────────
	gpuBufs := make(map[uint32]*wgpu.Buffer, len(items)*2)

	// Phase A: slot-shared intermediate buffers.
	slotMaxElems := make(map[int]int64)
	for i, item := range items {
		out := item.Bufs[0]
		if out.Slot >= 0 {
			actualElems := symElemCount(out, binding, items[i].SymVars)
			if actualElems > slotMaxElems[out.Slot] {
				slotMaxElems[out.Slot] = actualElems
			}
		}
	}
	slotGPUBuf := make(map[int]*wgpu.Buffer, len(slotMaxElems))
	for slot, maxElems := range slotMaxElems {
		buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("symslot%d", slot),
			Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
			Size:  uint64(maxElems) * 4,
		})
		if err != nil {
			return nil, fmt.Errorf("webgpu: RunSymbolic alloc slot %d: %w", slot, err)
		}
		slotGPUBuf[slot] = buf
	}
	for i, item := range items {
		out := item.Bufs[0]
		if out.Slot >= 0 {
			gpuBufs[out.UOpIdx] = slotGPUBuf[out.Slot]
		}
		_ = i
	}

	// Phase B: dedicated final output buffers (Slot == -1, written by a kernel).
	for i, item := range items {
		out := item.Bufs[0]
		if out.Slot < 0 {
			if _, ok := gpuBufs[out.UOpIdx]; !ok {
				actualElems := symElemCount(out, binding, items[i].SymVars)
				buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
					Label: fmt.Sprintf("symout%d", out.UOpIdx),
					Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
					Size:  uint64(actualElems) * 4,
				})
				if err != nil {
					return nil, fmt.Errorf("webgpu: RunSymbolic alloc output: %w", err)
				}
				gpuBufs[out.UOpIdx] = buf
			}
		}
	}

	// Phase C: leaf input buffers (never written by any kernel).
	for i, item := range items {
		for _, buf := range item.Bufs[1:] {
			if _, written := writtenBy[buf.UOpIdx]; written {
				continue
			}
			if _, ok := gpuBufs[buf.UOpIdx]; ok {
				continue
			}
			actualElems := symElemCount(buf, binding, items[i].SymVars)
			if actualElems == 0 {
				actualElems = 1
			}
			gpuBuf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: fmt.Sprintf("symleaf%d", buf.UOpIdx),
				Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopyDst,
				Size:  uint64(actualElems) * 4,
			})
			if err != nil {
				return nil, fmt.Errorf("webgpu: RunSymbolic alloc leaf: %w", err)
			}
			gpuBufs[buf.UOpIdx] = gpuBuf
		}
	}

	// ── Upload leaf inputs ────────────────────────────────────────────────
	symBufDType := make(map[uint32]*uop.DType, len(items)*4)
	for _, item := range items {
		for _, buf := range item.Bufs {
			symBufDType[buf.UOpIdx] = buf.DType
		}
	}
	for uopIdx, data := range inputs {
		gpuBuf, ok := gpuBufs[uopIdx]
		if !ok {
			continue
		}
		var raw []byte
		sdt := symBufDType[uopIdx]
		switch {
		case sdt != nil && sdt.Scalar() == uop.Dtypes.Float16:
			raw = float32sToF16Bytes(data)
		case sdt != nil && sdt.Scalar() == uop.Dtypes.BFloat16:
			raw = float32sToBF16U32Bytes(data)
		default:
			raw = float32sToBytes(data)
		}
		if err := d.queue.WriteBuffer(gpuBuf, 0, raw); err != nil {
			return nil, fmt.Errorf("webgpu: RunSymbolic upload: %w", err)
		}
	}

	// ── Execute kernels in schedule order ─────────────────────────────────
	// Symbolic kernels use runSymKernelWithHandle; concrete kernels use runKernel.
	// Static kernel GPU resources (shader, pipeline, bg) must outlive GPU completion.
	staticRess := make([]runKernelRes, len(items))
	for i, item := range items {
		if handles[i] != nil {
			if err := d.runSymKernelWithHandle(item, handles[i], binding, gpuBufs); err != nil {
				return nil, fmt.Errorf("webgpu: RunSymbolic kernel %d: %w", i, err)
			}
		} else {
			res, err := d.runKernel(item, wgsls[i], gpuBufs)
			if err != nil {
				return nil, fmt.Errorf("webgpu: RunSymbolic kernel %d: %w", i, err)
			}
			staticRess[i] = res
		}
	}

	// ── Read back final outputs ───────────────────────────────────────────
	outputs := make(map[uint32][]float32)
	for i, item := range items {
		out := item.Bufs[0]
		if readByAny[out.UOpIdx] {
			continue
		}
		actualElems := symElemCount(out, binding, items[i].SymVars)
		data, err := d.readBuffer(gpuBufs[out.UOpIdx], actualElems, out.DType)
		if err != nil {
			return nil, fmt.Errorf("webgpu: RunSymbolic readback: %w", err)
		}
		outputs[out.UOpIdx] = data
	}

	// ── Release static kernel GPU resources ───────────────────────────────
	// GPU is now idle (readBuffer syncs). Release static kernel resources so GC
	// finalizers for these objects become no-ops and don't race future dispatches.
	for i := len(staticRess) - 1; i >= 0; i-- {
		r := staticRess[i]
		if r.shader != nil {
			r.bg.Release()
			r.pipeline.Release()
			r.pipelineLayout.Release()
			r.bgLayout.Release()
			r.shader.Release()
		}
	}

	// ── Release GPU buffers ───────────────────────────────────────────────
	released := make(map[*wgpu.Buffer]bool, len(gpuBufs)+len(slotGPUBuf))
	for _, buf := range gpuBufs {
		if !released[buf] {
			released[buf] = true
			buf.Release()
		}
	}
	for _, buf := range slotGPUBuf {
		if !released[buf] {
			released[buf] = true
			buf.Release()
		}
	}

	return outputs, nil
}

// runSymKernelWithHandle dispatches one symbolic kernel using pre-allocated GPU
// buffers from gpuBufs. It creates a fresh params_n buffer per dispatch (4 bytes,
// holds the concrete batch size n) and submits a compute pass.
func (d *Device) runSymKernelWithHandle(item schedule.ExecItem, handle *SymKernelHandle, binding map[string]int64, gpuBufs map[uint32]*wgpu.Buffer) error {
	// Resolve n (symbolic batch size) from the first symbolic var in this kernel.
	n := int64(1)
	if len(item.SymVars) > 0 {
		if bv, ok := binding[item.SymVars[0]]; ok {
			n = bv
		}
	}

	// Params uniform buffer: ParamsN struct { data: array<u32, 4> } = 16 bytes.
	// Using a uniform buffer (not storage) avoids consuming a storage slot and
	// keeps the total storage buffer count within Metal's 8-per-stage limit.
	paramsBytes := make([]byte, 16)
	binary.LittleEndian.PutUint32(paramsBytes, uint32(n))
	paramsBuf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "sym_params",
		Usage: gputypes.BufferUsageUniform | gputypes.BufferUsageCopyDst,
		Size:  16,
	})
	if err != nil {
		return fmt.Errorf("alloc params: %w", err)
	}
	defer paramsBuf.Release()
	if err := d.queue.WriteBuffer(paramsBuf, 0, paramsBytes); err != nil {
		return fmt.Errorf("upload params: %w", err)
	}

	// Bind group: [data0, data1, ..., params_n]
	entries := make([]wgpu.BindGroupEntry, len(item.Bufs)+1)
	for i, buf := range item.Bufs {
		gpuBuf := gpuBufs[buf.UOpIdx]
		if gpuBuf == nil {
			return fmt.Errorf("missing GPU buf for UOpIdx %d (param %d)", buf.UOpIdx, i)
		}
		actualElems := symElemCount(buf, binding, item.SymVars)
		if actualElems == 0 {
			actualElems = n
		}
		entries[i] = wgpu.BindGroupEntry{
			Binding: uint32(i),
			Buffer:  gpuBuf,
			Size:    uint64(actualElems) * 4,
		}
	}
	entries[len(item.Bufs)] = wgpu.BindGroupEntry{
		Binding: uint32(len(item.Bufs)),
		Buffer:  paramsBuf,
		Size:    16,
	}

	bg, err := d.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  handle.bgLayout,
		Entries: entries,
	})
	if err != nil {
		return fmt.Errorf("CreateBindGroup: %w", err)
	}
	defer bg.Release()

	// Dispatch over total output elements (n × concrete trailing dims).
	outElems := symElemCount(item.Bufs[0], binding, item.SymVars)
	if outElems == 0 {
		outElems = n
	}
	wgs := uint32((outElems + workgroupSize - 1) / workgroupSize)
	if wgs == 0 {
		wgs = 1
	}

	enc, err := d.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("CreateCommandEncoder: %w", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return fmt.Errorf("BeginComputePass: %w", err)
	}
	pass.SetPipeline(handle.pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(wgs, 1, 1)
	if err := pass.End(); err != nil {
		return fmt.Errorf("ComputePass.End: %w", err)
	}
	cmd, err := enc.Finish()
	if err != nil {
		return fmt.Errorf("CommandEncoder.Finish: %w", err)
	}
	if _, err := d.queue.Submit(cmd); err != nil {
		return fmt.Errorf("Queue.Submit: %w", err)
	}
	return nil
}

// symVarName returns the DefineVar name of the first symbolic AxisLoop range
// in item's kernel AST, or "" if the kernel has no symbolic ranges.
func symVarName(item schedule.ExecItem) string {
	if !item.Ast.Valid() {
		return "" // Ast zeroed in cached item; SymVars is the ground truth
	}
	sink := item.Ast
	if sink.Op() != uop.OpSink || sink.NSrc() == 0 {
		return ""
	}
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return ""
	}
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			if ra, ok := r.Arg().(uop.RangeArg); ok && ra.Symbolic {
				return ra.VarName
			}
		}
	}
	return ""
}

// SymCompiledCount returns the number of distinct WGSL programs compiled and
// cached by RunSymbolic. A value of 1 after multiple dispatches of the same
// kernel structure proves compile-once behaviour.
func (d *Device) SymCompiledCount() int { return len(d.symCache) }

// ── Symbolic-shapes spike (SLICE 1) ──────────────────────────────────────────
//
// A SymKernelHandle holds a compiled GPU pipeline for a kernel that contains at
// least one symbolic (runtime-sized) loop range.  The WGSL shader reads loop
// bounds from a trailing params_n storage buffer rather than using compile-time
// literals, so the same compiled pipeline can be dispatched with different dim
// values without recompilation.
//
// Usage:
//
//	k, err := dev.CompileSymKernel(item)   // compiles the WGSL exactly once
//	defer k.Release()
//	out8, grid8, err := dev.DispatchSymKernel(k, 8, inputs8)
//	out128, grid128, err := dev.DispatchSymKernel(k, 128, inputs128)

// SymKernelHandle is an opaque handle to a compiled symbolic kernel.
type SymKernelHandle struct {
	shader         *wgpu.ShaderModule
	bgLayout       *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
	numDataParams  int // number of data buffer bindings (PARAM count from KernelInfo)
	wgsl           string
}

// WGSL returns the compiled shader source, useful for debugging.
func (k *SymKernelHandle) WGSL() string { return k.wgsl }

// Release frees GPU resources held by the handle.
func (k *SymKernelHandle) Release() {
	if k.pipeline != nil {
		k.pipeline.Release()
	}
	if k.pipelineLayout != nil {
		k.pipelineLayout.Release()
	}
	if k.bgLayout != nil {
		k.bgLayout.Release()
	}
	if k.shader != nil {
		k.shader.Release()
	}
}

// itemHasSymDim reports whether the kernel represented by item contains at least
// one symbolic AxisLoop range (i.e. a range whose bound is not known at compile time).
func itemHasSymDim(item schedule.ExecItem) bool {
	if !item.Ast.Valid() {
		return len(item.SymVars) > 0 // Ast zeroed in cached item; SymVars is ground truth
	}
	sink := item.Ast
	if sink.Op() != uop.OpSink || sink.NSrc() == 0 {
		return false
	}
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return false
	}
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			if ra, ok := r.Arg().(uop.RangeArg); ok && ra.Symbolic {
				return true
			}
		}
	}
	return false
}

// CompileSymKernel compiles the WGSL shader for item exactly once and returns a
// reusable handle.  item must contain at least one symbolic OpRange node.
//
// The bind group layout always has an extra read-only params_n binding at slot
// ki.NumParams, immediately after all data bindings — this matches what
// codegen.RenderWGSL emits for symbolic kernels.
func (d *Device) CompileSymKernel(item schedule.ExecItem) (*SymKernelHandle, error) {
	var handle *SymKernelHandle
	err := d.onGPU(func() error {
		var cerr error
		handle, cerr = d.compileSymKernelLocked(item)
		return cerr
	})
	return handle, err
}

// compileSymKernelLocked is CompileSymKernel's body; it assumes it is already
// executing on the GPU-owner goroutine and must not call onGPU.
func (d *Device) compileSymKernelLocked(item schedule.ExecItem) (*SymKernelHandle, error) {
	var wgsl string
	if item.WGSL != "" {
		wgsl = item.WGSL
	} else {
		wgsl = codegen.RenderWGSL(item)
	}
	var numData int
	if item.Ast.Valid() {
		numData = item.Ast.Arg().(uop.KernelInfo).NumParams
	} else {
		numData = len(item.Bufs) // Ast zeroed by cache; NumParams == len(Bufs) invariant
	}

	// Bind group layout: numData data buffers + 1 params_n buffer.
	totalBindings := numData + 1 // params_n is always present for symbolic kernels
	layoutEntries := make([]gputypes.BindGroupLayoutEntry, totalBindings)
	for i := 0; i < numData; i++ {
		bt := gputypes.BufferBindingTypeReadOnlyStorage
		if i == 0 {
			bt = gputypes.BufferBindingTypeStorage // output: read_write
		}
		layoutEntries[i] = gputypes.BindGroupLayoutEntry{
			Binding:    uint32(i),
			Visibility: gputypes.ShaderStageCompute,
			Buffer:     &gputypes.BufferBindingLayout{Type: bt},
		}
	}
	layoutEntries[numData] = gputypes.BindGroupLayoutEntry{
		Binding:    uint32(numData),
		Visibility: gputypes.ShaderStageCompute,
		Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform},
	}

	shader, err := d.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: wgsl})
	if err != nil {
		return nil, fmt.Errorf("CompileSymKernel CreateShaderModule: %w\n--- WGSL ---\n%s", err, wgsl)
	}

	bgLayout, err := d.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: layoutEntries,
	})
	if err != nil {
		shader.Release()
		return nil, fmt.Errorf("CompileSymKernel CreateBindGroupLayout: %w", err)
	}

	pipelineLayout, err := d.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		bgLayout.Release()
		shader.Release()
		return nil, fmt.Errorf("CompileSymKernel CreatePipelineLayout: %w", err)
	}

	pipeline, err := d.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "main",
	})
	if err != nil {
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return nil, fmt.Errorf("CompileSymKernel CreateComputePipeline: %w\n--- WGSL ---\n%s", err, wgsl)
	}

	return &SymKernelHandle{
		shader:         shader,
		bgLayout:       bgLayout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
		numDataParams:  numData,
		wgsl:           wgsl,
	}, nil
}

// DispatchSymKernel runs k with the given symbolic dimension n.
// inputs[i] provides the float32 data for PARAM(i+1) (i.e. the first input is
// inputs[0], the second is inputs[1], etc.; PARAM(0) is the output).
//
// Returns the output elements, the dispatch workgroup count (for proof-of-grid
// variance), and any error.
func (d *Device) DispatchSymKernel(k *SymKernelHandle, n int64, inputs [][]float32) (output []float32, workgroups uint32, err error) {
	var out []float32
	var wgs uint32
	rerr := d.onGPU(func() error {
		var derr error
		out, wgs, derr = d.dispatchSymKernelLocked(k, n, inputs)
		return derr
	})
	return out, wgs, rerr
}

// dispatchSymKernelLocked is DispatchSymKernel's body; it assumes it is already
// executing on the GPU-owner goroutine and must not call onGPU.
func (d *Device) dispatchSymKernelLocked(k *SymKernelHandle, n int64, inputs [][]float32) (output []float32, workgroups uint32, err error) {
	nInputs := len(inputs)

	// ── Allocate GPU buffers ──────────────────────────────────────────────
	outBuf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "sym_out",
		Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
		Size:  uint64(n) * 4,
	})
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel alloc output: %w", err)
	}
	defer outBuf.Release()

	inBufs := make([]*wgpu.Buffer, nInputs)
	for i, data := range inputs {
		buf, berr := d.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("sym_in%d", i),
			Usage: gputypes.BufferUsageStorage | gputypes.BufferUsageCopyDst,
			Size:  uint64(len(data)) * 4,
		})
		if berr != nil {
			for j := 0; j < i; j++ {
				inBufs[j].Release()
			}
			return nil, 0, fmt.Errorf("DispatchSymKernel alloc input %d: %w", i, berr)
		}
		inBufs[i] = buf
	}
	defer func() {
		for _, b := range inBufs {
			if b != nil {
				b.Release()
			}
		}
	}()

	// ── Upload input data ─────────────────────────────────────────────────
	for i, data := range inputs {
		raw := float32sToBytes(data)
		if werr := d.queue.WriteBuffer(inBufs[i], 0, raw); werr != nil {
			return nil, 0, fmt.Errorf("DispatchSymKernel upload input %d: %w", i, werr)
		}
	}

	// ── Params uniform buffer: ParamsN { data: array<u32, 4> } = 16 bytes ──
	paramsBytes := make([]byte, 16)
	binary.LittleEndian.PutUint32(paramsBytes, uint32(n))
	paramsBuf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "sym_params",
		Usage: gputypes.BufferUsageUniform | gputypes.BufferUsageCopyDst,
		Size:  16,
	})
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel alloc params: %w", err)
	}
	defer paramsBuf.Release()
	if werr := d.queue.WriteBuffer(paramsBuf, 0, paramsBytes); werr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel upload params: %w", werr)
	}

	// ── Bind group ────────────────────────────────────────────────────────
	// Layout: [out(rw), in0(r), in1(r), ..., params_n(uniform)]
	entries := make([]wgpu.BindGroupEntry, 1+nInputs+1)
	entries[0] = wgpu.BindGroupEntry{
		Binding: 0,
		Buffer:  outBuf,
		Size:    uint64(n) * 4,
	}
	for i, buf := range inBufs {
		entries[1+i] = wgpu.BindGroupEntry{
			Binding: uint32(1 + i),
			Buffer:  buf,
			Size:    uint64(len(inputs[i])) * 4,
		}
	}
	entries[1+nInputs] = wgpu.BindGroupEntry{
		Binding: uint32(1 + nInputs),
		Buffer:  paramsBuf,
		Size:    16,
	}
	bg, err := d.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  k.bgLayout,
		Entries: entries,
	})
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel CreateBindGroup: %w", err)
	}
	defer bg.Release()

	// ── Dispatch ──────────────────────────────────────────────────────────
	wgs := uint32((n + workgroupSize - 1) / workgroupSize)
	if wgs == 0 {
		wgs = 1
	}

	enc, err := d.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel CreateCommandEncoder: %w", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel BeginComputePass: %w", err)
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(wgs, 1, 1)
	if perr := pass.End(); perr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel ComputePass.End: %w", perr)
	}
	cmd, err := enc.Finish()
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel CommandEncoder.Finish: %w", err)
	}
	if _, serr := d.queue.Submit(cmd); serr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel Queue.Submit: %w", serr)
	}

	// ── Read back output ──────────────────────────────────────────────────
	out, rerr := d.readBuffer(outBuf, n, uop.Dtypes.Float32)
	if rerr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel readback: %w", rerr)
	}
	return out, wgs, nil
}
