package webgpu

import (
	"context"
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
func (d *Device) Run(items []schedule.ExecItem, inputs map[uint32][]float32) (map[uint32][]float32, error) {
	if len(items) == 0 {
		return nil, nil
	}

	// ── Render all WGSL shaders before touching the GPU ───────────────────
	// This also validates arena lifetime: codegen accesses the arena; the arena
	// is alive as long as the ExecItems are, which is for the duration of this call.
	wgsls := make([]string, len(items))
	for i, item := range items {
		wgsls[i] = codegen.RenderWGSL(item)
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
	for uopIdx, data := range inputs {
		gpuBuf, ok := gpuBufs[uopIdx]
		if !ok {
			continue // caller provided data for a buffer not in this schedule — skip
		}
		raw := float32sToBytes(data)
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
	ki := item.Ast.Arg().(uop.KernelInfo)
	nParams := ki.NumParams // Bufs[0]=output, Bufs[1..N-1]=inputs

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

	if err := staging.Map(context.Background(), wgpu.MapModeRead, 0, byteSize); err != nil {
		return nil, fmt.Errorf("Map: %w", err)
	}
	rng, err := staging.MappedRange(0, byteSize)
	if err != nil {
		return nil, fmt.Errorf("MappedRange: %w", err)
	}

	raw := rng.Bytes()
	result := make([]float32, nElems)
	for i := int64(0); i < nElems; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		result[i] = math.Float32frombits(bits)
	}
	rng.Release()
	return result, nil
}

// elemBytes returns the GPU buffer element size in bytes for a dtype.
// In v1 all dtypes use 4 bytes (bool→u32, int8/16→i32, etc.), matching
// wgsl.go's wgslBufferElemType promotion rules.
func elemBytes(d *uop.DType) uint64 {
	return 4
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
