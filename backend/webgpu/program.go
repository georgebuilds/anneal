package webgpu

import (
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/georgebuilds/anneal/backend"
)

// program wraps a compiled compute pipeline plus its bind-group layout. It is
// the WebGPU implementation of backend.Program.
//
// Lifetime: programs are owned by the compiler's cache (compiler.cache) — they
// outlive a single Run / RunSymbolic call and are released only when the device
// is closed.
//
// Per-dispatch resources (bind group, params uniform buffer if any) are created
// fresh inside Dispatch and released at end of Dispatch (after submit).
//
// Threading: Dispatch and Release touch *wgpu.* and must run on the GPU-owner
// goroutine.
type program struct {
	dev *Device

	shader         *wgpu.ShaderModule
	bgLayout       *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline

	numStorageBufs   int
	hasParamsUniform bool
}

// Dispatch records and submits one compute pass for this program.
//
// args.Buffers must contain exactly NumStorageBuffers entries in binding-index
// order (Buffers[0] is the output / read_write binding). args.Params, when
// non-nil, is uploaded to a fresh uniform buffer at the trailing binding slot.
// args.WorkgroupCount is the [x, y, z] dispatch grid; a zeroed x is silently
// promoted to (1, 1, 1) to preserve the static path's tolerance for degenerate
// schedules.
func (p *program) Dispatch(args backend.DispatchArgs) error {
	if len(args.Buffers) != p.numStorageBufs {
		return fmt.Errorf("program.Dispatch: expected %d storage buffers, got %d", p.numStorageBufs, len(args.Buffers))
	}
	if p.hasParamsUniform && args.Params == nil {
		return fmt.Errorf("program.Dispatch: program declares params uniform but args.Params is nil")
	}

	// ── Build bind group ─────────────────────────────────────────────────
	nEntries := p.numStorageBufs
	if p.hasParamsUniform {
		nEntries++
	}
	entries := make([]wgpu.BindGroupEntry, nEntries)
	for i, b := range args.Buffers {
		db, ok := b.(*deviceBuffer)
		if !ok {
			return fmt.Errorf("program.Dispatch: buffer at index %d is not a webgpu.deviceBuffer (got %T)", i, b)
		}
		if db.buf == nil {
			return fmt.Errorf("program.Dispatch: buffer at index %d is already released", i)
		}
		entries[i] = wgpu.BindGroupEntry{
			Binding: uint32(i),
			Buffer:  db.buf,
			Size:    bufferByteSize(db.elems, db.dt),
		}
	}

	// ── Params uniform: created per dispatch and released at end ─────────
	var paramsBuf *wgpu.Buffer
	if p.hasParamsUniform {
		var err error
		paramsBuf, err = p.dev.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "sym_params",
			Usage: gputypes.BufferUsageUniform | gputypes.BufferUsageCopyDst,
			Size:  uint64(len(args.Params)),
		})
		if err != nil {
			return fmt.Errorf("program.Dispatch alloc params: %w", err)
		}
		defer paramsBuf.Release()
		if err := p.dev.queue.WriteBuffer(paramsBuf, 0, args.Params); err != nil {
			return fmt.Errorf("program.Dispatch upload params: %w", err)
		}
		entries[p.numStorageBufs] = wgpu.BindGroupEntry{
			Binding: uint32(p.numStorageBufs),
			Buffer:  paramsBuf,
			Size:    uint64(len(args.Params)),
		}
	}

	bg, err := p.dev.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  p.bgLayout,
		Entries: entries,
	})
	if err != nil {
		return fmt.Errorf("program.Dispatch CreateBindGroup: %w", err)
	}
	defer bg.Release()

	wc := args.WorkgroupCount
	if wc[0] == 0 {
		wc = [3]int{1, 1, 1}
	}

	enc, err := p.dev.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("program.Dispatch CreateCommandEncoder: %w", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return fmt.Errorf("program.Dispatch BeginComputePass: %w", err)
	}
	pass.SetPipeline(p.pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(uint32(wc[0]), uint32(wc[1]), uint32(wc[2]))
	if err := pass.End(); err != nil {
		return fmt.Errorf("program.Dispatch ComputePass.End: %w", err)
	}
	cmd, err := enc.Finish()
	if err != nil {
		return fmt.Errorf("program.Dispatch CommandEncoder.Finish: %w", err)
	}
	if _, err := p.dev.queue.Submit(cmd); err != nil {
		return fmt.Errorf("program.Dispatch Queue.Submit: %w", err)
	}
	return nil
}

// Release is a no-op when the Program is owned by the compiler cache: the
// orchestrator must not release cached pipelines. Use releaseGPU (called from
// compiler.releaseAll on Device.Close) for the actual native release.
func (p *program) Release() {}

func (p *program) releaseGPU() {
	if p.pipeline != nil {
		p.pipeline.Release()
		p.pipeline = nil
	}
	if p.pipelineLayout != nil {
		p.pipelineLayout.Release()
		p.pipelineLayout = nil
	}
	if p.bgLayout != nil {
		p.bgLayout.Release()
		p.bgLayout = nil
	}
	if p.shader != nil {
		p.shader.Release()
		p.shader = nil
	}
}

// Compile-time assertion that *program satisfies backend.Program.
var _ backend.Program = (*program)(nil)
