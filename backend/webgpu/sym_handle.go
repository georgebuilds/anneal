package webgpu

import (
	"encoding/binary"
	"fmt"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// SymKernelHandle is an opaque handle to a compiled symbolic kernel.
//
// In the post-refactor world the heavy lifting (pipeline cache, dispatch) lives
// on compiler / program / allocator collaborators. SymKernelHandle remains as
// the public surface because external tests / cmd code construct and dispatch
// kernels directly via CompileSymKernel + DispatchSymKernel. It carries the
// lowerer-computed sizing metadata (LocalSize / WorkgroupCount / SymDispatch)
// alongside a reference to the cached *program.
type SymKernelHandle struct {
	prog           *program
	wgsl           string
	numDataParams  int
	LocalSize      [3]int
	WorkgroupCount [3]int
	SymDispatch    [3]schedule.DimDispatch
}

// WGSL returns the compiled shader source, useful for debugging.
func (k *SymKernelHandle) WGSL() string { return k.wgsl }

// Release is a no-op. The underlying compute pipeline is owned by the
// compiler cache and freed when Device.Close runs; per-handle Release is
// kept only for backwards-compatible test ergonomics (defer k.Release()).
func (k *SymKernelHandle) Release() {}

// CompileSymKernel compiles the WGSL shader for item exactly once and returns a
// reusable handle. item must contain at least one symbolic OpRange node.
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

// compileSymKernelLocked is CompileSymKernel's body; assumes it is already
// executing on the GPU-owner goroutine and must not call onGPU.
func (d *Device) compileSymKernelLocked(item schedule.ExecItem) (*SymKernelHandle, error) {
	var wgsl string
	var ws, wc [3]int
	var sd [3]schedule.DimDispatch
	if item.WGSL != "" {
		wgsl = item.WGSL
		ws = item.LocalSize
		wc = item.WorkgroupCount
		sd = item.SymDispatch
	} else {
		res := d.renderer.Render(item)
		wgsl = res.WGSL
		ws = res.LocalSize
		wc = res.WorkgroupCount
		sd = res.SymDispatch
	}
	numData := kernelNumParams(item)

	p, err := d.compiler.Compile(wgsl, backend.KernelMeta{
		NumStorageBuffers: numData,
		HasParamsUniform:  true,
	})
	if err != nil {
		return nil, err
	}

	return &SymKernelHandle{
		prog:           p.(*program),
		wgsl:           wgsl,
		numDataParams:  numData,
		LocalSize:      ws,
		WorkgroupCount: wc,
		SymDispatch:    sd,
	}, nil
}

// DispatchSymKernelWithBinding runs k with multiple symbolic variables.
//
// varNames lists the kernel's symbolic vars in name-sorted slot order (the
// same ordering as schedule.ExecItem.SymVars / codegen's symSlot). binding
// maps each var name to its concrete int64 value for this dispatch. outElems
// is the number of output elements (used to size the output buffer and the
// dispatch grid). inputs[i] is the float32 data for PARAM(i+1).
//
// Returns the output, the dispatch workgroup count, and any error. Returns an
// error if binding is missing an entry for a var named in varNames.
func (d *Device) DispatchSymKernelWithBinding(k *SymKernelHandle, varNames []string, binding map[string]int64, outElems int64, inputs [][]float32) (output []float32, workgroups uint32, err error) {
	var out []float32
	var wgs uint32
	rerr := d.onGPU(func() error {
		var derr error
		out, wgs, derr = d.dispatchSymKernelBindingLocked(k, varNames, binding, outElems, inputs)
		return derr
	})
	return out, wgs, rerr
}

func (d *Device) dispatchSymKernelBindingLocked(k *SymKernelHandle, varNames []string, binding map[string]int64, outElems int64, inputs [][]float32) ([]float32, uint32, error) {
	alloc := newAllocator(d)
	defer alloc.Reset()

	outBuf, err := alloc.Alloc(outElems, uop.Dtypes.Float32, backend.BufferUsageIO, "sym_out")
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernelWithBinding alloc output: %w", err)
	}

	bufs := make([]backend.DeviceBuffer, 1+len(inputs))
	bufs[0] = outBuf
	for i, data := range inputs {
		ib, ierr := alloc.Alloc(int64(len(data)), uop.Dtypes.Float32, backend.BufferUsageLeafInput, fmt.Sprintf("sym_in%d", i))
		if ierr != nil {
			return nil, 0, fmt.Errorf("DispatchSymKernelWithBinding alloc input %d: %w", i, ierr)
		}
		raw := float32sToBytes(data)
		if werr := ib.Write(raw); werr != nil {
			return nil, 0, fmt.Errorf("DispatchSymKernelWithBinding upload input %d: %w", i, werr)
		}
		bufs[1+i] = ib
	}

	params, perr := buildParamsBytes(varNames, binding)
	if perr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernelWithBinding: %w", perr)
	}

	wc, derr := computeSymDispatchWC(k, binding)
	if derr != nil {
		return nil, 0, derr
	}
	wgs := uint32(wc[0])

	if err := k.prog.Dispatch(backend.DispatchArgs{
		WorkgroupCount: wc,
		Buffers:        bufs,
		Params:         params,
	}); err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernelWithBinding: %w", err)
	}

	raw, rerr := outBuf.Read()
	if rerr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernelWithBinding readback: %w", rerr)
	}
	return DecodeBytesToFloat32(raw, outElems, uop.Dtypes.Float32), wgs, nil
}

// DispatchSymKernel runs k with the given symbolic dimension n.
// inputs[i] provides the float32 data for PARAM(i+1). PARAM(0) is the output.
//
// Returns the output elements, the dispatch workgroup count (for
// proof-of-grid-variance), and any error.
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

func (d *Device) dispatchSymKernelLocked(k *SymKernelHandle, n int64, inputs [][]float32) ([]float32, uint32, error) {
	alloc := newAllocator(d)
	defer alloc.Reset()

	outBuf, err := alloc.Alloc(n, uop.Dtypes.Float32, backend.BufferUsageIO, "sym_out")
	if err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel alloc output: %w", err)
	}

	bufs := make([]backend.DeviceBuffer, 1+len(inputs))
	bufs[0] = outBuf
	for i, data := range inputs {
		ib, ierr := alloc.Alloc(int64(len(data)), uop.Dtypes.Float32, backend.BufferUsageLeafInput, fmt.Sprintf("sym_in%d", i))
		if ierr != nil {
			return nil, 0, fmt.Errorf("DispatchSymKernel alloc input %d: %w", i, ierr)
		}
		raw := float32sToBytes(data)
		if werr := ib.Write(raw); werr != nil {
			return nil, 0, fmt.Errorf("DispatchSymKernel upload input %d: %w", i, werr)
		}
		bufs[1+i] = ib
	}

	// Single-var sym path: synthesize a binding from the kernel's SymDispatch
	// (this entry-point assumes one var, matching its API contract).
	binding := map[string]int64{}
	for axis := 0; axis < 3; axis++ {
		for _, be := range k.SymDispatch[axis].SymBounds {
			collectVarNames(be, binding)
		}
	}
	for name := range binding {
		binding[name] = n
	}

	// Single-var path historically packed n into the first 16-byte slot of the
	// params uniform regardless of SymDispatch contents; preserve that.
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params, uint32(n))

	wc, werr := computeSymDispatchWC(k, binding)
	if werr != nil {
		return nil, 0, werr
	}
	wgs := uint32(wc[0])

	if err := k.prog.Dispatch(backend.DispatchArgs{
		WorkgroupCount: wc,
		Buffers:        bufs,
		Params:         params,
	}); err != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel: %w", err)
	}

	raw, rerr := outBuf.Read()
	if rerr != nil {
		return nil, 0, fmt.Errorf("DispatchSymKernel readback: %w", rerr)
	}
	return DecodeBytesToFloat32(raw, n, uop.Dtypes.Float32), wgs, nil
}

// computeSymDispatchWC computes the per-dim workgroup count for a symbolic
// kernel from k.SymDispatch and the runtime binding. For each dim with at
// least one symbolic range it evaluates the bound expressions and overrides
// wc[d]; dims with no sym ranges keep k.WorkgroupCount[d] (byte-identical with
// the static path). After per-dim computation, when only dim 0 is in use
// (1D-sym kernel) and wc[0] > 65535, excess workgroup count spreads into Y/Z
// to stay within WebGPU's per-dim limit. Skipped for genuine multi-dim sym
// (wc[1] or wc[2] > 1).
func computeSymDispatchWC(handle *SymKernelHandle, binding map[string]int64) ([3]int, error) {
	wc := handle.WorkgroupCount
	for axis := 0; axis < 3; axis++ {
		dd := handle.SymDispatch[axis]
		if len(dd.SymBounds) == 0 {
			continue
		}
		extent := dd.Const
		if extent <= 0 {
			panic(fmt.Sprintf("webgpu: SymDispatch[%d].Const=%d (must be >=1) — lowerer invariant violated", axis, extent))
		}
		for fi, be := range dd.SymBounds {
			v, err := be.Eval(binding)
			if err != nil {
				return wc, fmt.Errorf("webgpu: SymDispatch[%d].SymBounds[%d]: %w", axis, fi, err)
			}
			if v <= 0 {
				panic(fmt.Sprintf("webgpu: SymDispatch[%d].SymBounds[%d] evaluates to %d (binding %v) — non-positive sym extent",
					axis, fi, v, binding))
			}
			extent *= v
		}
		ls := int64(handle.LocalSize[axis])
		if ls < 1 {
			ls = 1
		}
		wc[axis] = int((extent + ls - 1) / ls)
		if wc[axis] == 0 {
			wc[axis] = 1
		}
	}
	if wc[1] == 1 && wc[2] == 1 && wc[0] > 65535 {
		totalWGs := int64(wc[0])
		wc[0] = 65535
		wc[1] = int((totalWGs + 65534) / 65535)
		if wc[1] > 65535 {
			totalWGs2 := int64(wc[1])
			wc[1] = 65535
			wc[2] = int((totalWGs2 + 65534) / 65535)
		}
	}
	return wc, nil
}
