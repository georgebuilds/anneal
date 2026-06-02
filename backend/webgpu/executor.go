package webgpu

import (
	"encoding/binary"
	"fmt"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// executor.go is the slim WebGPU orchestrator. It composes Renderer, Compiler,
// Allocator, Program and Buffer collaborators (see renderer.go / compiler.go /
// allocator.go / program.go / buffer.go) into the Run / RunSymbolic / Benchmark
// public methods, and funnels every native call through Device.onGPU.
//
// Top-level data flow for one Run:
//
//	render-all  → compile-all  → allocate-all  → upload-inputs
//	            → dispatch-each → readback-outputs → release-buffers
//
// Each step delegates to the relevant collaborator; the orchestrator only
// handles classification (which buffer is a leaf input vs intermediate vs
// final output) and the slot-shared allocation maximum-element calculation.

// Run executes a compiled schedule on this device.
// inputs maps Buffer.UOpIdx → flat float32 data for leaf (external-input) buffers.
// Returns output data keyed by Buffer.UOpIdx for final output buffers (not read
// by any subsequent kernel in the schedule).
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

// runLocked is Run's body; assumes it is already executing on the GPU-owner
// goroutine and must not call onGPU.
func (d *Device) runLocked(items []schedule.ExecItem, inputs map[uint32][]float32) (map[uint32][]float32, error) {
	if err := requireShaderF16(d, items); err != nil {
		return nil, err
	}

	// ── Render all WGSL shaders before touching the GPU ─────────────────
	wgsls := make([]string, len(items))
	for i := range items {
		if items[i].WGSL != "" {
			wgsls[i] = items[i].WGSL // pre-rendered by cache; Ast may be zeroed
		} else {
			res := d.renderer.Render(items[i])
			wgsls[i] = res.WGSL
			items[i].WGSL = res.WGSL
			items[i].LocalSize = res.LocalSize
			items[i].WorkgroupCount = res.WorkgroupCount
		}
	}

	// ── Compile all programs (cached by WGSL source) ────────────────────
	progs := make([]backend.Program, len(items))
	for i, item := range items {
		nParams := kernelNumParams(item)
		p, err := d.compiler.Compile(wgsls[i], backend.KernelMeta{NumStorageBuffers: nParams})
		if err != nil {
			return nil, fmt.Errorf("webgpu: compile kernel %d: %w", i, err)
		}
		progs[i] = p
	}

	// ── Allocate GPU buffers ─────────────────────────────────────────────
	alloc := newAllocator(d)
	defer alloc.Reset() // dedup'd batch release after final readback

	gpuBufs, err := d.allocateBuffers(items, alloc, staticElems{}, "")
	if err != nil {
		return nil, err
	}

	// ── Upload leaf input data ───────────────────────────────────────────
	bufDType := buildBufDTypeMap(items)
	if err := uploadInputs(gpuBufs, inputs, bufDType); err != nil {
		return nil, err
	}

	// ── Execute kernels in schedule order ───────────────────────────────
	for i, item := range items {
		args, derr := buildDispatchArgs(item, item.WorkgroupCount, gpuBufs, nil)
		if derr != nil {
			return nil, fmt.Errorf("webgpu: kernel %d: %w", i, derr)
		}
		if err := progs[i].Dispatch(args); err != nil {
			return nil, fmt.Errorf("webgpu: kernel %d: %w", i, err)
		}
	}

	// ── Read back final outputs (implicit GPU sync) ─────────────────────
	outputs, err := readbackOutputs(items, gpuBufs, staticElems{})
	if err != nil {
		return nil, err
	}
	return outputs, nil
}

// RunSymbolic executes a schedule that may contain symbolic kernels.
// binding maps DefineVar name → concrete int64 value for this dispatch.
// Symbolic kernels are compiled once (keyed by WGSL source) and reused across
// calls with the same kernel structure but different binding values.
// Schedules may contain a mix of symbolic and fully-concrete kernels.
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

// runSymbolicLocked is RunSymbolic's body; assumes it is already executing on
// the GPU-owner goroutine and must not call onGPU.
func (d *Device) runSymbolicLocked(items []schedule.ExecItem, inputs map[uint32][]float32, binding map[string]int64) (map[uint32][]float32, error) {
	if err := requireShaderF16(d, items); err != nil {
		return nil, err
	}

	// ── Render + compile all kernels (cached by WGSL source) ────────────
	wgsls := make([]string, len(items))
	progs := make([]backend.Program, len(items))
	isSym := make([]bool, len(items))
	for i := range items {
		if items[i].WGSL != "" {
			wgsls[i] = items[i].WGSL // pre-rendered by cache; Ast may be zeroed
		} else {
			res := d.renderer.Render(items[i])
			wgsls[i] = res.WGSL
			items[i].WGSL = res.WGSL
			items[i].LocalSize = res.LocalSize
			items[i].WorkgroupCount = res.WorkgroupCount
			items[i].SymDispatch = res.SymDispatch
		}
		nParams := kernelNumParams(items[i])
		hasSym := len(items[i].SymVars) > 0
		isSym[i] = hasSym
		p, err := d.compiler.Compile(wgsls[i], backend.KernelMeta{
			NumStorageBuffers: nParams,
			HasParamsUniform:  hasSym,
		})
		if err != nil {
			return nil, fmt.Errorf("webgpu: RunSymbolic compile kernel %d: %w", i, err)
		}
		progs[i] = p
	}

	// ── Allocate GPU buffers ─────────────────────────────────────────────
	alloc := newAllocator(d)
	defer alloc.Reset()

	elems := symElems{binding: binding, items: items}
	gpuBufs, err := d.allocateBuffers(items, alloc, elems, "sym")
	if err != nil {
		return nil, err
	}

	// ── Upload leaf input data ───────────────────────────────────────────
	bufDType := buildBufDTypeMap(items)
	if err := uploadInputs(gpuBufs, inputs, bufDType); err != nil {
		return nil, err
	}

	// ── Execute kernels in schedule order ───────────────────────────────
	for i, item := range items {
		var params []byte
		wc := item.WorkgroupCount
		if isSym[i] {
			pb, perr := buildParamsBytes(item.SymVars, binding)
			if perr != nil {
				return nil, fmt.Errorf("webgpu: RunSymbolic kernel %d: %w", i, perr)
			}
			params = pb
			// Compute per-dim workgroup count from binding + SymDispatch.
			wcEval, werr := computeSymDispatchWCFromItem(item, binding)
			if werr != nil {
				return nil, fmt.Errorf("webgpu: RunSymbolic kernel %d: %w", i, werr)
			}
			wc = wcEval
		}
		args, derr := buildDispatchArgs(item, wc, gpuBufs, params)
		if derr != nil {
			return nil, fmt.Errorf("webgpu: RunSymbolic kernel %d: %w", i, derr)
		}
		if err := progs[i].Dispatch(args); err != nil {
			return nil, fmt.Errorf("webgpu: RunSymbolic kernel %d: %w", i, err)
		}
	}

	// ── Read back final outputs ─────────────────────────────────────────
	outputs, err := readbackOutputs(items, gpuBufs, elems)
	if err != nil {
		return nil, err
	}
	return outputs, nil
}

// SymCompiledCount returns the number of distinct symbolic-WGSL programs
// compiled and cached by RunSymbolic / CompileSymKernel. A value of 1 after
// multiple dispatches of the same kernel structure proves compile-once
// behaviour. Static (non-symbolic) kernels do not contribute.
func (d *Device) SymCompiledCount() int { return d.compiler.SymbolicCount() }

// ── Shared helpers ──────────────────────────────────────────────────────────

// requireShaderF16 fails closed if any kernel in items needs f16 but the
// device does not have shader-f16 enabled.
func requireShaderF16(d *Device, items []schedule.ExecItem) error {
	if d.HasShaderF16 {
		return nil
	}
	for _, item := range items {
		for _, buf := range item.Bufs {
			if buf.DType != nil && buf.DType.Scalar() == uop.Dtypes.Float16 {
				return fmt.Errorf("webgpu: kernel requires shader-f16 but adapter does not support it — enable the extension at device open time or use f32")
			}
		}
	}
	return nil
}

// kernelNumParams returns the storage-buffer count for a kernel: either from
// the AST's KernelInfo.NumParams or, if the AST has been zeroed by the cache,
// from len(Bufs) (which equals NumParams by schedule invariant).
func kernelNumParams(item schedule.ExecItem) int {
	if item.Ast.Valid() {
		return item.Ast.Arg().(uop.KernelInfo).NumParams
	}
	return len(item.Bufs)
}

// buildBufDTypeMap collects a UOpIdx → dtype lookup over every buffer in the
// schedule. Used by uploadInputs to encode host float32 data into the right
// per-element layout (f16, bf16, f32).
func buildBufDTypeMap(items []schedule.ExecItem) map[uint32]*uop.DType {
	m := make(map[uint32]*uop.DType, len(items)*4)
	for _, item := range items {
		for _, buf := range item.Bufs {
			m[buf.UOpIdx] = buf.DType
		}
	}
	return m
}

// elemCounter abstracts how an output / leaf buffer's element count is
// determined. The static path uses schedule.Buffer.Size directly; the
// symbolic path evaluates binding × per-dim multipliers via symElemCount.
type elemCounter interface {
	elems(buf schedule.Buffer, item schedule.ExecItem) int64
}

// staticElems is the elemCounter used by Run: returns buf.Size unchanged.
type staticElems struct{}

func (staticElems) elems(buf schedule.Buffer, _ schedule.ExecItem) int64 { return buf.Size }

// symElems is the elemCounter used by RunSymbolic: evaluates each symbolic
// dim against the binding (delegating to symElemCount).
type symElems struct {
	binding map[string]int64
	items   []schedule.ExecItem
}

func (s symElems) elems(buf schedule.Buffer, item schedule.ExecItem) int64 {
	return symElemCount(buf, s.binding, item.SymVars)
}

// allocateBuffers performs the three-phase allocation pattern used by both
// runLocked and runSymbolicLocked:
//
//	Phase A: slot-shared intermediate buffers (Slot >= 0), one allocation per
//	         slot sized to the max consumer.
//	Phase B: dedicated final-output buffers (Slot == -1, written by a kernel).
//	Phase C: leaf input buffers (read by some kernel, written by none).
//
// labelPrefix is prepended to the per-buffer debug label (e.g. "sym" makes
// labels read "symslot0", "symout42", "symleaf17"); empty string yields the
// static-path labels ("slot0", "out42", "leaf17").
func (d *Device) allocateBuffers(items []schedule.ExecItem, alloc *allocator, ec elemCounter, labelPrefix string) (map[uint32]backend.DeviceBuffer, error) {
	writtenBy := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenBy[item.Bufs[0].UOpIdx] = i
	}

	gpuBufs := make(map[uint32]backend.DeviceBuffer, len(items)*2)

	// Phase A: slot-shared intermediates.
	slotMaxElems := make(map[int]int64)
	for _, item := range items {
		out := item.Bufs[0]
		if out.Slot >= 0 {
			if n := ec.elems(out, item); n > slotMaxElems[out.Slot] {
				slotMaxElems[out.Slot] = n
			}
		}
	}
	for slot, maxElems := range slotMaxElems {
		label := fmt.Sprintf("%sslot%d", labelPrefix, slot)
		buf, err := alloc.AllocSlot(slot, maxElems, nil, label)
		if err != nil {
			return nil, fmt.Errorf("webgpu: alloc slot %d: %w", slot, err)
		}
		_ = buf
	}
	for _, item := range items {
		out := item.Bufs[0]
		if out.Slot >= 0 {
			buf, err := alloc.AllocSlot(out.Slot, 0, nil, "")
			if err != nil {
				return nil, fmt.Errorf("webgpu: lookup slot %d: %w", out.Slot, err)
			}
			gpuBufs[out.UOpIdx] = buf
		}
	}

	// Phase B: dedicated final outputs.
	for _, item := range items {
		out := item.Bufs[0]
		if out.Slot < 0 {
			if _, ok := gpuBufs[out.UOpIdx]; !ok {
				label := fmt.Sprintf("%sout%d", labelPrefix, out.UOpIdx)
				n := ec.elems(out, item)
				buf, err := alloc.Alloc(n, out.DType, backend.BufferUsageIO, label)
				if err != nil {
					return nil, fmt.Errorf("webgpu: alloc output buf %d: %w", out.UOpIdx, err)
				}
				gpuBufs[out.UOpIdx] = buf
			}
		}
	}

	// Phase C: leaf inputs (never written by any kernel in this schedule).
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			if _, written := writtenBy[buf.UOpIdx]; written {
				continue
			}
			if _, ok := gpuBufs[buf.UOpIdx]; ok {
				continue
			}
			label := fmt.Sprintf("%sleaf%d", labelPrefix, buf.UOpIdx)
			n := ec.elems(buf, item)
			if n == 0 {
				n = 1 // preserve the symbolic-path "actualElems==0 → 1" floor
			}
			db, err := alloc.Alloc(n, buf.DType, backend.BufferUsageLeafInput, label)
			if err != nil {
				return nil, fmt.Errorf("webgpu: alloc leaf buf %d: %w", buf.UOpIdx, err)
			}
			gpuBufs[buf.UOpIdx] = db
		}
	}
	return gpuBufs, nil
}

// uploadInputs writes leaf input data to GPU buffers, encoding host float32
// data into the per-dtype layout (f16 / bf16 / f32) via EncodeFloat32Input.
// Inputs for UOpIdx values not present in gpuBufs (e.g. for buffers not in
// this schedule) are silently skipped — same as the original behaviour.
func uploadInputs(gpuBufs map[uint32]backend.DeviceBuffer, inputs map[uint32][]float32, bufDType map[uint32]*uop.DType) error {
	for uopIdx, data := range inputs {
		db, ok := gpuBufs[uopIdx]
		if !ok {
			continue
		}
		raw := EncodeFloat32Input(data, bufDType[uopIdx])
		if err := db.Write(raw); err != nil {
			return fmt.Errorf("webgpu: upload buf %d: %w", uopIdx, err)
		}
	}
	return nil
}

// buildDispatchArgs assembles the DispatchArgs for one kernel: it resolves
// each schedule.Buffer to its DeviceBuffer in gpuBufs, in PARAM-index order.
// params is non-nil for symbolic kernels and is forwarded verbatim.
func buildDispatchArgs(item schedule.ExecItem, wc [3]int, gpuBufs map[uint32]backend.DeviceBuffer, params []byte) (backend.DispatchArgs, error) {
	bufs := make([]backend.DeviceBuffer, len(item.Bufs))
	for i, buf := range item.Bufs {
		db, ok := gpuBufs[buf.UOpIdx]
		if !ok {
			return backend.DispatchArgs{}, fmt.Errorf("missing GPU buffer for UOpIdx %d (param %d)", buf.UOpIdx, i)
		}
		bufs[i] = db
	}
	return backend.DispatchArgs{WorkgroupCount: wc, Buffers: bufs, Params: params}, nil
}

// readbackOutputs reads back every buffer that is written by some kernel and
// read by no kernel (i.e. final outputs) and decodes its bytes to float32 via
// DecodeBytesToFloat32. Implicit GPU sync happens here (readBuffer issues a
// full barrier via Poll(PollWait) inside the MapAsync path).
func readbackOutputs(items []schedule.ExecItem, gpuBufs map[uint32]backend.DeviceBuffer, ec elemCounter) (map[uint32][]float32, error) {
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}
	outputs := make(map[uint32][]float32)
	for _, item := range items {
		out := item.Bufs[0]
		if readByAny[out.UOpIdx] {
			continue
		}
		db := gpuBufs[out.UOpIdx]
		raw, err := db.Read()
		if err != nil {
			return nil, fmt.Errorf("webgpu: readback buf %d: %w", out.UOpIdx, err)
		}
		outputs[out.UOpIdx] = DecodeBytesToFloat32(raw, ec.elems(out, item), out.DType)
	}
	return outputs, nil
}

// buildParamsBytes packs the binding values for item.SymVars into the WGSL
// params_n uniform buffer layout: one u32 per var in slot order, padded up to
// a multiple of 16 bytes (codegen rounds the ParamsN field count up to a
// multiple of 4 to match WGSL's uniform-struct alignment rule).
func buildParamsBytes(symVars []string, binding map[string]int64) ([]byte, error) {
	nVars := len(symVars)
	n := (nVars*4 + 15) &^ 15
	if n < 16 {
		n = 16
	}
	out := make([]byte, n)
	for i, name := range symVars {
		bv, ok := binding[name]
		if !ok {
			return nil, fmt.Errorf("symbolic kernel missing binding for DefineVar %q (expected in binding map)", name)
		}
		binary.LittleEndian.PutUint32(out[i*4:], uint32(bv))
	}
	return out, nil
}

// ── symElemCount and friends (preserved from the original executor.go) ───────

// symElemCount returns the actual element count for buf in a symbolic schedule.
// For concrete buffers (Size>0) it returns Size directly. For symbolic buffers
// (Size==0), it multiplies the concrete dims in Shape by the binding values for
// the symbolic dims (marked as 0 in Shape). nil Shape falls back to binding[symVars[0]].
//
// When buf.SymDimMul is non-nil, each symbolic dim's contribution is multiplied
// by the corresponding entry (parallel to symbolic positions in Shape) to
// support reshape-merge derived bounds like [n*4] (multiplier 4 with var n).
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
	// Multi-dim symbolic: product over dims, using binding × per-dim multiplier
	// for symbolic (Shape[i]==0) dims.
	n := int64(1)
	symIdx := 0
	for _, s := range buf.Shape {
		if s == 0 {
			var dimSize int64
			useAffine := symIdx < len(buf.SymDimAffine)
			if useAffine {
				entry := buf.SymDimAffine[symIdx]
				dimSize = entry.Offset
				for _, t := range entry.Terms {
					bv, ok := binding[t.VarName]
					if !ok {
						bv = 0
					}
					dimSize += t.Mul * bv
				}
			} else {
				mul := int64(1)
				if symIdx < len(buf.SymDimMul) {
					mul = buf.SymDimMul[symIdx]
				}
				var name string
				if symIdx < len(buf.SymDimVar) {
					name = buf.SymDimVar[symIdx]
				} else if symIdx < len(symVars) {
					name = symVars[symIdx]
				}
				if name != "" {
					if bv, ok := binding[name]; ok {
						dimSize = mul * bv
					}
				}
			}
			n *= dimSize
			symIdx++
		} else {
			n *= s
		}
	}
	return n
}

// computeSymDispatchWCFromItem is the in-Run dispatch-WC evaluator used by
// runSymbolicLocked. It mirrors computeSymDispatchWC (which operates on a
// SymKernelHandle) but reads the per-dim DimDispatch straight off the
// ExecItem — the inputs are equivalent because RenderWGSL populates both.
func computeSymDispatchWCFromItem(item schedule.ExecItem, binding map[string]int64) ([3]int, error) {
	wc := item.WorkgroupCount
	for axis := 0; axis < 3; axis++ {
		dd := item.SymDispatch[axis]
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
		ls := int64(item.LocalSize[axis])
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

// collectVarNames walks a BoundExpr tree, inserting every VarName it
// encounters as a key in the binding map (with zero value). Used by the
// single-var DispatchSymKernel entry-point to discover what binding keys
// are needed before resolving them to the supplied n.
func collectVarNames(be schedule.BoundExpr, binding map[string]int64) {
	if be.Op == schedule.BoundOpVar {
		binding[be.VarName] = 0
		return
	}
	for _, c := range be.Children {
		collectVarNames(c, binding)
	}
}

// elemBytes returns the GPU buffer element size in bytes for a dtype, via the
// single per-dtype WGSL metadata table in codegen.
func elemBytes(d *uop.DType) uint64 {
	return codegen.WGSLTypeInfoFor(d).SizeBytes
}
