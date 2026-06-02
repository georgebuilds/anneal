package backend

// DispatchArgs bundles the runtime parameters for one kernel dispatch.
//
// WorkgroupCount is the [x, y, z] workgroup grid for this launch. For symbolic
// kernels, the orchestrator computes this from the binding before calling
// Dispatch — the Program itself is binding-agnostic.
//
// Buffers are the storage-buffer arguments in binding-index order, matching the
// layout the Compiler was given when creating the Program. The output buffer is
// Buffers[0] (read_write); the rest are read-only inputs.
//
// Params, when non-nil, is the raw byte payload of the params_n uniform buffer
// (one u32 per symbolic var in name-sorted slot order, padded to a multiple of
// 16 bytes per WGSL's uniform-struct alignment rule). nil iff the Program's
// KernelMeta had HasParamsUniform=false.
type DispatchArgs struct {
	WorkgroupCount [3]int
	Buffers        []DeviceBuffer
	Params         []byte
}

// Program is one compiled, dispatchable kernel pipeline.
//
// Dispatch records and submits a compute pass with the given args. It does not
// block on GPU completion (use the buffer Read path or the device's sync to do
// so); successive Dispatch calls within one orchestration may overlap on the
// queue.
//
// Release is a backend-controlled lifecycle hook. Backends that cache Programs
// for the device lifetime (as the WebGPU impl does) may implement Release as a
// no-op and rely on Compiler-side teardown at device Close. Backends whose
// Programs hold per-dispatch native handles should release them here. The
// orchestrator does not call Release between dispatches.
//
// Threading: Dispatch and Release touch native GPU state and must run on the
// backend's GPU-owner goroutine.
type Program interface {
	Dispatch(args DispatchArgs) error
	Release()
}
