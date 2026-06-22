package backend

// KernelMeta carries binding-layout info that Compiler needs beyond raw kernel
// source. Source alone underspecifies the bind-group layout because the layout
// depends on which bindings are storage vs. uniform, and how many of each - info
// the renderer already knows.
//
// NumStorageBuffers is the number of @group(0) storage-buffer bindings (the
// PARAM(0..N-1) bindings; PARAM(0) is read_write, the rest are read-only).
// HasParamsUniform is true when the kernel also has a trailing uniform-buffer
// binding for symbolic parameters (the params_n binding the codegen emits for
// kernels that contain at least one symbolic range).
type KernelMeta struct {
	NumStorageBuffers int
	HasParamsUniform  bool
}

// Compiler compiles a backend-specific kernel source to a ready-to-dispatch
// Program. It owns the pipeline cache: Compile is allowed (and expected) to
// return a cached Program when called with the same source. The backend's
// implementation keys by exact rendered source string. Identifier-stability in
// the renderer (SPEC §7.7c) ensures that two semantically identical kernels
// produce byte-identical source, so this collapses correctly without explicit
// normalization at the cache layer.
//
// src is the sole source of truth for binding shape: on a cache hit, meta is
// not re-validated, because the renderer guarantees that the same src implies
// the same binding layout. Callers must not pass inconsistent meta for the
// same src; doing so is a programming error, not a recoverable condition.
//
// Threading: Compile touches native GPU state (shader-module / pipeline
// creation) and must run on the backend's GPU-owner goroutine.
//
// Lifetime: the pipeline cache is unbounded for the device's lifetime; entries
// are released only at device Close. Workloads that produce an open-ended
// number of distinct kernel sources (rare in practice; the schedule cache and
// structural-key fusion bound the typical universe) should be aware of this.
type Compiler interface {
	Compile(src string, meta KernelMeta) (Program, error)
}
