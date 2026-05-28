package schedule

import "github.com/georgebuilds/anneal/uop"

// Buffer identifies one global materialized buffer in the schedule.
type Buffer struct {
	UOpIdx uint32     // arena index of the BUFFER uop — unique within this schedule
	Size   int64      // number of elements
	Shape  []int64    // per-dimension sizes; product == Size
	DType  *uop.DType
	Slot   int // slot assigned by memory_planner; -1 = not aliased (leaf/output)
}

// ExecItem is one executable kernel in the ordered schedule.
// Ast is the kernel SINK-rooted UOp tree (what Phase 8 codegen renders).
// Bufs[N] is the runtime buffer for the kernel's PARAM(arg=N).
// PARAM(arg=0) is always the kernel's output; PARAM(arg=1..N-1) are inputs.
// SymVars[symParamIdx] is the DefineVar name for each symbolic range parameter;
// nil for static-only kernels.
// WGSL, when non-empty, is a pre-rendered shader source that supersedes
// re-rendering Ast via codegen.  Set by the cache when Ast has been zeroed to
// release the arena reference.
// LocalSize is the [x, y, z] workgroup size computed by codegen.
// WorkgroupCount is the [x, y, z] workgroup count computed by codegen.
type ExecItem struct {
	Ast            uop.UOp
	Bufs           []Buffer
	SymVars        []string
	WGSL           string
	LocalSize      [3]int
	WorkgroupCount [3]int
}
