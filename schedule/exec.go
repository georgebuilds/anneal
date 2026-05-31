package schedule

import "github.com/georgebuilds/anneal/uop"

// Buffer identifies one global materialized buffer in the schedule.
type Buffer struct {
	UOpIdx uint32     // arena index of the BUFFER uop — unique within this schedule
	Size   int64      // number of elements
	Shape  []int64    // per-dimension sizes; product == Size; 0 marks a symbolic dim
	DType  *uop.DType
	Slot   int // slot assigned by memory_planner; -1 = not aliased (leaf/output)

	// SymDimMul / SymDimVar, when non-nil, are parallel slices indexed by the
	// *symbolic positions* of Shape (positions where Shape[i] == 0), in dim
	// order. For symbolic dim k:
	//
	//   actual size = SymDimMul[k] * binding[SymDimVar[k]]
	//
	// where SymDimVar[k] is the variable name (matching the var name carried
	// in the executor's SymVars list). Length == count of zeros in Shape.
	// Both nil means "all multipliers are 1 and dim-order matches
	// name-sorted symVar order" (the Slice 1/2 bare-DefineVar case).
	//
	// Slice 3 introduces multipliers > 1 to support reshape-merge derived
	// bounds like [n,4] → [n*4] where the output dim's bound is Mul(n, 4)
	// and its actual size is 4 * binding[n].
	SymDimMul []int64
	SymDimVar []string

	// SymDimAffine, when non-nil, supersedes (SymDimMul, SymDimVar) for the
	// symbolic positions of Shape. Each entry encodes one dim as an affine
	// sum: size = sum(Terms[i].Mul * binding[Terms[i].VarName]) + Offset.
	// Used by Option B Slice 5 for pad/shrink output buffers whose bounds
	// are Add expressions over distinct DefineVars — outside what the
	// single-term (Mul, VarName) encoding can carry.
	//
	// Indexed parallel to the symbolic positions of Shape (positions where
	// Shape[i] == 0), in dim order — same as SymDimMul / SymDimVar.
	SymDimAffine []SymDimAffineEntry
}

// SymDimAffineEntry is the per-symbolic-dim affine bound used when the
// (SymDimMul, SymDimVar) single-term encoding is insufficient.
type SymDimAffineEntry struct {
	Terms  []uop.AffineTerm
	Offset int64
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
