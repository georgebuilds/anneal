package schedule

import "github.com/georgebuilds/anneal/uop"

// Buffer identifies one global materialized buffer in the schedule.
type Buffer struct {
	UOpIdx uint32  // arena index of the BUFFER uop — unique within this schedule
	Size   int64   // number of elements
	Shape  []int64 // per-dimension sizes; product == Size; 0 marks a symbolic dim
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

// BoundExprOp identifies a node in a serializable expression tree used to
// evaluate a symbolic range bound at dispatch time. The supported set mirrors
// codegen's renderSymBoundExpr — Const / Var / Add / Sub / Mul / IDiv / Mod —
// which together cover bare DefineVars, reshape-merge bounds, pad/shrink Adds,
// and the LOCAL workgroup-count formula `(n + L - 1) / L`.
type BoundExprOp uint8

const (
	BoundOpConst BoundExprOp = iota + 1
	BoundOpVar
	BoundOpAdd
	BoundOpSub
	BoundOpMul
	BoundOpIDiv
	BoundOpMod
)

// BoundExpr is one node in a symbolic-bound expression tree, evaluable at
// dispatch time against a binding map. Leaves use Const or VarName; binary
// ops use Children[0] and Children[1]. Construction lives in codegen
// (boundExprFromUOp), which walks the corresponding UOp.
type BoundExpr struct {
	Op       BoundExprOp
	Const    int64
	VarName  string
	Children []BoundExpr
}

// Eval returns e's integer value under binding. Reports an error for an
// unbound variable; all other failure modes panic (malformed tree).
func (e BoundExpr) Eval(binding map[string]int64) (int64, error) {
	switch e.Op {
	case BoundOpConst:
		return e.Const, nil
	case BoundOpVar:
		v, ok := binding[e.VarName]
		if !ok {
			return 0, &boundEvalErr{kind: "missing binding for var", name: e.VarName}
		}
		return v, nil
	case BoundOpAdd, BoundOpSub, BoundOpMul, BoundOpIDiv, BoundOpMod:
		if len(e.Children) != 2 {
			panic("schedule.BoundExpr: binary op without 2 children")
		}
		a, err := e.Children[0].Eval(binding)
		if err != nil {
			return 0, err
		}
		b, err := e.Children[1].Eval(binding)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case BoundOpAdd:
			return a + b, nil
		case BoundOpSub:
			return a - b, nil
		case BoundOpMul:
			return a * b, nil
		case BoundOpIDiv:
			if b == 0 {
				panic("schedule.BoundExpr: IDiv by zero")
			}
			return a / b, nil
		case BoundOpMod:
			if b == 0 {
				panic("schedule.BoundExpr: Mod by zero")
			}
			return a % b, nil
		}
	}
	panic("schedule.BoundExpr: unknown op")
}

type boundEvalErr struct {
	kind string
	name string
}

func (e *boundEvalErr) Error() string {
	return "schedule.BoundExpr.Eval: " + e.kind + " " + e.name
}

// DimDispatch carries the per-dispatch-dim extent decomposition for a kernel.
// At runtime: dim extent = Const * Π_k SymBounds[k].Eval(binding).
// For an all-concrete dim, SymBounds is nil and extent == Const (matches the
// lowerer-computed dimSizes[d]). For a dim with at least one symbolic range,
// the executor uses this to compute workgroupCount[d] from the binding.
type DimDispatch struct {
	Const     int64
	SymBounds []BoundExpr
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
// SymDispatch is the per-dim extent decomposition for symbolic kernels; entries
// with non-empty SymFactors instruct the runtime to override WorkgroupCount[d]
// after evaluating the affine sum against the binding.
type ExecItem struct {
	Ast            uop.UOp
	Bufs           []Buffer
	SymVars        []string
	WGSL           string
	LocalSize      [3]int
	WorkgroupCount [3]int
	SymDispatch    [3]DimDispatch
}
