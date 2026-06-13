package codegen

import (
	"fmt"
	"math"
	"strings"

	"github.com/georgebuilds/anneal/rewrite/rules"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// InstrKind identifies the type of a linearized instruction.
type InstrKind int

const (
	// InstrBoundsCheck emits: if (gid_x >= <SymBoundExpr>) { return; }
	// for symbolic kernels (Slice 7b: SymBoundExpr is the full trailingProduct
	// over all loopRanges, possibly involving multiple sym vars). For static
	// kernels the bound is encoded by workgroupCount * workgroupSize already
	// matching totalOut, so the bounds check is a no-op (Symbolic=false).
	InstrBoundsCheck InstrKind = iota
	// InstrGIDVar emits: let r_RangeID: i32 = i32((gid.x / Stride) % RangeSize);
	InstrGIDVar
	// InstrLoopBegin emits: for (var r_RangeID: i32 = 0; r_RangeID < RangeSize; r_RangeID++) {
	InstrLoopBegin
	// InstrLoopEnd emits: }
	InstrLoopEnd
	// InstrAccInit emits: var acc_AccIdx: WGSLType = Identity;
	InstrAccInit
	// InstrAccUpdate emits: acc_AccIdx = combine(AccOp, acc_AccIdx, Expr);
	InstrAccUpdate
	// InstrLet emits: let t_NodeIdx: WGSLType = Expr;
	InstrLet
	// InstrStore emits: data0[IndexExpr] = Expr;
	InstrStore
	// InstrDefineLocal emits: var<workgroup> LocalName: array<WGSLType, LocalSize>;
	InstrDefineLocal
	// InstrBarrier emits: workgroupBarrier();
	InstrBarrier
	// InstrIf emits: if (Cond) {
	InstrIf
	// InstrEndIf emits: }
	InstrEndIf
	// InstrAssign emits: IndexExpr = Expr;
	InstrAssign
	// InstrImgLaneBegin opens the image slot-dispatch lane pass: guards the
	// thread to one vec4 output slot (gid_x), declares the thread-private
	// vec4 accumulator, and opens the 4-lane loop with the tail mask
	// `if (_img_flat < TotalN)`. TotalN is the logical element count.
	InstrImgLaneBegin
	// InstrImgLaneStore assigns Expr to the _img_out component selected by
	// the current _img_lane (static-swizzle cascade on a private var).
	InstrImgLaneStore
	// InstrImgLaneEnd closes the tail mask and lane loop, then writes the
	// whole vec4 slot once: data0[gid_x] = _img_out;
	InstrImgLaneEnd
)

// Instr is one linearized instruction in the kernel. Fields are interpreted
// according to Kind; unused fields are zero.
type Instr struct {
	Kind InstrKind

	// InstrBoundsCheck, InstrStore (scalar guard)
	TotalN int64

	// InstrGIDVar, InstrLoopBegin
	RangeID   int
	RangeSize int64

	// InstrGIDVar only
	Stride    int64
	Component int // 0:x, 1:y, 2:z
	Level     int // 0:Global (gid), 1:Workgroup (wid), 2:Local (lid)

	// InstrGIDVar (symbolic-stride case): WGSL u32 expression to use as the
	// divisor instead of the literal Stride int64. Set when the stride product
	// to the inner side of this range involves a symbolic factor (Slice 7b:
	// non-outermost symbolic dim). When non-empty, supersedes Stride; the
	// renderer emits `(base / <StrideExpr>) % rangeSize` (or `base / <StrideExpr>`
	// when Symbolic). When empty, the renderer falls back to the int64 Stride
	// path (byte-identical Slice 1–7a output).
	StrideExpr string

	// InstrBoundsCheck, InstrGIDVar, InstrLoopBegin: true when the range size is
	// symbolic (read from the params_n storage buffer at runtime).
	Symbolic bool

	// InstrLoopBegin (symbolic only): which params_n slot holds the loop
	// bound when the bound is a bare DefineVar. Used only when SymBoundExpr
	// is empty; the ALU-bound path populates SymBoundExpr directly and
	// supersedes this. InstrBoundsCheck always populates SymBoundExpr and
	// ignores this field.
	SymParamIdx int

	// InstrLoopBegin / InstrBoundsCheck (symbolic only): the symbolic bound
	// rendered as a WGSL u32 expression (e.g. "(params_n.n0 * 4u)" for
	// reshape-merge derived bounds). When non-empty, supersedes
	// SymParamIdx — the renderer uses this expression directly as the
	// loop bound / dispatch multiplier. Populated by renderSymBoundExpr
	// in the lowerer for ALU-typed OpRange bounds.
	SymBoundExpr string

	// InstrLoopBegin (symbolic only): true when rules.IndexDtypeForBound for
	// this loop's bound would have selected Int64 (vmax doesn't fit in int32).
	// WGSL has no i64, so the renderer emits an acknowledging comment but
	// still produces i32 — mirroring tinygrad PR #8268's WebGPU edge case.
	// On a future non-WebGPU backend the dtype decision would be honored.
	Int64Downgraded bool

	// InstrGIDVar (symbolic only): the rendered WGSL u32 expression for this
	// range's bound, used to emit a per-axis guard `if (r{N} >= <expr>) { return; }`
	// after the r{N} let-binding. Populated for sym ranges in multi-dim sym
	// dispatch; empty for static (which uses ins.RangeSize as the literal) and
	// for the legacy 1D-flatten path. Cooperates with the static-path
	// `if (r{N} >= RangeSize)` guard at wgsl.go:207.
	AxisGuardExpr string

	// InstrAccInit, InstrAccUpdate
	AccIdx   int
	WGSLType string // for InstrAccInit
	Identity string // for InstrAccInit
	AccOp    uop.Op // for InstrAccUpdate

	// InstrLet, InstrDefineLocal
	NodeIdx uint32
	DType   *uop.DType

	// InstrDefineLocal
	LocalName string
	LocalSize int

	// InstrLet, InstrAccUpdate, InstrStore, InstrIf, InstrAssign
	Expr      string
	IndexExpr string // for InstrStore, InstrAssign (LHS)

	// Name overrides the auto-derived `t{NodeIdx}` naming for InstrLet
	// (used by the B3 register-blocking codegen to emit named rA_k_mr /
	// rB_k_nr per-K register loads).
	Name string
}

// Lower converts one kernel's SINK AST into a linear instruction sequence.
// Instructions are in emit order; loop nesting depth is tracked by the renderer.
// symDispatch carries per-dim runtime extent info for symbolic kernels; entries
// with non-empty SymFactors instruct the executor to override workgroupCount[d]
// per binding.
func Lower(item schedule.ExecItem) ([]Instr, [3]int, [3]int, [3]schedule.DimDispatch) {
	l := &lowerer{
		item:   item,
		exprOf: make(map[uint32]string),
	}
	instrs := l.lowerSink()
	return instrs, l.workgroupSize, l.workgroupCount, l.symDispatch
}

type rangeGroup struct {
	u   uop.UOp
	ra  uop.RangeArg
	lvl int // 0:Global, 1:Workgroup, 2:Local
	idx int // original index in loopRanges
}

type lowerer struct {
	item           schedule.ExecItem
	instrs         []Instr
	exprOf         map[uint32]string // arenaIdx → WGSL expression / variable name
	accCnt         int               // counter for accumulator variable names
	widenF16       bool              // when true, f16 loads/ops are widened to f32 in reduce body
	workgroupSize  [3]int            // computed from AxisLocal ranges
	workgroupCount [3]int
	symDispatch    [3]schedule.DimDispatch // per-dim runtime extent (sym kernels)
	loopRanges     []uop.UOp
	dims           [3][]rangeGroup

	// symSlot maps DefineVar arena index → params_n slot index (0..N-1).
	// Slots are assigned at the top of lowerSink in name-sorted order so the
	// mapping is a pure function of graph structure (SPEC §10). Populated
	// even when the kernel has no symbolic ranges (then empty).
	symSlot map[uint32]int

	// symSlotByName maps DefineVar name → params_n slot index. Parallel to
	// symSlot but keyed by name, so callers with only the var name (e.g.
	// emitIndex resolving Buffer.SymDimVar entries) can recover the slot
	// without walking the DefineVar UOps. Populated alongside symSlot.
	symSlotByName map[string]int

	// Per-dim AxisUpcast info (B3 register blocking).
	// upcastByDim[d] = the AxisUpcast range UOp for dim d (Valid() iff factor > 1).
	// upcastFactorByDim[d] = the upcast factor (1 if no upcast on dim d).
	upcastByDim       [3]uop.UOp
	upcastFactorByDim [3]int64

	// Per-dim AxisVectorize info (B3.7 vec4 widening).
	// vectorizeByDim[d] = the AxisVectorize range UOp for dim d.
	// vectorizeFactorByDim[d] = the vector width (1 if no vectorize on dim d).
	vectorizeByDim       [3]uop.UOp
	vectorizeFactorByDim [3]int64
	// During emitTiledReduce expansion, these record the MR/NR accumulator-name
	// templates so the final InstrStore can be expanded into MR*NR stores.
	upcastTileActive bool
	upcastMR         int
	upcastNR         int
	upcastTS         int                     // tile size from the matched OptTile
	upcastAccName    func(mr, nr int) string // returns the WGSL acc name for cell (mr, nr)
	upcastOutMSize   int64                   // real M extent (from output buffer shape) for store mask
	upcastOutNSize   int64                   // real N extent for store mask
	upcastMWgID      int                     // RangeID of M-Workgroup outer (after OptUpcast)
	upcastMLocID     int                     // RangeID of M-Local
	upcastNWgID      int                     // RangeID of N-Workgroup outer
	upcastNLocID     int                     // RangeID of N-Local

	// B3.7 OptVectorize state: set by emitTiledReduce, consumed by lowerSink store section.
	vecTileActive  bool
	vecW           int // vector width (4 for vec4<f32>)
	vecNLocOuterID int // RangeID of N_loc_outer (lid.x ranges over TS/W)
	vecNReal       int64
}

// computeDType returns the effective WGSL dtype for u.
func (l *lowerer) computeDType(u uop.UOp) *uop.DType {
	d := u.DType()
	if d == nil {
		return d
	}
	s := d.Scalar()
	if s == uop.Dtypes.BFloat16 || s == uop.Dtypes.FP8E4M3 || s == uop.Dtypes.FP8E5M2 {
		return uop.Dtypes.Float32
	}
	if l.widenF16 && s == uop.Dtypes.Float16 {
		return uop.Dtypes.Float32
	}
	return d
}

func (l *lowerer) emit(ins Instr) { l.instrs = append(l.instrs, ins) }

// symParamIdxFor returns the params_n slot index for a bare-DefineVar
// symbolic OpRange node. Returns (-1, false) if the bound is not a bare
// DefineVar (e.g. an ALU expression like 4*n); callers should fall back to
// renderSymBoundExpr in that case.
func (l *lowerer) symParamIdxFor(r uop.UOp) (int, bool) {
	bound := r.Src(0)
	if bound.Op() != uop.OpDefineVar {
		return -1, false
	}
	slot, ok := l.symSlot[bound.Index()]
	if !ok {
		panic(fmt.Sprintf("codegen: DefineVar %s not in symSlot map", bound.Arg().(uop.VarArg).Name))
	}
	return slot, true
}

// renderSymBoundExpr renders a symbolic bound UOp expression as a WGSL u32
// expression string. Used for OpRange bounds that aren't bare DefineVars
// (e.g. 4*n produced by reshape merge [n,4]→[n*4]). The result is a
// parenthesised expression suitable for use as a loop bound or buffer-size
// multiplier in WGSL.
//
// Supported ops: OpDefineVar (renders as "params_n.n{slot}"), OpConst
// (renders as "{N}u"), and OpAdd/OpSub/OpMul/OpIDiv/OpMod (recursive binary).
// Panics on any other op so that an unsupported bound shape surfaces
// loudly rather than silently rendering garbage WGSL.
func (l *lowerer) renderSymBoundExpr(b uop.UOp) string {
	switch b.Op() {
	case uop.OpDefineVar:
		slot, ok := l.symSlot[b.Index()]
		if !ok {
			panic(fmt.Sprintf("codegen: renderSymBoundExpr: DefineVar %s not in symSlot map", b.Arg().(uop.VarArg).Name))
		}
		return fmt.Sprintf("params_n.n%d", slot)
	case uop.OpConst:
		v, ok := b.Arg().(int64)
		if !ok {
			panic(fmt.Sprintf("codegen: renderSymBoundExpr: OpConst arg type %T (expected int64)", b.Arg()))
		}
		if v < 0 {
			// u32 cannot encode a negative; tinygrad's symbolic bounds are always
			// non-negative dimension sizes. Surface the unexpected case.
			panic(fmt.Sprintf("codegen: renderSymBoundExpr: negative constant %d in symbolic bound", v))
		}
		return fmt.Sprintf("%du", v)
	case uop.OpAdd:
		return fmt.Sprintf("(%s + %s)", l.renderSymBoundExpr(b.Src(0)), l.renderSymBoundExpr(b.Src(1)))
	case uop.OpSub:
		return fmt.Sprintf("(%s - %s)", l.renderSymBoundExpr(b.Src(0)), l.renderSymBoundExpr(b.Src(1)))
	case uop.OpMul:
		return fmt.Sprintf("(%s * %s)", l.renderSymBoundExpr(b.Src(0)), l.renderSymBoundExpr(b.Src(1)))
	case uop.OpIDiv:
		return fmt.Sprintf("(%s / %s)", l.renderSymBoundExpr(b.Src(0)), l.renderSymBoundExpr(b.Src(1)))
	case uop.OpMod:
		return fmt.Sprintf("(%s %% %s)", l.renderSymBoundExpr(b.Src(0)), l.renderSymBoundExpr(b.Src(1)))
	default:
		panic(fmt.Sprintf("codegen: renderSymBoundExpr: unsupported op %s in symbolic bound", b.Op()))
	}
}

// symBoundEmission selects the WGSL bound encoding for a symbolic OpRange.
// For bare-DefineVar bounds it returns (slot, "") matching pre-Slice-3 form.
// For ALU bounds (derived by reshape merge / split) it returns (-1, expr)
// where expr is the rendered WGSL u32 expression.
func (l *lowerer) symBoundEmission(r uop.UOp) (slot int, expr string) {
	if s, ok := l.symParamIdxFor(r); ok {
		return s, ""
	}
	return -1, l.renderSymBoundExpr(r.Src(0))
}

// strideAcc accumulates a WGSL u32 stride/bound expression as a product of a
// concrete int64 part and an optional symbolic-WGSL part. Multiplication is
// commutative; rendering folds the parts into a single u32 expression and
// preserves byte-identical output for all-concrete accumulators (Slice 1–7a
// regression bar).
//
// Invariants:
//   - constPart >= 1
//   - symPart is either "" or a WGSL u32 expression like "params_n.n0" or
//     "(params_n.n0 * 4u)". Already parenthesised at composition boundaries
//     so simple `a * b` concatenation is safe.
type strideAcc struct {
	constPart int64  // accumulated concrete multiplier (>= 1)
	symPart   string // accumulated WGSL u32 expression of symbolic factors ("" if none)
}

func newStrideAcc() strideAcc { return strideAcc{constPart: 1} }

// mulConst returns acc * k where k is a concrete int64. Multiplying by 1 is a
// no-op so existing accs stay unchanged for size-1 dims.
//
// Panics with a diagnostic on int64 overflow: strides drive memory-address
// arithmetic, so a silent wrap would corrupt downstream codegen. There is no
// give-up channel at this layer (every dim must produce a concrete stride),
// so fail-loud is the only safe response.
func (acc strideAcc) mulConst(k int64) strideAcc {
	if k == 1 {
		return acc
	}
	product, ok := uop.MulInt64Checked(acc.constPart, k)
	if !ok {
		panic(fmt.Sprintf("codegen: strideAcc.mulConst: int64 overflow: %d * %d (constPart * k)", acc.constPart, k))
	}
	return strideAcc{constPart: product, symPart: acc.symPart}
}

// mulSym returns acc * <expr> where expr is a WGSL u32 expression for a single
// symbolic factor. Composition uses textual ` * ` joins; renderSymBoundExpr
// already parenthesises ALU subexpressions, so no extra parens are needed.
func (acc strideAcc) mulSym(expr string) strideAcc {
	out := strideAcc{constPart: acc.constPart}
	if acc.symPart == "" {
		out.symPart = expr
	} else {
		out.symPart = acc.symPart + " * " + expr
	}
	return out
}

// isConcrete reports whether the accumulator is a pure int64 (no symbolic
// factor accumulated yet). When true, downstream emission can use the int64
// path; when false, the WGSL string path applies.
func (acc strideAcc) isConcrete() bool { return acc.symPart == "" }

// renderU32 returns a WGSL u32 expression for the accumulator. Concrete-only
// accs render as `Nu`; sym-only as the bare symPart; mixed as
// `(<symPart> * Nu)` — parenthesised so the expression composes safely when
// used as a divisor (e.g. `base / <renderU32()>` in InstrGIDVar). WGSL `*`
// and `/` are left-associative with equal precedence, so without parens
// `base / params_n.n0 * 4u` parses as `(base / params_n.n0) * 4u` —
// silently emitting wrong indices for the [4, n, 4] / sym-middle shape.
// For 1: `1u`.
func (acc strideAcc) renderU32() string {
	if acc.symPart == "" {
		return fmt.Sprintf("%du", acc.constPart)
	}
	if acc.constPart == 1 {
		return acc.symPart
	}
	return fmt.Sprintf("(%s * %du)", acc.symPart, acc.constPart)
}

// renderI32StrideFactor returns the multiplicative i32 factor for the
// accumulator, suitable as the RHS of `(<dim> * <factor>)` in emitIndex's
// load index. For concrete strides it returns `%d` matching the Slice 1–7a
// byte-exact format (no `u` suffix). For symbolic strides it casts the u32
// expression to i32 so the surrounding i32 arithmetic stays well-typed.
// Returns ("", true) when the factor is exactly 1 — callers omit the
// multiplication entirely (mirrors the old `strides[d] == 1` branch).
func (acc strideAcc) renderI32StrideFactor() (string, bool) {
	if acc.symPart == "" {
		if acc.constPart == 1 {
			return "", true
		}
		return fmt.Sprintf("%d", acc.constPart), false
	}
	if acc.constPart == 1 {
		return fmt.Sprintf("i32(%s)", acc.symPart), false
	}
	return fmt.Sprintf("i32(%s * %du)", acc.symPart, acc.constPart), false
}

// boundExprFromUOp converts a symbolic-bound UOp expression to the
// dispatch-time-evaluable schedule.BoundExpr tree. Supports the same op set
// as renderSymBoundExpr — Const / DefineVar / Add / Sub / Mul / IDiv / Mod —
// so any bound the WGSL renderer can emit, the runtime can evaluate. Panics
// on any other op so a future codegen extension that introduces a new bound
// shape gets surfaced immediately rather than silently misdispatching.
func boundExprFromUOp(u uop.UOp) schedule.BoundExpr {
	switch u.Op() {
	case uop.OpConst:
		v, ok := u.Arg().(int64)
		if !ok {
			panic(fmt.Sprintf("codegen.boundExprFromUOp: OpConst arg type %T (expected int64)", u.Arg()))
		}
		return schedule.BoundExpr{Op: schedule.BoundOpConst, Const: v}
	case uop.OpDefineVar:
		return schedule.BoundExpr{Op: schedule.BoundOpVar, VarName: u.Arg().(uop.VarArg).Name}
	case uop.OpAdd:
		return schedule.BoundExpr{Op: schedule.BoundOpAdd, Children: []schedule.BoundExpr{boundExprFromUOp(u.Src(0)), boundExprFromUOp(u.Src(1))}}
	case uop.OpSub:
		return schedule.BoundExpr{Op: schedule.BoundOpSub, Children: []schedule.BoundExpr{boundExprFromUOp(u.Src(0)), boundExprFromUOp(u.Src(1))}}
	case uop.OpMul:
		return schedule.BoundExpr{Op: schedule.BoundOpMul, Children: []schedule.BoundExpr{boundExprFromUOp(u.Src(0)), boundExprFromUOp(u.Src(1))}}
	case uop.OpIDiv:
		return schedule.BoundExpr{Op: schedule.BoundOpIDiv, Children: []schedule.BoundExpr{boundExprFromUOp(u.Src(0)), boundExprFromUOp(u.Src(1))}}
	case uop.OpMod:
		return schedule.BoundExpr{Op: schedule.BoundOpMod, Children: []schedule.BoundExpr{boundExprFromUOp(u.Src(0)), boundExprFromUOp(u.Src(1))}}
	}
	panic(fmt.Sprintf("codegen.boundExprFromUOp: unsupported op %s in symbolic bound", u.Op()))
}

// rangeBoundFactor returns the strideAcc factor for one OpRange or OpConst
// loopRange entry — the contribution of that range to a containing product
// (e.g. a stride product or the trailingProduct bounds expression). For an
// OpConst placeholder (size-1 dim already collapsed by freshRanges to const 0)
// the factor is 1; for a concrete OpRange it's the int64 RangeSize; for a
// symbolic OpRange it's the renderSymBoundExpr of the loop's bound UOp.
func (l *lowerer) rangeBoundFactor(r uop.UOp) strideAcc {
	if r.Op() == uop.OpConst {
		// Size-1 dim placeholder; product is unchanged.
		return newStrideAcc()
	}
	if uop.RangeIsSymbolic(r) {
		return newStrideAcc().mulSym(l.renderSymBoundExpr(r.Src(0)))
	}
	return newStrideAcc().mulConst(uop.RangeSize(r))
}

func (l *lowerer) lowerSink() []Instr {
	sink := l.item.Ast
	if sink.Op() != uop.OpSink {
		panic(fmt.Sprintf("codegen.Lower: expected SINK, got %s", sink.Op()))
	}
	end := sink.Src(0)   // OpEnd
	store := end.Src(0)  // OpStore
	body := store.Src(1) // kernel body expression

	// Assign params_n slots for every DefineVar reachable from this kernel.
	// Sorted by VarArg.Name inside VariablesOf for deterministic ordering.
	// symSlotByName is a parallel name → slot map used by emitIndex when
	// resolving Buffer.SymDimVar entries (which carry the var name, not the
	// arena index) to their params_n slot.
	l.symSlot = map[uint32]int{}
	l.symSlotByName = map[string]int{}
	for i, v := range uop.VariablesOf(sink) {
		l.symSlot[v.Index()] = i
		l.symSlotByName[v.Arg().(uop.VarArg).Name] = i
	}

	// Collect AxisLoop/Workgroup/Local ranges from END.src[1:]. AxisUpcast and
	// AxisVectorize ranges are tracked separately: they don't contribute to dispatch
	// dims but are paired with the immediately-preceding parallel range.
	var loopRanges []uop.UOp
	type upcastPair struct {
		upcast    uop.UOp
		outerLRIx int // index of outer in loopRanges
	}
	type vectorizePair struct {
		vec       uop.UOp
		outerLRIx int
	}
	var upcastPairs []upcastPair
	var vectorizePairs []vectorizePair
	lastNonConstIdx := -1
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			ra := r.Arg().(uop.RangeArg)
			switch ra.Type {
			case uop.AxisLoop, uop.AxisWorkgroup, uop.AxisLocal:
				loopRanges = append(loopRanges, r)
				lastNonConstIdx = len(loopRanges) - 1
			case uop.AxisUpcast:
				upcastPairs = append(upcastPairs, upcastPair{upcast: r, outerLRIx: lastNonConstIdx})
			case uop.AxisVectorize:
				vectorizePairs = append(vectorizePairs, vectorizePair{vec: r, outerLRIx: lastNonConstIdx})
			}
		} else if r.Op() == uop.OpConst {
			loopRanges = append(loopRanges, r)
			lastNonConstIdx = len(loopRanges) - 1
		}
	}
	l.loopRanges = loopRanges

	// Total output elements = product of all loop range sizes.
	totalOut := int64(1)
	hasSymRange := false
	for _, r := range loopRanges {
		if r.Op() == uop.OpConst {
			continue
		}
		if uop.RangeIsSymbolic(r) {
			totalOut = 0
			hasSymRange = true
		} else if !hasSymRange {
			totalOut *= uop.RangeSize(r)
		}
	}

	// Compute global strides for the final output store flat index. Walks
	// loopRanges right-to-left so each stride is the product of all dims to
	// its right (output-dim convention: outer dims have larger strides). The
	// stride for dim i is the product of the bounds of dims (i+1)..(n-1); when
	// any of those bounds is symbolic, the stride carries a symbolic factor —
	// rendered as a WGSL u32 expression via rangeBoundFactor. Slice 7b: fixes
	// the STOP-2 regression where left-of-sym strides silently defaulted to 0
	// for non-outermost symbolic dims (preflight §9c.STOP-2).
	globalStrides := make([]strideAcc, len(loopRanges))
	if len(loopRanges) > 0 {
		globalStrides[len(loopRanges)-1] = newStrideAcc()
		for i := len(loopRanges) - 2; i >= 0; i-- {
			rNext := loopRanges[i+1]
			factor := l.rangeBoundFactor(rNext)
			acc := globalStrides[i+1]
			if !factor.isConcrete() {
				acc = acc.mulSym(factor.symPart)
			}
			if factor.constPart != 1 {
				acc = acc.mulConst(factor.constPart)
			}
			globalStrides[i] = acc
		}
	}

	// Image-output kernels: deterministic vec4 slot dispatch — one thread per
	// vec4 output slot, four logical outputs per thread, whole slot written by
	// its single owner. Keyed on the output buffer dtype (always on, not an
	// Opt) so correctness never depends on opt selection; removes the
	// unaligned-row-stride store race of the legacy per-lane cascade.
	// Two excluded shapes:
	//   - symbolic kernels keep the legacy cascade: the per-lane flat
	//     decomposition needs concrete strides, and unaligned image strides
	//     were never supported symbolically (LIMITATIONS.md);
	//   - opt-transformed kernels (Workgroup/Local/Upcast/Vectorize ranges or
	//     tile-tagged reduces) fail loud — ActionSpace (beam.go) filters
	//     image-output kernels so BEAM never produces them; reaching here
	//     means a hand-applied opt that would reintroduce the store race.
	if l.paramIsImage(0) && !hasSymRange {
		opted := len(upcastPairs) > 0 || len(vectorizePairs) > 0 || KernelHasTiledReduce(sink)
		for _, r := range loopRanges {
			if r.Op() == uop.OpRange && r.Arg().(uop.RangeArg).Type != uop.AxisLoop {
				opted = true
			}
		}
		if opted {
			panic("codegen: image-output kernel has opt-transformed ranges — vec4 slot dispatch requires the unopted form (image kernels are excluded from the BEAM action space; do not hand-apply opts)")
		}
		return l.lowerImageSlot(body, loopRanges, globalStrides, totalOut)
	}

	// Group ranges into Axes.
	// We assign ranges to Dimensions starting from the INMOST (last) range to X.
	// Matmul: [Row, Col] -> Col is X, Row is Y.
	// The cyclic dimIdx % 3 assignment is structural (depends on loopRanges
	// position, not arena order) and is shared between static and symbolic
	// kernels — the legacy `if hasSymRange { targetDim = 0 }` collapse that
	// forced the 1D-flatten layout for sym was the only sym-specific deviation
	// and is now dropped.
	dims := [3][]rangeGroup{}
	dimIdx := 0
	for i := len(loopRanges) - 1; i >= 0; i-- {
		targetDim := dimIdx % 3
		r := loopRanges[i]
		if r.Op() == uop.OpConst {
			dims[targetDim] = append(dims[targetDim], rangeGroup{u: r, lvl: 0, idx: i})
			dimIdx++
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		switch ra.Type {
		case uop.AxisLoop:
			dims[targetDim] = append(dims[targetDim], rangeGroup{u: r, ra: ra, lvl: 0, idx: i})
			dimIdx++
		case uop.AxisLocal:
			// Expect AxisWorkgroup partner next (at i-1)
			dims[targetDim] = append(dims[targetDim], rangeGroup{u: r, ra: ra, lvl: 2, idx: i})
			if i-1 >= 0 {
				rwg := loopRanges[i-1]
				if rwg.Op() == uop.OpRange {
					rawg := rwg.Arg().(uop.RangeArg)
					if rawg.Type == uop.AxisWorkgroup {
						dims[targetDim] = append(dims[targetDim], rangeGroup{u: rwg, ra: rawg, lvl: 1, idx: i - 1})
						i--
					}
				}
			}
			dimIdx++
		case uop.AxisWorkgroup:
			dims[targetDim] = append(dims[targetDim], rangeGroup{u: r, ra: ra, lvl: 1, idx: i})
			dimIdx++
		default:
			dimIdx++
		}
	}
	l.dims = dims

	// Pair each AxisUpcast with its outer's dim. The upcast contributes a
	// per-thread "stripe factor" in that dim but does NOT participate in
	// dispatch (workgroup_size or workgroup_count). Register a placeholder
	// expression — emitTiledReduce overrides this per (mr, nr) iteration.
	l.upcastFactorByDim = [3]int64{1, 1, 1}
	for _, p := range upcastPairs {
		if p.outerLRIx < 0 || p.outerLRIx >= len(loopRanges) {
			continue
		}
		outer := loopRanges[p.outerLRIx]
		for d := 0; d < 3; d++ {
			found := false
			for _, rg := range dims[d] {
				if rg.u == outer {
					l.upcastByDim[d] = p.upcast
					l.upcastFactorByDim[d] *= uop.RangeSize(p.upcast)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		l.exprOf[p.upcast.Index()] = "0"
	}

	// Pair each AxisVectorize with its outer's dim. Like upcast, the vector inner
	// does not participate in dispatch. Register placeholder "0" — emitTiledReduce
	// overrides this with component-indexed expressions in the vec4 path.
	l.vectorizeFactorByDim = [3]int64{1, 1, 1}
	for _, p := range vectorizePairs {
		if p.outerLRIx < 0 || p.outerLRIx >= len(loopRanges) {
			continue
		}
		outer := loopRanges[p.outerLRIx]
		for d := 0; d < 3; d++ {
			found := false
			for _, rg := range dims[d] {
				if rg.u == outer {
					l.vectorizeByDim[d] = p.vec
					l.vectorizeFactorByDim[d] = uop.RangeSize(p.vec)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		l.exprOf[p.vec.Index()] = "0"
	}

	// Compute strides and local sizes for each (dimension, level).
	l.workgroupSize = [3]int{1, 1, 1}
	l.workgroupCount = [3]int{1, 1, 1}
	l.symDispatch = [3]schedule.DimDispatch{{Const: 1}, {Const: 1}, {Const: 1}}
	dimSizes := [3]int64{1, 1, 1}

	for d := 0; d < 3; d++ {
		for _, rg := range dims[d] {
			if rg.u.Op() == uop.OpConst {
				continue
			}
			if uop.RangeIsSymbolic(rg.u) {
				// Symbolic dim contributes 0 to the static dispatch count;
				// the actual size is resolved per-launch from params_n.
				// The trailing `workgroupCount[d] == 0 → 1` guard preserves
				// a valid static plan that runSymKernelWithHandle overrides.
				dimSizes[d] *= 0
				// Per-dim sym dispatch (multi-dim sym): capture the range's
				// bound as a BoundExpr; the executor evaluates the product
				// (Const × Π SymBounds) at dispatch time to compute
				// workgroupCount[d]. boundExprFromUOp panics on unsupported
				// shapes so a regression surfaces immediately.
				l.symDispatch[d].SymBounds = append(l.symDispatch[d].SymBounds, boundExprFromUOp(rg.u.Src(0)))
				continue
			}
			dimSizes[d] *= uop.RangeSize(rg.u)
			l.symDispatch[d].Const *= uop.RangeSize(rg.u)
		}

		for lvl := 0; lvl < 3; lvl++ {
			var levelRanges []rangeGroup
			for _, rg := range dims[d] {
				if rg.lvl == lvl {
					levelRanges = append(levelRanges, rg)
				}
			}
			if len(levelRanges) == 0 {
				continue
			}

			// Stride calculation within a dimension/level.
			// Since we collected ranges in reverse order, levelRanges are also reversed.
			// Matmul Col/Row -> Row is in dim 1, Col is in dim 0.
			// If we had multi-range components, the outermost (first in loopRanges)
			// would be at the end of levelRanges.
			//
			// Slice 7b: strides accumulate as strideAcc to support symbolic factors
			// when a sym range is inner to another range in the same (dim, level)
			// group. For all-concrete kernels the constPart-only path is byte-
			// identical with the previous int64 representation.
			strides := make([]strideAcc, len(levelRanges))
			strides[0] = newStrideAcc()
			for i := 1; i < len(levelRanges); i++ {
				rPrev := levelRanges[i-1].u
				factor := l.rangeBoundFactor(rPrev)
				acc := strides[i-1]
				if !factor.isConcrete() {
					acc = acc.mulSym(factor.symPart)
				}
				if factor.constPart != 1 {
					acc = acc.mulConst(factor.constPart)
				}
				strides[i] = acc
			}

			for i, rg := range levelRanges {
				if rg.u.Op() == uop.OpConst {
					l.exprOf[rg.u.Index()] = "0u"
					continue
				}
				sym := uop.RangeIsSymbolic(rg.u)
				var rangeSize int64
				if !sym {
					rangeSize = uop.RangeSize(rg.u)
				}
				// Slice 7b: when the stride product carries a symbolic factor,
				// emit a StrideExpr WGSL expression; otherwise use the int64
				// Stride for byte-identical Slice 1–7a output. Defensive panic:
				// if a concrete stride for a non-Const range comes out as 0,
				// that means the old "zero-default" sentinel leaked into the
				// new path (per design call F).
				instr := Instr{
					Kind:      InstrGIDVar,
					RangeID:   rg.ra.ID,
					RangeSize: rangeSize,
					Symbolic:  sym,
					Component: d,
					Level:     lvl,
				}
				if strides[i].isConcrete() {
					instr.Stride = strides[i].constPart
					if instr.Stride == 0 {
						panic(fmt.Sprintf("codegen: per-(dim=%d, lvl=%d) stride=0 for RangeID=%d — old zero-default sentinel leaked into Slice 7b path",
							d, lvl, rg.ra.ID))
					}
				} else {
					instr.StrideExpr = strides[i].renderU32()
				}
				// Slice 7b: for a NON-OUTERMOST symbolic range, the WGSL must
				// apply a mod against the range's bound to extract that dim's
				// contribution from the flat 1D dispatch index. Outermost-sym
				// is exempt — its (dim, level) group has no factor above it,
				// so the gid_x / outer-stride is already in [0, bound).
				// levelRanges is inmost-first, so the outermost is at index len-1.
				//
				// Multi-dim sym dispatch (this slice): per-axis guards mask out
				// LOCAL-padding when L∤bound — populated for every sym range
				// so the wgsl renderer emits `if (r{N} >= AxisGuardExpr) { return; }`
				// after the let-binding. Redundant-but-correct when paired with
				// a mod, since mod constrains r to [0, bound) already.
				if sym && i != len(levelRanges)-1 {
					instr.SymBoundExpr = l.renderSymBoundExpr(rg.u.Src(0))
				}
				if sym {
					instr.AxisGuardExpr = l.renderSymBoundExpr(rg.u.Src(0))
				}
				l.emit(instr)
				l.exprOf[rg.u.Index()] = fmt.Sprintf("r%d", rg.ra.ID)
			}

			if lvl == 2 {
				totalLocal := int64(1)
				for _, rg := range levelRanges {
					if rg.u.Op() != uop.OpConst {
						totalLocal *= uop.RangeSize(rg.u)
					}
				}
				l.workgroupSize[d] = int(totalLocal)
			}
		}

		if d == 0 && l.workgroupSize[0] == 1 {
			hasGlobal := false
			for _, rg := range dims[0] {
				if (rg.lvl == 0 || rg.lvl == 1) && rg.u.Op() != uop.OpConst {
					hasGlobal = true
					break
				}
			}
			if hasGlobal {
				l.workgroupSize[0] = 64
			}
		}

		l.workgroupCount[d] = int((dimSizes[d] + int64(l.workgroupSize[d]) - 1) / int64(l.workgroupSize[d]))
		if l.workgroupCount[d] == 0 {
			l.workgroupCount[d] = 1
		}
	}

	l.spreadWorkgroupCount()

	var indexTerms []string
	for i, r := range loopRanges {
		if r.Op() == uop.OpConst {
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		term := fmt.Sprintf("u32(r%d)", ra.ID)
		stride := globalStrides[i]
		if stride.isConcrete() {
			if stride.constPart == 0 {
				// Defensive panic per design call F. globalStrides[i] coming out
				// as concrete 0 is the old zero-default sentinel that Slice 7b
				// eliminates; only Const(0) loopRange placeholders (size-1 dims)
				// reach here, and those are skipped above.
				panic(fmt.Sprintf("codegen: globalStrides[%d]=0 for non-Const range — old zero-default sentinel leaked into Slice 7b path", i))
			}
			if stride.constPart > 1 {
				term = fmt.Sprintf("(%s * %du)", term, stride.constPart)
			}
		} else {
			term = fmt.Sprintf("(%s * %s)", term, stride.renderU32())
		}
		indexTerms = append(indexTerms, term)
	}
	indexExpr := joinPlus(indexTerms)
	if len(indexTerms) == 0 {
		indexExpr = "0u"
	}

	// Multi-dim sym dispatch (this slice): per-axis guards on each InstrGIDVar
	// (populated above via AxisGuardExpr) supersede the legacy single-flat
	// `if (gid_x >= trailingProduct)` bound. The bounds-check instr is still
	// emitted for symmetry with the static path but with Symbolic=false so the
	// renderer skips it — same effective behavior as static, where per-axis
	// guards are the sole padding mask.
	l.emit(Instr{Kind: InstrBoundsCheck, TotalN: totalOut, Symbolic: false})

	// Pre-emit cross-scope shared ALU UOps at the kernel top so they live in
	// scope for every reduce-body that references them AND for any post-reduce
	// code that references them. Without this, emitReduce caches an inside-loop
	// `let t<i>` in exprOf and a later emitExpr call reuses the cached name
	// outside the closed loop, producing WGSL like:
	//
	//   for (var r51 ...) { ... let t1288: i32 = ...; ... }
	//   let t1291: i32 = (t1288 / 4);  // unresolved: t1288 went out of scope
	//
	// The pre-pass identifies any ALU UOp reachable from inside an OpReduce
	// elem-subtree AND from outside it, then runs emitExpr on each at the
	// kernel-top scope. The renderer puts those Lets before the reduce loop,
	// so subsequent emitExpr calls hit the cache regardless of scope.
	l.hoistCrossScopeShared(body)
	bodyExpr := l.emitExpr(body)

	var outBufDType *uop.DType
	if len(l.item.Bufs) > 0 {
		outBufDType = l.item.Bufs[0].DType
	}

	if l.upcastTileActive {
		// B3 register-blocking: each thread emits MR*NR masked stores. The
		// flat output index is built directly from (M, N) coordinates that
		// include the (mr, nr) stripe offset, so the loopRanges-derived
		// indexExpr (which only covers the shrunken dispatch grid) isn't
		// used here.
		MR := l.upcastMR
		NR := l.upcastNR
		TS := l.upcastTS
		Mreal := l.upcastOutMSize
		Nreal := l.upcastOutNSize
		if l.vecTileActive {
			// B3.7: each (mr, nr) accumulator is vec4<f32> covering W=4 consecutive N values.
			// lid.x ranges over 0..TS/W-1 (the N_loc_outer); actual N = nWgID*NR*TS + nr*TS + lid.x*W + component.
			W := int64(l.vecW)
			for mr := 0; mr < MR; mr++ {
				for nr := 0; nr < NR; nr++ {
					Mexpr := fmt.Sprintf("(u32(r%d) * %du + %du + u32(r%d))",
						l.upcastMWgID, MR*TS, mr*TS, l.upcastMLocID)
					NexprBase := fmt.Sprintf("(u32(r%d) * %du + %du + u32(r%d) * %du)",
						l.upcastNWgID, NR*TS, nr*TS, l.vecNLocOuterID, W)
					components := [4]string{"x", "y", "z", "w"}
					for v := int64(0); v < W; v++ {
						var Nexpr string
						if v == 0 {
							Nexpr = NexprBase
						} else {
							Nexpr = fmt.Sprintf("(%s + %du)", NexprBase, v)
						}
						cond := fmt.Sprintf("(%s < %du) && (%s < %du)", Mexpr, Mreal, Nexpr, Nreal)
						idx := fmt.Sprintf("(%s * %du + %s)", Mexpr, Nreal, Nexpr)
						l.emit(Instr{Kind: InstrIf, Expr: cond})
						l.emit(Instr{Kind: InstrStore,
							Expr:      fmt.Sprintf("%s.%s", l.upcastAccName(mr, nr), components[v]),
							IndexExpr: idx,
							DType:     outBufDType})
						l.emit(Instr{Kind: InstrEndIf})
					}
				}
			}
		} else {
			for mr := 0; mr < MR; mr++ {
				for nr := 0; nr < NR; nr++ {
					Mexpr := fmt.Sprintf("(u32(r%d) * %du + %du + u32(r%d))",
						l.upcastMWgID, MR*TS, mr*TS, l.upcastMLocID)
					Nexpr := fmt.Sprintf("(u32(r%d) * %du + %du + u32(r%d))",
						l.upcastNWgID, NR*TS, nr*TS, l.upcastNLocID)
					cond := fmt.Sprintf("(%s < %du) && (%s < %du)", Mexpr, Mreal, Nexpr, Nreal)
					idx := fmt.Sprintf("(%s * %du + %s)", Mexpr, Nreal, Nexpr)
					l.emit(Instr{Kind: InstrIf, Expr: cond})
					l.emit(Instr{Kind: InstrStore, Expr: l.upcastAccName(mr, nr), IndexExpr: idx, DType: outBufDType})
					l.emit(Instr{Kind: InstrEndIf})
				}
			}
		}
	} else {
		l.emit(Instr{Kind: InstrStore, TotalN: totalOut, Symbolic: hasSymRange, Expr: bodyExpr, IndexExpr: indexExpr, DType: outBufDType})
	}

	return l.instrs
}

// spreadWorkgroupCount folds an X workgroup count above the WebGPU per-dim
// 65535 limit into Y (and Z if needed). Applies only when Y and Z are unused,
// i.e. the 1D-flatten dispatch layouts; the renderer's flat_gid_x/gid_x
// linearization undoes the spread inside the shader.
func (l *lowerer) spreadWorkgroupCount() {
	if l.workgroupCount[0] > 65535 && l.workgroupCount[1] == 1 && l.workgroupCount[2] == 1 {
		totalWGs := int64(l.workgroupCount[0])
		l.workgroupCount[0] = 65535
		l.workgroupCount[1] = int((totalWGs + 65534) / 65535)
		if l.workgroupCount[1] > 65535 {
			totalY := int64(l.workgroupCount[1])
			l.workgroupCount[1] = 65535
			l.workgroupCount[2] = int((totalY + 65534) / 65535)
		}
	}
}

// lowerImageSlot lowers an image-output kernel as one thread per vec4 output
// slot: the thread computes the four logical elements flat = slot*4 + lane
// (lane 0..3) in a lane loop and writes the whole vec4 slot once. Single-
// thread slot ownership eliminates, by construction, the store race the
// legacy per-lane cascade has when the output row stride is not a multiple
// of 4 — a slot that straddles a dim boundary still has exactly one writer.
//
// loopRanges/globalStrides are the structures the static path derives; all
// strides are concrete here (the caller excludes symbolic kernels). Range
// indices are re-derived per lane from the flat logical index _img_flat via
// the same (flat / stride) % size decomposition the static InstrGIDVar path
// uses, so the kernel body lowers unchanged. Tail lanes (flat >= totalOut)
// are masked by the InstrImgLaneBegin guard: the body — and therefore every
// load — is skipped and the slot component keeps its 0.0 initialization; the
// allocator pads image buffers to whole slots (BufferByteSize), so the full-
// slot store stays in bounds.
func (l *lowerer) lowerImageSlot(body uop.UOp, loopRanges []uop.UOp, globalStrides []strideAcc, totalOut int64) []Instr {
	// totalOut >= 1 by construction (product of range sizes; the caller
	// excludes symbolic kernels whose sentinel is 0), so numSlots >= 1.
	numSlots := (totalOut + 3) / 4

	// Dispatch one thread per slot: 64-wide workgroups in X (the static-path
	// default), spread into Y/Z above the per-dim 65535 limit.
	l.workgroupSize = [3]int{64, 1, 1}
	l.workgroupCount = [3]int{int((numSlots + 63) / 64), 1, 1}
	l.symDispatch = [3]schedule.DimDispatch{{Const: numSlots}, {Const: 1}, {Const: 1}}
	l.spreadWorkgroupCount()

	l.emit(Instr{Kind: InstrImgLaneBegin, TotalN: totalOut})

	// Per-lane range index derivation, mirroring the static InstrGIDVar
	// arithmetic with _img_flat as the base instead of gid_x.
	for i, r := range loopRanges {
		if r.Op() == uop.OpConst {
			l.exprOf[r.Index()] = "0u"
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		stride := globalStrides[i]
		if !stride.isConcrete() || stride.constPart == 0 {
			panic(fmt.Sprintf("codegen: lowerImageSlot: non-concrete or zero stride for RangeID=%d — symbolic kernels must not reach the image slot path", ra.ID))
		}
		var expr string
		if stride.constPart == 1 {
			expr = fmt.Sprintf("i32(_img_flat %% %du)", uop.RangeSize(r))
		} else {
			expr = fmt.Sprintf("i32((_img_flat / %du) %% %du)", stride.constPart, uop.RangeSize(r))
		}
		l.emit(Instr{Kind: InstrLet, Name: fmt.Sprintf("r%d", ra.ID), WGSLType: "i32", Expr: expr})
		l.exprOf[r.Index()] = fmt.Sprintf("r%d", ra.ID)
	}

	l.hoistCrossScopeShared(body)
	l.emit(Instr{Kind: InstrImgLaneStore, Expr: l.emitExpr(body)})
	l.emit(Instr{Kind: InstrImgLaneEnd})
	return l.instrs
}

// hoistCrossScopeShared pre-emits ALU/index UOps whose let-binding would
// otherwise be emitted inside a reduce-loop scope but also referenced from a
// different scope where that identifier is not visible.
//
// Hash-consing in the UOp arena interns identical sub-expressions to a single
// node, so a loop-invariant ALU/index UOp that appears in multiple scopes
// resolves to the same arena UOp. The default emitExpr walk caches the first
// emission keyed by arena index; a later emitExpr from a different scope hits
// the cache and reuses the identifier — producing WGSL that references an
// out-of-scope `t<N>` (Naga rejects with "unresolved identifier").
//
// The pre-pass partitions the body into scope "colors":
//
//   - Color 0: the outer (kernel-top + post-reduce) scope — `body` walked with
//     each OpReduce treated as a barrier (don't descend into its elemNode).
//   - One additional color per top-level OpReduce reachable from body: the
//     elemNode subtree of that reduce. "Top-level" means reachable from body
//     without descending through another OpReduce's elemNode — nested reduces
//     are NOT colored separately because their let-bindings live INSIDE the
//     outer reduce's loop, where hoisting to kernel-top would lift them out of
//     scope of an enclosing reduce-local Range.
//
// A UOp is hoist-shared iff it is reached by 2+ distinct colors. We then walk
// the body topo, filtered to that set, and call emitExpr — emitting InstrLet
// at the current emit depth (kernel-top, before any reduce loop opens) and
// populating l.exprOf so every subsequent emitExpr call (from any scope) hits
// the cached identifier.
//
// Two manifestations covered:
//
//  1. Outer ↔ reduce: shared between outer scope and a reduce body.
//     Originally fixed for the Block-backward / LayerNorm-shaped graph in the
//     L0'-fix slice (see block_crossscope_regression_test.go). Color count
//     {outer, reduce_k} = 2 → hoist.
//
//  2. Reduce ↔ reduce: shared between two sibling top-level reduces. Surfaces
//     on ResNet-9 backward where conv-grad fuses many sibling reduces over
//     the same input-index expression. The pre-Slice fix missed this because
//     it only considered (inner ∩ outer); sibling-only shares are in inner
//     but never in outer. Color count {reduce_a, reduce_b} = 2 → hoist.
//
// We do NOT hoist Range / Const / DefineVar / Param / Buffer / DefineLocal /
// Barrier / Reduce / End / Store / Sink / After nodes (those either have no
// Let or have scope-specific naming that emitExpr handles correctly).
func (l *lowerer) hoistCrossScopeShared(body uop.UOp) {
	// Color 0: outer scope — walk body, treat OpReduce as a barrier (skip its
	// elemNode but still visit its range sources at the outer level).
	outerReachable := make(map[uint32]bool)
	// topLevelReduces collected during the same walk: any OpReduce visited at
	// the outer level is a "top-level" reduce (its elemNode hasn't been entered
	// through another reduce's elemNode).
	var topLevelReduces []uop.UOp
	seenReduce := make(map[uint32]bool)
	var walkOuter func(u uop.UOp)
	walkOuter = func(u uop.UOp) {
		if outerReachable[u.Index()] {
			return
		}
		outerReachable[u.Index()] = true
		if u.Op() == uop.OpReduce {
			if !seenReduce[u.Index()] {
				seenReduce[u.Index()] = true
				topLevelReduces = append(topLevelReduces, u)
			}
			// Skip Src(0) (elemNode), but still visit the range sources so
			// their UOps are seen at the outer level.
			for i := 1; i < u.NSrc(); i++ {
				walkOuter(u.Src(i))
			}
			return
		}
		for i := 0; i < u.NSrc(); i++ {
			walkOuter(u.Src(i))
		}
	}
	walkOuter(body)
	if len(topLevelReduces) == 0 {
		return
	}

	// Color count per UOp index. Each color increments the counter for every
	// UOp it reaches (once per color). UOps with count >= 2 are cross-scope-
	// shared and must be hoisted.
	colorCount := make(map[uint32]int, len(outerReachable))
	for idx := range outerReachable {
		colorCount[idx] = 1
	}

	// Color k (>=1): each top-level reduce's elemNode subtree. We walk WITHOUT
	// treating nested OpReduce as a barrier — a UOp inside a nested reduce IS
	// inside its enclosing top-level reduce's color. (Hoisting nested-reduce-
	// local nodes is suppressed by hoistEligible/Range-dependency: an ALU that
	// depends on a nested reduce's Range cannot be hoisted to kernel-top, but
	// such a node only appears in ONE top-level color and never in color 0, so
	// it never gets count >= 2 anyway.)
	mark := func(u uop.UOp) {
		visited := make(map[uint32]bool)
		var walk func(v uop.UOp)
		walk = func(v uop.UOp) {
			if visited[v.Index()] {
				return
			}
			visited[v.Index()] = true
			colorCount[v.Index()]++
			for i := 0; i < v.NSrc(); i++ {
				walk(v.Src(i))
			}
		}
		walk(u)
	}
	for _, r := range topLevelReduces {
		if r.NSrc() < 1 {
			continue
		}
		mark(r.Src(0))
	}

	// Build the shared set. A UOp shared by 2+ colors needs hoisting.
	shared := make(map[uint32]bool)
	for idx, n := range colorCount {
		if n >= 2 {
			shared[idx] = true
		}
	}
	if len(shared) == 0 {
		return
	}

	// Safety filter: never hoist a UOp that transitively depends on an OpRange
	// owned by any reduce (i.e. a reduce-loop-local range). Such a node, if
	// emitted at kernel-top, would reference an undefined `r<N>`. In practice
	// the color logic above already excludes these (a reduce-local-range
	// dependent only ever has color N for that single reduce, never color 0
	// or another reduce), but we belt-and-suspenders here in case a future
	// scheduler hoists a Range to a shared position.
	reduceRangeIdxs := make(map[uint32]bool)
	for _, r := range topLevelReduces {
		for i := 1; i < r.NSrc(); i++ {
			rng := r.Src(i)
			if rng.Op() == uop.OpRange {
				reduceRangeIdxs[rng.Index()] = true
			}
		}
	}
	// dependsOnReduceRange caches whether a UOp transitively uses a reduce-
	// local range. Memoized by arena index.
	depCache := make(map[uint32]bool)
	var dependsOnReduceRange func(u uop.UOp) bool
	dependsOnReduceRange = func(u uop.UOp) bool {
		if v, ok := depCache[u.Index()]; ok {
			return v
		}
		if u.Op() == uop.OpRange && reduceRangeIdxs[u.Index()] {
			depCache[u.Index()] = true
			return true
		}
		for i := 0; i < u.NSrc(); i++ {
			if dependsOnReduceRange(u.Src(i)) {
				depCache[u.Index()] = true
				return true
			}
		}
		depCache[u.Index()] = false
		return false
	}

	// Topologically order the shared UOps so we emit producers before consumers.
	// Reuse the body topo, filtering by `shared`.
	order := uop.TopoSort(body)
	for _, u := range order {
		if !shared[u.Index()] {
			continue
		}
		if !hoistEligible(u) {
			continue
		}
		if dependsOnReduceRange(u) {
			continue
		}
		// Trigger emission of u (and any unmemoized deps it references).
		// emitExpr is the canonical entry; calling it at kernel-top depth
		// emits InstrLet at this depth and caches the name in exprOf.
		l.emitExpr(u)
	}
}

// hoistEligible reports whether u should participate in the cross-scope hoist.
// Excludes nodes that emitExpr handles specially (no Let, or scope-specific
// naming the outer caller registers — Ranges/Consts/DefineVars/Params/etc).
func hoistEligible(u uop.UOp) bool {
	switch u.Op() {
	case uop.OpConst, uop.OpRange, uop.OpDefineVar, uop.OpParam,
		uop.OpBuffer, uop.OpDefineLocal, uop.OpBarrier,
		uop.OpReduce, uop.OpEnd, uop.OpStore, uop.OpSink, uop.OpAfter:
		return false
	}
	return true
}

func (l *lowerer) emitExpr(u uop.UOp) string {
	if e, ok := l.exprOf[u.Index()]; ok {
		return e
	}
	switch u.Op() {
	case uop.OpConst:
		e := constLiteral(u)
		l.exprOf[u.Index()] = e
		return e
	case uop.OpRange:
		panic(fmt.Sprintf("codegen: Range(id=%v) not registered before use", u.Arg()))
	case uop.OpDefineVar:
		// Symbolic dim referenced from a kernel body (Option B Slice 5: pad/
		// shrink amounts may be symbolic, so the predicate `r < N + n` and
		// the offset `r - lo` carry DefineVar nodes inline). params_n.n{slot}
		// is u32 in WGSL but DefineVar is Index-dtype (i32); cast at the seam
		// so the surrounding integer arithmetic stays in i32 — matches how
		// loop indices are cast at register-init time (wgsl.go:178+).
		slot, ok := l.symSlot[u.Index()]
		if !ok {
			name := "?"
			if va, ok2 := u.Arg().(uop.VarArg); ok2 {
				name = va.Name
			}
			panic(fmt.Sprintf("codegen: emitExpr: DefineVar %q not in symSlot map", name))
		}
		e := fmt.Sprintf("i32(params_n.n%d)", slot)
		l.exprOf[u.Index()] = e
		return e
	case uop.OpParam:
		e := fmt.Sprintf("data%d", int(u.Arg().(int64)))
		l.exprOf[u.Index()] = e
		return e
	case uop.OpIndex:
		return l.emitIndex(u)
	case uop.OpGatherIdx:
		return l.emitGatherIdx(u)
	case uop.OpReduce:
		return l.emitReduce(u)
	case uop.OpDefineLocal:
		name := fmt.Sprintf("sm%d", u.Index())
		l.emit(Instr{Kind: InstrDefineLocal, NodeIdx: u.Index(), LocalName: name, LocalSize: int(u.Arg().(int64)), DType: u.DType()})
		l.exprOf[u.Index()] = name
		return name
	case uop.OpBarrier:
		l.emit(Instr{Kind: InstrBarrier})
		return ""
	default:
		return l.emitALU(u)
	}
}

func (l *lowerer) emitIndex(u uop.UOp) string {
	paramNode := u.Src(0)
	isLocal := paramNode.Op() == uop.OpDefineLocal
	var paramIdx int
	var localName string
	if isLocal {
		localName = l.emitExpr(paramNode)
	} else {
		paramIdx = int(paramNode.Arg().(int64))
	}

	nDims := u.NSrc() - 1
	var flatExpr string
	switch nDims {
	case 0:
		flatExpr = "0u"
	case 1:
		flatExpr = l.emitExpr(u.Src(1))
	default:
		// Slice 7d: stride accumulation widened from int64 to strideAcc so a
		// symbolic dim contributes a WGSL u32 expression (params_n.n{slot} ×
		// SymDimMul) rather than the 0-sentinel that silently zeroed every
		// stride to its left. For local tiles and all-concrete input shapes
		// renderI32StrideFactor emits the bare int literal — byte-identical
		// to the pre-Slice-7d format `(<dim> * %d)`.
		strides := make([]strideAcc, nDims)
		strides[nDims-1] = newStrideAcc()
		for i := nDims - 2; i >= 0; i-- {
			var factor strideAcc
			if isLocal {
				// Local tiles are square 2D matmul scratchpads (sz × sz);
				// dim 1's size is the inner edge length.
				sz := int64(math.Sqrt(float64(paramNode.Arg().(int64))))
				factor = newStrideAcc().mulConst(sz)
			} else {
				factor = l.paramDimFactor(paramIdx, i+1)
			}
			acc := strides[i+1]
			if !factor.isConcrete() {
				acc = acc.mulSym(factor.symPart)
			}
			if factor.constPart != 1 {
				acc = acc.mulConst(factor.constPart)
			}
			strides[i] = acc
		}
		var terms []string
		for d := 0; d < nDims; d++ {
			dimExpr := l.emitExpr(u.Src(d + 1))
			// Defensive panic per design call D (Slice 7d closure): a
			// concrete stride of 0 for a non-trailing dim means the old
			// shape[i]==0 sentinel leaked into the new strideAcc path —
			// the exact bug class this rewrite eliminates. The trailing
			// dim's stride is always 1 (newStrideAcc()), so this only
			// fires on non-trailing dims.
			if strides[d].isConcrete() && strides[d].constPart == 0 {
				panic(fmt.Sprintf("codegen: emitIndex stride[d=%d]=0 for paramIdx=%d nDims=%d — sym-shape sentinel leak", d, paramIdx, nDims))
			}
			factor, isOne := strides[d].renderI32StrideFactor()
			if isOne {
				terms = append(terms, dimExpr)
			} else {
				terms = append(terms, fmt.Sprintf("(%s * %s)", dimExpr, factor))
			}
		}
		flatExpr = joinPlus(terms)
	}
	var rhs string
	if isLocal {
		rhs = fmt.Sprintf("%s[%s]", localName, flatExpr)
	} else {
		// Image storage: buffer is bound as array<vec4<f32>>; logical
		// element idx lives at data{i}[idx / 4u].{x,y,z,w}. The DefineLocal
		// path above is unaffected — image storage is a GPU buffer
		// concept and workgroup scratchpads stay scalar. We avoid
		// runtime-indexed component access (data{i}[slot][lane]) because
		// the naga WGSL→MSL pipeline silently degrades dynamic component
		// indexing on storage to static-lane-0 reads; a select-chain over
		// the four static swizzles is portable. flatExpr is i32-typed;
		// cast to u32 once so the / and % operators bind without
		// triggering the ambiguous-overload error.
		if l.paramIsImage(paramIdx) {
			rhs = fmt.Sprintf(
				"select(select(select(data%d[u32(%s) / 4u].w, data%d[u32(%s) / 4u].z, (u32(%s) %% 4u) == 2u), data%d[u32(%s) / 4u].y, (u32(%s) %% 4u) == 1u), data%d[u32(%s) / 4u].x, (u32(%s) %% 4u) == 0u)",
				paramIdx, flatExpr, paramIdx, flatExpr, flatExpr, paramIdx, flatExpr, flatExpr, paramIdx, flatExpr, flatExpr)
		} else {
			rhs = fmt.Sprintf("data%d[%s]", paramIdx, flatExpr)
		}
	}
	emitDType := u.DType()
	if emitDType != nil {
		s := emitDType.Scalar()
		if s == uop.Dtypes.BFloat16 || s == uop.Dtypes.FP8E4M3 || s == uop.Dtypes.FP8E5M2 {
			// bf16 and fp8 share the decoded-storage scheme: the u32 word is
			// the quantized value's f32 bit pattern, so a load is a bitcast.
			rhs = fmt.Sprintf("bitcast<f32>(%s)", rhs)
			emitDType = uop.Dtypes.Float32
		} else if l.widenF16 && s == uop.Dtypes.Float16 {
			rhs = fmt.Sprintf("f32(%s)", rhs)
			emitDType = uop.Dtypes.Float32
		}
	}
	letName := fmt.Sprintf("t%d", u.Index())
	l.emit(Instr{Kind: InstrLet, NodeIdx: u.Index(), DType: emitDType, Expr: rhs})
	l.exprOf[u.Index()] = letName
	return letName
}

// emitGatherIdx emits the scalar i32 result of a single indirect-index
// expression (an OpGatherIdx node), bound as a let so the surrounding
// emitIndex expression can splice it into a flat-offset arithmetic chain
// like any other dim index.
//
// OpGatherIdx.Src(0) is itself a complete OpIndex over the index BUFFER, so
// the actual load is already emitted by emitIndex via emitExpr; we delegate
// to it. The positional carriers in Src(1:) exist only to mark this node
// as position-dependent at the schedule level and are not emitted here.
//
// The result dtype is Dtypes.Index (i32); the inner OpIndex's dtype carries
// the index buffer's storage dtype (Int32 or UInt32), both rendered as i32
// by wgslDType.
func (l *lowerer) emitGatherIdx(u uop.UOp) string {
	inner := l.emitExpr(u.Src(0))
	l.exprOf[u.Index()] = inner
	return inner
}

func (l *lowerer) emitReduce(u uop.UOp) string {
	if tag := u.Tag(); tag != nil {
		if s, ok := tag.(string); ok && strings.HasPrefix(s, "tile:") {
			var ts int
			if _, err := fmt.Sscanf(s, "tile:%d", &ts); err != nil {
				panic(fmt.Sprintf("lower: malformed tile tag %q: %v", s, err))
			}
			return l.emitTiledReduce(u, ts)
		}
	}

	accOp := u.Arg().(uop.Op)
	elemNode := u.Src(0)
	accIdx := l.accCnt
	l.accCnt++
	outDType := u.DType()
	isF16Reduce := outDType != nil && outDType.Scalar() == uop.Dtypes.Float16
	isBF16Reduce := outDType != nil && outDType.Scalar() == uop.Dtypes.BFloat16
	isFP8Reduce := outDType != nil &&
		(outDType.Scalar() == uop.Dtypes.FP8E4M3 || outDType.Scalar() == uop.Dtypes.FP8E5M2)
	var wt, id string
	if isF16Reduce || isBF16Reduce || isFP8Reduce {
		wt = "f32"
		id = reduceIdentity(accOp, uop.Dtypes.Float32)
	} else {
		wt = wgslDType(outDType)
		id = reduceIdentity(accOp, outDType)
	}
	l.emit(Instr{Kind: InstrAccInit, AccIdx: accIdx, WGSLType: wt, Identity: id})
	redRanges := make([]uop.UOp, u.NSrc()-1)
	for i := 1; i < u.NSrc(); i++ {
		redRanges[i-1] = u.Src(i)
	}
	hasLoop := make([]bool, len(redRanges))
	for i, r := range redRanges {
		if r.Op() == uop.OpConst {
			l.exprOf[r.Index()] = constLiteral(r)
		} else {
			ra := r.Arg().(uop.RangeArg)
			if uop.RangeIsSymbolic(r) {
				slot, expr := l.symBoundEmission(r)
				downgrade := rules.IndexDtypeForBound(r.Src(0)) == uop.Dtypes.Int64
				l.emit(Instr{Kind: InstrLoopBegin, RangeID: ra.ID, Symbolic: true, SymParamIdx: slot, SymBoundExpr: expr, Int64Downgraded: downgrade})
			} else {
				l.emit(Instr{Kind: InstrLoopBegin, RangeID: ra.ID, RangeSize: uop.RangeSize(r)})
			}
			l.exprOf[r.Index()] = fmt.Sprintf("r%d", ra.ID)
			hasLoop[i] = true
		}
	}
	if isF16Reduce {
		l.widenF16 = true
	}
	elemExpr := l.emitExpr(elemNode)
	if isF16Reduce {
		l.widenF16 = false
	}
	l.emit(Instr{Kind: InstrAccUpdate, AccIdx: accIdx, AccOp: accOp, Expr: elemExpr})
	for i := range redRanges {
		if hasLoop[i] {
			l.emit(Instr{Kind: InstrLoopEnd})
		}
	}
	var name string
	if isF16Reduce {
		name = fmt.Sprintf("f16(acc%d)", accIdx)
	} else {
		name = fmt.Sprintf("acc%d", accIdx)
	}
	l.exprOf[u.Index()] = name
	return name
}

func (l *lowerer) emitTiledReduce(u uop.UOp, TS int) string {
	accOp := u.Arg().(uop.Op)
	elemNode := u.Src(0)
	rk_outer := u.Src(1)
	rk_inner := u.Src(2)

	outDType := u.DType()
	wt := wgslDType(outDType)
	id := reduceIdentity(accOp, outDType)

	if elemNode.Op() != uop.OpMul {
		panic("Tiled reduce currently only supports Mul element node")
	}
	idxA := elemNode.Src(0)
	idxB := elemNode.Src(1)
	if idxA.Op() != uop.OpIndex || idxB.Op() != uop.OpIndex {
		panic("Tiled reduce currently only supports Index sources for Mul")
	}

	// Detect B3 register-blocking upcasts paired with this tile.
	MR := int(l.upcastFactorByDim[1])
	NR := int(l.upcastFactorByDim[0])
	if MR < 1 {
		MR = 1
	}
	if NR < 1 {
		NR = 1
	}
	upcast := MR > 1 || NR > 1

	// 2. Identify dimensions M, N, K.
	raOuter := rk_outer.Arg().(uop.RangeArg)
	raInner := rk_inner.Arg().(uop.RangeArg)
	K_outer_size := uop.RangeSize(rk_outer)

	// Register rk_outer and rk_inner nodes.
	l.exprOf[rk_outer.Index()] = fmt.Sprintf("r%d", raOuter.ID)
	l.exprOf[rk_inner.Index()] = fmt.Sprintf("r%d", raInner.ID)

	// Walk the M and N index expressions of A/B once to populate exprOf
	// caches (the body's M, N references are otherwise consumed by the
	// reduce path; we want them computed once for shape inference).
	if idxA.NSrc() == 2 {
		oldExpr := l.exprOf[rk_inner.Index()]
		l.exprOf[rk_inner.Index()] = "0"
		l.emitExpr(idxA.Src(1))
		l.exprOf[rk_inner.Index()] = oldExpr
	}
	if idxB.NSrc() == 2 {
		oldExpr := l.exprOf[rk_inner.Index()]
		l.exprOf[rk_inner.Index()] = "0"
		l.emitExpr(idxB.Src(idxB.NSrc() - 1))
		l.exprOf[rk_inner.Index()] = oldExpr
	}

	paramA := int(idxA.Src(0).Arg().(int64))
	paramB := int(idxB.Src(0).Arg().(int64))

	// Use the real operand extents so upcast stripes that overshoot a padded
	// grid are masked correctly (regression guard for irregular shapes).
	M_real := l.paramShape(paramA)[0]
	K_real := l.paramShape(paramA)[1]
	N_real := l.paramShape(paramB)[1]
	K_size := K_outer_size * int64(TS)
	if K_size < K_real {
		K_size = K_real
	}

	// Identify M and N workgroup/local range IDs in the post-upcast dim layout.
	var mWgID, mLocID, nWgID, nLocID int
	for _, rg := range l.dims[1] {
		if rg.lvl == 1 {
			mWgID = rg.ra.ID
		}
		if rg.lvl == 2 {
			mLocID = rg.ra.ID
		}
	}
	for _, rg := range l.dims[0] {
		if rg.lvl == 1 {
			nWgID = rg.ra.ID
		}
		if rg.lvl == 2 {
			nLocID = rg.ra.ID
		}
	}

	K_stride_A := int64(1)
	M_stride_A := l.paramShape(paramA)[1]
	if l.paramShape(paramA)[0] == 1 {
		M_stride_A = 0
	}
	N_stride_B := int64(1)
	K_stride_B := l.paramShape(paramB)[1]
	if l.paramShape(paramB)[0] == 1 {
		K_stride_B = 0
	}

	zeroA := reduceIdentity(uop.OpAdd, idxA.DType())
	zeroB := reduceIdentity(uop.OpAdd, idxB.DType())

	if !upcast {
		// ── Original B2 tiled-reduce path (single accumulator per thread) ──
		accIdx := l.accCnt
		l.accCnt++
		l.emit(Instr{Kind: InstrAccInit, AccIdx: accIdx, WGSLType: wt, Identity: id})

		smA := l.emitExpr(l.item.Ast.Arena().New(uop.OpDefineLocal, idxA.DType(), nil, int64(TS*TS), nil))
		smB := l.emitExpr(l.item.Ast.Arena().New(uop.OpDefineLocal, idxB.DType(), nil, int64(TS*TS), nil))

		M_size := int64(l.workgroupCount[1] * l.workgroupSize[1])
		N_size := int64(l.workgroupCount[0] * l.workgroupSize[0])

		row_A := fmt.Sprintf("(u32(r%d) * %du + lid.y)", mWgID, TS)
		col_A := fmt.Sprintf("(u32(r%d) * %du + lid.x)", raOuter.ID, TS)
		row_B := fmt.Sprintf("(u32(r%d) * %du + lid.y)", raOuter.ID, TS)
		col_B := fmt.Sprintf("(u32(r%d) * %du + lid.x)", nWgID, TS)

		flat_store := fmt.Sprintf("(lid.y * %du + lid.x)", TS)

		condA := fmt.Sprintf("(%s < %du) && (%s < %du)", row_A, M_size, col_A, K_size)
		condB := fmt.Sprintf("(%s < %du) && (%s < %du)", row_B, K_size, col_B, N_size)

		loadA := fmt.Sprintf("data%d[%s * %du + %s * %du]", paramA, row_A, M_stride_A, col_A, K_stride_A)
		loadB := fmt.Sprintf("data%d[%s * %du + %s * %du]", paramB, row_B, K_stride_B, col_B, N_stride_B)

		l.emit(Instr{Kind: InstrLoopBegin, RangeID: raOuter.ID, RangeSize: K_outer_size})
		l.emit(Instr{Kind: InstrAssign, IndexExpr: fmt.Sprintf("%s[%s]", smA, flat_store),
			Expr: fmt.Sprintf("select(%s, %s, %s)", zeroA, loadA, condA)})
		l.emit(Instr{Kind: InstrAssign, IndexExpr: fmt.Sprintf("%s[%s]", smB, flat_store),
			Expr: fmt.Sprintf("select(%s, %s, %s)", zeroB, loadB, condB)})
		l.emit(Instr{Kind: InstrBarrier})
		for i := 0; i < TS; i++ {
			termA := fmt.Sprintf("%s[lid.y * %du + %du]", smA, TS, i)
			termB := fmt.Sprintf("%s[%du * %du + lid.x]", smB, i, TS)
			l.emit(Instr{Kind: InstrAccUpdate, AccIdx: accIdx, AccOp: accOp, Expr: fmt.Sprintf("(%s * %s)", termA, termB)})
		}
		l.emit(Instr{Kind: InstrBarrier})
		l.emit(Instr{Kind: InstrLoopEnd})

		l.exprOf[u.Index()] = fmt.Sprintf("acc%d", accIdx)
		return fmt.Sprintf("acc%d", accIdx)
	}

	// ── B3.7 OptVectorize path: vec4 widening on the N (X-dim) axis ──
	// Requires OptTile+OptUpcast to already be active (upcast must be true).
	// workgroup_size.x = TS/W (e.g. 4 for W=4, TS=16). Each thread (lid.x, lid.y)
	// covers W=4 consecutive N values and MR scalar M values.
	// Contiguous vec4 loads: smB[nr*TS*TS + k*TS + lid.x*W + 0..3] are stride-1. ✓
	// Global B loads: W consecutive N values from the same K row are stride-1 in B. ✓
	vecW := int(l.vectorizeFactorByDim[0])
	vecN := upcast && vecW > 1

	if vecN {
		W := vecW
		// vec4 accumulators: MR*NR vec4<f32>, one per (mr, nr) output cell.
		accBase := l.accCnt
		l.accCnt += MR * NR
		for mr := 0; mr < MR; mr++ {
			for nr := 0; nr < NR; nr++ {
				l.emit(Instr{Kind: InstrAccInit, AccIdx: accBase + mr*NR + nr,
					WGSLType: "vec4<f32>", Identity: "vec4<f32>(0.0)"})
			}
		}

		smA := l.emitExpr(l.item.Ast.Arena().New(uop.OpDefineLocal, idxA.DType(), nil, int64(MR*TS*TS), nil))
		smB := l.emitExpr(l.item.Ast.Arena().New(uop.OpDefineLocal, idxB.DType(), nil, int64(NR*TS*TS), nil))

		l.emit(Instr{Kind: InstrLoopBegin, RangeID: raOuter.ID, RangeSize: K_outer_size})

		// A tile load — MR stripes, each thread loads one element per stripe (scalar, unchanged).
		for mr := 0; mr < MR; mr++ {
			rowA := fmt.Sprintf("(u32(r%d) * %du + %du + lid.y)", mWgID, MR*TS, mr*TS)
			colA := fmt.Sprintf("(u32(r%d) * %du + lid.x)", raOuter.ID, TS)
			condA := fmt.Sprintf("(%s < %du) && (%s < %du)", rowA, M_real, colA, K_real)
			loadA := fmt.Sprintf("data%d[%s * %du + %s * %du]", paramA, rowA, M_stride_A, colA, K_stride_A)
			smIdx := fmt.Sprintf("(%du + lid.y * %du + lid.x)", mr*TS*TS, TS)
			l.emit(Instr{Kind: InstrAssign,
				IndexExpr: fmt.Sprintf("%s[%s]", smA, smIdx),
				Expr:      fmt.Sprintf("select(%s, %s, %s)", zeroA, loadA, condA)})
		}
		// B tile load — NR stripes, each thread loads W consecutive N values.
		// colB_base = nWgID*NR*TS + nr*TS + lid.x*W  (contiguous in N for fixed row)
		for nr := 0; nr < NR; nr++ {
			rowB := fmt.Sprintf("(u32(r%d) * %du + lid.y)", raOuter.ID, TS)
			colBBase := fmt.Sprintf("(u32(r%d) * %du + %du + lid.x * %du)", nWgID, NR*TS, nr*TS, W)
			for v := 0; v < W; v++ {
				var colBv string
				if v == 0 {
					colBv = colBBase
				} else {
					colBv = fmt.Sprintf("(%s + %du)", colBBase, v)
				}
				condBv := fmt.Sprintf("(%s < %du) && (%s < %du)", rowB, K_real, colBv, N_real)
				loadBv := fmt.Sprintf("data%d[%s * %du + %s * %du]", paramB, rowB, K_stride_B, colBv, N_stride_B)
				smIdxv := fmt.Sprintf("(%du + lid.y * %du + lid.x * %du + %du)", nr*TS*TS, TS, W, v)
				l.emit(Instr{Kind: InstrAssign,
					IndexExpr: fmt.Sprintf("%s[%s]", smB, smIdxv),
					Expr:      fmt.Sprintf("select(%s, %s, %s)", zeroB, loadBv, condBv)})
			}
		}
		l.emit(Instr{Kind: InstrBarrier})

		// Unrolled inner-K loop. Each k step:
		//   - Loads MR scalar rA values from smA (unchanged from B3).
		//   - Loads NR vec4 rBv values from smB: 4 contiguous N elements per stripe.
		//     smB[nr*TS*TS + k*TS + lid.x*W + 0..3] are stride-1. ✓
		//   - Issues MR*NR (scalar * vec4) FMAs → updates vec4 accumulators.
		regDTA := idxA.DType()
		for k := 0; k < TS; k++ {
			for mr := 0; mr < MR; mr++ {
				name := fmt.Sprintf("rA_%d_%d", k, mr)
				expr := fmt.Sprintf("%s[%du + lid.y * %du + %du]", smA, mr*TS*TS, TS, k)
				l.emit(Instr{Kind: InstrLet, Name: name, DType: regDTA, Expr: expr})
			}
			for nr := 0; nr < NR; nr++ {
				name := fmt.Sprintf("rBv_%d_%d", k, nr)
				base := fmt.Sprintf("%du + %du + lid.x * %du", nr*TS*TS, k*TS, W)
				expr := fmt.Sprintf("vec4<f32>(%s[%s + 0u], %s[%s + 1u], %s[%s + 2u], %s[%s + 3u])",
					smB, base, smB, base, smB, base, smB, base)
				l.emit(Instr{Kind: InstrLet, Name: name, WGSLType: "vec4<f32>", Expr: expr})
			}
			for mr := 0; mr < MR; mr++ {
				for nr := 0; nr < NR; nr++ {
					expr := fmt.Sprintf("(rA_%d_%d * rBv_%d_%d)", k, mr, k, nr)
					l.emit(Instr{Kind: InstrAccUpdate, AccIdx: accBase + mr*NR + nr, AccOp: accOp, Expr: expr})
				}
			}
		}
		l.emit(Instr{Kind: InstrBarrier})
		l.emit(Instr{Kind: InstrLoopEnd})

		// Hand off state to lowerSink store section.
		l.upcastTileActive = true
		l.vecTileActive = true
		l.vecW = W
		l.vecNLocOuterID = nLocID // N_loc_outer keeps original N_loc ID (applyVectorize outer policy)
		l.vecNReal = N_real
		l.upcastMR = MR
		l.upcastNR = NR
		l.upcastTS = TS
		l.upcastOutMSize = M_real
		l.upcastOutNSize = N_real
		l.upcastMWgID = mWgID
		l.upcastMLocID = mLocID
		l.upcastNWgID = nWgID
		l.upcastNLocID = nLocID
		l.upcastAccName = func(mr, nr int) string {
			return fmt.Sprintf("acc%d", accBase+mr*NR+nr)
		}
		l.exprOf[u.Index()] = fmt.Sprintf("acc%d", accBase)
		return fmt.Sprintf("acc%d", accBase)
	}

	// ── B3 OptUpcast register-blocking path (scalar, no vectorize) ──
	// Workgroup output tile: (MR*TS) rows × (NR*TS) cols, with workgroup_size = (TS, TS).
	// Each thread (lid.y, lid.x) owns MR×NR output cells, separated by TS in each dim.
	// A tile in smem: MR stripes of TS×TS. B tile in smem: NR stripes of TS×TS.
	// Per outer-K-tile step: each thread does MR + NR cooperative tile loads,
	// then TS unrolled k-steps; each k-step loads MR+NR registers from smem and
	// performs MR×NR FMAs into private accumulators.
	accBase := l.accCnt
	l.accCnt += MR * NR
	for mr := 0; mr < MR; mr++ {
		for nr := 0; nr < NR; nr++ {
			l.emit(Instr{Kind: InstrAccInit, AccIdx: accBase + mr*NR + nr, WGSLType: wt, Identity: id})
		}
	}

	smA := l.emitExpr(l.item.Ast.Arena().New(uop.OpDefineLocal, idxA.DType(), nil, int64(MR*TS*TS), nil))
	smB := l.emitExpr(l.item.Ast.Arena().New(uop.OpDefineLocal, idxB.DType(), nil, int64(NR*TS*TS), nil))

	l.emit(Instr{Kind: InstrLoopBegin, RangeID: raOuter.ID, RangeSize: K_outer_size})

	// A tile load — MR stripes, each thread loads one element per stripe.
	for mr := 0; mr < MR; mr++ {
		rowA := fmt.Sprintf("(u32(r%d) * %du + %du + lid.y)", mWgID, MR*TS, mr*TS)
		colA := fmt.Sprintf("(u32(r%d) * %du + lid.x)", raOuter.ID, TS)
		condA := fmt.Sprintf("(%s < %du) && (%s < %du)", rowA, M_real, colA, K_real)
		loadA := fmt.Sprintf("data%d[%s * %du + %s * %du]", paramA, rowA, M_stride_A, colA, K_stride_A)
		smIdx := fmt.Sprintf("(%du + lid.y * %du + lid.x)", mr*TS*TS, TS)
		l.emit(Instr{Kind: InstrAssign,
			IndexExpr: fmt.Sprintf("%s[%s]", smA, smIdx),
			Expr:      fmt.Sprintf("select(%s, %s, %s)", zeroA, loadA, condA)})
	}
	// B tile load — NR stripes.
	for nr := 0; nr < NR; nr++ {
		rowB := fmt.Sprintf("(u32(r%d) * %du + lid.y)", raOuter.ID, TS)
		colB := fmt.Sprintf("(u32(r%d) * %du + %du + lid.x)", nWgID, NR*TS, nr*TS)
		condB := fmt.Sprintf("(%s < %du) && (%s < %du)", rowB, K_real, colB, N_real)
		loadB := fmt.Sprintf("data%d[%s * %du + %s * %du]", paramB, rowB, K_stride_B, colB, N_stride_B)
		smIdx := fmt.Sprintf("(%du + lid.y * %du + lid.x)", nr*TS*TS, TS)
		l.emit(Instr{Kind: InstrAssign,
			IndexExpr: fmt.Sprintf("%s[%s]", smB, smIdx),
			Expr:      fmt.Sprintf("select(%s, %s, %s)", zeroB, loadB, condB)})
	}
	l.emit(Instr{Kind: InstrBarrier})

	// Unrolled inner-K loop. Each k step pre-loads MR rA + NR rB registers
	// from smem (giving Naga one chance to CSE per k), then issues MR*NR FMAs.
	regDT := idxA.DType()
	for k := 0; k < TS; k++ {
		for mr := 0; mr < MR; mr++ {
			name := fmt.Sprintf("rA_%d_%d", k, mr)
			expr := fmt.Sprintf("%s[%du + lid.y * %du + %du]", smA, mr*TS*TS, TS, k)
			l.emit(Instr{Kind: InstrLet, Name: name, DType: regDT, Expr: expr})
		}
		for nr := 0; nr < NR; nr++ {
			name := fmt.Sprintf("rB_%d_%d", k, nr)
			expr := fmt.Sprintf("%s[%du + %du * %du + lid.x]", smB, nr*TS*TS, k, TS)
			l.emit(Instr{Kind: InstrLet, Name: name, DType: regDT, Expr: expr})
		}
		for mr := 0; mr < MR; mr++ {
			for nr := 0; nr < NR; nr++ {
				expr := fmt.Sprintf("(rA_%d_%d * rB_%d_%d)", k, mr, k, nr)
				l.emit(Instr{Kind: InstrAccUpdate, AccIdx: accBase + mr*NR + nr, AccOp: accOp, Expr: expr})
			}
		}
	}

	l.emit(Instr{Kind: InstrBarrier})
	l.emit(Instr{Kind: InstrLoopEnd})

	// Hand off state to the final-store expansion in lowerSink.
	l.upcastTileActive = true
	l.upcastMR = MR
	l.upcastNR = NR
	l.upcastTS = TS
	l.upcastOutMSize = M_real
	l.upcastOutNSize = N_real
	l.upcastMWgID = mWgID
	l.upcastMLocID = mLocID
	l.upcastNWgID = nWgID
	l.upcastNLocID = nLocID
	l.upcastAccName = func(mr, nr int) string {
		return fmt.Sprintf("acc%d", accBase+mr*NR+nr)
	}

	// Return a sentinel — the final store layer ignores this and emits MR*NR
	// stores by acc name. Any non-store ancestor of u in the body would be a
	// bug for now (B3 reduces are always the kernel's terminal expression).
	l.exprOf[u.Index()] = fmt.Sprintf("acc%d", accBase)
	return fmt.Sprintf("acc%d", accBase)
}

func (l *lowerer) emitALU(u uop.UOp) string {
	srcs := make([]string, u.NSrc())
	for i := range srcs {
		srcs[i] = l.emitExpr(u.Src(i))
	}
	dt := l.computeDType(u)
	rhs := aluExpr(u.Op(), srcs, dt)
	letName := fmt.Sprintf("t%d", u.Index())
	l.emit(Instr{Kind: InstrLet, NodeIdx: u.Index(), DType: dt, Expr: rhs})
	l.exprOf[u.Index()] = letName
	return letName
}

func (l *lowerer) paramShape(paramIdx int) []int64 {
	if paramIdx >= len(l.item.Bufs) {
		return []int64{1}
	}
	return l.item.Bufs[paramIdx].Shape
}

// paramIsImage reports whether paramIdx's buffer dtype is image-storage
// (DType.IsImage). Drives the codegen fork between scalar `data{i}[idx]`
// indexing and the vec4-packed `data{i}[idx / 4u][idx % 4u]` form. The
// bounds-check is rolled into the predicate so a malformed item never
// indexes out of Bufs (paranoia mirror of paramShape).
func (l *lowerer) paramIsImage(paramIdx int) bool {
	return paramIdx < len(l.item.Bufs) && l.item.Bufs[paramIdx].DType != nil && l.item.Bufs[paramIdx].DType.IsImage()
}

// paramDimFactor returns a strideAcc representing the size of dim `dim` of
// buffer paramIdx. For concrete dims (Shape[dim] != 0) it returns the
// constant int64 size. For symbolic dims (Shape[dim] == 0, the SPEC §10
// sentinel) it consults SymDimAffine / (SymDimMul, SymDimVar) to produce a
// WGSL u32 expression in terms of params_n.n{slot} for the bound DefineVars.
// For `dim` beyond Shape length, returns the identity (constPart=1), matching
// the old emitIndex implicit-size-1 fallback when shape is shorter than nDims.
//
// This is the codegen-time analogue of executor.symElemCount: same encoding
// surface (Shape[i]==0 + SymDimMul/SymDimVar or SymDimAffine), but resolves
// the var name to a params_n slot rather than to a binding value.
//
// Slice 7d closure: replaces the implicit `shape[i+1]` int64 multiplication
// in emitIndex, which silently produced stride=0 when shape[i+1]==0 (sym
// non-outermost) — the latent flagged in the Slice 7b report.
func (l *lowerer) paramDimFactor(paramIdx int, dim int) strideAcc {
	if paramIdx >= len(l.item.Bufs) {
		return newStrideAcc()
	}
	buf := l.item.Bufs[paramIdx]
	if dim < 0 || dim >= len(buf.Shape) {
		return newStrideAcc()
	}
	s := buf.Shape[dim]
	if s != 0 {
		return newStrideAcc().mulConst(s)
	}
	symIdx := 0
	for k := 0; k < dim; k++ {
		if buf.Shape[k] == 0 {
			symIdx++
		}
	}
	if symIdx < len(buf.SymDimAffine) {
		entry := buf.SymDimAffine[symIdx]
		var parts []string
		for _, t := range entry.Terms {
			slot, ok := l.symSlotByName[t.VarName]
			if !ok {
				panic(fmt.Sprintf("codegen: paramDimFactor: SymDimAffine var %q (paramIdx=%d dim=%d) not in symSlot map", t.VarName, paramIdx, dim))
			}
			if t.Mul == 1 {
				parts = append(parts, fmt.Sprintf("params_n.n%d", slot))
			} else {
				parts = append(parts, fmt.Sprintf("params_n.n%d * %du", slot, t.Mul))
			}
		}
		if entry.Offset != 0 {
			parts = append(parts, fmt.Sprintf("%du", entry.Offset))
		}
		var expr string
		switch len(parts) {
		case 0:
			return newStrideAcc()
		case 1:
			expr = parts[0]
		default:
			expr = "(" + strings.Join(parts, " + ") + ")"
		}
		return newStrideAcc().mulSym(expr)
	}
	var name string
	if symIdx < len(buf.SymDimVar) {
		name = buf.SymDimVar[symIdx]
	} else if symIdx < len(l.item.SymVars) {
		name = l.item.SymVars[symIdx]
	}
	if name == "" {
		panic(fmt.Sprintf("codegen: paramDimFactor: no SymDimVar for paramIdx=%d dim=%d symIdx=%d (Shape=%v SymDimVar=%v)", paramIdx, dim, symIdx, buf.Shape, buf.SymDimVar))
	}
	slot, ok := l.symSlotByName[name]
	if !ok {
		panic(fmt.Sprintf("codegen: paramDimFactor: var %q (paramIdx=%d dim=%d) not in symSlot map", name, paramIdx, dim))
	}
	mul := int64(1)
	if symIdx < len(buf.SymDimMul) {
		if m := buf.SymDimMul[symIdx]; m > 0 {
			mul = m
		}
	}
	acc := newStrideAcc().mulSym(fmt.Sprintf("params_n.n%d", slot))
	if mul != 1 {
		acc = acc.mulConst(mul)
	}
	return acc
}

func joinPlus(terms []string) string {
	if len(terms) == 0 {
		return "0"
	}
	s := terms[0]
	for _, t := range terms[1:] {
		s += " + " + t
	}
	return s
}
