package codegen

import (
	"fmt"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// InstrKind identifies the type of a linearized instruction.
type InstrKind int

const (
	// InstrBoundsCheck emits: if (gid_x >= TotalN) { return; }
	InstrBoundsCheck InstrKind = iota
	// InstrGIDVar emits: let r_RangeID: i32 = i32((gid_x / Stride) % RangeSize);
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
	// InstrStore emits: data0[gid_x] = Expr;  (or data0[0] for scalar output)
	InstrStore
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
	Stride int64

	// InstrBoundsCheck, InstrGIDVar: true when the range size is symbolic (read
	// from the params_n storage buffer at runtime, not a compile-time literal).
	Symbolic bool

	// InstrAccInit, InstrAccUpdate
	AccIdx   int
	WGSLType string // for InstrAccInit
	Identity string // for InstrAccInit
	AccOp    uop.Op // for InstrAccUpdate

	// InstrLet
	NodeIdx uint32
	DType   *uop.DType

	// InstrLet, InstrAccUpdate, InstrStore
	Expr string
}

// Lower converts one kernel's SINK AST into a linear instruction sequence.
// Instructions are in emit order; loop nesting depth is tracked by the renderer.
func Lower(item schedule.ExecItem) []Instr {
	l := &lowerer{
		item:   item,
		exprOf: make(map[uint32]string),
	}
	return l.lowerSink()
}

type lowerer struct {
	item   schedule.ExecItem
	instrs []Instr
	exprOf map[uint32]string // arenaIdx → WGSL expression / variable name
	accCnt int               // counter for accumulator variable names
}

func (l *lowerer) emit(ins Instr) { l.instrs = append(l.instrs, ins) }

func (l *lowerer) lowerSink() []Instr {
	sink := l.item.Ast
	if sink.Op() != uop.OpSink {
		panic(fmt.Sprintf("codegen.Lower: expected SINK, got %s", sink.Op()))
	}
	end := sink.Src(0)   // OpEnd
	store := end.Src(0)  // OpStore
	body := store.Src(1) // kernel body expression

	// Collect AxisLoop ranges from END.src[1:] (AxisReduce ranges are emitted
	// lazily inside emitReduce when we encounter a REDUCE node in the body).
	var loopRanges []uop.UOp
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			if r.Arg().(uop.RangeArg).Type == uop.AxisLoop {
				loopRanges = append(loopRanges, r)
			}
		}
	}

	// Total output elements = product of AxisLoop range sizes.
	// TotalN == 0 is the sentinel for "symbolic" (size unknown at compile time).
	totalOut := int64(1)
	hasSymRange := false
	for _, r := range loopRanges {
		ra := r.Arg().(uop.RangeArg)
		if ra.Symbolic {
			totalOut = 0 // 0 = symbolic sentinel; renderer handles this
			hasSymRange = true
		} else {
			if !hasSymRange {
				totalOut *= ra.Size
			}
		}
	}

	// Bounds guard: only thread IDs in [0, totalOut) produce output.
	l.emit(Instr{Kind: InstrBoundsCheck, TotalN: totalOut, Symbolic: hasSymRange})

	// GID decomposition: each AxisLoop range gets a let binding derived from
	// gid_x via row-major stride arithmetic.
	// strides[i] = product(size[i+1], size[i+2], ...)
	// Symbolic dims have Size==0; strides that flow through them are unreliable
	// for multi-dim symbolic kernels (out of scope for SLICE 1).
	strides := make([]int64, len(loopRanges))
	if len(loopRanges) > 0 {
		strides[len(loopRanges)-1] = 1
		for i := len(loopRanges) - 2; i >= 0; i-- {
			ra := loopRanges[i+1].Arg().(uop.RangeArg)
			if !ra.Symbolic {
				strides[i] = strides[i+1] * ra.Size
			}
		}
	}
	for i, r := range loopRanges {
		ra := r.Arg().(uop.RangeArg)
		l.emit(Instr{Kind: InstrGIDVar, RangeID: ra.ID, RangeSize: ra.Size, Stride: strides[i], Symbolic: ra.Symbolic})
		l.exprOf[r.Index()] = fmt.Sprintf("r%d", ra.ID)
	}

	// Emit body expression tree (reduce loops + ALU tree as side effects).
	bodyExpr := l.emitExpr(body)

	// Output store: flat index is gid_x for multi-element output, 0 for scalar.
	l.emit(Instr{Kind: InstrStore, TotalN: totalOut, Symbolic: hasSymRange, Expr: bodyExpr})

	return l.instrs
}

// emitExpr returns the WGSL expression name for u, emitting any necessary
// instructions as side effects. Results are cached in exprOf.
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
		// AxisLoop ranges are pre-registered in lowerSink.
		// AxisReduce ranges are registered by emitReduce before recursing into the
		// element expression. Reaching here means a range was referenced before
		// its loop was opened — this is a bug in the lowerer.
		panic(fmt.Sprintf("codegen: Range(id=%v) not registered before use", u.Arg()))

	case uop.OpParam:
		// PARAM only appears as src[0] of INDEX; shouldn't be emitted standalone.
		e := fmt.Sprintf("data%d", int(u.Arg().(int64)))
		l.exprOf[u.Index()] = e
		return e

	case uop.OpIndex:
		return l.emitIndex(u)

	case uop.OpReduce:
		return l.emitReduce(u)

	default:
		return l.emitALU(u)
	}
}

// emitIndex handles INDEX(PARAM(N), idx_0, idx_1, ...) — a flat buffer read.
func (l *lowerer) emitIndex(u uop.UOp) string {
	paramNode := u.Src(0)
	paramIdx := int(paramNode.Arg().(int64))
	nDims := u.NSrc() - 1

	var flatExpr string
	switch {
	case nDims == 0:
		flatExpr = "0u"
	case nDims == 1:
		flatExpr = l.emitExpr(u.Src(1))
	default:
		// Multi-dim access: compute flat = sum(idx_i * stride_i).
		// Strides come from the buffer's per-dim shape.
		shape := l.paramShape(paramIdx)
		strides := make([]int64, nDims)
		strides[nDims-1] = 1
		for i := nDims - 2; i >= 0; i-- {
			if i+1 < len(shape) {
				strides[i] = strides[i+1] * shape[i+1]
			} else {
				strides[i] = 1
			}
		}
		var terms []string
		for d := 0; d < nDims; d++ {
			dimExpr := l.emitExpr(u.Src(d + 1))
			if strides[d] == 1 {
				terms = append(terms, dimExpr)
			} else {
				terms = append(terms, fmt.Sprintf("(%s * %d)", dimExpr, strides[d]))
			}
		}
		flatExpr = joinPlus(terms)
	}

	// Emit as a let binding so multi-use nodes aren't re-evaluated.
	rhs := fmt.Sprintf("data%d[%s]", paramIdx, flatExpr)
	letName := fmt.Sprintf("t%d", u.Index())
	l.emit(Instr{Kind: InstrLet, NodeIdx: u.Index(), DType: u.DType(), Expr: rhs})
	l.exprOf[u.Index()] = letName
	return letName
}

// emitReduce handles REDUCE(acc_op, elem_expr, *reduce_ranges).
// Emits: accumulator init, loop begins, element expr, accumulator update, loop ends.
// Some reduce ranges may be OpConst(0) when rangeify folded a size-1 dimension to
// a constant; those require no loop — the index is always 0.
func (l *lowerer) emitReduce(u uop.UOp) string {
	accOp := u.Arg().(uop.Op)
	elemNode := u.Src(0)
	accIdx := l.accCnt
	l.accCnt++

	wt := wgslDType(u.DType())
	id := reduceIdentity(accOp, u.DType())
	l.emit(Instr{Kind: InstrAccInit, AccIdx: accIdx, WGSLType: wt, Identity: id})

	// Emit loop begins for each AxisReduce range and register them in exprOf
	// before recursing into the element expression.
	redRanges := make([]uop.UOp, u.NSrc()-1)
	for i := 1; i < u.NSrc(); i++ {
		redRanges[i-1] = u.Src(i)
	}
	hasLoop := make([]bool, len(redRanges))
	for i, r := range redRanges {
		if r.Op() == uop.OpConst {
			// Size-1 reduce dimension: rangeify folds it to OpConst(0) rather than
			// creating an OpRange. No loop needed; register the index as 0.
			l.exprOf[r.Index()] = constLiteral(r)
		} else {
			ra := r.Arg().(uop.RangeArg)
			l.emit(Instr{Kind: InstrLoopBegin, RangeID: ra.ID, RangeSize: ra.Size})
			l.exprOf[r.Index()] = fmt.Sprintf("r%d", ra.ID)
			hasLoop[i] = true
		}
	}

	elemExpr := l.emitExpr(elemNode)
	l.emit(Instr{Kind: InstrAccUpdate, AccIdx: accIdx, AccOp: accOp, Expr: elemExpr})

	for i := range redRanges {
		if hasLoop[i] {
			l.emit(Instr{Kind: InstrLoopEnd})
		}
	}

	name := fmt.Sprintf("acc%d", accIdx)
	l.exprOf[u.Index()] = name
	return name
}

// emitALU handles unary/binary/ternary ALU ops and casts, emitting a let binding.
func (l *lowerer) emitALU(u uop.UOp) string {
	srcs := make([]string, u.NSrc())
	for i := range srcs {
		srcs[i] = l.emitExpr(u.Src(i))
	}

	rhs := aluExpr(u.Op(), srcs, u.DType())

	letName := fmt.Sprintf("t%d", u.Index())
	l.emit(Instr{Kind: InstrLet, NodeIdx: u.Index(), DType: u.DType(), Expr: rhs})
	l.exprOf[u.Index()] = letName
	return letName
}

// paramShape returns the per-dimension shape of the buffer behind PARAM(paramIdx).
// Shape is captured into schedule.Buffer at schedule time, so codegen is a pure
// function of ExecItem and never reaches back into the arena.
func (l *lowerer) paramShape(paramIdx int) []int64 {
	if paramIdx >= len(l.item.Bufs) {
		return []int64{1}
	}
	return l.item.Bufs[paramIdx].Shape
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
