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

	// InstrBoundsCheck, InstrGIDVar, InstrLoopBegin: true when the range size is
	// symbolic (read from the params_n storage buffer at runtime).
	Symbolic bool

	// InstrLoopBegin (symbolic only): which params_n slot holds the loop bound.
	SymParamIdx int

	// InstrAccInit, InstrAccUpdate
	AccIdx   int
	WGSLType string // for InstrAccInit
	Identity string // for InstrAccInit
	AccOp    uop.Op // for InstrAccUpdate

	// InstrLet
	NodeIdx uint32
	DType   *uop.DType

	// InstrBoundsCheck: product of concrete dims trailing the symbolic dim.
	// Used to emit "params_n[0] * N" bounds checks for multi-dim symbolic outputs.
	ConcreteTrailing int64

	// InstrLet, InstrAccUpdate, InstrStore
	Expr      string
	IndexExpr string // for InstrStore
}

// Lower converts one kernel's SINK AST into a linear instruction sequence.
// Instructions are in emit order; loop nesting depth is tracked by the renderer.
func Lower(item schedule.ExecItem) ([]Instr, [3]int, [3]int) {
	l := &lowerer{
		item:   item,
		exprOf: make(map[uint32]string),
	}
	instrs := l.lowerSink()
	return instrs, l.workgroupSize, l.workgroupCount
}

type lowerer struct {
	item           schedule.ExecItem
	instrs         []Instr
	exprOf         map[uint32]string // arenaIdx → WGSL expression / variable name
	accCnt         int               // counter for accumulator variable names
	widenF16       bool              // when true, f16 loads/ops are widened to f32 in reduce body
	workgroupSize  [3]int            // computed from AxisLocal ranges
	workgroupCount [3]int
}

// computeDType returns the effective WGSL dtype for u.
func (l *lowerer) computeDType(u uop.UOp) *uop.DType {
	d := u.DType()
	if d == nil {
		return d
	}
	s := d.Scalar()
	if s == uop.Dtypes.BFloat16 {
		return uop.Dtypes.Float32
	}
	if l.widenF16 && s == uop.Dtypes.Float16 {
		return uop.Dtypes.Float32
	}
	return d
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

	// Collect AxisLoop/Workgroup/Local ranges from END.src[1:].
	var loopRanges []uop.UOp
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			ra := r.Arg().(uop.RangeArg)
			if ra.Type == uop.AxisLoop || ra.Type == uop.AxisWorkgroup || ra.Type == uop.AxisLocal {
				loopRanges = append(loopRanges, r)
			}
		} else if r.Op() == uop.OpConst {
			loopRanges = append(loopRanges, r)
		}
	}

	// Total output elements = product of all loop range sizes.
	totalOut := int64(1)
	hasSymRange := false
	for _, r := range loopRanges {
		if r.Op() == uop.OpConst {
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		if ra.Symbolic {
			totalOut = 0
			hasSymRange = true
		} else if !hasSymRange {
			totalOut *= ra.Size
		}
	}

	// Compute global strides for the final output store flat index.
	globalStrides := make([]int64, len(loopRanges))
	if len(loopRanges) > 0 {
		globalStrides[len(loopRanges)-1] = 1
		for i := len(loopRanges) - 2; i >= 0; i-- {
			rNext := loopRanges[i+1]
			if rNext.Op() == uop.OpConst {
				globalStrides[i] = globalStrides[i+1]
				continue
			}
			ra := rNext.Arg().(uop.RangeArg)
			if !ra.Symbolic {
				globalStrides[i] = globalStrides[i+1] * ra.Size
			}
		}
	}

	// Group ranges into Axes.
	// We assign ranges to Dimensions starting from the INMOST (last) range to X.
	// Matmul: [Row, Col] -> Col is X, Row is Y.
	type rangeGroup struct {
		u   uop.UOp
		ra  uop.RangeArg
		lvl int // 0:Global, 1:Workgroup, 2:Local
		idx int // original index in loopRanges
	}
	dims := [3][]rangeGroup{}
	dimIdx := 0
	for i := len(loopRanges) - 1; i >= 0; i-- {
		targetDim := dimIdx % 3
		if hasSymRange {
			targetDim = 0
		}
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

	// Compute strides and local sizes for each (dimension, level).
	l.workgroupSize = [3]int{1, 1, 1}
	l.workgroupCount = [3]int{1, 1, 1}
	dimSizes := [3]int64{1, 1, 1}

	for d := 0; d < 3; d++ {
		for _, rg := range dims[d] {
			if rg.u.Op() != uop.OpConst {
				dimSizes[d] *= rg.ra.Size
			}
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
			strides := make([]int64, len(levelRanges))
			strides[0] = 1 // inmost range of this level
			for i := 1; i < len(levelRanges); i++ {
				rPrev := levelRanges[i-1].u
				if rPrev.Op() == uop.OpConst {
					strides[i] = strides[i-1]
				} else {
					ra := rPrev.Arg().(uop.RangeArg)
					if !ra.Symbolic {
						strides[i] = strides[i-1] * ra.Size
					}
				}
			}

			for i, rg := range levelRanges {
				if rg.u.Op() == uop.OpConst {
					l.exprOf[rg.u.Index()] = "0u"
					continue
				}
				l.emit(Instr{
					Kind:      InstrGIDVar,
					RangeID:   rg.ra.ID,
					RangeSize: rg.ra.Size,
					Stride:    strides[i],
					Symbolic:  rg.ra.Symbolic,
					Component: d,
					Level:     lvl,
				})
				l.exprOf[rg.u.Index()] = fmt.Sprintf("r%d", rg.ra.ID)
			}

			if lvl == 2 {
				totalLocal := int64(1)
				for _, rg := range levelRanges {
					if rg.u.Op() != uop.OpConst {
						totalLocal *= rg.ra.Size
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

	var indexTerms []string
	for i, r := range loopRanges {
		if r.Op() == uop.OpConst {
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		term := fmt.Sprintf("u32(r%d)", ra.ID)
		if globalStrides[i] > 1 {
			term = fmt.Sprintf("(%s * %du)", term, globalStrides[i])
		}
		indexTerms = append(indexTerms, term)
	}
	indexExpr := joinPlus(indexTerms)
	if len(indexTerms) == 0 {
		indexExpr = "0u"
	}

	concreteTrailing := int64(1)
	seenSym := false
	for _, r := range loopRanges {
		if r.Op() == uop.OpConst {
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		if ra.Symbolic {
			seenSym = true
		} else if seenSym {
			concreteTrailing *= ra.Size
		}
	}

	l.emit(Instr{Kind: InstrBoundsCheck, TotalN: totalOut, Symbolic: hasSymRange, ConcreteTrailing: concreteTrailing})
	bodyExpr := l.emitExpr(body)

	var outBufDType *uop.DType
	if len(l.item.Bufs) > 0 {
		outBufDType = l.item.Bufs[0].DType
	}
	l.emit(Instr{Kind: InstrStore, TotalN: totalOut, Symbolic: hasSymRange, Expr: bodyExpr, IndexExpr: indexExpr, DType: outBufDType})

	return l.instrs
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
	case uop.OpParam:
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
	rhs := fmt.Sprintf("data%d[%s]", paramIdx, flatExpr)
	emitDType := u.DType()
	if emitDType != nil {
		s := emitDType.Scalar()
		if s == uop.Dtypes.BFloat16 {
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

func (l *lowerer) emitReduce(u uop.UOp) string {
	accOp := u.Arg().(uop.Op)
	elemNode := u.Src(0)
	accIdx := l.accCnt
	l.accCnt++
	outDType := u.DType()
	isF16Reduce := outDType != nil && outDType.Scalar() == uop.Dtypes.Float16
	isBF16Reduce := outDType != nil && outDType.Scalar() == uop.Dtypes.BFloat16
	var wt, id string
	if isF16Reduce || isBF16Reduce {
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
			if ra.Symbolic {
				l.emit(Instr{Kind: InstrLoopBegin, RangeID: ra.ID, Symbolic: true, SymParamIdx: ra.SymParamIdx})
			} else {
				l.emit(Instr{Kind: InstrLoopBegin, RangeID: ra.ID, RangeSize: ra.Size})
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
