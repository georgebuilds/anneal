package codegen

import (
	"fmt"
	"math"
	"strings"

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

	// InstrLet, InstrDefineLocal
	NodeIdx uint32
	DType   *uop.DType

	// InstrDefineLocal
	LocalName string
	LocalSize int

	// InstrBoundsCheck: product of concrete dims trailing the symbolic dim.
	// Used to emit "params_n[0] * N" bounds checks for multi-dim symbolic outputs.
	ConcreteTrailing int64

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
func Lower(item schedule.ExecItem) ([]Instr, [3]int, [3]int) {
	l := &lowerer{
		item:   item,
		exprOf: make(map[uint32]string),
	}
	instrs := l.lowerSink()
	return instrs, l.workgroupSize, l.workgroupCount
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
	loopRanges     []uop.UOp
	dims           [3][]rangeGroup

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
	vecTileActive bool
	vecW          int   // vector width (4 for vec4<f32>)
	vecNLocOuterID int  // RangeID of N_loc_outer (lid.x ranges over TS/W)
	vecNReal       int64
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
					ra := p.upcast.Arg().(uop.RangeArg)
					l.upcastByDim[d] = p.upcast
					l.upcastFactorByDim[d] *= ra.Size
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
					ra := p.vec.Arg().(uop.RangeArg)
					l.vectorizeByDim[d] = p.vec
					l.vectorizeFactorByDim[d] = ra.Size
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
	switch {
	case nDims == 0:
		flatExpr = "0u"
	case nDims == 1:
		flatExpr = l.emitExpr(u.Src(1))
	default:
		var shape []int64
		if isLocal {
			// For now, assume 2D local tiles for matmul
			sz := int64(math.Sqrt(float64(paramNode.Arg().(int64))))
			shape = []int64{sz, sz}
		} else {
			shape = l.paramShape(paramIdx)
		}
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
	var rhs string
	if isLocal {
		rhs = fmt.Sprintf("%s[%s]", localName, flatExpr)
	} else {
		rhs = fmt.Sprintf("data%d[%s]", paramIdx, flatExpr)
	}
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
	if tag := u.Tag(); tag != nil {
		if s, ok := tag.(string); ok && strings.HasPrefix(s, "tile:") {
			var ts int
			fmt.Sscanf(s, "tile:%d", &ts)
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
	K_outer_size := raOuter.Size

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
		l.emitExpr(idxB.Src(idxB.NSrc()-1))
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

		l.emit(Instr{Kind: InstrLoopBegin, RangeID: raOuter.ID, RangeSize: raOuter.Size})
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

		l.emit(Instr{Kind: InstrLoopBegin, RangeID: raOuter.ID, RangeSize: raOuter.Size})

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

	l.emit(Instr{Kind: InstrLoopBegin, RangeID: raOuter.ID, RangeSize: raOuter.Size})

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
