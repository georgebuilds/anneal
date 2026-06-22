package schedule

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// extra_coverage_test.go - white-box (package schedule) coverage for the
// rangeify index-propagation branches that the existing end-to-end suite leaves
// dark: OpGather / OpScatterAdd dissolution, the symbolic-arg shapeOfNode
// branches (PadSintArg / ShrinkSintArg / ShapeSintArg buffers / OpBitcast), and
// the auto-Contiguous budget pass's deeper cut tiers. Each test pins real
// expected output (value oracle or structural invariant), not assertion-free
// line execution. All deterministic; no GPU, no network.

// ── A self-contained kernel interpreter that understands OpGatherIdx ───────────
//
// The shared schedule_test.go interpreter (kernelEval) cannot evaluate gather
// bodies because OpGatherIdx is an indirect scalar coordinate. This minimal
// evaluator handles the small op set produced by gather/scatter dissolution:
// Index, GatherIdx, Where, Reduce(+/max), CmpEq/CmpLt, And, Add/Sub/Mul, Const,
// Range, Cast.

type gatherEval struct {
	rangeVal map[uint32]int64
	fdata    map[uint32][]float32 // float buffers keyed by BUFFER index
	idata    map[uint32][]int32   // int buffers keyed by BUFFER index
	shape    map[uint32][]int64
}

// evalI returns the integer value of an index/int-typed expression.
func (e *gatherEval) evalI(u uop.UOp) int64 {
	switch u.Op() {
	case uop.OpConst:
		switch v := u.Arg().(type) {
		case int64:
			return v
		case float64:
			return int64(v)
		}
		panic("gatherEval.evalI: bad const")
	case uop.OpRange:
		return e.rangeVal[u.Index()]
	case uop.OpAdd:
		return e.evalI(u.Src(0)) + e.evalI(u.Src(1))
	case uop.OpSub:
		return e.evalI(u.Src(0)) - e.evalI(u.Src(1))
	case uop.OpMul:
		return e.evalI(u.Src(0)) * e.evalI(u.Src(1))
	case uop.OpCast:
		return e.evalI(u.Src(0))
	case uop.OpGatherIdx:
		// The indirect scalar: its src[0] is an INDEX over an integer buffer.
		return e.evalI(u.Src(0))
	case uop.OpIndex:
		buf := u.Src(0)
		sh := e.shape[buf.Index()]
		flat := 0
		for d := 0; d < len(sh); d++ {
			flat = flat*int(sh[d]) + int(e.evalI(u.Src(d+1)))
		}
		if dat, ok := e.idata[buf.Index()]; ok {
			return int64(dat[flat])
		}
		return int64(e.fdata[buf.Index()][flat])
	}
	panic("gatherEval.evalI: unhandled op " + u.Op().String())
}

// evalF returns the float value of a value-typed expression.
func (e *gatherEval) evalF(u uop.UOp) float32 {
	switch u.Op() {
	case uop.OpConst:
		switch v := u.Arg().(type) {
		case float64:
			return float32(v)
		case int64:
			return float32(v)
		case bool:
			if v {
				return 1
			}
			return 0
		}
		panic("gatherEval.evalF: bad const")
	case uop.OpIndex:
		buf := u.Src(0)
		sh := e.shape[buf.Index()]
		flat := 0
		for d := 0; d < len(sh); d++ {
			flat = flat*int(sh[d]) + int(e.evalI(u.Src(d+1)))
		}
		return e.fdata[buf.Index()][flat]
	case uop.OpWhere:
		if e.evalBool(u.Src(0)) {
			return e.evalF(u.Src(1))
		}
		return e.evalF(u.Src(2))
	case uop.OpAdd:
		return e.evalF(u.Src(0)) + e.evalF(u.Src(1))
	case uop.OpMul:
		return e.evalF(u.Src(0)) * e.evalF(u.Src(1))
	case uop.OpReduce:
		op := u.Arg().(uop.Op)
		var acc float32
		switch op {
		case uop.OpAdd:
			acc = 0
		case uop.OpMax:
			acc = float32(math.Inf(-1))
		default:
			panic("gatherEval: unhandled reduce op")
		}
		ranges := make([]uop.UOp, u.NSrc()-1)
		for i := 1; i < u.NSrc(); i++ {
			ranges[i-1] = u.Src(i)
		}
		e.enum(ranges, 0, func() {
			v := e.evalF(u.Src(0))
			switch op {
			case uop.OpAdd:
				acc += v
			case uop.OpMax:
				if v > acc {
					acc = v
				}
			}
		})
		return acc
	}
	panic("gatherEval.evalF: unhandled op " + u.Op().String())
}

func (e *gatherEval) evalBool(u uop.UOp) bool {
	switch u.Op() {
	case uop.OpCmpEq:
		return e.evalI(u.Src(0)) == e.evalI(u.Src(1))
	case uop.OpCmpLt:
		return e.evalI(u.Src(0)) < e.evalI(u.Src(1))
	case uop.OpAnd:
		return e.evalBool(u.Src(0)) && e.evalBool(u.Src(1))
	case uop.OpConst:
		return u.Arg().(bool)
	}
	panic("gatherEval.evalBool: unhandled op " + u.Op().String())
}

func (e *gatherEval) enum(ranges []uop.UOp, i int, fn func()) {
	if i == len(ranges) {
		fn()
		return
	}
	r := ranges[i]
	for v := int64(0); v < uop.RangeSize(r); v++ {
		e.rangeVal[r.Index()] = v
		e.enum(ranges, i+1, fn)
	}
}

// evalKernelGather runs one AFTER node (gather/scatter kernel) and returns the
// flat float output plus its shape.
func (e *gatherEval) evalKernelGather(after uop.UOp) ([]float32, []int64) {
	end := after.Src(1)
	store := end.Src(0)
	body := store.Src(1)

	var outRanges []uop.UOp
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			if ra, ok := r.Arg().(uop.RangeArg); ok && ra.Type == uop.AxisLoop {
				outRanges = append(outRanges, r)
			}
		}
	}
	outShape := make([]int64, len(outRanges))
	n := 1
	for i, r := range outRanges {
		outShape[i] = uop.RangeSize(r)
		n *= int(outShape[i])
	}
	out := make([]float32, n)
	var walk func(dim, flat int)
	walk = func(dim, flat int) {
		if dim == len(outRanges) {
			out[flat] = e.evalF(body)
			return
		}
		r := outRanges[dim]
		for v := int64(0); v < uop.RangeSize(r); v++ {
			e.rangeVal[r.Index()] = v
			walk(dim+1, flat*int(uop.RangeSize(r))+int(v))
		}
	}
	walk(0, 0)
	return out, outShape
}

// walkGraph visits every node reachable from root exactly once.
func walkGraph(root uop.UOp, visit func(uop.UOp)) {
	seen := map[uint32]bool{}
	var rec func(u uop.UOp)
	rec = func(u uop.UOp) {
		if !u.Valid() || seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		visit(u)
		for i := 0; i < u.NSrc(); i++ {
			rec(u.Src(i))
		}
	}
	rec(root)
}

// opsInGraph reports (sawGatherIdx, sawGather) over the live graph rooted at root.
func opsInGraph(root uop.UOp) (sawGatherIdx, sawGather bool) {
	walkGraph(root, func(u uop.UOp) {
		switch u.Op() {
		case uop.OpGatherIdx:
			sawGatherIdx = true
		case uop.OpGather:
			sawGather = true
		}
	})
	return
}

// opSurvivesInGraph reports whether op appears anywhere in the live graph.
func opSurvivesInGraph(root uop.UOp, op uop.Op) bool {
	found := false
	walkGraph(root, func(u uop.UOp) {
		if u.Op() == op {
			found = true
		}
	})
	return found
}

func findKernel(root uop.UOp) uop.UOp {
	a := root.Arena()
	for i := 0; i < a.Len(); i++ {
		u := a.At(uint32(i))
		if u.Op() == uop.OpAfter && u.NSrc() == 2 && u.Src(1).Op() == uop.OpEnd {
			return u
		}
	}
	return uop.UOp{}
}

// ── OpGather dissolution: shape + value oracle ───────────────────────────────

// TestGatherDissolvesAndValues builds OpGather(data[3,2], idx[4]) along dim 0
// and verifies the scheduled kernel computes torch-gather semantics:
// out[i, j] = data[idx[i], j].
func TestGatherDissolvesAndValues(t *testing.T) {
	a := uop.NewArena(512)
	const D0, D1, IDX = int64(3), int64(2), int64(4)

	data := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{D0, D1}, nil)
	idx := a.New(uop.OpBuffer, uop.Dtypes.Int32, nil, []int64{IDX}, nil)
	gather := a.New(uop.OpGather, uop.Dtypes.Float32, []uop.UOp{data, idx}, int64(0), nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{gather}, nil, nil)

	result := GetKernelGraph(sink, "cpu")

	// Structural: gather must dissolve (no OpGather survives in the scheduled
	// kernel body) and the body must carry an OpGatherIdx indirect coordinate.
	// Walk only the live kernel graph (not the arena, which keeps all interned
	// nodes including the pre-rewrite originals).
	sawGatherIdx, sawSurvivingGather := opsInGraph(result)
	if sawSurvivingGather {
		t.Errorf("OpGather survived scheduling; index propagation missing")
	}
	if !sawGatherIdx {
		t.Errorf("no OpGatherIdx produced; gather not dissolved into indirect index")
	}

	// Value oracle.
	dataVals := []float32{10, 11, 20, 21, 30, 31} // [3,2] row-major
	idxVals := []int32{2, 0, 1, 2}
	ev := &gatherEval{
		rangeVal: map[uint32]int64{},
		fdata:    map[uint32][]float32{data.Index(): dataVals},
		idata:    map[uint32][]int32{idx.Index(): idxVals},
		shape:    map[uint32][]int64{data.Index(): {D0, D1}, idx.Index(): {IDX}},
	}
	kernel := findKernel(result)
	if !kernel.Valid() {
		t.Fatalf("no kernel found")
	}
	got, gotShape := ev.evalKernelGather(kernel)
	if len(gotShape) != 2 || gotShape[0] != IDX || gotShape[1] != D1 {
		t.Fatalf("out shape = %v, want [%d %d]", gotShape, IDX, D1)
	}
	// want[i,j] = data[idx[i], j]
	want := make([]float32, IDX*D1)
	for i := int64(0); i < IDX; i++ {
		for j := int64(0); j < D1; j++ {
			want[i*D1+j] = dataVals[int64(idxVals[i])*D1+j]
		}
	}
	for k := range want {
		if got[k] != want[k] {
			t.Errorf("out[%d] = %v, want %v (full got=%v want=%v)", k, got[k], want[k], got, want)
		}
	}
}

// ── OpScatterAdd dissolution: shape + value oracle ───────────────────────────

// TestScatterAddDissolvesAndValues builds OpScatterAdd(zeros[V,D], grad[B,D],
// sortedIdx[B], perm[B]) and verifies the scheduled kernel computes the
// segment-sum embedding-backward semantics:
//
//	out[v, t] = sum_{b: sortedIdx[b]==v} grad[perm[b], t]
func TestScatterAddDissolvesAndValues(t *testing.T) {
	a := uop.NewArena(512)
	const V, D, B = int64(3), int64(2), int64(4)

	zeros := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{V, D}, nil)
	grad := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{B, D}, nil)
	sorted := a.New(uop.OpBuffer, uop.Dtypes.Int32, nil, []int64{B}, nil)
	perm := a.New(uop.OpBuffer, uop.Dtypes.Int32, nil, []int64{B}, nil)
	scatter := a.New(uop.OpScatterAdd, uop.Dtypes.Float32,
		[]uop.UOp{zeros, grad, sorted, perm}, int64(0), nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{scatter}, nil, nil)

	result := GetKernelGraph(sink, "cpu")

	if opSurvivesInGraph(result, uop.OpScatterAdd) {
		t.Errorf("OpScatterAdd survived scheduling; index propagation missing")
	}

	// Value oracle.
	gradVals := []float32{1, 2, 3, 4, 5, 6, 7, 8} // [4,2]
	sortedVals := []int32{0, 0, 1, 2}             // segment ids per b (sorted)
	permVals := []int32{0, 1, 2, 3}               // identity permutation
	ev := &gatherEval{
		rangeVal: map[uint32]int64{},
		fdata:    map[uint32][]float32{grad.Index(): gradVals},
		idata:    map[uint32][]int32{sorted.Index(): sortedVals, perm.Index(): permVals},
		shape: map[uint32][]int64{
			grad.Index(): {B, D}, sorted.Index(): {B}, perm.Index(): {B},
		},
	}
	kernel := findKernel(result)
	if !kernel.Valid() {
		t.Fatalf("no kernel found")
	}
	got, gotShape := ev.evalKernelGather(kernel)
	if len(gotShape) != 2 || gotShape[0] != V || gotShape[1] != D {
		t.Fatalf("out shape = %v, want [%d %d]", gotShape, V, D)
	}
	want := make([]float32, V*D)
	for b := int64(0); b < B; b++ {
		v := int64(sortedVals[b])
		p := int64(permVals[b])
		for tt := int64(0); tt < D; tt++ {
			want[v*D+tt] += gradVals[p*D+tt]
		}
	}
	for k := range want {
		if got[k] != want[k] {
			t.Errorf("out[%d] = %v, want %v (got=%v want=%v)", k, got[k], want[k], got, want)
		}
	}
}

// TestScatterAddNonZeroDimPanics pins the Slice-D scope guard.
func TestScatterAddNonZeroDimPanics(t *testing.T) {
	a := uop.NewArena(256)
	zeros := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{3, 2}, nil)
	grad := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4, 2}, nil)
	sorted := a.New(uop.OpBuffer, uop.Dtypes.Int32, nil, []int64{4}, nil)
	perm := a.New(uop.OpBuffer, uop.Dtypes.Int32, nil, []int64{4}, nil)
	scatter := a.New(uop.OpScatterAdd, uop.Dtypes.Float32,
		[]uop.UOp{zeros, grad, sorted, perm}, int64(1), nil) // dim=1 unsupported
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{scatter}, nil, nil)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for scatter-add dim != 0")
		}
	}()
	_ = GetKernelGraph(sink, "cpu")
}

// ── shapeOfNode: symbolic-arg branches ───────────────────────────────────────

// TestShapeOfNodeGatherAndGatherIdx pins the OpGather (data dim replaced by
// idx shape) and OpGatherIdx (scalar) shape rules.
func TestShapeOfNodeGatherAndGatherIdx(t *testing.T) {
	a := uop.NewArena(256)
	data := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{3, 5}, nil)
	idx := a.New(uop.OpBuffer, uop.Dtypes.Int32, nil, []int64{4}, nil)

	gather := a.New(uop.OpGather, uop.Dtypes.Float32, []uop.UOp{data, idx}, int64(0), nil)
	wantConcrete(t, shapeOf(gather), 4, 5) // dim0 (=3) replaced by idx shape (=4)

	// Gather along dim 1: [3, *idxShape].
	gather1 := a.New(uop.OpGather, uop.Dtypes.Float32, []uop.UOp{data, idx}, int64(1), nil)
	wantConcrete(t, shapeOf(gather1), 3, 4)

	// OpGatherIdx is a scalar (rank 0).
	gIdx := a.New(uop.OpGatherIdx, uop.Dtypes.Index, []uop.UOp{idx}, nil, nil)
	if sh := shapeOf(gIdx); len(sh) != 0 {
		t.Errorf("GatherIdx rank = %d, want 0", len(sh))
	}
}

// TestShapeOfNodeBitcast pins the OpBitcast shape-passthrough branch.
func TestShapeOfNodeBitcast(t *testing.T) {
	a := uop.NewArena(128)
	buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 3}, nil)
	bc := a.New(uop.OpBitcast, uop.Dtypes.Int32, []uop.UOp{buf}, nil, nil)
	wantConcrete(t, shapeOf(bc), 2, 3)
}

// TestShapeOfNodeSymbolicArgs pins the symbolic-arg (PadSintArg / ShrinkSintArg
// / ShapeSintArg) branches of shapeOfNode.
func TestShapeOfNodeSymbolicArgs(t *testing.T) {
	a := uop.NewArena(256)
	nNode := a.DefineVar("n", 1, 16)

	// Symbolic 1-D buffer via ShapeSintArg.
	symBuf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil,
		uop.ShapeSintArg{{Sym: true, VarName: "n", Mul: 1}}, nil)
	sh := shapeOf(symBuf)
	if len(sh) != 1 {
		t.Fatalf("symbuf rank = %d, want 1", len(sh))
	}
	if _, ok := sh[0].ConstValue(); ok {
		t.Errorf("symbuf dim 0 should be symbolic")
	}

	// Concrete buffer to feed Pad/Shrink with symbolic args (need a [n] src).
	src := a.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{nNode}, nil, nil)

	// OpPad with PadSintArg: lo/hi are concrete ints over a symbolic src dim.
	pad := a.New(uop.OpPad, uop.Dtypes.Float32, []uop.UOp{src},
		uop.PadSintArg{{{V: 1}, {V: 2}}}, nil)
	psh := shapeOf(pad)
	if len(psh) != 1 {
		t.Fatalf("pad rank = %d, want 1", len(psh))
	}
	if _, ok := psh[0].ConstValue(); ok {
		t.Errorf("pad dim should stay symbolic (n + 3)")
	}

	// OpShrink with ShrinkSintArg over the symbolic src dim → still symbolic
	// (hi=n, lo=0) so the result dim is n.
	shr := a.New(uop.OpShrink, uop.Dtypes.Float32, []uop.UOp{src},
		uop.ShrinkSintArg{{{V: 0}, {Sym: true, VarName: "n", Mul: 1}}}, nil)
	ssh := shapeOf(shr)
	if len(ssh) != 1 {
		t.Fatalf("shrink rank = %d, want 1", len(ssh))
	}
}

// ── Budget pass: deeper cut tiers ────────────────────────────────────────────

// TestBudgetDeepCutChain stresses chooseCut beyond the tier-1 happy path: a
// chain of reductions plus many leaves where a single cut cannot fit both
// halves under cap, forcing tier-2 (max-shed) selection across iterations.
// The oracle is the invariant: every resulting kernel stays within the cap.
func TestBudgetDeepCutChain(t *testing.T) {
	a := uop.NewArena(2048)
	const D = int64(4)
	// 16 leaf buffers added together: y = b0 + b1 + ... + b15.
	leaves := make([]uop.UOp, 16)
	for i := range leaves {
		leaves[i] = a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{D}, nil)
	}
	acc := leaves[0]
	for i := 1; i < len(leaves); i++ {
		acc = a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{acc, leaves[i]}, nil, nil)
	}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{acc}, nil, nil)

	result := GetKernelGraph(sink, "cpu")

	// Walk every kernel and assert binding count <= MaxBuffersPerKernel.
	for i := 0; i < result.Arena().Len(); i++ {
		u := result.Arena().At(uint32(i))
		if u.Op() != uop.OpAfter || u.NSrc() != 2 || u.Src(1).Op() != uop.OpEnd {
			continue
		}
		bufs := map[uint32]bool{u.Src(0).Index(): true}
		seen := map[uint32]bool{}
		var walk func(n uop.UOp)
		walk = func(n uop.UOp) {
			if seen[n.Index()] {
				return
			}
			seen[n.Index()] = true
			if n.Op() == uop.OpBuffer {
				bufs[n.Index()] = true
				return
			}
			for k := 0; k < n.NSrc(); k++ {
				walk(n.Src(k))
			}
		}
		walk(u.Src(1))
		if len(bufs) > MaxBuffersPerKernel {
			t.Errorf("kernel %d has %d storage buffers (cap %d)", i, len(bufs), MaxBuffersPerKernel)
		}
	}
}

// TestIsCutCandidate pins isCutCandidate's accept/reject classification across
// the op categories it gates on.
func TestIsCutCandidate(t *testing.T) {
	a := uop.NewArena(256)
	buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)

	cases := []struct {
		name string
		u    uop.UOp
		want bool
	}{
		{"buffer", buf, false},
		{"add", a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{buf, buf}, nil, nil), true},
		{"where", a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{buf, buf, buf}, nil, nil), true},
		{"reshape", a.New(uop.OpReshape, uop.Dtypes.Float32, []uop.UOp{buf}, []int64{4}, nil), true},
		{"cast", a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{buf}, nil, nil), true},
		{"contiguous", a.New(uop.OpContiguous, uop.Dtypes.Float32, []uop.UOp{buf}, nil, nil), false},
		{"reduceaxis", a.New(uop.OpReduceAxis, uop.Dtypes.Float32, []uop.UOp{buf},
			uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0}}, nil), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isCutCandidate(tc.u); got != tc.want {
				t.Errorf("isCutCandidate(%s) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// ── BoundExpr.Eval panic paths ───────────────────────────────────────────────

func mustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

// TestBoundExprEvalPanics pins the malformed-tree panic paths: division/modulo
// by zero and an unknown op code.
func TestBoundExprEvalPanics(t *testing.T) {
	binding := map[string]int64{}

	mustPanic(t, "idiv-by-zero", func() {
		_, _ = bBin(BoundOpIDiv, bConst(4), bConst(0)).Eval(binding)
	})
	mustPanic(t, "mod-by-zero", func() {
		_, _ = bBin(BoundOpMod, bConst(4), bConst(0)).Eval(binding)
	})
	mustPanic(t, "unknown-op", func() {
		_, _ = BoundExpr{Op: BoundExprOp(255)}.Eval(binding)
	})
	mustPanic(t, "binary-without-2-children", func() {
		_, _ = BoundExpr{Op: BoundOpAdd, Children: []BoundExpr{bConst(1)}}.Eval(binding)
	})
}

// ── indexExprNode arg-type panics ────────────────────────────────────────────

// TestIndexExprNodeArgPanics pins the "unexpected arg type" guards in the
// movement-op branches: feeding a node a malformed arg must panic rather than
// silently miscompile. These guards protect against IR corruption.
func TestIndexExprNodeArgPanics(t *testing.T) {
	mk := func(op uop.Op, arg any) func() {
		return func() {
			a := uop.NewArena(64)
			buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)
			node := a.New(op, uop.Dtypes.Float32, []uop.UOp{buf}, arg, nil)
			rc := newRangeCtx(a)
			cache := map[uint32][]shape.Sint{
				buf.Index():  {shape.Const(4)},
				node.Index(): {shape.Const(4)},
			}
			idx := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
			indexExprNode(a, node, []uop.UOp{idx}, cache, rc, 0)
		}
	}

	// Permute / Flip with a non-[]int64 arg, Reshape with a bogus arg.
	mustPanic(t, "permute-bad-arg", mk(uop.OpPermute, "nope"))
	mustPanic(t, "flip-bad-arg", mk(uop.OpFlip, "nope"))
	mustPanic(t, "reshape-bad-arg", mk(uop.OpReshape, "nope"))
}
