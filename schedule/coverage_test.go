package schedule

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// coverage_test.go - white-box (package schedule) direct unit coverage for the
// pure scheduling/index/symbolic helpers that production exercises only through
// full end-to-end graphs. These pin the helpers' semantics so a regression
// cannot ship silently, and they raise statement coverage of the cheap-to-test
// leaves (BoundExpr.Eval, isZeroConst, shapeSintArgToSints, newSymRange,
// candidateSize, the cut comparators, shapeOfNode op branches, identityConst,
// dimToUOp, DebugBufRangesFlush). All deterministic; no GPU, no network.

// ── BoundExpr.Eval / boundEvalErr.Error ──────────────────────────────────────

func bConst(v int64) BoundExpr   { return BoundExpr{Op: BoundOpConst, Const: v} }
func bVar(name string) BoundExpr { return BoundExpr{Op: BoundOpVar, VarName: name} }
func bBin(op BoundExprOp, a, b BoundExpr) BoundExpr {
	return BoundExpr{Op: op, Children: []BoundExpr{a, b}}
}

func TestBoundExprEval(t *testing.T) {
	binding := map[string]int64{"n": 7, "m": 3}
	cases := []struct {
		name string
		expr BoundExpr
		want int64
	}{
		{"const", bConst(5), 5},
		{"var", bVar("n"), 7},
		{"add", bBin(BoundOpAdd, bVar("n"), bConst(2)), 9},
		{"sub", bBin(BoundOpSub, bVar("n"), bVar("m")), 4},
		{"mul", bBin(BoundOpMul, bVar("n"), bConst(3)), 21},
		{"idiv", bBin(BoundOpIDiv, bVar("n"), bVar("m")), 2},
		{"mod", bBin(BoundOpMod, bVar("n"), bVar("m")), 1},
		// (n + L - 1) / L workgroup-count form, L=4, n=7 → 2.
		{"wgcount", bBin(BoundOpIDiv,
			bBin(BoundOpSub, bBin(BoundOpAdd, bVar("n"), bConst(4)), bConst(1)),
			bConst(4)), 2},
		{"nested mul-add", bBin(BoundOpAdd, bBin(BoundOpMul, bVar("n"), bVar("m")), bConst(1)), 22},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.expr.Eval(binding)
			if err != nil {
				t.Fatalf("Eval(%s) err = %v", tc.name, err)
			}
			if got != tc.want {
				t.Errorf("Eval(%s) = %d, want %d", tc.name, got, tc.want)
			}
		})
	}
}

func TestBoundExprEvalMissingVar(t *testing.T) {
	binding := map[string]int64{"n": 7}
	// Direct missing var.
	if _, err := bVar("missing").Eval(binding); err == nil {
		t.Fatalf("Eval(missing var) returned nil error")
	} else if err.Error() == "" {
		t.Fatalf("boundEvalErr.Error() returned empty string")
	}

	// Missing var propagated through both binary operands.
	left := bBin(BoundOpAdd, bVar("ghost"), bConst(1))
	if _, err := left.Eval(binding); err == nil {
		t.Errorf("Eval with missing left operand returned nil error")
	}
	right := bBin(BoundOpAdd, bConst(1), bVar("ghost"))
	if _, err := right.Eval(binding); err == nil {
		t.Errorf("Eval with missing right operand returned nil error")
	}
}

func TestBoundEvalErrMessage(t *testing.T) {
	e := &boundEvalErr{kind: "missing binding for var", name: "n"}
	want := "schedule.BoundExpr.Eval: missing binding for var n"
	if got := e.Error(); got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

// ── isZeroConst ──────────────────────────────────────────────────────────────

func TestIsZeroConst(t *testing.T) {
	a := uop.NewArena(64)
	zero := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	one := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	floatZero := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0), nil)
	notConst := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{zero, one}, nil, nil)

	cases := []struct {
		name string
		u    uop.UOp
		want bool
	}{
		{"int64 zero", zero, true},
		{"int64 one", one, false},
		{"float zero (not int64)", floatZero, false},
		{"non-const op", notConst, false},
		{"invalid uop", uop.UOp{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isZeroConst(tc.u); got != tc.want {
				t.Errorf("isZeroConst(%s) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// ── shapeSintArgToSints ──────────────────────────────────────────────────────

func TestShapeSintArgToSints(t *testing.T) {
	a := uop.NewArena(64)
	// Pre-create the DefineVar so RebuildSymBound interns to a known node.
	nNode := a.DefineVar("n", 1, 33)

	arg := uop.ShapeSintArg{
		{V: 4, Sym: false},                // concrete dim
		{Sym: true, VarName: "n", Mul: 1}, // bare symbolic dim
		{Sym: true, VarName: "n", Mul: 3}, // symbolic * 3
	}
	got := shapeSintArgToSints(a, arg)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	// Dim 0: concrete 4.
	if v, ok := got[0].ConstValue(); !ok || v != 4 {
		t.Errorf("got[0] concrete = (%d,%v), want 4,true", v, ok)
	}
	// Dim 1: bare symbolic - must be a SymInt aliased to nNode.
	sym1, ok := got[1].(shape.SymInt)
	if !ok {
		t.Fatalf("got[1] is %T, want shape.SymInt", got[1])
	}
	if sym1.Node != nNode {
		t.Errorf("got[1].Node not interned to original DefineVar")
	}
	// Dim 2: symbolic * 3 - symbolic and not equal to the bare var node.
	if _, ok := got[2].(shape.SymInt); !ok {
		t.Fatalf("got[2] is %T, want shape.SymInt", got[2])
	}
	if _, isConst := got[2].ConstValue(); isConst {
		t.Errorf("got[2] reports concrete, want symbolic")
	}
}

// ── rangeCtx.newSymRange / newRangeCtx / freshRanges ─────────────────────────

func TestNewSymRange(t *testing.T) {
	a := uop.NewArena(64)
	rc := newRangeCtx(a)
	rc.startKernel()

	bound := a.DefineVar("n", 1, 33)
	r := rc.newSymRange(bound, uop.AxisLoop)

	if r.Op() != uop.OpRange {
		t.Fatalf("newSymRange produced %v, want OpRange", r.Op())
	}
	if r.NSrc() != 1 || r.Src(0) != bound {
		t.Errorf("newSymRange src[0] should be the bound DefineVar")
	}
	ra, ok := r.Arg().(uop.RangeArg)
	if !ok {
		t.Fatalf("Range arg is %T, want uop.RangeArg", r.Arg())
	}
	if ra.Type != uop.AxisLoop {
		t.Errorf("Range arg type = %v, want AxisLoop", ra.Type)
	}
	// ID must increment with each new range; the sym range is accumulated.
	if len(rc.kernelRanges) != 1 || rc.kernelRanges[0] != r {
		t.Errorf("newSymRange did not accumulate into kernelRanges")
	}

	r2 := rc.newSymRange(bound, uop.AxisReduce)
	ra2 := r2.Arg().(uop.RangeArg)
	if ra2.ID == ra.ID {
		t.Errorf("two ranges share ID %d", ra.ID)
	}
}

func TestFreshRangesMixed(t *testing.T) {
	a := uop.NewArena(64)
	rc := newRangeCtx(a)
	rc.startKernel()

	sh := []shape.Sint{
		shape.Const(4), // concrete > 1 → OpRange
		shape.Const(1), // size-1 → OpConst(0)
		shape.SymInt{Node: a.DefineVar("n", 1, 9)}, // symbolic → newSymRange
	}
	ranges := rc.freshRanges(sh, uop.AxisLoop)
	if len(ranges) != 3 {
		t.Fatalf("len = %d, want 3", len(ranges))
	}
	if ranges[0].Op() != uop.OpRange {
		t.Errorf("ranges[0] = %v, want OpRange", ranges[0].Op())
	}
	if ranges[1].Op() != uop.OpConst {
		t.Errorf("ranges[1] = %v, want OpConst (size-1 collapse)", ranges[1].Op())
	}
	if ranges[2].Op() != uop.OpRange {
		t.Errorf("ranges[2] = %v, want OpRange (symbolic)", ranges[2].Op())
	}
}

// ── shapeOfNode op branches ──────────────────────────────────────────────────

// shapeOf is a small helper: topo-walk u and return its computed shape.
func shapeOf(u uop.UOp) []shape.Sint {
	cache := make(map[uint32][]shape.Sint)
	for _, n := range uop.TopoSort(u) {
		shapeOfNode(n, cache)
	}
	return cache[u.Index()]
}

func wantConcrete(t *testing.T, sh []shape.Sint, want ...int64) {
	t.Helper()
	if len(sh) != len(want) {
		t.Fatalf("rank %d, want %d (%v)", len(sh), len(want), sh)
	}
	for i, w := range want {
		v, ok := sh[i].ConstValue()
		if !ok || v != w {
			t.Errorf("dim %d = (%d,%v), want %d", i, v, ok, w)
		}
	}
}

func TestShapeOfNodeBranches(t *testing.T) {
	a := uop.NewArena(256)

	// OpConst → scalar.
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if sh := shapeOf(c); len(sh) != 0 {
		t.Errorf("const shape rank = %d, want 0", len(sh))
	}

	// OpBuffer with []int64 arg → that shape.
	buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 3}, nil)
	wantConcrete(t, shapeOf(buf), 2, 3)

	// OpBuffer with int64 arg → 1D.
	buf1 := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, int64(5), nil)
	wantConcrete(t, shapeOf(buf1), 5)

	// OpReshape with []int64 arg.
	rsh := a.New(uop.OpReshape, uop.Dtypes.Float32, []uop.UOp{buf}, []int64{6}, nil)
	wantConcrete(t, shapeOf(rsh), 6)

	// OpExpand with []int64 arg.
	exp := a.New(uop.OpExpand, uop.Dtypes.Float32, []uop.UOp{buf}, []int64{2, 3}, nil)
	wantConcrete(t, shapeOf(exp), 2, 3)

	// OpPermute reorders src shape.
	perm := a.New(uop.OpPermute, uop.Dtypes.Float32, []uop.UOp{buf}, []int64{1, 0}, nil)
	wantConcrete(t, shapeOf(perm), 3, 2)

	// OpPad ([][2]int64) adds lo+hi per axis.
	pad := a.New(uop.OpPad, uop.Dtypes.Float32, []uop.UOp{buf}, [][2]int64{{1, 1}, {0, 2}}, nil)
	wantConcrete(t, shapeOf(pad), 4, 5)

	// OpShrink ([][2]int64) → hi-lo per axis.
	shr := a.New(uop.OpShrink, uop.Dtypes.Float32, []uop.UOp{buf}, [][2]int64{{0, 1}, {1, 3}}, nil)
	wantConcrete(t, shapeOf(shr), 1, 2)

	// OpFlip preserves shape.
	flip := a.New(uop.OpFlip, uop.Dtypes.Float32, []uop.UOp{buf}, []int64{0}, nil)
	wantConcrete(t, shapeOf(flip), 2, 3)

	// OpCast preserves shape.
	cast := a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{buf}, nil, nil)
	wantConcrete(t, shapeOf(cast), 2, 3)

	// OpReduceAxis drops reduced axes.
	red := a.New(uop.OpReduceAxis, uop.Dtypes.Float32, []uop.UOp{buf},
		uop.ReduceArg{Op: uop.OpAdd, Axes: []int{1}}, nil)
	wantConcrete(t, shapeOf(red), 2)

	// OpReduceAxis over all axes → scalar.
	redAll := a.New(uop.OpReduceAxis, uop.Dtypes.Float32, []uop.UOp{buf},
		uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0, 1}}, nil)
	if sh := shapeOf(redAll); len(sh) != 0 {
		t.Errorf("reduce-all rank = %d, want 0", len(sh))
	}

	// ALU default branch → same shape as src[0].
	alu := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{buf, buf}, nil, nil)
	wantConcrete(t, shapeOf(alu), 2, 3)
}

func TestShapeOfNodeSymbolicBuffer(t *testing.T) {
	a := uop.NewArena(128)
	nNode := a.DefineVar("n", 1, 17)
	// 1D symbolic buffer: src[0]=DefineVar, arg=nil.
	buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{nNode}, nil, nil)
	sh := shapeOf(buf)
	if len(sh) != 1 {
		t.Fatalf("rank = %d, want 1", len(sh))
	}
	sym, ok := sh[0].(shape.SymInt)
	if !ok || sym.Node != nNode {
		t.Errorf("dim 0 = %T, want SymInt over the DefineVar", sh[0])
	}
}

// ── candidateSize ────────────────────────────────────────────────────────────

func TestCandidateSize(t *testing.T) {
	a := uop.NewArena(128)

	// Scalar const → size 1.
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0), nil)
	if got := candidateSize(c); got != 1 {
		t.Errorf("scalar size = %d, want 1", got)
	}

	// Concrete 2x3x4 buffer → 24.
	buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 3, 4}, nil)
	if got := candidateSize(buf); got != 24 {
		t.Errorf("2x3x4 size = %d, want 24", got)
	}

	// Symbolic dim contributes the sentinel weight (1<<40), so the product is
	// much larger than any plausible concrete shape.
	nNode := a.DefineVar("n", 1, 17)
	symBuf := a.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{nNode}, nil, nil)
	if got := candidateSize(symBuf); got < (1 << 40) {
		t.Errorf("symbolic size = %d, want >= 1<<40", got)
	}
}

// ── cutLessTier1 / cutLessTier2 comparators ──────────────────────────────────

func TestCutComparators(t *testing.T) {
	a := uop.NewArena(128)
	small := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2}, nil)    // size 2
	large := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 8}, nil) // size 16
	scoreOf := map[uint32]int{small.Index(): 1, large.Index(): 9}

	// Tier1 primary key: smaller materialised tensor ranks first.
	if !cutLessTier1(small, large, scoreOf) {
		t.Errorf("tier1: small should rank before large by size")
	}
	if cutLessTier1(large, small, scoreOf) {
		t.Errorf("tier1: large should not rank before small by size")
	}

	// Tier1 size-tie → larger score first. Build two equal-size buffers.
	eqA := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)
	eqB := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)
	scoreEq := map[uint32]int{eqA.Index(): 5, eqB.Index(): 2}
	if !cutLessTier1(eqA, eqB, scoreEq) {
		t.Errorf("tier1 size-tie: higher score (eqA=5) should rank first")
	}

	// Tier2 primary key: larger shed (score) ranks first.
	if !cutLessTier2(large, small, scoreOf) {
		t.Errorf("tier2: larger score (large=9) should rank first")
	}
	if cutLessTier2(small, large, scoreOf) {
		t.Errorf("tier2: smaller score should not rank first")
	}
	// Tier2 score-tie → smaller size first.
	scoreTie := map[uint32]int{small.Index(): 3, large.Index(): 3}
	if !cutLessTier2(small, large, scoreTie) {
		t.Errorf("tier2 score-tie: smaller size should rank first")
	}
}

// ── identityConst ────────────────────────────────────────────────────────────

func TestIdentityConst(t *testing.T) {
	a := uop.NewArena(128)

	cases := []struct {
		name  string
		op    uop.Op
		dtype *uop.DType
		want  any
	}{
		{"add float", uop.OpAdd, uop.Dtypes.Float32, float64(0)},
		{"add int", uop.OpAdd, uop.Dtypes.Int32, int64(0)},
		{"add bool", uop.OpAdd, uop.Dtypes.Bool, false},
		{"mul float", uop.OpMul, uop.Dtypes.Float32, float64(1)},
		{"mul int", uop.OpMul, uop.Dtypes.Int32, int64(1)},
		{"mul bool", uop.OpMul, uop.Dtypes.Bool, true},
		{"max float", uop.OpMax, uop.Dtypes.Float32, math.Inf(-1)},
		{"max bool", uop.OpMax, uop.Dtypes.Bool, false},
		{"max int32", uop.OpMax, uop.Dtypes.Int32, int64(math.MinInt32)},
		{"max int8", uop.OpMax, uop.Dtypes.Int8, int64(math.MinInt8)},
		{"max int16", uop.OpMax, uop.Dtypes.Int16, int64(math.MinInt16)},
		{"max int64", uop.OpMax, uop.Dtypes.Int64, int64(math.MinInt64)},
		{"max uint32", uop.OpMax, uop.Dtypes.UInt32, int64(0)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u := identityConst(a, tc.op, tc.dtype)
			if u.Op() != uop.OpConst {
				t.Fatalf("identityConst op = %v, want OpConst", u.Op())
			}
			if u.Arg() != tc.want {
				t.Errorf("identityConst(%s) arg = %v (%T), want %v (%T)",
					tc.name, u.Arg(), u.Arg(), tc.want, tc.want)
			}
		})
	}
}

// ── dimToUOp ─────────────────────────────────────────────────────────────────

func TestDimToUOp(t *testing.T) {
	a := uop.NewArena(64)

	// Concrete Sint → OpConst with the value.
	cu := dimToUOp(a, shape.Const(7))
	if cu.Op() != uop.OpConst {
		t.Fatalf("concrete dimToUOp op = %v, want OpConst", cu.Op())
	}
	if v, ok := cu.Arg().(int64); !ok || v != 7 {
		t.Errorf("concrete dimToUOp arg = %v, want int64 7", cu.Arg())
	}

	// Symbolic Sint → the carried Node.
	nNode := a.DefineVar("n", 1, 9)
	su := dimToUOp(a, shape.SymInt{Node: nNode})
	if su != nNode {
		t.Errorf("symbolic dimToUOp should return the carried Node")
	}
}

// ── DebugBufRangesFlush ──────────────────────────────────────────────────────

func TestDebugBufRangesFlush(t *testing.T) {
	// Save and restore the package-global to avoid cross-test contamination.
	saved := DebugBufRanges
	t.Cleanup(func() { DebugBufRanges = saved })

	DebugBufRanges = []string{"line-a", "line-b"}
	path := filepath.Join(t.TempDir(), "bufranges.txt")
	DebugBufRangesFlush(path)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read flushed file: %v", err)
	}
	if string(data) != "line-a\nline-b\n" {
		t.Errorf("flushed content = %q, want %q", string(data), "line-a\nline-b\n")
	}
	if DebugBufRanges != nil {
		t.Errorf("DebugBufRanges not cleared after flush")
	}
}

// ── bufSize / bufShape (kernels.go) over all arg encodings ───────────────────

// mkBuf builds a BUFFER node carrying the given arg. For symbolic-arg cases the
// DefineVar(s) are not needed since these helpers read the arg structurally.
func mkBuf(a *uop.Arena, arg any) uop.UOp {
	return a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, arg, nil)
}

func TestBufSize(t *testing.T) {
	a := uop.NewArena(64)
	cases := []struct {
		name string
		arg  any
		want int64
	}{
		{"nil symbolic", nil, 0},
		{"int64 total", int64(12), 12},
		{"[]int64 product", []int64{2, 3, 4}, 24},
		{"ShapeSintArg dynamic", uop.ShapeSintArg{{V: 4}, {Sym: true, VarName: "n", Mul: 1}}, 0},
		{"BoundExprArg dynamic", uop.BoundExprArg{{V: 4}}, 0},
		{"string special", "randn", 1},
		{"default fallback", float64(3.14), 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := bufSize(mkBuf(a, tc.arg)); got != tc.want {
				t.Errorf("bufSize(%s) = %d, want %d", tc.name, got, tc.want)
			}
		})
	}
}

func TestBufShape(t *testing.T) {
	a := uop.NewArena(64)
	eq := func(got, want []int64) bool {
		if len(got) != len(want) {
			return false
		}
		for i := range got {
			if got[i] != want[i] {
				return false
			}
		}
		return true
	}
	cases := []struct {
		name string
		arg  any
		want []int64 // nil-want is handled specially below
		nilW bool
	}{
		{"int64 → [n]", int64(5), []int64{5}, false},
		{"[]int64 passthrough", []int64{2, 3}, []int64{2, 3}, false},
		{"ShapeSintArg sym→0", uop.ShapeSintArg{{Sym: true, VarName: "n", Mul: 1}, {V: 3}}, []int64{0, 3}, false},
		{"BoundExprArg sym→0", uop.BoundExprArg{{Sym: true}, {V: 7}}, []int64{0, 7}, false},
		{"string → [1]", "randn", []int64{1}, false},
		{"default → [1]", float64(1), []int64{1}, false},
		{"nil → nil", nil, nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := bufShape(mkBuf(a, tc.arg))
			if tc.nilW {
				if got != nil {
					t.Errorf("bufShape(%s) = %v, want nil", tc.name, got)
				}
				return
			}
			if !eq(got, tc.want) {
				t.Errorf("bufShape(%s) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// ── bufSymDimMul / bufSymDimAffine ───────────────────────────────────────────

func TestBufSymDimMul(t *testing.T) {
	a := uop.NewArena(64)

	// Non-ShapeSintArg arg → (nil, nil).
	if muls, vars := bufSymDimMul(mkBuf(a, []int64{2, 3})); muls != nil || vars != nil {
		t.Errorf("non-ShapeSintArg → (%v,%v), want (nil,nil)", muls, vars)
	}

	// All-concrete ShapeSintArg → (nil, nil) (no symbolic dim).
	allConc := uop.ShapeSintArg{{V: 4}, {V: 3}}
	if muls, vars := bufSymDimMul(mkBuf(a, allConc)); muls != nil || vars != nil {
		t.Errorf("all-concrete ShapeSintArg → (%v,%v), want (nil,nil)", muls, vars)
	}

	// Symbolic dims → parallel (mul, var) slices in dim order.
	arg := uop.ShapeSintArg{
		{V: 4},                            // concrete - skipped
		{Sym: true, VarName: "n", Mul: 1}, // bare
		{Sym: true, VarName: "k", Mul: 3}, // merged ×3
	}
	muls, vars := bufSymDimMul(mkBuf(a, arg))
	if len(muls) != 2 || len(vars) != 2 {
		t.Fatalf("len(muls,vars) = (%d,%d), want (2,2)", len(muls), len(vars))
	}
	if muls[0] != 1 || vars[0] != "n" || muls[1] != 3 || vars[1] != "k" {
		t.Errorf("got muls=%v vars=%v, want [1 3] [n k]", muls, vars)
	}
}

func TestBufSymDimAffine(t *testing.T) {
	a := uop.NewArena(64)

	// Non-BoundExprArg arg → nil.
	if got := bufSymDimAffine(mkBuf(a, uop.ShapeSintArg{{V: 4}})); got != nil {
		t.Errorf("non-BoundExprArg → %v, want nil", got)
	}

	// BoundExprArg with one concrete + one symbolic affine dim.
	arg := uop.BoundExprArg{
		{V: 4}, // concrete - skipped
		{Sym: true, Terms: []uop.AffineTerm{{Mul: 1, VarName: "n"}, {Mul: 1, VarName: "p"}}, Offset: 2},
	}
	out := bufSymDimAffine(mkBuf(a, arg))
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	e := out[0]
	if e.Offset != 2 || len(e.Terms) != 2 {
		t.Fatalf("entry = %+v, want Offset 2 with 2 terms", e)
	}
	if e.Terms[0].VarName != "n" || e.Terms[1].VarName != "p" {
		t.Errorf("terms = %v, want vars n,p", e.Terms)
	}
}

// ── symVarsFromKernel ────────────────────────────────────────────────────────

func TestSymVarsFromKernel(t *testing.T) {
	a := uop.NewArena(128)

	// Invalid AST → nil.
	if got := symVarsFromKernel(uop.UOp{}); got != nil {
		t.Errorf("invalid AST → %v, want nil", got)
	}

	// AST with no DefineVar → nil.
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if got := symVarsFromKernel(c); got != nil {
		t.Errorf("no-sym AST → %v, want nil", got)
	}

	// AST referencing two DefineVars → sorted names. Build an Add over two
	// DefineVar-bounded ranges so VariablesOf finds both.
	nVar := a.DefineVar("n", 1, 9)
	kVar := a.DefineVar("k", 1, 9)
	expr := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{nVar, kVar}, nil, nil)
	got := symVarsFromKernel(expr)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2 (got %v)", len(got), got)
	}
	// VariablesOf returns name-sorted, so "k" precedes "n".
	if got[0] != "k" || got[1] != "n" {
		t.Errorf("got %v, want [k n]", got)
	}
}
