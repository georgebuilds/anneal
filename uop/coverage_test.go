package uop_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── TopoSort ──────────────────────────────────────────────────────────────────

func TestTopoSortLinear(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	add := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)

	order := uop.TopoSort(add)
	if len(order) != 3 {
		t.Fatalf("TopoSort length = %d, want 3", len(order))
	}
	// Each src must appear strictly before its parent.
	pos := map[uop.UOp]int{}
	for i, u := range order {
		pos[u] = i
	}
	if pos[x] >= pos[add] || pos[y] >= pos[add] {
		t.Errorf("TopoSort did not place srcs before parent: pos x=%d y=%d add=%d",
			pos[x], pos[y], pos[add])
	}
}

func TestTopoSortSharedSubgraph(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	s := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)
	// mul(s, s) - same subgraph appears twice in src, must appear once in topo
	mul := a.New(uop.OpMul, uop.Dtypes.Float32, []uop.UOp{s, s}, nil, nil)

	order := uop.TopoSort(mul)
	if len(order) != 4 {
		t.Fatalf("TopoSort length = %d, want 4 (shared subgraph dedup)", len(order))
	}
}

func TestTopoSortDeep(t *testing.T) {
	a := uop.NewArena(64)
	root := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0), nil)
	for i := 0; i < 30; i++ {
		root = a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{root}, nil, nil)
	}
	order := uop.TopoSort(root)
	if len(order) != 31 {
		t.Fatalf("TopoSort deep chain length = %d, want 31", len(order))
	}
	if order[len(order)-1] != root {
		t.Error("TopoSort: root must be last in forward order")
	}
}

// ── RebuildSymBound ───────────────────────────────────────────────────────────

func TestRebuildSymBoundBare(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("N", 1, 100)
	rebuilt := uop.RebuildSymBound(a, uop.ShapeDim{Sym: true, VarName: "N", Mul: 1})
	if rebuilt != dv {
		t.Errorf("RebuildSymBound(N, mul=1) must alias the DefineVar (interning)")
	}
}

func TestRebuildSymBoundScaled(t *testing.T) {
	a := uop.NewArena(8)
	a.DefineVar("N", 1, 100)
	r := uop.RebuildSymBound(a, uop.ShapeDim{Sym: true, VarName: "N", Mul: 4})
	if r.Op() != uop.OpMul {
		t.Errorf("RebuildSymBound mul>1: op = %s, want OpMul", r.Op())
	}
	if r.NSrc() != 2 {
		t.Fatalf("OpMul NSrc = %d, want 2", r.NSrc())
	}
	if r.Src(0).Op() != uop.OpDefineVar {
		t.Errorf("OpMul.Src(0) must be DefineVar, got %s", r.Src(0).Op())
	}
	if r.Src(1).Op() != uop.OpConst || r.Src(1).Arg().(int64) != 4 {
		t.Errorf("OpMul.Src(1) must be Const(4), got %s arg=%v", r.Src(1).Op(), r.Src(1).Arg())
	}
}

func TestRebuildSymBoundPanicsMissing(t *testing.T) {
	a := uop.NewArena(8)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unknown VarName, got none")
		}
	}()
	uop.RebuildSymBound(a, uop.ShapeDim{Sym: true, VarName: "missing", Mul: 1})
}

// ── Range helpers ─────────────────────────────────────────────────────────────

func TestRangeBoundAndSize(t *testing.T) {
	a := uop.NewArena(8)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(16), nil)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)

	if uop.RangeBound(r) != c {
		t.Errorf("RangeBound != src[0]")
	}
	if uop.RangeIsSymbolic(r) {
		t.Errorf("Const-bound range reported symbolic")
	}
	if uop.RangeSize(r) != 16 {
		t.Errorf("RangeSize = %d, want 16", uop.RangeSize(r))
	}
}

func TestRangeIsSymbolic(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("N", 1, 100)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{dv}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)
	if !uop.RangeIsSymbolic(r) {
		t.Errorf("DefineVar-bound range not reported symbolic")
	}
}

func TestRangeSizePanicsSymbolic(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("M", 1, 99)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{dv}, uop.RangeArg{ID: 2, Type: uop.AxisLoop}, nil)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on RangeSize for symbolic, got none")
		}
	}()
	_ = uop.RangeSize(r)
}

// ── VariablesOf / sortVarsByName ─────────────────────────────────────────────

func TestVariablesOfInvalidReturnsNil(t *testing.T) {
	var u uop.UOp
	if uop.VariablesOf(u) != nil {
		t.Error("VariablesOf(zero) must return nil")
	}
}

func TestVariablesOfNoneFound(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	got := uop.VariablesOf(x)
	if len(got) != 0 {
		t.Errorf("VariablesOf(const) = %d vars, want 0", len(got))
	}
}

func TestVariablesOfMultipleSorted(t *testing.T) {
	a := uop.NewArena(16)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	// Add DefineVars in reverse-name order to exercise sortVarsByName.
	dz := a.DefineVar("Z", 1, 10)
	dn := a.DefineVar("N", 1, 10)
	da := a.DefineVar("A", 1, 10)

	// Build root = a + (n + (z * 2))
	t1 := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{dz, c}, nil, nil)
	t2 := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{dn, t1}, nil, nil)
	root := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{da, t2}, nil, nil)

	got := uop.VariablesOf(root)
	if len(got) != 3 {
		t.Fatalf("VariablesOf len = %d, want 3", len(got))
	}
	names := []string{
		got[0].Arg().(uop.VarArg).Name,
		got[1].Arg().(uop.VarArg).Name,
		got[2].Arg().(uop.VarArg).Name,
	}
	want := []string{"A", "N", "Z"}
	if !reflect.DeepEqual(names, want) {
		t.Errorf("VariablesOf order = %v, want %v", names, want)
	}
}

// ── BoundToAffine / SymBoundFactor / mergeAffineTerms ────────────────────────

func TestBoundToAffineConst(t *testing.T) {
	a := uop.NewArena(8)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(7), nil)
	terms, off, ok := uop.BoundToAffine(c)
	if !ok || off != 7 || len(terms) != 0 {
		t.Errorf("Const(7): got terms=%v off=%d ok=%v", terms, off, ok)
	}
}

func TestBoundToAffineDefineVar(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("N", 1, 10)
	terms, off, ok := uop.BoundToAffine(dv)
	if !ok || off != 0 || len(terms) != 1 || terms[0].Mul != 1 || terms[0].VarName != "N" {
		t.Errorf("DefineVar(N): got terms=%v off=%d ok=%v", terms, off, ok)
	}
}

func TestBoundToAffineMulConstVar(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("N", 1, 10)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	m := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{c, dv}, nil, nil)
	terms, off, ok := uop.BoundToAffine(m)
	if !ok || off != 0 || len(terms) != 1 || terms[0].Mul != 4 || terms[0].VarName != "N" {
		t.Errorf("Mul(C,N): got terms=%v off=%d ok=%v", terms, off, ok)
	}
}

func TestBoundToAffineMulVarConst(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("N", 1, 10)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
	m := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{dv, c}, nil, nil)
	terms, off, ok := uop.BoundToAffine(m)
	if !ok || off != 0 || len(terms) != 1 || terms[0].Mul != 3 || terms[0].VarName != "N" {
		t.Errorf("Mul(N,C): got terms=%v off=%d ok=%v", terms, off, ok)
	}
}

func TestBoundToAffineMulConstConst(t *testing.T) {
	a := uop.NewArena(8)
	c1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
	c2 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(5), nil)
	m := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{c1, c2}, nil, nil)
	terms, off, ok := uop.BoundToAffine(m)
	if !ok || off != 15 || len(terms) != 0 {
		t.Errorf("Mul(C,C): got terms=%v off=%d ok=%v", terms, off, ok)
	}
}

func TestBoundToAffineAddSub(t *testing.T) {
	a := uop.NewArena(16)
	n := a.DefineVar("N", 1, 10)
	m := a.DefineVar("M", 1, 10)
	c5 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(5), nil)
	c3 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
	// (3*N) + M - 5
	mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{c3, n}, nil, nil)
	add := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, m}, nil, nil)
	sub := a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{add, c5}, nil, nil)
	terms, off, ok := uop.BoundToAffine(sub)
	if !ok {
		t.Fatalf("BoundToAffine ok=false")
	}
	if off != -5 {
		t.Errorf("offset = %d, want -5", off)
	}
	if len(terms) != 2 {
		t.Fatalf("terms = %d, want 2", len(terms))
	}
	byName := map[string]int64{}
	for _, t2 := range terms {
		byName[t2.VarName] = t2.Mul
	}
	if byName["N"] != 3 || byName["M"] != 1 {
		t.Errorf("term coefficients: %v", byName)
	}
}

func TestBoundToAffineUnsupported(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y := a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{x}, nil, nil)
	_, _, ok := uop.BoundToAffine(y)
	if ok {
		t.Error("BoundToAffine(neg) must report ok=false")
	}
}

func TestSymBoundFactorBare(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("K", 1, 10)
	name, mul := uop.SymBoundFactor(dv)
	if name != "K" || mul != 1 {
		t.Errorf("got (%q,%d), want (K,1)", name, mul)
	}
}

func TestSymBoundFactorMulConstVar(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("K", 1, 10)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(8), nil)
	m := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{c, dv}, nil, nil)
	name, mul := uop.SymBoundFactor(m)
	if name != "K" || mul != 8 {
		t.Errorf("got (%q,%d), want (K,8)", name, mul)
	}
}

func TestSymBoundFactorMulVarConst(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("K", 1, 10)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	m := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{dv, c}, nil, nil)
	name, mul := uop.SymBoundFactor(m)
	if name != "K" || mul != 2 {
		t.Errorf("got (%q,%d), want (K,2)", name, mul)
	}
}

func TestSymBoundFactorPanics(t *testing.T) {
	a := uop.NewArena(8)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(5), nil)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on unsupported shape")
		}
	}()
	_ = a // capture to keep linter happy
	uop.SymBoundFactor(c)
}

// ── FindDefineVar ─────────────────────────────────────────────────────────────

func TestFindDefineVarHit(t *testing.T) {
	a := uop.NewArena(8)
	dv := a.DefineVar("N", 1, 10)
	got, ok := a.FindDefineVar("N")
	if !ok {
		t.Fatal("FindDefineVar ok=false for existing var")
	}
	if got != dv {
		t.Error("FindDefineVar returned different UOp than original")
	}
}

func TestFindDefineVarMiss(t *testing.T) {
	a := uop.NewArena(8)
	a.DefineVar("X", 1, 10)
	_, ok := a.FindDefineVar("Y")
	if ok {
		t.Error("FindDefineVar ok=true for unknown name")
	}
}

func TestFindDefineVarEmptyArena(t *testing.T) {
	a := uop.NewArena(8)
	if _, ok := a.FindDefineVar("Z"); ok {
		t.Error("empty arena: ok=true for any name")
	}
}

// ── Leaf storage ──────────────────────────────────────────────────────────────

func TestLeafSetAndGet(t *testing.T) {
	a := uop.NewArena(8)
	u := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)
	data := []float32{1, 2, 3, 4}
	a.SetLeaf(u.Index(), data)

	got, ok := a.Leaf(u.Index())
	if !ok {
		t.Fatal("Leaf ok=false after SetLeaf")
	}
	if !reflect.DeepEqual(got, data) {
		t.Errorf("Leaf data mismatch: got %v", got)
	}
}

func TestLeafMissing(t *testing.T) {
	a := uop.NewArena(4)
	if _, ok := a.Leaf(999); ok {
		t.Error("Leaf ok=true for never-set index")
	}
}

func TestLeafClearedByReset(t *testing.T) {
	a := uop.NewArena(4)
	u := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{1}, nil)
	a.SetLeaf(u.Index(), []float32{42})
	a.Reset()
	if _, ok := a.Leaf(0); ok {
		t.Error("Leaf survived Reset")
	}
}

// ── DType BitSize / Name ──────────────────────────────────────────────────────

func TestDTypeBitSize(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want int
	}{
		{uop.Dtypes.Bool, 1},
		{uop.Dtypes.Int8, 8},
		{uop.Dtypes.Float16, 16},
		{uop.Dtypes.Float32, 32},
		{uop.Dtypes.Float64, 64},
		{uop.Dtypes.Float32.Vec(4), 128},
	}
	for _, c := range cases {
		if got := c.dt.BitSize(); got != c.want {
			t.Errorf("%s.BitSize() = %d, want %d", c.dt, got, c.want)
		}
	}
}

func TestDTypeName(t *testing.T) {
	if uop.Dtypes.Float32.Name() != "float" {
		t.Errorf("Float32.Name() = %q, want %q", uop.Dtypes.Float32.Name(), "float")
	}
	if uop.Dtypes.Int32.Name() != "int" {
		t.Errorf("Int32.Name() = %q, want %q", uop.Dtypes.Int32.Name(), "int")
	}
}

// ── CmpDType (lift coverage from 17.8%) ──────────────────────────────────────

func TestCmpDTypeEqual(t *testing.T) {
	if uop.CmpDType(uop.Dtypes.Float32, uop.Dtypes.Float32) != 0 {
		t.Error("equal pointers must compare 0")
	}
}

func TestCmpDTypeNil(t *testing.T) {
	if uop.CmpDType(nil, uop.Dtypes.Int32) != -1 {
		t.Error("nil < any dtype")
	}
	if uop.CmpDType(uop.Dtypes.Int32, nil) != 1 {
		t.Error("any dtype > nil")
	}
	if uop.CmpDType(nil, nil) != 0 {
		t.Error("nil == nil")
	}
}

func TestCmpDTypePtrVsScalar(t *testing.T) {
	p := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	s := uop.Dtypes.Float32
	if uop.CmpDType(s, p) != -1 {
		t.Error("scalar < ptr")
	}
	if uop.CmpDType(p, s) != 1 {
		t.Error("ptr > scalar")
	}
}

func TestCmpDTypeByPriority(t *testing.T) {
	if uop.CmpDType(uop.Dtypes.Int8, uop.Dtypes.Int16) != -1 {
		t.Error("Int8 priority < Int16 priority")
	}
	if uop.CmpDType(uop.Dtypes.Int16, uop.Dtypes.Int8) != 1 {
		t.Error("Int16 priority > Int8 priority")
	}
}

func TestCmpDTypeByCount(t *testing.T) {
	v2 := uop.Dtypes.Float32.Vec(2)
	v4 := uop.Dtypes.Float32.Vec(4)
	if uop.CmpDType(v2, v4) != -1 {
		t.Error("Vec(2) < Vec(4)")
	}
	if uop.CmpDType(v4, v2) != 1 {
		t.Error("Vec(4) > Vec(2)")
	}
}

func TestCmpDTypePtrAddrSpace(t *testing.T) {
	g := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	l := uop.Dtypes.Float32.Ptr(-1, uop.Local)
	// Global < Local because Global=0, Local=1.
	if uop.CmpDType(g, l) != -1 {
		t.Errorf("Global-ptr < Local-ptr; got %d", uop.CmpDType(g, l))
	}
	if uop.CmpDType(l, g) != 1 {
		t.Errorf("Local-ptr > Global-ptr; got %d", uop.CmpDType(l, g))
	}
}

func TestCmpDTypePtrSize(t *testing.T) {
	p64 := uop.Dtypes.Float32.Ptr(64, uop.Global)
	p128 := uop.Dtypes.Float32.Ptr(128, uop.Global)
	if uop.CmpDType(p64, p128) != -1 {
		t.Error("ptr(64) < ptr(128)")
	}
	if uop.CmpDType(p128, p64) != 1 {
		t.Error("ptr(128) > ptr(64)")
	}
}

func TestCmpDTypePtrBase(t *testing.T) {
	pf := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	pi := uop.Dtypes.Int32.Ptr(-1, uop.Global)
	// Float32.priority=13 > Int32.priority=5 → pf > pi by priority, before base check
	if uop.CmpDType(pi, pf) != -1 {
		t.Errorf("ptr<int> < ptr<float>, got %d", uop.CmpDType(pi, pf))
	}
}

// ── AddrSpace.String ──────────────────────────────────────────────────────────

func TestAddrSpaceString(t *testing.T) {
	if uop.Global.String() != "Global" {
		t.Errorf("Global.String() = %q", uop.Global.String())
	}
	if uop.Local.String() != "Local" {
		t.Errorf("Local.String() = %q", uop.Local.String())
	}
	if uop.Reg.String() != "Reg" {
		t.Errorf("Reg.String() = %q", uop.Reg.String())
	}
	s := uop.AddrSpace(99).String()
	if !strings.Contains(s, "99") {
		t.Errorf("unknown AddrSpace.String() = %q", s)
	}
}

// ── Vec / Ptr panic paths ─────────────────────────────────────────────────────

func TestVecOnVectorPanics(t *testing.T) {
	v := uop.Dtypes.Float32.Vec(4)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when Vec'ing a vector")
		}
	}()
	_ = v.Vec(2)
}

func TestPtrOnPtrPanics(t *testing.T) {
	p := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when Ptr'ing a pointer dtype")
		}
	}()
	_ = p.Ptr(-1, uop.Global)
}

// ── DType.String nil ──────────────────────────────────────────────────────────

func TestDTypeStringNil(t *testing.T) {
	var d *uop.DType
	if s := d.String(); !strings.Contains(s, "nil") {
		t.Errorf("nil DType.String() = %q, want to contain 'nil'", s)
	}
}

// ── OpFromString case-insensitive ─────────────────────────────────────────────

func TestOpFromStringCaseInsensitive(t *testing.T) {
	op, ok := uop.OpFromString("ADD")
	if !ok || op != uop.OpAdd {
		t.Errorf("OpFromString(ADD) = (%s,%v), want (Add,true)", op, ok)
	}
}

func TestOpFromStringUnknown(t *testing.T) {
	if _, ok := uop.OpFromString("NotAnOp"); ok {
		t.Error("OpFromString(unknown) returned ok=true")
	}
}

// ── Phase.String ──────────────────────────────────────────────────────────────

func TestPhaseString(t *testing.T) {
	if uop.PhaseForward.String() != "forward" {
		t.Errorf("PhaseForward.String() = %q", uop.PhaseForward.String())
	}
	if uop.PhaseBackward.String() != "backward" {
		t.Errorf("PhaseBackward.String() = %q", uop.PhaseBackward.String())
	}
}

// ── UOp.String full path (with arg+tag) ──────────────────────────────────────

func TestUOpStringWithTag(t *testing.T) {
	a := uop.NewArena(4)
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), "mytag")
	s := u.String()
	if !strings.Contains(s, "mytag") {
		t.Errorf("String() = %q, want tag content", s)
	}
}
