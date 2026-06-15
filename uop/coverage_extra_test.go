package uop_test

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── DType.Base ────────────────────────────────────────────────────────────────

// TestBaseNonPtrReturnsSelf covers the non-pointer branch of Base, which returns
// the receiver unchanged.
func TestBaseNonPtrReturnsSelf(t *testing.T) {
	if got := uop.Dtypes.Float32.Base(); got != uop.Dtypes.Float32 {
		t.Errorf("Float32.Base() = %v, want Float32", got)
	}
	// Vector dtype is also non-ptr; Base returns it as-is.
	v := uop.Dtypes.Float32.Vec(4)
	if got := v.Base(); got != v {
		t.Errorf("Float32×4.Base() = %v, want self", got)
	}
}

// TestBasePtrReturnsElement covers the pointer branch of Base.
func TestBasePtrReturnsElement(t *testing.T) {
	p := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	if got := p.Base(); got != uop.Dtypes.Float32 {
		t.Errorf("ptr(Float32).Base() = %v, want Float32", got)
	}
}

// ── DType.String ──────────────────────────────────────────────────────────────

// TestDTypeStringVectorCount covers the count != 1 branch (e.g. "float×4").
func TestDTypeStringVectorCount(t *testing.T) {
	s := uop.Dtypes.Float32.Vec(4).String()
	if s != "float×4" {
		t.Errorf("Float32.Vec(4).String() = %q, want %q", s, "float×4")
	}
}

// TestDTypeStringPtrSized covers the sized-pointer branch where ptrSize != -1.
func TestDTypeStringPtrSized(t *testing.T) {
	s := uop.Dtypes.Float32.Ptr(16, uop.Global).String()
	if s != "float.ptr(16)" {
		t.Errorf("ptr(16) String = %q, want %q", s, "float.ptr(16)")
	}
}

// TestDTypeStringPtrSizedLocal combines sized + non-global address space.
func TestDTypeStringPtrSizedLocal(t *testing.T) {
	s := uop.Dtypes.Float32.Ptr(8, uop.Local).String()
	if s != "float.ptr(8)[Local]" {
		t.Errorf("ptr(8)[Local] String = %q, want %q", s, "float.ptr(8)[Local]")
	}
}

// ── CmpDType reachable orderings ──────────────────────────────────────────────

// TestCmpDTypeByBitSize covers the bitsize-difference branch (and its symmetry).
func TestCmpDTypeByBitSize(t *testing.T) {
	// Float16 (16 bits) vs Float32 (32 bits): same isPtr, differing priority is
	// possible, so pick two with equal priority. Use Vec to vary bitsize while
	// holding priority/name constant.
	small := uop.Dtypes.Float32      // bitsize 32
	big := uop.Dtypes.Float32.Vec(2) // bitsize 64, same priority+name
	if got := uop.CmpDType(small, big); got != -1 {
		t.Errorf("Cmp(f32, f32x2) = %d, want -1", got)
	}
	if got := uop.CmpDType(big, small); got != 1 {
		t.Errorf("Cmp(f32x2, f32) = %d, want 1", got)
	}
}

// TestCmpDTypePtrBaseRecursion covers the recursive base comparison for two
// pointers that match on all shallow fields but point to different elements.
func TestCmpDTypePtrBaseRecursion(t *testing.T) {
	pf := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	pi := uop.Dtypes.Int32.Ptr(-1, uop.Global)
	c := uop.CmpDType(pf, pi)
	if c == 0 {
		t.Fatal("distinct-base pointers compared equal")
	}
	// Antisymmetry.
	if uop.CmpDType(pi, pf) != -c {
		t.Errorf("Cmp not antisymmetric: %d vs %d", c, uop.CmpDType(pi, pf))
	}
}

// ── UOp.String (no arg/no tag branch) ─────────────────────────────────────────

// TestUOpStringNoArgNoTag covers the short-form String branch (arg==nil && tag==nil).
func TestUOpStringNoArgNoTag(t *testing.T) {
	a := uop.NewArena(8)
	// OpAdd of two consts: consts carry args, but a plain op with no arg/tag is
	// hard via real ops; build a node with explicitly nil arg+tag.
	c := a.New(uop.OpNoop, uop.Dtypes.Void, nil, nil, nil)
	s := c.String()
	if s == "" || s == "<invalid UOp>" {
		t.Fatalf("unexpected String() = %q", s)
	}
	// Must NOT contain "arg=" since both arg and tag are nil.
	for i := 0; i+4 <= len(s); i++ {
		if s[i:i+4] == "arg=" {
			t.Errorf("String() = %q, should omit arg= for nil-arg/nil-tag node", s)
		}
	}
}

// ── sortVarsByName (swap path) ────────────────────────────────────────────────

// TestVariablesOfSortSwaps builds vars out of name order so the insertion-sort
// swap branch in sortVarsByName executes.
func TestVariablesOfSortSwaps(t *testing.T) {
	a := uop.NewArena(8)
	// Names deliberately reverse-sorted: "z", "m", "a".
	vz := a.DefineVar("z", 0, 10)
	vm := a.DefineVar("m", 0, 10)
	va := a.DefineVar("a", 0, 10)
	root := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{
		a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{vz, vm}, nil, nil),
		va,
	}, nil, nil)
	vars := uop.VariablesOf(root)
	if len(vars) != 3 {
		t.Fatalf("got %d vars, want 3", len(vars))
	}
	wantOrder := []string{"a", "m", "z"}
	for i, v := range vars {
		got := v.Arg().(uop.VarArg).Name
		if got != wantOrder[i] {
			t.Errorf("vars[%d] = %q, want %q", i, got, wantOrder[i])
		}
	}
}

// ── BoundToAffine / mergeAffineTerms additional branches ──────────────────────

// TestBoundToAffineAddDistinctVars exercises mergeAffineTerms' not-found append
// branch (b's term has a VarName absent from a).
func TestBoundToAffineAddDistinctVars(t *testing.T) {
	a := uop.NewArena(8)
	n := a.DefineVar("n", 1, 100)
	m := a.DefineVar("m", 1, 100)
	expr := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{n, m}, nil, nil)
	terms, off, ok := uop.BoundToAffine(expr)
	if !ok || off != 0 {
		t.Fatalf("BoundToAffine(n+m) ok=%v off=%d", ok, off)
	}
	if len(terms) != 2 {
		t.Fatalf("got %d terms, want 2", len(terms))
	}
	names := map[string]int64{}
	for _, tm := range terms {
		names[tm.VarName] = tm.Mul
	}
	if names["n"] != 1 || names["m"] != 1 {
		t.Errorf("terms = %+v, want n:1 m:1", names)
	}
}

// TestBoundToAffineCancellingVars exercises mergeAffineTerms' zero-coefficient
// drop: n - n must collapse to no terms.
func TestBoundToAffineCancellingVars(t *testing.T) {
	a := uop.NewArena(8)
	n := a.DefineVar("n", 1, 100)
	expr := a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{n, n}, nil, nil)
	terms, off, ok := uop.BoundToAffine(expr)
	if !ok || off != 0 {
		t.Fatalf("BoundToAffine(n-n) ok=%v off=%d", ok, off)
	}
	if len(terms) != 0 {
		t.Errorf("n-n terms = %+v, want empty", terms)
	}
}

// TestBoundToAffineAddOfSums covers mergeAffineTerms where the left arg already
// has terms (the copy+accumulate path) with a repeated var (2n).
func TestBoundToAffineAddOfSums(t *testing.T) {
	a := uop.NewArena(8)
	n := a.DefineVar("n", 1, 100)
	expr := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{n, n}, nil, nil)
	terms, off, ok := uop.BoundToAffine(expr)
	if !ok || off != 0 {
		t.Fatalf("BoundToAffine(n+n) ok=%v off=%d", ok, off)
	}
	if len(terms) != 1 || terms[0].VarName != "n" || terms[0].Mul != 2 {
		t.Errorf("n+n terms = %+v, want [{2 n}]", terms)
	}
}

// ── hashArg / equalArg for Pad/Shrink/BoundExpr discriminators ─────────────────

// makeArgNode interns a node carrying arg payload; identical args must intern to
// the same node, differing args to distinct nodes.
func makeArgNode(a *uop.Arena, arg any) uop.UOp {
	return a.New(uop.OpNoop, uop.Dtypes.Void, nil, arg, nil)
}

// TestPadSintArgInterning covers PadSintArg in both hashArg and equalArg.
func TestPadSintArgInterning(t *testing.T) {
	a := uop.NewArena(8)
	p1 := uop.PadSintArg{
		{{V: 0}, {V: 2}},
		{{Sym: true, VarName: "n", Mul: 1}, {V: 0}},
	}
	p2 := uop.PadSintArg{
		{{V: 0}, {V: 2}},
		{{Sym: true, VarName: "n", Mul: 1}, {V: 0}},
	}
	if makeArgNode(a, p1) != makeArgNode(a, p2) {
		t.Error("identical PadSintArg did not intern to same node")
	}
	// Differ in a concrete value.
	p3 := uop.PadSintArg{
		{{V: 0}, {V: 3}},
		{{Sym: true, VarName: "n", Mul: 1}, {V: 0}},
	}
	if makeArgNode(a, p1) == makeArgNode(a, p3) {
		t.Error("differing PadSintArg interned to same node")
	}
	// Differ in length.
	p4 := uop.PadSintArg{{{V: 0}, {V: 2}}}
	if makeArgNode(a, p1) == makeArgNode(a, p4) {
		t.Error("different-length PadSintArg interned to same node")
	}
}

// TestShrinkSintArgInterning covers ShrinkSintArg in hashArg and equalArg.
func TestShrinkSintArgInterning(t *testing.T) {
	a := uop.NewArena(8)
	s1 := uop.ShrinkSintArg{{{V: 1}, {Sym: true, VarName: "m", Mul: 2}}}
	s2 := uop.ShrinkSintArg{{{V: 1}, {Sym: true, VarName: "m", Mul: 2}}}
	if makeArgNode(a, s1) != makeArgNode(a, s2) {
		t.Error("identical ShrinkSintArg did not intern to same node")
	}
	s3 := uop.ShrinkSintArg{{{V: 1}, {Sym: true, VarName: "m", Mul: 3}}}
	if makeArgNode(a, s1) == makeArgNode(a, s3) {
		t.Error("differing-Mul ShrinkSintArg interned to same node")
	}
}

// TestBoundExprArgInterning covers BoundExprArg sym and concrete branches in
// both hashArg and equalArg, including length, Offset and Terms differences.
func TestBoundExprArgInterning(t *testing.T) {
	a := uop.NewArena(8)
	b1 := uop.BoundExprArg{
		{V: 4},
		{Sym: true, Terms: []uop.AffineTerm{{Mul: 1, VarName: "n"}}, Offset: 2},
	}
	b2 := uop.BoundExprArg{
		{V: 4},
		{Sym: true, Terms: []uop.AffineTerm{{Mul: 1, VarName: "n"}}, Offset: 2},
	}
	if makeArgNode(a, b1) != makeArgNode(a, b2) {
		t.Error("identical BoundExprArg did not intern to same node")
	}
	// Differ in Offset.
	b3 := uop.BoundExprArg{
		{V: 4},
		{Sym: true, Terms: []uop.AffineTerm{{Mul: 1, VarName: "n"}}, Offset: 5},
	}
	if makeArgNode(a, b1) == makeArgNode(a, b3) {
		t.Error("differing-Offset BoundExprArg interned to same node")
	}
	// Differ in a Term coefficient.
	b4 := uop.BoundExprArg{
		{V: 4},
		{Sym: true, Terms: []uop.AffineTerm{{Mul: 3, VarName: "n"}}, Offset: 2},
	}
	if makeArgNode(a, b1) == makeArgNode(a, b4) {
		t.Error("differing-Term BoundExprArg interned to same node")
	}
	// Differ in concrete V.
	b5 := uop.BoundExprArg{
		{V: 8},
		{Sym: true, Terms: []uop.AffineTerm{{Mul: 1, VarName: "n"}}, Offset: 2},
	}
	if makeArgNode(a, b1) == makeArgNode(a, b5) {
		t.Error("differing-V BoundExprArg interned to same node")
	}
	// Differ in length.
	b6 := uop.BoundExprArg{{V: 4}}
	if makeArgNode(a, b1) == makeArgNode(a, b6) {
		t.Error("different-length BoundExprArg interned to same node")
	}
}

// TestArgTypeCrossDiscriminators ensures Pad/Shrink/Bound args with the same
// underlying layout do NOT collide across the type discriminators.
func TestArgTypeCrossDiscriminators(t *testing.T) {
	a := uop.NewArena(8)
	pad := uop.PadSintArg{{{V: 1}, {V: 2}}}
	shrink := uop.ShrinkSintArg{{{V: 1}, {V: 2}}}
	if makeArgNode(a, pad) == makeArgNode(a, shrink) {
		t.Error("PadSintArg and ShrinkSintArg with same layout collided")
	}
}
