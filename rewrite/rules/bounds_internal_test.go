package rules

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// White-box tests for the canonicalization comparator used by hCanonicalize.
// cmpUOp must impose a stable total order so commutative operands sort
// deterministically; an unstable order would make GraphRewrite non-confluent.

// ── typeTag ───────────────────────────────────────────────────────────────────

func TestTypeTagOrdering(t *testing.T) {
	cases := []struct {
		v   any
		tag int
	}{
		{nil, 0},
		{true, 1},
		{int64(3), 2},
		{float64(1.5), 3},
		{"x", 4},
		{uop.VarArg{Name: "n"}, 5}, // unknown type → 5
	}
	for _, tc := range cases {
		if got := typeTag(tc.v); got != tc.tag {
			t.Errorf("typeTag(%T) = %d, want %d", tc.v, got, tc.tag)
		}
	}
}

// ── cmpAny ────────────────────────────────────────────────────────────────────

func TestCmpAnyCrossType(t *testing.T) {
	// nil < bool < int64 < float64 < string, by typeTag.
	pairs := []struct {
		a, b any
		want int
	}{
		{nil, true, -1},
		{true, int64(0), -1},
		{int64(0), float64(0), -1},
		{float64(0), "a", -1},
		{"a", nil, 1}, // reverse
	}
	for _, tc := range pairs {
		if got := cmpAny(tc.a, tc.b); got != tc.want {
			t.Errorf("cmpAny(%v,%v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestCmpAnySameType(t *testing.T) {
	cases := []struct {
		a, b any
		want int
	}{
		{nil, nil, 0},
		{false, false, 0},
		{false, true, -1},
		{true, false, 1},
		{int64(1), int64(2), -1},
		{int64(2), int64(1), 1},
		{int64(5), int64(5), 0},
		{float64(1), float64(2), -1},
		{float64(2), float64(1), 1},
		{float64(2), float64(2), 0},
		{"a", "b", -1},
		{"b", "a", 1},
		{"a", "a", 0},
	}
	for _, tc := range cases {
		if got := cmpAny(tc.a, tc.b); got != tc.want {
			t.Errorf("cmpAny(%v,%v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

// TestCmpAnyUnknownType covers the final fall-through return 0 for two values of
// the same unknown type tag (5).
func TestCmpAnyUnknownType(t *testing.T) {
	if got := cmpAny(uop.VarArg{Name: "a"}, uop.VarArg{Name: "b"}); got != 0 {
		t.Errorf("cmpAny(unknown,unknown) = %d, want 0", got)
	}
}

// ── cmpUOp ────────────────────────────────────────────────────────────────────

func TestCmpUOpIdentity(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(3), nil)
	if cmpUOp(x, x) != 0 {
		t.Error("cmpUOp(x,x) != 0")
	}
}

func TestCmpUOpByOp(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(2), nil)
	add := a.New(uop.OpAdd, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
	mul := a.New(uop.OpMul, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
	c := cmpUOp(add, mul)
	if c == 0 {
		t.Fatal("Add and Mul compared equal")
	}
	// Antisymmetric.
	if cmpUOp(mul, add) != -c {
		t.Error("cmpUOp not antisymmetric across ops")
	}
	// Order follows Op constant ordering.
	want := -1
	if uop.OpAdd > uop.OpMul {
		want = 1
	}
	if c != want {
		t.Errorf("cmpUOp(Add,Mul) = %d, want %d", c, want)
	}
}

func TestCmpUOpByArg(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(2), nil)
	if cmpUOp(x, y) != -1 {
		t.Errorf("cmpUOp(Const 1, Const 2) = %d, want -1", cmpUOp(x, y))
	}
	if cmpUOp(y, x) != 1 {
		t.Errorf("cmpUOp(Const 2, Const 1) = %d, want 1", cmpUOp(y, x))
	}
}

func TestCmpUOpByDType(t *testing.T) {
	a := uop.NewArena(16)
	// Same op + same arg, differing dtype.
	xi := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	xl := a.New(uop.OpConst, uop.Dtypes.Int64, nil, int64(1), nil)
	c := cmpUOp(xi, xl)
	if c == 0 {
		t.Fatal("nodes with differing dtype compared equal")
	}
	if cmpUOp(xl, xi) != -c {
		t.Error("cmpUOp not antisymmetric across dtype")
	}
}

func TestCmpUOpByNSrc(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	// Same op (Noop), differing src count.
	n0 := a.New(uop.OpNoop, uop.Dtypes.Void, nil, nil, nil)
	n1 := a.New(uop.OpNoop, uop.Dtypes.Void, []uop.UOp{x}, nil, nil)
	if cmpUOp(n0, n1) != -1 {
		t.Errorf("cmpUOp(0-src, 1-src) = %d, want -1", cmpUOp(n0, n1))
	}
	if cmpUOp(n1, n0) != 1 {
		t.Errorf("cmpUOp(1-src, 0-src) = %d, want 1", cmpUOp(n1, n0))
	}
}

func TestCmpUOpRecursesIntoSrc(t *testing.T) {
	a := uop.NewArena(16)
	c1 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	c2 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(2), nil)
	// Same op + same arity + same dtype; differ only in a child's arg.
	addA := a.New(uop.OpNeg, uop.Dtypes.Int32, []uop.UOp{c1}, nil, nil)
	addB := a.New(uop.OpNeg, uop.Dtypes.Int32, []uop.UOp{c2}, nil, nil)
	c := cmpUOp(addA, addB)
	if c != -1 {
		t.Errorf("cmpUOp recursing into src: got %d, want -1", c)
	}
}

// TestCmpUOpDepthLimit forces the depth==0 arena-index fallback by passing
// depth 0 directly to cmpUOpD.
func TestCmpUOpDepthLimit(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(2), nil)
	// At depth 0, comparison falls back to arena index order.
	c := cmpUOpD(x, y, 0)
	wantC := -1
	if x.Index() > y.Index() {
		wantC = 1
	}
	if c != wantC {
		t.Errorf("cmpUOpD depth 0 = %d, want %d (index fallback)", c, wantC)
	}
	// Equal handle short-circuits before depth check.
	if cmpUOpD(x, x, 0) != 0 {
		t.Error("cmpUOpD(x,x,0) != 0")
	}
}

// ── BoundsOf edge cases ───────────────────────────────────────────────────────

func TestBoundsOfBoolConst(t *testing.T) {
	a := uop.NewArena(16)
	bt := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	bf := a.New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil)
	if b := BoundsOf(bt); !b.Valid || b.Min != 1 || b.Max != 1 {
		t.Errorf("BoundsOf(true) = %+v, want exact 1", b)
	}
	if b := BoundsOf(bf); !b.Valid || b.Min != 0 || b.Max != 0 {
		t.Errorf("BoundsOf(false) = %+v, want exact 0", b)
	}
}

func TestBoundsOfRange(t *testing.T) {
	a := uop.NewArena(16)
	upper := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(8), nil)
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{upper}, uop.RangeArg{ID: 0}, nil)
	b := BoundsOf(rng)
	if !b.Valid || b.Min != 0 || b.Max != 7 {
		t.Errorf("BoundsOf(Range(8)) = %+v, want [0,7]", b)
	}
}

// TestBoundsOfRangeZeroUpper covers the invalid path where upper bound is not > 0.
func TestBoundsOfRangeZeroUpper(t *testing.T) {
	a := uop.NewArena(16)
	upper := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{upper}, uop.RangeArg{ID: 1}, nil)
	if b := BoundsOf(rng); b.Valid {
		t.Errorf("BoundsOf(Range(0)) = %+v, want invalid", b)
	}
}

// TestBoundsOfUnsupportedBinaryOp covers the trailing fallthrough: a 2-src
// integer ALU op not handled by the bounds switch returns invalid.
func TestBoundsOfUnsupportedBinaryOp(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(3), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(2), nil)
	// OpShl is a binary int op but not in the BoundsOf switch.
	node := a.New(uop.OpShl, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
	if b := BoundsOf(node); b.Valid {
		t.Errorf("BoundsOf(Shl) = %+v, want invalid (unsupported)", b)
	}
}

// TestBoundsOfNonIntConstArg covers the OpConst branch where the arg is neither
// int nor bool (e.g. a float arg on an int-typed const is not representable).
func TestBoundsOfFloatConstInvalid(t *testing.T) {
	a := uop.NewArena(16)
	// Float dtype short-circuits at the top of BoundsOf.
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1.5), nil)
	if b := BoundsOf(c); b.Valid {
		t.Errorf("BoundsOf(float const) = %+v, want invalid", b)
	}
}
