package rules_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/rewrite/rules"
	"github.com/georgebuilds/anneal/uop"
)

// ── arena helpers ─────────────────────────────────────────────────────────────

func arena() *uop.Arena { return uop.NewArena(256) }

func ci(a *uop.Arena, v int64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Int32, nil, v, nil)
}

func ci64(a *uop.Arena, v int64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Int64, nil, v, nil)
}

func cf(a *uop.Arena, v float64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Float32, nil, v, nil)
}

func cb(a *uop.Arena, v bool) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Bool, nil, v, nil)
}

func bop(a *uop.Arena, op uop.Op, dtype *uop.DType, x, y uop.UOp) uop.UOp {
	return a.New(op, dtype, []uop.UOp{x, y}, nil, nil)
}

func add(a *uop.Arena, x, y uop.UOp) uop.UOp { return bop(a, uop.OpAdd, uop.Dtypes.Int32, x, y) }
func mul(a *uop.Arena, x, y uop.UOp) uop.UOp { return bop(a, uop.OpMul, uop.Dtypes.Int32, x, y) }
func sub(a *uop.Arena, x, y uop.UOp) uop.UOp { return bop(a, uop.OpSub, uop.Dtypes.Int32, x, y) }
func idiv(a *uop.Arena, x, y uop.UOp) uop.UOp { return bop(a, uop.OpIDiv, uop.Dtypes.Int32, x, y) }
func mod(a *uop.Arena, x, y uop.UOp) uop.UOp  { return bop(a, uop.OpMod, uop.Dtypes.Int32, x, y) }
func xor(a *uop.Arena, x, y uop.UOp) uop.UOp  { return bop(a, uop.OpXor, uop.Dtypes.Int32, x, y) }
func and(a *uop.Arena, x, y uop.UOp) uop.UOp  { return bop(a, uop.OpAnd, uop.Dtypes.Int32, x, y) }
func cmplt(a *uop.Arena, x, y uop.UOp) uop.UOp {
	return bop(a, uop.OpCmpLt, uop.Dtypes.Bool, x, y)
}
func cmpne(a *uop.Arena, x, y uop.UOp) uop.UOp {
	return bop(a, uop.OpCmpNe, uop.Dtypes.Bool, x, y)
}

func sym(root uop.UOp) uop.UOp { return rewrite.GraphRewrite(root, rules.Symbolic) }

// ── 1. Constant folding ───────────────────────────────────────────────────────

func TestConstFoldBinaryInt(t *testing.T) {
	tests := []struct {
		name string
		node func(a *uop.Arena) uop.UOp
		want int64
	}{
		{"add", func(a *uop.Arena) uop.UOp { return add(a, ci(a, 3), ci(a, 5)) }, 8},
		{"sub", func(a *uop.Arena) uop.UOp { return sub(a, ci(a, 10), ci(a, 3)) }, 7},
		{"mul", func(a *uop.Arena) uop.UOp { return mul(a, ci(a, 6), ci(a, 7)) }, 42},
		{"idiv floor", func(a *uop.Arena) uop.UOp { return idiv(a, ci(a, -7), ci(a, 3)) }, -3},
		{"mod floor", func(a *uop.Arena) uop.UOp { return mod(a, ci(a, -7), ci(a, 3)) }, 2},
		{"xor", func(a *uop.Arena) uop.UOp { return xor(a, ci(a, 5), ci(a, 3)) }, 6},
		{"and bitwise", func(a *uop.Arena) uop.UOp { return and(a, ci(a, 5), ci(a, 3)) }, 1},
		{"max", func(a *uop.Arena) uop.UOp { return bop(a, uop.OpMax, uop.Dtypes.Int32, ci(a, 3), ci(a, 7)) }, 7},
		{"shl", func(a *uop.Arena) uop.UOp { return bop(a, uop.OpShl, uop.Dtypes.Int32, ci(a, 1), ci(a, 4)) }, 16},
		{"shr", func(a *uop.Arena) uop.UOp { return bop(a, uop.OpShr, uop.Dtypes.Int32, ci(a, 16), ci(a, 2)) }, 4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			result := sym(tc.node(a))
			if result.Op() != uop.OpConst {
				t.Fatalf("expected Const, got %v", result.Op())
			}
			got, ok := result.Arg().(int64)
			if !ok {
				t.Fatalf("arg is %T, want int64", result.Arg())
			}
			if got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func TestConstFoldInt32Overflow(t *testing.T) {
	// 2_000_000_000 + 2_000_000_000 overflows int32 → wraps to -294_967_296
	a := arena()
	big := ci(a, 2_000_000_000)
	result := sym(add(a, big, big))
	if result.Op() != uop.OpConst {
		t.Fatal("expected Const after fold")
	}
	got := result.Arg().(int64)
	sum := int64(2_000_000_000) + int64(2_000_000_000) // 4_000_000_000
	want := int64(int32(sum))
	if got != want {
		t.Errorf("int32 overflow: got %d, want %d", got, want)
	}
}

func TestConstFoldComparison(t *testing.T) {
	tests := []struct {
		name string
		node func(*uop.Arena) uop.UOp
		want bool
	}{
		{"3<5=true", func(a *uop.Arena) uop.UOp { return cmplt(a, ci(a, 3), ci(a, 5)) }, true},
		{"5<3=false", func(a *uop.Arena) uop.UOp { return cmplt(a, ci(a, 5), ci(a, 3)) }, false},
		{"3!=5=true", func(a *uop.Arena) uop.UOp { return cmpne(a, ci(a, 3), ci(a, 5)) }, true},
		{"3!=3=false", func(a *uop.Arena) uop.UOp { return cmpne(a, ci(a, 3), ci(a, 3)) }, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			result := sym(tc.node(a))
			if result.Op() != uop.OpConst {
				t.Fatalf("want Const, got %v", result.Op())
			}
			got, ok := result.Arg().(bool)
			if !ok {
				t.Fatalf("arg type %T, want bool", result.Arg())
			}
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestConstFoldFloatAdd(t *testing.T) {
	a := arena()
	result := sym(bop(a, uop.OpAdd, uop.Dtypes.Float32, cf(a, 1.5), cf(a, 2.5)))
	if result.Op() != uop.OpConst {
		t.Fatal("expected Const")
	}
	got := result.Arg().(float64)
	// float32 round-trip: float64(float32(4.0)) = 4.0
	if got != 4.0 {
		t.Errorf("float add: got %v, want 4.0", got)
	}
}

func TestConstFoldUnary(t *testing.T) {
	a := arena()
	// Neg(-5) = 5
	neg := a.New(uop.OpNeg, uop.Dtypes.Int32, []uop.UOp{ci(a, -5)}, nil, nil)
	result := sym(neg)
	if result.Op() != uop.OpConst {
		t.Fatal("expected Const after Neg fold")
	}
	if result.Arg().(int64) != 5 {
		t.Errorf("Neg(-5): got %v, want 5", result.Arg())
	}
}

func TestFloorDivModSemantics(t *testing.T) {
	tests := []struct{ a, b, wantDiv, wantMod int64 }{
		{7, 3, 2, 1},
		{-7, 3, -3, 2},  // Python: -7 // 3 = -3, -7 % 3 = 2
		{7, -3, -3, -2}, // Python: 7 // -3 = -3, 7 % -3 = -2
	}
	for _, tc := range tests {
		a := arena()
		divResult := sym(idiv(a, ci(a, tc.a), ci(a, tc.b)))
		modResult := sym(mod(a, ci(a, tc.a), ci(a, tc.b)))
		if divResult.Arg().(int64) != tc.wantDiv {
			t.Errorf("%d//%d: got %d, want %d", tc.a, tc.b, divResult.Arg().(int64), tc.wantDiv)
		}
		if modResult.Arg().(int64) != tc.wantMod {
			t.Errorf("%d%%%d: got %d, want %d", tc.a, tc.b, modResult.Arg().(int64), tc.wantMod)
		}
	}
}

// ── 2. Cast folding ────────────────────────────────────────────────────────────

func TestCastConstFold(t *testing.T) {
	a := arena()
	// Cast Int32 → Float32
	c5 := ci(a, 5)
	castNode := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{c5}, nil, nil)
	result := sym(castNode)
	if result.Op() != uop.OpConst {
		t.Fatalf("expected Const, got %v", result.Op())
	}
	if result.DType() != uop.Dtypes.Float32 {
		t.Errorf("dtype: got %v, want Float32", result.DType())
	}
	got := result.Arg().(float64)
	if got != 5.0 {
		t.Errorf("Cast(5): got %v, want 5.0", got)
	}
}

func TestIdentityCast(t *testing.T) {
	a := arena()
	x := ci(a, 7)
	castNode := a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{x}, nil, nil)
	result := sym(castNode)
	if result != x {
		t.Errorf("identity cast: expected x, got different node")
	}
}

// ── 3. Arithmetic identities ──────────────────────────────────────────────────

func TestAddZero(t *testing.T) {
	a := arena()
	x := ci(a, 7)
	// x + 0 (canonical)
	if r := sym(add(a, x, ci(a, 0))); r != x {
		t.Errorf("x+0: expected x, got %v", r.Op())
	}
	// 0 + x (needs commutative swap then identity)
	if r := sym(add(a, ci(a, 0), x)); r != x {
		t.Errorf("0+x: expected x, got %v", r.Op())
	}
}

func TestMulOne(t *testing.T) {
	a := arena()
	x := ci(a, 7)
	if r := sym(mul(a, x, ci(a, 1))); r != x {
		t.Errorf("x*1: expected x")
	}
	if r := sym(mul(a, ci(a, 1), x)); r != x {
		t.Errorf("1*x: expected x")
	}
}

func TestMulZero(t *testing.T) {
	a := arena()
	x := ci(a, 99)
	result := sym(mul(a, x, ci(a, 0)))
	if result.Op() != uop.OpConst || result.Arg().(int64) != 0 {
		t.Errorf("x*0: expected Const(0), got %v arg=%v", result.Op(), result.Arg())
	}
}

func TestSubZero(t *testing.T) {
	a := arena()
	x := ci(a, 7)
	if r := sym(sub(a, x, ci(a, 0))); r != x {
		t.Errorf("x-0: expected x")
	}
}

func TestXorZero(t *testing.T) {
	a := arena()
	x := ci(a, 42)
	if r := sym(xor(a, x, ci(a, 0))); r != x {
		t.Errorf("x^0: expected x")
	}
}

func TestIDiv(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	// x // 1 → x  (x is a non-const variable)
	if r := sym(idiv(a, x, ci(a, 1))); r != x {
		t.Errorf("x//1: expected x, got %v", r.Op())
	}
	// x // -1 → Neg(x)
	neg := sym(idiv(a, x, ci(a, -1)))
	if neg.Op() != uop.OpNeg {
		t.Errorf("x//-1: expected Neg, got %v", neg.Op())
	}
}

// ── 4. Self-cancellation ──────────────────────────────────────────────────────

func TestSelfCancellation(t *testing.T) {
	a := arena()
	x := ci(a, 99)

	// x - x → 0
	if r := sym(sub(a, x, x)); r.Op() != uop.OpConst || r.Arg().(int64) != 0 {
		t.Errorf("x-x: expected 0, got op=%v arg=%v", r.Op(), r.Arg())
	}
	// x ^ x → 0
	if r := sym(xor(a, x, x)); r.Op() != uop.OpConst || r.Arg().(int64) != 0 {
		t.Errorf("x^x: expected 0")
	}
	// x % x → 0
	if r := sym(mod(a, x, x)); r.Op() != uop.OpConst || r.Arg().(int64) != 0 {
		t.Errorf("x%%x: expected 0")
	}
	// x // x → 1
	if r := sym(idiv(a, x, x)); r.Op() != uop.OpConst || r.Arg().(int64) != 1 {
		t.Errorf("x//x: expected 1, got %v", r.Arg())
	}
	// x < x → false
	if r := sym(cmplt(a, x, x)); r.Op() != uop.OpConst || r.Arg().(bool) != false {
		t.Errorf("x<x: expected false")
	}
	// x != x → false
	if r := sym(cmpne(a, x, x)); r.Op() != uop.OpConst || r.Arg().(bool) != false {
		t.Errorf("x!=x: expected false")
	}
}

// ── 5. Idempotent ─────────────────────────────────────────────────────────────

func TestIdempotent(t *testing.T) {
	a := arena()
	x := bop(a, uop.OpAdd, uop.Dtypes.Int32, ci(a, 1), ci(a, 2)) // a non-const x

	ops := []struct {
		name string
		op   uop.Op
	}{
		{"Or", uop.OpOr},
		{"And", uop.OpAnd},
		{"Max", uop.OpMax},
	}
	for _, tc := range ops {
		t.Run(tc.name, func(t *testing.T) {
			node := bop(a, tc.op, uop.Dtypes.Int32, x, x)
			r := sym(node)
			if r != x {
				// x|x, x&x, max(x,x) should reduce to x
				// Note: the folding of Add(1,2)→3 happens too, so result might be Const(3)
				// Check that it's the folded form of x.
				xFolded := sym(x)
				if r != xFolded {
					t.Errorf("%s(x,x): expected x (or its folded form), got op=%v", tc.name, r.Op())
				}
			}
		})
	}
}

// ── 6. Canonicalization: a+b and b+a intern to the same node ─────────────────

func TestCommutativeCanonIntern(t *testing.T) {
	a := arena()
	// Build two variables (simulated by different constants)
	p := ci(a, 17) // "p"
	q := ci(a, 31) // "q"

	pq := sym(add(a, p, q))
	qp := sym(add(a, q, p))

	// After simplification both should be the same node (both are Const(48) from folding).
	if pq != qp {
		t.Errorf("p+q and q+p produced different nodes: %v vs %v", pq.Index(), qp.Index())
	}
}

func TestCommutativeCanonWithVar(t *testing.T) {
	// Use a non-constant "variable" node by using OpDefineVar (no rewrite rules match it).
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "idx", nil)
	c5 := ci(a, 5)

	// x+5 (already canonical: non-const at 0, const at 1)
	xPlus5 := add(a, x, c5)
	r1 := sym(xPlus5)

	// 5+x (non-canonical: const at 0) → canonicalize → x+5
	fivePlusX := add(a, c5, x)
	r2 := sym(fivePlusX)

	// Both should end up as the same interned node.
	if r1 != r2 {
		t.Errorf("x+5 and 5+x produced different nodes after canonicalization")
	}
}

// ── 7. Double-XOR and modulo idempotent ──────────────────────────────────────

func TestDoubleXOR(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	y := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "y", nil)

	// (x ^ y) ^ y → x
	inner := xor(a, x, y)
	outer := xor(a, inner, y)
	result := sym(outer)
	if result != x {
		t.Errorf("(x^y)^y: expected x, got op=%v", result.Op())
	}
}

func TestModIdem(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	y := ci(a, 7)

	// (x % y) % y → x % y
	inner := mod(a, x, y)
	outer := mod(a, inner, y)
	result := sym(outer)
	if result != inner {
		t.Errorf("(x%%y)%%y: expected inner x%%y node, got op=%v", result.Op())
	}
}

// ── 8. Boolean algebra ────────────────────────────────────────────────────────

func TestBoolAlgebra(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
	y := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "y", nil)

	// bool * bool → bool & bool
	mulNode := a.New(uop.OpMul, uop.Dtypes.Bool, []uop.UOp{x, y}, nil, nil)
	r := sym(mulNode)
	if r.Op() != uop.OpAnd {
		t.Errorf("bool*bool: expected And, got %v", r.Op())
	}

	// bool + bool → bool | bool
	addNode := a.New(uop.OpAdd, uop.Dtypes.Bool, []uop.UOp{x, y}, nil, nil)
	r = sym(addNode)
	if r.Op() != uop.OpOr {
		t.Errorf("bool+bool: expected Or, got %v", r.Op())
	}
}

// ── 9. Where ─────────────────────────────────────────────────────────────────

func TestWhereIdentical(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "cond", nil)
	v := ci(a, 42)

	whereNode := a.New(uop.OpWhere, uop.Dtypes.Int32, []uop.UOp{x, v, v}, nil, nil)
	result := sym(whereNode)
	if result != v {
		t.Errorf("where(cond,v,v): expected v, got op=%v", result.Op())
	}
}

func TestWhereConstCond(t *testing.T) {
	a := arena()
	trueC := cb(a, true)
	falseC := cb(a, false)
	v1 := ci(a, 10)
	v2 := ci(a, 20)

	r1 := sym(a.New(uop.OpWhere, uop.Dtypes.Int32, []uop.UOp{trueC, v1, v2}, nil, nil))
	if r1 != v1 {
		t.Errorf("where(true,v1,v2): expected v1")
	}
	r2 := sym(a.New(uop.OpWhere, uop.Dtypes.Int32, []uop.UOp{falseC, v1, v2}, nil, nil))
	if r2 != v2 {
		t.Errorf("where(false,v1,v2): expected v2")
	}
}

// ── 10. Bound propagation ─────────────────────────────────────────────────────

func TestBoundsOfConst(t *testing.T) {
	a := arena()
	c7 := ci(a, 7)
	b := rules.BoundsOf(c7)
	if !b.Valid || b.Min != 7 || b.Max != 7 {
		t.Errorf("BoundsOf(Const(7)) = %+v, want {7,7,true}", b)
	}
}

func TestBoundsOfAdd(t *testing.T) {
	a := arena()
	// Add(Const(3), Const(10)): bounds should be [13,13] but const folding happens first.
	// Test with non-const vars to isolate bounds logic.
	// Use range-like DefineVar with two-src form to test bounds propagation.
	lo := ci(a, 0)
	hi := ci(a, 10)
	v := a.New(uop.OpDefineVar, uop.Dtypes.Int32, []uop.UOp{lo, hi}, "x", nil)
	bv := rules.BoundsOf(v)
	if !bv.Valid || bv.Min != 0 || bv.Max != 9 {
		t.Errorf("BoundsOf(DefineVar(0,10)) = %+v, want {0,9,true}", bv)
	}

	// Add(v, Const(5)): bounds should be [5, 14]
	addNode := add(a, v, ci(a, 5))
	ba := rules.BoundsOf(addNode)
	if !ba.Valid || ba.Min != 5 || ba.Max != 14 {
		t.Errorf("BoundsOf(v+5) = %+v, want {5,14,true}", ba)
	}
}

func TestBoundsOfMul(t *testing.T) {
	a := arena()
	lo := ci(a, 1)
	hi := ci(a, 4)
	v := a.New(uop.OpDefineVar, uop.Dtypes.Int32, []uop.UOp{lo, hi}, "x", nil)
	// v in [1,3], Const(2): mul bounds [2, 6]
	m := mul(a, v, ci(a, 2))
	bm := rules.BoundsOf(m)
	if !bm.Valid || bm.Min != 2 || bm.Max != 6 {
		t.Errorf("BoundsOf(v*2) = %+v, want {2,6,true}", bm)
	}
}

// ── 11. Bound-based comparison folding ───────────────────────────────────────

func TestCmpLtBoundsFold(t *testing.T) {
	a := arena()
	lo := ci(a, 0)
	hi := ci(a, 5)
	v := a.New(uop.OpDefineVar, uop.Dtypes.Int32, []uop.UOp{lo, hi}, "x", nil)
	// v in [0,4], Const(10): v < 10 → always true
	lt := cmplt(a, v, ci(a, 10))
	result := sym(lt)
	if result.Op() != uop.OpConst || result.Arg().(bool) != true {
		t.Errorf("v<10 where v in [0,4]: expected true, got op=%v arg=%v", result.Op(), result.Arg())
	}

	// v in [0,4], Const(-1): v < -1 → always false
	lt2 := cmplt(a, v, ci(a, -1))
	result2 := sym(lt2)
	if result2.Op() != uop.OpConst || result2.Arg().(bool) != false {
		t.Errorf("v<-1 where v in [0,4]: expected false, got op=%v arg=%v", result2.Op(), result2.Arg())
	}
}

func TestCmpNeBoundsFold(t *testing.T) {
	a := arena()
	// Disjoint ranges: Const(3) and Const(7) → always not equal
	c3 := ci(a, 3)
	c7 := ci(a, 7)
	ne := cmpne(a, c3, c7)
	result := sym(ne)
	if result.Op() != uop.OpConst || result.Arg().(bool) != true {
		t.Errorf("3!=7: expected true, got op=%v arg=%v", result.Op(), result.Arg())
	}
}

// ── 12. Fixpoint convergence over a multi-rule chain ──────────────────────────

// TestFixpointMultiRule: (0 + x) * 1 → x  (requires add-zero + mul-one, with
// the commutative swap for 0+x → x+0 en route).
func TestFixpointMultiRule(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	// Build ((0+x)*1)
	inner := add(a, ci(a, 0), x) // 0+x
	outer := mul(a, inner, ci(a, 1)) // (0+x)*1
	result := sym(outer)
	if result != x {
		t.Errorf("(0+x)*1: expected x, got op=%v", result.Op())
	}
}

// ── 13. No oscillation ────────────────────────────────────────────────────────

// TestNoOscillation verifies that the ruleset converges on a sampling of
// expressions and does not oscillate (applying rules infinitely).
// We use the fixpoint driver (apply pm until stable) and count iterations.
func TestNoOscillation(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	y := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "y", nil)

	// A collection of expressions; each should converge in ≤ 3 rewrite passes.
	exprs := []uop.UOp{
		add(a, ci(a, 0), x),
		mul(a, x, ci(a, 1)),
		sub(a, x, x),
		xor(a, add(a, x, y), add(a, x, y)),
		add(a, ci(a, 3), ci(a, 4)),
		mul(a, ci(a, 0), x),
		xor(a, xor(a, x, y), y),
	}

	for i, expr := range exprs {
		r1 := rewrite.GraphRewrite(expr, rules.Symbolic)
		r2 := rewrite.GraphRewrite(r1, rules.Symbolic)
		if r1 != r2 {
			t.Errorf("expr[%d] did not converge after 2 passes: pass1=%v pass2=%v", i, r1.Op(), r2.Op())
		}
	}
}

// ── 14. Float constant special values ────────────────────────────────────────

func TestFloatSpecialValues(t *testing.T) {
	a := arena()
	// Log2(0) → -Inf
	logZero := a.New(uop.OpLog2, uop.Dtypes.Float32, []uop.UOp{cf(a, 0)}, nil, nil)
	r := sym(logZero)
	if r.Op() != uop.OpConst {
		t.Fatal("Log2(0): expected Const")
	}
	got := r.Arg().(float64)
	if !math.IsInf(got, -1) {
		t.Errorf("Log2(0): got %v, want -Inf", got)
	}

	// Sqrt(-1) → NaN
	sqrtNeg := a.New(uop.OpSqrt, uop.Dtypes.Float32, []uop.UOp{cf(a, -1)}, nil, nil)
	r = sym(sqrtNeg)
	if r.Op() != uop.OpConst {
		t.Fatal("Sqrt(-1): expected Const")
	}
	if !math.IsNaN(r.Arg().(float64)) {
		t.Errorf("Sqrt(-1): got %v, want NaN", r.Arg())
	}
}

// ── 15. Int64 constant folding ────────────────────────────────────────────────

func TestInt64ConstFold(t *testing.T) {
	a := arena()
	// Int64 doesn't truncate on overflow the way int32 does within 64 bits.
	big := ci64(a, 1<<40)
	two := ci64(a, 2)
	node := a.New(uop.OpMul, uop.Dtypes.Int64, []uop.UOp{big, two}, nil, nil)
	r := sym(node)
	want := int64(1 << 41)
	if r.Op() != uop.OpConst || r.Arg().(int64) != want {
		t.Errorf("int64 mul: got op=%v arg=%v, want %d", r.Op(), r.Arg(), want)
	}
}

// ── Differential: generated matcher vs v0 runtime ────────────────────────────

// TestDifferentialSymbolic runs both the generated matcher (Symbolic) and the
// v0 runtime matcher (SymbolicV0) on a representative set of input graphs and
// asserts they produce bit-identical rewrites. This directly proves equivalence
// rather than inferring it from the rest of the test suite passing.
func TestDifferentialSymbolic(t *testing.T) {
	cases := []struct {
		name  string
		build func(a *uop.Arena) uop.UOp
	}{
		// ── constant folding ──
		{"fold_add_int", func(a *uop.Arena) uop.UOp { return add(a, ci(a, 7), ci(a, 3)) }},
		{"fold_mul_float", func(a *uop.Arena) uop.UOp {
			return a.New(uop.OpMul, uop.Dtypes.Float32, []uop.UOp{cf(a, 2), cf(a, 3)}, nil, nil)
		}},
		{"fold_unary_sqrt", func(a *uop.Arena) uop.UOp {
			return a.New(uop.OpSqrt, uop.Dtypes.Float32, []uop.UOp{cf(a, 4)}, nil, nil)
		}},
		// ── identity rules ──
		{"add_zero_int", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return add(a, x, ci(a, 0))
		}},
		{"mul_one_float", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Float32, nil, "x", nil)
			return a.New(uop.OpMul, uop.Dtypes.Float32, []uop.UOp{x, cf(a, 1)}, nil, nil)
		}},
		{"mul_zero_int", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return mul(a, x, ci(a, 0))
		}},
		{"idiv_neg_one", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return idiv(a, x, ci(a, -1))
		}},
		// ── boolean rules ──
		{"and_true", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
			return a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{x, cb(a, true)}, nil, nil)
		}},
		{"or_false", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
			return a.New(uop.OpOr, uop.Dtypes.Bool, []uop.UOp{x, cb(a, false)}, nil, nil)
		}},
		{"and_false_absorb", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
			return a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{x, cb(a, false)}, nil, nil)
		}},
		{"bool_mul_becomes_and", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
			y := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "y", nil)
			return a.New(uop.OpMul, uop.Dtypes.Bool, []uop.UOp{x, y}, nil, nil)
		}},
		// ── self-cancellation ──
		{"sub_self", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return sub(a, x, x)
		}},
		{"xor_self", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return xor(a, x, x)
		}},
		{"cmplt_self", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return cmplt(a, x, x)
		}},
		// ── double-XOR ──
		{"double_xor", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			y := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "y", nil)
			return xor(a, xor(a, x, y), y)
		}},
		// ── modulo idempotent ──
		{"mod_idempotent", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			y := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "y", nil)
			return mod(a, mod(a, x, y), y)
		}},
		// ── where/ternary ──
		{"where_identical", func(a *uop.Arena) uop.UOp {
			c := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "c", nil)
			v := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "v", nil)
			return a.New(uop.OpWhere, uop.Dtypes.Int32, []uop.UOp{c, v, v}, nil, nil)
		}},
		{"where_true_const", func(a *uop.Arena) uop.UOp {
			aa := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "a", nil)
			bb := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "b", nil)
			return a.New(uop.OpWhere, uop.Dtypes.Int32, []uop.UOp{cb(a, true), aa, bb}, nil, nil)
		}},
		{"where_false_const", func(a *uop.Arena) uop.UOp {
			aa := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "a", nil)
			bb := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "b", nil)
			return a.New(uop.OpWhere, uop.Dtypes.Int32, []uop.UOp{cb(a, false), aa, bb}, nil, nil)
		}},
		// ── bounds-based comparison folding ──
		{"cmplt_always_true", func(a *uop.Arena) uop.UOp {
			v := a.New(uop.OpDefineVar, uop.Dtypes.Int32, []uop.UOp{ci(a, 0), ci(a, 5)}, "x", nil)
			return cmplt(a, v, ci(a, 10))
		}},
		{"cmpne_disjoint", func(a *uop.Arena) uop.UOp { return cmpne(a, ci(a, 3), ci(a, 7)) }},
		// ── canonicalization ──
		{"canon_add_const_left", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return add(a, ci(a, 5), x) // const at src[0] → should move to src[1]
		}},
		// ── multi-rule chain ──
		{"chain_0_plus_x_times_1", func(a *uop.Arena) uop.UOp {
			x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
			return mul(a, add(a, ci(a, 0), x), ci(a, 1))
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Both matchers share the same arena so identical rewrites produce
			// the same interned UOp (same index → equality by ==).
			a := arena()
			input := tc.build(a)
			gen := rewrite.GraphRewrite(input, rules.Symbolic)
			v0 := rewrite.GraphRewrite(input, rules.SymbolicV0)
			if gen != v0 {
				t.Errorf("generated and v0 matchers diverged:\n  generated: op=%v dtype=%v arg=%v nsrc=%d\n  v0:        op=%v dtype=%v arg=%v nsrc=%d",
					gen.Op(), gen.DType(), gen.Arg(), gen.NSrc(),
					v0.Op(), v0.DType(), v0.Arg(), v0.NSrc())
			}
		})
	}
}

// ── 16. Dtype-tiebreaker regression ──────────────────────────────────────────

// TestCmpUOpDtypeTiebreaker verifies that cmpUOp's ordering between two nodes
// that share (op, arg, nsrc) but differ in dtype is determined by dtype content,
// not by arena allocation order.
//
// The old code fell through to the arena-index tiebreaker for this case, so
// the canonical operand order of CmpNe(x_int, x_float) could differ between
// builds depending on which node was allocated first. With the dtype comparison
// inserted, the order is always int32 < float32 (lower priority) regardless of
// allocation sequence.
func TestCmpUOpDtypeTiebreaker(t *testing.T) {
	// Arena A: int32 var allocated before float32 var (lower index).
	a1 := arena()
	intVar1 := a1.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "v", nil)
	floatVar1 := a1.New(uop.OpDefineVar, uop.Dtypes.Float32, nil, "v", nil)
	// forward: int at src[0], float at src[1]
	ne1fwd := sym(a1.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{intVar1, floatVar1}, nil, nil))
	// reverse: float at src[0], int at src[1] — needs a swap
	ne1rev := sym(a1.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{floatVar1, intVar1}, nil, nil))
	if ne1fwd != ne1rev {
		t.Errorf("arena A: CmpNe(int,float) and CmpNe(float,int) canonicalized to different nodes")
	}

	// Arena B: float32 var allocated before int32 var (lower index) — reversed order.
	// With the old index tiebreaker this would flip the canonical src ordering.
	a2 := arena()
	floatVar2 := a2.New(uop.OpDefineVar, uop.Dtypes.Float32, nil, "v", nil)
	intVar2 := a2.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "v", nil)
	ne2fwd := sym(a2.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{intVar2, floatVar2}, nil, nil))
	ne2rev := sym(a2.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{floatVar2, intVar2}, nil, nil))
	if ne2fwd != ne2rev {
		t.Errorf("arena B: CmpNe(int,float) and CmpNe(float,int) canonicalized to different nodes")
	}

	// The canonical form in both arenas must place the same dtype at src[0].
	// int32 has lower priority than float32, so int32 < float32 in cmpUOp order,
	// meaning the canonical node has int32-typed var at src[0].
	if ne1fwd.Src(0).DType() != uop.Dtypes.Int32 {
		t.Errorf("arena A canonical src[0]: got dtype %v, want Int32", ne1fwd.Src(0).DType())
	}
	if ne2fwd.Src(0).DType() != uop.Dtypes.Int32 {
		t.Errorf("arena B canonical src[0]: got dtype %v, want Int32", ne2fwd.Src(0).DType())
	}
}
