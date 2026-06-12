package rules_test

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// These tests drive the named symbolic handlers and their generated matchers
// (symbolic_gen.go) through the public rewrite engine via the shared sym()
// helper. Each case constructs a small UOp DAG that triggers exactly one rule
// and asserts the rewritten result, covering both the handler in
// symbolic_handlers.go and the matcher it is dispatched from.

// ── Boolean absorbing / neutral elements ──────────────────────────────────────

// hOrTrue: (Or x true) → Const(true), regardless of x.
func TestOrTrueFolds(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
	node := a.New(uop.OpOr, uop.Dtypes.Bool, []uop.UOp{x, cb(a, true)}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpConst {
		t.Fatalf("x|true: expected Const, got %v", r.Op())
	}
	if got, ok := r.Arg().(bool); !ok || got != true {
		t.Errorf("x|true: expected true, got %v (%T)", r.Arg(), r.Arg())
	}
}

// hAndZeroInt: (And x 0) → Const(0) for integer dtype (bitwise-and absorbing).
func TestAndZeroIntFolds(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	node := a.New(uop.OpAnd, uop.Dtypes.Int32, []uop.UOp{x, ci(a, 0)}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpConst {
		t.Fatalf("x&0: expected Const, got %v", r.Op())
	}
	if got, ok := r.Arg().(int64); !ok || got != 0 {
		t.Errorf("x&0: expected 0, got %v (%T)", r.Arg(), r.Arg())
	}
}

// ── Self-cancellation: mod / idiv ──────────────────────────────────────────────

// hModSelf: (Mod x x) → Const(0).
func TestModSelfFolds(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	r := sym(mod(a, x, x))
	if r.Op() != uop.OpConst {
		t.Fatalf("x%%x: expected Const, got %v", r.Op())
	}
	if got, ok := r.Arg().(int64); !ok || got != 0 {
		t.Errorf("x%%x: expected 0, got %v (%T)", r.Arg(), r.Arg())
	}
}

// hIDivSelf: (IDiv x x) → Const(1).
func TestIDivSelfFolds(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	r := sym(idiv(a, x, x))
	if r.Op() != uop.OpConst {
		t.Fatalf("x//x: expected Const, got %v", r.Op())
	}
	if got, ok := r.Arg().(int64); !ok || got != 1 {
		t.Errorf("x//x: expected 1, got %v (%T)", r.Arg(), r.Arg())
	}
}

// ── Boolean algebra: max → or ──────────────────────────────────────────────────

// hBoolMax: (Max x y) with bool srcs → (Or x y).
func TestBoolMaxBecomesOr(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "x", nil)
	y := a.New(uop.OpDefineVar, uop.Dtypes.Bool, nil, "y", nil)
	node := a.New(uop.OpMax, uop.Dtypes.Bool, []uop.UOp{x, y}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpOr {
		t.Fatalf("max(boolx,booly): expected Or, got %v", r.Op())
	}
	if r.NSrc() != 2 {
		t.Fatalf("Or should have 2 srcs, got %d", r.NSrc())
	}
}

// ── Identity cast / bitcast ────────────────────────────────────────────────────

// hIdentityCast (via genMatchBitcast): Bitcast to the same dtype → x.
func TestIdentityBitcastReturnsX(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	node := a.New(uop.OpBitcast, uop.Dtypes.Int32, []uop.UOp{x}, nil, nil)
	if r := sym(node); r != x {
		t.Errorf("bitcast<i32>(x:i32): expected x, got op=%v", r.Op())
	}
}

// hIdentityCast (via Cast matcher): Cast to the same dtype on a non-const → x.
func TestIdentityCastReturnsX(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	node := a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{x}, nil, nil)
	if r := sym(node); r != x {
		t.Errorf("cast<i32>(x:i32): expected x, got op=%v", r.Op())
	}
}

// A differing-dtype Bitcast must NOT be folded by hIdentityCast (guard branch).
func TestNonIdentityBitcastUnchanged(t *testing.T) {
	a := arena()
	x := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "x", nil)
	node := a.New(uop.OpBitcast, uop.Dtypes.Float32, []uop.UOp{x}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpBitcast {
		t.Errorf("bitcast<f32>(x:i32): expected unchanged Bitcast, got %v", r.Op())
	}
}

// ── Bind folding ───────────────────────────────────────────────────────────────

// hBindFold (via genMatchBind): Bind(DefineVar, val) → Const(val).
func TestBindFoldsToConst(t *testing.T) {
	a := arena()
	v := a.DefineVar("n", 0, 16)
	b := a.Bind(v, 7)
	r := sym(b)
	if r.Op() != uop.OpConst {
		t.Fatalf("Bind(DefineVar,7): expected Const, got %v", r.Op())
	}
	if got, ok := r.Arg().(int64); !ok || got != 7 {
		t.Errorf("Bind(DefineVar,7): expected 7, got %v (%T)", r.Arg(), r.Arg())
	}
}

// ── Generated unary/binary fold matchers ──────────────────────────────────────
// These exercise genMatchExp2/Sin/Reciprocal/Trunc/CmpEq/FDiv/MulAcc by feeding
// const operands so foldConstALU collapses them to a single Const.

func TestUnaryFloatFolds(t *testing.T) {
	tests := []struct {
		name string
		op   uop.Op
		in   float64
		want float64
	}{
		{"exp2", uop.OpExp2, 3, 8},           // 2^3
		{"sin0", uop.OpSin, 0, 0},            // sin(0)
		{"recip", uop.OpReciprocal, 4, 0.25}, // 1/4
		{"trunc", uop.OpTrunc, 2.9, 2},       // trunc(2.9)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			node := a.New(tc.op, uop.Dtypes.Float32, []uop.UOp{cf(a, tc.in)}, nil, nil)
			r := sym(node)
			if r.Op() != uop.OpConst {
				t.Fatalf("%s: expected Const, got %v", tc.name, r.Op())
			}
			got, ok := r.Arg().(float64)
			if !ok {
				t.Fatalf("%s: arg %T, want float64", tc.name, r.Arg())
			}
			if got != tc.want {
				t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestFDivFolds(t *testing.T) {
	a := arena()
	node := a.New(uop.OpFDiv, uop.Dtypes.Float32, []uop.UOp{cf(a, 9), cf(a, 3)}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpConst {
		t.Fatalf("9/3: expected Const, got %v", r.Op())
	}
	if got := r.Arg().(float64); got != 3 {
		t.Errorf("9/3: got %v, want 3", got)
	}
}

func TestCmpEqFolds(t *testing.T) {
	tests := []struct {
		name string
		x, y float64
		want bool
	}{
		{"eq", 5, 5, true},
		{"ne", 5, 6, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			node := a.New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{cf(a, tc.x), cf(a, tc.y)}, nil, nil)
			r := sym(node)
			if r.Op() != uop.OpConst {
				t.Fatalf("%s: expected Const, got %v", tc.name, r.Op())
			}
			if got := r.Arg().(bool); got != tc.want {
				t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// genMatchMulAcc: MulAcc(a,b,c) on consts → a*b+c.
func TestMulAccFolds(t *testing.T) {
	a := arena()
	node := a.New(uop.OpMulAcc, uop.Dtypes.Float32,
		[]uop.UOp{cf(a, 2), cf(a, 3), cf(a, 4)}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpConst {
		t.Fatalf("mulacc(2,3,4): expected Const, got %v", r.Op())
	}
	if got := r.Arg().(float64); got != 10 {
		t.Errorf("mulacc(2,3,4): got %v, want 10", got)
	}
}

// genMatchPow: the Pow matcher fires on two const operands. Pow is not handled
// by execALU, so foldConstALU returns false and the node is left unchanged —
// this still exercises the generated matcher path.
func TestPowMatcherLeavesNodeWhenUnfoldable(t *testing.T) {
	a := arena()
	node := a.New(uop.OpPow, uop.Dtypes.Float32, []uop.UOp{cf(a, 2), cf(a, 3)}, nil, nil)
	r := sym(node)
	if r.Op() != uop.OpPow {
		t.Errorf("pow(2,3): expected unchanged Pow, got %v", r.Op())
	}
}

// ── Cast(Const) folding ────────────────────────────────────────────────────────
// hCastConstFold drives castValue, which fans out into asFloat/asBool/asInt and
// truncateInt/truncateFloat. Each case casts a Const of fromDtype to toDtype and
// asserts the folded Const arg, covering the conversion + truncation matrix.

func constOf(a *uop.Arena, dt *uop.DType, v any) uop.UOp {
	return a.New(uop.OpConst, dt, nil, v, nil)
}

func TestCastConstFoldMatrix(t *testing.T) {
	// Runtime variables so narrowing conversions are not constant expressions
	// (which would trip the vet "overflows" check at compile time).
	var (
		v200 int64 = 200
		v40k int64 = 40000
		v5e9 int64 = 5_000_000_000
	)
	tests := []struct {
		name string
		from *uop.DType
		to   *uop.DType
		in   any
		want any
	}{
		// → bool
		{"f32→bool nonzero", uop.Dtypes.Float32, uop.Dtypes.Bool, float64(2.5), true},
		{"f32→bool zero", uop.Dtypes.Float32, uop.Dtypes.Bool, float64(0), false},
		{"i32→bool nonzero", uop.Dtypes.Int32, uop.Dtypes.Bool, int64(5), true},
		{"i32→bool zero", uop.Dtypes.Int32, uop.Dtypes.Bool, int64(0), false},
		// → float
		{"f64→f32 roundtrip", uop.Dtypes.Float64, uop.Dtypes.Float32, float64(1.5), float64(float32(1.5))},
		{"bool→f32 true", uop.Dtypes.Bool, uop.Dtypes.Float32, true, float64(1)},
		{"bool→f32 false", uop.Dtypes.Bool, uop.Dtypes.Float32, false, float64(0)},
		{"i32→f32", uop.Dtypes.Int32, uop.Dtypes.Float32, int64(7), float64(7)},
		// → int (truncation matrix)
		{"f32→i32 trunc", uop.Dtypes.Float32, uop.Dtypes.Int32, float64(3.9), int64(3)},
		{"i64→i8 wrap", uop.Dtypes.Int64, uop.Dtypes.Int8, v200, int64(int8(v200))},
		{"i64→u8 wrap", uop.Dtypes.Int64, uop.Dtypes.UInt8, int64(-1), int64(uint8(0xff))},
		{"i64→i16 wrap", uop.Dtypes.Int64, uop.Dtypes.Int16, v40k, int64(int16(v40k))},
		{"i64→u16 wrap", uop.Dtypes.Int64, uop.Dtypes.UInt16, int64(-1), int64(uint16(0xffff))},
		{"i64→i32 wrap", uop.Dtypes.Int64, uop.Dtypes.Int32, v5e9, int64(int32(v5e9))},
		{"i64→u32 wrap", uop.Dtypes.Int64, uop.Dtypes.UInt32, int64(-1), int64(uint32(0xffffffff))},
		{"i32→i64 widen", uop.Dtypes.Int32, uop.Dtypes.Int64, int64(123), int64(123)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			src := constOf(a, tc.from, tc.in)
			node := a.New(uop.OpCast, tc.to, []uop.UOp{src}, nil, nil)
			r := sym(node)
			if r.Op() != uop.OpConst {
				t.Fatalf("%s: expected Const, got %v", tc.name, r.Op())
			}
			if r.Arg() != tc.want {
				t.Errorf("%s: got %v (%T), want %v (%T)", tc.name, r.Arg(), r.Arg(), tc.want, tc.want)
			}
		})
	}
}

// ── Bound-based CmpNe folding ──────────────────────────────────────────────────
// hCmpNeBounds: a!=b where ranges prove always-unequal → true; where both sides
// are the same single point → false.

func TestCmpNeBoundsFolds(t *testing.T) {
	t.Run("disjoint ranges → true", func(t *testing.T) {
		a := arena()
		lo := a.DefineVar("lo", 0, 3)  // [0,3]
		hi := a.DefineVar("hi", 5, 10) // [5,10]
		r := sym(cmpne(a, lo, hi))
		if r.Op() != uop.OpConst || r.Arg().(bool) != true {
			t.Errorf("disjoint a!=b: expected Const(true), got op=%v arg=%v", r.Op(), r.Arg())
		}
	})
	t.Run("same point → false", func(t *testing.T) {
		a := arena()
		p1 := a.DefineVar("p1", 4, 4)
		p2 := a.DefineVar("p2", 4, 4)
		r := sym(cmpne(a, p1, p2))
		if r.Op() != uop.OpConst || r.Arg().(bool) != false {
			t.Errorf("same-point a!=b: expected Const(false), got op=%v arg=%v", r.Op(), r.Arg())
		}
	})
}
