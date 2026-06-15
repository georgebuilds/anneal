package rules

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// White-box unit tests for the ALU constant-fold helpers. These functions are
// the numeric core of const folding; testing them directly (rather than only
// end-to-end through the ruleset, where other simplification rules may fire
// first) pins down their exact semantics op-by-op.

// ── conversion helpers ────────────────────────────────────────────────────────

func TestAsIntConversions(t *testing.T) {
	cases := []struct {
		in     any
		want   int64
		wantOK bool
	}{
		{int64(7), 7, true},
		{true, 1, true},
		{false, 0, true},
		{float64(3.0), 0, false}, // float is not accepted by asInt
		{"x", 0, false},
	}
	for _, tc := range cases {
		got, ok := asInt(tc.in)
		if ok != tc.wantOK || (ok && got != tc.want) {
			t.Errorf("asInt(%v) = (%d,%v), want (%d,%v)", tc.in, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestAsFloatConversions(t *testing.T) {
	cases := []struct {
		in     any
		want   float64
		wantOK bool
	}{
		{float64(2.5), 2.5, true},
		{int64(4), 4.0, true},
		{true, 1.0, true},
		{false, 0.0, true},
		{"x", 0, false},
	}
	for _, tc := range cases {
		got, ok := asFloat(tc.in)
		if ok != tc.wantOK || (ok && got != tc.want) {
			t.Errorf("asFloat(%v) = (%g,%v), want (%g,%v)", tc.in, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestAsBoolConversions(t *testing.T) {
	cases := []struct {
		in     any
		want   bool
		wantOK bool
	}{
		{true, true, true},
		{false, false, true},
		{int64(0), false, true},
		{int64(5), true, true},
		{float64(0), false, true},
		{float64(1.5), true, true},
		{"x", false, false},
	}
	for _, tc := range cases {
		got, ok := asBool(tc.in)
		if ok != tc.wantOK || (ok && got != tc.want) {
			t.Errorf("asBool(%v) = (%v,%v), want (%v,%v)", tc.in, got, ok, tc.want, tc.wantOK)
		}
	}
}

// ── truncation ────────────────────────────────────────────────────────────────

func TestTruncateIntBitWidths(t *testing.T) {
	// Runtime vars dodge the vet "overflows" constant-conversion check.
	var (
		v200 int64 = 200
		v40k int64 = 40000
		v5e9 int64 = 5_000_000_000
	)
	cases := []struct {
		v     int64
		dtype *uop.DType
		want  int64
	}{
		{5, uop.Dtypes.Bool, 1},
		{0, uop.Dtypes.Bool, 0},
		{v200, uop.Dtypes.Int8, int64(int8(v200))},
		{v200, uop.Dtypes.UInt8, int64(uint8(v200))},
		{-1, uop.Dtypes.UInt8, 255},
		{v40k, uop.Dtypes.Int16, int64(int16(v40k))},
		{-1, uop.Dtypes.UInt16, 65535},
		{v5e9, uop.Dtypes.Int32, int64(int32(v5e9))},
		{-1, uop.Dtypes.UInt32, int64(uint32(0xffffffff))},
		{123456789012, uop.Dtypes.Int64, 123456789012}, // 64-bit: unchanged
	}
	for _, tc := range cases {
		if got := truncateInt(tc.v, tc.dtype); got != tc.want {
			t.Errorf("truncateInt(%d, %s) = %d, want %d", tc.v, tc.dtype, got, tc.want)
		}
	}
}

func TestTruncateFloatPrecision(t *testing.T) {
	// Float32 path rounds to f32 precision.
	v := 1.0000001
	if got := truncateFloat(v, uop.Dtypes.Float32); got != float64(float32(v)) {
		t.Errorf("truncateFloat f32 = %v, want %v", got, float64(float32(v)))
	}
	// Float64 path is identity.
	if got := truncateFloat(v, uop.Dtypes.Float64); got != v {
		t.Errorf("truncateFloat f64 = %v, want %v", got, v)
	}
}

// ── execALUFloat: every op ────────────────────────────────────────────────────

func f(v float64) any { return v }

func TestExecALUFloatOps(t *testing.T) {
	cases := []struct {
		name string
		op   uop.Op
		vals []any
		want any
	}{
		{"neg", uop.OpNeg, []any{f(2)}, -2.0},
		{"exp2", uop.OpExp2, []any{f(3)}, 8.0},
		{"log2", uop.OpLog2, []any{f(8)}, 3.0},
		{"log2 zero", uop.OpLog2, []any{f(0)}, math.Inf(-1)},
		{"sqrt", uop.OpSqrt, []any{f(9)}, 3.0},
		{"reciprocal", uop.OpReciprocal, []any{f(4)}, 0.25},
		{"trunc", uop.OpTrunc, []any{f(3.9)}, 3.0},
		{"add", uop.OpAdd, []any{f(1), f(2)}, 3.0},
		{"sub", uop.OpSub, []any{f(5), f(2)}, 3.0},
		{"mul", uop.OpMul, []any{f(3), f(4)}, 12.0},
		{"fdiv", uop.OpFDiv, []any{f(6), f(2)}, 3.0},
		{"max", uop.OpMax, []any{f(1), f(9)}, 9.0},
		{"min", uop.OpMin, []any{f(1), f(9)}, 1.0},
		{"cmplt true", uop.OpCmpLt, []any{f(1), f(2)}, true},
		{"cmpne true", uop.OpCmpNe, []any{f(1), f(2)}, true},
		{"cmpeq true", uop.OpCmpEq, []any{f(2), f(2)}, true},
		{"where true", uop.OpWhere, []any{true, f(5), f(6)}, 5.0},
		{"where false", uop.OpWhere, []any{false, f(5), f(6)}, 6.0},
		{"mulacc", uop.OpMulAcc, []any{f(2), f(3), f(4)}, 10.0},
	}
	for _, tc := range cases {
		got, ok := execALUFloat(tc.op, tc.vals)
		if !ok {
			t.Errorf("%s: execALUFloat returned ok=false", tc.name)
			continue
		}
		if got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestExecALUFloatSpecialValues(t *testing.T) {
	// log2 of negative → NaN
	if r, _ := execALUFloat(uop.OpLog2, []any{f(-1)}); !math.IsNaN(r.(float64)) {
		t.Errorf("log2(-1) = %v, want NaN", r)
	}
	// sqrt of negative → NaN
	if r, _ := execALUFloat(uop.OpSqrt, []any{f(-4)}); !math.IsNaN(r.(float64)) {
		t.Errorf("sqrt(-4) = %v, want NaN", r)
	}
	// reciprocal of 0 → +Inf
	if r, _ := execALUFloat(uop.OpReciprocal, []any{f(0)}); !math.IsInf(r.(float64), 1) {
		t.Errorf("1/0 = %v, want +Inf", r)
	}
	// sin of Inf → NaN
	if r, _ := execALUFloat(uop.OpSin, []any{f(math.Inf(1))}); !math.IsNaN(r.(float64)) {
		t.Errorf("sin(Inf) = %v, want NaN", r)
	}
	// unsupported op → ok=false
	if _, ok := execALUFloat(uop.OpAnd, []any{f(1), f(2)}); ok {
		t.Error("execALUFloat(And) should be unsupported")
	}
}

// ── execALUInt: every op ──────────────────────────────────────────────────────

func i(v int64) any { return v }

func TestExecALUIntOps(t *testing.T) {
	cases := []struct {
		name string
		op   uop.Op
		vals []any
		want any
	}{
		{"neg", uop.OpNeg, []any{i(3)}, int64(-3)},
		{"trunc identity", uop.OpTrunc, []any{i(7)}, int64(7)},
		{"add", uop.OpAdd, []any{i(1), i(2)}, int64(3)},
		{"sub", uop.OpSub, []any{i(5), i(2)}, int64(3)},
		{"mul", uop.OpMul, []any{i(3), i(4)}, int64(12)},
		{"idiv floor", uop.OpIDiv, []any{i(-7), i(2)}, int64(-4)},
		{"mod floor", uop.OpMod, []any{i(-7), i(3)}, int64(2)},
		{"max", uop.OpMax, []any{i(1), i(9)}, int64(9)},
		{"max lhs", uop.OpMax, []any{i(9), i(1)}, int64(9)},
		{"min", uop.OpMin, []any{i(1), i(9)}, int64(1)},
		{"min lhs", uop.OpMin, []any{i(0), i(9)}, int64(0)},
		{"shl", uop.OpShl, []any{i(1), i(4)}, int64(16)},
		{"shr", uop.OpShr, []any{i(16), i(2)}, int64(4)},
		{"xor", uop.OpXor, []any{i(6), i(3)}, int64(5)},
		{"or", uop.OpOr, []any{i(4), i(1)}, int64(5)},
		{"and", uop.OpAnd, []any{i(6), i(3)}, int64(2)},
		{"cmplt", uop.OpCmpLt, []any{i(1), i(2)}, true},
		{"cmpne", uop.OpCmpNe, []any{i(1), i(2)}, true},
		{"cmpeq", uop.OpCmpEq, []any{i(2), i(2)}, true},
		{"where true", uop.OpWhere, []any{true, i(5), i(6)}, int64(5)},
		{"where false", uop.OpWhere, []any{false, i(5), i(6)}, int64(6)},
		{"mulacc", uop.OpMulAcc, []any{i(2), i(3), i(4)}, int64(10)},
	}
	for _, tc := range cases {
		got, ok := execALUInt(tc.op, tc.vals)
		if !ok {
			t.Errorf("%s: execALUInt returned ok=false", tc.name)
			continue
		}
		if got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestExecALUIntDivByZero(t *testing.T) {
	if _, ok := execALUInt(uop.OpIDiv, []any{i(5), i(0)}); ok {
		t.Error("idiv by zero should return ok=false")
	}
	if _, ok := execALUInt(uop.OpMod, []any{i(5), i(0)}); ok {
		t.Error("mod by zero should return ok=false")
	}
	// unsupported op
	if _, ok := execALUInt(uop.OpExp2, []any{i(2)}); ok {
		t.Error("execALUInt(Exp2) should be unsupported")
	}
}

// ── floorDiv / floorMod sign matrix ───────────────────────────────────────────

func TestFloorDivMod(t *testing.T) {
	cases := []struct {
		a, b       int64
		wdiv, wmod int64
	}{
		{7, 2, 3, 1},
		{-7, 2, -4, 1},
		{7, -2, -4, -1},
		{-7, -2, 3, -1},
		{6, 3, 2, 0}, // exact: no floor adjustment
	}
	for _, tc := range cases {
		if d := floorDiv(tc.a, tc.b); d != tc.wdiv {
			t.Errorf("floorDiv(%d,%d) = %d, want %d", tc.a, tc.b, d, tc.wdiv)
		}
		if m := floorMod(tc.a, tc.b); m != tc.wmod {
			t.Errorf("floorMod(%d,%d) = %d, want %d", tc.a, tc.b, m, tc.wmod)
		}
		// Invariant: a == div*b + mod.
		if floorDiv(tc.a, tc.b)*tc.b+floorMod(tc.a, tc.b) != tc.a {
			t.Errorf("floor identity broke for (%d,%d)", tc.a, tc.b)
		}
	}
}

// ── foldConstALU guard paths ──────────────────────────────────────────────────

func TestFoldConstALUNonConstSrc(t *testing.T) {
	a := uop.NewArena(16)
	v := a.DefineVar("n", 0, 10) // not a Const
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	node := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{v, c}, nil, nil)
	if _, ok := foldConstALU(node); ok {
		t.Error("foldConstALU should not fold a node with a non-const src")
	}
}

func TestFoldConstALUZeroSrc(t *testing.T) {
	a := uop.NewArena(16)
	node := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(0), nil)
	if _, ok := foldConstALU(node); ok {
		t.Error("foldConstALU should not fold a zero-src node")
	}
}

func TestFoldConstALUUndefined(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(5), nil)
	z := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(0), nil)
	node := a.New(uop.OpIDiv, uop.Dtypes.Int32, []uop.UOp{x, z}, nil, nil)
	if _, ok := foldConstALU(node); ok {
		t.Error("foldConstALU should not fold idiv-by-zero")
	}
}

func TestFoldConstALUFolds(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(3), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(4), nil)
	node := a.New(uop.OpMul, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
	r, ok := foldConstALU(node)
	if !ok {
		t.Fatal("foldConstALU should fold const*const")
	}
	if r.Op() != uop.OpConst || r.Arg().(int64) != 12 {
		t.Errorf("3*4 folded to %v arg=%v, want Const 12", r.Op(), r.Arg())
	}
}
