package codegen

import (
	"math"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── aluExpr: exercise every supported branch ─────────────────────────────────

func TestAluExprAllOps(t *testing.T) {
	a := []string{"a", "b", "c"}
	cases := []struct {
		op   uop.Op
		want string
	}{
		{uop.OpExp2, "exp2(a)"},
		{uop.OpLog2, "log2(a)"},
		{uop.OpSin, "sin(a)"},
		{uop.OpSqrt, "sqrt(a)"},
		{uop.OpReciprocal, "(1.0 / a)"},
		{uop.OpNeg, "(-a)"},
		{uop.OpTrunc, "trunc(a)"},
		{uop.OpErf, "erf_anneal(a)"},
		{uop.OpAdd, "(a + b)"},
		{uop.OpSub, "(a - b)"},
		{uop.OpMul, "(a * b)"},
		{uop.OpFDiv, "(a / b)"},
		{uop.OpIDiv, "(a / b)"},
		{uop.OpMod, "(a % b)"},
		{uop.OpMax, "max(a, b)"},
		{uop.OpMin, "min(a, b)"},
		{uop.OpShl, "(a << u32(b))"},
		{uop.OpShr, "(a >> u32(b))"},
		{uop.OpAnd, "(a & b)"},
		{uop.OpOr, "(a | b)"},
		{uop.OpXor, "(a ^ b)"},
		{uop.OpCmpLt, "(a < b)"},
		{uop.OpCmpNe, "(a != b)"},
		{uop.OpCmpEq, "(a == b)"},
		{uop.OpPow, "pow(a, b)"},
		{uop.OpWhere, "select(c, b, a)"},
		{uop.OpMulAcc, "(a + (b * c))"},
	}
	for _, c := range cases {
		if got := aluExpr(c.op, a, uop.Dtypes.Float32); got != c.want {
			t.Errorf("aluExpr(%s)=%q, want %q", c.op, got, c.want)
		}
	}
}

func TestAluExprCastAndBitcast(t *testing.T) {
	srcs := []string{"v"}
	if got := aluExpr(uop.OpCast, srcs, uop.Dtypes.Float32); !strings.Contains(got, "(v)") {
		t.Errorf("Cast expr = %q", got)
	}
	if got := aluExpr(uop.OpBitcast, srcs, uop.Dtypes.Int32); !strings.HasPrefix(got, "bitcast<") {
		t.Errorf("Bitcast expr = %q", got)
	}
}

func TestAluExprUnknownPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unknown ALU op")
		}
	}()
	_ = aluExpr(uop.OpRange, []string{}, uop.Dtypes.Float32)
}

// ── constLiteral: NaN / +Inf / -Inf, ints, bools, f16 variants ───────────────

func newArena() *uop.Arena { return uop.NewArena(64) }

func TestConstLiteralFloat32Plain(t *testing.T) {
	a := newArena()
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2.5), nil)
	got := constLiteral(u)
	if !strings.Contains(got, "2.5") {
		t.Errorf("got %q", got)
	}
}

func TestConstLiteralFloat32Integral(t *testing.T) {
	a := newArena()
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3), nil)
	got := constLiteral(u)
	if got != "3.0" {
		t.Errorf("got %q, want '3.0' (trailing .0 forced)", got)
	}
}

func TestConstLiteralFloat32Inf(t *testing.T) {
	a := newArena()
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, math.Inf(1), nil)
	got := constLiteral(u)
	if got != "bitcast<f32>(0x7f7fffffu)" {
		t.Errorf("+Inf f32 = %q", got)
	}
	un := a.New(uop.OpConst, uop.Dtypes.Float32, nil, math.Inf(-1), nil)
	if g := constLiteral(un); g != "bitcast<f32>(0xff7fffffu)" {
		t.Errorf("-Inf f32 = %q", g)
	}
}

func TestConstLiteralFloat32NaN(t *testing.T) {
	a := newArena()
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, math.NaN(), nil)
	got := constLiteral(u)
	if got != "bitcast<f32>(0x7fc00000u)" {
		t.Errorf("NaN f32 = %q", got)
	}
}

func TestConstLiteralFloat16(t *testing.T) {
	a := newArena()
	cases := []struct {
		v    float64
		want string
	}{
		{2.5, "f16(2.5)"},
		{math.Inf(1), "bitcast<f16>(0x7BFFu)"},
		{math.Inf(-1), "bitcast<f16>(0xFBFFu)"},
		{math.NaN(), "bitcast<f16>(0x7E00u)"},
	}
	for _, c := range cases {
		u := a.New(uop.OpConst, uop.Dtypes.Float16, nil, c.v, nil)
		if got := constLiteral(u); got != c.want {
			t.Errorf("f16(%v) = %q, want %q", c.v, got, c.want)
		}
	}
}

func TestConstLiteralInt(t *testing.T) {
	a := newArena()
	u := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(42), nil)
	if got := constLiteral(u); got != "42" {
		t.Errorf("int const = %q", got)
	}
}

func TestConstLiteralBool(t *testing.T) {
	a := newArena()
	tu := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	fu := a.New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil)
	if got := constLiteral(tu); got != "true" {
		t.Errorf("true const = %q", got)
	}
	if got := constLiteral(fu); got != "false" {
		t.Errorf("false const = %q", got)
	}
}

func TestConstLiteralUnknown(t *testing.T) {
	a := newArena()
	// nil arg falls through to "0"
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, nil, nil)
	if got := constLiteral(u); got != "0" {
		t.Errorf("nil-arg const = %q, want '0'", got)
	}
}

// ── reduceIdentity: each op × dtype family ───────────────────────────────────

func TestReduceIdentityAdd(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want string
	}{
		{uop.Dtypes.Float32, "0.0"},
		{uop.Dtypes.Float16, "f16(0.0)"},
		{uop.Dtypes.Int32, "0"},
	}
	for _, c := range cases {
		if got := reduceIdentity(uop.OpAdd, c.dt); got != c.want {
			t.Errorf("Add identity %s = %q, want %q", c.dt, got, c.want)
		}
	}
}

func TestReduceIdentityMul(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want string
	}{
		{uop.Dtypes.Float32, "1.0"},
		{uop.Dtypes.Float16, "f16(1.0)"},
		{uop.Dtypes.Int32, "1"},
	}
	for _, c := range cases {
		if got := reduceIdentity(uop.OpMul, c.dt); got != c.want {
			t.Errorf("Mul identity %s = %q, want %q", c.dt, got, c.want)
		}
	}
}

func TestReduceIdentityMax(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want string
	}{
		{uop.Dtypes.Float32, "bitcast<f32>(0xff7fffffu)"},
		{uop.Dtypes.Float16, "bitcast<f16>(0xFBFFu)"},
		{uop.Dtypes.UInt32, "0u"},
		{uop.Dtypes.Int32, "-2147483648"},
	}
	for _, c := range cases {
		if got := reduceIdentity(uop.OpMax, c.dt); got != c.want {
			t.Errorf("Max identity %s = %q, want %q", c.dt, got, c.want)
		}
	}
}

func TestReduceIdentityMin(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want string
	}{
		{uop.Dtypes.Float32, "bitcast<f32>(0x7f7fffffu)"},
		{uop.Dtypes.Float16, "bitcast<f16>(0x7BFFu)"},
		{uop.Dtypes.UInt32, "4294967295u"},
		{uop.Dtypes.Int32, "2147483647"},
	}
	for _, c := range cases {
		if got := reduceIdentity(uop.OpMin, c.dt); got != c.want {
			t.Errorf("Min identity %s = %q, want %q", c.dt, got, c.want)
		}
	}
}

func TestReduceIdentityFallback(t *testing.T) {
	// Unknown op falls through to additive identity by dtype family.
	if got := reduceIdentity(uop.OpXor, uop.Dtypes.Float32); got != "0.0" {
		t.Errorf("fallback f32 = %q", got)
	}
	if got := reduceIdentity(uop.OpXor, uop.Dtypes.Float16); got != "f16(0.0)" {
		t.Errorf("fallback f16 = %q", got)
	}
	if got := reduceIdentity(uop.OpXor, uop.Dtypes.Int32); got != "0" {
		t.Errorf("fallback int = %q", got)
	}
}

// ── accUpdateExpr ─────────────────────────────────────────────────────────────

func TestAccUpdateExpr(t *testing.T) {
	cases := []struct {
		op   uop.Op
		want string
	}{
		{uop.OpAdd, "acc + e"},
		{uop.OpMul, "acc * e"},
		{uop.OpMax, "max(acc, e)"},
		{uop.OpMin, "min(acc, e)"},
		{uop.OpXor, "acc + e"}, // fallthrough
	}
	for _, c := range cases {
		if got := accUpdateExpr(c.op, "acc", "e"); got != c.want {
			t.Errorf("%s: got %q, want %q", c.op, got, c.want)
		}
	}
}

// ── normalizeWGSL / beamWGSLHash ─────────────────────────────────────────────

func TestNormalizeWGSLDeterministic(t *testing.T) {
	src := `let t7: f32 = data1[r3]; let t12: f32 = sm5[r3];`
	a := normalizeWGSL(src)
	b := normalizeWGSL(src)
	if a != b {
		t.Errorf("normalizeWGSL not deterministic:\n  a=%q\n  b=%q", a, b)
	}
	if strings.Contains(a, "t7") || strings.Contains(a, "r3") || strings.Contains(a, "sm5") {
		t.Errorf("normalizeWGSL did not rewrite indices: %q", a)
	}
	if !strings.Contains(a, "_v0") || !strings.Contains(a, "_v1") {
		t.Errorf("normalizeWGSL missing _v0/_v1 placeholders: %q", a)
	}
}

func TestNormalizeWGSLStableAcrossRenames(t *testing.T) {
	// Same structural WGSL with different arena-index identifiers must
	// normalize to the same string.
	src1 := `let t1: f32 = data1[r2]; let t3: f32 = t1 + 1.0;`
	src2 := `let t10: f32 = data1[r20]; let t30: f32 = t10 + 1.0;`
	if normalizeWGSL(src1) != normalizeWGSL(src2) {
		t.Error("normalizeWGSL not stable across index renames")
	}
}

func TestBeamWGSLHashMatchesExport(t *testing.T) {
	src := `let t1: f32 = 0.0;`
	if BeamWGSLHash(src) != beamWGSLHash(src) {
		t.Error("BeamWGSLHash != beamWGSLHash")
	}
}

// ── strideAcc ─────────────────────────────────────────────────────────────────

func TestStrideAccConstOnly(t *testing.T) {
	acc := newStrideAcc()
	if !acc.isConcrete() {
		t.Error("fresh acc must be concrete")
	}
	acc2 := acc.mulConst(8).mulConst(4)
	if !acc2.isConcrete() {
		t.Error("const-only chain must remain concrete")
	}
	if got := acc2.renderU32(); got != "32u" {
		t.Errorf("renderU32 = %q, want '32u'", got)
	}
	expr, isOne := acc2.renderI32StrideFactor()
	if isOne {
		t.Error("32 should not report isOne")
	}
	if expr != "32" {
		t.Errorf("i32 factor = %q, want '32'", expr)
	}
}

func TestStrideAccConstOneRendersOne(t *testing.T) {
	acc := newStrideAcc()
	if got := acc.renderU32(); got != "1u" {
		t.Errorf("renderU32 of 1 = %q", got)
	}
	expr, isOne := acc.renderI32StrideFactor()
	if !isOne {
		t.Error("constPart=1, symPart='' should report isOne=true")
	}
	if expr != "" {
		t.Errorf("i32 factor for 1 = %q, want empty", expr)
	}
	// mulConst(1) is a no-op
	if got := acc.mulConst(1); got != acc {
		t.Error("mulConst(1) must be identity")
	}
}

func TestStrideAccSymOnly(t *testing.T) {
	acc := newStrideAcc().mulSym("params_n.n0")
	if acc.isConcrete() {
		t.Error("sym chain must not be concrete")
	}
	if got := acc.renderU32(); got != "params_n.n0" {
		t.Errorf("sym-only renderU32 = %q", got)
	}
	expr, isOne := acc.renderI32StrideFactor()
	if isOne {
		t.Error("sym factor cannot be 1")
	}
	if expr != "i32(params_n.n0)" {
		t.Errorf("sym i32 = %q", expr)
	}
}

func TestStrideAccSymCompose(t *testing.T) {
	acc := newStrideAcc().mulSym("a").mulSym("b")
	if got := acc.renderU32(); got != "a * b" {
		t.Errorf("compose sym = %q", got)
	}
}

func TestStrideAccMixed(t *testing.T) {
	acc := newStrideAcc().mulConst(4).mulSym("params_n.n0")
	if got := acc.renderU32(); got != "(params_n.n0 * 4u)" {
		t.Errorf("mixed renderU32 = %q", got)
	}
	expr, _ := acc.renderI32StrideFactor()
	if expr != "i32(params_n.n0 * 4u)" {
		t.Errorf("mixed i32 = %q", expr)
	}
}

func TestStrideAccMulConstOverflowPanics(t *testing.T) {
	acc := newStrideAcc().mulConst(1 << 40)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on int64 overflow")
		}
	}()
	_ = acc.mulConst(1 << 30) // 2^70 overflows
}

// ── boundExprFromUOp ─────────────────────────────────────────────────────────

func TestBoundExprFromUOpAllSupportedOps(t *testing.T) {
	a := uop.NewArena(32)
	dv := a.DefineVar("N", 1, 16)
	c2 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)

	cs := func(op uop.Op) uop.UOp {
		return a.New(op, uop.Dtypes.Index, []uop.UOp{dv, c2}, nil, nil)
	}
	for _, op := range []uop.Op{uop.OpAdd, uop.OpSub, uop.OpMul, uop.OpIDiv, uop.OpMod} {
		got := boundExprFromUOp(cs(op))
		if len(got.Children) != 2 {
			t.Errorf("%s: %d children, want 2", op, len(got.Children))
		}
	}
	c := boundExprFromUOp(a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(7), nil))
	if c.Const != 7 {
		t.Errorf("const child = %d, want 7", c.Const)
	}
	v := boundExprFromUOp(dv)
	if v.VarName != "N" {
		t.Errorf("var name = %q, want N", v.VarName)
	}
}

func TestBoundExprFromUOpUnsupportedPanics(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	bad := a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{x}, nil, nil)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on unsupported op")
		}
	}()
	_ = boundExprFromUOp(bad)
}

func TestBoundExprFromUOpConstFloatPanics(t *testing.T) {
	a := uop.NewArena(8)
	u := a.New(uop.OpConst, uop.Dtypes.Index, nil, float64(1), nil)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on Const(float)")
		}
	}()
	_ = boundExprFromUOp(u)
}

// ── WGSLTypeInfoFor: cover small-int / int64 / bool / ptr / nil paths ────────

func TestWGSLTypeInfoForExhaustive(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want WGSLTypeInfo
	}{
		{nil, WGSLTypeInfo{"f32", "f32", 4}},
		{uop.Dtypes.Void, WGSLTypeInfo{"f32", "f32", 4}},
		{uop.Dtypes.Float32, WGSLTypeInfo{"f32", "f32", 4}},
		{uop.Dtypes.Float16, WGSLTypeInfo{"f16", "f16", 2}},
		{uop.Dtypes.Int32, WGSLTypeInfo{"i32", "i32", 4}},
		{uop.Dtypes.UInt32, WGSLTypeInfo{"u32", "u32", 4}},
		{uop.Dtypes.Index, WGSLTypeInfo{"i32", "i32", 4}},
		{uop.Dtypes.Bool, WGSLTypeInfo{"bool", "u32", 4}}, // bool promoted to u32 in storage
		{uop.Dtypes.Int8, WGSLTypeInfo{"i32", "i32", 4}},
		{uop.Dtypes.Int16, WGSLTypeInfo{"i32", "i32", 4}},
		{uop.Dtypes.UInt8, WGSLTypeInfo{"u32", "u32", 4}},
		{uop.Dtypes.UInt16, WGSLTypeInfo{"u32", "u32", 4}},
		{uop.Dtypes.Int64, WGSLTypeInfo{"i32", "i32", 4}},
		{uop.Dtypes.UInt64, WGSLTypeInfo{"u32", "u32", 4}},
		{uop.Dtypes.BFloat16, WGSLTypeInfo{"f32", "u32", 4}}, // bf16 storage promoted to u32
	}
	for _, c := range cases {
		got := WGSLTypeInfoFor(c.dt)
		if got != c.want {
			t.Errorf("%v: got %+v, want %+v", c.dt, got, c.want)
		}
	}
}

func TestWGSLTypeInfoForPtrUnwrapsBase(t *testing.T) {
	p := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	got := WGSLTypeInfoFor(p)
	if got.Scalar != "f32" {
		t.Errorf("ptr.Scalar = %q, want f32 (unwrap base)", got.Scalar)
	}
}
