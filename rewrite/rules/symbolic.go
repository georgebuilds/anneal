package rules

import (
	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/uop"
)

// Symbolic is the generated matcher for arithmetic simplification, constant folding,
// and commutative canonicalization. Produced from symbolic.upat by go:generate.
// Rule ordering and semantics are identical to SymbolicV0 (verified by TestSymbolicDifferential).
var Symbolic rewrite.Matcher = symbolicGenerated

// SymbolicV0 is the v0 runtime PatternMatcher kept for differential equivalence testing.
// It is not the default path. Use Symbolic for all production rewrites.
var SymbolicV0 = buildSymbolicV0()

// buildSymbolicV0 builds the v0 PatternMatcher using UPat structs.
// This is the original buildSymbolic renamed for differential testing.
//
// Rule ordering (earlier rules win):
//  1. Constant folding — all ALU ops with all-Const sources.
//  2. Cast/Bitcast folding — cast of a single Const source.
//  3. Identity cast — Cast(x) → x when dtypes already match.
//  4. Arithmetic identities — x+0, x*1, x*0, x−0, x^0, x//1, x//−1.
//  5. Boolean neutral/absorbing elements — x&true, x|false, x&false, x|true, x&0.
//  6. Self-cancellation (same-name captures) — x−x, x^x, x%x, x//x, x<x, x!=x.
//  7. Idempotent — x|x, x&x, max(x,x).
//  8. Boolean algebra — bool*bool→bool&bool, bool+bool→bool|bool.
//  9. Double-XOR / modulo idempotent — (x^y)^y→x, (x%y)%y→x%y.
// 10. Where (ternary) — cond.where(v,v)→v; const_cond.where(a,b)→a or b.
// 11. Bound-based comparison folding — CmpLt/CmpNe when intervals resolve result.
// 12. Commutative canonicalization — sort binary operands so constants end up at
//
//	src[1], enabling identity rules on the re-processed canonical node.
func buildSymbolicV0() *rewrite.PatternMatcher {
	anyConst := rewrite.Pat(uop.OpConst)
	w := func() *rewrite.UPat { return rewrite.WildPat() }

	// ── 1. Constant folding ───────────────────────────────────────────────────

	unaryFold := rewrite.Rule{
		Pat: rewrite.Pats(aluUnaryOps()...).WithName("node").WithSrc(anyConst),
		Fn:  hFoldUnary,
	}
	binaryFold := rewrite.Rule{
		Pat: rewrite.Pats(aluBinaryOps()...).WithName("node").WithSrc(anyConst, anyConst),
		Fn:  hFoldBinary,
	}
	ternaryFold := rewrite.Rule{
		Pat: rewrite.Pats(uop.OpWhere, uop.OpMulAcc).WithName("node").WithSrc(anyConst, anyConst, anyConst),
		Fn:  hFoldTernary,
	}

	// ── 2. Cast of Const → Const(newDtype, castValue) ────────────────────────

	castConstFold := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpCast).WithName("root").WithSrc(
			rewrite.Pat(uop.OpConst).WithName("c"),
		),
		Fn: hCastConstFold,
	}

	// ── 3. Identity cast ─────────────────────────────────────────────────────

	identityCast := rewrite.Rule{
		Pat: rewrite.Pats(uop.OpCast, uop.OpBitcast).WithName("root").WithSrc(w().WithName("x")),
		Fn:  hIdentityCast,
	}

	// ── 4. Arithmetic identities ──────────────────────────────────────────────

	addZeroI := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpAdd).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(0))),
		Fn:  hReturnX,
	}
	addZeroF := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpAdd).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(float64(0))),
		Fn:  hReturnX,
	}
	mulOneI := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMul).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(1))),
		Fn:  hReturnX,
	}
	mulOneF := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMul).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(float64(1))),
		Fn:  hReturnX,
	}
	mulZeroI := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMul).WithName("node").WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(0))),
		Fn:  hMulZero,
	}
	subZeroI := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpSub).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(0))),
		Fn:  hReturnX,
	}
	subZeroF := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpSub).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(float64(0))),
		Fn:  hReturnX,
	}
	xorZero := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpXor).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(0))),
		Fn:  hReturnX,
	}
	idivOne := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpIDiv).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(1))),
		Fn:  hReturnX,
	}
	idivNegOne := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpIDiv).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(-1))),
		Fn:  hIDivNegOne,
	}

	// ── 5. Boolean neutral/absorbing elements ────────────────────────────────

	andTrue := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpAnd).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(true)),
		Fn:  hReturnX,
	}
	orFalse := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpOr).WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(false)),
		Fn:  hReturnX,
	}
	andFalse := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpAnd).WithName("node").WithSrc(w(), rewrite.Pat(uop.OpConst).WithArg(false)),
		Fn:  hAndFalse,
	}
	orTrue := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpOr).WithName("node").WithSrc(w(), rewrite.Pat(uop.OpConst).WithArg(true)),
		Fn:  hOrTrue,
	}
	andZeroInt := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpAnd).WithName("node").WithSrc(w().WithName("x"), rewrite.Pat(uop.OpConst).WithArg(int64(0))),
		Fn:  hAndZeroInt,
	}

	// ── 6. Self-cancellation (same-name captures enforce equality) ───────────

	subSelf := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpSub).WithName("node").WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hSubSelf,
	}
	xorSelf := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpXor).WithName("node").WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hXorSelf,
	}
	modSelf := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMod).WithName("node").WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hModSelf,
	}
	idivSelf := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpIDiv).WithName("node").WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hIDivSelf,
	}
	cmpLtSelf := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpCmpLt).WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hCmpSelf,
	}
	cmpNeSelf := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpCmpNe).WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hCmpSelf,
	}

	// ── 7. Idempotent: x op x → x  (OR, AND, MAX) ────────────────────────────

	idempotent := rewrite.Rule{
		Pat: rewrite.Pats(uop.OpOr, uop.OpAnd, uop.OpMax).WithSrc(w().WithName("x"), w().WithName("x")),
		Fn:  hReturnX,
	}

	// ── 8. Boolean algebra conversions ───────────────────────────────────────

	boolMul := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMul).WithName("node").WithSrc(
			w().WithName("x").WithDtype(uop.Dtypes.Bool),
			w().WithName("y").WithDtype(uop.Dtypes.Bool),
		),
		Fn: hBoolMul,
	}
	boolAdd := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpAdd).WithName("node").WithSrc(
			w().WithName("x").WithDtype(uop.Dtypes.Bool),
			w().WithName("y").WithDtype(uop.Dtypes.Bool),
		),
		Fn: hBoolAdd,
	}
	boolMax := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMax).WithName("node").WithSrc(
			w().WithName("x").WithDtype(uop.Dtypes.Bool),
			w().WithName("y").WithDtype(uop.Dtypes.Bool),
		),
		Fn: hBoolMax,
	}

	// ── 9. Double-XOR and modulo idempotent ───────────────────────────────────

	doubleXOR := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpXor).WithSrc(
			rewrite.Pat(uop.OpXor).WithSrc(w().WithName("x"), w().WithName("y")),
			w().WithName("y"),
		),
		Fn: hReturnX,
	}
	modIdem := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpMod).WithSrc(
			rewrite.Pat(uop.OpMod).WithSrc(w(), w().WithName("y")).WithName("base"),
			w().WithName("y"),
		),
		Fn: hReturnBase,
	}

	// ── 10. Where (ternary) ──────────────────────────────────────────────────

	whereIdentical := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpWhere).WithSrc(w(), w().WithName("v"), w().WithName("v")),
		Fn:  hReturnV,
	}
	whereTrue := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpWhere).WithSrc(
			rewrite.Pat(uop.OpConst).WithArg(true),
			w().WithName("a"),
			w(),
		),
		Fn: hReturnA,
	}
	whereFalse := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpWhere).WithSrc(
			rewrite.Pat(uop.OpConst).WithArg(false),
			w(),
			w().WithName("b"),
		),
		Fn: hReturnB,
	}

	// ── 1b. Bind folding ─────────────────────────────────────────────────────
	bindFold := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpBind).WithName("node").WithSrc(rewrite.Pat(uop.OpDefineVar)),
		Fn:  hBindFold,
	}

	// ── 11. Bound-based comparison folding ───────────────────────────────────

	cmpLtBounds := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpCmpLt).WithSrc(w().WithName("a"), w().WithName("b")),
		Fn:  hCmpLtBounds,
	}
	cmpNeBounds := rewrite.Rule{
		Pat: rewrite.Pat(uop.OpCmpNe).WithSrc(w().WithName("a"), w().WithName("b")),
		Fn:  hCmpNeBounds,
	}

	// ── 12. Commutative canonicalization ─────────────────────────────────────

	canonicalize := rewrite.Rule{
		Pat: rewrite.Pats(commutativeOpSlice()...).WithName("x"),
		Fn:  hCanonicalize,
	}

	return rewrite.NewPatternMatcher([]rewrite.Rule{
		unaryFold, binaryFold, ternaryFold,
		bindFold,
		castConstFold, identityCast,
		addZeroI, addZeroF,
		mulOneI, mulOneF,
		mulZeroI,
		subZeroI, subZeroF,
		xorZero,
		idivOne, idivNegOne,
		andTrue, orFalse, andFalse, orTrue, andZeroInt,
		subSelf, xorSelf, modSelf, idivSelf, cmpLtSelf, cmpNeSelf,
		idempotent,
		boolMul, boolAdd, boolMax,
		doubleXOR, modIdem,
		whereIdentical, whereTrue, whereFalse,
		cmpLtBounds, cmpNeBounds,
		canonicalize,
	})
}

// ── helpers ───────────────────────────────────────────────────────────────────

// aluUnaryOps returns all unary ALU op codes handled by execALU.
func aluUnaryOps() []uop.Op {
	return []uop.Op{
		uop.OpExp2, uop.OpLog2, uop.OpSin, uop.OpSqrt,
		uop.OpReciprocal, uop.OpNeg, uop.OpTrunc,
	}
}

// aluBinaryOps returns all binary ALU op codes handled by execALU (no ThreeFry).
func aluBinaryOps() []uop.Op {
	return []uop.Op{
		uop.OpAdd, uop.OpMul, uop.OpShl, uop.OpShr, uop.OpIDiv,
		uop.OpMax, uop.OpMod, uop.OpCmpLt, uop.OpCmpNe, uop.OpCmpEq,
		uop.OpXor, uop.OpOr, uop.OpAnd, uop.OpSub, uop.OpFDiv, uop.OpPow,
	}
}

// commutativeOpSlice converts GroupCommutative to a slice for Pats().
func commutativeOpSlice() []uop.Op {
	ops := make([]uop.Op, 0, len(uop.GroupCommutative))
	for op := range uop.GroupCommutative {
		ops = append(ops, op)
	}
	return ops
}
