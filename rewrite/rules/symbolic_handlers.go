package rules

import (
	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/uop"
)

// Named handler functions for symbolic rules.
// Both the v0 PatternMatcher (buildSymbolicV0) and the generated matcher call these.
// Signature matches rewrite.MatchFn: func(map[string]uop.UOp, any) (uop.UOp, bool).

func hReturnX(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	if x, ok := c["x"]; ok {
		return x, true
	}
	return uop.UOp{}, false
}

func hReturnV(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	if v, ok := c["v"]; ok {
		return v, true
	}
	return uop.UOp{}, false
}

func hReturnA(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	if a, ok := c["a"]; ok {
		return a, true
	}
	return uop.UOp{}, false
}

func hReturnB(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	if b, ok := c["b"]; ok {
		return b, true
	}
	return uop.UOp{}, false
}

func hReturnBase(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	if base, ok := c["base"]; ok {
		return base, true
	}
	return uop.UOp{}, false
}

func hFoldUnary(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return foldConstALU(c["node"])
}

func hFoldBinary(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return foldConstALU(c["node"])
}

func hFoldTernary(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return foldConstALU(c["node"])
}

func hCastConstFold(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	root, src := c["root"], c["c"]
	v := castValue(src.Arg(), src.DType(), root.DType())
	return root.Arena().New(uop.OpConst, root.DType(), nil, v, nil), true
}

func hIdentityCast(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	root, x := c["root"], c["x"]
	if root.DType() == x.DType() {
		return x, true
	}
	return uop.UOp{}, false
}

func hMulZero(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	node := c["node"]
	if node.DType().IsInt() || node.DType().IsBool() {
		return constLike(node, int64(0)), true
	}
	return uop.UOp{}, false
}

func hIDivNegOne(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	x := c["x"]
	return x.Arena().New(uop.OpNeg, x.DType(), []uop.UOp{x}, nil, nil), true
}

func hAndFalse(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	node := c["node"]
	return node.Arena().New(uop.OpConst, node.DType(), nil, false, nil), true
}

func hOrTrue(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	node := c["node"]
	return node.Arena().New(uop.OpConst, node.DType(), nil, true, nil), true
}

func hAndZeroInt(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	node := c["node"]
	if node.DType().IsInt() {
		return constLike(node, int64(0)), true
	}
	return uop.UOp{}, false
}

func hSubSelf(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return constLike(c["node"], int64(0)), true
}

func hXorSelf(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return constLike(c["node"], int64(0)), true
}

func hModSelf(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return constLike(c["node"], int64(0)), true
}

func hIDivSelf(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	return constLike(c["node"], int64(1)), true
}

func hCmpSelf(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	x := c["x"]
	return x.Arena().New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil), true
}

func hBoolMul(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	x, y, node := c["x"], c["y"], c["node"]
	return x.Arena().New(uop.OpAnd, node.DType(), []uop.UOp{x, y}, nil, nil), true
}

func hBoolAdd(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	x, y, node := c["x"], c["y"], c["node"]
	return x.Arena().New(uop.OpOr, node.DType(), []uop.UOp{x, y}, nil, nil), true
}

func hBoolMax(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	x, y, node := c["x"], c["y"], c["node"]
	return x.Arena().New(uop.OpOr, node.DType(), []uop.UOp{x, y}, nil, nil), true
}

func hCmpLtBounds(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	a, b := c["a"], c["b"]
	ba, bb := BoundsOf(a), BoundsOf(b)
	if !ba.Valid || !bb.Valid {
		return uop.UOp{}, false
	}
	if ba.Max < bb.Min {
		return a.Arena().New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil), true
	}
	if ba.Min >= bb.Max {
		return a.Arena().New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil), true
	}
	return uop.UOp{}, false
}

func hCmpNeBounds(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	a, b := c["a"], c["b"]
	ba, bb := BoundsOf(a), BoundsOf(b)
	if !ba.Valid || !bb.Valid {
		return uop.UOp{}, false
	}
	neverEqual := (ba.Max < bb.Min) || (bb.Max < ba.Min)
	alwaysEqual := ba.Min == ba.Max && ba.Min == bb.Min && bb.Min == bb.Max
	if neverEqual {
		return a.Arena().New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil), true
	}
	if alwaysEqual {
		return a.Arena().New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil), true
	}
	return uop.UOp{}, false
}

func hCanonicalize(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
	x := c["x"]
	if x.NSrc() != 2 {
		return uop.UOp{}, false
	}
	s0, s1 := x.Src(0), x.Src(1)
	if cmpUOp(s1, s0) >= 0 {
		return uop.UOp{}, false
	}
	return x.Arena().New(x.Op(), x.DType(), []uop.UOp{s1, s0}, x.Arg(), x.Tag()), true
}

// handlerTable maps handler names (as used in .upat files) to their MatchFn implementations.
// Used by the v0 PatternMatcher builder; the generated matcher calls handlers directly.
var handlerTable = map[string]rewrite.MatchFn{
	"hReturnX":      hReturnX,
	"hReturnV":      hReturnV,
	"hReturnA":      hReturnA,
	"hReturnB":      hReturnB,
	"hReturnBase":   hReturnBase,
	"hFoldUnary":    hFoldUnary,
	"hFoldBinary":   hFoldBinary,
	"hFoldTernary":  hFoldTernary,
	"hCastConstFold": hCastConstFold,
	"hIdentityCast": hIdentityCast,
	"hMulZero":      hMulZero,
	"hIDivNegOne":   hIDivNegOne,
	"hAndFalse":     hAndFalse,
	"hOrTrue":       hOrTrue,
	"hAndZeroInt":   hAndZeroInt,
	"hSubSelf":      hSubSelf,
	"hXorSelf":      hXorSelf,
	"hModSelf":      hModSelf,
	"hIDivSelf":     hIDivSelf,
	"hCmpSelf":      hCmpSelf,
	"hBoolMul":      hBoolMul,
	"hBoolAdd":      hBoolAdd,
	"hBoolMax":      hBoolMax,
	"hCmpLtBounds":  hCmpLtBounds,
	"hCmpNeBounds":  hCmpNeBounds,
	"hCanonicalize": hCanonicalize,
}
