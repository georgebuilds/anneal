// Code generated from symbolic.upat by rewrite/gen. DO NOT EDIT.

package rules

import (
	"math"

	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/uop"
)

// symbolicGenerated is the generated matcher, produced from symbolic.upat.
// It replaces the v0 UPat struct traversal with inlined per-op match functions.
// No reflection; captures are local variables built only on full match.
var symbolicGenerated rewrite.Matcher = newSymbolicGenerated()

// genMatcher dispatches rewrite attempts via a fixed array indexed by op.
// Array indexing is O(1) with no map allocation.
type genMatcher struct {
	fns [uop.OpCount]func(uop.UOp, any) (uop.UOp, bool)
}

func (gm *genMatcher) Rewrite(u uop.UOp, ctx any) (uop.UOp, bool) {
	if fn := gm.fns[u.Op()]; fn != nil {
		return fn(u, ctx)
	}
	return uop.UOp{}, false
}

func newSymbolicGenerated() *genMatcher {
	gm := &genMatcher{}
	gm.fns[uop.OpExp2] = genMatchExp2
	gm.fns[uop.OpLog2] = genMatchLog2
	gm.fns[uop.OpSin] = genMatchSin
	gm.fns[uop.OpSqrt] = genMatchSqrt
	gm.fns[uop.OpReciprocal] = genMatchReciprocal
	gm.fns[uop.OpNeg] = genMatchNeg
	gm.fns[uop.OpTrunc] = genMatchTrunc
	gm.fns[uop.OpAdd] = genMatchAdd
	gm.fns[uop.OpMul] = genMatchMul
	gm.fns[uop.OpShl] = genMatchShl
	gm.fns[uop.OpShr] = genMatchShr
	gm.fns[uop.OpIDiv] = genMatchIDiv
	gm.fns[uop.OpMax] = genMatchMax
	gm.fns[uop.OpMod] = genMatchMod
	gm.fns[uop.OpCmpLt] = genMatchCmpLt
	gm.fns[uop.OpCmpNe] = genMatchCmpNe
	gm.fns[uop.OpCmpEq] = genMatchCmpEq
	gm.fns[uop.OpXor] = genMatchXor
	gm.fns[uop.OpOr] = genMatchOr
	gm.fns[uop.OpAnd] = genMatchAnd
	gm.fns[uop.OpSub] = genMatchSub
	gm.fns[uop.OpFDiv] = genMatchFDiv
	gm.fns[uop.OpPow] = genMatchPow
	gm.fns[uop.OpWhere] = genMatchWhere
	gm.fns[uop.OpMulAcc] = genMatchMulAcc
	gm.fns[uop.OpCast] = genMatchCast
	gm.fns[uop.OpBitcast] = genMatchBitcast
	return gm
}

// genArgEq reports whether a == b for arg matching purposes.
// NaN float64 values compare equal when bits match (same constant → same node).
func genArgEq(a, b any) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	switch av := a.(type) {
	case int64:
		bv, ok := b.(int64)
		return ok && av == bv
	case float64:
		bv, ok := b.(float64)
		return ok && math.Float64bits(av) == math.Float64bits(bv)
	case bool:
		bv, ok := b.(bool)
		return ok && av == bv
	}
	return false
}

// ── per-op match functions ─────────────────────────────────────────────────────

func genMatchExp2(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchLog2(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchSin(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchSqrt(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchReciprocal(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchNeg(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchTrunc(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 24: => hFoldUnary
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hFoldUnary(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchAdd(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 40: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(0)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 41: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), float64(0)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 76: => hBoolAdd
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.DType() == uop.Dtypes.Bool {
			_cap_x := _s0
			_s1 := u.Src(1)
			if _s1.DType() == uop.Dtypes.Bool {
				_cap_y := _s1
				_caps := map[string]uop.UOp{"node": u, "x": _cap_x, "y": _cap_y}
				if _r, _ok := hBoolAdd(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchMul(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 42: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(1)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 43: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), float64(1)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 44: => hMulZero
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(0)) {
			_caps := map[string]uop.UOp{"node": u, "x": _cap_x}
			if _r, _ok := hMulZero(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 75: => hBoolMul
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.DType() == uop.Dtypes.Bool {
			_cap_x := _s0
			_s1 := u.Src(1)
			if _s1.DType() == uop.Dtypes.Bool {
				_cap_y := _s1
				_caps := map[string]uop.UOp{"node": u, "x": _cap_x, "y": _cap_y}
				if _r, _ok := hBoolMul(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchShl(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchShr(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchIDiv(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 48: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(1)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 49: => hIDivNegOne
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(-1)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hIDivNegOne(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 65: => hIDivSelf
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"node": u, "x": _cap_x}
			if _r, _ok := hIDivSelf(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchMax(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 71: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 77: => hBoolMax
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.DType() == uop.Dtypes.Bool {
			_cap_x := _s0
			_s1 := u.Src(1)
			if _s1.DType() == uop.Dtypes.Bool {
				_cap_y := _s1
				_caps := map[string]uop.UOp{"node": u, "x": _cap_x, "y": _cap_y}
				if _r, _ok := hBoolMax(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchMod(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 64: => hModSelf
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"node": u, "x": _cap_x}
			if _r, _ok := hModSelf(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 82: => hReturnBase
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpMod {
			_cap_base := _s0
			if _s0.NSrc() == 2 {
				_s1__s0 := _s0.Src(1)
				_cap_y := _s1__s0
				_s1 := u.Src(1)
				if _s1 == _cap_y {
					_caps := map[string]uop.UOp{"base": _cap_base, "y": _cap_y}
					if _r, _ok := hReturnBase(_caps, ctx); _ok {
						return _r, true
					}
				}
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchCmpLt(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 66: => hCmpSelf
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hCmpSelf(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 92: => hCmpLtBounds
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_a := _s0
		_s1 := u.Src(1)
		_cap_b := _s1
		_caps := map[string]uop.UOp{"a": _cap_a, "b": _cap_b}
		if _r, _ok := hCmpLtBounds(_caps, ctx); _ok {
			return _r, true
		}
	}
	return uop.UOp{}, false
}

func genMatchCmpNe(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 67: => hCmpSelf
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hCmpSelf(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 93: => hCmpNeBounds
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_a := _s0
		_s1 := u.Src(1)
		_cap_b := _s1
		_caps := map[string]uop.UOp{"a": _cap_a, "b": _cap_b}
		if _r, _ok := hCmpNeBounds(_caps, ctx); _ok {
			return _r, true
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchCmpEq(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchXor(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 47: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(0)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 63: => hXorSelf
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"node": u, "x": _cap_x}
			if _r, _ok := hXorSelf(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 81: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpXor {
			if _s0.NSrc() == 2 {
				_s0__s0 := _s0.Src(0)
				_cap_x := _s0__s0
				_s1__s0 := _s0.Src(1)
				_cap_y := _s1__s0
				_s1 := u.Src(1)
				if _s1 == _cap_y {
					_caps := map[string]uop.UOp{"x": _cap_x, "y": _cap_y}
					if _r, _ok := hReturnX(_caps, ctx); _ok {
						return _r, true
					}
				}
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchOr(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 54: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), false) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 56: => hOrTrue
	if u.NSrc() == 2 {
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), true) {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hOrTrue(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 71: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchAnd(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 53: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), true) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 55: => hAndFalse
	if u.NSrc() == 2 {
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), false) {
			_caps := map[string]uop.UOp{"node": u}
			if _r, _ok := hAndFalse(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 57: => hAndZeroInt
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(0)) {
			_caps := map[string]uop.UOp{"node": u, "x": _cap_x}
			if _r, _ok := hAndZeroInt(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 71: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 99: => hCanonicalize
	_caps := map[string]uop.UOp{"x": u}
	if _r, _ok := hCanonicalize(_caps, ctx); _ok {
		return _r, true
	}
	return uop.UOp{}, false
}

func genMatchSub(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	// Rule line 45: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), int64(0)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 46: => hReturnX
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1.Op() == uop.OpConst && genArgEq(_s1.Arg(), float64(0)) {
			_caps := map[string]uop.UOp{"x": _cap_x}
			if _r, _ok := hReturnX(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 62: => hSubSelf
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_s1 := u.Src(1)
		if _s1 == _cap_x {
			_caps := map[string]uop.UOp{"node": u, "x": _cap_x}
			if _r, _ok := hSubSelf(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchFDiv(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchPow(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 25: => hFoldBinary
	if u.NSrc() == 2 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_caps := map[string]uop.UOp{"node": u}
				if _r, _ok := hFoldBinary(_caps, ctx); _ok {
					return _r, true
				}
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchWhere(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 26: => hFoldTernary
	if u.NSrc() == 3 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_s2 := u.Src(2)
				if _s2.Op() == uop.OpConst {
					_caps := map[string]uop.UOp{"node": u}
					if _r, _ok := hFoldTernary(_caps, ctx); _ok {
						return _r, true
					}
				}
			}
		}
	}
	// Rule line 86: => hReturnV
	if u.NSrc() == 3 {
		_s1 := u.Src(1)
		_cap_v := _s1
		_s2 := u.Src(2)
		if _s2 == _cap_v {
			_caps := map[string]uop.UOp{"v": _cap_v}
			if _r, _ok := hReturnV(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 87: => hReturnA
	if u.NSrc() == 3 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst && genArgEq(_s0.Arg(), true) {
			_s1 := u.Src(1)
			_cap_a := _s1
			_caps := map[string]uop.UOp{"a": _cap_a}
			if _r, _ok := hReturnA(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 88: => hReturnB
	if u.NSrc() == 3 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst && genArgEq(_s0.Arg(), false) {
			_s2 := u.Src(2)
			_cap_b := _s2
			_caps := map[string]uop.UOp{"b": _cap_b}
			if _r, _ok := hReturnB(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchMulAcc(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 26: => hFoldTernary
	if u.NSrc() == 3 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_s1 := u.Src(1)
			if _s1.Op() == uop.OpConst {
				_s2 := u.Src(2)
				if _s2.Op() == uop.OpConst {
					_caps := map[string]uop.UOp{"node": u}
					if _r, _ok := hFoldTernary(_caps, ctx); _ok {
						return _r, true
					}
				}
			}
		}
	}
	return uop.UOp{}, false
}

func genMatchCast(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 30: => hCastConstFold
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		if _s0.Op() == uop.OpConst {
			_cap_c := _s0
			_caps := map[string]uop.UOp{"c": _cap_c, "root": u}
			if _r, _ok := hCastConstFold(_caps, ctx); _ok {
				return _r, true
			}
		}
	}
	// Rule line 34: => hIdentityCast
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_caps := map[string]uop.UOp{"root": u, "x": _cap_x}
		if _r, _ok := hIdentityCast(_caps, ctx); _ok {
			return _r, true
		}
	}
	return uop.UOp{}, false
}

func genMatchBitcast(u uop.UOp, ctx any) (uop.UOp, bool) {
	// Rule line 34: => hIdentityCast
	if u.NSrc() == 1 {
		_s0 := u.Src(0)
		_cap_x := _s0
		_caps := map[string]uop.UOp{"root": u, "x": _cap_x}
		if _r, _ok := hIdentityCast(_caps, ctx); _ok {
			return _r, true
		}
	}
	return uop.UOp{}, false
}
