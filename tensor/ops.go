package tensor

import (
	"github.com/georgebuilds/anneal/uop"
)

// ── Cast ──────────────────────────────────────────────────────────────────────

// Cast produces a CAST graph node converting t's dtype to dtype.
// No-op when dtypes are already identical.
func (t *Tensor) Cast(dtype *uop.DType) *Tensor {
	if t.dtype == dtype {
		return t
	}
	node := t.arena().New(uop.OpCast, dtype, []uop.UOp{t.node}, nil, nil)
	return fromNode(node, t.st, dtype, t.device)
}

// ── Unary primitives ──────────────────────────────────────────────────────────

func (t *Tensor) unary(op uop.Op) *Tensor {
	node := t.arena().New(op, t.dtype, []uop.UOp{t.node}, nil, nil)
	return fromNode(node, t.st, t.dtype, t.device)
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor { return t.unary(uop.OpNeg) }

// Exp2 returns 2^t (primitive).
func (t *Tensor) Exp2() *Tensor { return t.unary(uop.OpExp2) }

// Log2 returns log₂(t) (primitive).
func (t *Tensor) Log2() *Tensor { return t.unary(uop.OpLog2) }

// Sin returns sin(t) (primitive).
func (t *Tensor) Sin() *Tensor { return t.unary(uop.OpSin) }

// Sqrt returns √t (primitive).
func (t *Tensor) Sqrt() *Tensor { return t.unary(uop.OpSqrt) }

// Recip returns 1/t (primitive reciprocal).
func (t *Tensor) Recip() *Tensor { return t.unary(uop.OpReciprocal) }

// Trunc returns the truncated value (primitive).
func (t *Tensor) Trunc() *Tensor { return t.unary(uop.OpTrunc) }

// ── Derived unary ops ─────────────────────────────────────────────────────────

// Exp returns eˣ expressed as exp2(x / ln2)
func (t *Tensor) Exp() *Tensor {
	scale := Full(t.arena(), t.Shape(), 1.0/ln2, t.dtype, t.device)
	return t.Mul(scale).Exp2()
}

// Log returns ln(x) expressed as log2(x) * ln2.
func (t *Tensor) Log() *Tensor {
	scale := Full(t.arena(), t.Shape(), ln2, t.dtype, t.device)
	return t.Log2().Mul(scale)
}

// Abs returns |t| via maximum(t, -t).
func (t *Tensor) Abs() *Tensor { return t.Maximum(t.Neg()) }

// ── Binary ops ────────────────────────────────────────────────────────────────

// binaryOp broadcasts both operands to a common shape, promotes dtypes, and
// emits a single binary ALU node.
func (t *Tensor) binaryOp(op uop.Op, other *Tensor) *Tensor {
	dtype := uop.LeastUpperDType(t.dtype, other.dtype)
	a, b := broadcast(t.Cast(dtype), other.Cast(dtype))
	node := t.arena().New(op, dtype, []uop.UOp{a.node, b.node}, nil, nil)
	return fromNode(node, a.st, dtype, t.device)
}

// Add returns t + other with broadcasting and dtype promotion.
func (t *Tensor) Add(other *Tensor) *Tensor { return t.binaryOp(uop.OpAdd, other) }

// Sub returns t - other.
func (t *Tensor) Sub(other *Tensor) *Tensor { return t.binaryOp(uop.OpSub, other) }

// Mul returns t * other.
func (t *Tensor) Mul(other *Tensor) *Tensor { return t.binaryOp(uop.OpMul, other) }

// Div returns t / other.
// Float types: mul-reciprocal. Integer types: integer division.
func (t *Tensor) Div(other *Tensor) *Tensor {
	dtype := uop.LeastUpperDType(t.dtype, other.dtype)
	if dtype.IsFloat() {
		return t.Cast(dtype).Mul(other.Cast(dtype).Recip())
	}
	return t.binaryOp(uop.OpIDiv, other)
}

// Maximum returns element-wise max(t, other).
func (t *Tensor) Maximum(other *Tensor) *Tensor { return t.binaryOp(uop.OpMax, other) }

// CmpLt returns t < other as a bool tensor.
func (t *Tensor) CmpLt(other *Tensor) *Tensor {
	dtype := uop.LeastUpperDType(t.dtype, other.dtype)
	a, b := broadcast(t.Cast(dtype), other.Cast(dtype))
	node := t.arena().New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{a.node, b.node}, nil, nil)
	return fromNode(node, a.st, uop.Dtypes.Bool, t.device)
}

// CmpEq returns t == other as a bool tensor.
func (t *Tensor) CmpEq(other *Tensor) *Tensor {
	dtype := uop.LeastUpperDType(t.dtype, other.dtype)
	a, b := broadcast(t.Cast(dtype), other.Cast(dtype))
	node := t.arena().New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{a.node, b.node}, nil, nil)
	return fromNode(node, a.st, uop.Dtypes.Bool, t.device)
}

// Where returns a tensor selecting x where cond is true, y otherwise.
// Broadcasts all three operands to a common shape.
func Where(cond, x, y *Tensor) *Tensor {
	dtype := uop.LeastUpperDType(x.dtype, y.dtype)
	xp, yp := broadcast(x.Cast(dtype), y.Cast(dtype))
	c := BroadcastTo(cond, xp.Shape())
	node := x.arena().New(uop.OpWhere, dtype, []uop.UOp{c.node, xp.node, yp.node}, nil, nil)
	return fromNode(node, xp.st, dtype, x.device)
}
