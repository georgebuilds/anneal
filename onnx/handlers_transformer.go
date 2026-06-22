package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Erf ──────────────────────────────────────────────────────────────────────

// handleErf dispatches to the true elementwise Gauss error function. Required
// for erf-based GELU subgraphs (BERT, some GPT-2 exports). WGSL has no stdlib
// erf; the OpErf lowering injects a polynomial helper (Abramowitz-Stegun 7.1.26,
// max abs error ~1.5e-7).
func handleErf(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Erf")
	if err != nil {
		return nil, err
	}
	return []Value{Device(x.Erf())}, nil
}

// ── Where ────────────────────────────────────────────────────────────────────

// handleWhere implements the ternary select. cond is treated as boolean (any
// non-zero is true). Broadcasts cond, x, y to a common shape.
func handleWhere(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 3 {
		return nil, fmt.Errorf("where: expected 3 inputs (cond, x, y), got %d", len(ctx.Inputs))
	}
	for i := 0; i < 3; i++ {
		if !ctx.Inputs[i].IsDevice() {
			return nil, fmt.Errorf("where: input %d is not a device tensor", i)
		}
	}
	cond := ctx.Inputs[0].Tensor()
	x := ctx.Inputs[1].Tensor()
	y := ctx.Inputs[2].Tensor()
	// tensor.Where expects a bool cond; cast if needed.
	if cond.DType() != uop.Dtypes.Bool {
		cond = cond.Cast(uop.Dtypes.Bool)
	}
	return []Value{Device(tensor.Where(cond, x, y))}, nil
}

// ── ReduceMin ───────────────────────────────────────────────────────────────

// handleReduceMin uses the OpMin reduction primitive added in Phase 3. Same
// shape as the other reductions in handlers_reduction.go.
func handleReduceMin(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "ReduceMin")
	if err != nil {
		return nil, err
	}
	axes, err := readReduceAxes(ctx)
	if err != nil {
		return nil, fmt.Errorf("reduceMin: %w", err)
	}
	keepdim := ctx.Node.Attrs["keepdims"].Int(1) != 0
	out := reduceWithF32Accumulator(x, axes, keepdim,
		func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor {
			return t.Min(axes, keepdim)
		})
	return []Value{Device(out)}, nil
}

// ── Softmax ─────────────────────────────────────────────────────────────────

// handleSoftmax implements numerically-stable softmax over a single axis.
//
// Opset semantics:
//   - opset 1..12: axis (default 1) flattens trailing dims at axis, then
//     softmax acts over the (now last) flattened axis. We model this by
//     reshaping to [outer, inner] where inner is the product of dims from
//     `axis` to end, softmaxing the last dim, then reshaping back.
//   - opset 13+: axis (default -1) acts on the single specified axis only.
//
// Recipe (both branches use it on their target axis):
//
//	m = max(x, axis, keepdim=true)
//	e = exp(x - m)
//	s = sum(e, axis, keepdim=true)
//	y = e / s
//
// Reductions go through the f32 accumulator pattern from handlers_reduction.go
// so f16/bf16 inputs don't overflow.
func handleSoftmax(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Softmax")
	if err != nil {
		return nil, err
	}
	rank := x.Rank()
	if rank == 0 {
		return nil, fmt.Errorf("softmax: input must have rank >= 1")
	}

	// Default axis depends on opset.
	defaultAxis := int64(-1)
	if ctx.Opset > 0 && ctx.Opset < 13 {
		defaultAxis = 1
	}
	axisAttr := ctx.Node.Attrs["axis"].Int(defaultAxis)
	axis := int(axisAttr)
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("softmax: axis %d out of range for rank %d", axisAttr, rank)
	}

	if ctx.Opset > 0 && ctx.Opset < 13 {
		// Pre-opset-13: coerce to 2D [outer, inner] at axis, softmax over inner,
		// reshape back. Requires concrete dims for the flatten.
		sh := x.ShapeSints()
		outer := shape.Const(1)
		for i := 0; i < axis; i++ {
			outer = shape.Mul(outer, sh[i])
		}
		inner := shape.Const(1)
		for i := axis; i < rank; i++ {
			inner = shape.Mul(inner, sh[i])
		}
		flat := x.ReshapeSints([]shape.Sint{outer, inner})
		y := softmaxOverLastAxis(ctx, flat)
		return []Value{Device(y.ReshapeSints(sh))}, nil
	}
	// opset >= 13 or unknown: act on the single axis only.
	return []Value{Device(softmaxOverAxis(ctx, x, axis))}, nil
}

// softmaxOverLastAxis is a convenience for the opset<13 path: softmax over
// the trailing axis with keepdim-true reductions.
func softmaxOverLastAxis(ctx *HandlerCtx, x *tensor.Tensor) *tensor.Tensor {
	return softmaxOverAxis(ctx, x, x.Rank()-1)
}

// softmaxOverAxis computes the numerically-stable softmax of x over `axis`,
// preserving x's shape. Reductions use keepdim=false followed by an explicit
// Reshape (matching the nn.LayerNorm convention) so the autodiff shape tracker
// - and the cpuEval test helper - see the rank-adding step explicitly.
// Reductions go through the f32 accumulator pattern so fp16/bf16 inputs don't
// lose precision.
func softmaxOverAxis(ctx *HandlerCtx, x *tensor.Tensor, axis int) *tensor.Tensor {
	_ = ctx
	axes := []int{axis}
	xSh := x.Shape()
	keepShape := make([]int64, len(xSh))
	copy(keepShape, xSh)
	keepShape[axis] = 1
	maxT := reduceWithF32Accumulator(x, axes, false,
		func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor {
			return t.Max(axes, keepdim)
		}).Reshape(keepShape)
	shifted := x.Sub(maxT)
	expv := shifted.Exp()
	sumT := reduceWithF32Accumulator(expv, axes, false,
		func(t *tensor.Tensor, axes []int, keepdim bool) *tensor.Tensor {
			return t.Sum(axes, keepdim)
		}).Reshape(keepShape)
	return expv.Div(sumT)
}

// ── Comparison family (Less, LessOrEqual, Greater, GreaterOrEqual) ─────────

// handleLess: Less(a, b) = (a < b). Direct CmpLt.
func handleLess(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Less")
	if err != nil {
		return nil, err
	}
	return []Value{Device(a.CmpLt(b))}, nil
}

// handleGreater: Greater(a, b) = (b < a), since CmpLt is the only direct
// less-than primitive.
func handleGreater(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "Greater")
	if err != nil {
		return nil, err
	}
	return []Value{Device(b.CmpLt(a))}, nil
}

// boolNot inverts a bool tensor via 1-x (the bool tensor's 0/1 values flip).
// anneal has OpOr at the UOp level but it's bitwise, not logical-on-bool with
// a CmpEq-shaped result; the safest portable expression of !bool is the
// arithmetic identity, which is correctness-equivalent on the 0/1 domain.
func boolNot(t *tensor.Tensor) *tensor.Tensor {
	a := t.Arena()
	one := tensor.FullSints(a, t.ShapeSints(), 1.0, t.DType(), t.Device())
	return one.Sub(t)
}

// handleLessOrEqual: a <= b  ≡  !(a > b)  ≡  !(b < a).
func handleLessOrEqual(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "LessOrEqual")
	if err != nil {
		return nil, err
	}
	return []Value{Device(boolNot(b.CmpLt(a)))}, nil
}

// handleGreaterOrEqual: a >= b  ≡  !(a < b).
func handleGreaterOrEqual(ctx *HandlerCtx) ([]Value, error) {
	a, b, err := twoTensorInputs(ctx, "GreaterOrEqual")
	if err != nil {
		return nil, err
	}
	return []Value{Device(boolNot(a.CmpLt(b)))}, nil
}

// Compiler-touch: keep shape import alive.
var _ = shape.Const
