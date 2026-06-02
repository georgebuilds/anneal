package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── LayerNorm ────────────────────────────────────────────────────────────────

// LayerNorm normalizes activations across the last dimension.
//
// For input x of shape [..., normalizedShape], the forward pass computes:
//
//	mu    = mean(x, axis=-1, keepdim=true)
//	xc    = x - mu
//	var   = mean(xc * xc, axis=-1, keepdim=true)
//	xhat  = xc / sqrt(var + eps)
//	y     = xhat * Weight + Bias
//
// Weight (scale, init ones) and Bias (init zeros) are learnable parameters of
// shape [normalizedShape]. They broadcast across all leading dims of x.
type LayerNorm struct {
	Weight *Parameter // shape [normalizedShape]; init ones
	Bias   *Parameter // shape [normalizedShape]; init zeros
	Eps    float32
}

// NewLayerNorm constructs a LayerNorm module over the last dimension of size
// normalizedShape. Weight is initialized to 1.0 and Bias to 0.0. Eps is added
// inside the sqrt to prevent division by zero on near-constant inputs.
func NewLayerNorm(a *uop.Arena, normalizedShape int64, eps float32) *LayerNorm {
	if normalizedShape <= 0 {
		panic(fmt.Sprintf("nn: NewLayerNorm: normalizedShape must be positive, got %d", normalizedShape))
	}
	ln := &LayerNorm{
		Weight: NewParameter(a, []int64{normalizedShape}, uop.Dtypes.Float32, "webgpu"),
		Bias:   NewParameter(a, []int64{normalizedShape}, uop.Dtypes.Float32, "webgpu"),
		Eps:    eps,
	}
	for i := range ln.Weight.Value {
		ln.Weight.Value[i] = 1.0
	}
	// Bias.Value is already zero-initialized by NewParameter.
	return ln
}

// Forward applies LayerNorm to x along the last axis. x may have any rank ≥ 1
// whose last dim equals normalizedShape. Output shape equals input shape.
//
// Implementation note: reductions over the last axis use keepdim=false followed
// by an explicit Reshape that re-adds the singleton axis. This avoids a known
// limitation in the autodiff shapeOfNode pass where OpReduceAxis drops the
// reduced axis regardless of keepdim, which then mismatches the tensor-level
// logical shape on the next broadcast.
func (ln *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := x.Rank()
	if rank < 1 {
		panic("nn: LayerNorm.Forward: input must have rank ≥ 1")
	}
	lastAxis := rank - 1

	// keepShape is x's shape with the last dim replaced by 1: used to re-add
	// the singleton axis after each keepdim=false reduction.
	xShape := x.Shape()
	keepShape := make([]int64, rank)
	copy(keepShape, xShape)
	keepShape[lastAxis] = 1

	// mu = mean(x, axis=-1) reshaped to [..., 1].
	mu := x.Mean([]int{lastAxis}, false).Reshape(keepShape)

	// xc = x - mu, shape [..., d]; broadcasts mu's last dim.
	xc := x.Sub(mu)

	// variance = mean(xc * xc, axis=-1) reshaped to [..., 1].
	variance := xc.Mul(xc).Mean([]int{lastAxis}, false).Reshape(keepShape)

	// invStd = 1 / sqrt(var + eps), shape [..., 1].
	a := x.Arena()
	epsT := tensor.FullSints(a, variance.ShapeSints(), float64(ln.Eps), x.DType(), x.Device())
	invStd := variance.Add(epsT).Sqrt().Recip()

	// xhat = xc * invStd, shape [..., d].
	xhat := xc.Mul(invStd)

	// Affine transform with broadcast over leading dims:
	// Weight, Bias have shape [d]; xhat has shape [..., d].
	out := xhat.Mul(ln.Weight.T).Add(ln.Bias.T)
	return out
}

// Params returns all trainable parameters in deterministic order.
func (ln *LayerNorm) Params() []*Parameter {
	return []*Parameter{ln.Weight, ln.Bias}
}
