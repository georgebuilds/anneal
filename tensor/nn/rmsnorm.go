package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── RMSNorm ──────────────────────────────────────────────────────────────────

// RMSNorm is Root-Mean-Square layer normalization (Zhang & Sennrich 2019), the
// normalizer used by Llama-style transformers in place of LayerNorm. It is
// LayerNorm without the mean-subtraction and without a bias term:
//
//	rms  = sqrt(mean(x^2, axis=-1) + eps)
//	y    = (x / rms) * Weight
//
// Dropping the centering step makes RMSNorm cheaper and, empirically, just as
// stable for transformer pre-normalization. Weight (scale, init ones) is the
// only learnable parameter; it broadcasts across all leading dims of x.
type RMSNorm struct {
	Weight *Parameter // shape [normalizedShape]; init ones
	Eps    float32
}

// NewRMSNorm constructs an RMSNorm module over the last dimension of size
// normalizedShape. Weight is initialized to 1.0. Eps is added inside the sqrt
// to prevent division by zero on near-zero inputs.
func NewRMSNorm(a *uop.Arena, normalizedShape int64, eps float32) *RMSNorm {
	if normalizedShape <= 0 {
		panic(fmt.Sprintf("nn: NewRMSNorm: normalizedShape must be positive, got %d", normalizedShape))
	}
	rn := &RMSNorm{
		Weight: NewParameter(a, []int64{normalizedShape}, uop.Dtypes.Float32, "webgpu"),
		Eps:    eps,
	}
	for i := range rn.Weight.Value {
		rn.Weight.Value[i] = 1.0
	}
	return rn
}

// Forward applies RMSNorm to x along the last axis. x may have any rank ≥ 1
// whose last dim equals normalizedShape. Output shape equals input shape.
//
// Like LayerNorm.Forward, the last-axis reduction uses keepdim=false followed
// by an explicit Reshape that re-adds the singleton axis, so the autodiff
// shapeOfNode pass sees the rank-adding step (see layernorm.go for the rationale).
func (rn *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := x.Rank()
	if rank < 1 {
		panic("nn: RMSNorm.Forward: input must have rank ≥ 1")
	}
	lastAxis := rank - 1

	xShape := x.Shape()
	keepShape := make([]int64, rank)
	copy(keepShape, xShape)
	keepShape[lastAxis] = 1

	// meanSq = mean(x*x, axis=-1) reshaped to [..., 1].
	meanSq := x.Mul(x).Mean([]int{lastAxis}, false).Reshape(keepShape)

	// invRMS = 1 / sqrt(meanSq + eps), shape [..., 1].
	a := x.Arena()
	epsT := tensor.FullSints(a, meanSq.ShapeSints(), float64(rn.Eps), x.DType(), x.Device())
	invRMS := meanSq.Add(epsT).Sqrt().Recip()

	// y = (x * invRMS) * Weight (Weight broadcasts over leading dims).
	xhat := x.Mul(invRMS)
	return xhat.Mul(rn.Weight.T)
}

// Params returns the single trainable parameter (no bias).
func (rn *RMSNorm) Params() []*Parameter {
	return []*Parameter{rn.Weight}
}
