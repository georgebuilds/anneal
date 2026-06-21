package nn

import (
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── MLP block (transformer FFN) ───────────────────────────────────────────────

// MLP is the canonical transformer feed-forward block: a linear expansion to
// 4*nEmbd, an exact (erf-based) GELU activation, and a linear contraction back
// to nEmbd. The 4x expansion ratio is the GPT-2/-3 default.
type MLP struct {
	FC1 *Linear // nEmbd -> 4*nEmbd
	FC2 *Linear // 4*nEmbd -> nEmbd
}

// NewMLP constructs a transformer FFN block over nEmbd-dimensional embeddings.
// Weights are uninitialised parameter buffers; the caller seeds them before
// realize() runs. Both inner Linear layers carry a bias term, matching the
// GPT-2 reference implementation.
func NewMLP(a *uop.Arena, nEmbd int, dtype *uop.DType, device string) *MLP {
	hidden := int64(4 * nEmbd)
	return &MLP{
		FC1: NewLinear(a, int64(nEmbd), hidden, true, dtype, device),
		FC2: NewLinear(a, hidden, int64(nEmbd), true, dtype, device),
	}
}

// Forward computes FC2(gelu(FC1(x))).
// x shape: [..., nEmbd]; output shape: [..., nEmbd].
func (m *MLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := m.FC1.Forward(x)
	h = gelu(h)
	return m.FC2.Forward(h)
}

// Params returns all trainable parameters in deterministic order:
// [FC1.Weight, FC1.Bias, FC2.Weight, FC2.Bias].
func (m *MLP) Params() []*Parameter {
	return append(m.FC1.Params(), m.FC2.Params()...)
}

// ── exact (erf-based) GELU ────────────────────────────────────────────────────

// gelu implements the exact Gaussian Error Linear Unit:
//
//	gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// This uses the OpErf primitive (erf is bounded in [-1, 1]; its backward is
// (2/sqrt(pi))*exp(-x^2), which underflows harmlessly to 0 for large |x|). It
// replaces the former tanh-approximant, whose backward built tanh from
// exp-composites and whose explicit x^3 term overflowed f32 in both forward and
// backward once fine-tuning drifted activations up at GPT-2 scale (NaN). The
// exact and tanh-approximant GELUs agree to O(1e-3); the change is invisible to
// training quality but numerically stable. No x^3, no Tanh/Sigmoid dependency.
func gelu(x *tensor.Tensor) *tensor.Tensor {
	const invSqrt2 = float64(0.7071067811865476) // 1/sqrt(2)
	a := x.Arena()
	sh := x.ShapeSints()
	dt := x.DType()
	dev := x.Device()

	half := tensor.FullSints(a, sh, 0.5, dt, dev)
	one := tensor.FullSints(a, sh, 1.0, dt, dev)
	kInv := tensor.FullSints(a, sh, invSqrt2, dt, dev)

	// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
	return half.Mul(x).Mul(one.Add(x.Mul(kInv).Erf()))
}
