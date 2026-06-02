package nn

import (
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── MLP block (transformer FFN) ───────────────────────────────────────────────

// MLP is the canonical transformer feed-forward block: a linear expansion to
// 4*nEmbd, a tanh-approximant GELU activation, and a linear contraction back
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

// Forward computes FC2(gelu_tanh(FC1(x))).
// x shape: [..., nEmbd]; output shape: [..., nEmbd].
func (m *MLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := m.FC1.Forward(x)
	h = geluTanh(h)
	return m.FC2.Forward(h)
}

// Params returns all trainable parameters in deterministic order:
// [FC1.Weight, FC1.Bias, FC2.Weight, FC2.Bias].
func (m *MLP) Params() []*Parameter {
	return append(m.FC1.Params(), m.FC2.Params()...)
}

// ── tanh-approximant GELU ─────────────────────────────────────────────────────

// geluTanh implements the tanh-approximant Gaussian Error Linear Unit:
//
//	gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// This is the GPT-2 / Hendrycks-Gimpel approximation, chosen here because the
// codebase exposes Tanh as a composite (nn.Tanh) but does not yet implement
// erf, which the exact GELU requires. The approximation agrees with the exact
// erf-based GELU to within O(1e-4) over a wide range, which tests document.
func geluTanh(x *tensor.Tensor) *tensor.Tensor {
	const (
		c0 = float64(0.7978845608028654) // sqrt(2/pi)
		c1 = float64(0.044715)
	)
	a := x.Arena()
	sh := x.ShapeSints()
	dt := x.DType()
	dev := x.Device()

	half := tensor.FullSints(a, sh, 0.5, dt, dev)
	one := tensor.FullSints(a, sh, 1.0, dt, dev)
	kC0 := tensor.FullSints(a, sh, c0, dt, dev)
	kC1 := tensor.FullSints(a, sh, c1, dt, dev)

	// x^3 = x * x * x
	x2 := x.Mul(x)
	x3 := x2.Mul(x)

	// inner = c0 * (x + c1 * x^3)
	inner := kC0.Mul(x.Add(kC1.Mul(x3)))

	// gelu(x) = 0.5 * x * (1 + tanh(inner))
	return half.Mul(x).Mul(one.Add(Tanh(inner)))
}
