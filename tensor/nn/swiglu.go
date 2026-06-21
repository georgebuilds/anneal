package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── SwiGLU feed-forward ───────────────────────────────────────────────────────

// SwiGLU is the gated feed-forward block used by Llama-style transformers
// (Shazeer 2020), replacing the GELU MLP. It uses three bias-free linears:
//
//	SwiGLU(x) = Down( SiLU(Gate(x)) * Up(x) )
//
// where Gate and Up both project nEmbd -> hidden and Down projects hidden ->
// nEmbd. The element-wise product of the SiLU-gated and ungated branches is the
// "gated linear unit"; SiLU is the swish gate. Llama omits the bias on all
// three projections.
type SwiGLU struct {
	Gate *Linear // nEmbd -> hidden
	Up   *Linear // nEmbd -> hidden
	Down *Linear // hidden -> nEmbd
}

// SwiGLUHidden returns the Llama convention for the SwiGLU hidden width:
// round (2/3 * 4 * nEmbd) up to the nearest multiple of multipleOf. The 2/3
// factor keeps the parameter count comparable to a 4*nEmbd GELU MLP despite the
// extra (gate) projection.
func SwiGLUHidden(nEmbd, multipleOf int) int {
	if multipleOf <= 0 {
		multipleOf = 1
	}
	base := (2 * 4 * nEmbd) / 3
	// round up to a multiple of multipleOf
	return ((base + multipleOf - 1) / multipleOf) * multipleOf
}

// NewSwiGLU constructs a SwiGLU feed-forward block. hidden is the inner width
// (see SwiGLUHidden for the Llama default). All three projections are bias-free.
func NewSwiGLU(a *uop.Arena, nEmbd, hidden int, dtype *uop.DType, device string) *SwiGLU {
	if nEmbd <= 0 || hidden <= 0 {
		panic(fmt.Sprintf("nn: NewSwiGLU: nEmbd and hidden must be positive (nEmbd=%d hidden=%d)", nEmbd, hidden))
	}
	return &SwiGLU{
		Gate: NewLinear(a, int64(nEmbd), int64(hidden), false, dtype, device),
		Up:   NewLinear(a, int64(nEmbd), int64(hidden), false, dtype, device),
		Down: NewLinear(a, int64(hidden), int64(nEmbd), false, dtype, device),
	}
}

// Forward computes Down(SiLU(Gate(x)) * Up(x)).
// x shape: [..., nEmbd]; output shape: [..., nEmbd].
func (m *SwiGLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	gate := SiLU(m.Gate.Forward(x))
	up := m.Up.Forward(x)
	return m.Down.Forward(gate.Mul(up))
}

// Params returns all trainable parameters in deterministic order:
// [Gate.Weight, Up.Weight, Down.Weight] (no biases).
func (m *SwiGLU) Params() []*Parameter {
	ps := make([]*Parameter, 0, 3)
	ps = append(ps, m.Gate.Params()...)
	ps = append(ps, m.Up.Params()...)
	ps = append(ps, m.Down.Params()...)
	return ps
}
