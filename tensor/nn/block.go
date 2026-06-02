package nn

import (
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Block (pre-LayerNorm transformer block) ──────────────────────────────────
//
// Block is the standard pre-LayerNorm transformer block as used in GPT-2
// style transformers. It composes Wave 2 Slice I/J/K modules (LayerNorm,
// CausalSelfAttention, MLP) into the canonical two-residual pattern:
//
//	x = x + attn(ln1(x))
//	x = x + mlp(ln2(x))
//
// The "pre-LN" placement of the LayerNorms (before attention and FFN, inside
// the residual additions) is the GPT-2-and-later convention and is empirically
// more stable to train than the original post-LN placement.
//
// Shape conventions:
//
//	x       [B, T, nEmbd]
//	out     [B, T, nEmbd]
//
// where nEmbd must be divisible by nHead.
type Block struct {
	LN1  *LayerNorm
	Attn *CausalSelfAttention
	LN2  *LayerNorm
	MLP  *MLP
}

// NewBlock constructs a pre-LN transformer block.
//
//   - nEmbd is the embedding dimension; it must be divisible by nHead.
//   - nHead is the number of attention heads.
//   - blockSize is the maximum sequence length supported by the causal mask.
//
// LayerNorm Weight is initialised to 1 and Bias to 0 (the standard convention).
// QKV / Proj / FC1 / FC2 weights are zero-allocated by their constructors; the
// caller seeds them before the first forward pass, matching the convention
// used by NewLinear / NewConv2d / NewMLP.
func NewBlock(a *uop.Arena, nEmbd, nHead, blockSize int) *Block {
	const lnEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"

	return &Block{
		LN1:  NewLayerNorm(a, int64(nEmbd), lnEps),
		Attn: NewCausalSelfAttention(a, nEmbd, nHead, blockSize),
		LN2:  NewLayerNorm(a, int64(nEmbd), lnEps),
		MLP:  NewMLP(a, nEmbd, dtype, device),
	}
}

// Forward computes the pre-LN transformer block.
//
//	a = x + Attn(LN1(x))
//	y = a + MLP(LN2(a))
//
// Input  x: [B, T, nEmbd]    (T <= BlockSize)
// Output y: [B, T, nEmbd]
//
// The two residual additions short-circuit a path from input to output that
// keeps gradients well-conditioned at depth. The submodules' shapes already
// match (each maps [B,T,nEmbd] -> [B,T,nEmbd]) so no explicit broadcast or
// reshape is needed at the residual joins.
func (b *Block) Forward(x *tensor.Tensor) *tensor.Tensor {
	// First sub-block: residual around (LN1 -> Attn).
	h := x.Add(b.Attn.Forward(b.LN1.Forward(x)))
	// Second sub-block: residual around (LN2 -> MLP).
	return h.Add(b.MLP.Forward(b.LN2.Forward(h)))
}

// Params returns all trainable parameters in deterministic order:
//
//	LN1.Weight, LN1.Bias,
//	Attn.QKV.Weight, Attn.QKV.Bias, Attn.Proj.Weight, Attn.Proj.Bias,
//	LN2.Weight, LN2.Bias,
//	MLP.FC1.Weight, MLP.FC1.Bias, MLP.FC2.Weight, MLP.FC2.Bias.
//
// Total: 12 parameters (LayerNorm x2 = 4, Attention QKV + Proj = 4, MLP
// FC1 + FC2 = 4).
func (b *Block) Params() []*Parameter {
	ps := make([]*Parameter, 0, 12)
	ps = append(ps, b.LN1.Params()...)
	ps = append(ps, b.Attn.Params()...)
	ps = append(ps, b.LN2.Params()...)
	ps = append(ps, b.MLP.Params()...)
	return ps
}
