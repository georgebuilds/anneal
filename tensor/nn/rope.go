package nn

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── RoPE (Rotary Position Embedding) ──────────────────────────────────────────
//
// RoPE (Su et al. 2021) encodes absolute position by rotating each query/key
// vector in 2-D subspaces by an angle proportional to its position. It is the
// positional scheme used by Llama/Qwen/Gemma in place of learned absolute
// position embeddings. RoPE is applied to Q and K *after* the head split, on
// tensors of shape [B, H, T, D] (D = head dim, even).
//
// This implementation follows the HuggingFace "rotate_half" convention:
//
//	inv_freq[j] = base^(-2j/D),                 j in [0, D/2)
//	angle(p, d) = p * inv_freq[d mod (D/2)]     (first and second halves share freqs)
//	cos, sin    = cos(angle), sin(angle)        shape [T, D]
//	rotate_half(x) = concat(-x[..., D/2:], x[..., :D/2])
//	rope(x)     = x * cos + rotate_half(x) * sin
//
// The cos/sin tables are precomputed on the host (no in-graph trig op exists)
// and uploaded as fixed leaves per Forward. They are not trainable parameters.
type RoPE struct {
	HeadDim   int
	MaxSeqLen int
	Base      float64

	// cos, sin are row-major [MaxSeqLen * HeadDim] host tables; cos[p*D+d] holds
	// cos(angle(p, d)). Sliced to the current T per Apply call.
	cos, sin []float32

	dtype  *uop.DType
	device string
}

// NewRoPE precomputes the rotary cos/sin tables for positions [0, maxSeqLen)
// over a head dimension of headDim (which must be even). base is the rotary
// frequency base (10000 in the original paper and in Llama).
func NewRoPE(headDim, maxSeqLen int, base float64) *RoPE {
	if headDim <= 0 || headDim%2 != 0 {
		panic(fmt.Sprintf("nn: NewRoPE: headDim must be positive and even, got %d", headDim))
	}
	if maxSeqLen <= 0 {
		panic(fmt.Sprintf("nn: NewRoPE: maxSeqLen must be positive, got %d", maxSeqLen))
	}
	if base <= 0 {
		panic(fmt.Sprintf("nn: NewRoPE: base must be positive, got %g", base))
	}

	half := headDim / 2
	cos := make([]float32, maxSeqLen*headDim)
	sin := make([]float32, maxSeqLen*headDim)
	for p := 0; p < maxSeqLen; p++ {
		for d := 0; d < headDim; d++ {
			j := d % half // first half: d; second half: d-half
			invFreq := math.Pow(base, -float64(2*j)/float64(headDim))
			angle := float64(p) * invFreq
			cos[p*headDim+d] = float32(math.Cos(angle))
			sin[p*headDim+d] = float32(math.Sin(angle))
		}
	}
	return &RoPE{
		HeadDim:   headDim,
		MaxSeqLen: maxSeqLen,
		Base:      base,
		cos:       cos,
		sin:       sin,
		dtype:     uop.Dtypes.Float32,
		device:    "webgpu",
	}
}

// Apply rotates x (shape [B, H, T, D], D == HeadDim, T <= MaxSeqLen) by the
// precomputed rotary tables and returns the rotated tensor of the same shape.
func (r *RoPE) Apply(x *tensor.Tensor) *tensor.Tensor {
	sh := x.Shape()
	if len(sh) != 4 {
		panic(fmt.Sprintf("nn: RoPE.Apply: input must be rank 4 [B,H,T,D], got rank %d", len(sh)))
	}
	B, H, T, D := sh[0], sh[1], sh[2], sh[3]
	if int(D) != r.HeadDim {
		panic(fmt.Sprintf("nn: RoPE.Apply: head dim %d != RoPE headDim %d", D, r.HeadDim))
	}
	if T > int64(r.MaxSeqLen) {
		panic(fmt.Sprintf("nn: RoPE.Apply: T=%d exceeds maxSeqLen=%d", T, r.MaxSeqLen))
	}

	a := x.Arena()

	// Build [1,1,T,D] cos/sin leaves from the precomputed tables, then broadcast.
	n := T * D
	cosBits := make([]float32, n)
	sinBits := make([]float32, n)
	copy(cosBits, r.cos[:n])
	copy(sinBits, r.sin[:n])
	cosLeaf := tensor.NewLeaf(a, []int64{1, 1, T, D}, r.dtype, r.device)
	cosLeaf.SetData(cosBits)
	sinLeaf := tensor.NewLeaf(a, []int64{1, 1, T, D}, r.dtype, r.device)
	sinLeaf.SetData(sinBits)
	cosB := cosLeaf.Expand([]int64{B, H, T, D})
	sinB := sinLeaf.Expand([]int64{B, H, T, D})

	// rotate_half(x) = concat(-x2, x1) along the last (static) axis.
	half := D / 2
	x1 := x.Shrink([][2]int64{{0, B}, {0, H}, {0, T}, {0, half}})
	x2 := x.Shrink([][2]int64{{0, B}, {0, H}, {0, T}, {half, D}})
	rot := tensor.Concat([]*tensor.Tensor{x2.Neg(), x1}, -1)

	return x.Mul(cosB).Add(rot.Mul(sinB))
}
