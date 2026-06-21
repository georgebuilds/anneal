package nn

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Grouped-Query Causal Self-Attention (Llama-style) ─────────────────────────
//
// GQAttention implements causal multi-head self-attention with Grouped-Query
// Attention (Ainslie et al. 2023) and Rotary Position Embeddings, as used by
// Llama/Qwen/Gemma. It differs from CausalSelfAttention in three ways:
//
//   - Separate, bias-free Q/K/V projections (not a fused QKV with bias). Q has
//     NHead heads; K and V have NKVHead heads (NKVHead divides NHead). Each KV
//     head is shared by NHead/NKVHead query heads.
//   - RoPE is applied to Q and K (per head) before the attention scores.
//   - K and V are repeated from NKVHead to NHead heads (via Expand) before the
//     scaled-dot-product, so the rest of the math is identical to dense MHA.
//
// The output projection is also bias-free. The causal mask and the
// multiplicative-mask + clamp-before-exp softmax mirror CausalSelfAttention
// exactly (see attention.go for the numerical-stability rationale).
//
// Shape conventions (D = NEmbd / NHead = head dim):
//
//	x       [B, T, NEmbd]
//	q       [B, T, NHead*D]   -> [B, NHead,   T, D]
//	k, v    [B, T, NKVHead*D] -> [B, NKVHead, T, D] -> (repeat) [B, NHead, T, D]
//	att     [B, NHead, T, T]
//	out     [B, NHead, T, D]  -> [B, T, NEmbd]
type GQAttention struct {
	Q    *Linear // NEmbd -> NHead*D   (bias-free)
	K    *Linear // NEmbd -> NKVHead*D (bias-free)
	V    *Linear // NEmbd -> NKVHead*D (bias-free)
	Proj *Linear // NHead*D -> NEmbd   (bias-free)

	rope *RoPE

	NHead     int
	NKVHead   int
	HeadDim   int
	NEmbd     int
	BlockSize int

	maskData []float32

	dtype  *uop.DType
	device string
}

// NewGQAttention constructs a grouped-query causal self-attention module.
// nEmbd must be divisible by nHead, and nHead must be divisible by nKVHead.
// rope must have been built with headDim == nEmbd/nHead and maxSeqLen >=
// blockSize; it is shared across layers (it carries no per-layer state).
func NewGQAttention(a *uop.Arena, nEmbd, nHead, nKVHead, blockSize int, rope *RoPE) *GQAttention {
	if nEmbd%nHead != 0 {
		panic(fmt.Sprintf("nn: GQAttention: nEmbd %d not divisible by nHead %d", nEmbd, nHead))
	}
	if nKVHead <= 0 || nHead%nKVHead != 0 {
		panic(fmt.Sprintf("nn: GQAttention: nHead %d not divisible by nKVHead %d", nHead, nKVHead))
	}
	if blockSize <= 0 {
		panic("nn: GQAttention: blockSize must be positive")
	}
	if rope == nil {
		panic("nn: GQAttention: rope must be non-nil")
	}
	headDim := nEmbd / nHead
	if rope.HeadDim != headDim {
		panic(fmt.Sprintf("nn: GQAttention: rope.HeadDim %d != nEmbd/nHead %d", rope.HeadDim, headDim))
	}

	dtype := uop.Dtypes.Float32
	device := "webgpu"

	att := &GQAttention{
		Q:         NewLinear(a, int64(nEmbd), int64(nHead*headDim), false, dtype, device),
		K:         NewLinear(a, int64(nEmbd), int64(nKVHead*headDim), false, dtype, device),
		V:         NewLinear(a, int64(nEmbd), int64(nKVHead*headDim), false, dtype, device),
		Proj:      NewLinear(a, int64(nHead*headDim), int64(nEmbd), false, dtype, device),
		rope:      rope,
		NHead:     nHead,
		NKVHead:   nKVHead,
		HeadDim:   headDim,
		NEmbd:     nEmbd,
		BlockSize: blockSize,
		dtype:     dtype,
		device:    device,
	}

	// Precomputed causal mask, row-major [blockSize, blockSize]: 1 on/below the
	// diagonal (keep), 0 above (drop). Applied multiplicatively post-exp.
	att.maskData = make([]float32, blockSize*blockSize)
	for i := 0; i < blockSize; i++ {
		for j := 0; j < blockSize; j++ {
			if j <= i {
				att.maskData[i*blockSize+j] = maskKeep
			} else {
				att.maskData[i*blockSize+j] = maskDrop
			}
		}
	}
	return att
}

// Forward computes grouped-query causal self-attention.
//
// Input  x: [B, T, NEmbd]    (T <= BlockSize)
// Output y: [B, T, NEmbd]
func (m *GQAttention) Forward(x *tensor.Tensor) *tensor.Tensor {
	xShape := x.Shape()
	if len(xShape) != 3 {
		panic(fmt.Sprintf("nn: GQAttention: input must be rank 3 [B,T,NEmbd], got rank %d", len(xShape)))
	}
	B, T, E := xShape[0], xShape[1], xShape[2]
	if int(E) != m.NEmbd {
		panic(fmt.Sprintf("nn: GQAttention: input nEmbd=%d != module nEmbd=%d", E, m.NEmbd))
	}
	if T > int64(m.BlockSize) {
		panic(fmt.Sprintf("nn: GQAttention: T=%d exceeds blockSize=%d", T, m.BlockSize))
	}

	a := x.Arena()
	HQ := int64(m.NHead)
	HKV := int64(m.NKVHead)
	D := int64(m.HeadDim)
	g := HQ / HKV // query heads per kv head

	// Projections (bias-free). q:[B,T,HQ*D]  k,v:[B,T,HKV*D].
	q := m.Q.Forward(x)
	k := m.K.Forward(x)
	v := m.V.Forward(x)

	// Split into heads: [B,T,H*D] -> [B,T,H,D] -> [B,H,T,D].
	splitHeads := func(t *tensor.Tensor, H int64) *tensor.Tensor {
		return t.Reshape([]int64{B, T, H, D}).Permute([]int{0, 2, 1, 3})
	}
	q = splitHeads(q, HQ)  // [B, HQ, T, D]
	k = splitHeads(k, HKV) // [B, HKV, T, D]
	v = splitHeads(v, HKV) // [B, HKV, T, D]

	// Rotary position embedding on Q and K.
	q = m.rope.Apply(q)
	k = m.rope.Apply(k)

	// Repeat KV heads from HKV to HQ: [B,HKV,T,D] -> [B,HKV,1,T,D]
	// -> Expand [B,HKV,g,T,D] -> Reshape [B,HQ,T,D]. Each kv head feeds g
	// consecutive query heads (Llama repeat_kv layout).
	repeatKV := func(t *tensor.Tensor) *tensor.Tensor {
		if g == 1 {
			return t
		}
		return t.Reshape([]int64{B, HKV, 1, T, D}).
			Expand([]int64{B, HKV, g, T, D}).
			Reshape([]int64{B, HQ, T, D})
	}
	k = repeatKV(k)
	v = repeatKV(v)

	// Scaled dot-product: att = (q @ k^T) / sqrt(D), shape [B, HQ, T, T].
	kT := k.Transpose() // [B, HQ, D, T]
	att := q.Matmul(kT)
	invSqrtD := tensor.FullSints(a, att.ShapeSints(),
		1.0/math.Sqrt(float64(D)), m.dtype, m.device)
	att = att.Mul(invSqrtD)

	// Masked softmax (multiplicative mask + clamp-before-exp; see attention.go).
	clampAtt := tensor.FullSints(a, att.ShapeSints(), 40.0, m.dtype, m.device)
	att = att.Minimum(clampAtt)

	maskData := make([]float32, T*T)
	for i := int64(0); i < T; i++ {
		for j := int64(0); j < T; j++ {
			maskData[i*T+j] = m.maskData[i*int64(m.BlockSize)+j]
		}
	}
	maskLeaf := tensor.NewLeaf(a, []int64{T, T}, m.dtype, m.device)
	maskLeaf.SetData(maskData)
	maskBroadcast := maskLeaf.Reshape([]int64{1, 1, T, T}).Expand([]int64{B, HQ, T, T})

	expv := att.Exp()
	expvMasked := expv.Mul(maskBroadcast)
	sumRed := expvMasked.Sum([]int{3}, false)
	sumExp := sumRed.Reshape([]int64{B, HQ, T, 1})
	att = expvMasked.Div(sumExp)

	// out = att @ v -> [B, HQ, T, D] -> [B, T, HQ*D] -> Proj.
	out := att.Matmul(v)
	out = out.Permute([]int{0, 2, 1, 3}).Reshape([]int64{B, T, int64(m.NHead) * D})
	return m.Proj.Forward(out)
}

// Params returns the trainable parameters in deterministic order:
// Q.Weight, K.Weight, V.Weight, Proj.Weight (all bias-free).
func (m *GQAttention) Params() []*Parameter {
	ps := make([]*Parameter, 0, 4)
	ps = append(ps, m.Q.Params()...)
	ps = append(ps, m.K.Params()...)
	ps = append(ps, m.V.Params()...)
	ps = append(ps, m.Proj.Params()...)
	return ps
}
