package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Llama block + model (modern decoder stack) ────────────────────────────────
//
// Llama composes the modern small-LM primitive stack - RMSNorm, grouped-query
// attention with RoPE, and a SwiGLU feed-forward - into a decoder-only
// transformer, the architecture shared by Llama/Qwen/Gemma. It is the
// "level-up" counterpart to GPT (nn.GPT): same autoregressive shape, but with
// the four primitives that replaced LayerNorm / learned-absolute-position /
// vanilla-MHA / GELU-MLP across 2024-2026 small models.
//
//	idx [B, T]
//	  -> Tok: token embedding              -> [B, T, nEmbd]
//	  -> N x LlamaBlock (pre-RMSNorm)      -> [B, T, nEmbd]   (RoPE inside attn)
//	  -> NormF: final RMSNorm              -> [B, T, nEmbd]
//	  -> LMHead: Linear(nEmbd, vocab)      -> [B, T, vocab]   (weight tied to Tok)
//
// There is no learned position embedding: position is injected by RoPE inside
// each attention block. The LM-head weight is tied to the token embedding (the
// Llama default), so Params() returns it once.

// LlamaBlock is a pre-RMSNorm transformer block:
//
//	a = x + Attn(Norm1(x))
//	y = a + MLP(Norm2(a))
type LlamaBlock struct {
	Norm1 *RMSNorm
	Attn  *GQAttention
	Norm2 *RMSNorm
	MLP   *SwiGLU
}

// NewLlamaBlock constructs a pre-RMSNorm Llama block. rope is shared across all
// blocks (it carries no per-layer state). hidden is the SwiGLU inner width.
func NewLlamaBlock(a *uop.Arena, nEmbd, nHead, nKVHead, hidden, blockSize int, eps float32, rope *RoPE) *LlamaBlock {
	dtype := uop.Dtypes.Float32
	device := "webgpu"
	return &LlamaBlock{
		Norm1: NewRMSNorm(a, int64(nEmbd), eps),
		Attn:  NewGQAttention(a, nEmbd, nHead, nKVHead, blockSize, rope),
		Norm2: NewRMSNorm(a, int64(nEmbd), eps),
		MLP:   NewSwiGLU(a, nEmbd, hidden, dtype, device),
	}
}

// Forward computes the two-residual pre-RMSNorm block.
func (b *LlamaBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := x.Add(b.Attn.Forward(b.Norm1.Forward(x)))
	return h.Add(b.MLP.Forward(b.Norm2.Forward(h)))
}

// Params returns all trainable parameters in deterministic order:
// Norm1, Attn(Q,K,V,Proj), Norm2, MLP(Gate,Up,Down).
func (b *LlamaBlock) Params() []*Parameter {
	ps := make([]*Parameter, 0, 9)
	ps = append(ps, b.Norm1.Params()...)
	ps = append(ps, b.Attn.Params()...)
	ps = append(ps, b.Norm2.Params()...)
	ps = append(ps, b.MLP.Params()...)
	return ps
}

// Llama is the full decoder-only transformer.
type Llama struct {
	Tok    *Embedding // token embedding [vocab, nEmbd]
	Blocks []*LlamaBlock
	NormF  *RMSNorm // final RMSNorm over nEmbd
	LMHead *Linear  // language-model head nEmbd -> vocab (Weight tied to Tok)

	NLayer    int
	NHead     int
	NKVHead   int
	NEmbd     int
	Hidden    int
	BlockSize int
	Vocab     int

	// TieWeights indicates LMHead.Weight is the SAME *Parameter as Tok.Weight
	// (the Llama default). Params() returns each unique Parameter once.
	TieWeights bool
}

// NewLlama constructs a Llama model with weight tying (LMHead.Weight aliased to
// Tok.Weight). nEmbd must be divisible by nHead, and nHead by nKVHead. hidden
// is the SwiGLU inner width (see SwiGLUHidden for the Llama default). ropeBase
// is the rotary frequency base (10000 in Llama). Weights are zero-allocated;
// the caller seeds them before the first forward pass.
func NewLlama(a *uop.Arena, vocab, nLayer, nHead, nKVHead, nEmbd, hidden, blockSize int, ropeBase float64) *Llama {
	if vocab <= 0 || nLayer <= 0 || nHead <= 0 || nEmbd <= 0 || blockSize <= 0 || hidden <= 0 {
		panic(fmt.Sprintf("nn: NewLlama: all dims must be positive (vocab=%d nLayer=%d nHead=%d nEmbd=%d hidden=%d blockSize=%d)",
			vocab, nLayer, nHead, nEmbd, hidden, blockSize))
	}
	if nEmbd%nHead != 0 {
		panic(fmt.Sprintf("nn: NewLlama: nEmbd=%d not divisible by nHead=%d", nEmbd, nHead))
	}
	if nKVHead <= 0 || nHead%nKVHead != 0 {
		panic(fmt.Sprintf("nn: NewLlama: nHead=%d not divisible by nKVHead=%d", nHead, nKVHead))
	}

	const rmsEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"
	headDim := nEmbd / nHead
	rope := NewRoPE(headDim, blockSize, ropeBase)

	blocks := make([]*LlamaBlock, nLayer)
	for i := range blocks {
		blocks[i] = NewLlamaBlock(a, nEmbd, nHead, nKVHead, hidden, blockSize, rmsEps, rope)
	}

	m := &Llama{
		Tok:       NewEmbedding(a, int64(vocab), int64(nEmbd), dtype, device),
		Blocks:    blocks,
		NormF:     NewRMSNorm(a, int64(nEmbd), rmsEps),
		LMHead:    NewLinear(a, int64(nEmbd), int64(vocab), false, dtype, device),
		NLayer:    nLayer,
		NHead:     nHead,
		NKVHead:   nKVHead,
		NEmbd:     nEmbd,
		Hidden:    hidden,
		BlockSize: blockSize,
		Vocab:     vocab,
	}
	// Tie the LM head to the token embedding (both [vocab, nEmbd]).
	m.LMHead.Weight = m.Tok.Weight
	m.TieWeights = true
	return m
}

// Forward runs the full decoder stack on idx.
//
// Input  idx: [B, T] Int32, T <= BlockSize, values in [0, Vocab).
// Output:     [B, T, Vocab] logits (no softmax). idx.Data() must be populated.
func (m *Llama) Forward(idx *tensor.Tensor) *tensor.Tensor {
	if idx.Rank() != 2 {
		panic(fmt.Sprintf("nn: Llama.Forward: idx must be rank 2 [B, T], got rank %d", idx.Rank()))
	}
	if idx.DType() != uop.Dtypes.Int32 {
		panic(fmt.Sprintf("nn: Llama.Forward: idx dtype must be Int32, got %s", idx.DType()))
	}
	idxShape := idx.Shape()
	B, T := idxShape[0], idxShape[1]
	if T > int64(m.BlockSize) {
		panic(fmt.Sprintf("nn: Llama.Forward: T=%d exceeds blockSize=%d", T, m.BlockSize))
	}

	a := idx.Arena()
	device := "webgpu"

	// Token embedding via a flat [B*T] index leaf (the scatter-add gather
	// backward path requires a 1-D index), reshaped to [B, T, nEmbd].
	idxData := idx.Data()
	if idxData == nil {
		panic("nn: Llama.Forward: idx.Data() is nil; call idx.SetData(...) before Forward")
	}
	if int64(len(idxData)) != B*T {
		panic(fmt.Sprintf("nn: Llama.Forward: idx.Data() length %d != B*T=%d", len(idxData), B*T))
	}
	idxFlat := tensor.NewLeaf(a, []int64{B * T}, uop.Dtypes.Int32, device)
	flatBits := make([]float32, B*T)
	copy(flatBits, idxData)
	idxFlat.SetData(flatBits)

	tokEmbFlat := m.Tok.Forward(idxFlat)                   // [B*T, nEmbd]
	x := tokEmbFlat.Reshape([]int64{B, T, int64(m.NEmbd)}) // [B, T, nEmbd]

	// Transformer blocks (position is injected by RoPE inside each attention).
	for _, blk := range m.Blocks {
		x = blk.Forward(x)
	}

	// Final RMSNorm + tied LM head -> [B, T, vocab].
	x = m.NormF.Forward(x)
	return m.LMHead.Forward(x)
}

// Params returns all trainable parameters in deterministic order:
//
//	Tok.Weight,
//	Block[i]: Norm1, Q, K, V, Proj, Norm2, Gate, Up, Down  (9 per block),
//	NormF.Weight,
//	(LMHead.Weight is tied to Tok.Weight; not returned separately.)
func (m *Llama) Params() []*Parameter {
	ps := make([]*Parameter, 0, 9*m.NLayer+2)
	ps = append(ps, m.Tok.Params()...)
	for _, blk := range m.Blocks {
		ps = append(ps, blk.Params()...)
	}
	ps = append(ps, m.NormF.Params()...)
	// LMHead.Weight is the same *Parameter as Tok.Weight when tied; the Tok
	// path already added it. LMHead has no bias.
	if !m.TieWeights {
		ps = append(ps, m.LMHead.Params()...)
	}
	return ps
}
