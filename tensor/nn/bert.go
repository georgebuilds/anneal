package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── BERT container (bidirectional encoder + masked-LM head) ───────────────────
//
// BERT composes a token embedding, a learned absolute position embedding, a
// stack of non-causal pre-LN encoder blocks, a final LayerNorm, and a
// language-model head into the canonical BERT-style masked-language-model
// architecture (Devlin et al, 2018):
//
//	idx [B, T]
//	  -> Wte:   token embedding              -> [B, T, nEmbd]
//	  -> + PosEmb (learned [seqLen, nEmbd], sliced to T, broadcast over batch)
//	  -> N x ViTBlock (non-causal, pre-LN)   -> [B, T, nEmbd]
//	  -> LNf:   final LayerNorm              -> [B, T, nEmbd]
//	  -> Head:  Linear(nEmbd, vocab)         -> [B, T, vocab]
//
// This is pure composition over the existing kit; it adds no new primitive.
// The only architectural choice that distinguishes a BERT encoder from the GPT
// decoder is the attention mask: BERT is bidirectional, so the encoder blocks
// use ViTBlock (built on NewSelfAttention, an all-ones mask) rather than the
// causal nn.Block. The MLM training objective (mask a fraction of input tokens
// and predict them) lives example-side; the container is objective-agnostic and
// simply emits per-position vocab logits.
//
// Design calls, mirroring nn.ViT / nn.GPT:
//   - Encoder block: reuse ViTBlock (pre-LN, non-causal MHSA plus 4x erf-GELU
//     FFN plus LayerNorm). Pre-LN is the modern, more trainable variant; the
//     only structural delta from original post-LN BERT, a deliberate upgrade.
//   - Position embedding: a learned [seqLen, nEmbd] Parameter, broadcast-added
//     (the original BERT "learned absolute" choice), sliced to the actual T so
//     T <= seqLen works. Segment embeddings are omitted (single-sequence demo).
//   - MLM head: a separate Linear(nEmbd, vocab). Tying it to the token
//     embedding (real-BERT behaviour) is a proven option but kept untied here
//     for the lowest-risk gradient path, mirroring ViT.Head / the untied GPT.
type BERT struct {
	Wte    *Embedding  // token embedding   [vocab, nEmbd]
	PosEmb *Parameter  // learned absolute position embedding [seqLen, nEmbd]
	Blocks []*ViTBlock // N non-causal pre-LN encoder blocks
	LNf    *LayerNorm  // final LayerNorm over nEmbd
	Head   *Linear     // masked-LM head: nEmbd -> vocab

	NLayer    int
	NHead     int
	NEmbd     int
	BlockSize int // maximum sequence length the position embedding / mask cover
	Vocab     int
}

// NewBERT constructs a BERT model with the given configuration.
//
//   - vocab:   number of distinct input tokens (Wte first dim, Head out dim).
//     For masked-LM with a [MASK] sentinel this is baseVocab + 1.
//   - nLayer:  number of stacked encoder blocks.
//   - nHead:   attention heads per block (nEmbd must divide cleanly).
//   - nEmbd:   embedding / residual stream width.
//   - seqLen:  maximum sequence length (the position embedding rows and the
//     attention mask span; forward accepts any T <= seqLen).
//
// All weights are zero-allocated; the caller seeds them before the first
// forward pass, matching the convention used elsewhere in tensor/nn.
func NewBERT(a *uop.Arena, vocab, nLayer, nHead, nEmbd, seqLen int) *BERT {
	if vocab <= 0 || nLayer <= 0 || nHead <= 0 || nEmbd <= 0 || seqLen <= 0 {
		panic(fmt.Sprintf("nn: NewBERT: all dims must be positive (vocab=%d nLayer=%d nHead=%d nEmbd=%d seqLen=%d)",
			vocab, nLayer, nHead, nEmbd, seqLen))
	}
	if nEmbd%nHead != 0 {
		panic(fmt.Sprintf("nn: NewBERT: nEmbd=%d not divisible by nHead=%d", nEmbd, nHead))
	}

	const lnEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"

	blocks := make([]*ViTBlock, nLayer)
	for i := range blocks {
		blocks[i] = NewViTBlock(a, nEmbd, nHead, seqLen)
	}

	return &BERT{
		Wte:       NewEmbedding(a, int64(vocab), int64(nEmbd), dtype, device),
		PosEmb:    NewParameter(a, []int64{int64(seqLen), int64(nEmbd)}, dtype, device),
		Blocks:    blocks,
		LNf:       NewLayerNorm(a, int64(nEmbd), lnEps),
		Head:      NewLinear(a, int64(nEmbd), int64(vocab), true, dtype, device),
		NLayer:    nLayer,
		NHead:     nHead,
		NEmbd:     nEmbd,
		BlockSize: seqLen,
		Vocab:     vocab,
	}
}

// Forward runs the full BERT encoder stack on idx.
//
// Input  idx: [B, T] Int32, T <= BlockSize, values in [0, Vocab).
// Output:     [B, T, Vocab] (per-position logits, no softmax applied).
//
// idx.Data() must already be populated (set via SetData before Forward). As in
// GPT.Forward, the token-embedding gather backward (scatter-add) requires a 1-D
// index, so Forward flattens [B, T] to a fresh [B*T] Int32 leaf, gathers, and
// reshapes back. The position embedding is sliced to the first T rows and
// broadcast over the batch.
func (m *BERT) Forward(idx *tensor.Tensor) *tensor.Tensor {
	if idx.Rank() != 2 {
		panic(fmt.Sprintf("nn: BERT.Forward: idx must be rank 2 [B, T], got rank %d", idx.Rank()))
	}
	if idx.DType() != uop.Dtypes.Int32 {
		panic(fmt.Sprintf("nn: BERT.Forward: idx dtype must be Int32, got %s", idx.DType()))
	}
	idxShape := idx.Shape()
	B, T := idxShape[0], idxShape[1]
	if T > int64(m.BlockSize) {
		panic(fmt.Sprintf("nn: BERT.Forward: T=%d exceeds seqLen=%d", T, m.BlockSize))
	}

	a := idx.Arena()
	device := "webgpu"

	// ── Token embedding (flatten-to-[B*T] gather, GPT pattern) ────────────────
	idxData := idx.Data()
	if idxData == nil {
		panic("nn: BERT.Forward: idx.Data() is nil; call idx.SetData(...) before Forward")
	}
	if int64(len(idxData)) != B*T {
		panic(fmt.Sprintf("nn: BERT.Forward: idx.Data() length %d != B*T=%d", len(idxData), B*T))
	}
	idxFlat := tensor.NewLeaf(a, []int64{B * T}, uop.Dtypes.Int32, device)
	flatBits := make([]float32, B*T)
	copy(flatBits, idxData)
	idxFlat.SetData(flatBits)

	tokEmbFlat := m.Wte.Forward(idxFlat)                        // [B*T, nEmbd]
	tokEmb := tokEmbFlat.Reshape([]int64{B, T, int64(m.NEmbd)}) // [B, T, nEmbd]

	// ── Learned position embedding ────────────────────────────────────────────
	// Slice PosEmb [seqLen, nEmbd] to the first T rows, then broadcast-prepend a
	// batch dim to match tokEmb [B, T, nEmbd] and add.
	posSlice := m.PosEmb.T.Shrink([][2]int64{{0, T}, {0, int64(m.NEmbd)}}) // [T, nEmbd]
	posB := tensor.BroadcastToSints(posSlice, tokEmb.ShapeSints())         // [B, T, nEmbd]
	x := tokEmb.Add(posB)

	// ── Bidirectional encoder blocks ──────────────────────────────────────────
	for _, blk := range m.Blocks {
		x = blk.Forward(x)
	}

	// ── Final LayerNorm + masked-LM head ──────────────────────────────────────
	x = m.LNf.Forward(x)
	return m.Head.Forward(x) // [B, T, Vocab]
}

// Params returns all trainable parameters in deterministic order:
//
//	Wte.Weight,
//	PosEmb,
//	Block[0..L-1].(12 params each, order from ViTBlock.Params()),
//	LNf.{W,B},
//	Head.{W,B}.
//
// Total: 1 (Wte) + 1 (PosEmb) + 12 * nLayer + 2 (LNf) + 2 (Head)
//
//	= 12 * nLayer + 6.
func (m *BERT) Params() []*Parameter {
	ps := make([]*Parameter, 0, 12*m.NLayer+6)
	ps = append(ps, m.Wte.Params()...)
	ps = append(ps, m.PosEmb)
	for _, blk := range m.Blocks {
		ps = append(ps, blk.Params()...)
	}
	ps = append(ps, m.LNf.Params()...)
	ps = append(ps, m.Head.Params()...)
	return ps
}
