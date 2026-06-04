package nn

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── GPT container (full transformer stack) ───────────────────────────────────
//
// GPT composes the Wave 2 Slice E/I/J/K/L modules (Embedding, LayerNorm,
// CausalSelfAttention, MLP, Block) into a complete GPT-2 style transformer:
//
//	idx [B, T]
//	  -> Wte:  token embedding             -> [B, T, nEmbd]
//	  -> Wpe:  positional embedding        -> [T, nEmbd], broadcast-add
//	  -> N x Block (pre-LayerNorm)         -> [B, T, nEmbd]
//	  -> LNf:  final LayerNorm             -> [B, T, nEmbd]
//	  -> LMHead: Linear(nEmbd, vocab)      -> [B, T, vocab]
//
// Weight tying (sharing LMHead.Weight with Wte.Weight as GPT-2 does) is
// deferred to Wave 2 Slice O. Each module here owns its own parameters.
//
// Positional indices [0, 1, ..., T-1] are host-precomputed and uploaded as a
// fresh Int32 BUFFER leaf on every Forward call. Per Phase 0 outcomes in
// notes/roadmap.md, anneal has no on-graph arange / iota op; the positional
// indices and causal mask are precomputed host-side and fed in as constant
// buffers. The freshly-allocated leaf is bound to the current arena so the
// dispatch path treats it identically to any other leaf input.
//
// Input idx shape contract: GPT.Forward accepts a rank-2 [B, T] Int32 tensor
// whose data has been set via SetData (which is the universal precondition
// for any leaf that participates in a Realize call). The Embedding/Gather
// backward pipeline (Slice D scatterAdd) currently requires a 1-D index;
// Forward therefore builds a fresh 1-D [B*T] leaf from idx.Data() so the
// Wte path participates in autodiff without modifying embedding.go.
type GPT struct {
	Wte    *Embedding // token embedding   [vocab, nEmbd]
	Wpe    *Embedding // position embedding [blockSize, nEmbd]
	Blocks []*Block   // N pre-LN transformer blocks
	LNf    *LayerNorm // final LayerNorm over nEmbd
	LMHead *Linear    // language-model head: nEmbd -> vocab

	NLayer    int
	NHead     int
	NEmbd     int
	BlockSize int
	Vocab     int

	// TieWeights, when true, indicates LMHead.Weight is the SAME *Parameter
	// object as Wte.Weight (canonical GPT-2 weight tying). Params() returns
	// each unique Parameter once, so the tied count drops by one and
	// LMHead.Weight does not appear independently in the list. Forward-only
	// for this slice (Slice O); training with tied weights is deferred.
	TieWeights bool
}

// NewGPT constructs a GPT model with the given configuration.
//
//   - vocab:     number of distinct input tokens (Wte first dim, LMHead out)
//   - nLayer:    number of stacked transformer Blocks
//   - nHead:     number of attention heads per Block (nEmbd must divide cleanly)
//   - nEmbd:     embedding / residual stream width
//   - blockSize: maximum sequence length supported by Wpe / causal mask
//
// Wte, Wpe, LMHead weights and biases are zero-allocated by their constructors;
// the caller seeds them before the first forward pass, matching the convention
// used by NewLinear / NewConv2d / NewMLP / NewBlock. LNf weight is 1.0 and
// bias 0.0 per the LayerNorm convention.
func NewGPT(a *uop.Arena, vocab, nLayer, nHead, nEmbd, blockSize int) *GPT {
	if vocab <= 0 || nLayer <= 0 || nHead <= 0 || nEmbd <= 0 || blockSize <= 0 {
		panic(fmt.Sprintf("nn: NewGPT: all dims must be positive (vocab=%d nLayer=%d nHead=%d nEmbd=%d blockSize=%d)",
			vocab, nLayer, nHead, nEmbd, blockSize))
	}
	if nEmbd%nHead != 0 {
		panic(fmt.Sprintf("nn: NewGPT: nEmbd=%d not divisible by nHead=%d", nEmbd, nHead))
	}

	const lnEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"

	blocks := make([]*Block, nLayer)
	for i := range blocks {
		blocks[i] = NewBlock(a, nEmbd, nHead, blockSize)
	}

	return &GPT{
		Wte:       NewEmbedding(a, int64(vocab), int64(nEmbd), dtype, device),
		Wpe:       NewEmbedding(a, int64(blockSize), int64(nEmbd), dtype, device),
		Blocks:    blocks,
		LNf:       NewLayerNorm(a, int64(nEmbd), lnEps),
		LMHead:    NewLinear(a, int64(nEmbd), int64(vocab), true, dtype, device),
		NLayer:    nLayer,
		NHead:     nHead,
		NEmbd:     nEmbd,
		BlockSize: blockSize,
		Vocab:     vocab,
	}
}

// NewGPTWithTiedHead constructs a GPT model with the language-model head's
// Weight tied to the token-embedding Weight, matching canonical GPT-2 (the
// HuggingFace gpt2 safetensors file omits lm_head.weight entirely; on load
// the LM head reuses wte.weight). Shape-wise this is sound: both tensors
// are [vocab, nEmbd] and Linear.Forward computes x @ Weight.T which yields
// the [..., vocab] logits.
//
// Implementation: build a plain GPT via NewGPT, then alias LMHead.Weight to
// the same *Parameter object as Wte.Weight. This is a structural swap, so
// any subsequent Load(arena) on Wte.Weight automatically updates LMHead's
// view via the shared Parameter: there is one canonical Value slice.
//
// LMHead.Bias is left as the freshly-allocated zero buffer from NewLinear.
// HF GPT-2 has no bias on the LM head (the standalone Linear's Bias.Value
// is all zeros), and a zero Bias added to logits is the identity; the
// constructor's symmetry with NewLinear is preserved so existing call sites
// that inspect LMHead.Bias do not need to special-case nil.
//
// Slice O scope: forward-only. The standard Backward path would route two
// gradient sources into Wte.Weight (Embedding's scatter-add path and the
// LM head's Matmul path); supporting that cleanly is a separate slice and
// is intentionally not exercised here.
func NewGPTWithTiedHead(a *uop.Arena, vocab, nLayer, nHead, nEmbd, blockSize int) *GPT {
	g := NewGPT(a, vocab, nLayer, nHead, nEmbd, blockSize)
	// Alias the LM head's Weight to the embedding's Weight. Both have shape
	// [vocab, nEmbd] (Embedding stores [numEmbeddings, embeddingDim]; Linear
	// stores [outFeatures, inFeatures]); Linear.Forward applies a transpose
	// inside Matmul so the math reduces to wte @ x.T which is the canonical
	// tied head.
	g.LMHead.Weight = g.Wte.Weight
	g.TieWeights = true
	return g
}

// Forward runs the full transformer stack on idx.
//
// Input  idx: [B, T] Int32, T <= BlockSize, values in [0, Vocab).
// Output:     [B, T, Vocab] (logits, no softmax applied).
//
// idx.Data() must already be populated (the test driver sets it via SetData
// before Forward is called). Forward reads idx.Data() to construct the
// flattened [B*T] index leaf passed to Wte.Forward; this keeps the gather
// backward (Slice D scatterAdd) on its 1-D-index code path while still
// exposing the natural [B, T] shape contract to callers. The position
// indices [0, 1, ..., T-1] are host-precomputed and uploaded as a fresh
// [T] Int32 BUFFER leaf on every call.
func (g *GPT) Forward(idx *tensor.Tensor) *tensor.Tensor {
	if idx.Rank() != 2 {
		panic(fmt.Sprintf("nn: GPT.Forward: idx must be rank 2 [B, T], got rank %d", idx.Rank()))
	}
	if idx.DType() != uop.Dtypes.Int32 {
		panic(fmt.Sprintf("nn: GPT.Forward: idx dtype must be Int32, got %s", idx.DType()))
	}
	idxShape := idx.Shape()
	B, T := idxShape[0], idxShape[1]
	if T > int64(g.BlockSize) {
		panic(fmt.Sprintf("nn: GPT.Forward: T=%d exceeds blockSize=%d", T, g.BlockSize))
	}

	a := idx.Arena()
	device := "webgpu"

	// ── Token embedding ──────────────────────────────────────────────────────
	//
	// Slice D's scatterAdd backward requires a 1-D index leaf. We construct a
	// fresh [B*T] Int32 BUFFER leaf from idx's underlying data and call
	// Embedding.Forward on it, then reshape the [B*T, nEmbd] result to
	// [B, T, nEmbd]. This keeps Wte's gradient path on the supported
	// dim=0, 1-D-index code path without modifying embedding.go.
	idxData := idx.Data()
	if idxData == nil {
		panic("nn: GPT.Forward: idx.Data() is nil; call idx.SetData(...) before Forward")
	}
	if int64(len(idxData)) != B*T {
		panic(fmt.Sprintf("nn: GPT.Forward: idx.Data() length %d != B*T=%d", len(idxData), B*T))
	}
	idxFlat := tensor.NewLeaf(a, []int64{B * T}, uop.Dtypes.Int32, device)
	flatBits := make([]float32, B*T)
	copy(flatBits, idxData)
	idxFlat.SetData(flatBits)

	tokEmbFlat := g.Wte.Forward(idxFlat)                        // [B*T, nEmbd]
	tokEmb := tokEmbFlat.Reshape([]int64{B, T, int64(g.NEmbd)}) // [B, T, nEmbd]

	// ── Positional embedding ─────────────────────────────────────────────────
	//
	// Positional indices [0, 1, ..., T-1] are host-precomputed and uploaded
	// as a fresh [T] Int32 BUFFER leaf each call. anneal has no on-graph
	// arange / iota op (Phase 0 outcome, notes/roadmap.md); the conservative
	// per-call fresh-leaf path is the supported pattern. The leaf lives in
	// the current arena and is discarded with it after Realize.
	posBits := make([]float32, T)
	for i := int64(0); i < T; i++ {
		posBits[i] = math.Float32frombits(uint32(int32(i)))
	}
	positions := tensor.NewLeaf(a, []int64{T}, uop.Dtypes.Int32, device)
	positions.SetData(posBits)

	posEmb := g.Wpe.Forward(positions) // [T, nEmbd]

	// Broadcast posEmb [T, nEmbd] across the batch dim to match tokEmb
	// [B, T, nEmbd] and add. tokEmb already carries the [B, T, nEmbd] shape;
	// BroadcastToSints prepends a rank-1 dim and Expands B.
	posEmbB := tensor.BroadcastToSints(posEmb, tokEmb.ShapeSints())

	x := tokEmb.Add(posEmbB)

	// ── Transformer Blocks ───────────────────────────────────────────────────
	for _, blk := range g.Blocks {
		x = blk.Forward(x)
	}

	// ── Final LayerNorm + LM head ────────────────────────────────────────────
	x = g.LNf.Forward(x)

	// LMHead: [B, T, nEmbd] -> [B, T, vocab]. Linear's Matmul handles the
	// rank-3 leading dims via the existing batched-matmul path used by
	// CausalSelfAttention.
	logits := g.LMHead.Forward(x)

	return logits
}

// ForwardKVStep runs one decode step on a single new token using the KV cache.
//
// idxNew is the token id for the next position, shape [1, 1] Int32. cache must
// be a fully sized KVCache compatible with this model (NumLayers == g.NLayer,
// NumHeads == g.NHead, HeadDim == g.NEmbd / g.NHead, MaxSeqLen <= g.BlockSize).
// cache.Pos must be in [0, g.BlockSize) and identifies which Wpe row the
// positional embedding for this token is drawn from.
//
// Returns logits of shape [1, 1, Vocab] and the per-layer kNews / vNews each
// of shape [1, NHead, 1, HeadDim]. After Realize-ing all returned tensors the
// caller is expected to extract kNews[i].Data() and vNews[i].Data() and call
// cache.StoreLayerKV(i, ...) for each layer, then cache.Advance() to bump Pos
// for the next step. Calling tensor.Realize once on the full output tuple
// schedules the whole forward in a single pass (Realize is variadic).
//
// All kernel shapes depend only on the model config and the cache's MaxSeqLen,
// not on cache.Pos, so the WGSL pipeline cache reuses every kernel from step
// to step. Per-step variation is in the BUFFER payloads only.
func (g *GPT) ForwardKVStep(idxNew *tensor.Tensor, cache *KVCache) (logits *tensor.Tensor, kNews, vNews []*tensor.Tensor) {
	if idxNew.Rank() != 2 {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: idxNew must be rank 2 [1, 1], got rank %d", idxNew.Rank()))
	}
	idxShape := idxNew.Shape()
	if idxShape[0] != 1 || idxShape[1] != 1 {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: idxNew must be shape [1, 1], got %v", idxShape))
	}
	if idxNew.DType() != uop.Dtypes.Int32 {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: idxNew dtype must be Int32, got %s", idxNew.DType()))
	}
	if cache == nil {
		panic("nn: GPT.ForwardKVStep: cache is nil")
	}
	if cache.NumLayers != g.NLayer || cache.NumHeads != g.NHead || cache.HeadDim != g.NEmbd/g.NHead {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: cache geometry (NumLayers=%d NumHeads=%d HeadDim=%d) does not match model (NLayer=%d NHead=%d HeadDim=%d)",
			cache.NumLayers, cache.NumHeads, cache.HeadDim, g.NLayer, g.NHead, g.NEmbd/g.NHead))
	}
	if cache.MaxSeqLen > g.BlockSize {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: cache.MaxSeqLen=%d exceeds model BlockSize=%d", cache.MaxSeqLen, g.BlockSize))
	}
	if cache.Pos < 0 || cache.Pos >= cache.MaxSeqLen {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: cache.Pos=%d out of range [0, %d)", cache.Pos, cache.MaxSeqLen))
	}

	a := idxNew.Arena()
	device := "webgpu"

	// Token embedding for the single new token. Wte.Forward expects a 1-D
	// index leaf; build one of length 1 from idxNew.Data() and reshape the
	// [1, NEmbd] output to [1, 1, NEmbd].
	idxData := idxNew.Data()
	if idxData == nil {
		panic("nn: GPT.ForwardKVStep: idxNew.Data() is nil; call idxNew.SetData(...) before ForwardKVStep")
	}
	if len(idxData) != 1 {
		panic(fmt.Sprintf("nn: GPT.ForwardKVStep: idxNew.Data() length %d != 1", len(idxData)))
	}
	idxFlat := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Int32, device)
	idxFlatData := []float32{idxData[0]}
	idxFlat.SetData(idxFlatData)

	tokEmbFlat := g.Wte.Forward(idxFlat)                        // [1, NEmbd]
	tokEmb := tokEmbFlat.Reshape([]int64{1, 1, int64(g.NEmbd)}) // [1, 1, NEmbd]

	// Positional embedding for slot cache.Pos. Wpe.Forward also wants a 1-D
	// index leaf; build one with the single value cache.Pos.
	posIdx := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Int32, device)
	posIdx.SetData([]float32{math.Float32frombits(uint32(int32(cache.Pos)))})

	posEmbRow := g.Wpe.Forward(posIdx)                         // [1, NEmbd]
	posEmb := posEmbRow.Reshape([]int64{1, 1, int64(g.NEmbd)}) // [1, 1, NEmbd]

	x := tokEmb.Add(posEmb)

	dtype := g.Wte.Weight.dtype
	posOneHot := cache.UploadPosOneHotLeaf(a, dtype, device)
	lenMask := cache.UploadLengthMaskLeaf(a, dtype, device)

	kNews = make([]*tensor.Tensor, g.NLayer)
	vNews = make([]*tensor.Tensor, g.NLayer)
	for i, blk := range g.Blocks {
		kCache := cache.UploadKLeaf(a, i, dtype, device)
		vCache := cache.UploadVLeaf(a, i, dtype, device)
		var kNew, vNew *tensor.Tensor
		x, kNew, vNew = blk.ForwardKVStep(x, kCache, vCache, posOneHot, lenMask)
		kNews[i] = kNew
		vNews[i] = vNew
	}

	x = g.LNf.Forward(x)
	logits = g.LMHead.Forward(x)
	return logits, kNews, vNews
}

// Params returns all trainable parameters in deterministic order:
//
//	Wte.Weight,
//	Wpe.Weight,
//	Block[0].LN1.{W,B}, Block[0].QKV.{W,B}, Block[0].Proj.{W,B},
//	Block[0].LN2.{W,B}, Block[0].FC1.{W,B}, Block[0].FC2.{W,B},
//	... repeated for each block (12 params per Block) ...
//	LNf.Weight, LNf.Bias,
//	LMHead.Weight, LMHead.Bias.
//
// Total: 12 * nLayer + 6 parameters
//
//	(Wte = 1, Wpe = 1, LNf = 2, LMHead = 2, Blocks = 12 * nLayer).
func (g *GPT) Params() []*Parameter {
	ps := make([]*Parameter, 0, 12*g.NLayer+6)
	ps = append(ps, g.Wte.Params()...)
	ps = append(ps, g.Wpe.Params()...)
	for _, blk := range g.Blocks {
		ps = append(ps, blk.Params()...)
	}
	ps = append(ps, g.LNf.Params()...)
	// When weight tying is on LMHead.Weight is the same *Parameter as
	// Wte.Weight; the Wte path has already added it. Only Bias is unique.
	if g.TieWeights {
		if g.LMHead.Bias != nil {
			ps = append(ps, g.LMHead.Bias)
		}
		return ps
	}
	ps = append(ps, g.LMHead.Params()...)
	return ps
}
