package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// KVCache owns the per-layer key and value buffers that survive arena resets,
// the same way Parameter.Value owns the canonical weight data: a []float32
// allocated once on the Go heap, copied into a fresh BUFFER leaf at the start
// of every generation step.
//
// Buffer shape per layer is fixed: [1, NumHeads, MaxSeqLen, HeadDim], row-major.
// Pos is the next slot to write (positions in [0, Pos) are populated). Keeping
// the buffer shape fixed across steps is the load-bearing decision: it keeps
// every attention kernel WGSL source byte-identical across token steps, so
// the compiler.go pipeline cache reuses pipelines and per-step recompilation
// does not happen. Unwritten positions are zero on first use; the LengthMask
// passed into the attention forward zeros their contribution before softmax.
//
// Cache lifetime is strictly inference. Backward / autodiff is not modelled
// here. KVCache instances do not participate in Params() and are not visible
// to optimizers.
type KVCache struct {
	NumLayers int
	NumHeads  int
	HeadDim   int
	MaxSeqLen int

	// Pos is the next slot to write into. Pos == 0 means no tokens are cached
	// yet; Pos == MaxSeqLen means the cache is full and Store will panic.
	Pos int

	// Keys[layer] holds NumHeads * MaxSeqLen * HeadDim floats in head-major,
	// position-major layout. The byte at offset h*MaxSeqLen*HeadDim + p*HeadDim
	// + d is the value for head h, position p, dim d.
	Keys [][]float32

	// Values[layer] mirrors Keys.
	Values [][]float32
}

// NewKVCache allocates a fresh cache with all slots zero. Pos starts at 0.
// Allocations are pure Go heap; no arena reference is kept.
func NewKVCache(numLayers, numHeads, headDim, maxSeqLen int) *KVCache {
	if numLayers <= 0 || numHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		panic(fmt.Sprintf("nn: NewKVCache: all dims must be positive (numLayers=%d numHeads=%d headDim=%d maxSeqLen=%d)",
			numLayers, numHeads, headDim, maxSeqLen))
	}
	perLayer := numHeads * maxSeqLen * headDim
	c := &KVCache{
		NumLayers: numLayers,
		NumHeads:  numHeads,
		HeadDim:   headDim,
		MaxSeqLen: maxSeqLen,
		Keys:      make([][]float32, numLayers),
		Values:    make([][]float32, numLayers),
	}
	for i := 0; i < numLayers; i++ {
		c.Keys[i] = make([]float32, perLayer)
		c.Values[i] = make([]float32, perLayer)
	}
	return c
}

// Reset clears Pos and zeroes every populated slot, so the cache can be reused
// for a new prompt without reallocating. Unpopulated slots were already zero;
// repopulated slots are explicitly cleared so a stale value cannot leak into
// the masked region of the next run.
func (c *KVCache) Reset() {
	for i := range c.Keys {
		for j := range c.Keys[i] {
			c.Keys[i][j] = 0
		}
		for j := range c.Values[i] {
			c.Values[i][j] = 0
		}
	}
	c.Pos = 0
}

// PosOneHotData returns a fresh [MaxSeqLen]float32 buffer with 1.0 at index
// Pos and 0.0 elsewhere. The caller uploads it as a [1, 1, MaxSeqLen, 1] leaf
// per step. The forward kernel uses it to inject the new token's K and V into
// the fixed-shape cache buffer at the right slot via broadcast multiply plus
// add (see attention.go ForwardKVStep).
//
// When Pos == MaxSeqLen the returned buffer is all zeros, which is the
// degenerate case that ForwardKVStep guards against at call time.
func (c *KVCache) PosOneHotData() []float32 {
	m := make([]float32, c.MaxSeqLen)
	if c.Pos >= 0 && c.Pos < c.MaxSeqLen {
		m[c.Pos] = 1.0
	}
	return m
}

// LengthMaskData returns a fresh [MaxSeqLen]float32 buffer with 1.0 at every
// index in [0, Pos] and 0.0 above. Used as the multiplicative mask post-exp
// in masked softmax (mirrors the [T, T] mask pattern in CausalSelfAttention.
// Forward, restricted to a single query row). Positions 0..Pos-1 are the
// historical cache entries; position Pos is the new token slot that
// PosOneHotData injects K and V into in the same step.
func (c *KVCache) LengthMaskData() []float32 {
	m := make([]float32, c.MaxSeqLen)
	upto := c.Pos
	if upto >= c.MaxSeqLen {
		upto = c.MaxSeqLen - 1
	}
	for i := 0; i <= upto; i++ {
		m[i] = 1.0
	}
	return m
}

// StoreLayerKV writes kNew and vNew into slot c.Pos for layer. Each input is
// length NumHeads * HeadDim, head-major: input[h*HeadDim + d] is the value
// for head h, dim d. This is the layout produced by the graph's K_new and
// V_new outputs after splitHeads + permute, with B=1 and T=1.
//
// Does NOT advance Pos. Call Advance after every layer in the step has been
// stored, so all layers write the same slot.
func (c *KVCache) StoreLayerKV(layer int, kNew, vNew []float32) {
	if layer < 0 || layer >= c.NumLayers {
		panic(fmt.Sprintf("nn: KVCache.StoreLayerKV: layer %d out of range [0, %d)", layer, c.NumLayers))
	}
	expected := c.NumHeads * c.HeadDim
	if len(kNew) != expected {
		panic(fmt.Sprintf("nn: KVCache.StoreLayerKV: len(kNew)=%d, want %d", len(kNew), expected))
	}
	if len(vNew) != expected {
		panic(fmt.Sprintf("nn: KVCache.StoreLayerKV: len(vNew)=%d, want %d", len(vNew), expected))
	}
	if c.Pos < 0 || c.Pos >= c.MaxSeqLen {
		panic(fmt.Sprintf("nn: KVCache.StoreLayerKV: Pos=%d out of range [0, %d)", c.Pos, c.MaxSeqLen))
	}
	for h := 0; h < c.NumHeads; h++ {
		srcOff := h * c.HeadDim
		dstOff := h*c.MaxSeqLen*c.HeadDim + c.Pos*c.HeadDim
		copy(c.Keys[layer][dstOff:dstOff+c.HeadDim], kNew[srcOff:srcOff+c.HeadDim])
		copy(c.Values[layer][dstOff:dstOff+c.HeadDim], vNew[srcOff:srcOff+c.HeadDim])
	}
}

// Advance increments Pos by one. Caller invokes this after StoreLayerKV has
// run for every layer in the current step.
func (c *KVCache) Advance() {
	if c.Pos >= c.MaxSeqLen {
		panic(fmt.Sprintf("nn: KVCache.Advance: Pos=%d already at MaxSeqLen=%d", c.Pos, c.MaxSeqLen))
	}
	c.Pos++
}

// kvLeaf is the per-leaf upload helper shared by UploadKLeaf, UploadVLeaf,
// UploadPosOneHotLeaf, UploadLengthMaskLeaf. Mirrors the way Parameter.Load
// rebuilds a fresh BUFFER leaf at the start of every step.
func kvLeaf(a *uop.Arena, sh []int64, dtype *uop.DType, device string, data []float32) *tensor.Tensor {
	leaf := tensor.NewLeaf(a, sh, dtype, device)
	leaf.SetData(data)
	return leaf
}

// UploadKLeaf returns a fresh leaf in a backed by c.Keys[layer], shape
// [1, NumHeads, MaxSeqLen, HeadDim]. dtype must match the attention module
// the cache feeds. The leaf is owned by a; the underlying host data is
// owned by c and outlives a.
func (c *KVCache) UploadKLeaf(a *uop.Arena, layer int, dtype *uop.DType, device string) *tensor.Tensor {
	if layer < 0 || layer >= c.NumLayers {
		panic(fmt.Sprintf("nn: KVCache.UploadKLeaf: layer %d out of range [0, %d)", layer, c.NumLayers))
	}
	sh := []int64{1, int64(c.NumHeads), int64(c.MaxSeqLen), int64(c.HeadDim)}
	return kvLeaf(a, sh, dtype, device, c.Keys[layer])
}

// UploadVLeaf mirrors UploadKLeaf for the value cache.
func (c *KVCache) UploadVLeaf(a *uop.Arena, layer int, dtype *uop.DType, device string) *tensor.Tensor {
	if layer < 0 || layer >= c.NumLayers {
		panic(fmt.Sprintf("nn: KVCache.UploadVLeaf: layer %d out of range [0, %d)", layer, c.NumLayers))
	}
	sh := []int64{1, int64(c.NumHeads), int64(c.MaxSeqLen), int64(c.HeadDim)}
	return kvLeaf(a, sh, dtype, device, c.Values[layer])
}

// UploadPosOneHotLeaf returns a fresh leaf shaped [1, 1, MaxSeqLen, 1] with
// 1.0 at the slot Pos and 0.0 elsewhere.
func (c *KVCache) UploadPosOneHotLeaf(a *uop.Arena, dtype *uop.DType, device string) *tensor.Tensor {
	sh := []int64{1, 1, int64(c.MaxSeqLen), 1}
	return kvLeaf(a, sh, dtype, device, c.PosOneHotData())
}

// UploadLengthMaskLeaf returns a fresh leaf shaped [1, 1, 1, MaxSeqLen] with
// 1.0 at every position in [0, Pos] and 0.0 above.
func (c *KVCache) UploadLengthMaskLeaf(a *uop.Arena, dtype *uop.DType, device string) *tensor.Tensor {
	sh := []int64{1, 1, 1, int64(c.MaxSeqLen)}
	return kvLeaf(a, sh, dtype, device, c.LengthMaskData())
}
