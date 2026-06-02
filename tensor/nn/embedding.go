package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Embedding ─────────────────────────────────────────────────────────────────

// Embedding is a learnable lookup table that maps integer indices to dense
// vectors. Mirrors torch.nn.Embedding: a Weight matrix of shape
// [NumEmbeddings, EmbeddingDim] is queried by a 1-D Int32 index tensor of
// length B; the forward pass returns rows of Weight stacked into a [B,
// EmbeddingDim] output.
//
// Backward is provided by the Slice D scatter-add path: dW[i] accumulates
// adj rows for every b where idx[b] == i, deterministically via host-side
// sort plus segment-sum on the device. Indices that never appear in idx
// receive a zero gradient.
//
// Initialisation follows the rest of tensor/nn: NewEmbedding allocates a
// zero-valued Parameter and leaves seeding to the caller (mirrors NewLinear
// and NewConv2d). Tests in this package use a fixed-seed math/rand source
// to fill Weight.Value with small normal samples before the first Load.
type Embedding struct {
	Weight *Parameter
}

// NewEmbedding constructs an Embedding layer with a Weight matrix shaped
// [numEmbeddings, embeddingDim]. Weight.Value is zero-initialised; callers
// seed it before the first forward pass, the same convention NewLinear uses.
func NewEmbedding(a *uop.Arena, numEmbeddings, embeddingDim int64, dtype *uop.DType, device string) *Embedding {
	if numEmbeddings <= 0 || embeddingDim <= 0 {
		panic(fmt.Sprintf("nn: NewEmbedding: dims must be positive, got [%d, %d]", numEmbeddings, embeddingDim))
	}
	return &Embedding{
		Weight: NewParameter(a, []int64{numEmbeddings, embeddingDim}, dtype, device),
	}
}

// Forward selects Weight rows for each entry of idx. idx must be a 1-D
// Int32 (or castable integer) tensor of length B; the output has shape
// [B, EmbeddingDim]. Composes directly with the Slice B-D Gather machinery,
// so the backward pass is scatter-add with deterministic count-of-occurrences
// accumulation.
func (e *Embedding) Forward(idx *tensor.Tensor) *tensor.Tensor {
	return e.Weight.T.Gather(0, idx)
}

// Params returns the single trainable parameter (Weight) as a one-element
// slice, matching the Linear / Conv2d Params() contract.
func (e *Embedding) Params() []*Parameter {
	return []*Parameter{e.Weight}
}
