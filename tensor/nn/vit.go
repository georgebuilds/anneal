package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── PatchEmbed (Vision Transformer patch embedding) ──────────────────────────
//
// PatchEmbed splits a batch of images into a sequence of non-overlapping
// patches and projects each patch into an embedding vector. The canonical
// ViT formulation is a single Conv2d with kernel_size=patch_size and
// stride=patch_size; we implement the same math via movement-ops-plus-Linear,
// matching the roadmap note "Conv2d-as-Linear reshape+linear".
//
// Concretely, for an input x of shape [B, C, H, W] with patch size P
// (assumes H and W are divisible by P), the forward pass:
//
//  1. reshape  [B, C, H,        W      ]
//     -> [B, C, H/P,  P,  W/P, P]
//  2. permute (0, 2, 4, 1, 3, 5)
//     -> [B, H/P, W/P, C, P, P]
//  3. reshape -> [B, N, C*P*P]                    where N = (H/P)*(W/P)
//  4. Linear:   x @ W^T + b  with W shape [embedDim, C*P*P]
//     -> [B, N, embedDim]
//
// Every step except (4) is a pure rangeify movement op (no data copy); only
// the Linear is a real kernel. This is the same algebraic operation as a
// Conv2d with kernel = stride = P, just expressed as the equivalent
// patch-flatten-plus-dense formulation that the ViT paper introduced.
type PatchEmbed struct {
	Proj *Linear // patch projection: C*P*P -> embedDim

	ImageH, ImageW int64 // input spatial dims (must be divisible by Patch)
	Patch          int64 // patch side length (kernel == stride)
	InCh           int64 // input channel count C
	EmbedDim       int64 // output embedding width
}

// NewPatchEmbed constructs a PatchEmbed module.
//
//   - imageH, imageW: input spatial dims; both must be divisible by patch.
//   - patch:          patch side length (kernel == stride).
//   - inCh:           input channel count (3 for RGB, 1 for grayscale).
//   - embedDim:       output embedding width per patch.
//
// The Linear's weights are zero-initialised; the caller seeds them before the
// first forward pass, matching the convention used by NewLinear / NewConv2d.
func NewPatchEmbed(a *uop.Arena, imageH, imageW, patch, inCh, embedDim int64) *PatchEmbed {
	if patch <= 0 {
		panic(fmt.Sprintf("nn: NewPatchEmbed: patch must be positive, got %d", patch))
	}
	if imageH%patch != 0 || imageW%patch != 0 {
		panic(fmt.Sprintf("nn: NewPatchEmbed: imageH=%d, imageW=%d must be divisible by patch=%d",
			imageH, imageW, patch))
	}
	if inCh <= 0 || embedDim <= 0 {
		panic(fmt.Sprintf("nn: NewPatchEmbed: inCh=%d, embedDim=%d must be positive", inCh, embedDim))
	}

	dtype := uop.Dtypes.Float32
	device := "webgpu"
	patchFlat := inCh * patch * patch

	return &PatchEmbed{
		Proj:     NewLinear(a, patchFlat, embedDim, true, dtype, device),
		ImageH:   imageH,
		ImageW:   imageW,
		Patch:    patch,
		InCh:     inCh,
		EmbedDim: embedDim,
	}
}

// Forward applies the patch embedding.
//
// Input  x: [B, C, H, W]
// Output:   [B, N, embedDim]   where N = (H/P) * (W/P)
//
// The reshape / permute / reshape chain is pure index arithmetic under the
// rangeify model (no materialised intermediate buffer); only the Linear at
// the end is a real kernel.
func (p *PatchEmbed) Forward(x *tensor.Tensor) *tensor.Tensor {
	xShape := x.Shape()
	if len(xShape) != 4 {
		panic(fmt.Sprintf("nn: PatchEmbed.Forward: input must be rank 4 [B,C,H,W], got rank %d", len(xShape)))
	}
	B, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	if C != p.InCh {
		panic(fmt.Sprintf("nn: PatchEmbed.Forward: input C=%d != module inCh=%d", C, p.InCh))
	}
	if H != p.ImageH || W != p.ImageW {
		panic(fmt.Sprintf("nn: PatchEmbed.Forward: input H=%d, W=%d != module ImageH=%d, ImageW=%d",
			H, W, p.ImageH, p.ImageW))
	}

	P := p.Patch
	nH, nW := H/P, W/P
	N := nH * nW
	patchFlat := C * P * P

	// 1. Split the spatial dims into (num_h, patch_h) and (num_w, patch_w).
	//    [B, C, H, W] -> [B, C, nH, P, nW, P]
	y := x.Reshape([]int64{B, C, nH, P, nW, P})

	// 2. Group the patch axes together: bring (nH, nW) to the front of the
	//    spatial group and (C, patch_h, patch_w) to the end.
	//    [B, C, nH, P, nW, P] (axes 0..5) -> [B, nH, nW, C, P, P]
	y = y.Permute([]int{0, 2, 4, 1, 3, 5})

	// 3. Flatten the patch contents to one feature dim per patch token.
	//    [B, nH, nW, C, P, P] -> [B, N, C*P*P]
	y = y.Reshape([]int64{B, N, patchFlat})

	// 4. Project: [B, N, C*P*P] @ W^T -> [B, N, embedDim] (+ bias).
	return p.Proj.Forward(y)
}

// Params returns the trainable parameters.
func (p *PatchEmbed) Params() []*Parameter {
	return p.Proj.Params()
}

// ── ViTBlock (pre-LayerNorm encoder block, non-causal) ───────────────────────
//
// ViTBlock is the standard pre-LN transformer encoder block as used in the
// Vision Transformer (Dosovitskiy et al, 2020). Same canonical residual
// pattern as nn.Block:
//
//	x = x + attn(ln1(x))
//	x = x + mlp(ln2(x))
//
// The only structural difference from nn.Block is the attention module: ViT
// uses non-causal (full bidirectional) self-attention. The MLP (tanh-GELU,
// 4x expansion) and LayerNorm shapes are identical to nn.Block.
//
// We could not reuse nn.Block directly because its Attn field is constructed
// through NewCausalSelfAttention at NewBlock time; rather than thread a
// causal/non-causal flag through the existing constructor, we author a
// parallel block that constructs a non-causal attention via NewSelfAttention.
// The struct is intentionally unexported because ViT is the only consumer.
type ViTBlock struct {
	LN1  *LayerNorm
	Attn *CausalSelfAttention // built via NewSelfAttention; mask is all-ones
	LN2  *LayerNorm
	MLP  *MLP
}

// NewViTBlock constructs a pre-LN ViT encoder block over a sequence of length
// seqLen (one entry per patch token). MLP uses the standard 4x expansion
// inherited from NewMLP; if a future caller wants the more typical ViT 2x
// or 4x ratio it can either parameterise this or wrap a custom MLP.
func NewViTBlock(a *uop.Arena, nEmbd, nHead, seqLen int) *ViTBlock {
	const lnEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"

	return &ViTBlock{
		LN1:  NewLayerNorm(a, int64(nEmbd), lnEps),
		Attn: NewSelfAttention(a, nEmbd, nHead, seqLen),
		LN2:  NewLayerNorm(a, int64(nEmbd), lnEps),
		MLP:  NewMLP(a, nEmbd, dtype, device),
	}
}

// Forward computes the pre-LN ViT encoder block.
//
// Input  x: [B, N, nEmbd]    (N <= seqLen)
// Output y: [B, N, nEmbd]
func (b *ViTBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := x.Add(b.Attn.Forward(b.LN1.Forward(x)))
	return h.Add(b.MLP.Forward(b.LN2.Forward(h)))
}

// Params returns all trainable parameters in deterministic order:
// LN1.{W,B}, Attn.QKV.{W,B}, Attn.Proj.{W,B}, LN2.{W,B}, MLP.FC1.{W,B}, MLP.FC2.{W,B}.
func (b *ViTBlock) Params() []*Parameter {
	ps := make([]*Parameter, 0, 12)
	ps = append(ps, b.LN1.Params()...)
	ps = append(ps, b.Attn.Params()...)
	ps = append(ps, b.LN2.Params()...)
	ps = append(ps, b.MLP.Params()...)
	return ps
}

// ── ViT container (Vision Transformer) ───────────────────────────────────────
//
// ViT composes patch embedding, a learned positional embedding, a stack of
// non-causal encoder blocks, a final LayerNorm, and a linear classification
// head into the canonical ViT architecture (Dosovitskiy et al, 2020).
//
//	x [B, C, H, W]
//	  -> PatchEmbed                     -> [B, N, embedDim]
//	  -> + PosEmb (learned [N, embedDim], broadcast over batch)
//	  -> N x ViTBlock                   -> [B, N, embedDim]
//	  -> LNf                            -> [B, N, embedDim]
//	  -> mean over N                    -> [B, embedDim]    (mean-pool head)
//	  -> Head: Linear(embedDim, numCls) -> [B, numClasses]
//
// Design call (mean-pool vs CLS token): we use mean-pooling over the patch
// tokens for the classifier input. Justification: (a) ViT-paper-correct on
// CIFAR-scale (the paper itself reports mean-pool and CLS within noise on
// JFT-finetuned CIFAR-10), (b) avoids a CLS-token leaf and a positional
// embedding entry for it, keeping the parameter and shape contract minimal,
// (c) the demo's point is the patch-embed plus encoder stack, not the head.
// The CLS-token variant is a one-Parameter delta if a future caller wants it.
//
// Positional embedding is a learned [N, embedDim] Parameter (not the
// 2D-sinusoid fixed table). One Parameter, broadcast-added; matches the
// original ViT paper's "1D learnable position embedding" choice.
type ViT struct {
	Patch  *PatchEmbed
	PosEmb *Parameter  // [N, embedDim], learnable
	Blocks []*ViTBlock // L encoder blocks
	LNf    *LayerNorm  // final LayerNorm over embedDim
	Head   *Linear     // classifier head: embedDim -> numClasses

	ImageH, ImageW int64
	PatchSize      int64
	InCh           int64
	EmbedDim       int64
	NHead          int
	NLayer         int
	NumClasses     int64
	N              int64 // (ImageH/Patch) * (ImageW/Patch); cached
}

// NewViT constructs a ViT model with the given configuration.
//
//   - imageH, imageW: input image spatial dims; both must be divisible by patch.
//   - patch:          patch side length.
//   - inCh:           input channel count (3 for RGB).
//   - embedDim:       embedding / residual stream width.
//   - nLayer:         number of stacked encoder blocks.
//   - nHead:          attention heads per block (embedDim must divide cleanly).
//   - numClasses:     output class count.
//
// All weights are zero-allocated; the caller seeds them before the first
// forward pass, matching the convention used elsewhere in tensor/nn.
func NewViT(a *uop.Arena, imageH, imageW, patch, inCh, embedDim int64,
	nLayer, nHead int, numClasses int64) *ViT {
	if imageH <= 0 || imageW <= 0 || patch <= 0 || inCh <= 0 || embedDim <= 0 ||
		nLayer <= 0 || nHead <= 0 || numClasses <= 0 {
		panic(fmt.Sprintf("nn: NewViT: all dims must be positive (imageH=%d imageW=%d patch=%d inCh=%d embedDim=%d nLayer=%d nHead=%d numClasses=%d)",
			imageH, imageW, patch, inCh, embedDim, nLayer, nHead, numClasses))
	}
	if imageH%patch != 0 || imageW%patch != 0 {
		panic(fmt.Sprintf("nn: NewViT: imageH=%d, imageW=%d must be divisible by patch=%d",
			imageH, imageW, patch))
	}
	if int(embedDim)%nHead != 0 {
		panic(fmt.Sprintf("nn: NewViT: embedDim=%d not divisible by nHead=%d", embedDim, nHead))
	}

	const lnEps = float32(1e-5)
	dtype := uop.Dtypes.Float32
	device := "webgpu"
	N := (imageH / patch) * (imageW / patch)

	blocks := make([]*ViTBlock, nLayer)
	for i := range blocks {
		blocks[i] = NewViTBlock(a, int(embedDim), nHead, int(N))
	}

	return &ViT{
		Patch:      NewPatchEmbed(a, imageH, imageW, patch, inCh, embedDim),
		PosEmb:     NewParameter(a, []int64{N, embedDim}, dtype, device),
		Blocks:     blocks,
		LNf:        NewLayerNorm(a, embedDim, lnEps),
		Head:       NewLinear(a, embedDim, numClasses, true, dtype, device),
		ImageH:     imageH,
		ImageW:     imageW,
		PatchSize:  patch,
		InCh:       inCh,
		EmbedDim:   embedDim,
		NHead:      nHead,
		NLayer:     nLayer,
		NumClasses: numClasses,
		N:          N,
	}
}

// Forward runs the full ViT stack on a batch of images.
//
// Input  x: [B, C, H, W]    (C == InCh, H == ImageH, W == ImageW)
// Output:   [B, numClasses] (logits, no softmax applied)
func (v *ViT) Forward(x *tensor.Tensor) *tensor.Tensor {
	xShape := x.Shape()
	if len(xShape) != 4 {
		panic(fmt.Sprintf("nn: ViT.Forward: input must be rank 4 [B,C,H,W], got rank %d", len(xShape)))
	}
	B := xShape[0]

	// ── Patch embed + positional embed ───────────────────────────────────────
	// [B, C, H, W] -> [B, N, embedDim]
	h := v.Patch.Forward(x)

	// PosEmb is [N, embedDim]; broadcast-prepend a batch dim and Expand to
	// match h's [B, N, embedDim].
	posB := tensor.BroadcastToSints(v.PosEmb.T, h.ShapeSints())
	h = h.Add(posB)

	// ── Encoder blocks ───────────────────────────────────────────────────────
	for _, blk := range v.Blocks {
		h = blk.Forward(h)
	}

	// ── Final LayerNorm ──────────────────────────────────────────────────────
	h = v.LNf.Forward(h)

	// ── Mean-pool over patch tokens ──────────────────────────────────────────
	// [B, N, embedDim] -> [B, embedDim].
	// keepdim=false then explicit Reshape mirrors LayerNorm's pattern to keep
	// the autodiff shape tracker honest about the reduced rank.
	pooled := h.Mean([]int{1}, false) // [B, embedDim]
	pooled = pooled.Reshape([]int64{B, v.EmbedDim})

	// ── Classifier head ──────────────────────────────────────────────────────
	return v.Head.Forward(pooled) // [B, numClasses]
}

// Params returns all trainable parameters in deterministic order:
//
//	Patch.Proj.{W,B},
//	PosEmb,
//	Block[0..L-1].(12 params each, order from ViTBlock.Params()),
//	LNf.{W,B},
//	Head.{W,B}.
//
// Total: 2 (Patch) + 1 (PosEmb) + 12 * nLayer + 2 (LNf) + 2 (Head)
//
//	= 12 * nLayer + 7.
func (v *ViT) Params() []*Parameter {
	ps := make([]*Parameter, 0, 12*v.NLayer+7)
	ps = append(ps, v.Patch.Params()...)
	ps = append(ps, v.PosEmb)
	for _, blk := range v.Blocks {
		ps = append(ps, blk.Params()...)
	}
	ps = append(ps, v.LNf.Params()...)
	ps = append(ps, v.Head.Params()...)
	return ps
}
