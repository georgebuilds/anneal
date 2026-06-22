package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── DiT primitives (Diffusion Transformer, adaLN-zero) ───────────────────────
//
// Foundational pieces for a Diffusion Transformer (Peebles & Xie, 2022): a
// norm-only LayerNorm (AdaLNNorm), the per-token modulation that adaLN-zero
// applies on top of it (Modulate), and the inverse of PatchEmbed (Unpatchify).
// The DiT container (a later slice) composes these with the existing
// PatchEmbed, non-causal attention blocks, and the DDPM schedule/time-embed.
//
// These are deliberately small, op-only helpers: no new tensor ops, no new IR,
// just compositions of the existing reshape/permute/reduce/elementwise surface.

// AdaLNNorm normalizes x across its last axis WITHOUT a learnable affine:
//
//	xhat = (x - mean(x)) / sqrt(var(x) + eps)
//
// Standard LayerNorm folds in a learnable scale/shift; DiT's adaptive LayerNorm
// instead receives the scale/shift from a conditioning vector (see Modulate), so
// the normalization itself must not apply its own affine. The reduction pattern
// (keepdim=false then Reshape to re-add the singleton axis) mirrors LayerNorm so
// the autodiff shape tracker stays honest about the reduced rank.
func AdaLNNorm(x *tensor.Tensor, eps float32) *tensor.Tensor {
	rank := x.Rank()
	if rank < 1 {
		panic("nn: AdaLNNorm: input must have rank >= 1")
	}
	lastAxis := rank - 1

	xShape := x.Shape()
	keepShape := make([]int64, rank)
	copy(keepShape, xShape)
	keepShape[lastAxis] = 1

	mu := x.Mean([]int{lastAxis}, false).Reshape(keepShape)
	xc := x.Sub(mu)
	variance := xc.Mul(xc).Mean([]int{lastAxis}, false).Reshape(keepShape)

	a := x.Arena()
	epsT := tensor.FullSints(a, variance.ShapeSints(), float64(eps), x.DType(), x.Device())
	invStd := variance.Add(epsT).Sqrt().Recip()
	return xc.Mul(invStd)
}

// Modulate applies adaLN-zero per-token modulation to a token sequence:
//
//	out = x * (1 + scale) + shift
//
// x is [B, N, D] (a sequence of N tokens of width D); scale and shift are [B, D]
// conditioning vectors that broadcast across the N token axis. The "1 +" makes a
// zero-valued scale and shift act as the identity, which is exactly what lets
// adaLN-zero zero-initialise its conditioning projection so every block starts
// as a no-op and the network begins training as the identity flow.
func Modulate(x, scale, shift *tensor.Tensor) *tensor.Tensor {
	if x.Rank() != 3 {
		panic(fmt.Sprintf("nn: Modulate: x must be rank 3 [B,N,D], got rank %d", x.Rank()))
	}
	if scale.Rank() != 2 || shift.Rank() != 2 {
		panic(fmt.Sprintf("nn: Modulate: scale and shift must be rank 2 [B,D], got %d and %d",
			scale.Rank(), shift.Rank()))
	}
	xs := x.Shape()
	B, N, D := xs[0], xs[1], xs[2]

	bcast := func(c *tensor.Tensor) *tensor.Tensor {
		cs := c.Shape()
		if cs[0] != B || cs[1] != D {
			panic(fmt.Sprintf("nn: Modulate: cond shape [%d,%d] != [B,D]=[%d,%d]", cs[0], cs[1], B, D))
		}
		return c.Reshape([]int64{B, 1, D}).Expand([]int64{B, N, D})
	}
	scaleB := bcast(scale)
	shiftB := bcast(shift)

	a := x.Arena()
	onesT := tensor.FullSints(a, scaleB.ShapeSints(), 1.0, x.DType(), x.Device())
	return x.Mul(scaleB.Add(onesT)).Add(shiftB)
}

// Unpatchify is the inverse of PatchEmbed's spatial fold: it reassembles a grid
// of per-patch feature tokens back into an image. Given tokens of shape
// [B, N, inCh*patch*patch] (the per-patch pixel/feature payload, e.g. the output
// of a final Linear that projects the residual stream to patch pixels), it
// returns [B, inCh, imageH, imageW].
//
// The fold in PatchEmbed.Forward is:
//
//	[B,C,nH,P,nW,P] --permute(0,2,4,1,3,5)--> [B,nH,nW,C,P,P] --reshape--> [B,N,C*P*P]
//
// so the exact inverse is:
//
//	[B,N,C*P*P] --reshape--> [B,nH,nW,C,P,P] --permute(0,3,1,4,2,5)--> [B,C,nH,P,nW,P] --reshape--> [B,C,H,W]
//
// Every step is a pure rangeify movement op (index arithmetic, no data copy).
func Unpatchify(tokens *tensor.Tensor, imageH, imageW, patch, inCh int64) *tensor.Tensor {
	if tokens.Rank() != 3 {
		panic(fmt.Sprintf("nn: Unpatchify: tokens must be rank 3 [B,N,inCh*patch*patch], got rank %d", tokens.Rank()))
	}
	if patch <= 0 {
		panic(fmt.Sprintf("nn: Unpatchify: patch must be positive, got %d", patch))
	}
	if imageH%patch != 0 || imageW%patch != 0 {
		panic(fmt.Sprintf("nn: Unpatchify: imageH=%d, imageW=%d must be divisible by patch=%d", imageH, imageW, patch))
	}
	ts := tokens.Shape()
	B, N, feat := ts[0], ts[1], ts[2]
	nH, nW := imageH/patch, imageW/patch
	if N != nH*nW {
		panic(fmt.Sprintf("nn: Unpatchify: N=%d != (imageH/patch)*(imageW/patch)=%d", N, nH*nW))
	}
	if feat != inCh*patch*patch {
		panic(fmt.Sprintf("nn: Unpatchify: feature dim=%d != inCh*patch*patch=%d", feat, inCh*patch*patch))
	}

	y := tokens.Reshape([]int64{B, nH, nW, inCh, patch, patch})
	y = y.Permute([]int{0, 3, 1, 4, 2, 5})
	return y.Reshape([]int64{B, inCh, imageH, imageW})
}

// ── DiTBlock (adaLN-zero conditioned transformer block) ──────────────────────
//
// DiTBlock is a pre-norm transformer block with adaptive-LayerNorm-zero
// conditioning (Peebles & Xie, 2022). Unlike ViTBlock (which carries its own
// LayerNorm affine), DiTBlock normalizes with AdaLNNorm (no affine) and takes
// the scale/shift/gate from a conditioning vector c through one modulation
// projection:
//
//	c6 = Mod(SiLU(c))                                     -> [B, 6D]
//	(shift1, scale1, gate1, shift2, scale2, gate2) = split(c6)
//	x = x + gate1 * Attn(Modulate(AdaLNNorm(x), scale1, shift1))
//	x = x + gate2 * MLP (Modulate(AdaLNNorm(x), scale2, shift2))
//
// Mod is meant to be zero-initialised (the "zero" in adaLN-zero): at init every
// gate is 0, so each block is the exact identity and the network begins as the
// identity flow, then learns to deviate. Attention is non-causal (the all-ones
// mask of NewSelfAttention), as DiT operates on an unordered patch set.
type DiTBlock struct {
	Attn *CausalSelfAttention // non-causal via NewSelfAttention (all-ones mask)
	MLP  *MLP
	Mod  *Linear // SiLU(c) -> 6*embedDim modulation; zero-init for adaLN-zero
	Eps  float32
	D    int64
}

// NewDiTBlock constructs a DiT block over a sequence of seqLen patch tokens of
// width embedDim. Mod's weights are zero (the adaLN-zero init); the caller seeds
// the other submodules before the first forward.
func NewDiTBlock(a *uop.Arena, embedDim int64, nHead, seqLen int) *DiTBlock {
	if embedDim <= 0 || nHead <= 0 || seqLen <= 0 {
		panic(fmt.Sprintf("nn: NewDiTBlock: dims must be positive (embedDim=%d nHead=%d seqLen=%d)", embedDim, nHead, seqLen))
	}
	if int(embedDim)%nHead != 0 {
		panic(fmt.Sprintf("nn: NewDiTBlock: embedDim=%d not divisible by nHead=%d", embedDim, nHead))
	}
	dtype := uop.Dtypes.Float32
	device := "webgpu"
	return &DiTBlock{
		Attn: NewSelfAttention(a, int(embedDim), nHead, seqLen),
		MLP:  NewMLP(a, int(embedDim), dtype, device),
		Mod:  NewLinear(a, embedDim, 6*embedDim, true, dtype, device),
		Eps:  1e-6,
		D:    embedDim,
	}
}

// Forward applies the adaLN-zero block. x is [B, N, D]; c is the conditioning
// vector [B, D]. Output is [B, N, D].
func (b *DiTBlock) Forward(x, c *tensor.Tensor) *tensor.Tensor {
	if x.Rank() != 3 {
		panic(fmt.Sprintf("nn: DiTBlock.Forward: x must be rank 3 [B,N,D], got rank %d", x.Rank()))
	}
	D := b.D
	m := b.Mod.Forward(SiLU(c)) // [B, 6D]
	B := m.Shape()[0]
	chunk := func(k int64) *tensor.Tensor {
		return m.Shrink([][2]int64{{0, B}, {k * D, (k + 1) * D}})
	}
	shift1, scale1, gate1 := chunk(0), chunk(1), chunk(2)
	shift2, scale2, gate2 := chunk(3), chunk(4), chunk(5)

	attn := b.Attn.Forward(Modulate(AdaLNNorm(x, b.Eps), scale1, shift1))
	x = x.Add(applyGate(attn, gate1))
	mlp := b.MLP.Forward(Modulate(AdaLNNorm(x, b.Eps), scale2, shift2))
	x = x.Add(applyGate(mlp, gate2))
	return x
}

// Params returns the block's parameters in deterministic order: Mod, then Attn,
// then MLP.
func (b *DiTBlock) Params() []*Parameter {
	ps := make([]*Parameter, 0, 2+len(b.Attn.Params())+len(b.MLP.Params()))
	ps = append(ps, b.Mod.Params()...)
	ps = append(ps, b.Attn.Params()...)
	ps = append(ps, b.MLP.Params()...)
	return ps
}

// applyGate broadcasts a [B, D] gate over the N token axis of sub [B, N, D] and
// multiplies. A zero gate produces a zero contribution, which is what makes the
// adaLN-zero residual the identity at init.
func applyGate(sub, gate *tensor.Tensor) *tensor.Tensor {
	s := sub.Shape()
	B, N, D := s[0], s[1], s[2]
	gateB := gate.Reshape([]int64{B, 1, D}).Expand([]int64{B, N, D})
	return sub.Mul(gateB)
}

// ── DiT container (Diffusion Transformer) ────────────────────────────────────
//
// DiT composes patch embedding, a learned positional embedding, a conditioning
// projection, a stack of adaLN-zero blocks, a final adaLN, a linear head, and
// unpatchify into the canonical Diffusion Transformer (Peebles & Xie, 2022):
//
//	x [B, C, H, W]
//	  -> PatchEmbed                          -> [B, N, D]
//	  -> + PosEmb (learned [N, D])
//	  -> blocks (each conditioned on c = CondProj(cond))
//	  -> Modulate(AdaLNNorm(x), finalScale, finalShift)
//	  -> Linear(D -> outCh*P*P)              -> [B, N, outCh*P*P]
//	  -> Unpatchify                          -> [B, outCh, H, W]   (eps prediction)
//
// cond is a [B, condDim] conditioning input the caller assembles (for example a
// timestep embedding plus a class embedding); the container projects it to the
// residual width once and shares it across blocks.
type DiT struct {
	Patch    *PatchEmbed
	PosEmb   *Parameter // [N, embedDim], learned
	CondProj *Linear    // condDim -> embedDim
	Blocks   []*DiTBlock
	FinalMod *Linear // SiLU(c) -> 2*embedDim (finalShift, finalScale); zero-init
	FinalLin *Linear // embedDim -> outCh*patch*patch

	ImageH, ImageW int64
	PatchSize      int64
	InCh, OutCh    int64
	EmbedDim       int64
	CondDim        int64
	N              int64
	NHead, NLayer  int
	Eps            float32
}

// NewDiT constructs a DiT. All weights except the (deliberately zero) modulation
// projections are zero-allocated; the caller seeds them before the first forward,
// leaving every block's Mod and the FinalMod at zero for the adaLN-zero init.
func NewDiT(a *uop.Arena, imageH, imageW, patch, inCh, outCh, embedDim, condDim int64,
	nLayer, nHead int) *DiT {
	if imageH <= 0 || imageW <= 0 || patch <= 0 || inCh <= 0 || outCh <= 0 ||
		embedDim <= 0 || condDim <= 0 || nLayer <= 0 || nHead <= 0 {
		panic(fmt.Sprintf("nn: NewDiT: all dims must be positive (imageH=%d imageW=%d patch=%d inCh=%d outCh=%d embedDim=%d condDim=%d nLayer=%d nHead=%d)",
			imageH, imageW, patch, inCh, outCh, embedDim, condDim, nLayer, nHead))
	}
	if imageH%patch != 0 || imageW%patch != 0 {
		panic(fmt.Sprintf("nn: NewDiT: imageH=%d, imageW=%d must be divisible by patch=%d", imageH, imageW, patch))
	}
	if int(embedDim)%nHead != 0 {
		panic(fmt.Sprintf("nn: NewDiT: embedDim=%d not divisible by nHead=%d", embedDim, nHead))
	}

	const lnEps = float32(1e-6)
	dtype := uop.Dtypes.Float32
	device := "webgpu"
	N := (imageH / patch) * (imageW / patch)

	blocks := make([]*DiTBlock, nLayer)
	for i := range blocks {
		blocks[i] = NewDiTBlock(a, embedDim, nHead, int(N))
	}

	return &DiT{
		Patch:     NewPatchEmbed(a, imageH, imageW, patch, inCh, embedDim),
		PosEmb:    NewParameter(a, []int64{N, embedDim}, dtype, device),
		CondProj:  NewLinear(a, condDim, embedDim, true, dtype, device),
		Blocks:    blocks,
		FinalMod:  NewLinear(a, embedDim, 2*embedDim, true, dtype, device),
		FinalLin:  NewLinear(a, embedDim, outCh*patch*patch, true, dtype, device),
		ImageH:    imageH,
		ImageW:    imageW,
		PatchSize: patch,
		InCh:      inCh,
		OutCh:     outCh,
		EmbedDim:  embedDim,
		CondDim:   condDim,
		N:         N,
		NHead:     nHead,
		NLayer:    nLayer,
		Eps:       lnEps,
	}
}

// Forward runs the DiT. x is [B, InCh, ImageH, ImageW]; cond is [B, CondDim].
// Output is [B, OutCh, ImageH, ImageW] (the predicted noise for eps-prediction
// diffusion training).
func (m *DiT) Forward(x, cond *tensor.Tensor) *tensor.Tensor {
	if x.Rank() != 4 {
		panic(fmt.Sprintf("nn: DiT.Forward: x must be rank 4 [B,C,H,W], got rank %d", x.Rank()))
	}
	if cond.Rank() != 2 {
		panic(fmt.Sprintf("nn: DiT.Forward: cond must be rank 2 [B,condDim], got rank %d", cond.Rank()))
	}

	h := m.Patch.Forward(x) // [B, N, D]
	posB := tensor.BroadcastToSints(m.PosEmb.T, h.ShapeSints())
	h = h.Add(posB)

	c := m.CondProj.Forward(cond) // [B, D]
	for _, blk := range m.Blocks {
		h = blk.Forward(h, c)
	}

	// Final adaLN, then project each token to its patch pixels and unpatchify.
	fm := m.FinalMod.Forward(SiLU(c)) // [B, 2D]
	B := fm.Shape()[0]
	D := m.EmbedDim
	fshift := fm.Shrink([][2]int64{{0, B}, {0, D}})
	fscale := fm.Shrink([][2]int64{{0, B}, {D, 2 * D}})
	h = Modulate(AdaLNNorm(h, m.Eps), fscale, fshift)

	h = m.FinalLin.Forward(h) // [B, N, OutCh*P*P]
	return Unpatchify(h, m.ImageH, m.ImageW, m.PatchSize, m.OutCh)
}

// Params returns all trainable parameters in deterministic order:
// Patch, PosEmb, CondProj, Blocks[0..L-1], FinalMod, FinalLin.
func (m *DiT) Params() []*Parameter {
	ps := make([]*Parameter, 0, 8+len(m.Blocks)*8)
	ps = append(ps, m.Patch.Params()...)
	ps = append(ps, m.PosEmb)
	ps = append(ps, m.CondProj.Params()...)
	for _, blk := range m.Blocks {
		ps = append(ps, blk.Params()...)
	}
	ps = append(ps, m.FinalMod.Params()...)
	ps = append(ps, m.FinalLin.Params()...)
	return ps
}
