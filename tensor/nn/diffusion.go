package nn

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── SiLU activation ───────────────────────────────────────────────────────────

// SiLU returns x * sigmoid(x), aka Swish. One-liner used by DDPM/U-Net stacks.
func SiLU(x *tensor.Tensor) *tensor.Tensor {
	return x.Mul(Sigmoid(x))
}

// ── DDPM schedule helpers ─────────────────────────────────────────────────────

// MakeLinearBetas returns a length-T schedule of variance values linearly
// interpolated between betaStart and betaEnd. T must be >= 2.
func MakeLinearBetas(T int, betaStart, betaEnd float32) []float32 {
	if T < 2 {
		panic(fmt.Sprintf("nn: MakeLinearBetas: T must be >= 2, got %d", T))
	}
	out := make([]float32, T)
	denom := float32(T - 1)
	for i := 0; i < T; i++ {
		out[i] = betaStart + (betaEnd-betaStart)*float32(i)/denom
	}
	return out
}

// MakeAlphas returns 1 - β_t for each step.
func MakeAlphas(betas []float32) []float32 {
	out := make([]float32, len(betas))
	for i, b := range betas {
		out[i] = 1.0 - b
	}
	return out
}

// MakeAlphaBars returns the cumulative product of alphas, i.e.
// ᾱ_t = ∏_{s=0}^{t} α_s.
func MakeAlphaBars(alphas []float32) []float32 {
	out := make([]float32, len(alphas))
	acc := float32(1.0)
	for i, a := range alphas {
		acc *= a
		out[i] = acc
	}
	return out
}

// ── Sinusoidal time embedding ────────────────────────────────────────────────

// SinusoidalTimeEmbed builds a [B, embedDim] float32 leaf encoding each
// timestep t in tValues as the canonical transformer sinusoidal embedding:
//
//	emb[b, 2i]   = sin(t / 10000^(2i/embedDim))
//	emb[b, 2i+1] = cos(t / 10000^(2i/embedDim))
//
// embedDim must be even. The embedding is constructed entirely host-side and
// uploaded as a fresh BUFFER leaf — the time index is a host-controlled scalar
// per sample, not a graph computation.
func SinusoidalTimeEmbed(a *uop.Arena, tValues []int32, embedDim int64, dtype *uop.DType, device string) *tensor.Tensor {
	if embedDim <= 0 || embedDim%2 != 0 {
		panic(fmt.Sprintf("nn: SinusoidalTimeEmbed: embedDim must be positive even, got %d", embedDim))
	}
	B := int64(len(tValues))
	if B == 0 {
		panic("nn: SinusoidalTimeEmbed: tValues must be non-empty")
	}
	half := embedDim / 2
	freqs := make([]float64, half)
	for i := int64(0); i < half; i++ {
		freqs[i] = math.Pow(10000.0, -float64(2*i)/float64(embedDim))
	}
	data := make([]float32, B*embedDim)
	for b := int64(0); b < B; b++ {
		t := float64(tValues[b])
		for i := int64(0); i < half; i++ {
			ang := t * freqs[i]
			data[b*embedDim+2*i] = float32(math.Sin(ang))
			data[b*embedDim+2*i+1] = float32(math.Cos(ang))
		}
	}
	leaf := tensor.NewLeaf(a, []int64{B, embedDim}, dtype, device)
	leaf.SetData(data)
	return leaf
}

// ── DDPMDenoiser ─────────────────────────────────────────────────────────────

// DDPMDenoiser is a tiny constant-resolution conv denoiser that predicts the
// noise component of a noised input x_t given a time embedding. Three 3×3
// conv blocks at constant spatial resolution with SiLU activations and a
// per-block time-embedding projection added as a channel-broadcast bias.
//
// Architecture:
//
//	Conv1: [N, InCh, H, W] → [N, C, H, W]
//	+ TEmbProj1(tEmb) broadcast as [N, C, 1, 1] → [N, C, H, W]
//	SiLU
//	Conv2: [N, C, H, W] → [N, C, H, W]
//	+ TEmbProj2(tEmb) broadcast as [N, C, 1, 1] → [N, C, H, W]
//	SiLU
//	Conv3: [N, C, H, W] → [N, InCh, H, W]  (predicted noise)
type DDPMDenoiser struct {
	Conv1     *Conv2d
	TEmbProj1 *Linear
	Conv2     *Conv2d
	TEmbProj2 *Linear
	Conv3     *Conv2d

	InCh         int64
	Channels     int64
	TimeEmbedDim int64
}

// NewDDPMDenoiser allocates the layer parameter shapes. Weights are
// zero-initialised; callers seed them before the first Load.
func NewDDPMDenoiser(a *uop.Arena, inCh, channels, timeEmbedDim int64, dtype *uop.DType, device string) *DDPMDenoiser {
	if inCh <= 0 || channels <= 0 || timeEmbedDim <= 0 {
		panic(fmt.Sprintf("nn: NewDDPMDenoiser: dims must be positive, got inCh=%d channels=%d timeEmbedDim=%d",
			inCh, channels, timeEmbedDim))
	}
	return &DDPMDenoiser{
		Conv1:        NewConv2d(a, inCh, channels, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, dtype, device),
		TEmbProj1:    NewLinear(a, timeEmbedDim, channels, true, dtype, device),
		Conv2:        NewConv2d(a, channels, channels, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, dtype, device),
		TEmbProj2:    NewLinear(a, timeEmbedDim, channels, true, dtype, device),
		Conv3:        NewConv2d(a, channels, inCh, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, dtype, device),
		InCh:         inCh,
		Channels:     channels,
		TimeEmbedDim: timeEmbedDim,
	}
}

// Forward predicts the noise component of x given a per-sample time embedding.
// x shape: [N, InCh, H, W]; tEmb shape: [N, TimeEmbedDim]. Output: [N, InCh, H, W].
func (d *DDPMDenoiser) Forward(x, tEmb *tensor.Tensor) *tensor.Tensor {
	if x.Rank() != 4 {
		panic(fmt.Sprintf("nn: DDPMDenoiser.Forward: x must be rank 4, got rank %d", x.Rank()))
	}
	if tEmb.Rank() != 2 {
		panic(fmt.Sprintf("nn: DDPMDenoiser.Forward: tEmb must be rank 2, got rank %d", tEmb.Rank()))
	}
	xShape := x.Shape()
	N, _, H, W := xShape[0], xShape[1], xShape[2], xShape[3]

	h := d.Conv1.Forward(x)
	h = h.Add(d.broadcastTEmb(d.TEmbProj1.Forward(tEmb), N, d.Channels, H, W))
	h = SiLU(h)

	h = d.Conv2.Forward(h)
	h = h.Add(d.broadcastTEmb(d.TEmbProj2.Forward(tEmb), N, d.Channels, H, W))
	h = SiLU(h)

	return d.Conv3.Forward(h)
}

// broadcastTEmb reshapes a [N, C] time-embedding projection into [N, C, 1, 1]
// and expands it to [N, C, H, W] for per-channel additive bias.
func (d *DDPMDenoiser) broadcastTEmb(proj *tensor.Tensor, N, C, H, W int64) *tensor.Tensor {
	return proj.Reshape([]int64{N, C, 1, 1}).Expand([]int64{N, C, H, W})
}

// Params returns the trainable parameters in deterministic order:
// Conv1.W, Conv1.B, TEmbProj1.W, TEmbProj1.B, Conv2.W, Conv2.B,
// TEmbProj2.W, TEmbProj2.B, Conv3.W, Conv3.B.
func (d *DDPMDenoiser) Params() []*Parameter {
	out := make([]*Parameter, 0, 10)
	out = append(out, d.Conv1.Params()...)
	out = append(out, d.TEmbProj1.Params()...)
	out = append(out, d.Conv2.Params()...)
	out = append(out, d.TEmbProj2.Params()...)
	out = append(out, d.Conv3.Params()...)
	return out
}
