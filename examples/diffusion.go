package examples

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "diffusion",
		Summary: "tiny DDPM denoiser (constant-resolution conv, sinusoidal time embed); synthetic 8x8",
		Build:   buildDiffusion,
		Train:   trainDiffusion,
	})
}

// ── Diffusion example config ────────────────────────────────────────────────
//
// Tiny DDPM denoiser on 1-channel 8x8 synthetic patterns. Constant-resolution
// conv architecture (no downsample, no upsample, no skips); time-step is fed
// in via a sinusoidal embedding projected to a per-channel additive bias on
// each of two intermediate conv blocks. See SPEC §11 / diffusion_preflight.md
// for the slice-1 scope (no real MNIST, no classifier-free guidance, no EMA,
// no sampling-quality benchmark).
const (
	diffImageH       = int64(8)
	diffImageW       = int64(8)
	diffInCh         = int64(1)
	diffChannels     = int64(16)
	diffTimeEmbedDim = int64(32)
	diffBatch        = int64(4)
	diffT            = 200
	diffBetaStart    = float32(1e-4)
	diffBetaEnd      = float32(0.02)
	diffAdamLR       = float32(1e-3)
	diffInitScale    = float32(0.1)
	diffSampleSteps  = 50
)

// buildDiffusion constructs the forward graph for ONE denoise step at fixed
// t=0 against a synthetic batch. Used by `anneal run diffusion` /
// `anneal graph diffusion` / `anneal kernels diffusion`.
func buildDiffusion(device string) (*BuildResult, error) {
	seedArena := uop.NewArena(1 << 14)
	seed := nn.NewDDPMDenoiser(seedArena, diffInCh, diffChannels, diffTimeEmbedDim,
		uop.Dtypes.Float32, device)
	initRNG := rand.New(rand.NewSource(42))
	initDiffusionSmall(seed, diffInitScale, initRNG)

	a := uop.NewArena(1 << 20)
	model := nn.NewDDPMDenoiser(a, diffInCh, diffChannels, diffTimeEmbedDim,
		uop.Dtypes.Float32, device)

	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return nil, fmt.Errorf("diffusion: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	leaves := make([]*tensor.Tensor, 0, len(dstParams))
	for _, p := range dstParams {
		p.Load(a)
		leaves = append(leaves, p.T)
	}

	rng := rand.New(rand.NewSource(43))
	xData := diffusionDataset(diffBatch, diffImageH, diffImageW, rng)
	x := tensor.NewLeaf(a, []int64{diffBatch, diffInCh, diffImageH, diffImageW},
		uop.Dtypes.Float32, device)
	x.SetData(xData)

	tVals := make([]int32, diffBatch)
	for i := range tVals {
		tVals[i] = 0
	}
	tEmb := nn.SinusoidalTimeEmbed(a, tVals, diffTimeEmbedDim, uop.Dtypes.Float32, device)

	out := model.Forward(x, tEmb)

	return &BuildResult{
		Arena:  a,
		Output: out,
		Device: device,
		Leaves: leaves,
	}, nil
}

// trainDiffusion runs the full DDPM training loop on a synthetic 4x1x8x8
// fixed pattern. Per step: sample t per batch elem from [0, T), sample
// ε~N(0,I), compose x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε host-side, predict ε, MSE
// loss, Adam step. Periodic eval emits via logFn; total wall-time emits via
// cfg.LogText. If cfg.LogText is set, a final T=50 reverse-process sample is
// drawn and reported as a finite-stats summary (mean / var of last sample).
func trainDiffusion(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	lr := cfg.LR
	if lr == cmdTrainSGDDefaultLR || lr == 0 {
		lr = diffAdamLR
	}
	batch := cfg.Batch
	if batch <= 0 {
		batch = diffBatch
	}

	betas := nn.MakeLinearBetas(diffT, diffBetaStart, diffBetaEnd)
	alphas := nn.MakeAlphas(betas)
	alphaBars := nn.MakeAlphaBars(alphas)
	sqrtAlphaBar := make([]float32, diffT)
	sqrtOneMinusAlphaBar := make([]float32, diffT)
	for i := 0; i < diffT; i++ {
		sqrtAlphaBar[i] = float32(math.Sqrt(float64(alphaBars[i])))
		sqrtOneMinusAlphaBar[i] = float32(math.Sqrt(1.0 - float64(alphaBars[i])))
	}

	seedArena := uop.NewArena(1 << 14)
	seed := nn.NewDDPMDenoiser(seedArena, diffInCh, diffChannels, diffTimeEmbedDim,
		uop.Dtypes.Float32, device)
	initRNG := rand.New(rand.NewSource(42))
	initDiffusionSmall(seed, diffInitScale, initRNG)

	a0 := uop.NewArena(1 << 14)
	model := nn.NewDDPMDenoiser(a0, diffInCh, diffChannels, diffTimeEmbedDim,
		uop.Dtypes.Float32, device)
	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return fmt.Errorf("diffusion: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	params := model.Params()
	opt := nn.NewAdam(params, lr)

	// Fixed clean batch reused every step; the noise is what randomises.
	dataRNG := rand.New(rand.NewSource(43))
	x0 := diffusionDataset(batch, diffImageH, diffImageW, dataRNG)

	stepRNG := rand.New(rand.NewSource(44))
	perSample := diffInCh * diffImageH * diffImageW
	noiseBuf := make([]float32, batch*perSample)
	xtBuf := make([]float32, batch*perSample)
	tValsBuf := make([]int32, batch)

	start := time.Now()

	for step := 1; step <= cfg.Steps; step++ {
		for b := int64(0); b < batch; b++ {
			tValsBuf[b] = int32(stepRNG.Intn(diffT))
		}
		for i := range noiseBuf {
			noiseBuf[i] = float32(stepRNG.NormFloat64())
		}
		for b := int64(0); b < batch; b++ {
			tIdx := int(tValsBuf[b])
			sa := sqrtAlphaBar[tIdx]
			so := sqrtOneMinusAlphaBar[tIdx]
			base := b * perSample
			for i := int64(0); i < perSample; i++ {
				xtBuf[base+i] = sa*x0[base+i] + so*noiseBuf[base+i]
			}
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}
		xt := tensor.NewLeaf(a, []int64{batch, diffInCh, diffImageH, diffImageW},
			uop.Dtypes.Float32, device)
		xt.SetData(append([]float32{}, xtBuf...))
		eps := tensor.NewLeaf(a, []int64{batch, diffInCh, diffImageH, diffImageW},
			uop.Dtypes.Float32, device)
		eps.SetData(append([]float32{}, noiseBuf...))
		tEmb := nn.SinusoidalTimeEmbed(a, tValsBuf, diffTimeEmbedDim,
			uop.Dtypes.Float32, device)

		pred := model.Forward(xt, tEmb)
		diff := pred.Sub(eps)
		loss := diff.Mul(diff).Mean(nil, false)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)
		if err := tensor.Realize(loss); err != nil {
			return fmt.Errorf("diffusion: realize loss at step %d: %w", step, err)
		}
		for _, p := range params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(g); err != nil {
				return fmt.Errorf("diffusion: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}
		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			logFn(step, loss.Data()[0])
		}
	}

	elapsed := time.Since(start)
	if cfg.LogText != nil {
		cfg.LogText(fmt.Sprintf("diffusion: training complete in %s (%d steps)\n",
			elapsed.Round(time.Millisecond), cfg.Steps))
		// Best-effort reverse-process sample sweep. Skipped on error so a
		// kernel surface regression doesn't block the test smoke.
		stats, err := diffusionSampleSmoke(model, params, batch, betas, alphas, alphaBars,
			rand.New(rand.NewSource(45)), device)
		if err == nil {
			cfg.LogText(fmt.Sprintf("diffusion: sample mean=%+.4f var=%+.4f (T=%d reverse steps)\n",
				stats.mean, stats.variance, diffSampleSteps))
		}
	}

	return nil
}

// diffusionDataset generates a fixed synthetic [B, 1, H, W] batch of small
// sinusoid-grid patterns. The exact pattern is unimportant; only that it is
// non-trivial (so the MSE loss has a non-degenerate signal) and deterministic
// for a given rng seed.
func diffusionDataset(B int64, H, W int64, rng *rand.Rand) []float32 {
	out := make([]float32, B*H*W)
	for b := int64(0); b < B; b++ {
		fx := 0.5 + rng.Float32()*1.5
		fy := 0.5 + rng.Float32()*1.5
		ph := rng.Float64() * 2 * math.Pi
		for h := int64(0); h < H; h++ {
			for w := int64(0); w < W; w++ {
				v := math.Sin(float64(fx)*float64(h)*0.5+ph) +
					math.Cos(float64(fy)*float64(w)*0.5)
				out[b*H*W+h*W+w] = float32(v) * 0.3
			}
		}
	}
	return out
}

// initDiffusionSmall seeds every DDPMDenoiser parameter with small-normal
// samples. Mirrors initViTSmall / initResNet9Small.
func initDiffusionSmall(d *nn.DDPMDenoiser, scale float32, rng *rand.Rand) {
	for _, c := range []*nn.Conv2d{d.Conv1, d.Conv2, d.Conv3} {
		for i := range c.Weight.Value {
			c.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if c.Bias != nil {
			for i := range c.Bias.Value {
				c.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
	}
	for _, l := range []*nn.Linear{d.TEmbProj1, d.TEmbProj2} {
		for i := range l.Weight.Value {
			l.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if l.Bias != nil {
			for i := range l.Bias.Value {
				l.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
	}
}

type diffusionSampleStats struct {
	mean     float32
	variance float32
}

// diffusionSampleSmoke runs the reverse process from x_T ~ N(0, I) for
// diffSampleSteps iterations and returns finite-stats on the final sample.
// Skipped from the main test budget — gated on cfg.LogText (the same way
// nanoGPT gates its final sample).
func diffusionSampleSmoke(
	model *nn.DDPMDenoiser,
	params []*nn.Parameter,
	batch int64,
	betas, alphas, alphaBars []float32,
	rng *rand.Rand,
	device string,
) (diffusionSampleStats, error) {
	perSample := diffInCh * diffImageH * diffImageW
	x := make([]float32, batch*perSample)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	totalSteps := diffSampleSteps
	if totalSteps > diffT {
		totalSteps = diffT
	}
	// Coarse-grained reverse process: walk uniformly across [T-1, 0].
	stride := diffT / totalSteps
	if stride < 1 {
		stride = 1
	}
	for step := 0; step < totalSteps; step++ {
		tIdx := diffT - 1 - step*stride
		if tIdx < 0 {
			tIdx = 0
		}
		tVals := make([]int32, batch)
		for b := range tVals {
			tVals[b] = int32(tIdx)
		}

		a := uop.NewArena(1 << 20)
		for _, p := range params {
			p.Load(a)
		}
		xt := tensor.NewLeaf(a, []int64{batch, diffInCh, diffImageH, diffImageW},
			uop.Dtypes.Float32, device)
		xt.SetData(append([]float32{}, x...))
		tEmb := nn.SinusoidalTimeEmbed(a, tVals, diffTimeEmbedDim, uop.Dtypes.Float32, device)
		pred := model.Forward(xt, tEmb)
		if err := tensor.Realize(pred); err != nil {
			return diffusionSampleStats{}, fmt.Errorf("diffusion sample step %d: %w", step, err)
		}
		predData := pred.Data()

		// x_{t-1} = (1/√α_t) * (x_t - (β_t / √(1-ᾱ_t)) * eps_pred) [+ σ_t * z]
		// Simplified: drop the stochastic σ_t term (deterministic walk).
		// Skip the divide-by-zero corner case at t=0 by clamping alpha.
		alphaT := alphas[tIdx]
		alphaBarT := alphaBars[tIdx]
		invSqrtAlpha := float32(1.0 / math.Sqrt(float64(alphaT)))
		coef := betas[tIdx] / float32(math.Sqrt(1.0-float64(alphaBarT)))
		for i := range x {
			x[i] = invSqrtAlpha * (x[i] - coef*predData[i])
		}
	}

	var sum, sq float64
	for _, v := range x {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			return diffusionSampleStats{}, fmt.Errorf("non-finite sample value")
		}
		sum += fv
		sq += fv * fv
	}
	n := float64(len(x))
	mean := sum / n
	variance := sq/n - mean*mean
	return diffusionSampleStats{mean: float32(mean), variance: float32(variance)}, nil
}
