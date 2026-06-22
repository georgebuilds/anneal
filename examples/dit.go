package examples

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	resnet9data "github.com/georgebuilds/anneal/examples/resnet9"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "dit",
		Summary: "Diffusion Transformer (adaLN-zero, classifier-free guidance) on CIFAR-10",
		Build:   buildDiT,
		Train:   trainDiT,
	})
}

// ── DiT example config ───────────────────────────────────────────────────────
//
// Class-conditional Diffusion Transformer (Peebles & Xie, 2022) trained with
// epsilon-prediction on CIFAR-10 (32x32 RGB). Architecture: patch-embed the
// image into a token grid, condition every adaLN-zero block on c = timeEmbed +
// classEmbed, predict the noise, unpatchify. Class conditioning is a one-hot
// projection (equivalent to an embedding table) so classifier-free guidance is
// a single null-class row; CFG dropout during training lets sampling trade off
// fidelity against diversity. The schedule, sinusoidal time embedding, noising,
// and DDPM reverse process reuse tensor/nn's diffusion helpers.
type ditConfig struct {
	imageH, imageW int64
	patch          int64
	inCh           int64 // == outCh for eps-prediction
	embedDim       int64
	condDim        int64
	timeEmbedDim   int64
	numClasses     int64
	nLayer         int
	nHead          int
	T              int
	betaStart      float32
	betaEnd        float32
	adamLR         float32
	initScale      float32
	cfgDropProb    float32 // probability of dropping the class label to null during training
}

// ditDefaultConfig is the production CIFAR-10 configuration.
func ditDefaultConfig() ditConfig {
	return ditConfig{
		imageH: 32, imageW: 32, patch: 4, inCh: 3,
		embedDim: 64, condDim: 64, timeEmbedDim: 64, numClasses: 10,
		nLayer: 2, nHead: 4, T: 200,
		betaStart: 1e-4, betaEnd: 0.02, adamLR: 1e-3, initScale: 0.02,
		cfgDropProb: 0.1,
	}
}

const (
	ditBatch       = int64(2) // tiny by default; the WGSL backward surface is the ceiling
	ditSampleSteps = 50
	ditGuidance    = float32(2.0)
)

// ── ditModel: DiT core plus the conditioning embedders ───────────────────────
//
// The nn.DiT container takes a [B, condDim] conditioning vector; ditModel owns
// the two embedders that assemble it: a timestep projection over the sinusoidal
// time embedding and a class projection over a one-hot label (numClasses + 1
// rows, the last being the CFG null class). cond = timeProj(t) + classProj(y).
type ditModel struct {
	core      *nn.DiT
	timeProj  *nn.Linear // timeEmbedDim -> condDim
	classProj *nn.Linear // (numClasses+1) -> condDim, no bias (one-hot input)
	dc        ditConfig
}

func newDitModel(a *uop.Arena, dc ditConfig, device string) *ditModel {
	dtype := uop.Dtypes.Float32
	return &ditModel{
		core: nn.NewDiT(a, dc.imageH, dc.imageW, dc.patch, dc.inCh, dc.inCh,
			dc.embedDim, dc.condDim, dc.nLayer, dc.nHead),
		timeProj:  nn.NewLinear(a, dc.timeEmbedDim, dc.condDim, true, dtype, device),
		classProj: nn.NewLinear(a, dc.numClasses+1, dc.condDim, false, dtype, device),
		dc:        dc,
	}
}

// Params returns every trainable parameter in deterministic order: the DiT core,
// then the timestep projection, then the class projection.
func (m *ditModel) Params() []*nn.Parameter {
	ps := m.core.Params()
	ps = append(ps, m.timeProj.Params()...)
	ps = append(ps, m.classProj.Params()...)
	return ps
}

// Forward predicts the noise for noised images x at timesteps tVals, conditioned
// on the one-hot class rows ohData ([B*(numClasses+1)] row-major). Builds the
// time/one-hot leaves in the given arena; params must already be Loaded into a.
func (m *ditModel) Forward(a *uop.Arena, device string, x *tensor.Tensor, tVals []int32, ohData []float32) *tensor.Tensor {
	dc := m.dc
	tEmb := nn.SinusoidalTimeEmbed(a, tVals, dc.timeEmbedDim, uop.Dtypes.Float32, device)
	tc := m.timeProj.Forward(tEmb)

	B := int64(len(tVals))
	oh := tensor.NewLeaf(a, []int64{B, dc.numClasses + 1}, uop.Dtypes.Float32, device)
	oh.SetData(append([]float32{}, ohData...))
	cc := m.classProj.Forward(oh)

	cond := tc.Add(cc)
	return m.core.Forward(x, cond)
}

// buildDiT constructs the forward graph for one denoise step at t=0 on a
// synthetic batch, used by `anneal run dit` / `anneal graph dit` /
// `anneal kernels dit`.
func buildDiT(device string) (*BuildResult, error) {
	dc := ditDefaultConfig()

	seedArena := uop.NewArena(1 << 16)
	seed := newDitModel(seedArena, dc, device)
	initDitSmall(seed, dc, rand.New(rand.NewSource(42)))

	a := uop.NewArena(1 << 22)
	model := newDitModel(a, dc, device)
	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return nil, fmt.Errorf("dit: param-count mismatch between seed (%d) and compute (%d) models",
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

	B := ditBatch
	rng := rand.New(rand.NewSource(43))
	xData := make([]float32, B*dc.inCh*dc.imageH*dc.imageW)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.3
	}
	x := tensor.NewLeaf(a, []int64{B, dc.inCh, dc.imageH, dc.imageW}, uop.Dtypes.Float32, device)
	x.SetData(xData)

	tVals := make([]int32, B) // all t=0
	oh := make([]float32, B*(dc.numClasses+1))
	ditClassOneHot(make([]int32, B), dc.numClasses, 0, rng, oh) // class 0, no dropout

	out := model.Forward(a, device, x, tVals, oh)
	return &BuildResult{Arena: a, Output: out, Device: device, Leaves: leaves}, nil
}

// trainDiT loads CIFAR-10 and runs the full DiT training loop.
func trainDiT(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	ds, err := resnet9data.Load()
	if err != nil {
		return fmt.Errorf("dit: load CIFAR-10: %w", err)
	}
	return runDiT(device, cfg, logFn, ds, ditDefaultConfig(), 42)
}

// runDiT is the shared trainer used by the production entry point (CIFAR-10 +
// default config) and the smoke tests (in-memory fixture + scaled-down config),
// mirroring runResNet9. Per step: sample t and a class-dropout mask, draw noise,
// compose x_t host-side, predict eps, MSE loss, Adam step.
func runDiT(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *resnet9data.CIFAR10,
	dc ditConfig,
	seed int64,
) error {
	lr := cfg.LR
	if lr == cmdTrainSGDDefaultLR || lr == 0 {
		lr = dc.adamLR
	}
	batch := cfg.Batch
	if batch <= 0 {
		batch = ditBatch
	}

	betas := nn.MakeLinearBetas(dc.T, dc.betaStart, dc.betaEnd)
	alphas := nn.MakeAlphas(betas)
	alphaBars := nn.MakeAlphaBars(alphas)
	sqrtAlphaBar := make([]float32, dc.T)
	sqrtOneMinusAlphaBar := make([]float32, dc.T)
	for i := 0; i < dc.T; i++ {
		sqrtAlphaBar[i] = float32(math.Sqrt(float64(alphaBars[i])))
		sqrtOneMinusAlphaBar[i] = float32(math.Sqrt(1.0 - float64(alphaBars[i])))
	}

	seedArena := uop.NewArena(1 << 16)
	seedModel := newDitModel(seedArena, dc, device)
	initDitSmall(seedModel, dc, rand.New(rand.NewSource(seed)))

	a0 := uop.NewArena(1 << 16)
	model := newDitModel(a0, dc, device)
	srcParams := seedModel.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return fmt.Errorf("dit: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	params := model.Params()
	opt := nn.NewAdam(params, lr)

	sampleRNG := rand.New(rand.NewSource(seed + 1))
	stepRNG := rand.New(rand.NewSource(seed + 2))

	C, H, W := dc.inCh, dc.imageH, dc.imageW
	perSample := C * H * W
	cols := dc.numClasses + 1
	xHost := make([]float32, batch*perSample)
	yHost := make([]int32, batch)
	noiseBuf := make([]float32, batch*perSample)
	xtBuf := make([]float32, batch*perSample)
	tValsBuf := make([]int32, batch)
	ohBuf := make([]float32, batch*cols)

	if cfg.LogEvery > 0 {
		l0 := ditEvalLoss(model, params, ds, sampleRNG, batch, dc, sqrtAlphaBar, sqrtOneMinusAlphaBar, device)
		logFn(0, l0)
	}

	start := time.Now()

	for step := 1; step <= cfg.Steps; step++ {
		ds.Batch(sampleRNG, int(batch), xHost, yHost)
		for b := int64(0); b < batch; b++ {
			tValsBuf[b] = int32(stepRNG.Intn(dc.T))
		}
		for i := range noiseBuf {
			noiseBuf[i] = float32(stepRNG.NormFloat64())
		}
		for b := int64(0); b < batch; b++ {
			tIdx := int(tValsBuf[b])
			sa, so := sqrtAlphaBar[tIdx], sqrtOneMinusAlphaBar[tIdx]
			base := b * perSample
			for i := int64(0); i < perSample; i++ {
				xtBuf[base+i] = sa*xHost[base+i] + so*noiseBuf[base+i]
			}
		}
		ditClassOneHot(yHost, dc.numClasses, dc.cfgDropProb, stepRNG, ohBuf)

		a := uop.NewArena(1 << 22)
		for _, p := range params {
			p.Load(a)
		}
		xt := tensor.NewLeaf(a, []int64{batch, C, H, W}, uop.Dtypes.Float32, device)
		xt.SetData(append([]float32{}, xtBuf...))
		epsTarget := tensor.NewLeaf(a, []int64{batch, C, H, W}, uop.Dtypes.Float32, device)
		epsTarget.SetData(append([]float32{}, noiseBuf...))

		pred := model.Forward(a, device, xt, tValsBuf, ohBuf)
		diff := pred.Sub(epsTarget)
		loss := diff.Mul(diff).Mean(nil, false)

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)
		if err := tensor.Realize(loss); err != nil {
			return fmt.Errorf("dit: realize loss at step %d: %w", step, err)
		}
		for _, p := range params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(g); err != nil {
				return fmt.Errorf("dit: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}
		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			lp := ditEvalLoss(model, params, ds, sampleRNG, batch, dc, sqrtAlphaBar, sqrtOneMinusAlphaBar, device)
			logFn(step, lp)
		}
	}

	elapsed := time.Since(start)
	if cfg.LogText != nil {
		cfg.LogText(fmt.Sprintf("dit: training complete in %s (%d steps)\n", elapsed.Round(time.Millisecond), cfg.Steps))
		stats, err := ditSampleSmoke(model, params, batch, dc, betas, alphas, alphaBars,
			rand.New(rand.NewSource(seed+3)), device)
		if err == nil {
			cfg.LogText(fmt.Sprintf("dit: CFG sample mean=%+.4f var=%+.4f (guidance=%.1f, %d reverse steps)\n",
				stats.mean, stats.variance, ditGuidance, ditSampleSteps))
		}
	}
	return nil
}

// ditClassOneHot writes a [B, numClasses+1] one-hot block from labels, applying
// classifier-free-guidance dropout: with probability dropProb a sample is routed
// to the null class (index numClasses) so the model also learns the
// unconditional score.
func ditClassOneHot(labels []int32, numClasses int64, dropProb float32, rng *rand.Rand, out []float32) {
	cols := numClasses + 1
	for i := range out {
		out[i] = 0
	}
	for b := range labels {
		cls := int64(labels[b])
		if cls < 0 || cls >= numClasses {
			cls = numClasses // out-of-range guard -> null
		}
		if dropProb > 0 && rng.Float32() < dropProb {
			cls = numClasses
		}
		out[int64(b)*cols+cls] = 1
	}
}

// initDitSmall seeds every parameter with small-normal samples, then restores
// the adaLN-zero invariant: each block's modulation projection, the final
// modulation, and the final linear stay at zero so the network starts as the
// identity flow predicting zero noise.
func initDitSmall(m *ditModel, dc ditConfig, rng *rand.Rand) {
	for _, p := range m.Params() {
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * dc.initScale
		}
	}
	zero := func(p *nn.Parameter) {
		for i := range p.Value {
			p.Value[i] = 0
		}
	}
	for _, blk := range m.core.Blocks {
		for _, p := range blk.Mod.Params() {
			zero(p)
		}
	}
	for _, p := range m.core.FinalMod.Params() {
		zero(p)
	}
	for _, p := range m.core.FinalLin.Params() {
		zero(p)
	}
}

// ditEvalLoss recomputes the eps-prediction MSE for one freshly-sampled batch in
// a fresh arena, mirroring resnet9EvalLoss.
func ditEvalLoss(
	m *ditModel,
	params []*nn.Parameter,
	ds *resnet9data.CIFAR10,
	rng *rand.Rand,
	B int64,
	dc ditConfig,
	sqrtAlphaBar, sqrtOneMinusAlphaBar []float32,
	device string,
) float32 {
	C, H, W := dc.inCh, dc.imageH, dc.imageW
	perSample := C * H * W
	cols := dc.numClasses + 1
	xHost := make([]float32, B*perSample)
	yHost := make([]int32, B)
	noiseBuf := make([]float32, B*perSample)
	xtBuf := make([]float32, B*perSample)
	tValsBuf := make([]int32, B)
	ohBuf := make([]float32, B*cols)

	ds.Batch(rng, int(B), xHost, yHost)
	for b := int64(0); b < B; b++ {
		tValsBuf[b] = int32(rng.Intn(dc.T))
	}
	for i := range noiseBuf {
		noiseBuf[i] = float32(rng.NormFloat64())
	}
	for b := int64(0); b < B; b++ {
		tIdx := int(tValsBuf[b])
		sa, so := sqrtAlphaBar[tIdx], sqrtOneMinusAlphaBar[tIdx]
		base := b * perSample
		for i := int64(0); i < perSample; i++ {
			xtBuf[base+i] = sa*xHost[base+i] + so*noiseBuf[base+i]
		}
	}
	ditClassOneHot(yHost, dc.numClasses, 0, rng, ohBuf) // eval: conditional, no dropout

	a := uop.NewArena(1 << 22)
	for _, p := range params {
		p.Load(a)
	}
	xt := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, device)
	xt.SetData(xtBuf)
	epsTarget := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, device)
	epsTarget.SetData(noiseBuf)

	pred := m.Forward(a, device, xt, tValsBuf, ohBuf)
	diff := pred.Sub(epsTarget)
	loss := diff.Mul(diff).Mean(nil, false)
	if err := tensor.Realize(loss); err != nil {
		return float32(math.NaN())
	}
	return loss.Data()[0]
}

type ditSampleStats struct {
	mean     float32
	variance float32
}

// ditSampleSmoke runs the classifier-free-guidance DDPM reverse process from
// x_T ~ N(0, I) and returns finite stats on the final sample. Each step runs the
// model twice (conditional on class 0, unconditional via the null class) and
// combines: eps = eps_uncond + guidance * (eps_cond - eps_uncond). Gated on
// cfg.LogText like the diffusion / nanoGPT samples.
func ditSampleSmoke(
	m *ditModel,
	params []*nn.Parameter,
	batch int64,
	dc ditConfig,
	betas, alphas, alphaBars []float32,
	rng *rand.Rand,
	device string,
) (ditSampleStats, error) {
	C, H, W := dc.inCh, dc.imageH, dc.imageW
	perSample := C * H * W
	cols := dc.numClasses + 1
	x := make([]float32, batch*perSample)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	// Conditional one-hot (class 0) and unconditional one-hot (null class).
	condOH := make([]float32, batch*cols)
	uncondOH := make([]float32, batch*cols)
	for b := int64(0); b < batch; b++ {
		condOH[b*cols+0] = 1
		uncondOH[b*cols+dc.numClasses] = 1
	}

	totalSteps := ditSampleSteps
	if totalSteps > dc.T {
		totalSteps = dc.T
	}
	stride := dc.T / totalSteps
	if stride < 1 {
		stride = 1
	}

	for step := 0; step < totalSteps; step++ {
		tIdx := dc.T - 1 - step*stride
		if tIdx < 0 {
			tIdx = 0
		}
		tVals := make([]int32, batch)
		for b := range tVals {
			tVals[b] = int32(tIdx)
		}

		predEps := func(oh []float32) ([]float32, error) {
			a := uop.NewArena(1 << 22)
			for _, p := range params {
				p.Load(a)
			}
			xt := tensor.NewLeaf(a, []int64{batch, C, H, W}, uop.Dtypes.Float32, device)
			xt.SetData(append([]float32{}, x...))
			pred := m.Forward(a, device, xt, tVals, oh)
			if err := tensor.Realize(pred); err != nil {
				return nil, err
			}
			return pred.Data(), nil
		}

		epsCond, err := predEps(condOH)
		if err != nil {
			return ditSampleStats{}, fmt.Errorf("dit sample step %d (cond): %w", step, err)
		}
		epsUncond, err := predEps(uncondOH)
		if err != nil {
			return ditSampleStats{}, fmt.Errorf("dit sample step %d (uncond): %w", step, err)
		}

		alphaT := alphas[tIdx]
		alphaBarT := alphaBars[tIdx]
		invSqrtAlpha := float32(1.0 / math.Sqrt(float64(alphaT)))
		coef := betas[tIdx] / float32(math.Sqrt(1.0-float64(alphaBarT)))
		for i := range x {
			eps := epsUncond[i] + ditGuidance*(epsCond[i]-epsUncond[i])
			x[i] = invSqrtAlpha * (x[i] - coef*eps)
		}
	}

	var sum, sq float64
	for _, v := range x {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			return ditSampleStats{}, fmt.Errorf("non-finite sample value")
		}
		sum += fv
		sq += fv * fv
	}
	n := float64(len(x))
	mean := sum / n
	return ditSampleStats{mean: float32(mean), variance: float32(sq/n - mean*mean)}, nil
}
