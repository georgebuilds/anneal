package examples

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	resnet9data "github.com/georgebuilds/anneal/examples/resnet9"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/tensor/safetensors"
	"github.com/georgebuilds/anneal/uop"
)

// MeanFlow (Geng et al. 2024, one-step generative modeling) trains a network to
// predict the AVERAGE velocity u(z_t, r, t) over an interval [r, t] of a
// flow-matching trajectory, instead of the instantaneous velocity. The defining
// identity,
//
//	u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t)   (stop-grad on the RHS),
//
// needs the TOTAL time-derivative of the network along the trajectory,
//
//	d/dt u = (du/dz) * v + (du/dr) * 0 + (du/dt_explicit) * 1,
//
// which is exactly ONE forward-mode JVP of the network forward with tangents
// (v, 0, 1) on (z, r, t). That is the capability tensor.JVP adds; MeanFlow is the
// example that uses it. Training is GPU-only (the CPU interpreter realizes neither
// the diffusion/DiT backward nor Sin); the backbone is the DiT from dit.go.

func init() {
	Register(&Example{
		Name:    "meanflow",
		Summary: "MeanFlow one-step generative model (average-velocity, forward-mode JVP) on CIFAR-10",
		Build:   buildMeanflow,
		Train:   trainMeanflow,
	})
}

const meanflowBatch = int64(2) // tiny by default; the WGSL backward surface is the ceiling

// meanflowConfig is the MeanFlow + DiT-backbone hyperparameter set. Unlike DiT it
// carries no diffusion schedule (time is continuous in [0,1]); it adds pEqual, the
// fraction of samples drawn with r == t so the model also learns the instantaneous
// velocity at the interval's collapse.
type meanflowConfig struct {
	imageH, imageW int64
	patch, inCh    int64
	embedDim       int64
	condDim        int64
	timeEmbedDim   int64
	numClasses     int64
	nLayer, nHead  int64
	adamLR         float32
	initScale      float32
	cfgDropProb    float32
	pEqual         float32
}

func meanflowDefaultConfig() meanflowConfig {
	return meanflowConfig{
		imageH: 32, imageW: 32, patch: 4, inCh: 3,
		embedDim: 64, condDim: 64, timeEmbedDim: 64, numClasses: 10,
		nLayer: 2, nHead: 4,
		adamLR: 1e-3, initScale: 0.02, cfgDropProb: 0.1, pEqual: 0.25,
	}
}

// meanflowTimeEmbed builds a sinusoidal embedding of a CONTINUOUS time t as an
// in-graph function, so a JVP can flow a tangent through t. (nn.SinusoidalTimeEmbed
// precomputes sin/cos host-side from integer steps and injects a constant leaf, so
// no tangent can flow through it.) t is [B,1]; the result is [B, embedDim] laid out
// as [sin(t*freqs) | cos(t*freqs)] with freqs = 10000^(-2i/embedDim). The sin/cos
// halves are assembled with Pad+Add since there is no concat op; every op here
// (Mul, Sin, Add, Pad) has a JVP rule, so the embedding is differentiable in t.
func meanflowTimeEmbed(a *uop.Arena, t *tensor.Tensor, embedDim int64, device string) *tensor.Tensor {
	if embedDim <= 0 || embedDim%2 != 0 {
		panic("examples: meanflowTimeEmbed: embedDim must be a positive even number")
	}
	half := embedDim / 2
	freqs := make([]float32, half)
	for i := int64(0); i < half; i++ {
		freqs[i] = float32(math.Pow(10000.0, -float64(2*i)/float64(embedDim)))
	}
	fLeaf := tensor.NewLeaf(a, []int64{1, half}, t.DType(), device)
	fLeaf.SetData(freqs)

	ang := t.Mul(fLeaf) // [B,1] x [1,half] -> [B,half]
	sinPart := ang.Sin()
	cosPart := ang.Add(tensor.FullSints(a, ang.ShapeSints(), math.Pi/2, t.DType(), device)).Sin()

	// Assemble [sin | cos] into [B, embedDim]: pad each half into its slot, add.
	sinPadded := sinPart.Pad([][2]int64{{0, 0}, {0, half}})
	cosPadded := cosPart.Pad([][2]int64{{0, 0}, {half, 0}})
	return sinPadded.Add(cosPadded)
}

// meanflowModel is u_theta(z, r, t, y): the DiT core conditioned on c =
// tProj(embed(t)) + rProj(embed(r)) + classProj(onehot(y)). The time embeddings
// are in-graph (meanflowTimeEmbed) so a JVP can flow a tangent through t.
type meanflowModel struct {
	dc        meanflowConfig
	core      *nn.DiT
	tProj     *nn.Linear // timeEmbedDim -> condDim
	rProj     *nn.Linear // timeEmbedDim -> condDim
	classProj *nn.Linear // numClasses+1 -> condDim, no bias (one-hot input)
}

func newMeanflowModel(a *uop.Arena, dc meanflowConfig, device string) *meanflowModel {
	dtype := uop.Dtypes.Float32
	core := nn.NewDiT(a, dc.imageH, dc.imageW, dc.patch, dc.inCh, dc.inCh,
		dc.embedDim, dc.condDim, int(dc.nLayer), int(dc.nHead))
	return &meanflowModel{
		dc:        dc,
		core:      core,
		tProj:     nn.NewLinear(a, dc.timeEmbedDim, dc.condDim, true, dtype, device),
		rProj:     nn.NewLinear(a, dc.timeEmbedDim, dc.condDim, true, dtype, device),
		classProj: nn.NewLinear(a, dc.numClasses+1, dc.condDim, false, dtype, device),
	}
}

// Params returns every trainable parameter in deterministic order: the DiT core,
// then the t/r/class projections.
func (m *meanflowModel) Params() []*nn.Parameter {
	ps := m.core.Params()
	ps = append(ps, m.tProj.Params()...)
	ps = append(ps, m.rProj.Params()...)
	ps = append(ps, m.classProj.Params()...)
	return ps
}

// Forward predicts the average velocity for states z at interval [r, t], class
// rows ohData. z, tLeaf ([B,1]) and rLeaf ([B,1]) are taken as caller-built leaves
// so a JVP can seed tangents on z and tLeaf; oh is built here. Params must already
// be Loaded into a.
func (m *meanflowModel) Forward(a *uop.Arena, device string, z, tLeaf, rLeaf *tensor.Tensor, ohData []float32) *tensor.Tensor {
	dc := m.dc
	te := meanflowTimeEmbed(a, tLeaf, dc.timeEmbedDim, device)
	re := meanflowTimeEmbed(a, rLeaf, dc.timeEmbedDim, device)

	B := tLeaf.Shape()[0]
	oh := tensor.NewLeaf(a, []int64{B, dc.numClasses + 1}, uop.Dtypes.Float32, device)
	oh.SetData(append([]float32{}, ohData...))

	cond := m.tProj.Forward(te).Add(m.rProj.Forward(re)).Add(m.classProj.Forward(oh))
	return m.core.Forward(z, cond)
}

// initMeanflowSmall seeds all params ~ N(0, initScale^2) then zeroes the adaLN-zero
// modulation + final projection so every block starts as the identity (the "zero"
// in adaLN-zero), mirroring initDitSmall.
func initMeanflowSmall(m *meanflowModel, dc meanflowConfig, rng *rand.Rand) {
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

// buildMeanflow constructs the forward graph for `anneal run/graph/kernels meanflow`.
func buildMeanflow(device string) (*BuildResult, error) {
	dc := meanflowDefaultConfig()

	seedArena := uop.NewArena(1 << 16)
	seed := newMeanflowModel(seedArena, dc, device)
	initMeanflowSmall(seed, dc, rand.New(rand.NewSource(42)))

	a := uop.NewArena(1 << 22)
	model := newMeanflowModel(a, dc, device)
	srcParams := seed.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return nil, fmt.Errorf("meanflow: param-count mismatch between seed (%d) and compute (%d) models",
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

	B := meanflowBatch
	rng := rand.New(rand.NewSource(43))
	zData := make([]float32, B*dc.inCh*dc.imageH*dc.imageW)
	for i := range zData {
		zData[i] = float32(rng.NormFloat64())
	}
	z := tensor.NewLeaf(a, []int64{B, dc.inCh, dc.imageH, dc.imageW}, uop.Dtypes.Float32, device)
	z.SetData(zData)
	tLeaf := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, device)
	tLeaf.SetData(make([]float32, B))
	rLeaf := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, device)
	rLeaf.SetData(make([]float32, B))
	oh := make([]float32, B*(dc.numClasses+1))
	meanflowClassOneHot(make([]int32, B), dc.numClasses, 0, rng, oh)

	out := model.Forward(a, device, z, tLeaf, rLeaf, oh)
	return &BuildResult{Arena: a, Output: out, Device: device, Leaves: leaves}, nil
}

// trainMeanflow loads CIFAR-10 and runs the full MeanFlow training loop.
func trainMeanflow(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	ds, err := resnet9data.Load()
	if err != nil {
		return fmt.Errorf("meanflow: load CIFAR-10: %w", err)
	}
	return runMeanflow(device, cfg, logFn, ds, meanflowDefaultConfig(), 42)
}

// runMeanflow is the shared trainer (CIFAR-10 / default config in production, an
// in-memory fixture / scaled-down config in the smoke). Per step: draw x, a class,
// noise eps, continuous t in [0,1] and r in [0,t] (a pEqual fraction with r==t);
// form the flow-matching state z_t = (1-t)x + t*eps and velocity v = eps - x;
// predict u; compute du/dt as one JVP; regress u against the stop-grad target
// v - (t-r)*du/dt; Adam step.
func runMeanflow(
	device string,
	cfg TrainConfig,
	logFn func(step int, loss float32),
	ds *resnet9data.CIFAR10,
	dc meanflowConfig,
	seed int64,
) error {
	lr := cfg.LR
	if lr == cmdTrainSGDDefaultLR || lr == 0 {
		lr = dc.adamLR
	}
	batch := cfg.Batch
	if batch <= 0 {
		batch = meanflowBatch
	}

	seedArena := uop.NewArena(1 << 16)
	seedModel := newMeanflowModel(seedArena, dc, device)
	initMeanflowSmall(seedModel, dc, rand.New(rand.NewSource(seed)))

	a0 := uop.NewArena(1 << 16)
	model := newMeanflowModel(a0, dc, device)
	srcParams := seedModel.Params()
	dstParams := model.Params()
	if len(srcParams) != len(dstParams) {
		return fmt.Errorf("meanflow: param-count mismatch between seed (%d) and compute (%d) models",
			len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyParam(dstParams[i], srcParams[i])
	}

	params := model.Params()
	opt := nn.NewAdam(params, lr)

	ckpt := meanflowCheckpointPath()
	if _, statErr := os.Stat(ckpt); statErr == nil {
		if err := safetensors.Load(ckpt, meanflowParamMap(params)); err != nil {
			if cfg.LogText != nil {
				cfg.LogText(fmt.Sprintf("meanflow: checkpoint at %s ignored (%v)\n", ckpt, err))
			}
		} else if cfg.LogText != nil {
			cfg.LogText(fmt.Sprintf("meanflow: resumed from checkpoint %s\n", ckpt))
		}
	}

	stepRNG := rand.New(rand.NewSource(seed + 2))
	sampleRNG := rand.New(rand.NewSource(seed + 1))

	C, H, W := dc.inCh, dc.imageH, dc.imageW
	perSample := C * H * W
	cols := dc.numClasses + 1
	xHost := make([]float32, batch*perSample)
	yHost := make([]int32, batch)
	st := meanflowStepBuffers{
		zt:    make([]float32, batch*perSample),
		v:     make([]float32, batch*perSample),
		t:     make([]float32, batch),
		r:     make([]float32, batch),
		tr:    make([]float32, batch),
		oh:    make([]float32, batch*cols),
		batch: batch,
	}

	if cfg.LogEvery > 0 {
		ds.Batch(sampleRNG, int(batch), xHost, yHost)
		meanflowDrawStep(sampleRNG, xHost, yHost, dc, &st)
		a := uop.NewArena(1 << 22)
		for _, p := range params {
			p.Load(a)
		}
		l0, err := meanflowStepLoss(a, model, device, dc, &st)
		if err != nil {
			return fmt.Errorf("meanflow: baseline loss: %w", err)
		}
		if err := tensor.Realize(l0); err != nil {
			return fmt.Errorf("meanflow: realize baseline loss: %w", err)
		}
		logFn(0, l0.Data()[0])
	}

	start := time.Now()

	for step := 1; step <= cfg.Steps; step++ {
		ds.Batch(stepRNG, int(batch), xHost, yHost)
		meanflowDrawStep(stepRNG, xHost, yHost, dc, &st)

		a := uop.NewArena(1 << 22)
		for _, p := range params {
			p.Load(a)
		}
		loss, err := meanflowStepLoss(a, model, device, dc, &st)
		if err != nil {
			return fmt.Errorf("meanflow: build step %d: %w", step, err)
		}

		leaves := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)
		if err := tensor.Realize(loss); err != nil {
			return fmt.Errorf("meanflow: realize loss at step %d: %w", step, err)
		}
		for _, p := range params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.Realize(g); err != nil {
				return fmt.Errorf("meanflow: realize grad for %q at step %d: %w", p.Name, step, err)
			}
		}
		opt.Step(grads)

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			if err := safetensors.Save(ckpt, meanflowParamMap(params)); err != nil && cfg.LogText != nil {
				cfg.LogText(fmt.Sprintf("meanflow: checkpoint save failed at step %d: %v\n", step, err))
			}
			logFn(step, loss.Data()[0])
		}
		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}
	}

	elapsed := time.Since(start)
	if err := safetensors.Save(ckpt, meanflowParamMap(params)); err != nil && cfg.LogText != nil {
		cfg.LogText(fmt.Sprintf("meanflow: final checkpoint save failed: %v\n", err))
	}
	if cfg.LogText != nil {
		cfg.LogText(fmt.Sprintf("meanflow: training complete in %s (%d steps)\n", elapsed.Round(time.Millisecond), cfg.Steps))
		stats, samples, err := meanflowSample(model, params, batch, dc, rand.New(rand.NewSource(seed+3)), device)
		if err == nil {
			cfg.LogText(fmt.Sprintf("meanflow: one-step sample mean=%+.4f var=%+.4f (guidance=%.1f)\n",
				stats.mean, stats.variance, meanflowGuidance))
			dir := meanflowSampleDir()
			if n, perr := ditSavePNGs(samples, batch, dc.inCh, dc.imageH, dc.imageW, dir); perr == nil {
				cfg.LogText(fmt.Sprintf("meanflow: wrote %d sample PNG(s) to %s\n", n, dir))
			} else {
				cfg.LogText(fmt.Sprintf("meanflow: PNG export failed: %v\n", perr))
			}
		} else {
			cfg.LogText(fmt.Sprintf("meanflow: sampling failed: %v\n", err))
		}
	}
	return nil
}

// meanflowStepBuffers holds the per-step host-side arrays so the loop reuses them.
type meanflowStepBuffers struct {
	zt, v    []float32 // [B, C*H*W]
	t, r, tr []float32 // [B]
	oh       []float32 // [B, numClasses+1]
	batch    int64
}

// meanflowDrawStep fills the step buffers from a drawn batch: continuous t ~ U(0,1),
// r ~ U(0,t) (a pEqual fraction with r==t), eps ~ N(0,1), the flow-matching state
// z_t = (1-t)x + t*eps and velocity v = eps - x.
func meanflowDrawStep(rng *rand.Rand, xHost []float32, yHost []int32, dc meanflowConfig, st *meanflowStepBuffers) {
	perSample := dc.inCh * dc.imageH * dc.imageW
	for b := int64(0); b < st.batch; b++ {
		tb := float32(rng.Float64())
		var rb float32
		if rng.Float32() < dc.pEqual {
			rb = tb
		} else {
			rb = tb * float32(rng.Float64())
		}
		st.t[b] = tb
		st.r[b] = rb
		st.tr[b] = tb - rb
		base := b * perSample
		for i := int64(0); i < perSample; i++ {
			eps := float32(rng.NormFloat64())
			x := xHost[base+i]
			st.zt[base+i] = (1-tb)*x + tb*eps
			st.v[base+i] = eps - x
		}
	}
	meanflowClassOneHot(yHost, dc.numClasses, dc.cfgDropProb, rng, st.oh)
}

// meanflowStepLoss builds the MeanFlow loss graph for the current step buffers: the
// forward u, the total time-derivative du/dt as one JVP, and the MSE against the
// stop-grad target v - (t-r)*du/dt (detached by realizing it and re-injecting a
// const leaf, since there is no Detach op). Returns the (unrealized) loss; the
// caller backpropagates through u only (the target is a constant).
func meanflowStepLoss(a *uop.Arena, m *meanflowModel, device string, dc meanflowConfig, st *meanflowStepBuffers) (*tensor.Tensor, error) {
	B, C, H, W := st.batch, dc.inCh, dc.imageH, dc.imageW

	zt := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, device)
	zt.SetData(append([]float32{}, st.zt...))
	vLeaf := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, device)
	vLeaf.SetData(append([]float32{}, st.v...))
	tLeaf := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, device)
	tLeaf.SetData(append([]float32{}, st.t...))
	rLeaf := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, device)
	rLeaf.SetData(append([]float32{}, st.r...))
	ones := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, device)
	onesData := make([]float32, B)
	for i := range onesData {
		onesData[i] = 1
	}
	ones.SetData(onesData)
	trData := make([]float32, B)
	copy(trData, st.tr)
	trLeaf := tensor.NewLeaf(a, []int64{B, 1, 1, 1}, uop.Dtypes.Float32, device)
	trLeaf.SetData(trData)

	u := m.Forward(a, device, zt, tLeaf, rLeaf, st.oh)

	// du/dt along the trajectory: tangent v on z, 1 on t, 0 on r (r not seeded).
	duDt, err := tensor.JVP(u, []*tensor.Tensor{zt, tLeaf}, []*tensor.Tensor{vLeaf, ones})
	if err != nil {
		return nil, fmt.Errorf("JVP: %w", err)
	}
	// Stop-grad target: realize v - (t-r)*du/dt and re-inject as a const leaf.
	tgt := vLeaf.Sub(trLeaf.Mul(duDt))
	if err := tensor.Realize(tgt); err != nil {
		return nil, fmt.Errorf("realize target: %w", err)
	}
	tgtConst := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, device)
	tgtConst.SetData(append([]float32{}, tgt.Data()...))

	diff := u.Sub(tgtConst)
	return diff.Mul(diff).Mean(nil, false), nil
}

const (
	meanflowGuidance = float32(2.0)
)

type meanflowStats struct {
	mean, variance float32
}

// meanflowSample draws Gaussian noise z1 and produces a ONE-STEP sample
// x0 = z1 - u_cfg(z1, r=0, t=1, y), where u_cfg blends a class-conditional and an
// unconditional average velocity by the guidance weight. Returns summary stats and
// the raw samples for PNG export.
func meanflowSample(m *meanflowModel, params []*nn.Parameter, batch int64, dc meanflowConfig, rng *rand.Rand, device string) (meanflowStats, []float32, error) {
	C, H, W := dc.inCh, dc.imageH, dc.imageW
	perSample := C * H * W
	cols := dc.numClasses + 1

	z1 := make([]float32, batch*perSample)
	for i := range z1 {
		z1[i] = float32(rng.NormFloat64())
	}
	condOH := make([]float32, batch*cols)
	uncondOH := make([]float32, batch*cols)
	labels := make([]int32, batch) // class 0
	meanflowClassOneHot(labels, dc.numClasses, 0, rng, condOH)
	nullLabels := make([]int32, batch)
	for i := range nullLabels {
		nullLabels[i] = int32(dc.numClasses) // null class
	}
	meanflowClassOneHot(nullLabels, dc.numClasses, 0, rng, uncondOH)

	predU := func(ohData []float32) ([]float32, error) {
		a := uop.NewArena(1 << 22)
		for _, p := range params {
			p.Load(a)
		}
		z := tensor.NewLeaf(a, []int64{batch, C, H, W}, uop.Dtypes.Float32, device)
		z.SetData(append([]float32{}, z1...))
		tLeaf := tensor.NewLeaf(a, []int64{batch, 1}, uop.Dtypes.Float32, device)
		tOnes := make([]float32, batch)
		for i := range tOnes {
			tOnes[i] = 1 // t = 1
		}
		tLeaf.SetData(tOnes)
		rLeaf := tensor.NewLeaf(a, []int64{batch, 1}, uop.Dtypes.Float32, device)
		rLeaf.SetData(make([]float32, batch)) // r = 0
		u := m.Forward(a, device, z, tLeaf, rLeaf, ohData)
		if err := tensor.Realize(u); err != nil {
			return nil, err
		}
		return append([]float32{}, u.Data()...), nil
	}

	uCond, err := predU(condOH)
	if err != nil {
		return meanflowStats{}, nil, err
	}
	uUncond, err := predU(uncondOH)
	if err != nil {
		return meanflowStats{}, nil, err
	}

	samples := make([]float32, batch*perSample)
	var sum, sumSq float64
	for i := range samples {
		u := uUncond[i] + meanflowGuidance*(uCond[i]-uUncond[i])
		x0 := z1[i] - u // one-step: integrate the average velocity over [0,1]
		samples[i] = x0
		sum += float64(x0)
		sumSq += float64(x0) * float64(x0)
	}
	n := float64(len(samples))
	mean := sum / n
	return meanflowStats{mean: float32(mean), variance: float32(sumSq/n - mean*mean)}, samples, nil
}

// meanflowClassOneHot writes a [B, numClasses+1] one-hot block with CFG dropout,
// identical in contract to ditClassOneHot.
func meanflowClassOneHot(labels []int32, numClasses int64, dropProb float32, rng *rand.Rand, out []float32) {
	ditClassOneHot(labels, numClasses, dropProb, rng, out)
}

func meanflowCheckpointPath() string {
	return filepath.Join(ditCacheDir(), "meanflow-checkpoint.safetensors")
}
func meanflowSampleDir() string {
	dir := filepath.Join(ditCacheDir(), "meanflow-samples")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

func meanflowParamMap(params []*nn.Parameter) map[string]*nn.Parameter {
	m := make(map[string]*nn.Parameter, len(params))
	for i, p := range params {
		m[fmt.Sprintf("p%04d", i)] = p
	}
	return m
}
