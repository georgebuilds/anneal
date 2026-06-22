package examples

// MeanFlow de-risk smoke (preflight R1): exercise the full MeanFlow training
// mechanism in miniature on the GPU BEFORE building the real example, namely that
//   1. a JVP of a model-like forward realizes on the GPU,
//   2. the stop-gradient target (realize-the-value, re-inject as a const leaf,
//      since there is no Detach op) works, and
//   3. Backward through the primary term (with the detached JVP target) realizes
//      and yields finite gradients.
// If this passes, the rest of MeanFlow is scaffolding mirrored from dit.go.

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestMeanFlowRegistered(t *testing.T) {
	ex, err := Get("meanflow")
	if err != nil {
		t.Fatalf("Get(meanflow): %v", err)
	}
	if ex.Build == nil || ex.Train == nil {
		t.Fatal("meanflow example missing Build or Train")
	}
	if !strings.Contains(ex.Summary, "MeanFlow") {
		t.Errorf("Summary should mention MeanFlow: %q", ex.Summary)
	}
}

func TestBuildMeanflowConstructs(t *testing.T) {
	br, err := buildMeanflow("webgpu")
	if err != nil {
		t.Fatalf("buildMeanflow: %v", err)
	}
	if br == nil || br.Arena == nil || br.Output == nil {
		t.Fatal("buildMeanflow returned nil arena/output")
	}
	if len(br.Leaves) == 0 {
		t.Fatal("buildMeanflow returned no parameter leaves")
	}
	dc := meanflowDefaultConfig()
	sh := br.Output.Shape()
	if len(sh) != 4 || sh[0] != meanflowBatch || sh[1] != dc.inCh || sh[2] != dc.imageH || sh[3] != dc.imageW {
		t.Fatalf("buildMeanflow output shape: got %v, want [%d, %d, %d, %d]",
			sh, meanflowBatch, dc.inCh, dc.imageH, dc.imageW)
	}
}

// TestRunMeanFlowFewStepsSmoke runs the full MeanFlow training loop (forward +
// JVP target + Backward + Adam) for a couple of steps on the GPU with a small
// config and an in-memory CIFAR-10 fixture, then the one-step CFG sample. Loss
// values are checked finite; convergence is a separate multi-session GPU run.
func TestRunMeanFlowFewStepsSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU MeanFlow smoke")
	}
	requireGPUForDiTTest(t)
	cacheDir := t.TempDir()
	t.Setenv("ANNEAL_CACHE_DIR", cacheDir)

	ds := synthCIFAR10(4, rand.New(rand.NewSource(7)))
	dc := meanflowConfig{
		imageH: 32, imageW: 32, patch: 8, inCh: 3,
		embedDim: 32, condDim: 32, timeEmbedDim: 32, numClasses: 10,
		nLayer: 1, nHead: 2, adamLR: 1e-3, initScale: 0.02, cfgDropProb: 0.1, pEqual: 0.25,
	}

	var captured strings.Builder
	var losses []float32
	cfg := TrainConfig{
		Steps:    2,
		LR:       0, // exercises the 0 -> dc.adamLR swap
		Batch:    2,
		LogEvery: 1,
		LogText:  func(s string) { captured.WriteString(s) },
	}
	if err := runMeanflow("webgpu", cfg, func(_ int, l float32) {
		losses = append(losses, l)
	}, ds, dc, 7); err != nil {
		t.Fatalf("runMeanflow: %v", err)
	}

	if len(losses) == 0 {
		t.Fatal("no losses logged")
	}
	for i, l := range losses {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) || l < 0 {
			t.Fatalf("loss[%d] not a valid MSE: %v", i, l)
		}
	}
	if !strings.Contains(captured.String(), "training complete") {
		t.Errorf("LogText did not receive completion line; got %q", captured.String())
	}
	if !strings.Contains(captured.String(), "one-step sample") {
		t.Errorf("LogText did not receive sample line; got %q", captured.String())
	}
	if _, err := os.Stat(filepath.Join(cacheDir, "meanflow-checkpoint.safetensors")); err != nil {
		t.Errorf("expected a checkpoint file under %s: %v", cacheDir, err)
	}
	pngs, _ := filepath.Glob(filepath.Join(cacheDir, "meanflow-samples", "*.png"))
	if len(pngs) == 0 {
		t.Errorf("expected sample PNGs under %s/meanflow-samples", cacheDir)
	}
}

// TestMeanflowDrawStep checks the host-side flow-matching draw: t in [0,1], r in
// [0,t], tr == t-r, and the interpolant/velocity invariant z_t == x + t*v (from
// z_t = (1-t)x + t*eps and v = eps - x).
func TestMeanflowDrawStep(t *testing.T) {
	dc := meanflowConfig{imageH: 2, imageW: 2, inCh: 3, numClasses: 10, pEqual: 0.25, cfgDropProb: 0}
	const B = int64(2)
	perSample := dc.inCh * dc.imageH * dc.imageW
	xHost := make([]float32, B*perSample)
	for i := range xHost {
		xHost[i] = float32(i)*0.1 - 0.5
	}
	yHost := []int32{0, 1}
	st := meanflowStepBuffers{
		zt: make([]float32, B*perSample), v: make([]float32, B*perSample),
		t: make([]float32, B), r: make([]float32, B), tr: make([]float32, B),
		oh: make([]float32, B*(dc.numClasses+1)), batch: B,
	}
	meanflowDrawStep(rand.New(rand.NewSource(1)), xHost, yHost, dc, &st)
	for b := int64(0); b < B; b++ {
		if st.t[b] < 0 || st.t[b] > 1 {
			t.Errorf("t[%d]=%v out of [0,1]", b, st.t[b])
		}
		if st.r[b] < 0 || st.r[b] > st.t[b]+1e-6 {
			t.Errorf("r[%d]=%v not in [0,t=%v]", b, st.r[b], st.t[b])
		}
		if d := st.tr[b] - (st.t[b] - st.r[b]); math.Abs(float64(d)) > 1e-6 {
			t.Errorf("tr[%d]=%v != t-r", b, st.tr[b])
		}
		base := b * perSample
		for i := int64(0); i < perSample; i++ {
			want := xHost[base+i] + st.t[b]*st.v[base+i]
			if d := st.zt[base+i] - want; math.Abs(float64(d)) > 1e-5 {
				t.Errorf("zt[%d,%d]=%v, want x+t*v=%v", b, i, st.zt[base+i], want)
			}
		}
	}
}

// TestMeanflowTimeEmbedGPU validates the in-graph continuous-time embedding math
// against a host sin/cos oracle (first half = sin(t*freqs), second half = cos).
func TestMeanflowTimeEmbedGPU(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU time-embed oracle")
	}
	requireGPUForDiTTest(t)
	a := uop.NewArena(1 << 20)
	const B, D = int64(2), int64(8)
	tVals := []float32{0.3, 0.7}
	tL := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, "webgpu")
	tL.SetData(tVals)
	emb := meanflowTimeEmbed(a, tL, D, "webgpu")
	if err := tensor.Realize(emb); err != nil {
		t.Fatalf("realize embed: %v", err)
	}
	got := emb.Data()
	if sh := emb.Shape(); len(sh) != 2 || sh[0] != B || sh[1] != D {
		t.Fatalf("embed shape %v, want [%d,%d]", sh, B, D)
	}
	half := D / 2
	for b := int64(0); b < B; b++ {
		for i := int64(0); i < half; i++ {
			freq := float32(math.Pow(10000.0, -float64(2*i)/float64(D)))
			ang := float64(tVals[b] * freq)
			if d := math.Abs(float64(got[b*D+i]) - math.Sin(ang)); d > 1e-4 {
				t.Errorf("sin[%d,%d]=%v want %v", b, i, got[b*D+i], math.Sin(ang))
			}
			if d := math.Abs(float64(got[b*D+half+i]) - math.Cos(ang)); d > 1e-4 {
				t.Errorf("cos[%d,%d]=%v want %v", b, i, got[b*D+half+i], math.Cos(ang))
			}
		}
	}
}

func TestMeanFlowJVPBackwardSmokeGPU(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: skipping GPU MeanFlow JVP smoke")
	}
	requireGPUForDiTTest(t) // shared GPU bootstrap (dit_test.go)

	a := uop.NewArena(1 << 20)
	const B, D = int64(2), int64(4)
	dev := "webgpu"

	lin := nn.NewLinear(a, D, D, true, uop.Dtypes.Float32, dev)
	rng := rand.New(rand.NewSource(7))
	for _, p := range lin.Params() {
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * 0.1
		}
		p.Load(a)
	}

	zData := make([]float32, B*D)
	vData := make([]float32, B*D)
	for i := range zData {
		zData[i] = float32(rng.NormFloat64())
		vData[i] = float32(rng.NormFloat64())
	}
	z := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, dev)
	z.SetData(zData)
	v := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, dev)
	v.SetData(vData)
	tL := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, dev)
	tL.SetData([]float32{0.3, 0.7})
	ones := tensor.NewLeaf(a, []int64{B, 1}, uop.Dtypes.Float32, dev)
	ones.SetData([]float32{1, 1})

	// Miniature u(z,t): a linear map of z plus a small continuous-time embedding,
	// differentiable in both z and t (mirrors the real u_theta structure).
	te := meanflowTimeEmbed(a, tL, D, dev) // [B,D]
	scale := tensor.FullSints(a, te.ShapeSints(), 0.1, uop.Dtypes.Float32, dev)
	u := lin.Forward(z.Add(te.Mul(scale))) // [B,D]

	// du/dt along (z->v, t->1): the MeanFlow total time-derivative as one JVP.
	duDt, err := tensor.JVP(u, []*tensor.Tensor{z, tL}, []*tensor.Tensor{v, ones})
	if err != nil {
		t.Fatalf("JVP: %v", err)
	}
	if err := tensor.Realize(duDt); err != nil {
		t.Fatalf("realize du/dt on GPU: %v", err)
	}

	// Stop-grad target = v - (t-r)*du/dt with r=0; detach by realizing + reinjecting.
	tgt := v.Sub(tL.Mul(duDt)) // [B,1] x [B,D] broadcast -> [B,D]
	if err := tensor.Realize(tgt); err != nil {
		t.Fatalf("realize target: %v", err)
	}
	tgtConst := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, dev)
	tgtConst.SetData(append([]float32{}, tgt.Data()...))

	diff := u.Sub(tgtConst)
	loss := diff.Mul(diff).Mean(nil, false)
	leaves := make([]*tensor.Tensor, 0, len(lin.Params()))
	for _, p := range lin.Params() {
		leaves = append(leaves, p.T)
	}
	grads := tensor.Backward(loss, leaves)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("realize loss: %v", err)
	}
	lv := loss.Data()[0]
	if math.IsNaN(float64(lv)) || math.IsInf(float64(lv), 0) || lv < 0 {
		t.Fatalf("MeanFlow smoke loss not a valid MSE: %v", lv)
	}
	for _, p := range lin.Params() {
		g, ok := grads[p.T]
		if !ok {
			t.Fatalf("no gradient for %q (Backward through u with detached target failed)", p.Name)
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("realize grad %q: %v", p.Name, err)
		}
		for _, gv := range g.Data() {
			if math.IsNaN(float64(gv)) || math.IsInf(float64(gv), 0) {
				t.Fatalf("non-finite gradient for %q: %v", p.Name, gv)
			}
		}
	}
	t.Logf("MeanFlow JVP->detach->Backward smoke OK: loss=%.4f", lv)
}
