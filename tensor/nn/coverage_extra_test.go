package nn_test

// Additional CPU-path coverage for tensor/nn. These tests exercise optimizer
// update arithmetic (via SetData-backed gradient tensors, no GPU realize),
// constructor validation panics, and graph-construction shape contracts of the
// larger models (ResNet9 / GPT / ViT) that the GPU-gated suites only reach
// behind a live device.

import (
	"math"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func mustPanic(t *testing.T, substr string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got none", substr)
		}
		msg := ""
		switch v := r.(type) {
		case string:
			msg = v
		case error:
			msg = v.Error()
		}
		if !strings.Contains(msg, substr) {
			t.Fatalf("panic value %v does not contain %q", r, substr)
		}
	}()
	fn()
}

// gradTensor builds a realized-looking leaf carrying grad data so that g.Data()
// returns it. Optimizer Step paths read grad via Data() only; no GPU needed.
func gradTensor(a *uop.Arena, data []float32) *tensor.Tensor {
	g := tensor.NewLeaf(a, []int64{int64(len(data))}, uop.Dtypes.Float32, "cpu")
	g.SetData(append([]float32{}, data...))
	return g
}

// ── nn.go: SGD ────────────────────────────────────────────────────────────────

func TestSGDStep_UpdatesValue(t *testing.T) {
	a := uop.NewArena(256)
	p := nn.NewParameter(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	p.Value = []float32{1, 2, 3}
	p.SGDStep([]float32{0.5, 0.5, 0.5}, 2.0)
	want := []float32{0, 1, 2}
	for i := range want {
		if p.Value[i] != want[i] {
			t.Fatalf("Value[%d]: got %v want %v", i, p.Value[i], want[i])
		}
	}
}

func TestSGDStep_LengthMismatchPanics(t *testing.T) {
	a := uop.NewArena(256)
	p := nn.NewParameter(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	mustPanic(t, "gradient length", func() { p.SGDStep([]float32{1, 2}, 0.1) })
}

func TestSGD_Step_AppliesAndSkips(t *testing.T) {
	a := uop.NewArena(1024)
	p1 := nn.NewParameter(a, []int64{2}, uop.Dtypes.Float32, "cpu")
	p1.Value = []float32{10, 20}
	p2 := nn.NewParameter(a, []int64{2}, uop.Dtypes.Float32, "cpu")
	p2.Value = []float32{5, 5}

	// Load both so p.T is the current-step leaf used as the grads key.
	l1 := p1.Load(a)
	p2.Load(a)

	opt := nn.NewSGD([]*nn.Parameter{p1, p2}, 1.0)
	// Provide a gradient only for p1; p2 must be left untouched (skip branch).
	grads := map[*tensor.Tensor]*tensor.Tensor{l1: gradTensor(a, []float32{1, 2})}
	opt.Step(grads)

	if p1.Value[0] != 9 || p1.Value[1] != 18 {
		t.Fatalf("p1 not updated: %v", p1.Value)
	}
	if p2.Value[0] != 5 || p2.Value[1] != 5 {
		t.Fatalf("p2 should be unchanged: %v", p2.Value)
	}
}

// ── optim.go: Adam ────────────────────────────────────────────────────────────

func TestAdam_ZeroGradIsNoop(t *testing.T) {
	a := uop.NewArena(256)
	p := nn.NewParameter(a, []int64{2}, uop.Dtypes.Float32, "cpu")
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.1)
	opt.ZeroGrad() // must not panic, must not alter state
	if opt.T != 0 {
		t.Fatalf("ZeroGrad must not advance step counter, got T=%d", opt.T)
	}
}

func TestAdam_Step_SignUpdateAtStep1(t *testing.T) {
	// At step 1 with default betas, m_hat=g and v_hat=g^2, so the update is
	// -lr * g / (|g| + eps) ≈ -lr * sign(g).
	a := uop.NewArena(1024)
	p := nn.NewParameter(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	p.Value = []float32{0, 0, 0}
	leaf := p.Load(a)
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.01)
	grads := map[*tensor.Tensor]*tensor.Tensor{leaf: gradTensor(a, []float32{2, -3, 0.0001})}
	opt.Step(grads)
	if opt.T != 1 {
		t.Fatalf("T: got %d want 1", opt.T)
	}
	// g=2 -> -0.01; g=-3 -> +0.01; g tiny positive -> ~ -0.01 (still sign).
	if math.Abs(float64(p.Value[0]+0.01)) > 1e-5 {
		t.Fatalf("Value[0]: got %v want ~-0.01", p.Value[0])
	}
	if math.Abs(float64(p.Value[1]-0.01)) > 1e-5 {
		t.Fatalf("Value[1]: got %v want ~+0.01", p.Value[1])
	}
}

func TestAdam_Step_SkipsMissingGrad(t *testing.T) {
	a := uop.NewArena(1024)
	p := nn.NewParameter(a, []int64{2}, uop.Dtypes.Float32, "cpu")
	p.Value = []float32{1, 1}
	p.Load(a)
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.1)
	opt.Step(map[*tensor.Tensor]*tensor.Tensor{}) // empty grads
	if p.Value[0] != 1 || p.Value[1] != 1 {
		t.Fatalf("missing-grad param should be unchanged: %v", p.Value)
	}
	if opt.T != 1 {
		t.Fatalf("Step still increments T even when nothing applies, got %d", opt.T)
	}
}

func TestAdam_GradLengthMismatchPanics(t *testing.T) {
	a := uop.NewArena(1024)
	p := nn.NewParameter(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	leaf := p.Load(a)
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.1)
	mustPanic(t, "gradient length", func() {
		opt.Step(map[*tensor.Tensor]*tensor.Tensor{leaf: gradTensor(a, []float32{1, 2})})
	})
}

// ── MaxPool2D validation ──────────────────────────────────────────────────────

func TestMaxPool2D_RankPanics(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	mustPanic(t, "rank 4", func() { nn.MaxPool2D(x, 2, 2, 2, 2) })
}

func TestMaxPool2D_BadKernelPanics(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{1, 1, 4, 4}, uop.Dtypes.Float32, "cpu")
	mustPanic(t, "kernel size must be positive", func() { nn.MaxPool2D(x, 0, 2, 2, 2) })
}

func TestMaxPool2D_BadStridePanics(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{1, 1, 4, 4}, uop.Dtypes.Float32, "cpu")
	mustPanic(t, "stride must be positive", func() { nn.MaxPool2D(x, 2, 2, 0, 2) })
}

func TestMaxPool2D_KernelLargerThanInputPanics(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{1, 1, 2, 2}, uop.Dtypes.Float32, "cpu")
	mustPanic(t, "smaller than kernel", func() { nn.MaxPool2D(x, 4, 4, 1, 1) })
}

func TestMaxPool2D_NonOverlappingShape(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{1, 2, 4, 4}, uop.Dtypes.Float32, "cpu")
	out := nn.MaxPool2D(x, 2, 2, 2, 2) // kH<=sH path
	if got := out.Shape(); got[2] != 2 || got[3] != 2 {
		t.Fatalf("non-overlapping pool shape: got %v want [...,2,2]", got)
	}
}

func TestMaxPool2D_OverlappingShape(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{1, 1, 6, 6}, uop.Dtypes.Float32, "cpu")
	// kH=3 > sH=2 -> overlapping binary-max chain branch. oH=(6-3)/2+1=2; the
	// largest kernel offset reads rows [2, 2+oH*sH=6) which fits in H=6.
	out := nn.MaxPool2D(x, 3, 3, 2, 2)
	if got := out.Shape(); got[2] != 2 || got[3] != 2 {
		t.Fatalf("overlapping pool shape: got %v want [...,2,2]", got)
	}
}

// ── Embedding validation ──────────────────────────────────────────────────────

func TestNewEmbedding_BadDimsPanics(t *testing.T) {
	a := uop.NewArena(256)
	mustPanic(t, "dims must be positive", func() {
		nn.NewEmbedding(a, 0, 8, uop.Dtypes.Float32, "cpu")
	})
}

func TestNewEmbedding_ParamsAndShape(t *testing.T) {
	a := uop.NewArena(256)
	e := nn.NewEmbedding(a, 10, 4, uop.Dtypes.Float32, "cpu")
	if ps := e.Params(); len(ps) != 1 {
		t.Fatalf("Params: got %d want 1", len(ps))
	}
	if sh := e.Weight.T.Shape(); sh[0] != 10 || sh[1] != 4 {
		t.Fatalf("weight shape: got %v want [10,4]", sh)
	}
}

// ── PatchEmbed / ViT validation + graph shape ─────────────────────────────────

func TestNewPatchEmbed_NonDivisiblePanics(t *testing.T) {
	a := uop.NewArena(256)
	mustPanic(t, "divisible by patch", func() {
		nn.NewPatchEmbed(a, 30, 32, 4, 3, 8)
	})
}

func TestNewPatchEmbed_BadPatchPanics(t *testing.T) {
	a := uop.NewArena(256)
	mustPanic(t, "patch must be positive", func() {
		nn.NewPatchEmbed(a, 32, 32, 0, 3, 8)
	})
}

func TestPatchEmbed_ForwardShape(t *testing.T) {
	a := uop.NewArena(1 << 14)
	pe := nn.NewPatchEmbed(a, 8, 8, 4, 3, 16)
	x := tensor.NewLeaf(a, []int64{2, 3, 8, 8}, uop.Dtypes.Float32, "webgpu")
	out := pe.Forward(x)
	// N = (8/4)*(8/4) = 4 tokens, embedDim 16.
	sh := out.Shape()
	if sh[0] != 2 || sh[1] != 4 || sh[2] != 16 {
		t.Fatalf("PatchEmbed forward shape: got %v want [2,4,16]", sh)
	}
}

func TestPatchEmbed_ForwardChannelMismatchPanics(t *testing.T) {
	a := uop.NewArena(1 << 12)
	pe := nn.NewPatchEmbed(a, 8, 8, 4, 3, 16)
	x := tensor.NewLeaf(a, []int64{2, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
	mustPanic(t, "!= module inCh", func() { pe.Forward(x) })
}

func TestNewViT_BadHeadDivisionPanics(t *testing.T) {
	a := uop.NewArena(1 << 12)
	mustPanic(t, "not divisible by nHead", func() {
		nn.NewViT(a, 8, 8, 4, 3, 18 /*embedDim*/, 1, 4 /*nHead*/, 10)
	})
}

func TestNewViT_NonDivisibleImagePanics(t *testing.T) {
	a := uop.NewArena(1 << 12)
	mustPanic(t, "must be divisible by patch", func() {
		nn.NewViT(a, 10, 8, 4, 3, 16, 1, 4, 10)
	})
}

func TestViT_ForwardRankPanics(t *testing.T) {
	a := uop.NewArena(1 << 16)
	v := nn.NewViT(a, 8, 8, 4, 3, 16, 1, 4, 10)
	x := tensor.NewLeaf(a, []int64{8, 8}, uop.Dtypes.Float32, "webgpu")
	mustPanic(t, "rank 4", func() { v.Forward(x) })
}

func TestViT_ForwardGraphShape(t *testing.T) {
	a := uop.NewArena(1 << 18)
	v := nn.NewViT(a, 8, 8, 4, 3, 16, 1, 4, 10)
	x := tensor.NewLeaf(a, []int64{2, 3, 8, 8}, uop.Dtypes.Float32, "webgpu")
	out := v.Forward(x)
	sh := out.Shape()
	if sh[0] != 2 || sh[1] != 10 {
		t.Fatalf("ViT forward shape: got %v want [2,10]", sh)
	}
}

// ── GPT validation + tied head ────────────────────────────────────────────────

func TestNewGPT_BadDivisionPanics(t *testing.T) {
	a := uop.NewArena(1 << 12)
	mustPanic(t, "not divisible by nHead", func() {
		nn.NewGPT(a, 16, 1, 3 /*nHead*/, 8 /*nEmbd*/, 8)
	})
}

func TestNewGPT_NonPositivePanics(t *testing.T) {
	a := uop.NewArena(1 << 12)
	mustPanic(t, "must be positive", func() {
		nn.NewGPT(a, 0, 1, 2, 8, 8)
	})
}

func TestNewGPTWithTiedHead_SharesWeight(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a, 16, 2, 2, 8, 8)
	if !g.TieWeights {
		t.Fatal("TieWeights should be true")
	}
	if g.LMHead.Weight != g.Wte.Weight {
		t.Fatal("tied head should alias LMHead.Weight to Wte.Weight (same *Parameter)")
	}
	// Tied: Params returns each unique Parameter once, so the tied weight is
	// not double-counted. Compare to an untied GPT of identical config.
	untied := nn.NewGPT(a, 16, 2, 2, 8, 8)
	if len(g.Params()) >= len(untied.Params()) {
		t.Fatalf("tied param count %d should be < untied %d", len(g.Params()), len(untied.Params()))
	}
}

func TestGPT_ForwardRankPanics(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPT(a, 16, 1, 2, 8, 8)
	idx := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Int32, "webgpu") // rank 1
	mustPanic(t, "rank 2", func() { g.Forward(idx) })
}

func TestGPT_ForwardDtypePanics(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPT(a, 16, 1, 2, 8, 8)
	idx := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Float32, "webgpu") // wrong dtype
	mustPanic(t, "dtype must be Int32", func() { g.Forward(idx) })
}

func TestGPT_ForwardNilDataPanics(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPT(a, 16, 1, 2, 8, 8)
	idx := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Int32, "webgpu")
	// idx.Data() is nil because SetData was never called.
	mustPanic(t, "idx.Data() is nil", func() { g.Forward(idx) })
}

// ── ResNet9 graph + maintenance ───────────────────────────────────────────────

func TestResNet9_ForwardBadShapePanics(t *testing.T) {
	a := uop.NewArena(1 << 18)
	m := nn.NewResNet9(a, 10, uop.Dtypes.Float32, "webgpu")
	x := tensor.NewLeaf(a, []int64{1, 3, 16, 16}, uop.Dtypes.Float32, "webgpu")
	mustPanic(t, "expected [B,3,32,32]", func() { m.Forward(x) })
}

func TestResNet9_ForwardRankPanics(t *testing.T) {
	a := uop.NewArena(1 << 18)
	m := nn.NewResNet9(a, 10, uop.Dtypes.Float32, "webgpu")
	x := tensor.NewLeaf(a, []int64{3, 32, 32}, uop.Dtypes.Float32, "webgpu")
	mustPanic(t, "expected 4-D input", func() { m.Forward(x) })
}

func TestResNet9_ForwardGraphShape(t *testing.T) {
	a := uop.NewArena(1 << 22)
	m := nn.NewResNet9(a, 10, uop.Dtypes.Float32, "webgpu")
	x := tensor.NewLeaf(a, []int64{2, 3, 32, 32}, uop.Dtypes.Float32, "webgpu")
	out := m.Forward(x) // graph-construction only; no realize
	sh := out.Shape()
	if sh[0] != 2 || sh[1] != 10 {
		t.Fatalf("ResNet9 forward shape: got %v want [2,10]", sh)
	}
}

func TestResNet9_LoadRebindsParams(t *testing.T) {
	a := uop.NewArena(1 << 22)
	m := nn.NewResNet9(a, 10, uop.Dtypes.Float32, "webgpu")
	// Capture a param's current leaf, then Load and confirm it was rebound.
	p := m.Params()[0]
	before := p.T
	a2 := uop.NewArena(1 << 22)
	m.Load(a2)
	if p.T == before {
		t.Fatal("Load should rebind p.T to a fresh leaf in the new arena")
	}
}

func TestResNet9_PostStepNoForwardIsClean(t *testing.T) {
	a := uop.NewArena(1 << 18)
	m := nn.NewResNet9(a, 10, uop.Dtypes.Float32, "webgpu")
	m.Train()
	m.Eval()
	// No forward ran, so every BN's lastBatchMean/Var is nil -> PostStep is a
	// clean no-op returning nil (no GPU realize attempted).
	if err := m.PostStep(); err != nil {
		t.Fatalf("PostStep with no forward should be nil, got %v", err)
	}
}
