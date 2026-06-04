package examples

// Additional CPU-only coverage. Mirrors coverage_test.go: no Realize, no
// GPU. These tests exercise:
//   - loss-graph constructors that compose UOp nodes only
//     (crossEntropyLoss, vitCrossEntropy)
//   - the GPT small-init helper
//   - emitSample's LogText / stdout fallback
//   - the ANNEAL_OFFLINE error path through loadShakespeareDataset and
//     the example Build/Train entry points that depend on it
//   - the no-GPU error paths inside the eval-loss + generation helpers
//     (Realize fails fast when no executor is registered, which still
//     exercises every line up to the Realize call)
//   - the train* entry points run with Steps=0 / LogEvery=0 (no Realize,
//     no GPU) so the setup blocks are covered
//
// Anything that needs a live device stays GPU-gated in the existing test
// files.

import (
	"bytes"
	"context"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"

	resnet9data "github.com/georgebuilds/anneal/examples/resnet9"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── loss-graph constructors ──────────────────────────────────────────────────

func TestCrossEntropyLossConstructsScalar(t *testing.T) {
	const B, T, V = int64(2), int64(3), int64(4)
	a := uop.NewArena(1 << 16)
	logits := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, "webgpu")
	oh := tensor.NewLeaf(a, []int64{B, T, V}, uop.Dtypes.Float32, "webgpu")
	loss := crossEntropyLoss(logits, oh, B, T, V)
	if loss == nil {
		t.Fatal("crossEntropyLoss returned nil")
	}
	if got := len(loss.Shape()); got != 0 {
		t.Fatalf("loss rank: got %d, want 0 (scalar); shape=%v", got, loss.Shape())
	}
}

func TestVitCrossEntropyConstructsScalar(t *testing.T) {
	const B, C = int64(4), int64(10)
	a := uop.NewArena(1 << 16)
	logits := tensor.NewLeaf(a, []int64{B, C}, uop.Dtypes.Float32, "webgpu")
	oh := tensor.NewLeaf(a, []int64{B, C}, uop.Dtypes.Float32, "webgpu")
	loss := vitCrossEntropy(logits, oh, B, C)
	if loss == nil {
		t.Fatal("vitCrossEntropy returned nil")
	}
	if got := len(loss.Shape()); got != 0 {
		t.Fatalf("loss rank: got %d, want 0 (scalar); shape=%v", got, loss.Shape())
	}
}

// ── initGPTSmall ─────────────────────────────────────────────────────────────

func TestInitGPTSmallTouchesAllTensors(t *testing.T) {
	a := uop.NewArena(1 << 16)
	// Tiny config so the init loop is cheap but still covers every branch
	// (LayerNorm, attn QKV/Proj weights+bias, MLP FC1/FC2 weights+bias, LM
	// head bias). Bias=true on every NewLinear inside NewGPT so all the
	// optional branches in initGPTSmall fire.
	g := nn.NewGPT(a, 8, 2, 2, 16, 8)
	rng := rand.New(rand.NewSource(7))
	initGPTSmall(g, 0.02, rng)
	// Verify at least one parameter slice is non-zero in each major group.
	groups := [][]float32{
		g.Wte.Weight.Value,
		g.Wpe.Weight.Value,
		g.LNf.Weight.Value,
		g.LMHead.Weight.Value,
	}
	for _, b := range g.Blocks {
		groups = append(groups,
			b.LN1.Weight.Value, b.LN1.Bias.Value,
			b.LN2.Weight.Value, b.LN2.Bias.Value,
			b.Attn.QKV.Weight.Value, b.Attn.Proj.Weight.Value,
			b.MLP.FC1.Weight.Value, b.MLP.FC2.Weight.Value,
		)
		if b.Attn.QKV.Bias != nil {
			groups = append(groups, b.Attn.QKV.Bias.Value)
		}
		if b.Attn.Proj.Bias != nil {
			groups = append(groups, b.Attn.Proj.Bias.Value)
		}
		if b.MLP.FC1.Bias != nil {
			groups = append(groups, b.MLP.FC1.Bias.Value)
		}
		if b.MLP.FC2.Bias != nil {
			groups = append(groups, b.MLP.FC2.Bias.Value)
		}
	}
	for i, buf := range groups {
		any := false
		for _, v := range buf {
			if v != 0 {
				any = true
				break
			}
		}
		if !any {
			t.Errorf("group %d all-zero after initGPTSmall", i)
		}
	}
}

// ── emitSample ───────────────────────────────────────────────────────────────

func TestEmitSampleViaLogText(t *testing.T) {
	var got string
	emitSample(TrainConfig{LogText: func(s string) { got = s }}, "hello")
	if !strings.Contains(got, "sample (") || !strings.Contains(got, "hello") {
		t.Errorf("LogText payload missing header / body: %q", got)
	}
}

func TestEmitSampleStdoutFallback(t *testing.T) {
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = orig }()

	done := make(chan struct{})
	var buf bytes.Buffer
	go func() {
		_, _ = io.Copy(&buf, r)
		close(done)
	}()

	emitSample(TrainConfig{}, "world")
	_ = w.Close()
	<-done

	if !strings.Contains(buf.String(), "world") {
		t.Errorf("stdout fallback missing body: %q", buf.String())
	}
}

// ── ANNEAL_OFFLINE error paths ───────────────────────────────────────────────

// withOfflineCacheMiss forces internal/assets to error on every fetch.
// ANNEAL_OFFLINE=1 + an empty cache root makes assets.Get fail before any
// network call.
func withOfflineCacheMiss(t *testing.T) {
	t.Helper()
	t.Setenv("ANNEAL_OFFLINE", "1")
	t.Setenv("ANNEAL_CACHE_DIR", t.TempDir())
}

func TestLoadShakespeareDatasetOffline(t *testing.T) {
	withOfflineCacheMiss(t)
	_, err := loadShakespeareDataset()
	if err == nil {
		t.Fatal("loadShakespeareDataset: expected error in offline cache-miss mode")
	}
	if !strings.Contains(err.Error(), "shakespeare") {
		t.Errorf("error should mention asset name: %v", err)
	}
}

func TestBuildNanoGPTOffline(t *testing.T) {
	withOfflineCacheMiss(t)
	_, err := buildNanoGPT("webgpu")
	if err == nil {
		t.Fatal("buildNanoGPT: expected error in offline cache-miss mode")
	}
}

func TestTrainNanoGPTOffline(t *testing.T) {
	withOfflineCacheMiss(t)
	err := trainNanoGPT("webgpu", TrainConfig{}, func(int, float32) {})
	if err == nil {
		t.Fatal("trainNanoGPT: expected error in offline cache-miss mode")
	}
}

func TestTrainResNet9Offline(t *testing.T) {
	withOfflineCacheMiss(t)
	err := trainResNet9("webgpu", TrainConfig{}, func(int, float32) {})
	if err == nil {
		t.Fatal("trainResNet9: expected error in offline cache-miss mode")
	}
}

// ── NanoGPTGenerateStream argument validation ───────────────────────────────

func TestNanoGPTGenerateStreamArgValidation(t *testing.T) {
	// onTok=nil: rejected before any asset / GPU access.
	if _, err := NanoGPTGenerateStream(context.Background(), "webgpu", "x", 1, nil); err == nil {
		t.Error("expected error when onTok is nil")
	}
	// nGen <= 0: rejected before any asset / GPU access.
	if _, err := NanoGPTGenerateStream(context.Background(), "webgpu", "x", 0,
		func(NanoGPTStreamToken) {}); err == nil {
		t.Error("expected error when nGen is 0")
	}
}

func TestNanoGPTGenerateStreamOffline(t *testing.T) {
	withOfflineCacheMiss(t)
	_, err := NanoGPTGenerateStream(context.Background(), "webgpu", "x", 1,
		func(NanoGPTStreamToken) {})
	if err == nil {
		t.Fatal("NanoGPTGenerateStream: expected error in offline cache-miss mode")
	}
}

// ── eval-loss helpers (no GPU → Realize errors, helper returns 0/NaN) ───────

func TestEvalMLPLossNoGPUReturnsZero(t *testing.T) {
	a := uop.NewArena(1 << 14)
	l1 := nn.NewLinear(a, 2, mlpHidden, true, uop.Dtypes.Float32, "webgpu")
	l2 := nn.NewLinear(a, mlpHidden, 1, true, uop.Dtypes.Float32, "webgpu")
	params := append(l1.Params(), l2.Params()...)
	xs, ys := toyDataset()
	fwd := func(x *tensor.Tensor) *tensor.Tensor {
		return l2.Forward(nn.ReLU(l1.Forward(x)))
	}
	if got := evalMLPLoss(fwd, params, xs, ys, "webgpu"); got != 0 {
		t.Errorf("evalMLPLoss: got %v, want 0 (Realize must fail without GPU)", got)
	}
}

func TestEvalConvLossNoGPUReturnsZero(t *testing.T) {
	a := uop.NewArena(1 << 16)
	m := newConvNetModel(a, "webgpu")
	imgs, labels := convDataset()
	if got := evalConvLoss(m.forward, m.convParams(), imgs, labels, "webgpu"); got != 0 {
		t.Errorf("evalConvLoss: got %v, want 0 (Realize must fail without GPU)", got)
	}
}

func TestEvalViTLossNoGPUReturnsNaN(t *testing.T) {
	a := uop.NewArena(1 << 16)
	v := nn.NewViT(a, vitImageH, vitImageW, vitPatch, vitInCh,
		vitEmbedDim, vitNLayer, vitNHead, vitNumClasses)
	images, labels := vitDataset(vitBatch, vitInCh, vitImageH, vitImageW,
		vitNumClasses, rand.New(rand.NewSource(1)))
	loss := evalViTLoss(v, v.Params(), images, labels, vitBatch, "webgpu")
	// Without a GPU Realize fails and the helper returns NaN.
	if !math.IsNaN(float64(loss)) {
		t.Errorf("evalViTLoss: expected NaN, got %v", loss)
	}
}

func TestEvalNanoGPTLossNoGPUReturnsNaN(t *testing.T) {
	a := uop.NewArena(1 << 16)
	g := nn.NewGPT(a, 8, 1, 2, 16, 8)
	initGPTSmall(g, 0.02, rand.New(rand.NewSource(1)))
	// Tiny dataset so x/y bit-packing is cheap.
	xs := make([]int32, 1*8)
	ys := make([]int32, 1*8)
	loss := evalNanoGPTLoss(g, g.Params(), xs, ys, 1, 8, 8, "webgpu")
	if !math.IsNaN(float64(loss)) {
		t.Errorf("evalNanoGPTLoss: expected NaN, got %v", loss)
	}
}

func TestGenerateNanoGPTNoGPU(t *testing.T) {
	a := uop.NewArena(1 << 16)
	cfg := nanoGPTConfig{Vocab: 8, NLayer: 1, NHead: 2, NEmbd: 16, BlockSize: 8}
	g := nn.NewGPT(a, cfg.Vocab, cfg.NLayer, cfg.NHead, cfg.NEmbd, cfg.BlockSize)
	initGPTSmall(g, 0.02, rand.New(rand.NewSource(1)))
	ds := newCharDatasetFromString("abcdefgh")
	_, err := generateNanoGPT(g, g.Params(), cfg, ds, "ab", 2,
		rand.New(rand.NewSource(1)), "webgpu")
	if err == nil {
		t.Fatal("generateNanoGPT: expected Realize error without GPU")
	}
}

func TestGenerateNanoGPTLongPrompt(t *testing.T) {
	// Prompt longer than block_size hits the "encoded >= T" branch.
	a := uop.NewArena(1 << 16)
	cfg := nanoGPTConfig{Vocab: 8, NLayer: 1, NHead: 2, NEmbd: 16, BlockSize: 4}
	g := nn.NewGPT(a, cfg.Vocab, cfg.NLayer, cfg.NHead, cfg.NEmbd, cfg.BlockSize)
	initGPTSmall(g, 0.02, rand.New(rand.NewSource(1)))
	ds := newCharDatasetFromString("abcdefgh")
	_, err := generateNanoGPT(g, g.Params(), cfg, ds, "abcdefgh", 1,
		rand.New(rand.NewSource(1)), "webgpu")
	if err == nil {
		t.Fatal("generateNanoGPT: expected Realize error without GPU")
	}
}

// ── trainX with Steps=0 (no Realize) ─────────────────────────────────────────

func TestTrainMLPZeroSteps(t *testing.T) {
	cfg := TrainConfig{Steps: 0, LR: 0.01, LogEvery: 0}
	if err := trainMLP("webgpu", cfg, func(int, float32) {}); err != nil {
		t.Fatalf("trainMLP zero-steps: %v", err)
	}
}

func TestTrainConvZeroSteps(t *testing.T) {
	cfg := TrainConfig{Steps: 0, LR: 0.01, LogEvery: 0}
	if err := trainConv("webgpu", cfg, func(int, float32) {}); err != nil {
		t.Fatalf("trainConv zero-steps: %v", err)
	}
}

func TestTrainDynMLPZeroSteps(t *testing.T) {
	cfg := TrainConfig{Steps: 0, LR: 0.01, LogEvery: 0, Batch: 4}
	if err := trainDynMLP("webgpu", cfg, func(int, float32) {}); err != nil {
		t.Fatalf("trainDynMLP zero-steps: %v", err)
	}
}

func TestTrainDynMLPZeroBatchFallback(t *testing.T) {
	cfg := TrainConfig{Steps: 0, LR: 0.01, LogEvery: 0, Batch: 0}
	if err := trainDynMLP("webgpu", cfg, func(int, float32) {}); err != nil {
		t.Fatalf("trainDynMLP zero-batch: %v", err)
	}
}

// TestRunViTLogTextEmits drives runViT with zero steps against a tiny
// in-memory CIFAR-10 fixture; the loop body is skipped (Steps=0) but the
// lr/batch fallback resolution, model assembly, and final wall-time
// LogText emission all execute. CPU-only — no GPU dispatch.
func TestRunViTLogTextEmits(t *testing.T) {
	ds := synthCIFAR10(8, rand.New(rand.NewSource(41)))
	var captured strings.Builder
	cfg := TrainConfig{
		Steps:   0,
		LR:      0, // exercises the zero -> vitAdamLR swap
		Batch:   0, // exercises the <=0 -> vitBatch swap
		LogText: func(s string) { captured.WriteString(s) },
	}
	if err := runViT("webgpu", cfg, func(int, float32) {}, ds, 1); err != nil {
		t.Fatalf("runViT (default fallbacks): %v", err)
	}
	if !strings.Contains(captured.String(), "training complete") {
		t.Errorf("LogText did not receive wall-time line; got %q", captured.String())
	}
}

// TestRunViTSentinelLR covers the cmdTrainSGDDefaultLR sentinel branch in
// the lr-fallback ladder. Steps=0 keeps us out of the Forward path so this
// stays CPU-only.
func TestRunViTSentinelLR(t *testing.T) {
	ds := synthCIFAR10(4, rand.New(rand.NewSource(42)))
	cfg := TrainConfig{Steps: 0, LR: cmdTrainSGDDefaultLR, Batch: 2}
	if err := runViT("webgpu", cfg, func(int, float32) {}, ds, 1); err != nil {
		t.Fatalf("runViT (sentinel LR): %v", err)
	}
}

// TestEvalViTLossCIFARNoGPUReturnsNaN covers the CIFAR-batch eval helper
// without a GPU; Realize fails fast and the helper returns NaN. Mirrors
// TestResNet9EvalLossNoGPUReturnsNaN.
func TestEvalViTLossCIFARNoGPUReturnsNaN(t *testing.T) {
	a := uop.NewArena(1 << 14)
	v := nn.NewViT(a, vitImageH, vitImageW, vitPatch, vitInCh,
		vitEmbedDim, vitNLayer, vitNHead, vitNumClasses)
	ds := synthCIFAR10(4, rand.New(rand.NewSource(43)))
	loss := evalViTLossCIFAR(v, v.Params(), ds, rand.New(rand.NewSource(44)), 2, "webgpu")
	if !math.IsNaN(float64(loss)) {
		t.Errorf("evalViTLossCIFAR: expected NaN, got %v", loss)
	}
}

// ── runNanoGPT setup paths (Steps=0, in-memory dataset) ─────────────────────

func TestRunNanoGPTZeroStepsLogTextEmits(t *testing.T) {
	// Steps=0 + a fixture corpus skips every Realize. The final sample
	// path is gated behind generateNanoGPT, which Realizes — so we expect
	// the helper to return a generation error here. The setup block up to
	// that point is what we're after.
	ds := newCharDatasetFromString(strings.Repeat("abcdefgh", 8))
	cfg := nanoGPTConfig{
		Vocab:        ds.VocabSize(),
		NLayer:       1,
		NHead:        2,
		NEmbd:        16,
		BlockSize:    4,
		SampleTokens: 1,
	}
	tcfg := TrainConfig{Steps: 0, LR: 0, LogEvery: 0, Batch: 2}
	err := runNanoGPT("webgpu", tcfg, func(int, float32) {}, ds, cfg, 1)
	if err == nil {
		t.Fatal("runNanoGPT: expected generation error without a GPU")
	}
	if !strings.Contains(err.Error(), "generation") {
		t.Errorf("error should originate in generation step, got: %v", err)
	}
}

func TestRunNanoGPTShortCorpusErrors(t *testing.T) {
	// Corpus shorter than block_size+1 is rejected up front.
	ds := newCharDatasetFromString("ab")
	cfg := nanoGPTConfig{
		Vocab:     ds.VocabSize(),
		NLayer:    1,
		NHead:     2,
		NEmbd:     16,
		BlockSize: 8,
	}
	tcfg := TrainConfig{Steps: 0, Batch: 1}
	err := runNanoGPT("webgpu", tcfg, func(int, float32) {}, ds, cfg, 1)
	if err == nil || !strings.Contains(err.Error(), "block_size") {
		t.Fatalf("expected block_size error, got: %v", err)
	}
}

// ── trainResNet9 with in-memory dataset (Steps=0) ───────────────────────────

// Step=0 paths are covered by TestRunResNet9LogTextEmits /
// TestRunResNet9SentinelLR in resnet9_test.go. The eval-loss helper has
// a GPU-gated smoke test there too; mirror it as a CPU-only NaN check so
// it runs under -short.

func TestResNet9EvalLossNoGPUReturnsNaN(t *testing.T) {
	a := uop.NewArena(1 << 14)
	m := nn.NewResNet9Scaled(a, [4]int64{2, 4, 8, 16}, 10, uop.Dtypes.Float32, "webgpu")
	initResNet9Small(m, 0.05, rand.New(rand.NewSource(1)))
	// Tiny CPU-only fixture mirroring synthCIFAR10 from resnet9_test.go;
	// kept private so we don't depend on a test-only helper here.
	const imagePixels = 3 * 32 * 32
	ds := &resnet9data.CIFAR10{
		Train:       make([]float32, 4*imagePixels),
		TrainLabels: []int32{0, 1, 2, 3},
	}
	loss := resnet9EvalLoss(m, m.Params(), ds, rand.New(rand.NewSource(2)), 2, "webgpu")
	if !math.IsNaN(float64(loss)) {
		t.Errorf("resnet9EvalLoss: expected NaN, got %v", loss)
	}
}

// ── listNames empty path is hit indirectly via Get("nosuch") in
// coverage_test.go, so listNames itself stays at >0%. The "len == 0"
// branch is unreachable in practice because init() registers every
// example; we don't game it here.
