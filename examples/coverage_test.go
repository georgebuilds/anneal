package examples

// Coverage-boosting tests that do NOT require a GPU. These exercise:
//   - the Example registry (Get, All, listNames, duplicate-register panic)
//   - pure data helpers (toyDataset, convDataset, dynBatchSlice,
//     heInit, copyParam, int32sAsBits, oneHotBits, oneHotBitsViT)
//   - the forward-graph Build functions for every CPU-buildable example
//     (mlp, conv, dynmlp, vit). Build constructs UOp nodes only — no
//     Realize, no executor, no kernel dispatch.
//   - the nanoGPT config helper and the in-memory dataset path.
//
// Anything that calls Realize / Backward / Train against a real device
// stays in the GPU-gated tests already in nanogpt_test.go.

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Registry ──────────────────────────────────────────────────────────────────

func TestRegistryGetKnown(t *testing.T) {
	for _, name := range []string{"mlp", "conv", "dynmlp", "vit", "nanogpt", "resnet9", "llama"} {
		ex, err := Get(name)
		if err != nil {
			t.Errorf("Get(%q): %v", name, err)
			continue
		}
		if ex.Name != name {
			t.Errorf("Get(%q).Name = %q", name, ex.Name)
		}
		if ex.Summary == "" {
			t.Errorf("Get(%q).Summary is empty", name)
		}
		if ex.Build == nil || ex.Train == nil {
			t.Errorf("Get(%q): Build or Train is nil", name)
		}
	}
}

func TestRegistryGetUnknown(t *testing.T) {
	_, err := Get("nosuch")
	if err == nil {
		t.Fatal("Get('nosuch') returned nil error")
	}
	if !strings.Contains(err.Error(), "available") {
		t.Errorf("error must list available examples: %v", err)
	}
}

func TestRegistryAllOrderStable(t *testing.T) {
	a := All()
	b := All()
	if len(a) != len(b) {
		t.Fatalf("All() returned different lengths: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("All() not stable at index %d", i)
		}
	}
	if len(a) < 4 {
		t.Errorf("expected at least 4 registered examples, got %d", len(a))
	}
	// All returns a copy: mutating the slice does not affect the registry.
	a[0] = nil
	c := All()
	if c[0] == nil {
		t.Error("All() returned an aliased slice (mutation leaked)")
	}
}

func TestRegistryDuplicateRegisterPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for duplicate Register")
		}
	}()
	Register(&Example{Name: "mlp"})
}

// ── Pure data helpers ────────────────────────────────────────────────────────

func TestToyDataset(t *testing.T) {
	xs, ys := toyDataset()
	if len(xs) != 32 || len(ys) != 16 {
		t.Fatalf("toyDataset: xs=%d ys=%d, want 32, 16", len(xs), len(ys))
	}
	// y[i] = x1^2 + x2^2 invariant.
	for i := 0; i < 16; i++ {
		x1, x2 := xs[i*2], xs[i*2+1]
		want := x1*x1 + x2*x2
		got := ys[i]
		if (got-want)*(got-want) > 1e-12 {
			t.Errorf("ys[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestConvDataset(t *testing.T) {
	imgs, labels := convDataset()
	if len(imgs) != 8*6*6 || len(labels) != 8 {
		t.Fatalf("convDataset: imgs=%d labels=%d", len(imgs), len(labels))
	}
	// Top-left 3x3 pixel must equal the label.
	for n := 0; n < 8; n++ {
		base := n * 36
		if imgs[base] != labels[n] {
			t.Errorf("image %d top-left = %v, label = %v", n, imgs[base], labels[n])
		}
	}
}

func TestDynBatchSlice(t *testing.T) {
	xFull, yFull := toyDataset()
	// Smaller-than-corpus batch.
	xb, yb := dynBatchSlice(xFull, yFull, 5)
	if len(xb) != 10 || len(yb) != 5 {
		t.Fatalf("len mismatch: xb=%d yb=%d", len(xb), len(yb))
	}
	// First sample must match the corpus.
	if xb[0] != xFull[0] || xb[1] != xFull[1] || yb[0] != yFull[0] {
		t.Error("dynBatchSlice did not copy first sample correctly")
	}
	// Larger-than-corpus batch must cycle.
	xb2, yb2 := dynBatchSlice(xFull, yFull, 20)
	if len(xb2) != 40 || len(yb2) != 20 {
		t.Fatalf("cycling len mismatch: xb=%d yb=%d", len(xb2), len(yb2))
	}
	// Sample at index 16 must equal the original sample 0 (16 % 16 == 0).
	if xb2[16*2] != xFull[0] {
		t.Error("dynBatchSlice did not cycle correctly")
	}
}

func TestHeInit(t *testing.T) {
	a := uop.NewArena(64)
	l := nn.NewLinear(a, 4, 8, true, uop.Dtypes.Float32, "webgpu")
	rng := rand.New(rand.NewSource(1))
	heInit(l.Weight, 4, rng)
	// Result must not be all zero (vanishingly small probability for random init).
	nonZero := 0
	for _, v := range l.Weight.Value {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Error("heInit produced all-zero weights")
	}
}

func TestCopyParamNilSafe(t *testing.T) {
	// nil src or dst must not panic.
	copyParam(nil, nil)
	a := uop.NewArena(8)
	l := nn.NewLinear(a, 2, 2, true, uop.Dtypes.Float32, "webgpu")
	copyParam(l.Weight, nil)
	copyParam(nil, l.Weight)
}

func TestCopyParamCopies(t *testing.T) {
	a := uop.NewArena(8)
	src := nn.NewLinear(a, 2, 2, true, uop.Dtypes.Float32, "webgpu")
	dst := nn.NewLinear(a, 2, 2, true, uop.Dtypes.Float32, "webgpu")
	for i := range src.Weight.Value {
		src.Weight.Value[i] = float32(i + 1)
	}
	copyParam(dst.Weight, src.Weight)
	for i, v := range dst.Weight.Value {
		if v != float32(i+1) {
			t.Errorf("dst[%d]=%v, want %v", i, v, float32(i+1))
		}
	}
}

// ── int32sAsBits / oneHotBits / oneHotBitsViT ────────────────────────────────

func TestInt32sAsBitsRoundtrip(t *testing.T) {
	in := []int32{0, 1, -1, 1 << 20, -42}
	out := int32sAsBits(in)
	if len(out) != len(in) {
		t.Fatalf("len mismatch")
	}
	// Compare at bit level — int32(-1) maps to NaN as float32 and NaN != NaN.
	for i, v := range in {
		gotBack := int32(math.Float32bits(out[i]))
		if gotBack != v {
			t.Errorf("roundtrip [%d]: stored bits decode to %d, want %d", i, gotBack, v)
		}
	}
}

func TestOneHotBits(t *testing.T) {
	got := oneHotBits([]int32{0, 2, 1}, 3)
	want := []float32{1, 0, 0, 0, 0, 1, 0, 1, 0}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("oneHotBits[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

func TestOneHotBitsOutOfRangeDropped(t *testing.T) {
	got := oneHotBits([]int32{-1, 5, 1}, 3)
	// Row 0 (-1) and row 1 (5) must be all-zero; row 2 has 1 at column 1.
	for i := 0; i < 3; i++ {
		if got[i] != 0 {
			t.Errorf("row0[%d] = %v, want 0", i, got[i])
		}
	}
	for i := 3; i < 6; i++ {
		if got[i] != 0 {
			t.Errorf("row1[%d] = %v, want 0", i, got[i])
		}
	}
	if got[6+1] != 1.0 {
		t.Errorf("row2[1] = %v, want 1", got[6+1])
	}
}

func TestOneHotBitsViT(t *testing.T) {
	got := oneHotBitsViT([]int32{2, 0}, 3)
	want := []float32{0, 0, 1, 1, 0, 0}
	if len(got) != len(want) {
		t.Fatalf("len = %d", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestOneHotBitsViTOutOfRangePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on out-of-range label")
		}
	}()
	_ = oneHotBitsViT([]int32{5}, 3)
}

// ── charDataset ──────────────────────────────────────────────────────────────

func TestNewCharDatasetFromStringEmptyPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty corpus")
		}
	}()
	_ = newCharDatasetFromString("")
}

func TestCharDatasetEncodeDecode(t *testing.T) {
	ds := newCharDatasetFromString("abcabc")
	if ds.VocabSize() != 3 {
		t.Errorf("vocab size = %d, want 3", ds.VocabSize())
	}
	ids := ds.Encode("abc")
	if len(ids) != 3 {
		t.Fatalf("encoded len = %d", len(ids))
	}
	got := ds.Decode(ids)
	if got != "abc" {
		t.Errorf("decode = %q", got)
	}
	// Out-of-vocab rune is silently dropped.
	if e := ds.Encode("aZb"); len(e) != 2 {
		t.Errorf("OOV encode length = %d, want 2", len(e))
	}
	// Invalid id is silently dropped during decode.
	if d := ds.Decode([]int32{-1, 99, 0}); d != "a" {
		t.Errorf("invalid-id decode = %q, want 'a'", d)
	}
}

func TestCharDatasetSampleBatchTooShort(t *testing.T) {
	ds := newCharDatasetFromString("ab")
	xs, ys := ds.SampleBatch(rand.New(rand.NewSource(1)), 1, 8)
	if xs != nil || ys != nil {
		t.Error("SampleBatch on short corpus must return nil, nil")
	}
}

func TestCharDatasetSampleBatchShape(t *testing.T) {
	ds := newCharDatasetFromString(strings.Repeat("abcd", 64))
	xs, ys := ds.SampleBatch(rand.New(rand.NewSource(2)), 3, 8)
	if len(xs) != 24 || len(ys) != 24 {
		t.Fatalf("SampleBatch len: xs=%d ys=%d", len(xs), len(ys))
	}
}

// ── defaultNanoGPTConfig ─────────────────────────────────────────────────────

func TestDefaultNanoGPTConfig(t *testing.T) {
	cfg := defaultNanoGPTConfig(123)
	if cfg.Vocab != 123 {
		t.Errorf("Vocab = %d, want 123", cfg.Vocab)
	}
	if cfg.NLayer <= 0 || cfg.NHead <= 0 || cfg.NEmbd <= 0 || cfg.BlockSize <= 0 {
		t.Errorf("invalid defaults: %+v", cfg)
	}
}

// ── Build functions (no GPU) ─────────────────────────────────────────────────

func TestBuildMLPConstructsForwardGraph(t *testing.T) {
	br, err := buildMLP("webgpu")
	if err != nil {
		t.Fatalf("buildMLP: %v", err)
	}
	if br.Arena == nil {
		t.Error("Arena is nil")
	}
	if br.Output == nil {
		t.Fatal("Output is nil")
	}
	if br.Device != "webgpu" {
		t.Errorf("Device = %q, want webgpu", br.Device)
	}
	// MLP output is [batch, 1] post-projection.
	sh := br.Output.Shape()
	if len(sh) != 2 || sh[0] != mlpBatch || sh[1] != 1 {
		t.Errorf("Output.Shape = %v, want [%d, 1]", sh, mlpBatch)
	}
	if len(br.Leaves) != 4 {
		t.Errorf("Leaves = %d, want 4", len(br.Leaves))
	}
}

func TestBuildConvConstructsForwardGraph(t *testing.T) {
	br, err := buildConv("webgpu")
	if err != nil {
		t.Fatalf("buildConv: %v", err)
	}
	if br.Output == nil {
		t.Fatal("Output is nil")
	}
	sh := br.Output.Shape()
	if len(sh) != 2 || sh[0] != convBatch || sh[1] != 1 {
		t.Errorf("Output.Shape = %v, want [%d, 1]", sh, convBatch)
	}
	// FC head is bias=true → 2 params; conv has weight only (bias=false) → 1.
	if len(br.Leaves) < 3 {
		t.Errorf("Leaves = %d, want ≥ 3", len(br.Leaves))
	}
}

func TestBuildDynMLPConstructsForwardGraph(t *testing.T) {
	br, err := buildDynMLP("webgpu")
	if err != nil {
		t.Fatalf("buildDynMLP: %v", err)
	}
	if br.Output == nil {
		t.Fatal("Output is nil")
	}
	if len(br.Leaves) != 4 {
		t.Errorf("Leaves = %d, want 4", len(br.Leaves))
	}
}

func TestBuildViTConstructsForwardGraph(t *testing.T) {
	br, err := buildViT("webgpu")
	if err != nil {
		t.Fatalf("buildViT: %v", err)
	}
	if br.Output == nil {
		t.Fatal("Output is nil")
	}
	sh := br.Output.Shape()
	if len(sh) != 2 || sh[0] != vitBatch || sh[1] != vitNumClasses {
		t.Errorf("Output.Shape = %v, want [%d, %d]", sh, vitBatch, vitNumClasses)
	}
	if len(br.Leaves) == 0 {
		t.Error("Leaves is empty")
	}
}

// ── vitDataset / initViTSmall ────────────────────────────────────────────────

func TestVitDatasetShape(t *testing.T) {
	images, labels := vitDataset(4, 3, 8, 8, 10, rand.New(rand.NewSource(1)))
	if len(images) != 4*3*8*8 || len(labels) != 4 {
		t.Fatalf("vitDataset shape: images=%d labels=%d", len(images), len(labels))
	}
	for _, k := range labels {
		if k < 0 || k >= 10 {
			t.Errorf("label %d out of range", k)
		}
	}
}

func TestInitViTSmallTouchesAllTensors(t *testing.T) {
	a := uop.NewArena(1 << 14)
	v := nn.NewViT(a, 32, 32, 4, 3, 32, 1, 2, 5)
	initViTSmall(v, 0.02, rand.New(rand.NewSource(7)))
	// At least one param value must be non-zero after init.
	any := false
	for _, p := range v.Params() {
		for _, x := range p.Value {
			if x != 0 {
				any = true
				break
			}
		}
		if any {
			break
		}
	}
	if !any {
		t.Error("initViTSmall produced all-zero weights")
	}
}
