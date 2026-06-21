package nn_test

// Vision Transformer (ViT) correctness tests.
//
// Five oracles, mirroring the rigor of gpt_test.go and mlp_block_test.go:
//
//  1. TestPatchEmbedShape
//     Forward pass produces the expected [B, N, embedDim] output shape from
//     [B, C, H, W]. Smoke gate on the patch unfold + Linear wiring.
//
//  2. TestViTShapeEndToEnd
//     Full ViT forward pass produces [B, numClasses] from [B, 3, 32, 32].
//
//  3. TestViTParamsCount
//     len(v.Params()) == 12*nLayer + 7. Pointer-identity checks at the
//     well-known boundaries (Patch.Proj.Weight first, Head.Bias last).
//
//  4. TestViTFDGradCheck
//     Tiny config (imageH=imageW=8, patch=4 -> N=4; embedDim=8, nHead=2,
//     nLayer=1, numClasses=4, B=1). Loss = output.Sum(). Tiered tolerance:
//       - Linear-only paths at 1e-3 relative: Block.Proj.{W,B},
//         Block.LN2.{W,B}, Block.FC1.{W,B}, Block.FC2.{W,B}, LNf.{W,B},
//         Head.{W,B}.
//       - Softmax-chain paths at 7e-2: Patch.Proj.{W,B}, PosEmb,
//         Block.QKV.{W,B}, Block.LN1.{W,B}. These were sign-wrong and
//         documented-skip under the OpExpand-backward bug (fixed 2026-06-18,
//         tensor/gradient_ruleset.go); they now agree with FD.
//
//  5. TestViTDeterminism
//     Same seed produces sha256-identical output logits across 3 fresh runs.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Helpers ──────────────────────────────────────────────────────────────────

// vitInitSmall seeds every learnable parameter in the ViT with small normal
// samples so the FD check operates in the regime where central-difference
// truncation error stays below the analytic-vs-FD tolerance budget. Mirrors
// gptInitSmall in gpt_test.go (same precision floor logic for softmax).
func vitInitSmall(v *nn.ViT, scale float32, rng *rand.Rand) {
	// PatchEmbed projection.
	for i := range v.Patch.Proj.Weight.Value {
		v.Patch.Proj.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	if v.Patch.Proj.Bias != nil {
		for i := range v.Patch.Proj.Bias.Value {
			v.Patch.Proj.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}

	// Positional embedding.
	for i := range v.PosEmb.Value {
		v.PosEmb.Value[i] = float32(rng.NormFloat64()) * scale
	}

	// Each encoder block.
	for _, b := range v.Blocks {
		for i := range b.LN1.Weight.Value {
			b.LN1.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
		for i := range b.LN1.Bias.Value {
			b.LN1.Bias.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.LN2.Weight.Value {
			b.LN2.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
		for i := range b.LN2.Bias.Value {
			b.LN2.Bias.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.Attn.QKV.Weight.Value {
			b.Attn.QKV.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.Attn.Proj.Weight.Value {
			b.Attn.Proj.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if b.Attn.QKV.Bias != nil {
			for i := range b.Attn.QKV.Bias.Value {
				b.Attn.QKV.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
		if b.Attn.Proj.Bias != nil {
			for i := range b.Attn.Proj.Bias.Value {
				b.Attn.Proj.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
		for i := range b.MLP.FC1.Weight.Value {
			b.MLP.FC1.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		for i := range b.MLP.FC2.Weight.Value {
			b.MLP.FC2.Weight.Value[i] = float32(rng.NormFloat64()) * scale
		}
		if b.MLP.FC1.Bias != nil {
			for i := range b.MLP.FC1.Bias.Value {
				b.MLP.FC1.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
		if b.MLP.FC2.Bias != nil {
			for i := range b.MLP.FC2.Bias.Value {
				b.MLP.FC2.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
			}
		}
	}

	// Final LayerNorm and classifier head.
	for i := range v.LNf.Weight.Value {
		v.LNf.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
	}
	for i := range v.LNf.Bias.Value {
		v.LNf.Bias.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for i := range v.Head.Weight.Value {
		v.Head.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	if v.Head.Bias != nil {
		for i := range v.Head.Bias.Value {
			v.Head.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
}

// evalViTOutput runs forward-only on the ViT and returns a copy of the
// realized [B, numClasses] logits.
func evalViTOutput(t *testing.T, v *nn.ViT, xData []float32, B int64) []float32 {
	t.Helper()
	a := uop.NewArena(1 << 18)
	for _, p := range v.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, v.InCh, v.ImageH, v.ImageW},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	y := v.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("evalViTOutput Realize: %v", err)
	}
	out := make([]float32, len(y.Data()))
	copy(out, y.Data())
	return out
}

// evalViTLoss runs forward-only and returns loss = sum(output).
func evalViTLoss(t *testing.T, v *nn.ViT, xData []float32, B int64) float32 {
	t.Helper()
	a := uop.NewArena(1 << 18)
	for _, p := range v.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, v.InCh, v.ImageH, v.ImageW},
		uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := v.Forward(x).Sum(nil, false)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("evalViTLoss Realize: %v", err)
	}
	return loss.Data()[0]
}

// ── 1. PatchEmbed shape ──────────────────────────────────────────────────────

func TestPatchEmbedShape(t *testing.T) {
	requireGPU(t)

	const (
		B        = int64(2)
		C        = int64(3)
		H        = int64(8)
		W        = int64(8)
		Patch    = int64(4)
		EmbedDim = int64(16)
	)

	a0 := uop.NewArena(1 << 14)
	pe := nn.NewPatchEmbed(a0, H, W, Patch, C, EmbedDim)
	rng := rand.New(rand.NewSource(1))
	for i := range pe.Proj.Weight.Value {
		pe.Proj.Weight.Value[i] = float32(rng.NormFloat64()) * 0.1
	}

	a := uop.NewArena(1 << 16)
	pe.Proj.Weight.Load(a)
	pe.Proj.Bias.Load(a)

	x := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, "webgpu")
	xData := make([]float32, B*C*H*W)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.1
	}
	x.SetData(xData)

	y := pe.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("PatchEmbed Realize: %v", err)
	}

	ys := y.Shape()
	N := (H / Patch) * (W / Patch)
	want := []int64{B, N, EmbedDim}
	if len(ys) != 3 || ys[0] != want[0] || ys[1] != want[1] || ys[2] != want[2] {
		t.Fatalf("PatchEmbed output shape: got %v, want %v", ys, want)
	}
	t.Logf("PatchEmbed [%d,%d,%d,%d] -> [%d,%d,%d] OK", B, C, H, W, ys[0], ys[1], ys[2])
}

// ── 2. ViT forward shape end-to-end ──────────────────────────────────────────

func TestViTShapeEndToEnd(t *testing.T) {
	requireGPU(t)

	const (
		B          = int64(2)
		C          = int64(3)
		H          = int64(32)
		W          = int64(32)
		Patch      = int64(4)
		EmbedDim   = int64(32)
		NLayer     = 2
		NHead      = 4
		NumClasses = int64(10)
	)

	a0 := uop.NewArena(1 << 14)
	v := nn.NewViT(a0, H, W, Patch, C, EmbedDim, NLayer, NHead, NumClasses)
	rng := rand.New(rand.NewSource(1))
	vitInitSmall(v, 0.1, rng)

	xData := make([]float32, B*C*H*W)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.1
	}

	out := evalViTOutput(t, v, xData, B)
	want := int(B * NumClasses)
	if len(out) != want {
		t.Fatalf("ViT output length %d != %d (expected [B=%d, numClasses=%d])",
			len(out), want, B, NumClasses)
	}
	t.Logf("ViT forward output shape [B=%d, numClasses=%d] = %d elements OK",
		B, NumClasses, len(out))
}

// ── 3. ViT params count + ordering ───────────────────────────────────────────

func TestViTParamsCount(t *testing.T) {
	const (
		H          = int64(32)
		W          = int64(32)
		Patch      = int64(4)
		C          = int64(3)
		EmbedDim   = int64(32)
		NLayer     = 2
		NHead      = 4
		NumClasses = int64(10)
	)

	a0 := uop.NewArena(1 << 14)
	v := nn.NewViT(a0, H, W, Patch, C, EmbedDim, NLayer, NHead, NumClasses)
	ps := v.Params()

	// 2 (Patch.Proj.{W,B}) + 1 (PosEmb) + 12*nLayer + 2 (LNf) + 2 (Head)
	// = 12*nLayer + 7.
	want := 12*NLayer + 7
	if len(ps) != want {
		t.Fatalf("ViT.Params(): got %d, want %d", len(ps), want)
	}

	// Pointer-identity at well-known boundaries.
	if ps[0] != v.Patch.Proj.Weight {
		t.Fatalf("Params()[0]: expected pointer-identity with Patch.Proj.Weight")
	}
	if ps[len(ps)-1] != v.Head.Bias {
		t.Fatalf("Params()[%d]: expected pointer-identity with Head.Bias", len(ps)-1)
	}

	// Spot-check the deterministic ordering across the full sequence.
	expected := []*nn.Parameter{v.Patch.Proj.Weight, v.Patch.Proj.Bias, v.PosEmb}
	for _, b := range v.Blocks {
		expected = append(expected, b.Params()...)
	}
	expected = append(expected, v.LNf.Weight, v.LNf.Bias, v.Head.Weight, v.Head.Bias)
	if len(expected) != len(ps) {
		t.Fatalf("expected-order length %d != Params() length %d", len(expected), len(ps))
	}
	for i, w := range expected {
		if ps[i] != w {
			t.Fatalf("Params()[%d]: pointer mismatch (parameter ordering changed)", i)
		}
	}
	t.Logf("ViT.Params() returns %d parameters in expected order (12*nLayer=%d + 7 = %d)",
		len(ps), 12*NLayer, want)
}

// ── 4. FD gradient check ─────────────────────────────────────────────────────

// fdViTGradParam returns the central-difference estimate of d loss / d p.Value[idx].
func fdViTGradParam(t *testing.T, v *nn.ViT, p *nn.Parameter,
	idx int, h float32, xData []float32, B int64) float32 {
	t.Helper()
	orig := p.Value[idx]
	p.Value[idx] = orig + h
	lp := evalViTLoss(t, v, xData, B)
	p.Value[idx] = orig - h
	lm := evalViTLoss(t, v, xData, B)
	p.Value[idx] = orig
	return (lp - lm) / (2 * h)
}

// TestViTFDGradCheck verifies analytic Backward gradients agree with
// central-difference FD on the Linear-only paths. Softmax-chain paths
// (Block.QKV.{W,B}, Block.LN1.{W,B}) are documented-skip per the GPT
// precedent (see gpt_test.go and notes/gather_slice_progress.md
// "Softmax gradient drift in autodiff"). Skipping these paths is the
// agreed Slice L policy: the analytic gradients still run through the
// full Backward dispatch; we just do not compare to FD.
func TestViTFDGradCheck(t *testing.T) {
	requireGPU(t)

	const (
		// Tiny config: 8x8 image, patch 4 -> N=4 patches. embedDim=8, 2 heads,
		// 1 layer, 4 classes, B=1. Smallest config that still exercises every
		// surface (patch unfold, positional embed broadcast, mean-pool, head).
		H          = int64(8)
		W          = int64(8)
		Patch      = int64(4)
		C          = int64(3)
		EmbedDim   = int64(8)
		NLayer     = 1
		NHead      = 2
		NumClasses = int64(4)
		B          = int64(1)

		fdH      = float32(1e-3)
		tolTight = float32(1e-3)
		// Softmax-chain paths (Patch.Proj, PosEmb, LN1, QKV): previously sign-wrong
		// and documented-skip under the OpExpand-backward bug (fixed 2026-06-18);
		// exp/div/rsqrt amplify FD truncation, so they carry this looser budget.
		tolSoftmax = float32(7e-2)
		nCheck     = 3
	)

	a0 := uop.NewArena(1 << 14)
	v := nn.NewViT(a0, H, W, Patch, C, EmbedDim, NLayer, NHead, NumClasses)
	rng := rand.New(rand.NewSource(7))
	vitInitSmall(v, 0.1, rng)

	xData := make([]float32, B*C*H*W)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}

	// ── Analytic gradient via Backward ───────────────────────────────────────
	a := uop.NewArena(1 << 18)
	for _, p := range v.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := v.Forward(x).Sum(nil, false)

	leaves := make([]*tensor.Tensor, 0, len(v.Params()))
	for _, p := range v.Params() {
		leaves = append(leaves, p.T)
	}
	grads := tensor.Backward(loss, leaves)
	for _, leaf := range leaves {
		if grd, ok := grads[leaf]; ok {
			if err := tensor.Realize(grd); err != nil {
				t.Fatalf("Realize grad: %v", err)
			}
		}
	}

	// Snapshot analytic gradients BEFORE FD (FD calls Load which rebuilds
	// p.T into a fresh arena, invalidating the grads map keys).
	gradOf := func(leaf *tensor.Tensor) []float32 {
		grd, ok := grads[leaf]
		if !ok {
			t.Fatalf("no gradient for leaf")
		}
		out := make([]float32, len(grd.Data()))
		copy(out, grd.Data())
		return out
	}

	patchWGrad := gradOf(v.Patch.Proj.Weight.T)
	patchBGrad := gradOf(v.Patch.Proj.Bias.T)
	posEmbGrad := gradOf(v.PosEmb.T)
	blk := v.Blocks[0]
	ln1WGrad := gradOf(blk.LN1.Weight.T)
	ln1BGrad := gradOf(blk.LN1.Bias.T)
	qkvWGrad := gradOf(blk.Attn.QKV.Weight.T)
	qkvBGrad := gradOf(blk.Attn.QKV.Bias.T)
	projWGrad := gradOf(blk.Attn.Proj.Weight.T)
	projBGrad := gradOf(blk.Attn.Proj.Bias.T)
	ln2WGrad := gradOf(blk.LN2.Weight.T)
	ln2BGrad := gradOf(blk.LN2.Bias.T)
	fc1WGrad := gradOf(blk.MLP.FC1.Weight.T)
	fc1BGrad := gradOf(blk.MLP.FC1.Bias.T)
	fc2WGrad := gradOf(blk.MLP.FC2.Weight.T)
	fc2BGrad := gradOf(blk.MLP.FC2.Bias.T)
	lnfWGrad := gradOf(v.LNf.Weight.T)
	lnfBGrad := gradOf(v.LNf.Bias.T)
	headWGrad := gradOf(v.Head.Weight.T)
	headBGrad := gradOf(v.Head.Bias.T)

	// ── FD comparison helper ─────────────────────────────────────────────────
	type groupStat struct {
		label  string
		maxRel float32
	}
	stats := make([]groupStat, 0, 16)

	checkParam := func(p *nn.Parameter, ag []float32, label string, useTol float32) {
		t.Helper()
		n := nCheck
		if n > len(ag) {
			n = len(ag)
		}
		maxRel := float32(0)
		for i := 0; i < n; i++ {
			fd := fdViTGradParam(t, v, p, i, fdH, xData, B)
			av := ag[i]
			diff := absF32(av - fd)
			scale := absF32(fd)
			if absF32(av) > scale {
				scale = absF32(av)
			}
			if scale < 1 {
				scale = 1
			}
			rel := diff / scale
			if rel > maxRel {
				maxRel = rel
			}
			t.Logf("%s[%d]: analytic=%+.6f  fd=%+.6f  rel=%.2e  (tol=%.0e)",
				label, i, av, fd, rel, useTol)
			if rel > useTol {
				t.Fatalf("%s[%d]: analytic=%.6f  fd=%.6f  diff=%.2e  rel=%.2e > tol=%.2e",
					label, i, av, fd, diff, rel, useTol)
			}
		}
		stats = append(stats, groupStat{label: label, maxRel: maxRel})
	}

	// PatchEmbed.Proj and PosEmb sit upstream of every softmax in the graph, so
	// their gradients previously inherited the OpExpand-backward sign-flip and
	// were documented-skip. With that bug fixed (2026-06-18) they agree with FD
	// inside the softmax-chain budget.
	//
	// The softmax-chain FD checks (Patch/PosEmb here, LN1/QKV below) compile many
	// extra GPU backward kernels; under -short (CI on lavapipe) that peak memory
	// OOMs the runner. Skip them on -short; the OpExpand-backward fix they prove
	// is independently covered on CPU by tensor/expand_backward_grad_test.go.
	// They run on full local/Metal runs.
	if !testing.Short() {
		checkParam(v.Patch.Proj.Weight, patchWGrad, "Patch.Proj.Weight", tolSoftmax)
		checkParam(v.Patch.Proj.Bias, patchBGrad, "Patch.Proj.Bias", tolSoftmax)
		checkParam(v.PosEmb, posEmbGrad, "PosEmb", tolSoftmax)
	}

	checkParam(blk.Attn.Proj.Weight, projWGrad, "Proj.Weight", tolTight)
	checkParam(blk.Attn.Proj.Bias, projBGrad, "Proj.Bias", tolTight)
	checkParam(blk.LN2.Weight, ln2WGrad, "LN2.Weight", tolTight)
	checkParam(blk.LN2.Bias, ln2BGrad, "LN2.Bias", tolTight)
	checkParam(blk.MLP.FC1.Weight, fc1WGrad, "FC1.Weight", tolTight)
	checkParam(blk.MLP.FC1.Bias, fc1BGrad, "FC1.Bias", tolTight)
	checkParam(blk.MLP.FC2.Weight, fc2WGrad, "FC2.Weight", tolTight)
	checkParam(blk.MLP.FC2.Bias, fc2BGrad, "FC2.Bias", tolTight)
	checkParam(v.LNf.Weight, lnfWGrad, "LNf.Weight", tolTight)
	checkParam(v.LNf.Bias, lnfBGrad, "LNf.Bias", tolTight)
	checkParam(v.Head.Weight, headWGrad, "Head.Weight", tolTight)
	checkParam(v.Head.Bias, headBGrad, "Head.Bias", tolTight)

	// Softmax/LN-chain paths (QKV.{W,B}, LN1.{W,B}): previously sign-wrong and
	// documented-skip under the OpExpand-backward bug (fixed 2026-06-18); now
	// FD-checked at the softmax-chain budget. Skipped on -short (see note above).
	if !testing.Short() {
		checkParam(blk.LN1.Weight, ln1WGrad, "LN1.Weight", tolSoftmax)
		checkParam(blk.LN1.Bias, ln1BGrad, "LN1.Bias", tolSoftmax)
		checkParam(blk.Attn.QKV.Weight, qkvWGrad, "QKV.Weight", tolSoftmax)
		checkParam(blk.Attn.QKV.Bias, qkvBGrad, "QKV.Bias", tolSoftmax)
	}

	t.Logf("ViT FD summary (max rel per group):")
	for _, s := range stats {
		t.Logf("  %-14s max-rel=%.2e", s.label, s.maxRel)
	}
}

// ── 5. Determinism ───────────────────────────────────────────────────────────

// TestViTDeterminism runs the same forward pass three times from a freshly
// seeded RNG and asserts the sha256 of the output logits is bit-identical.
// Catches nondeterminism in the dispatch / kernel-cache / scheduler path
// for the ViT stack.
func TestViTDeterminism(t *testing.T) {
	requireGPU(t)

	const (
		H          = int64(16)
		W          = int64(16)
		Patch      = int64(4)
		C          = int64(3)
		EmbedDim   = int64(16)
		NLayer     = 2
		NHead      = 4
		NumClasses = int64(10)
		B          = int64(1)
		seed       = int64(20260601)
	)

	hashFloats := func(xs []float32) string {
		h := sha256.New()
		buf := make([]byte, 4)
		for _, v := range xs {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
			_, _ = h.Write(buf)
		}
		return hex.EncodeToString(h.Sum(nil))
	}

	runOnce := func() string {
		a0 := uop.NewArena(1 << 14)
		v := nn.NewViT(a0, H, W, Patch, C, EmbedDim, NLayer, NHead, NumClasses)
		rng := rand.New(rand.NewSource(seed))
		vitInitSmall(v, 0.1, rng)

		xData := make([]float32, B*C*H*W)
		for i := range xData {
			xData[i] = float32(rng.NormFloat64()) * 0.1
		}
		out := evalViTOutput(t, v, xData, B)
		return hashFloats(out)
	}

	hashes := make([]string, 3)
	for i := range hashes {
		hashes[i] = runOnce()
		t.Logf("run %d: sha256=%s", i+1, hashes[i])
	}
	if hashes[0] != hashes[1] || hashes[1] != hashes[2] {
		t.Fatalf("non-determinism across 3 runs:\n  run1=%s\n  run2=%s\n  run3=%s",
			hashes[0], hashes[1], hashes[2])
	}
}
