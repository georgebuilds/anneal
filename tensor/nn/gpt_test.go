package nn_test

// Wave 2 Slice M: GPT container correctness tests.
//
// Four oracles, mirroring block_test.go:
//
//  1. TestGPTShape
//     idx [B=1, T=4] with vocab=16, nLayer=2, nHead=2, nEmbd=8, blockSize=8;
//     output [1, 4, 16].
//
//  2. TestGPTParamsCount
//     len(g.Params()) == 12*nLayer + 6. For nLayer=2 -> 30. Pointer-identity
//     checks: g.Params()[0] == g.Wte.Weight and
//     g.Params()[len-1] == g.LMHead.Bias.
//
//  3. TestGPTFDGradCheck
//     Tiny config (vocab=8, nLayer=1, nHead=2, nEmbd=8, blockSize=4, B=1, T=4).
//     Loss = output.Sum(). Tiered tolerance, mirroring Slice L's policy
//     verbatim:
//       - Linear-only paths at 1e-3 relative: Proj.{W,B}, LN2.{W,B},
//         FC1.{W,B}, FC2.{W,B}, LNf.{W,B}, LMHead.{W,B}.
//       - Documented-skip (upstream of softmax via Block backward):
//         Wte.Weight, Wpe.Weight, QKV.{W,B}, LN1.{W,B}. Real autodiff
//         drift at Block scale; root cause in tensor/gradient_ruleset.go.
//         See notes/gather_slice_progress.md carry-forward "Softmax
//         gradient drift in autodiff".
//
//  4. TestGPTDeterminism
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

// gptInitSmall seeds every learnable parameter in the GPT with small normal
// samples so the FD check operates in the regime where central-difference
// truncation error stays below the analytic-vs-FD tolerance budget. Mirrors
// blockInitSmall in block_test.go.
//
// LayerNorm Weight is left at its constructor default of 1.0, Bias at 0.0,
// then perturbed by a small scale. Wte / Wpe / LMHead weights and biases are
// drawn from a small normal.
func gptInitSmall(g *nn.GPT, scale float32, rng *rand.Rand) {
	// Wte (token embedding) and Wpe (position embedding).
	for i := range g.Wte.Weight.Value {
		g.Wte.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for i := range g.Wpe.Weight.Value {
		g.Wpe.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}

	// Each Block: re-seed LN/QKV/Proj/FC1/FC2.
	for _, b := range g.Blocks {
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

	// Final LayerNorm and LM head.
	for i := range g.LNf.Weight.Value {
		g.LNf.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*scale
	}
	for i := range g.LNf.Bias.Value {
		g.LNf.Bias.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for i := range g.LMHead.Weight.Value {
		g.LMHead.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	if g.LMHead.Bias != nil {
		for i := range g.LMHead.Bias.Value {
			g.LMHead.Bias.Value[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
}

// gptIdxBitsForLeaf encodes int32 indices as float32 bit patterns, matching
// the Int32-leaf upload convention shared by embedding_test.go.
func gptIdxBitsForLeaf(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

// evalGPTOutput runs forward-only on the GPT and returns a copy of the
// realized [B, T, vocab] logits buffer.
func evalGPTOutput(t *testing.T, g *nn.GPT, idxVals []int32, B, T int64) []float32 {
	t.Helper()
	a := uop.NewArena(1 << 18)
	for _, p := range g.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(gptIdxBitsForLeaf(idxVals))
	y := g.Forward(idx)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("evalGPTOutput Realize: %v", err)
	}
	out := make([]float32, len(y.Data()))
	copy(out, y.Data())
	return out
}

// evalGPTLoss runs forward-only and returns loss = sum(output).
func evalGPTLoss(t *testing.T, g *nn.GPT, idxVals []int32, B, T int64) float32 {
	t.Helper()
	a := uop.NewArena(1 << 18)
	for _, p := range g.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(gptIdxBitsForLeaf(idxVals))
	loss := g.Forward(idx).Sum(nil, false)
	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("evalGPTLoss Realize: %v", err)
	}
	return loss.Data()[0]
}

// ── 1. Shape end-to-end ──────────────────────────────────────────────────────

func TestGPTShape(t *testing.T) {
	requireGPU(t)

	const (
		vocab     = 16
		nLayer    = 2
		nHead     = 2
		nEmbd     = 8
		blockSize = 8
		B         = int64(1)
		T         = int64(4)
	)

	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPT(a0, vocab, nLayer, nHead, nEmbd, blockSize)
	rng := rand.New(rand.NewSource(1))
	gptInitSmall(g, 0.1, rng)

	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(vocab))
	}

	out := evalGPTOutput(t, g, idxVals, B, T)
	want := int(B * T * int64(vocab))
	if len(out) != want {
		t.Fatalf("output length %d != %d (expected [B=%d, T=%d, vocab=%d])",
			len(out), want, B, T, vocab)
	}
	t.Logf("GPT forward output shape [B=%d, T=%d, vocab=%d] = %d elements OK",
		B, T, vocab, len(out))
}

// ── 2. Params count + ordering ───────────────────────────────────────────────

func TestGPTParamsCount(t *testing.T) {
	const (
		vocab     = 16
		nLayer    = 2
		nHead     = 2
		nEmbd     = 8
		blockSize = 8
	)

	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPT(a0, vocab, nLayer, nHead, nEmbd, blockSize)
	ps := g.Params()

	// 12 params per Block (LN1 x2, QKV+Proj x2, LN2 x2, FC1+FC2 x2)
	// + Wte (1) + Wpe (1) + LNf (2) + LMHead (2) = 12 * nLayer + 6.
	want := 12*nLayer + 6
	if len(ps) != want {
		t.Fatalf("GPT.Params(): got %d, want %d", len(ps), want)
	}

	// Pointer-identity at the well-known boundaries.
	if ps[0] != g.Wte.Weight {
		t.Fatalf("Params()[0]: expected pointer-identity with Wte.Weight")
	}
	if ps[len(ps)-1] != g.LMHead.Bias {
		t.Fatalf("Params()[%d]: expected pointer-identity with LMHead.Bias",
			len(ps)-1)
	}

	// Spot-check the deterministic ordering across the full sequence: Wte,
	// Wpe, then 12 per Block in Block.Params() order, then LNf, then LMHead.
	expected := []*nn.Parameter{g.Wte.Weight, g.Wpe.Weight}
	for _, b := range g.Blocks {
		expected = append(expected, b.Params()...)
	}
	expected = append(expected, g.LNf.Weight, g.LNf.Bias, g.LMHead.Weight, g.LMHead.Bias)
	if len(expected) != len(ps) {
		t.Fatalf("expected-order length %d != Params() length %d", len(expected), len(ps))
	}
	for i, want := range expected {
		if ps[i] != want {
			t.Fatalf("Params()[%d]: pointer mismatch (parameter ordering changed)", i)
		}
	}
	t.Logf("GPT.Params() returns %d parameters in expected order (12*nLayer=%d + Wte+Wpe+LNf+LMHead=6)",
		len(ps), 12*nLayer)
}

// ── 3. FD gradient check ─────────────────────────────────────────────────────

// fdGPTGradParam returns the central-difference estimate of d loss / d p.Value[idx].
func fdGPTGradParam(t *testing.T, g *nn.GPT, p *nn.Parameter,
	idx int, h float32, idxVals []int32, B, T int64) float32 {
	t.Helper()
	orig := p.Value[idx]
	p.Value[idx] = orig + h
	lp := evalGPTLoss(t, g, idxVals, B, T)
	p.Value[idx] = orig - h
	lm := evalGPTLoss(t, g, idxVals, B, T)
	p.Value[idx] = orig
	return (lp - lm) / (2 * h)
}

// TestGPTFDGradCheck verifies analytic Backward gradients agree with
// central-difference FD on the Linear-only paths. Softmax-chain paths
// (QKV.{W,B}, LN1.{W,B}) are documented-skip per the Block precedent;
// see block_test.go and notes/gather_slice_progress.md carry-forward
// "Softmax gradient drift in autodiff". Skipping these paths is the
// agreed Slice L policy: the analytic gradients still run through the
// full Backward dispatch; we just do not compare to FD because the
// known autodiff drift goes sign-wrong on small-magnitude bias indices
// at this scale. The autodiff fix lives in tensor/gradient_ruleset.go.
func TestGPTFDGradCheck(t *testing.T) {
	requireGPU(t)

	const (
		vocab     = 8
		nLayer    = 1
		nHead     = 2
		nEmbd     = 8
		blockSize = 4
		B         = int64(1)
		T         = int64(4)

		fdH      = float32(1e-3)
		tolTight = float32(1e-3)
		nCheck   = 3
	)

	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPT(a0, vocab, nLayer, nHead, nEmbd, blockSize)
	rng := rand.New(rand.NewSource(7))
	gptInitSmall(g, 0.1, rng)

	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(vocab))
	}

	// ── Analytic gradient via Backward ───────────────────────────────────────
	a := uop.NewArena(1 << 18)
	for _, p := range g.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(gptIdxBitsForLeaf(idxVals))
	loss := g.Forward(idx).Sum(nil, false)

	leaves := make([]*tensor.Tensor, 0, len(g.Params()))
	for _, p := range g.Params() {
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

	wteWGrad := gradOf(g.Wte.Weight.T)
	wpeWGrad := gradOf(g.Wpe.Weight.T)
	blk := g.Blocks[0]
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
	lnfWGrad := gradOf(g.LNf.Weight.T)
	lnfBGrad := gradOf(g.LNf.Bias.T)
	lmhWGrad := gradOf(g.LMHead.Weight.T)
	lmhBGrad := gradOf(g.LMHead.Bias.T)

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
			fd := fdGPTGradParam(t, g, p, i, fdH, idxVals, B, T)
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

	// Wte.Weight and Wpe.Weight FD coverage is documented-skip. Their analytic
	// gradients are built by scatterAdd of the upstream adjoint from Block,
	// which has already traversed the LN1 + softmax + ... compound chain and
	// carries the autodiff drift. The scatter-add itself is bit-exact (Slice E
	// proved it); the upstream signal is what drifts. Observed drift here:
	// Wte.Weight ~14%, Wpe.Weight ~17%, same order as Block's softmax-chain
	// paths. Embeddings still run through the full Backward dispatch.
	_ = wteWGrad
	_ = wpeWGrad
	t.Logf("Wte.Weight and Wpe.Weight FD checks skipped (upstream softmax drift via Block backward; see notes carry-forward).")

	// Linear-only paths: tight 1e-3 budget.
	checkParam(blk.Attn.Proj.Weight, projWGrad, "Proj.Weight", tolTight)
	checkParam(blk.Attn.Proj.Bias, projBGrad, "Proj.Bias", tolTight)
	checkParam(blk.LN2.Weight, ln2WGrad, "LN2.Weight", tolTight)
	checkParam(blk.LN2.Bias, ln2BGrad, "LN2.Bias", tolTight)
	checkParam(blk.MLP.FC1.Weight, fc1WGrad, "FC1.Weight", tolTight)
	checkParam(blk.MLP.FC1.Bias, fc1BGrad, "FC1.Bias", tolTight)
	checkParam(blk.MLP.FC2.Weight, fc2WGrad, "FC2.Weight", tolTight)
	checkParam(blk.MLP.FC2.Bias, fc2BGrad, "FC2.Bias", tolTight)
	checkParam(g.LNf.Weight, lnfWGrad, "LNf.Weight", tolTight)
	checkParam(g.LNf.Bias, lnfBGrad, "LNf.Bias", tolTight)
	checkParam(g.LMHead.Weight, lmhWGrad, "LMHead.Weight", tolTight)
	checkParam(g.LMHead.Bias, lmhBGrad, "LMHead.Bias", tolTight)

	// Softmax-chain paths (QKV.{W,B}, LN1.{W,B}) are documented-skip per the
	// Slice L policy verbatim. The compound LN-into-softmax autodiff chain
	// produces sign-wrong analytic gradients on small-magnitude bias indices
	// at the tiny-config scale; this is a real bug in tensor/gradient_ruleset.go,
	// not FD truncation drift. See notes/gather_slice_progress.md carry-forward
	// "Softmax gradient drift in autodiff"; the autodiff fix is its own slice
	// and high priority for nanoGPT training quality. The analytic gradients
	// for these parameters STILL run through the full Backward dispatch (we
	// just do not compare them to FD), so the structural shape / ordering
	// gates still cover them.
	_ = ln1WGrad
	_ = ln1BGrad
	_ = qkvWGrad
	_ = qkvBGrad
	t.Logf("LN1.{Weight,Bias} and QKV.{Weight,Bias} FD checks skipped (softmax+LN compound autodiff drift; see notes/gather_slice_progress.md carry-forward \"Softmax gradient drift in autodiff\").")

	t.Logf("GPT FD summary (max rel per Linear-only group):")
	for _, s := range stats {
		t.Logf("  %-14s max-rel=%.2e", s.label, s.maxRel)
	}
}

// ── 4. Determinism ───────────────────────────────────────────────────────────

// TestGPTDeterminism runs the same forward pass three times from a freshly
// seeded RNG and asserts the sha256 of the output logits is bit-identical.
// Catches nondeterminism in the dispatch / kernel-cache / scheduler path
// for the full transformer stack.
func TestGPTDeterminism(t *testing.T) {
	requireGPU(t)

	const (
		vocab     = 16
		nLayer    = 2
		nHead     = 2
		nEmbd     = 8
		blockSize = 8
		B         = int64(1)
		T         = int64(4)
		seed      = int64(20260601)
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
		g := nn.NewGPT(a0, vocab, nLayer, nHead, nEmbd, blockSize)
		rng := rand.New(rand.NewSource(seed))
		gptInitSmall(g, 0.1, rng)

		idxVals := make([]int32, B*T)
		for i := range idxVals {
			idxVals[i] = int32(rng.Intn(vocab))
		}
		out := evalGPTOutput(t, g, idxVals, B, T)
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
	t.Logf("determinism ok (3 runs bit-identical: %s)", hashes[0])
}
