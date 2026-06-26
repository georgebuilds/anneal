package nn_test

// BERT container correctness tests.
//
// BERT is pure composition over the existing kit (Embedding + learned position
// Parameter + non-causal ViTBlock stack + LayerNorm + Linear head), so the
// oracle that matters is that the composed graph differentiates correctly. The
// finite-difference gradient check runs on BOTH the pure-Go CPU backend (always
// in CI; gradient rules are backend-independent) and the GPU (guarded, skipped
// on -short like the other GPU bursts), pinning analytic-vs-FD agreement across
// every parameter group: the Linear-only paths (Head, LNf, Proj, LN2, FC1, FC2)
// at a tight tolerance, and the embedding + softmax + layernorm chain (Wte,
// PosEmb, LN1, QKV) at the looser softmax-chain budget that exp/div/rsqrt FD
// truncation forces (same tiering as gpt_test.go / vit_test.go).

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// bertIdxBits packs a []int32 of token ids into the float32-bit layout a
// NewLeaf(..., Int32, ...).SetData expects.
func bertIdxBits(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

// bertInitSmall seeds every learnable parameter with small normal samples so
// the FD check operates where central-difference truncation stays below the
// analytic-vs-FD tolerance budget. Mirrors vitInitSmall / gptInitSmall.
func bertInitSmall(m *nn.BERT, scale float32, rng *rand.Rand) {
	fillN := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rng.NormFloat64()) * scale
		}
	}
	fillB := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rng.NormFloat64()) * scale * 0.5
		}
	}
	fillLN := func(w, b []float32) {
		for i := range w {
			w[i] = 1.0 + float32(rng.NormFloat64())*scale
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64()) * scale
		}
	}
	fillN(m.Wte.Weight.Value)
	fillN(m.PosEmb.Value)
	for _, blk := range m.Blocks {
		fillLN(blk.LN1.Weight.Value, blk.LN1.Bias.Value)
		fillN(blk.Attn.QKV.Weight.Value)
		fillN(blk.Attn.Proj.Weight.Value)
		if blk.Attn.QKV.Bias != nil {
			fillB(blk.Attn.QKV.Bias.Value)
		}
		if blk.Attn.Proj.Bias != nil {
			fillB(blk.Attn.Proj.Bias.Value)
		}
		fillLN(blk.LN2.Weight.Value, blk.LN2.Bias.Value)
		fillN(blk.MLP.FC1.Weight.Value)
		fillN(blk.MLP.FC2.Weight.Value)
		if blk.MLP.FC1.Bias != nil {
			fillB(blk.MLP.FC1.Bias.Value)
		}
		if blk.MLP.FC2.Bias != nil {
			fillB(blk.MLP.FC2.Bias.Value)
		}
	}
	fillLN(m.LNf.Weight.Value, m.LNf.Bias.Value)
	fillN(m.Head.Weight.Value)
	if m.Head.Bias != nil {
		fillB(m.Head.Bias.Value)
	}
}

// ── 1. Forward shape end-to-end (CPU) ─────────────────────────────────────────

func TestBERTShapeEndToEnd(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const (
		vocab  = 12
		nLayer = 2
		nHead  = 2
		nEmbd  = 8
		seqLen = 6
		B      = int64(2)
		T      = int64(6)
	)

	a0 := uop.NewArena(1 << 14)
	m := nn.NewBERT(a0, vocab, nLayer, nHead, nEmbd, seqLen)
	bertInitSmall(m, 0.1, rand.New(rand.NewSource(1)))

	a := uop.NewArena(1 << 16)
	for _, p := range m.Params() {
		p.Load(a)
	}
	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = int32(i % vocab)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "cpu")
	idx.SetData(bertIdxBits(idxVals))

	y := m.Forward(idx)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("BERT forward Realize: %v", err)
	}
	sh := y.Shape()
	if len(sh) != 3 || sh[0] != B || sh[1] != T || sh[2] != int64(vocab) {
		t.Fatalf("BERT output shape: got %v, want [%d %d %d]", sh, B, T, vocab)
	}
	t.Logf("BERT forward [B=%d,T=%d] -> [%d,%d,%d] OK", B, T, sh[0], sh[1], sh[2])
}

// ── 2. Params count + ordering ────────────────────────────────────────────────

func TestBERTParamsCount(t *testing.T) {
	const (
		vocab  = 12
		nLayer = 2
		nHead  = 2
		nEmbd  = 8
		seqLen = 6
	)
	a0 := uop.NewArena(1 << 14)
	m := nn.NewBERT(a0, vocab, nLayer, nHead, nEmbd, seqLen)
	ps := m.Params()

	// 1 (Wte) + 1 (PosEmb) + 12*nLayer + 2 (LNf) + 2 (Head) = 12*nLayer + 6.
	want := 12*nLayer + 6
	if len(ps) != want {
		t.Fatalf("BERT.Params(): got %d, want %d", len(ps), want)
	}
	if ps[0] != m.Wte.Weight {
		t.Fatalf("Params()[0]: expected pointer-identity with Wte.Weight")
	}
	if ps[1] != m.PosEmb {
		t.Fatalf("Params()[1]: expected pointer-identity with PosEmb")
	}
	if ps[len(ps)-1] != m.Head.Bias {
		t.Fatalf("Params()[last]: expected pointer-identity with Head.Bias")
	}

	expected := []*nn.Parameter{m.Wte.Weight, m.PosEmb}
	for _, b := range m.Blocks {
		expected = append(expected, b.Params()...)
	}
	expected = append(expected, m.LNf.Weight, m.LNf.Bias, m.Head.Weight, m.Head.Bias)
	if len(expected) != len(ps) {
		t.Fatalf("expected-order length %d != Params() length %d", len(expected), len(ps))
	}
	for i, w := range expected {
		if ps[i] != w {
			t.Fatalf("Params()[%d]: pointer mismatch (parameter ordering changed)", i)
		}
	}
	t.Logf("BERT.Params() returns %d parameters in expected order (12*%d + 6)", len(ps), nLayer)
}

// ── 3. Finite-difference gradient check (shared body) ─────────────────────────

// bertFDGradCheck verifies analytic Backward gradients agree with central
// differences for a tiny BERT on whatever executor the caller installed. device
// tags the input leaf (cosmetic on the CPU backend, required on GPU). It is
// driven by TestBERTFDGradCheckCPU and TestBERTFDGradCheckGPU.
func bertFDGradCheck(t *testing.T, device string) {
	t.Helper()
	const (
		vocab  = 10
		nLayer = 1
		nHead  = 2
		nEmbd  = 8
		seqLen = 4
		B      = int64(1)
		T      = int64(4)

		fdH        = float32(1e-3)
		tolTight   = float32(5e-3) // Linear-only paths
		tolSoftmax = float32(7e-2) // embedding + softmax + layernorm chain
		nCheck     = 4
	)

	rng := rand.New(rand.NewSource(7))
	a0 := uop.NewArena(1 << 14)
	m := nn.NewBERT(a0, vocab, nLayer, nHead, nEmbd, seqLen)
	bertInitSmall(m, 0.1, rng)

	// Cover every embedding row so Wte gradients are exercised.
	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = int32(i % vocab)
	}

	snap := make([][]float32, len(m.Params()))
	for i, p := range m.Params() {
		snap[i] = append([]float32(nil), p.Value...)
	}
	restore := func() {
		for i, p := range m.Params() {
			copy(p.Value, snap[i])
		}
	}

	// Analytic gradients via Backward.
	a := uop.NewArena(1 << 18)
	for _, p := range m.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(bertIdxBits(idxVals))
	loss := m.Forward(idx).Sum(nil, false)

	leaves := make([]*tensor.Tensor, 0, len(m.Params()))
	for _, p := range m.Params() {
		leaves = append(leaves, p.T)
	}
	grads := tensor.Backward(loss, leaves)
	// Realize each grad separately (one Realize per leaf). Batching all grads
	// into a single variadic Realize tripped the assignOutputs structural-key
	// ordering footgun (a [seqLen,nEmbd] PosEmb grad came back empty); the
	// per-grad loop is the pattern every other example/test uses.
	for _, leaf := range leaves {
		if g, ok := grads[leaf]; ok {
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("realize grad: %v", err)
			}
		}
	}
	gradOf := func(leaf *tensor.Tensor) []float32 {
		g, ok := grads[leaf]
		if !ok {
			t.Fatalf("no gradient for leaf")
		}
		return append([]float32(nil), g.Data()...)
	}

	blk := m.Blocks[0]
	type checkSpec struct {
		p     *nn.Parameter
		ag    []float32
		label string
		tol   float32
	}
	specs := []checkSpec{
		// Linear-only paths (tight).
		{m.Head.Weight, gradOf(m.Head.Weight.T), "Head.Weight", tolTight},
		{m.Head.Bias, gradOf(m.Head.Bias.T), "Head.Bias", tolTight},
		{m.LNf.Weight, gradOf(m.LNf.Weight.T), "LNf.Weight", tolTight},
		{m.LNf.Bias, gradOf(m.LNf.Bias.T), "LNf.Bias", tolTight},
		{blk.Attn.Proj.Weight, gradOf(blk.Attn.Proj.Weight.T), "Proj.Weight", tolTight},
		{blk.Attn.Proj.Bias, gradOf(blk.Attn.Proj.Bias.T), "Proj.Bias", tolTight},
		{blk.LN2.Weight, gradOf(blk.LN2.Weight.T), "LN2.Weight", tolTight},
		{blk.LN2.Bias, gradOf(blk.LN2.Bias.T), "LN2.Bias", tolTight},
		{blk.MLP.FC1.Weight, gradOf(blk.MLP.FC1.Weight.T), "FC1.Weight", tolTight},
		{blk.MLP.FC2.Weight, gradOf(blk.MLP.FC2.Weight.T), "FC2.Weight", tolTight},
		// Embedding + softmax + layernorm chain (loose).
		{m.Wte.Weight, gradOf(m.Wte.Weight.T), "Wte.Weight", tolSoftmax},
		{m.PosEmb, gradOf(m.PosEmb.T), "PosEmb", tolSoftmax},
		{blk.LN1.Weight, gradOf(blk.LN1.Weight.T), "LN1.Weight", tolSoftmax},
		{blk.Attn.QKV.Weight, gradOf(blk.Attn.QKV.Weight.T), "QKV.Weight", tolSoftmax},
	}

	lossVal := func() float32 {
		aa := uop.NewArena(1 << 18)
		for _, p := range m.Params() {
			p.Load(aa)
		}
		id := tensor.NewLeaf(aa, []int64{B, T}, uop.Dtypes.Int32, device)
		id.SetData(bertIdxBits(idxVals))
		l := m.Forward(id).Sum(nil, false)
		if err := tensor.Realize(l); err != nil {
			t.Fatalf("realize FD loss: %v", err)
		}
		return l.Data()[0]
	}

	for _, sp := range specs {
		n := nCheck
		if n > len(sp.ag) {
			n = len(sp.ag)
		}
		worst := float32(0)
		for i := 0; i < n; i++ {
			restore()
			sp.p.Value[i] += fdH
			up := lossVal()
			restore()
			sp.p.Value[i] -= fdH
			dn := lossVal()
			restore()
			fd := (up - dn) / (2 * fdH)
			av := sp.ag[i]
			diff := absF32(av - fd)
			scale := absF32(fd)
			if absF32(av) > scale {
				scale = absF32(av)
			}
			if scale < 1 {
				scale = 1
			}
			rel := diff / scale
			if rel > worst {
				worst = rel
			}
			if rel > sp.tol {
				t.Fatalf("%s[%d]: analytic=%.6f fd=%.6f diff=%.2e rel=%.2e > tol=%.2e",
					sp.label, i, av, fd, diff, rel, sp.tol)
			}
		}
		t.Logf("%-12s worst rel=%.2e over %d (tol=%.0e)", sp.label, worst, n, sp.tol)
	}
}

func TestBERTFDGradCheckCPU(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()
	bertFDGradCheck(t, "cpu")
}

func TestBERTFDGradCheckGPU(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode: GPU BERT backward burst is slow/OOM-prone on the software renderer; the CPU FD check covers the gradient logic")
	}
	requireGPU(t)
	bertFDGradCheck(t, "webgpu")
}

// ── 4. Constructor + Forward validation guards ────────────────────────────────

func mustPanicBERT(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

func TestNewBERTPanics(t *testing.T) {
	a := uop.NewArena(1 << 12)
	mustPanicBERT(t, "nEmbd%nHead!=0", func() { nn.NewBERT(a, 10, 1, 3, 8, 4) }) // 8 % 3 != 0
	mustPanicBERT(t, "nonpositive vocab", func() { nn.NewBERT(a, 0, 1, 2, 8, 4) })
	mustPanicBERT(t, "nonpositive seqLen", func() { nn.NewBERT(a, 10, 1, 2, 8, 0) })
}

func TestBERTForwardPanics(t *testing.T) {
	a := uop.NewArena(1 << 14)
	m := nn.NewBERT(a, 10, 1, 2, 8, 4) // seqLen 4
	for _, p := range m.Params() {
		p.Load(a)
	}
	// rank != 2
	mustPanicBERT(t, "rank!=2", func() {
		bad := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Int32, "cpu")
		bad.SetData(make([]float32, 4))
		m.Forward(bad)
	})
	// dtype != Int32
	mustPanicBERT(t, "dtype!=Int32", func() {
		bad := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Float32, "cpu")
		bad.SetData(make([]float32, 4))
		m.Forward(bad)
	})
	// T > seqLen (8 > 4)
	mustPanicBERT(t, "T>seqLen", func() {
		bad := tensor.NewLeaf(a, []int64{1, 8}, uop.Dtypes.Int32, "cpu")
		bad.SetData(make([]float32, 8))
		m.Forward(bad)
	})
	// idx.Data() nil (no SetData)
	mustPanicBERT(t, "nil data", func() {
		bad := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Int32, "cpu")
		m.Forward(bad)
	})
}

// ── 5. Determinism (CPU) ──────────────────────────────────────────────────────

// TestBERTDeterminism runs the same forward pass three times from a freshly
// seeded RNG and asserts the sha256 of the output logits is bit-identical,
// catching nondeterminism in the dispatch / kernel-cache / scheduler path.
func TestBERTDeterminism(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const (
		vocab  = 10
		nLayer = 2
		nHead  = 2
		nEmbd  = 8
		seqLen = 4
		B      = int64(1)
		T      = int64(4)
		seed   = int64(20260618)
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
		m := nn.NewBERT(a0, vocab, nLayer, nHead, nEmbd, seqLen)
		bertInitSmall(m, 0.1, rand.New(rand.NewSource(seed)))
		a := uop.NewArena(1 << 16)
		for _, p := range m.Params() {
			p.Load(a)
		}
		idxVals := make([]int32, B*T)
		for i := range idxVals {
			idxVals[i] = int32(i % vocab)
		}
		idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "cpu")
		idx.SetData(bertIdxBits(idxVals))
		y := m.Forward(idx)
		if err := tensor.Realize(y); err != nil {
			t.Fatalf("Realize: %v", err)
		}
		return hashFloats(y.Data())
	}

	h0 := runOnce()
	for i := 0; i < 2; i++ {
		if h := runOnce(); h != h0 {
			t.Fatalf("non-determinism across runs: %s != %s", h, h0)
		}
	}
	t.Logf("BERT forward deterministic across 3 runs: sha256=%s", h0)
}
