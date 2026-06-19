package nn_test

// Tied-head GPT gradient correctness (R1 gate for GPT-2 fine-tuning).
//
// In a tied-head GPT (NewGPTWithTiedHead) the LM-head weight IS the token
// embedding weight: LMHead.Weight and Wte.Weight are the same *Parameter. The
// shared leaf therefore receives gradient from two distinct paths in one
// backward pass:
//
//   1. the token-embedding gather  (scatter-add backward into the table), and
//   2. the LM-head matmul          (x @ Wte.T -> logits).
//
// tensor.Backward must accumulate BOTH contributions at the shared leaf to
// produce the correct total gradient. This was deferred ("Slice O", forward
// only) while the OpExpand-backward softmax bug corrupted attention gradients;
// with that fixed (tensor/gradient_ruleset.go, 2026-06-18) tied training is
// sound. This finite-difference check runs entirely on the pure-Go CPU backend
// (gradient rules are backend-independent) and pins the guarantee so GPT-2
// fine-tuning with canonical weight tying can rely on it.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestGPTTiedHeadGradCheck(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const (
		vocab     = 8
		nLayer    = 1
		nHead     = 2
		nEmbd     = 8
		blockSize = 4
		B         = int64(1)
		T         = int64(4)
		fdH       = float32(1e-3)
		// scatter-add embedding + LM-head matmul + softmax chain: same budget as
		// the softmax-chain paths in TestGPTFDGradCheck.
		tol    = float32(7e-2)
		nCheck = 12
	)

	rng := rand.New(rand.NewSource(7))
	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(vocab))
	}

	a0 := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a0, vocab, nLayer, nHead, nEmbd, blockSize)
	if g.LMHead.Weight != g.Wte.Weight {
		t.Fatalf("expected tied head: LMHead.Weight must be the same *Parameter as Wte.Weight")
	}
	for _, p := range g.Params() {
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * 0.1
		}
	}
	snap := make([][]float32, len(g.Params()))
	for i, p := range g.Params() {
		snap[i] = append([]float32(nil), p.Value...)
	}
	restore := func() {
		for i, p := range g.Params() {
			copy(p.Value, snap[i])
		}
	}

	// Analytic gradient of the tied Wte.Weight via Backward.
	a := uop.NewArena(1 << 18)
	for _, p := range g.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "cpu")
	idx.SetData(gptIdxBitsForLeaf(idxVals))
	loss := g.Forward(idx).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{g.Wte.Weight.T})
	gw := grads[g.Wte.Weight.T]
	if gw == nil {
		t.Fatalf("no gradient produced for tied Wte.Weight")
	}
	if err := tensor.Realize(gw); err != nil {
		t.Fatalf("realize tied grad: %v", err)
	}
	analytic := append([]float32(nil), gw.Data()...)

	lossVal := func() float32 {
		aa := uop.NewArena(1 << 18)
		for _, p := range g.Params() {
			p.Load(aa)
		}
		id := tensor.NewLeaf(aa, []int64{B, T}, uop.Dtypes.Int32, "cpu")
		id.SetData(gptIdxBitsForLeaf(idxVals))
		l := g.Forward(id).Sum(nil, false)
		if err := tensor.Realize(l); err != nil {
			t.Fatalf("realize loss: %v", err)
		}
		return l.Data()[0]
	}

	n := nCheck
	if n > len(analytic) {
		n = len(analytic)
	}
	worst := float32(0)
	for i := 0; i < n; i++ {
		restore()
		g.Wte.Weight.Value[i] += fdH
		up := lossVal()
		restore()
		g.Wte.Weight.Value[i] -= fdH
		dn := lossVal()
		restore()
		fd := (up - dn) / (2 * fdH)
		av := analytic[i]
		diff := float32(math.Abs(float64(av - fd)))
		scale := float32(math.Max(math.Max(math.Abs(float64(av)), math.Abs(float64(fd))), 1.0))
		rel := diff / scale
		if rel > worst {
			worst = rel
		}
		if math.Abs(float64(fd)) > 5e-3 && (av > 0) != (fd > 0) {
			t.Errorf("tied Wte.Weight[%d] sign-wrong: analytic=%.6f fd=%.6f", i, av, fd)
		}
		if rel > tol {
			t.Fatalf("tied Wte.Weight[%d]: analytic=%.6f fd=%.6f rel=%.2e > tol=%.2e", i, av, fd, rel, tol)
		}
	}
	t.Logf("tied Wte.Weight gradient agrees with FD (worst rel=%.2e over %d indices)", worst, n)
}
