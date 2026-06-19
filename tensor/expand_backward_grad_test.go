package tensor_test

// Regression test for the OpExpand backward broadcast-axis bug.
//
// Bug (fixed 2026-06-18): OpExpand backward in tensor/gradient_ruleset.go summed
// the adjoint over EVERY source axis of size 1, including axes that were not
// actually broadcast (source size 1 AND expanded size 1). Summing a non-broadcast
// size-1 axis is a value no-op in a correct reducer, but it emits a multi-axis
// reduce that the scheduler's index lowering miscompiles, sign-flipping the
// gradient whenever a reduce-normalized "diamond" value (softmax's div-by-sum,
// y = e / sum(e)) feeds a matmul. This silently corrupted the gradient of every
// softmax-attention parameter (nanoGPT / ViT / GPT-2) and propagated upstream
// into the token-embedding gradient. The fix: only sum axes that were genuinely
// broadcast (src==1 AND expanded>1).
//
// This test pins the fix on the pure-Go CPU backend (gradient rules are
// backend-independent), so it is fast and GPU-free. It was previously a
// documented-skip in the Block/GPT/ViT FD tests; those FD paths are re-enabled
// alongside this test.

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// diamondMatmulLoss builds the exact failing pattern: a softmax-style
// div-by-sum diamond whose output feeds a matmul, returning the score leaf and
// the scalar loss. B is the (possibly size-1) leading batch dim that triggered
// the bug.
func diamondMatmulLoss(a *uop.Arena, sData, vData []float32, B, H, T, D int64) (*tensor.Tensor, *tensor.Tensor) {
	s := tensor.NewLeaf(a, []int64{B, H, T, T}, uop.Dtypes.Float32, "cpu")
	s.SetData(sData)
	v := tensor.NewLeaf(a, []int64{B, H, T, D}, uop.Dtypes.Float32, "cpu")
	v.SetData(vData)
	expv := s.Exp()
	den := expv.Sum([]int{3}, false).Reshape([]int64{B, H, T, 1}) // diamond: expv used twice
	att := expv.Div(den)
	out := att.Matmul(v)
	return s, out.Sum(nil, false)
}

func TestExpandBackwardDiamondMatmulGradient(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	// B=1 is the critical case: the spurious size-1-axis reduce is on the batch
	// dim, so B=1 is exactly where the old code misbehaved. B=2 guards the
	// general case.
	for _, B := range []int64{1, 2} {
		B := B
		t.Run("B="+strconv.FormatInt(B, 10), func(t *testing.T) {
			const H, T, D = int64(2), int64(4), int64(3)
			rng := rand.New(rand.NewSource(7))
			sData := make([]float32, int(B*H*T*T))
			for i := range sData {
				sData[i] = float32(rng.NormFloat64()) * 0.5
			}
			vData := make([]float32, int(B*H*T*D))
			for i := range vData {
				vData[i] = float32(rng.NormFloat64()) * 0.5
			}

			a := uop.NewArena(1 << 18)
			sT, loss := diamondMatmulLoss(a, sData, vData, B, H, T, D)
			g := tensor.Backward(loss, []*tensor.Tensor{sT})[sT]
			if g == nil {
				t.Fatal("no gradient produced for scores")
			}
			if err := tensor.Realize(g); err != nil {
				t.Fatalf("realize grad: %v", err)
			}
			analytic := append([]float32(nil), g.Data()...)

			lossVal := func(sd []float32) float32 {
				aa := uop.NewArena(1 << 18)
				_, l := diamondMatmulLoss(aa, sd, vData, B, H, T, D)
				if err := tensor.Realize(l); err != nil {
					t.Fatalf("realize loss: %v", err)
				}
				return l.Data()[0]
			}

			const h = float32(1e-3)
			const tol = 1e-2 // central-difference truncation budget for exp/div chain
			maxDiff := 0.0
			worstIdx := -1
			for i := range sData {
				up := append([]float32(nil), sData...)
				up[i] += h
				dn := append([]float32(nil), sData...)
				dn[i] -= h
				fd := (lossVal(up) - lossVal(dn)) / (2 * h)
				d := math.Abs(float64(analytic[i] - fd))
				if d > maxDiff {
					maxDiff = d
					worstIdx = i
				}
				// Sign agreement is the load-bearing check: the bug flipped signs.
				if math.Abs(float64(fd)) > 5e-3 && (analytic[i] > 0) != (fd > 0) {
					t.Errorf("sign-flipped gradient at s[%d]: analytic=%.5f fd=%.5f", i, analytic[i], fd)
				}
			}
			if maxDiff > tol {
				t.Errorf("diamond->matmul gradient drifts from finite differences: maxAbsDiff=%.6f (tol=%.0e) at index %d",
					maxDiff, tol, worstIdx)
			}
		})
	}
}
