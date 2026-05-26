package nn_test

// Phase 9a: parameter-persistence design proof.
//
// Two GPU tests exercise the load→step→writeback lifecycle:
//   TestSGDQuadratic_Trajectory   — 4 SGD steps on sum((p-target)²); verifies
//                                   each p_n against the closed form p_n = target
//                                   + (p0-target)*0.8^n within float32 tolerance.
//   TestCrossResetPersistence     — confirms that an update applied in step 1
//                                   is visible in step 2 built from a completely
//                                   fresh arena, proving no arena-index identity
//                                   is used to locate the value.

import (
	"math"
	"runtime"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func requireGPU(t *testing.T) {
	t.Helper()
	// Metal NSAutoreleasePool is thread-local. Lock the goroutine to one OS
	// thread so that pool creation and pool drain always run on the same thread.
	// Without this, Go's scheduler may migrate the goroutine between cgo calls,
	// causing drain to run on a different thread than create → SIGSEGV.
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	t.Cleanup(func() {
		tensor.DefaultExecutor = nil
		dev.Close()
	})
	tensor.DefaultExecutor = dev
}

// runStep builds a fresh graph, runs backward on loss = sum((pLeaf-target)²),
// realizes the gradient on the GPU, applies one SGD step, and returns the
// realized gradient values.
func runStep(t *testing.T, p *nn.Parameter, targetVals []float32, lr float32) []float32 {
	t.Helper()
	n := int64(len(p.Value))

	a := uop.NewArena(65536)
	pLeaf := p.Load(a)

	tgt := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, targetVals...))

	diff := pLeaf.Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{pLeaf})
	gradTensor, ok := grads[pLeaf]
	if !ok {
		t.Fatal("Backward: no gradient for pLeaf")
	}

	if err := tensor.Realize(gradTensor); err != nil {
		t.Fatalf("Realize(grad): %v", err)
	}
	gradData := gradTensor.Data()

	p.SGDStep(gradData, lr)
	return gradData
}

// TestSGDQuadratic_Trajectory runs 4 SGD steps on loss = sum((p - 1)²) with
// p₀ = 3.0, lr = 0.1, and verifies the numeric trajectory against the closed
// form p_n = 1 + 2·0.8^n within float32 tolerance.
//
// Analytically: grad = 2(p-target), so p_{n+1} = p_n - lr·2(p_n-target).
// Contraction factor: 1 - 2·lr = 1 - 0.2 = 0.8.
//
//	n=0: p = 3.000000  (expected 3.000000)
//	n=1: p = 2.600000  (expected 2.600000)
//	n=2: p = 2.280000  (expected 2.280000)
//	n=3: p = 2.024000  (expected 2.024000)
func TestSGDQuadratic_Trajectory(t *testing.T) {
	requireGPU(t)

	const (
		lr       = float32(0.1)
		nElems   = 4
		nSteps   = 4
		p0Val    = float32(3.0)
		tgtVal   = float32(1.0)
		tol      = float32(1e-4)
	)

	// Allocate parameter. The seed arena is ephemeral; only p.Value persists.
	seedArena := uop.NewArena(16)
	p := nn.NewParameter(seedArena, []int64{nElems}, uop.Dtypes.Float32, "webgpu")
	for i := range p.Value {
		p.Value[i] = p0Val
	}

	targetVals := make([]float32, nElems)
	for i := range targetVals {
		targetVals[i] = tgtVal
	}

	// Verify p_n before each step, run GPU grad, apply SGD.
	for step := 0; step < nSteps; step++ {
		// Closed-form expected: p_n = tgtVal + (p0Val-tgtVal)*0.8^n
		pExpected := tgtVal + (p0Val-tgtVal)*float32(math.Pow(0.8, float64(step)))
		// Expected gradient: 2*(p_n - tgtVal)
		gExpected := 2 * (pExpected - tgtVal)

		for i, v := range p.Value {
			if absF32(v-pExpected) > tol {
				t.Fatalf("step %d: p.Value[%d] = %.6f, want %.6f (closed-form diff=%.2e)",
					step, i, v, pExpected, absF32(v-pExpected))
			}
		}
		t.Logf("step %d: p=%.6f  expected=%.6f  diff=%.2e  ✓",
			step, p.Value[0], pExpected, absF32(p.Value[0]-pExpected))

		// Run GPU: compute grad = 2*(p - target), apply SGD.
		gradData := runStep(t, p, targetVals, lr)

		// Verify GPU gradient matches closed form.
		for i, g := range gradData {
			if absF32(g-gExpected) > tol {
				t.Fatalf("step %d: grad[%d] = %.6f, want %.6f (diff=%.2e)",
					step, i, g, gExpected, absF32(g-gExpected))
			}
		}
		t.Logf("step %d: grad=%.6f  expected=%.6f  diff=%.2e  ✓",
			step, gradData[0], gExpected, absF32(gradData[0]-gExpected))
	}

	// Final value: p_nSteps = tgtVal + (p0Val-tgtVal)*0.8^nSteps
	pFinal := tgtVal + (p0Val-tgtVal)*float32(math.Pow(0.8, float64(nSteps)))
	for i, v := range p.Value {
		if absF32(v-pFinal) > tol {
			t.Fatalf("final p.Value[%d] = %.6f, want %.6f", i, v, pFinal)
		}
	}
	t.Logf("final: p=%.6f  expected=%.6f  ✓  (converging to target=%.1f)", p.Value[0], pFinal, tgtVal)
}

// TestCrossResetPersistence proves that a weight update in step 1 survives
// complete arena abandonment and is correctly loaded into a fresh arena in step 2.
//
// Structural proof: Load() seeds from p.Value (a Go slice on the Parameter),
// not from any arena index. If it used step-1's arena.leaves, step 2 would
// either see a stale value (3.0) or panic (wrong arena). The GPU confirms the
// loaded value matches the updated p.Value by realizing sum(pLeaf) in step 2.
func TestCrossResetPersistence(t *testing.T) {
	requireGPU(t)

	const (
		lr    = float32(0.1)
		nElems = int64(4)
	)

	seedArena := uop.NewArena(16)
	p := nn.NewParameter(seedArena, []int64{nElems}, uop.Dtypes.Float32, "webgpu")
	for i := range p.Value {
		p.Value[i] = 3.0
	}

	// ── Step 1: arena a1 ──────────────────────────────────────────────────────
	a1 := uop.NewArena(65536)
	pLeaf1 := p.Load(a1)

	tgt1 := tensor.NewLeaf(a1, []int64{nElems}, uop.Dtypes.Float32, "webgpu")
	tgt1.SetData([]float32{1, 1, 1, 1})

	diff1 := pLeaf1.Sub(tgt1)
	loss1 := diff1.Mul(diff1).Sum(nil, false)

	grads1 := tensor.Backward(loss1, []*tensor.Tensor{pLeaf1})
	g1, ok := grads1[pLeaf1]
	if !ok {
		t.Fatal("step 1: no gradient for pLeaf1")
	}
	if err := tensor.Realize(g1); err != nil {
		t.Fatalf("step 1: Realize(grad): %v", err)
	}
	p.SGDStep(g1.Data(), lr)

	// p.Value must be updated: p1 = 3.0 - 0.1*4.0 = 2.6
	wantP1 := float32(2.6)
	for i, v := range p.Value {
		if absF32(v-wantP1) > 1e-4 {
			t.Fatalf("after step 1: p.Value[%d] = %.6f, want %.6f — SGDStep did not apply",
				i, v, wantP1)
		}
	}
	t.Logf("after step 1: p.Value = %.6f  ✓ (updated from 3.0 to %.6f)", p.Value[0], wantP1)

	// a1 is now abandoned. Its leaves map (which held the step-1 leaf data at
	// arena index 0) is no longer referenced. p.Value is NOT stored there.

	// ── Step 2: completely fresh arena, no reference to a1 ───────────────────
	a2 := uop.NewArena(65536)
	// pLeaf1 was in a1 (index 0). pLeaf2 will be in a2 (index 0 in a2, a
	// completely different *Arena pointer). The value comes from p.Value.
	pLeaf2 := p.Load(a2)

	// Realize sum(pLeaf2) on GPU. Expected: 4 * 2.6 = 10.4
	// If p.Value were stale (3.0), sum would be 12.0.
	// If Load used a1's leaves via a stale arena index, it would panic or return 0.
	sum2 := pLeaf2.Sum(nil, false)
	if err := tensor.Realize(sum2); err != nil {
		t.Fatalf("step 2: Realize(sum): %v", err)
	}

	gotSum := sum2.Data()[0]
	wantSum := float32(nElems) * wantP1 // 4 * 2.6 = 10.4

	if absF32(gotSum-wantSum) > 1e-3 {
		t.Fatalf("cross-reset: step-2 sum(pLeaf) = %.6f, want %.6f — "+
			"update did not survive arena abandonment (stale=%.1f would give sum=%.1f)",
			gotSum, wantSum, float32(3.0), float32(nElems)*3.0)
	}
	t.Logf("cross-reset: step-2 sum(pLeaf) = %.6f, expected %.6f ✓ — "+
		"updated value (%.6f) loaded correctly from fresh arena", gotSum, wantSum, wantP1)

	// Identity proof: pLeaf1 and pLeaf2 are in different arenas.
	// Their node indices happen to both be 0 (first allocation), yet they carry
	// different values (3.0 vs 2.6). No cross-arena index lookup was used.
	if pLeaf1.Arena() == pLeaf2.Arena() {
		t.Fatal("test invariant: pLeaf1 and pLeaf2 must be in different arenas")
	}
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
