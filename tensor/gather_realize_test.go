package tensor_test

// Gather forward Realize tests: five value-oracle fixtures from
// notes/gather_slice_design.md §9 plus a JIT capture/replay smoke test (Slice
// C carried question 2).
//
// Oracles in this slice are in-process (no PyTorch):
//
//   - Direct row-slice oracle (always feasible). For Gather(W, idx, axis=0)
//     with W:[V,D], idx:[B] int32, the expected out[b, d] == W[idx[b], d].
//     We rebuild the expected tensor in Go from the same W bytes and require
//     max-abs-diff == 0.
//
//   - One-hot @ W oracle (V=8 fixture only). out == one_hot(idx, V) @ W.
//     Required exactly 0; skipped for V=50257 because the one-hot @ W path
//     is the very thing the gather pipeline is meant to avoid.
//
// Fixtures correspond 1:1 with notes/gather_slice_design.md §9. Fixture 4
// (symbolic batch) reuses fixture 3's idx values for cross-check.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── GPU setup ─────────────────────────────────────────────────────────────────

func requireGPUGather(t *testing.T) {
	t.Helper()
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

// ── Conversion helpers ────────────────────────────────────────────────────────

// i32sAsF32Bits packs int32 values as float32 bit patterns. Tensor.SetData
// uploads via float32sToBytes which preserves bit patterns; the buffer is
// declared array<i32> by the wgsl renderer (Int32 dtype), so WGSL reads the
// bytes as i32 values directly.
func i32sAsF32Bits(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

func maxAbsDiff(a, b []float32) float32 {
	var m float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

// directRowSliceOracle computes out[b, ...trail] = W[idx[b], ...trail] for
// Gather(axis=0) with W:[V, *trailShape] flattened row-major in wData.
func directRowSliceOracle(wData []float32, idxVals []int32, V int, trailSize int) []float32 {
	B := len(idxVals)
	out := make([]float32, B*trailSize)
	for b := 0; b < B; b++ {
		row := int(idxVals[b])
		copy(out[b*trailSize:(b+1)*trailSize], wData[row*trailSize:(row+1)*trailSize])
	}
	return out
}

// directRowSliceAxis1Oracle for W:[D0, V] with idx of arbitrary rank ridxShape,
// out shape is [D0, ...idxShape], out[d, j...] = W[d, idx[j...]].
func directRowSliceAxis1Oracle(wData []float32, D0, V int, idxVals []int32, idxRank int) []float32 {
	B := len(idxVals)
	out := make([]float32, D0*B)
	for d := 0; d < D0; d++ {
		for b := 0; b < B; b++ {
			out[d*B+b] = wData[d*V+int(idxVals[b])]
		}
	}
	return out
}

// oneHotAtWOracle computes one_hot(idx, V) @ W for W:[V, D] and idx:[B] →
// [B, D]. Independent path from gather; used as a second oracle for V=8.
func oneHotAtWOracle(wData []float32, idxVals []int32, V, D int) []float32 {
	B := len(idxVals)
	out := make([]float32, B*D)
	for b := 0; b < B; b++ {
		oh := make([]float32, V)
		oh[int(idxVals[b])] = 1
		for d := 0; d < D; d++ {
			var acc float32
			for v := 0; v < V; v++ {
				acc += oh[v] * wData[v*D+d]
			}
			out[b*D+d] = acc
		}
	}
	return out
}

// ── Fixtures from notes/gather_slice_design.md §9 ────────────────────────────

// Fixture 1: W=[8, 4] random, idx=[3] = [7, 0, 3], axis=0.
// Direct-slice oracle + one-hot @ W oracle. Both must match exactly.
func TestGatherRealize_Fixture1_DirectAndOneHot(t *testing.T) {
	requireGPUGather(t)

	const V, D = 8, 4
	a := uop.NewArena(1024)

	rng := rand.New(rand.NewSource(1))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := []int32{7, 0, 3}

	w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{int64(len(idxVals))}, uop.Dtypes.Int32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))

	out := w.Gather(0, idx)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	want := directRowSliceOracle(wData, idxVals, V, D)
	if got := maxAbsDiff(out.Data(), want); got != 0 {
		t.Fatalf("fixture 1 direct-slice oracle: max-abs-diff=%v (want 0)\nout=%v\nwant=%v", got, out.Data(), want)
	}

	wantOH := oneHotAtWOracle(wData, idxVals, V, D)
	if got := maxAbsDiff(out.Data(), wantOH); got != 0 {
		t.Fatalf("fixture 1 one-hot @ W oracle: max-abs-diff=%v (want 0)", got)
	}
}

// Fixture 2: W=[6, 6] random, idx=[5, 2] random in [0, 6), axis=1.
// Output shape is [6, 5, 2]. Direct-slice oracle only.
func TestGatherRealize_Fixture2_Axis1_MultiDimIndex(t *testing.T) {
	requireGPUGather(t)

	const D0, V = 6, 6
	const B0, B1 = 5, 2
	a := uop.NewArena(1024)

	rng := rand.New(rand.NewSource(2))
	wData := make([]float32, D0*V)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := make([]int32, B0*B1)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(V))
	}

	w := tensor.NewLeaf(a, []int64{D0, V}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{B0, B1}, uop.Dtypes.Int32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))

	out := w.Gather(1, idx)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	// out[d, j0, j1] = w[d, idx[j0, j1]]. Row-major flat:
	//   out_flat[d*(B0*B1) + j0*B1 + j1] = w[d*V + idx[j0*B1+j1]]
	want := directRowSliceAxis1Oracle(wData, D0, V, idxVals, 2)
	if got := maxAbsDiff(out.Data(), want); got != 0 {
		t.Fatalf("fixture 2 direct-slice oracle: max-abs-diff=%v (want 0)\nout(first 12)=%v\nwant(first 12)=%v",
			got, out.Data()[:12], want[:12])
	}
}

// Fixture 3: GPT-2 embedding shape. W=[50257, 768], idx=[16] random in
// [0, 50257), axis=0. Direct-slice oracle only.
func TestGatherRealize_Fixture3_GPT2EmbeddingDirect(t *testing.T) {
	requireGPUGather(t)

	const V, D = 50257, 768
	const B = 16
	a := uop.NewArena(8192)

	rng := rand.New(rand.NewSource(3))
	wData := make([]float32, V*D)
	for i := range wData {
		// Cheap deterministic pseudo-random; full NormFloat64 over 38M elts is
		// slow but tolerable. Keep this on a fixed seed.
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := make([]int32, B)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(V))
	}

	w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{B}, uop.Dtypes.Int32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))

	out := w.Gather(0, idx)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	want := directRowSliceOracle(wData, idxVals, V, D)
	if got := maxAbsDiff(out.Data(), want); got != 0 {
		t.Fatalf("fixture 3 direct-slice oracle: max-abs-diff=%v (want 0); sampled idx=%v", got, idxVals)
	}
}

// Fixture 4: symbolic batch case. W=[50257, 768], idx=[n] int32 where n is
// symbolic. Bind n=16, share idx values with fixture 3, require exact match.
func TestGatherRealize_Fixture4_SymbolicBatch(t *testing.T) {
	requireGPUGather(t)

	const V, D = 50257, 768
	const symVar = "n"
	const bindN = int64(16)
	a := uop.NewArena(8192)

	rng := rand.New(rand.NewSource(3)) // same seed as fixture 3 → same W and idx
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := make([]int32, bindN)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(V))
	}

	w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewSymbolicInput(a, symVar, 1, 32, uop.Dtypes.Int32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))

	out := w.Gather(0, idx)
	if err := tensor.RealizeWithBinding(map[string]int64{symVar: bindN}, out); err != nil {
		t.Fatalf("RealizeWithBinding: %v", err)
	}

	want := directRowSliceOracle(wData, idxVals, V, D)
	if got := maxAbsDiff(out.Data(), want); got != 0 {
		t.Fatalf("fixture 4 symbolic-batch oracle: max-abs-diff=%v (want 0); sampled idx=%v", got, idxVals)
	}
}

// Fixture 5: adversarial. W=[8, 4], idx=[6] = all-3s. Direct-slice oracle.
func TestGatherRealize_Fixture5_AdversarialDuplicates(t *testing.T) {
	requireGPUGather(t)

	const V, D = 8, 4
	a := uop.NewArena(1024)

	rng := rand.New(rand.NewSource(5))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := []int32{3, 3, 3, 3, 3, 3}

	w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{int64(len(idxVals))}, uop.Dtypes.Int32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))

	out := w.Gather(0, idx)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	want := directRowSliceOracle(wData, idxVals, V, D)
	if got := maxAbsDiff(out.Data(), want); got != 0 {
		t.Fatalf("fixture 5 adversarial oracle: max-abs-diff=%v (want 0)", got)
	}
}

// ── JIT capture/replay smoke test (Slice C carried question 2) ───────────────

// TestGatherRealize_JITCaptureReplayDifferentIdx captures a gather kernel
// then replays with a different idx tensor of the same shape; the replayed
// output must reflect the new idx values, not the captured ones. This proves
// the JIT cache is graph-keyed and the idx tensor flows through as runtime
// data on replay.
func TestGatherRealize_JITCaptureReplayDifferentIdx(t *testing.T) {
	requireGPUGather(t)

	const V, D = 8, 4
	wData := make([]float32, V*D)
	rng := rand.New(rand.NewSource(11))
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}

	runOnce := func(idxVals []int32) []float32 {
		a := uop.NewArena(1024)
		w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
		idx := tensor.NewLeaf(a, []int64{int64(len(idxVals))}, uop.Dtypes.Int32, "webgpu")
		w.SetData(append([]float32{}, wData...))
		idx.SetData(i32sAsF32Bits(idxVals))
		out := w.Gather(0, idx)
		return out.Data() // populated after jit.Realize below; we wrap externally
	}
	_ = runOnce // silence: helper kept for symmetry with other tests; we drive JIT inline.

	jit := tensor.NewJIT()

	// Capture step: idx = [0, 1, 2].
	idxA := []int32{0, 1, 2}
	a1 := uop.NewArena(1024)
	w1 := tensor.NewLeaf(a1, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx1 := tensor.NewLeaf(a1, []int64{int64(len(idxA))}, uop.Dtypes.Int32, "webgpu")
	w1.SetData(append([]float32{}, wData...))
	idx1.SetData(i32sAsF32Bits(idxA))
	outA := w1.Gather(0, idx1)
	if err := jit.Realize(outA); err != nil {
		t.Fatalf("JIT capture: %v", err)
	}
	wantA := directRowSliceOracle(wData, idxA, V, D)
	if d := maxAbsDiff(outA.Data(), wantA); d != 0 {
		t.Fatalf("JIT capture output: max-abs-diff=%v (want 0)", d)
	}

	// Replay step: same shapes, different idx values.
	idxB := []int32{5, 4, 7}
	a2 := uop.NewArena(1024)
	w2 := tensor.NewLeaf(a2, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx2 := tensor.NewLeaf(a2, []int64{int64(len(idxB))}, uop.Dtypes.Int32, "webgpu")
	w2.SetData(append([]float32{}, wData...))
	idx2.SetData(i32sAsF32Bits(idxB))
	outB := w2.Gather(0, idx2)
	if err := jit.Realize(outB); err != nil {
		t.Fatalf("JIT replay: %v", err)
	}
	wantB := directRowSliceOracle(wData, idxB, V, D)
	if d := maxAbsDiff(outB.Data(), wantB); d != 0 {
		t.Fatalf("JIT replay output: max-abs-diff=%v (want 0); output may be stale-captured\nout=%v\nwant=%v",
			d, outB.Data(), wantB)
	}

	caps, reps := jit.JITStats()
	if caps != 1 {
		t.Errorf("expected 1 capture, got %d", caps)
	}
	if reps != 1 {
		t.Errorf("expected 1 replay, got %d", reps)
	}
}
