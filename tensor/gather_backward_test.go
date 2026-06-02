package tensor_test

// Gather backward (scatter-add) Realize tests; Slice D value-oracle gates.
//
// All three backward fixtures from notes/gather_slice_design.md §9, plus:
//   - FD gradient check at 1e-3 relative tolerance.
//   - 3-run determinism (sha256-equal across runs).
//   - JIT capture/replay with a fresh idx tensor on replay.
//
// Oracles are constructed in-process (no PyTorch available). The expected
// dW for a (W, idx, dY) triple is computed by scanning idx in Go and
// accumulating dY rows into the right dW slots; correct by inspection for
// the simple fixtures used here.

import (
	"crypto/sha256"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── GPU setup ─────────────────────────────────────────────────────────────────

func requireGPUGatherBack(t *testing.T) {
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

// ── Oracle ─────────────────────────────────────────────────────────────────

// expectedDW computes dW[v, d] = sum over b where idx[b]==v of dY[b, d].
// Mirrors PyTorch's `torch.zeros(V, D).scatter_add_(0, idx_expand, dY)`.
func expectedDW(idx []int32, dY []float32, V, D int) []float32 {
	out := make([]float32, V*D)
	B := len(idx)
	for b := 0; b < B; b++ {
		v := int(idx[b])
		for d := 0; d < D; d++ {
			out[v*D+d] += dY[b*D+d]
		}
	}
	return out
}

// runEmbeddingBackward builds W:[V,D] + idx:[B] + dY:[B,D] tensors on a fresh
// arena, computes loss = sum(W.Gather(0, idx) * dY) (so the adjoint of the
// gather is exactly dY), and returns dW = Backward(loss, {W})[W]'s data.
func runEmbeddingBackward(t *testing.T, V, D int, idxVals []int32, wData, dY []float32) []float32 {
	t.Helper()
	a := uop.NewArena(2048)
	w := tensor.NewLeaf(a, []int64{int64(V), int64(D)}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{int64(len(idxVals))}, uop.Dtypes.Int32, "webgpu")
	dy := tensor.NewLeaf(a, []int64{int64(len(idxVals)), int64(D)}, uop.Dtypes.Float32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))
	dy.SetData(append([]float32{}, dY...))

	gather := w.Gather(0, idx)
	prod := gather.Mul(dy)
	loss := prod.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{w})
	dW, ok := grads[w]
	if !ok {
		t.Fatal("Backward returned no gradient for W")
	}
	if err := tensor.Realize(dW); err != nil {
		t.Fatalf("Realize dW: %v", err)
	}
	return dW.Data()
}

// ── Backward fixture 1: no collisions ────────────────────────────────────

// Fixture 1: W=[8, 4], idx=[3] = [7, 0, 3], dY random [3, 4].
// Expected: dW[7]=dY[0], dW[0]=dY[1], dW[3]=dY[2], rest zero.
func TestGatherBackward_Fixture1_NoCollisions(t *testing.T) {
	requireGPUGatherBack(t)

	const V, D = 8, 4
	rng := rand.New(rand.NewSource(101))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := []int32{7, 0, 3}
	dY := make([]float32, len(idxVals)*D)
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	got := runEmbeddingBackward(t, V, D, idxVals, wData, dY)
	want := expectedDW(idxVals, dY, V, D)

	if diff := maxAbsDiff(got, want); diff != 0 {
		t.Fatalf("fixture 1 backward: max-abs-diff=%v (want 0)\nidx=%v\nwant=%v\ngot=%v", diff, idxVals, want, got)
	}
}

// ── Backward fixture 2: heavy duplicate ─────────────────────────────────

// Fixture 2: idx=[2,2,2,5,2]. dW[2] accumulates four dY rows; dW[5] gets one.
// The host sort produces sortedIdx=[2,2,2,2,5] with perm=[0,1,2,4,3]; the
// segment-sum kernel reads dY rows in sorted order and accumulates them in a
// thread-private scalar. Sum order matches the natural integer order of B,
// so for normally-distributed dY the result is bit-identical to the naive
// row-major accumulation; diff is exactly 0.
func TestGatherBackward_Fixture2_HeavyDuplicate(t *testing.T) {
	requireGPUGatherBack(t)

	const V, D = 8, 4
	rng := rand.New(rand.NewSource(102))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := []int32{2, 2, 2, 5, 2}
	dY := make([]float32, len(idxVals)*D)
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	got := runEmbeddingBackward(t, V, D, idxVals, wData, dY)
	want := expectedDW(idxVals, dY, V, D)

	// Sort within a segment can reorder float-add ops; design §9 allows up
	// to 1e-6 here. The host preprocessor uses stable sort, and the kernel
	// reduces in sorted-position order, so reordering follows the natural
	// integer order of B (no change) - we expect a tighter bound but accept
	// the design budget defensively.
	if diff := maxAbsDiff(got, want); diff > 1e-6 {
		t.Fatalf("fixture 2 backward (heavy dup): max-abs-diff=%v (want <= 1e-6)\nidx=%v", diff, idxVals)
	}
}

// ── Backward fixture 3: symbolic-batch ──────────────────────────────────

// Fixture 3: W=[V, D] with symbolic idx[n] where n in [1, 32], bind n=8.
// Compare against the same idx-and-dY oracle on a static-shape rebuild.
func TestGatherBackward_Fixture3_SymbolicBatch(t *testing.T) {
	requireGPUGatherBack(t)

	const V, D = 16, 4
	const symVar = "n"
	const bindN = int64(8)

	rng := rand.New(rand.NewSource(103))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := make([]int32, bindN)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(V))
	}
	dY := make([]float32, int(bindN)*D)
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	a := uop.NewArena(2048)
	w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewSymbolicInput(a, symVar, 1, 32, uop.Dtypes.Int32, "webgpu")
	dy := tensor.NewSymbolicBatchInput(a, symVar, 1, 32, []int64{D}, uop.Dtypes.Float32, "webgpu")
	w.SetData(append([]float32{}, wData...))
	idx.SetData(i32sAsF32Bits(idxVals))
	dy.SetData(append([]float32{}, dY...))

	gather := w.Gather(0, idx)
	prod := gather.Mul(dy)
	loss := prod.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{w})
	dW, ok := grads[w]
	if !ok {
		t.Fatal("Backward returned no gradient for W under symbolic batch")
	}
	if err := tensor.RealizeWithBinding(map[string]int64{symVar: bindN}, dW); err != nil {
		t.Fatalf("RealizeWithBinding dW: %v", err)
	}

	want := expectedDW(idxVals, dY, V, D)
	if diff := maxAbsDiff(dW.Data(), want); diff > 1e-6 {
		t.Fatalf("fixture 3 symbolic backward: max-abs-diff=%v (want <= 1e-6)\nidx=%v", diff, idxVals)
	}
}

// ── Determinism: 3 runs of the same backward produce bit-identical dW ────

// TestGatherBackward_Determinism is the load-bearing architectural gate:
// host sort is deterministic, segment-sum is race-free, so three independent
// runs of the same backward computation must produce byte-identical dW.
// A regression here breaks the design's "no atomics, no nondeterministic
// reductions" guarantee.
func TestGatherBackward_Determinism(t *testing.T) {
	requireGPUGatherBack(t)

	const V, D = 8, 4
	rng := rand.New(rand.NewSource(201))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := []int32{2, 5, 2, 0, 2}
	dY := make([]float32, len(idxVals)*D)
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	hashes := make([][32]byte, 3)
	for run := 0; run < 3; run++ {
		got := runEmbeddingBackward(t, V, D, idxVals, wData, dY)
		hashes[run] = sha256.Sum256(float32sToBytesForHash(got))
	}
	if hashes[0] != hashes[1] || hashes[1] != hashes[2] {
		t.Fatalf("determinism gate FAILED: dW hashes differ across 3 runs\nrun0=%x\nrun1=%x\nrun2=%x",
			hashes[0], hashes[1], hashes[2])
	}
}

// float32sToBytesForHash converts a float32 slice to its little-endian byte
// representation for SHA-256 hashing. Mirrors the encoding the executor
// pushes to / receives from the GPU.
func float32sToBytesForHash(in []float32) []byte {
	out := make([]byte, 4*len(in))
	for i, v := range in {
		bits := math.Float32bits(v)
		out[4*i] = byte(bits)
		out[4*i+1] = byte(bits >> 8)
		out[4*i+2] = byte(bits >> 16)
		out[4*i+3] = byte(bits >> 24)
	}
	return out
}

// ── FD gradient check ────────────────────────────────────────────────────

// TestGatherBackward_FDCheck runs central-difference finite differences on
// fixture 1 (V=8, D=4, no collisions) and asserts max relative error <= 1e-3.
//
// The reduction is loss = sum(W.Gather(0, idx) * dY). dW[v, d] is exact
// (analytic) and equal to expectedDW(idx, dY)[v, d]. FD perturbs each W
// entry by ±h=1e-3 and compares.
func TestGatherBackward_FDCheck(t *testing.T) {
	requireGPUGatherBack(t)

	const V, D = 8, 4
	const h = 1e-3
	const relTol = 1e-3

	rng := rand.New(rand.NewSource(301))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	idxVals := []int32{7, 0, 3}
	dY := make([]float32, len(idxVals)*D)
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	dW := runEmbeddingBackward(t, V, D, idxVals, wData, dY)

	// Loss as a function of W (treating idx and dY as constants):
	//   loss(W) = sum_{b, d} W[idx[b], d] * dY[b, d]
	// FD probes each W entry; for entries that don't appear in idx the
	// analytic gradient is zero and the FD is also zero (loss flat), so
	// relative error is 0/0; skip those with an absolute-tolerance check.
	loss := func(w []float32) float64 {
		var s float64
		for b, ib := range idxVals {
			for d := 0; d < D; d++ {
				s += float64(w[int(ib)*D+d]) * float64(dY[b*D+d])
			}
		}
		return s
	}

	wPert := make([]float32, V*D)
	copy(wPert, wData)

	var maxRel float64
	for i := 0; i < V*D; i++ {
		orig := wPert[i]
		wPert[i] = orig + h
		lp := loss(wPert)
		wPert[i] = orig - h
		lm := loss(wPert)
		wPert[i] = orig
		fd := (lp - lm) / (2 * h)
		ana := float64(dW[i])
		diff := math.Abs(fd - ana)
		if math.Abs(ana) < 1e-6 && math.Abs(fd) < 1e-6 {
			continue
		}
		denom := math.Max(math.Abs(ana), math.Abs(fd))
		rel := diff / denom
		if rel > maxRel {
			maxRel = rel
		}
	}
	if maxRel > relTol {
		t.Fatalf("FD gradient check: max relative error %.6g exceeds tolerance %.0e", maxRel, relTol)
	}
}

// ── JIT capture + replay with a different idx ───────────────────────────

// TestGatherBackward_JITReplayDifferentIdx captures a backward at idx=[0,1,2]
// and replays at idx=[5,4,7] (same shape, different values). The replayed
// dW must reflect the new idx values: the host preprocessor reruns on the
// fresh arena every replay so sortedIdx / perm leaves carry current data,
// and the JIT plan stays graph-keyed. A stale dW indicates capture-time
// data leaking into replay.
func TestGatherBackward_JITReplayDifferentIdx(t *testing.T) {
	requireGPUGatherBack(t)

	const V, D = 8, 4
	rng := rand.New(rand.NewSource(401))
	wData := make([]float32, V*D)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	dY := make([]float32, 3*D)
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	captureBackward := func(jit *tensor.JIT, idxVals []int32) []float32 {
		a := uop.NewArena(2048)
		w := tensor.NewLeaf(a, []int64{int64(V), int64(D)}, uop.Dtypes.Float32, "webgpu")
		idx := tensor.NewLeaf(a, []int64{int64(len(idxVals))}, uop.Dtypes.Int32, "webgpu")
		dy := tensor.NewLeaf(a, []int64{int64(len(idxVals)), int64(D)}, uop.Dtypes.Float32, "webgpu")
		w.SetData(append([]float32{}, wData...))
		idx.SetData(i32sAsF32Bits(idxVals))
		dy.SetData(append([]float32{}, dY...))

		gather := w.Gather(0, idx)
		prod := gather.Mul(dy)
		loss := prod.Sum(nil, false)
		grads := tensor.Backward(loss, []*tensor.Tensor{w})
		dW := grads[w]
		if err := jit.Realize(dW); err != nil {
			t.Fatalf("JIT.Realize: %v", err)
		}
		out := make([]float32, V*D)
		copy(out, dW.Data())
		return out
	}

	jit := tensor.NewJIT()

	idxA := []int32{0, 1, 2}
	gotA := captureBackward(jit, idxA)
	wantA := expectedDW(idxA, dY, V, D)
	if diff := maxAbsDiff(gotA, wantA); diff != 0 {
		t.Fatalf("JIT capture: max-abs-diff=%v (want 0)", diff)
	}

	idxB := []int32{5, 4, 7}
	gotB := captureBackward(jit, idxB)
	wantB := expectedDW(idxB, dY, V, D)
	if diff := maxAbsDiff(gotB, wantB); diff != 0 {
		t.Fatalf("JIT replay with new idx: max-abs-diff=%v (want 0); idx leak suspected\ngot=%v\nwant=%v",
			diff, gotB, wantB)
	}

	caps, reps := jit.JITStats()
	if caps != 1 {
		t.Errorf("expected 1 capture, got %d", caps)
	}
	if reps != 1 {
		t.Errorf("expected 1 replay, got %d", reps)
	}
}
