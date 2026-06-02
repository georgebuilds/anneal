package nn_test

// Slice E value-oracle gates for nn.Embedding. Three GPU tests, one per
// design-doc gate:
//
//   TestEmbedding_ForwardGPT2Shape: Forward at GPT-2 scale (V=50257, D=768)
//     matches the direct row-slice oracle with max-abs-diff 0.
//   TestEmbedding_BackwardCountOracle: loss = embedding(idx).Sum() produces a
//     gradient whose row i equals count(i in idx) * 1.0 for every unique i,
//     zero elsewhere. Directly exercises the Slice D scatter-add at the
//     Module level.
//   TestEmbedding_TrainConvergence: small SGD loop on
//     loss = ||embedding(idx) - target||^2 drives the loss to <= 10 % of its
//     initial value within a fixed step budget.
//
// All three skip when no GPU device is available (same convention as the
// rest of tensor/nn). Random data is seeded so failures reproduce.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Helpers ───────────────────────────────────────────────────────────────────

// idxBitsForLeaf packs int32 index values as float32 bit patterns so they can
// be uploaded via tensor.SetData. The Int32 leaf reads the same bytes back as
// i32 on the device (mirrors tensor/gather_realize_test.i32sAsF32Bits).
func idxBitsForLeaf(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

// maxAbsDiffF32 returns the elementwise max-abs-diff of two float32 slices.
// Local copy to keep the test file self-contained.
func maxAbsDiffF32(a, b []float32) float32 {
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

// ── Gate 1: forward correctness at GPT-2 scale ────────────────────────────────

// TestEmbedding_ForwardGPT2Shape checks that Embedding(50257, 768).Forward(idx)
// returns the exact rows of Weight indexed by idx. Direct-row-slice oracle:
// out[b, d] == Weight.Value[idx[b]*D + d] for every (b, d). Max-abs-diff must
// be exactly 0 (integer indexing, no float rounding).
func TestEmbedding_ForwardGPT2Shape(t *testing.T) {
	requireGPU(t)

	const (
		V = int64(50257)
		D = int64(768)
		B = int64(16)
	)

	// Seed the Weight with small normal samples. The exact distribution does
	// not matter; the gate is bit-exact row equality.
	rng := rand.New(rand.NewSource(7))
	idxVals := make([]int32, B)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(int(V)))
	}

	a0 := uop.NewArena(64)
	emb := nn.NewEmbedding(a0, V, D, uop.Dtypes.Float32, "webgpu")
	for i := range emb.Weight.Value {
		emb.Weight.Value[i] = float32(rng.NormFloat64()) * 0.02
	}

	a := uop.NewArena(1 << 20)
	emb.Weight.Load(a)

	idx := tensor.NewLeaf(a, []int64{B}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(idxBitsForLeaf(idxVals))

	out := emb.Forward(idx)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize embedding forward: %v", err)
	}

	got := out.Data()
	if int64(len(got)) != B*D {
		t.Fatalf("output length %d != B*D=%d", len(got), B*D)
	}

	// Direct row-slice oracle from the master Weight.Value.
	want := make([]float32, B*D)
	for b := int64(0); b < B; b++ {
		row := int64(idxVals[b])
		copy(want[b*D:(b+1)*D], emb.Weight.Value[row*D:(row+1)*D])
	}

	if diff := maxAbsDiffF32(got, want); diff != 0 {
		t.Fatalf("forward direct-row-slice oracle: max-abs-diff=%v (want 0)\nidx=%v",
			diff, idxVals)
	}
	t.Logf("forward GPT-2 shape oracle ✓  V=%d D=%d B=%d  diff=0", V, D, B)
}

// ── Gate 2: backward "count of occurrences" oracle ────────────────────────────

// TestEmbedding_BackwardCountOracle exercises the Slice D scatter-add path
// at the Module wrapper level. With loss = sum(embedding(idx)), the upstream
// gradient is all-ones, so each gather output row contributes a +1 row to
// dW[idx[b]]. The expected gradient therefore has dW[i] == count(i in idx)
// for every row i in [0, V), with the row repeated across all D columns
// (since every dY entry is 1).
//
// idx is chosen with multiple duplicates so the gate also covers the
// segment-sum accumulation path: idx=[2, 2, 5, 0, 2] -> dW[0]=1, dW[2]=3,
// dW[5]=1, rest zero, every row uniformly across D columns.
//
// A regression in the Slice D scatter-add (e.g. dropping duplicates, double
// counting, off-by-one in segment boundaries) would fail this gate.
func TestEmbedding_BackwardCountOracle(t *testing.T) {
	requireGPU(t)

	const (
		V = int64(8)
		D = int64(4)
	)
	idxVals := []int32{2, 2, 5, 0, 2}
	B := int64(len(idxVals))

	// Initial Weight values are irrelevant to the count-of-occurrences gate:
	// d/dW (sum(gather(W, idx))) does not depend on W. Seed with non-zero
	// random data so any accidental "skip rows where W is zero" bug shows.
	rng := rand.New(rand.NewSource(11))

	a0 := uop.NewArena(64)
	emb := nn.NewEmbedding(a0, V, D, uop.Dtypes.Float32, "webgpu")
	for i := range emb.Weight.Value {
		emb.Weight.Value[i] = float32(rng.NormFloat64())
	}

	a := uop.NewArena(1 << 16)
	wLeaf := emb.Weight.Load(a)

	idx := tensor.NewLeaf(a, []int64{B}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(idxBitsForLeaf(idxVals))

	out := emb.Forward(idx)
	loss := out.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{wLeaf})
	dW, ok := grads[wLeaf]
	if !ok {
		t.Fatal("Backward returned no gradient for Weight")
	}
	if err := tensor.Realize(dW); err != nil {
		t.Fatalf("Realize dW: %v", err)
	}

	got := dW.Data()
	if int64(len(got)) != V*D {
		t.Fatalf("dW length %d != V*D=%d", len(got), V*D)
	}

	// Expected: every row equals count(row in idx) * [1, 1, ..., 1].
	counts := make([]int, V)
	for _, v := range idxVals {
		counts[v]++
	}
	want := make([]float32, V*D)
	for v := int64(0); v < V; v++ {
		c := float32(counts[v])
		for d := int64(0); d < D; d++ {
			want[v*D+d] = c
		}
	}

	if diff := maxAbsDiffF32(got, want); diff != 0 {
		t.Fatalf("backward count-of-occurrences oracle: max-abs-diff=%v (want 0)\nidx=%v\ncounts=%v\ngot=%v\nwant=%v",
			diff, idxVals, counts, got, want)
	}
	t.Logf("backward count oracle ✓  counts=%v  diff=0", counts)
}

// ── Gate 3: toy-task training convergence ─────────────────────────────────────

// TestEmbedding_TrainConvergence verifies end-to-end training through the
// embedding by minimising ||embedding(idx) - target||^2 over a small fixed
// (idx, target) batch. The only learnable parameter is the embedding Weight,
// so the optimal solution is Weight[idx[b]] = target[b] for every b (with
// repeats averaged). The loss is bounded below by zero and monotonically
// decreasable under gradient descent; we assert the loss at step N is at
// most 10 % of its step-0 value.
//
// A failure could mean: (a) scatter-add backward is wrong (mass not landing
// in dW[idx]), (b) gradient is the wrong sign, (c) LR mistuned. The report
// logs the trajectory so the operator can distinguish (c) from (a)/(b).
func TestEmbedding_TrainConvergence(t *testing.T) {
	requireGPU(t)

	const (
		V        = int64(16)
		D        = int64(8)
		B        = int64(6)
		lr       = float32(0.1)
		nSteps   = 200
		logEvery = 25
	)

	rng := rand.New(rand.NewSource(23))

	idxVals := make([]int32, B)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(int(V)))
	}
	targetVals := make([]float32, B*D)
	for i := range targetVals {
		targetVals[i] = float32(rng.NormFloat64()) * 0.5
	}

	a0 := uop.NewArena(64)
	emb := nn.NewEmbedding(a0, V, D, uop.Dtypes.Float32, "webgpu")
	// Init at zero so loss(0) = sum(target^2). This makes the initial loss
	// trivially predictable for the report.
	for i := range emb.Weight.Value {
		emb.Weight.Value[i] = 0
	}

	opt := nn.NewSGD(emb.Params(), lr)

	// evalLoss rebuilds the graph in a fresh arena and returns
	// loss = sum((embedding(idx) - target)^2). The arena is discarded after
	// each call; emb.Weight.Value persists across calls via Parameter.Load.
	evalLoss := func() float32 {
		t.Helper()
		a := uop.NewArena(1 << 16)
		emb.Weight.Load(a)
		idx := tensor.NewLeaf(a, []int64{B}, uop.Dtypes.Int32, "webgpu")
		idx.SetData(idxBitsForLeaf(idxVals))
		tgt := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(append([]float32{}, targetVals...))
		diff := emb.Forward(idx).Sub(tgt)
		loss := diff.Mul(diff).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("evalLoss: Realize: %v", err)
		}
		return loss.Data()[0]
	}

	// trainStep runs one forward + backward + SGD update on a fresh arena.
	trainStep := func() {
		t.Helper()
		a := uop.NewArena(1 << 16)
		wLeaf := emb.Weight.Load(a)
		idx := tensor.NewLeaf(a, []int64{B}, uop.Dtypes.Int32, "webgpu")
		idx.SetData(idxBitsForLeaf(idxVals))
		tgt := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(append([]float32{}, targetVals...))

		diff := emb.Forward(idx).Sub(tgt)
		loss := diff.Mul(diff).Sum(nil, false)

		grads := tensor.Backward(loss, []*tensor.Tensor{wLeaf})
		g, ok := grads[wLeaf]
		if !ok {
			t.Fatal("trainStep: no gradient for Weight")
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("trainStep: Realize grad: %v", err)
		}
		opt.Step(grads)
	}

	loss0 := evalLoss()
	t.Logf("step %4d: loss=%.6f", 0, loss0)
	if loss0 <= 0 {
		t.Fatalf("trivial initial loss=%v; expected positive (target has non-zero rows)", loss0)
	}

	var lossFinal float32
	for step := 1; step <= nSteps; step++ {
		trainStep()
		if step%logEvery == 0 || step == nSteps {
			l := evalLoss()
			t.Logf("step %4d: loss=%.6f  ratio=%.4f", step, l, l/loss0)
			if step == nSteps {
				lossFinal = l
			}
		}
	}

	ratio := lossFinal / loss0
	if ratio > 0.1 {
		t.Fatalf("embedding did not converge: loss0=%.6f loss%d=%.6f ratio=%.4f (want <= 0.1)",
			loss0, nSteps, lossFinal, ratio)
	}
	t.Logf("convergence ✓  initial=%.6f  final=%.6f  ratio=%.4f  (≤ 0.1)",
		loss0, lossFinal, ratio)
}
