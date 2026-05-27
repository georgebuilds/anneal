package webgpu_test

// GPU-side proofs for the schedule memoization cache (Phase cache).
//
// Proof 1 — Hit correctness: the same logical graph scheduled twice returns the
//   cached schedule, and running both produces byte-identical GPU output.
//
// Proof 2 — Symbolic one-schedule-two-bindings: a dynamic-batch graph produces
//   one cache entry (1 miss + 1 hit) and both batch=4 and batch=32 dispatch
//   correctly from that single cached schedule.
//
// Proof 3 — Training-loop hit count: N forward passes over the same graph
//   produce 1 miss + (N-1) hits, demonstrating that re-scheduling is skipped on
//   every subsequent step.

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Proof 1: hit correctness ──────────────────────────────────────────────────

// TestScheduleCache_HitCorrectness schedules a matmul-like graph twice on the
// same arena, asserts the second call is a hit, then runs both returned schedules
// on the GPU and confirms the outputs are bit-identical (max-abs-diff = 0).
//
// The correctness failure mode being guarded against: a cache that returns a
// structurally-matching but wrong schedule (stale arena indices, wrong buffers).
func TestScheduleCache_HitCorrectness(t *testing.T) {
	dev := requireDevice(t)
	schedule.ResetScheduleCache()

	const device = "webgpu"
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	a := uop.NewArena(1024)

	// A sum-reduce followed by element-wise exp2 — two kernels.
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, device)
	y := x.Sum([]int{1}, false) // [4]
	z := y.Exp2()               // [4]

	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = float32(i) * 0.1
	}
	x.SetData(xData)

	// First realize — cache miss.
	if err := tensor.Realize(z); err != nil {
		t.Fatalf("first Realize: %v", err)
	}
	out1 := make([]float32, len(z.Data()))
	copy(out1, z.Data())

	hits1, misses1 := schedule.ScheduleCacheStats()
	if misses1 != 1 || hits1 != 0 {
		t.Fatalf("after first realize: want 1 miss 0 hits, got misses=%d hits=%d", misses1, hits1)
	}

	// Second realize — cache hit.  Output data must match bit-for-bit.
	if err := tensor.Realize(z); err != nil {
		t.Fatalf("second Realize: %v", err)
	}
	out2 := z.Data()

	hits2, misses2 := schedule.ScheduleCacheStats()
	if misses2 != 1 || hits2 != 1 {
		t.Fatalf("after second realize: want 1 miss 1 hit, got misses=%d hits=%d", misses2, hits2)
	}

	if len(out1) != len(out2) {
		t.Fatalf("output length changed: %d vs %d", len(out1), len(out2))
	}
	var maxDiff float64
	for i := range out1 {
		if d := math.Abs(float64(out1[i] - out2[i])); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("=== PROOF 1: hit correctness ===")
	t.Logf("first output:  %v", out1)
	t.Logf("second output: %v", out2)
	t.Logf("max-abs-diff between first (miss) and second (hit) run: %.2e  (want 0)", maxDiff)
	if maxDiff != 0 {
		t.Errorf("FAIL: cached schedule produced different output; max-abs-diff=%.2e", maxDiff)
	}
}

// ── Proof 2: symbolic one-schedule-two-bindings ───────────────────────────────

// TestScheduleCache_SymbolicTwoBindings proves that a symbolic-batch graph
// produces exactly one cache entry (one miss on the first RealizeWithBinding,
// one hit on the second) and that both dispatches — with batch=4 and batch=32 —
// produce correct results against a CPU reference.
//
// This demonstrates that the binding value is NOT baked into the cache key and
// that the same compiled schedule handles all concrete batch sizes.
func TestScheduleCache_SymbolicTwoBindings(t *testing.T) {
	dev := requireDevice(t)
	schedule.ResetScheduleCache()

	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const device = "webgpu"
	a := uop.NewArena(512)

	// Element-wise scale: y = x * 2.0 where x has shape [n, 4].
	x := tensor.NewSymbolicBatchInput(a, "n", 1, 256, []int64{4}, uop.Dtypes.Float32, device)
	scale := tensor.Full(a, []int64{1, 4}, 2.0, uop.Dtypes.Float32, device)
	y := x.Mul(scale)

	// CPU reference.
	cpuScale := func(data []float32) []float32 {
		out := make([]float32, len(data))
		for i, v := range data {
			out[i] = v * 2.0
		}
		return out
	}

	runBatch := func(batch int) ([]float32, float64) {
		t.Helper()
		data := make([]float32, batch*4)
		for i := range data {
			data[i] = float32(i+1) * 0.5
		}
		x.SetData(data)
		if err := tensor.RealizeWithBinding(map[string]int64{"n": int64(batch)}, y); err != nil {
			t.Fatalf("RealizeWithBinding(batch=%d): %v", batch, err)
		}
		got := y.Data()
		want := cpuScale(data)
		var maxErr float64
		for i := range got {
			if e := math.Abs(float64(got[i] - want[i])); e > maxErr {
				maxErr = e
			}
		}
		return got, maxErr
	}

	// Dispatch 1: batch=4 — this is the first schedule call (cache miss).
	got4, err4 := runBatch(4)
	hits1, misses1 := schedule.ScheduleCacheStats()
	if misses1 != 1 || hits1 != 0 {
		t.Fatalf("after batch=4: want 1 miss 0 hits, got misses=%d hits=%d", misses1, hits1)
	}

	// Dispatch 2: batch=32 — same SINK (interned), same structural hash → cache HIT.
	_, err32 := runBatch(32)
	hits2, misses2 := schedule.ScheduleCacheStats()
	if misses2 != 1 || hits2 != 1 {
		t.Fatalf("after batch=32: want 1 miss 1 hit, got misses=%d hits=%d", misses2, hits2)
	}

	t.Logf("=== PROOF 2: symbolic one-schedule-two-bindings ===")
	t.Logf("batch=4  (miss): max abs error vs CPU = %.2e  (want < 1e-5)", err4)
	t.Logf("batch=32 (hit):  max abs error vs CPU = %.2e  (want < 1e-5)", err32)
	t.Logf("cache: %d miss, %d hit — single schedule entry served both batch sizes", misses2, hits2)
	t.Logf("first 4 values (batch=4): %v", got4[:4])

	if err4 > 1e-5 {
		t.Errorf("FAIL: batch=4 max error %.2e > 1e-5", err4)
	}
	if err32 > 1e-5 {
		t.Errorf("FAIL: batch=32 max error %.2e > 1e-5", err32)
	}
}

// ── Proof 3: training-loop hit count ─────────────────────────────────────────

// TestScheduleCache_TrainingLoopHits simulates N forward passes over the same
// computation graph (leaf data changes each step via SetData, graph structure
// does not).  After N steps the cache must show exactly 1 miss and N-1 hits.
//
// This is the primary performance proof: in a real training loop, re-scheduling
// is skipped on every step after the first.
func TestScheduleCache_TrainingLoopHits(t *testing.T) {
	dev := requireDevice(t)
	schedule.ResetScheduleCache()

	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const device = "webgpu"
	const N = 5

	a := uop.NewArena(1024)

	// A two-kernel graph (reduce then elemwise) so the schedule is non-trivial.
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, device)
	y := x.Sum([]int{1}, false) // [4]
	z := y.Exp2()               // [4]

	var lastOut []float32
	for step := 0; step < N; step++ {
		// Vary input data each step so we're actually computing different values.
		data := make([]float32, 32)
		for i := range data {
			data[i] = float32(step*32+i) * 0.01
		}
		x.SetData(data)

		if err := tensor.Realize(z); err != nil {
			t.Fatalf("step %d: Realize: %v", step, err)
		}
		lastOut = z.Data()

		hits, misses := schedule.ScheduleCacheStats()
		wantHits := int64(step)   // step 0 → 0 hits; step 1 → 1 hit; ...
		wantMisses := int64(1)    // always exactly 1 miss (step 0)
		if step == 0 {
			wantHits = 0
			wantMisses = 1
		}
		if misses != wantMisses || hits != wantHits {
			t.Errorf("step %d: want %d miss %d hits, got misses=%d hits=%d",
				step, wantMisses, wantHits, misses, hits)
		}
	}

	hits, misses := schedule.ScheduleCacheStats()
	t.Logf("=== PROOF 3: training-loop hit count ===")
	t.Logf("%d steps: %d miss + %d hits  (want 1 miss + %d hits)", N, misses, hits, N-1)
	t.Logf("last step output: %v", lastOut)

	if misses != 1 {
		t.Errorf("FAIL: want exactly 1 miss, got %d", misses)
	}
	if hits != N-1 {
		t.Errorf("FAIL: want %d hits, got %d", N-1, hits)
	}
}
