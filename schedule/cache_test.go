package schedule_test

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestScheduleCache_HitMiss verifies that scheduling the same logical graph twice
// produces one miss followed by one hit, and that the returned items slice is
// pointer-identical (same backing array) on the hit.
func TestScheduleCache_HitMiss(t *testing.T) {
	schedule.ResetScheduleCache()

	a := uop.NewArena(512)
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0}, false)
	z := y.Exp2()
	sink := makeSink(a, z)

	items1 := schedule.CreateSchedule(sink, "cpu")
	hits, misses := schedule.ScheduleCacheStats()
	if misses != 1 || hits != 0 {
		t.Fatalf("after first call: want 1 miss 0 hits, got misses=%d hits=%d", misses, hits)
	}

	items2 := schedule.CreateSchedule(sink, "cpu")
	hits, misses = schedule.ScheduleCacheStats()
	if misses != 1 || hits != 1 {
		t.Fatalf("after second call: want 1 miss 1 hit, got misses=%d hits=%d", misses, hits)
	}

	if len(items1) != len(items2) {
		t.Fatalf("cached schedule length %d != original %d", len(items2), len(items1))
	}
	// The returned slice must be pointer-identical (same backing array, not a copy)
	// EXCEPT when WGSLRenderFunc is set (which forces a copy to release the arena).
	if schedule.WGSLRenderFunc == nil && len(items1) > 0 && &items1[0] != &items2[0] {
		t.Error("cached items slice is a copy, not the same backing array")
	}
}

// TestScheduleCache_DifferentDevice verifies that the same graph with different
// device strings produces two separate cache entries (two misses, no hits).
func TestScheduleCache_DifferentDevice(t *testing.T) {
	schedule.ResetScheduleCache()

	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := x.Exp2()
	sink := makeSink(a, y)

	schedule.CreateSchedule(sink, "cpu")
	schedule.CreateSchedule(sink, "webgpu")

	hits, misses := schedule.ScheduleCacheStats()
	if hits != 0 || misses != 2 {
		t.Fatalf("want 2 misses 0 hits, got misses=%d hits=%d", misses, hits)
	}

	// A third call with "cpu" must hit.
	schedule.CreateSchedule(sink, "cpu")
	hits, misses = schedule.ScheduleCacheStats()
	if hits != 1 || misses != 2 {
		t.Fatalf("want 1 hit 2 misses, got misses=%d hits=%d", misses, hits)
	}
}

// TestScheduleCache_CrossArenaIsolation verifies that two arenas with structurally
// identical graphs never share a cache entry.  If they did, the second arena would
// receive ExecItems whose Buffer.UOpIdx values reference the first arena's nodes —
// a silent wrong-buffer corruption.
func TestScheduleCache_CrossArenaIsolation(t *testing.T) {
	schedule.ResetScheduleCache()

	buildSink := func() (uop.UOp, *uop.Arena) {
		a := uop.NewArena(256)
		x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
		y := x.Exp2()
		sink := makeSink(a, y)
		return sink, a
	}

	sink1, a1 := buildSink()
	sink2, a2 := buildSink()

	items1 := schedule.CreateSchedule(sink1, "cpu")
	items2 := schedule.CreateSchedule(sink2, "cpu")

	hits, misses := schedule.ScheduleCacheStats()
	if hits != 0 || misses != 2 {
		t.Fatalf("two distinct arenas: want 2 misses 0 hits, got misses=%d hits=%d", misses, hits)
	}

	// Each item set's buffer indices must belong to the correct arena.
	for i, item := range items1 {
		for j, buf := range item.Bufs {
			node := a1.At(buf.UOpIdx)
			if !node.Valid() {
				t.Errorf("items1[%d].Bufs[%d].UOpIdx=%d is invalid in arena1", i, j, buf.UOpIdx)
			}
		}
	}
	for i, item := range items2 {
		for j, buf := range item.Bufs {
			node := a2.At(buf.UOpIdx)
			if !node.Valid() {
				t.Errorf("items2[%d].Bufs[%d].UOpIdx=%d is invalid in arena2", i, j, buf.UOpIdx)
			}
		}
	}

	_ = a1
	_ = a2
}

// TestScheduleCache_StructuralKey verifies that the cache key is purely structural:
// two arenas with identical computation graphs but different construction orders
// produce the same schedule structure (kernel count, NumParams per kernel), and
// each independently misses then hits on a second call within its own arena.
//
// This is complementary to TestCreateSchedule_DeterminismBuildOrder which already
// tests cross-arena structural equality of schedules; here we only test that the
// cache correctly classifies same-arena repeated calls as hits.
func TestScheduleCache_SameArenaTwiceHits(t *testing.T) {
	schedule.ResetScheduleCache()

	a := uop.NewArena(512)
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0}, false)
	z := y.Reshape([]int64{1}).Expand([]int64{8}).Exp2()
	w := z.Sum([]int{0}, false)
	sink := makeSink(a, w)

	// First call: miss.
	items1 := schedule.CreateSchedule(sink, "cpu")
	hits, misses := schedule.ScheduleCacheStats()
	if misses != 1 || hits != 0 {
		t.Fatalf("first call: want 1 miss, got misses=%d hits=%d", misses, hits)
	}

	// Five more calls: all hits.
	for i := 0; i < 5; i++ {
		schedule.CreateSchedule(sink, "cpu")
	}
	hits, misses = schedule.ScheduleCacheStats()
	if misses != 1 || hits != 5 {
		t.Fatalf("after 6 total calls: want 1 miss 5 hits, got misses=%d hits=%d", misses, hits)
	}
	_ = items1
}

// TestScheduleCache_DeterminismUnchanged ensures the existing determinism test
// continues to pass after adding the cache.  Two independent arenas with identical
// graphs produce schedules with the same kernel count and NumParams.
func TestScheduleCache_DeterminismUnchanged(t *testing.T) {
	schedule.ResetScheduleCache()

	buildGraph := func() (uop.UOp, *uop.Arena) {
		a := uop.NewArena(1024)
		x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
		y := x.Sum([]int{0}, false)
		z := y.Exp2()
		srcs := []uop.UOp{z.Node()}
		sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
		return sink, a
	}

	sink1, _ := buildGraph()
	sink2, _ := buildGraph()

	items1 := schedule.CreateSchedule(sink1, "cpu")
	items2 := schedule.CreateSchedule(sink2, "cpu")

	// Both calls are misses (different arenas).
	hits, misses := schedule.ScheduleCacheStats()
	if hits != 0 || misses != 2 {
		t.Fatalf("want 2 misses 0 hits for distinct arenas, got misses=%d hits=%d", misses, hits)
	}

	if len(items1) != len(items2) {
		t.Fatalf("schedule lengths differ: %d vs %d", len(items1), len(items2))
	}
	for i := range items1 {
		ki1 := items1[i].Ast.Arg().(uop.KernelInfo)
		ki2 := items2[i].Ast.Arg().(uop.KernelInfo)
		if ki1.NumParams != ki2.NumParams {
			t.Errorf("item[%d]: NumParams %d vs %d", i, ki1.NumParams, ki2.NumParams)
		}
	}
}
