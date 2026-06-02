package codegen_test

// Gather forward lowering: WGSL render snapshots and the tiled-reduce
// containment-scan assertion (Slice C).
//
// Snapshot tests only verify structural invariants: an i32 binding for the
// index buffer, an indirect load of that binding inside the kernel body, and
// the standard render-result shape. The Realize-side correctness gates (all
// five forward fixtures with max-abs-diff 0) live in
// /Users/george/Code/anneal/tensor/gather_realize_test.go because they need
// a live GPU.
//
// The tile-skip assertion constructs a Gather(W, idx).Matmul(other) graph
// and confirms that ApplyOpt(OptTile, ...) returns the input sink unchanged,
// proving the containment scan in opt.go closes the perf-trap design §2
// flagged for indirect-indexed matmuls.

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// gatherFirstItem realizes the schedule for a single output and returns the
// first ExecItem (the gather kernel). Fails if no items materialise.
func gatherFirstItem(t *testing.T, sink uop.UOp) schedule.ExecItem {
	t.Helper()
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("CreateSchedule produced 0 items")
	}
	return items[0]
}

// ── Snapshot: forward Gather emits an i32 index binding and an indirect load.

func TestGatherForward_Snapshot_Rank2Axis0(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "webgpu")
	out := w.Gather(0, idx)

	item := gatherFirstItem(t, makeSink(a, out))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)
	// One of the two read-only bindings must be array<i32> (the index buffer).
	if !strings.Contains(wgsl, "var<storage, read> ") || !strings.Contains(wgsl, ": array<i32>") {
		t.Errorf("expected an i32 read-only binding for the index buffer\n%s", wgsl)
	}
	// The kernel must reference both input bindings: data1 and data2.
	assertContains(t, wgsl, "data1[", "data2[")
}

func TestGatherForward_Snapshot_GPT2EmbeddingShape(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{50257, 768}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Int32, "webgpu")
	out := w.Gather(0, idx)

	item := gatherFirstItem(t, makeSink(a, out))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)
	assertContains(t, wgsl, ": array<i32>")
}

func TestGatherForward_Snapshot_Axis1MultiDimIndex(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{6, 6}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{5, 2}, uop.Dtypes.Int32, "webgpu")
	out := w.Gather(1, idx)

	item := gatherFirstItem(t, makeSink(a, out))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)
	assertContains(t, wgsl, ": array<i32>")
}

// ── Tile-skip assertion (carried question 3) ─────────────────────────────────

// TestGatherForward_TileSkipsIndirectMatmul builds a `Gather(W, idx).Matmul(B)`
// kernel and verifies ApplyOpt(OptTile, ...) on its compiled sink returns the
// input unchanged. The containment scan in codegen/opt.go must reject tiling
// for indirect-indexed matmul bodies; this test pins the behaviour against
// regressions.
func TestGatherForward_TileSkipsIndirectMatmul(t *testing.T) {
	a := newArena()
	// W: [V=8, K=4]; gather rows → [B=3, K=4]; matmul against B: [K=4, N=5] → [B=3, N=5].
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "webgpu")
	other := tensor.NewLeaf(a, []int64{4, 5}, uop.Dtypes.Float32, "webgpu")

	gathered := w.Gather(0, idx)  // [3, 4]
	out := gathered.Matmul(other) // [3, 5]

	items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items")
	}

	// Locate the matmul kernel: it must have a non-zero reduce body.
	var mmIdx = -1
	for i, item := range items {
		// rough heuristic: matmul kernels have a Reduce somewhere in the body.
		if strings.Contains(codegen.RenderWGSL(item).WGSL, "for (") {
			mmIdx = i
			break
		}
	}
	if mmIdx < 0 {
		t.Fatal("no reduce-containing kernel found in schedule; gather-matmul setup is wrong")
	}

	mm := items[mmIdx]

	// Try every (axis=0, TS) candidate; the tiled rewrite must be a no-op.
	tiledSink := codegen.ApplyOpt(mm.Ast, codegen.Opt{Kind: codegen.OptTile, Axis: 0, Arg: 8})
	if tiledSink != mm.Ast {
		t.Fatalf("ApplyOpt(OptTile) produced a different sink for an indirect-indexed matmul; containment scan failed")
	}

	// Defence-in-depth: render the kernel and confirm no tile-specific WGSL
	// patterns slipped in via some other path.
	wgsl := codegen.RenderWGSL(mm).WGSL
	for _, bad := range []string{"var<workgroup>", "loadTile", "workgroupBarrier"} {
		if strings.Contains(wgsl, bad) {
			t.Errorf("indirect-indexed matmul WGSL contains tile-specific pattern %q\n%s", bad, wgsl)
		}
	}
}
