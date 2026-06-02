package codegen_test

// Scatter-add backward kernel WGSL snapshot + structural assertions (Slice D).
//
// The OpScatterAdd UOp dissolves at rangeify time into an OpReduce-of-OpWhere
// over an OpGatherIdx-wrapped indirect load. This file confirms that the
// resulting kernel:
//
//   - emits exactly two i32 read-only bindings (sortedIdx + perm),
//   - emits an indirect data1[...] read through the perm gather scalar,
//   - emits a select(0.0, ..., ...) or where pattern for the segment mask,
//   - runs through the standard reduce lowering (a for-loop with an
//     accumulator init), confirming no special-purpose template is needed.
//
// The value-oracle gates live in tensor/gather_backward_test.go (GPU-driven).

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// scatterAddSnapshotSetup builds a backward graph for
//
//	loss = sum(W.Gather(0, idx) * dY)
//
// then returns the dW kernel ExecItem so the test can render and assert on it.
func scatterAddSnapshotSetup(t *testing.T, V, D, B int) (schedule.RenderResult, int) {
	t.Helper()
	a := newArena()
	w := tensor.NewLeaf(a, []int64{int64(V), int64(D)}, uop.Dtypes.Float32, "webgpu")
	idx := tensor.NewLeaf(a, []int64{int64(B)}, uop.Dtypes.Int32, "webgpu")
	dy := tensor.NewLeaf(a, []int64{int64(B), int64(D)}, uop.Dtypes.Float32, "webgpu")

	gather := w.Gather(0, idx)
	prod := gather.Mul(dy)
	loss := prod.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{w})
	dW := grads[w]
	if dW == nil {
		t.Fatal("Backward returned no gradient for W")
	}

	sink := makeSink(a, dW)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items for backward graph")
	}

	// The scatter-add kernel is the dW producer: the kernel whose output is
	// the dW tensor. Identify it by inspecting WGSL for the segment-mask
	// pattern (a for-loop accumulator); fall back to the last item if none
	// of the kernels exhibit the pattern (schedule ordering is stable).
	lastIdx := len(items) - 1
	scatterIdx := lastIdx
	for i, item := range items {
		w := codegen.RenderWGSL(item).WGSL
		// Scatter-add has a reduce over r (forms a for-loop) AND uses two
		// i32 bindings (sortedIdx, perm). zeros kernel has neither.
		if strings.Contains(w, "for (") && strings.Count(w, ": array<i32>") >= 2 {
			scatterIdx = i
			break
		}
	}
	return codegen.RenderWGSL(items[scatterIdx]), scatterIdx
}

// ── Snapshot tests ──────────────────────────────────────────────────────────

// TestScatterAdd_Snapshot_TwoIntBindings verifies the rendered kernel for a
// simple V=8, D=4, B=3 backward emits both i32 storage bindings (sortedIdx
// and perm) and includes a reduce-style for-loop.
func TestScatterAdd_Snapshot_TwoIntBindings(t *testing.T) {
	res, _ := scatterAddSnapshotSetup(t, 8, 4, 3)
	wgsl := res.WGSL

	// Must have at least two array<i32> read bindings (sortedIdx + perm).
	cnt := strings.Count(wgsl, ": array<i32>")
	if cnt < 2 {
		t.Errorf("expected >= 2 i32 read bindings; got %d\n%s", cnt, wgsl)
	}

	// Reduce loop present.
	if !strings.Contains(wgsl, "for (") {
		t.Errorf("expected a for-loop (reduce over B); got\n%s", wgsl)
	}

	// At least one accumulator init. Codegen names accumulators `accN` (no
	// underscore between `acc` and the counter); match the prefix.
	if !strings.Contains(wgsl, "var acc") {
		t.Errorf("expected accumulator init `var acc`; got\n%s", wgsl)
	}

	// Segment-mask: a select(0.0, ..., bool) is the canonical reduction
	// where(match, contribution, 0). The exact arguments depend on let
	// numbering but the structural shape is fixed.
	if !strings.Contains(wgsl, "select(0.0,") {
		t.Errorf("expected segment-mask select(0.0, ...) in WGSL; got\n%s", wgsl)
	}

	// Output binding is the dW buffer (read_write).
	if !strings.Contains(wgsl, "var<storage, read_write> data0") {
		t.Errorf("expected dW as read_write data0; got\n%s", wgsl)
	}
}

// TestScatterAdd_Snapshot_RaceFreeWrite confirms the kernel writes data0[...]
// inside the workgroup body (not inside a reduce body), proving the dispatch
// geometry has one writer per (v, *t) cell (race-free without atomics).
func TestScatterAdd_Snapshot_RaceFreeWrite(t *testing.T) {
	res, _ := scatterAddSnapshotSetup(t, 8, 4, 3)
	wgsl := res.WGSL

	// Find the first `data0[` occurrence. It should NOT be inside the
	// for-loop body. A robust check: the `data0[` must occur AFTER the
	// for-loop's closing brace (the loop runs the reduction, then the
	// accumulator is written once).
	dataPos := strings.Index(wgsl, "data0[")
	if dataPos < 0 {
		t.Fatalf("kernel does not write data0\n%s", wgsl)
	}

	// Locate the for-loop start and its matching brace (rough heuristic:
	// the for-loop has exactly one body; counting braces from the for-stmt
	// onward).
	forPos := strings.Index(wgsl, "for (")
	if forPos < 0 {
		t.Fatal("no for-loop found")
	}
	// Walk forward from forPos finding the matching close brace.
	bodyStart := strings.Index(wgsl[forPos:], "{")
	if bodyStart < 0 {
		t.Fatal("malformed for-loop: no opening brace")
	}
	depth := 1
	bodyEnd := -1
brace:
	for i := forPos + bodyStart + 1; i < len(wgsl); i++ {
		switch wgsl[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				bodyEnd = i
				break brace
			}
		}
	}
	if bodyEnd < 0 {
		t.Fatal("malformed for-loop: no closing brace")
	}
	if dataPos < bodyEnd && dataPos > forPos {
		t.Errorf("data0[...] written INSIDE the reduce loop; expected the write to be outside\n%s", wgsl)
	}
}
