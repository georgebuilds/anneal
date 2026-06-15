package codegen_test

import (
	"os"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// reduceItem builds a single-kernel reduce schedule item whose ActionSpace is
// non-empty (so a real, applicable Opt is available for cache-injection tests).
func reduceItem(t *testing.T) schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{32, 32}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{1}, false)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}
	return items[0]
}

// TestBeamApplyToItemsValidCacheHit injects a cached non-identity opt with the
// CORRECT WGSL hash; BeamApplyToItems must apply it and pre-fill the rendered
// WGSL so the executor skips re-render.
func TestBeamApplyToItemsValidCacheHit(t *testing.T) {
	_ = os.Unsetenv("ANNEAL_BEAM")
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	item := reduceItem(t)
	acts := codegen.ActionSpace(item.Ast)
	if len(acts) == 0 {
		t.Skip("no actions for kernel")
	}
	opt := acts[0]

	// Compute the hash exactly as the apply path will.
	opted := codegen.ApplyOpts(item, []codegen.Opt{opt})
	rendered := codegen.RenderWGSL(opted)
	wantHash := codegen.BeamWGSLHash(rendered.WGSL)

	sk := codegen.KernelSK(item)
	codegen.BeamDiskCacheInject(sk, []codegen.Opt{opt}, wantHash)

	got := codegen.BeamApplyToItems([]schedule.ExecItem{item}, nil, nil)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if got[0].WGSL == "" {
		t.Error("valid cache hit should pre-fill WGSL")
	}
	// Raw WGSL carries arena-index-dependent var names, so compare the
	// index-independent normalized hash instead of the literal string.
	if codegen.BeamWGSLHash(got[0].WGSL) != wantHash {
		t.Error("pre-filled WGSL hash must match the rendered opted kernel")
	}
}

// TestBeamApplyToItemsSKCollisionGuard injects a non-identity opt with the
// WRONG WGSL hash; the value-identity guard must detect the mismatch and fall
// back to the original (un-opted) item.
func TestBeamApplyToItemsSKCollisionGuard(t *testing.T) {
	_ = os.Unsetenv("ANNEAL_BEAM")
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	item := reduceItem(t)
	acts := codegen.ActionSpace(item.Ast)
	if len(acts) == 0 {
		t.Skip("no actions for kernel")
	}

	sk := codegen.KernelSK(item)
	codegen.BeamDiskCacheInject(sk, []codegen.Opt{acts[0]}, "0000000000000000")

	got := codegen.BeamApplyToItems([]schedule.ExecItem{item}, nil, nil)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	// Guard fallback: result[i] is restored to the original input item, so its
	// Ast and (pre-existing) WGSL must match the unmodified item exactly.
	if got[0].Ast.Index() != item.Ast.Index() {
		t.Error("fallback item must equal the original input item")
	}
	if got[0].WGSL != item.WGSL {
		t.Error("fallback must restore the original item's WGSL, not the opted render")
	}
}

// TestBeamApplyToItemsSearchModeStores runs the apply path in search mode with
// stub exec/bench so a fresh search runs, persists, and applies its winner.
func TestBeamApplyToItemsSearchModeStores(t *testing.T) {
	t.Setenv("ANNEAL_BEAM", "1")
	codegen.BeamCacheReset()
	codegen.BeamDiskCacheReset()
	defer func() {
		codegen.BeamCacheReset()
		codegen.BeamDiskCacheReset()
	}()

	item := reduceItem(t)

	exec := &stubExec{}
	bench := &stubBench{}
	got := codegen.BeamApplyToItems([]schedule.ExecItem{item}, exec, bench)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	// Search mode must have invoked the executor at least once (baseline + cands).
	if exec.runCount.Load() == 0 {
		t.Error("search mode did not run the executor")
	}
}
