package codegen_test

import (
	"os"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// tiledMatmulItem builds a 64×64 matmul kernel ExecItem suitable for the
// OptTile / OptUpcast / OptVectorize codegen path. Returns the first schedule
// item (the matmul reduce kernel).
func tiledMatmulItem(t *testing.T) schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(1 << 16)
	A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items for matmul")
	}
	return items[0]
}

// ── emitTiledReduce: OptTile render ──────────────────────────────────────────

// TestRender_TiledMatmul renders a tiled matmul to WGSL, exercising
// emitTiledReduce and the tiled branches of lowerSink / emitExpr that the
// existing tests only reach via ApplyOpts (no render).
func TestRender_TiledMatmul(t *testing.T) {
	item := tiledMatmulItem(t)
	out := codegen.ApplyOpts(item, []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptTile, Axis: 0, Arg: 16},
	})
	wgsl := codegen.RenderWGSL(out).WGSL
	if wgsl == "" {
		t.Fatal("tiled matmul rendered empty WGSL")
	}
	// Tiling introduces workgroup-shared memory and a barrier.
	assertContains(t, wgsl, "var<workgroup>", "workgroupBarrier()", "@compute")
	// Braces must balance.
	if strings.Count(wgsl, "{") != strings.Count(wgsl, "}") {
		t.Errorf("unbalanced braces in tiled WGSL")
	}
}

// TestRender_TiledUpcastMatmul renders a tiled + register-blocked (OptUpcast)
// matmul, exercising the MR/NR expansion branches of emitTiledReduce and the
// multi-store fan-out in lowerSink.
func TestRender_TiledUpcastMatmul(t *testing.T) {
	item := tiledMatmulItem(t)
	out := codegen.ApplyOpts(item, []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptTile, Axis: 0, Arg: 16},
		{Kind: codegen.OptUpcast, Axis: 0, Arg: 4},
	})
	wgsl := codegen.RenderWGSL(out).WGSL
	if wgsl == "" {
		t.Fatal("tiled+upcast matmul rendered empty WGSL")
	}
	assertContains(t, wgsl, "var<workgroup>", "workgroupBarrier()")
	if strings.Count(wgsl, "{") != strings.Count(wgsl, "}") {
		t.Errorf("unbalanced braces in tiled+upcast WGSL")
	}
}

// TestRender_TiledUpcastVectorizeMatmul renders the full
// Tile→Upcast×2→Vectorize pipeline, hitting the vec4 store fan-out and the
// AxisVectorize branches in emitTiledReduce.
func TestRender_TiledUpcastVectorizeMatmul(t *testing.T) {
	item := tiledMatmulItem(t)
	out := codegen.ApplyOpts(item, []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptTile, Axis: 0, Arg: 16},
		{Kind: codegen.OptUpcast, Axis: 0, Arg: 4},
		{Kind: codegen.OptUpcast, Axis: 1, Arg: 4},
		{Kind: codegen.OptVectorize, Axis: 0, Arg: 4},
	})
	wgsl := codegen.RenderWGSL(out).WGSL
	if wgsl == "" {
		t.Fatal("tiled+upcast+vectorize matmul rendered empty WGSL")
	}
	assertContains(t, wgsl, "var<workgroup>", "workgroupBarrier()")
	if strings.Count(wgsl, "{") != strings.Count(wgsl, "}") {
		t.Errorf("unbalanced braces in vectorized WGSL")
	}
}

// ── BeamApplyToItems: SK-collision guard (hit with wrong hash) ───────────────

// TestBeamApplyToItems_CollisionGuard injects a disk-cache hit whose stored
// WGSL hash does not match what the opted kernel actually renders to, forcing
// the value-identity guard to fire and fall back to the original item.
func TestBeamApplyToItems_CollisionGuard(t *testing.T) {
	_ = os.Unsetenv("ANNEAL_BEAM")
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	item := tiledMatmulItem(t)
	items := []schedule.ExecItem{item}
	sk := codegen.KernelSK(item)

	// Inject non-empty opts with a deliberately wrong WGSL hash.
	codegen.BeamDiskCacheInject(sk, []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptTile, Axis: 0, Arg: 16},
	}, "0000000000000000")

	got := codegen.BeamApplyToItems(items, nil, nil)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	// Guard fired → original item returned unchanged.
	if got[0].Ast.Index() != items[0].Ast.Index() {
		t.Error("collision guard should fall back to the original item")
	}
}

// TestBeamApplyToItems_HitWithMatchingHash injects a hit whose stored hash
// matches the rendered opted kernel, so the opts are applied and WGSL is
// pre-filled (the cache-hit success branch).
func TestBeamApplyToItems_HitWithMatchingHash(t *testing.T) {
	_ = os.Unsetenv("ANNEAL_BEAM")
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	item := tiledMatmulItem(t)
	items := []schedule.ExecItem{item}
	sk := codegen.KernelSK(item)

	opts := []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptTile, Axis: 0, Arg: 16},
	}
	// Compute the true hash the same way BeamApplyToItems does.
	opted := codegen.ApplyOpts(item, opts)
	opted.WGSL = ""
	rendered := codegen.RenderWGSL(opted)
	hash := codegen.BeamWGSLHash(rendered.WGSL)

	codegen.BeamDiskCacheInject(sk, opts, hash)

	got := codegen.BeamApplyToItems(items, nil, nil)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if got[0].WGSL == "" {
		t.Error("matching-hash hit should pre-fill WGSL")
	}
	if got[0].Ast.Index() == items[0].Ast.Index() {
		t.Error("matching-hash hit should swap in the opted Ast")
	}
}
