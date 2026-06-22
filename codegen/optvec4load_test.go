package codegen_test

// OptVec4Load - apply-time eligibility, tag encoding, renderer bindings, and
// BEAM action-space integration. GPU value oracles live in
// backend/webgpu/vec4load_test.go.

import (
	"fmt"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// tileOpts is the canonical OptLocal×2 + OptTile prologue (B2 path).
func tileOpts(TS int) []codegen.Opt {
	return []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptTile, Axis: 0, Arg: TS},
	}
}

func applySeq(t *testing.T, item schedule.ExecItem, opts []codegen.Opt) uop.UOp {
	t.Helper()
	sink := item.Ast
	for _, o := range opts {
		sink = codegen.ApplyOptBufs(sink, o, item.Bufs)
	}
	return sink
}

// TestOptVec4Load_TagEncodingAndIdempotence pins the encoding: applying
// OptVec4Load after OptTile re-tags the reduce "tile:<TS>:vec4" (a NEW
// interned node - the no-op filter must see a real transform), and a second
// application refuses (idempotent).
func TestOptVec4Load_TagEncodingAndIdempotence(t *testing.T) {
	item := matmulItem(t, 64, 64, 64)
	tiled := applySeq(t, item, tileOpts(16))

	v1 := codegen.ApplyOptBufs(tiled, codegen.Opt{Kind: codegen.OptVec4Load}, item.Bufs)
	if v1.Index() == tiled.Index() {
		t.Fatal("OptVec4Load refused on an eligible tiled 64³ matmul")
	}
	// The reduce tag must now carry the :vec4 suffix.
	foundTag := false
	for _, u := range uop.TopoSort(v1) {
		if u.Op() != uop.OpReduce {
			continue
		}
		if s, ok := u.Tag().(string); ok && s == "tile:16:vec4" {
			foundTag = true
		}
	}
	if !foundTag {
		t.Error("opted kernel has no OpReduce tagged \"tile:16:vec4\"")
	}
	// Composability gate intact: KernelHasTiledReduce still sees a tile tag.
	if !codegen.KernelHasTiledReduce(v1) {
		t.Error("KernelHasTiledReduce false after OptVec4Load - OptUpcast/OptVectorize composition broken")
	}
	// Idempotence: second application refuses.
	if v2 := codegen.ApplyOptBufs(v1, codegen.Opt{Kind: codegen.OptVec4Load}, item.Bufs); v2.Index() != v1.Index() {
		t.Error("second OptVec4Load application transformed the sink - must refuse (idempotent)")
	}
}

// TestOptVec4Load_Refusals pins every refusal edge: the opt must return the
// sink unchanged (applyTile inapplicability convention) so ActionSpace's
// no-op filter excludes it.
func TestOptVec4Load_Refusals(t *testing.T) {
	vec4 := codegen.Opt{Kind: codegen.OptVec4Load}

	t.Run("without_OptTile", func(t *testing.T) {
		item := matmulItem(t, 64, 64, 64)
		if got := codegen.ApplyOptBufs(item.Ast, vec4, item.Bufs); got.Index() != item.Ast.Index() {
			t.Error("OptVec4Load applied without OptTile - must refuse (compose-after contract)")
		}
	})

	t.Run("K_non_div_4", func(t *testing.T) {
		// K=17: A's stride-1 extent 17 % 4 != 0 → whole opt refuses
		// (both-or-nothing v1), even though B's extent (N=64) is aligned.
		item := matmulItem(t, 64, 17, 64)
		tiled := applySeq(t, item, tileOpts(16))
		if got := codegen.ApplyOptBufs(tiled, vec4, item.Bufs); got.Index() != tiled.Index() {
			t.Error("OptVec4Load applied with K=17 - A tile loads would need partial vec4 slots")
		}
	})

	t.Run("N_non_div_4", func(t *testing.T) {
		item := matmulItem(t, 64, 64, 30)
		tiled := applySeq(t, item, tileOpts(16))
		if got := codegen.ApplyOptBufs(tiled, vec4, item.Bufs); got.Index() != tiled.Index() {
			t.Error("OptVec4Load applied with N=30 - B tile loads would need partial vec4 slots")
		}
	})

	t.Run("non_tilable_kernel", func(t *testing.T) {
		item := elemwiseItem(t) // [6,6] add - no Mul(Index,Index) reduce
		if got := codegen.ApplyOptBufs(item.Ast, vec4, item.Bufs); got.Index() != item.Ast.Index() {
			t.Error("OptVec4Load applied to a non-tilable elementwise kernel")
		}
	})

	t.Run("symbolic_kernel", func(t *testing.T) {
		// Symbolic batch matmul: [n,64] @ [64,64]. The A param carries a
		// symbolic dim (Shape[0]==0 sentinel) and the kernel has symbolic
		// ranges - both independently force refusal.
		a := uop.NewArena(1 << 16)
		A := tensor.NewSymbolicBatchInput(a, "n", 1, 64, []int64{64}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
		items := schedule.CreateSchedule(sink, "webgpu")
		if len(items) == 0 {
			t.Fatal("symbolic schedule produced 0 items")
		}
		item := items[0]
		// OptTile refuses on the symbolic reduce? K=64 here is concrete, so
		// tiling may apply; OptVec4Load must refuse either way.
		tiled := applySeq(t, item, tileOpts(16))
		if got := codegen.ApplyOptBufs(tiled, vec4, item.Bufs); got.Index() != tiled.Index() {
			t.Error("OptVec4Load applied to a symbolic-batch kernel")
		}
	})

	t.Run("image_dtype_param", func(t *testing.T) {
		// Doctor the buffer table: same tiled matmul AST, but param 1 claims
		// image storage. Image dtypes already bind array<vec4<f32>> with the
		// lane-select load path - OptVec4Load must not stack on top.
		item := matmulItem(t, 64, 64, 64)
		tiled := applySeq(t, item, tileOpts(16))
		bufs := append([]schedule.Buffer{}, item.Bufs...)
		bufs[1].DType = uop.Dtypes.ImageFloat32
		if got := codegen.ApplyOptBufs(tiled, vec4, bufs); got.Index() != tiled.Index() {
			t.Error("OptVec4Load applied to an image-dtype param")
		}
	})

	t.Run("non_f32_param", func(t *testing.T) {
		item := matmulItem(t, 64, 64, 64)
		tiled := applySeq(t, item, tileOpts(16))
		bufs := append([]schedule.Buffer{}, item.Bufs...)
		bufs[2].DType = uop.Dtypes.Float16
		if got := codegen.ApplyOptBufs(tiled, vec4, bufs); got.Index() != tiled.Index() {
			t.Error("OptVec4Load applied to an f16 param")
		}
	})

	t.Run("nil_bufs_via_ApplyOpt", func(t *testing.T) {
		item := matmulItem(t, 64, 64, 64)
		tiled := applySeq(t, item, tileOpts(16))
		if got := codegen.ApplyOpt(tiled, vec4); got.Index() != tiled.Index() {
			t.Error("bare ApplyOpt (no buffer table) applied OptVec4Load - shape eligibility cannot be proven without Bufs")
		}
	})

	t.Run("TS_non_div_4", func(t *testing.T) {
		// TS must be a multiple of 4 for aligned vec4 column bases. Use a
		// hand-tiled TS that is not: OptTile with TS=6 on a 24-extent K.
		item := matmulItem(t, 64, 24, 64)
		tiled := applySeq(t, item, []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: 8},
			{Kind: codegen.OptLocal, Axis: 0, Arg: 8},
			{Kind: codegen.OptTile, Axis: 0, Arg: 6},
		})
		if got := codegen.ApplyOptBufs(tiled, vec4, item.Bufs); got.Index() != tiled.Index() {
			t.Error("OptVec4Load applied with TS=6 - tile column bases would be misaligned")
		}
	})
}

// TestOptVec4Load_WGSLStructure checks the rendered shader: converted input
// params bind array<vec4<f32>>, the output stays array<f32>, the tile fills
// load whole vec4 slots, and the un-opted render of the same kernel carries
// no vec4 f32 bindings.
func TestOptVec4Load_WGSLStructure(t *testing.T) {
	stacks := []struct {
		name string
		opts []codegen.Opt
		vReg string // expected vec4 register prefix in the fill
	}{
		{"tile_only_B2", append(tileOpts(16), codegen.Opt{Kind: codegen.OptVec4Load}), "vA"},
		{"b3_upcast", append(append(tileOpts(16),
			codegen.Opt{Kind: codegen.OptUpcast, Axis: 0, Arg: 4},
			codegen.Opt{Kind: codegen.OptUpcast, Axis: 1, Arg: 4}),
			codegen.Opt{Kind: codegen.OptVec4Load}), "vA"},
		{"b37_vectorize", append(append(tileOpts(16),
			codegen.Opt{Kind: codegen.OptUpcast, Axis: 0, Arg: 4},
			codegen.Opt{Kind: codegen.OptUpcast, Axis: 1, Arg: 4},
			codegen.Opt{Kind: codegen.OptVectorize, Axis: 1, Arg: 4}),
			codegen.Opt{Kind: codegen.OptVec4Load}), "vA_0"},
	}

	for _, tc := range stacks {
		t.Run(tc.name, func(t *testing.T) {
			item := matmulItem(t, 64, 64, 64)
			identity := codegen.RenderWGSL(item).WGSL
			if strings.Contains(identity, "array<vec4<f32>>") {
				t.Fatal("identity f32 matmul already binds array<vec4<f32>> - test premise broken")
			}

			opted := codegen.ApplyOpts(item, tc.opts)
			w := codegen.RenderWGSL(opted).WGSL

			for _, want := range []string{
				"@group(0) @binding(1) var<storage, read> data1: array<vec4<f32>>;",
				"@group(0) @binding(2) var<storage, read> data2: array<vec4<f32>>;",
				"@group(0) @binding(0) var<storage, read_write> data0: array<f32>;",
				"let " + tc.vReg + ": vec4<f32> = select(vec4<f32>(0.0), data1[",
			} {
				if !strings.Contains(w, want) {
					t.Errorf("opted WGSL missing %q\n--- WGSL head ---\n%s", want, w[:min(len(w), 1200)])
				}
			}
			// No scalar loads from the converted params may remain: every
			// data1/data2 subscript must be a vec4 slot read into a vN register.
			for _, line := range strings.Split(w, "\n") {
				if strings.Contains(line, "data1[") || strings.Contains(line, "data2[") {
					if !strings.Contains(line, "vec4<f32>") {
						t.Errorf("scalar-style load from a vec4-bound param: %s", strings.TrimSpace(line))
					}
				}
			}
		})
	}
}

// TestOptVec4Load_RenderReproducible mirrors TestB37_ScheduleCache_HitCorrect:
// two independent builds render byte-identical WGSL, and the normalized beam
// hash is stable (persistent-disk-cache contract).
func TestOptVec4Load_RenderReproducible(t *testing.T) {
	opts := append(tileOpts(16), codegen.Opt{Kind: codegen.OptVec4Load})
	build := func() string {
		item := codegen.ApplyOpts(matmulItem(t, 64, 64, 64), opts)
		return codegen.RenderWGSL(item).WGSL
	}
	w1, w2 := build(), build()
	if w1 != w2 {
		t.Fatal("WGSL render not reproducible under OptVec4Load (cache key risk)")
	}
	if codegen.BeamWGSLHash(w1) != codegen.BeamWGSLHash(w2) {
		t.Fatal("normalized beam hash unstable under OptVec4Load")
	}
}

// retagTiledReduce rebuilds sink with its tiled reduce's tag replaced -
// bypassing applyVec4Load's eligibility gate to pin the lowerer's fail-loud
// backstops against hand-built tags.
func retagTiledReduce(t *testing.T, sink uop.UOp, newTag string) uop.UOp {
	t.Helper()
	arena := sink.Arena()
	end := sink.Src(0)
	store := end.Src(0)
	reduce := store.Src(1)
	if reduce.Op() != uop.OpReduce {
		t.Fatalf("retagTiledReduce: store.Src(1) is %s, want OpReduce", reduce.Op())
	}
	srcs := make([]uop.UOp, reduce.NSrc())
	for i := range srcs {
		srcs[i] = reduce.Src(i)
	}
	newReduce := arena.New(uop.OpReduce, reduce.DType(), srcs, reduce.Arg(), newTag)
	newStore := arena.New(uop.OpStore, store.DType(), []uop.UOp{store.Src(0), newReduce}, store.Arg(), store.Tag())
	newEndSrcs := make([]uop.UOp, end.NSrc())
	newEndSrcs[0] = newStore
	for i := 1; i < end.NSrc(); i++ {
		newEndSrcs[i] = end.Src(i)
	}
	newEnd := arena.New(uop.OpEnd, end.DType(), newEndSrcs, end.Arg(), end.Tag())
	return arena.New(uop.OpSink, sink.DType(), []uop.UOp{newEnd}, sink.Arg(), sink.Tag())
}

func expectRenderPanic(t *testing.T, item schedule.ExecItem, wantSubstr string) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected render panic containing %q, got none", wantSubstr)
		}
		msg := fmt.Sprint(r)
		if !strings.Contains(msg, wantSubstr) {
			t.Errorf("panic message missing %q\nfull message: %s", wantSubstr, msg)
		}
	}()
	codegen.RenderWGSL(item)
}

// TestOptVec4Load_LowererFailLoudBackstops pins the lowerer panics that
// guard against tags the eligibility gate never produces (hand-built or
// hand-composed sequences only; BEAM cannot reach these).
func TestOptVec4Load_LowererFailLoudBackstops(t *testing.T) {
	t.Run("unaligned_TS_tag", func(t *testing.T) {
		// TS=6 tile (legitimate) re-tagged ":vec4" (illegitimate - TS%4!=0).
		item := matmulItem(t, 64, 24, 64)
		tiled := applySeq(t, item, []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: 8},
			{Kind: codegen.OptLocal, Axis: 0, Arg: 8},
			{Kind: codegen.OptTile, Axis: 0, Arg: 6},
		})
		item.SetAst(retagTiledReduce(t, tiled, "tile:6:vec4"))
		expectRenderPanic(t, item, "unaligned extents")
	})

	t.Run("malformed_tile_tag", func(t *testing.T) {
		item := matmulItem(t, 64, 64, 64)
		tiled := applySeq(t, item, tileOpts(16))
		item.SetAst(retagTiledReduce(t, tiled, "tile:notanumber"))
		expectRenderPanic(t, item, "malformed tile tag")
	})

	t.Run("vectorize_width_2", func(t *testing.T) {
		// applyVec4Load doesn't gate on OptVectorize width (BEAM only
		// proposes W=4); the vecN lowering path refuses anything else.
		item := matmulItem(t, 64, 64, 64)
		opts := append(b3TestOpts(16, 4, 4),
			codegen.Opt{Kind: codegen.OptVectorize, Axis: 1, Arg: 2},
			codegen.Opt{Kind: codegen.OptVec4Load})
		expectRenderPanic(t, codegen.ApplyOpts(item, opts), "requires OptVectorize width 4")
	})

	t.Run("upcast_factor_8", func(t *testing.T) {
		// MR=8 needs 2 vec4 loads per thread per operand; the distributed
		// fill only supports ≤ 1 (upcast factors are ≤ 4 everywhere else).
		item := matmulItem(t, 256, 256, 256)
		opts := append(b3TestOpts(16, 8, 4), codegen.Opt{Kind: codegen.OptVec4Load})
		expectRenderPanic(t, codegen.ApplyOpts(item, opts), "distributed fill")
	})
}

// b3TestOpts mirrors backend/webgpu's b3Opts (OptLocal×2+OptTile+OptUpcast×2).
func b3TestOpts(TS, MR, NR int) []codegen.Opt {
	return append(tileOpts(TS),
		codegen.Opt{Kind: codegen.OptUpcast, Axis: 0, Arg: MR},
		codegen.Opt{Kind: codegen.OptUpcast, Axis: 1, Arg: NR})
}

// TestVec4LoadParams_Edges pins the nil/refusal branches of the renderer's
// derivation helper and the remaining buffer-eligibility edges.
func TestVec4LoadParams_Edges(t *testing.T) {
	if got := codegen.Vec4LoadParams(uop.UOp{}); got != nil {
		t.Errorf("Vec4LoadParams(invalid sink) = %v, want nil", got)
	}
	// Malformed/synthetic sinks must report nil, not panic (the structural
	// matcher's NSrc guards; renderer unit tests build SINKs like these).
	{
		a := uop.NewArena(64)
		emptyEnd := a.New(uop.OpEnd, uop.Dtypes.Void, nil, nil, nil)
		sinkEmptyEnd := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{emptyEnd}, nil, nil)
		if got := codegen.Vec4LoadParams(sinkEmptyEnd); got != nil {
			t.Errorf("Vec4LoadParams(END with no srcs) = %v, want nil", got)
		}
		dst := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
		store1 := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{dst}, nil, nil)
		end1 := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store1}, nil, nil)
		sinkStore1 := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end1}, nil, nil)
		if got := codegen.Vec4LoadParams(sinkStore1); got != nil {
			t.Errorf("Vec4LoadParams(STORE with 1 src) = %v, want nil", got)
		}
	}
	if got := codegen.Vec4LoadParams(elemwiseItem(t).Ast); got != nil {
		t.Errorf("Vec4LoadParams(non-tilable kernel) = %v, want nil", got)
	}
	item := matmulItem(t, 64, 64, 64)
	tiled := applySeq(t, item, tileOpts(16))
	if got := codegen.Vec4LoadParams(tiled); got != nil {
		t.Errorf("Vec4LoadParams(tiled, no vec4 tag) = %v, want nil", got)
	}
	v4 := codegen.ApplyOptBufs(tiled, codegen.Opt{Kind: codegen.OptVec4Load}, item.Bufs)
	if got := codegen.Vec4LoadParams(v4); len(got) != 2 || !got[1] || !got[2] {
		t.Errorf("Vec4LoadParams(vec4 kernel) = %v, want {1:true, 2:true}", got)
	}

	// Buffer-eligibility edges via doctored buffer tables.
	vec4 := codegen.Opt{Kind: codegen.OptVec4Load}
	refuse := func(name string, doctor func(b []schedule.Buffer) []schedule.Buffer) {
		t.Helper()
		bufs := doctor(append([]schedule.Buffer{}, item.Bufs...))
		if got := codegen.ApplyOptBufs(tiled, vec4, bufs); got.Index() != tiled.Index() {
			t.Errorf("%s: OptVec4Load applied, want refusal", name)
		}
	}
	refuse("nil_dtype", func(b []schedule.Buffer) []schedule.Buffer { b[1].DType = nil; return b })
	refuse("rank1_shape", func(b []schedule.Buffer) []schedule.Buffer { b[1].Shape = []int64{4096}; return b })
	refuse("symbolic_dim_sentinel", func(b []schedule.Buffer) []schedule.Buffer { b[1].Shape = []int64{0, 64}; return b })
	refuse("size_non_div_4", func(b []schedule.Buffer) []schedule.Buffer { b[2].Size = 4094; return b })
	refuse("short_buf_table", func(b []schedule.Buffer) []schedule.Buffer { return b[:2] })
}

// TestActionSpace_Vec4Load pins BEAM integration: proposed exactly when the
// kernel is an already-tiled, 4-aligned f32 matmul AND buffer info is present.
func TestActionSpace_Vec4Load(t *testing.T) {
	hasVec4 := func(actions []codegen.Opt) bool {
		for _, a := range actions {
			if a.Kind == codegen.OptVec4Load {
				return true
			}
		}
		return false
	}

	item := matmulItem(t, 64, 64, 64)
	if hasVec4(codegen.ActionSpaceBufs(item.Ast, item.Bufs)) {
		t.Error("ActionSpace proposed OptVec4Load on an un-tiled kernel")
	}

	tiled := applySeq(t, item, tileOpts(16))
	if !hasVec4(codegen.ActionSpaceBufs(tiled, item.Bufs)) {
		t.Error("ActionSpace did not propose OptVec4Load on a tiled 64³ matmul")
	}
	if hasVec4(codegen.ActionSpace(tiled)) {
		t.Error("ActionSpace without buffer info proposed OptVec4Load - shape gate cannot run")
	}

	irr := matmulItem(t, 64, 17, 64)
	irrTiled := applySeq(t, irr, tileOpts(16))
	if hasVec4(codegen.ActionSpaceBufs(irrTiled, irr.Bufs)) {
		t.Error("ActionSpace proposed OptVec4Load on a K=17 matmul")
	}
}
