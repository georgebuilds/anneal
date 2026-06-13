package codegen_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// buildImageElementwise1D builds a one-kernel schedule copying an n-element
// image-typed input to an image-typed output via Mul by a scalar const.
func buildImageElementwise1D(t *testing.T, n int64) schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.ImageFloat32, "webgpu")
	two := tensor.ConstScalar(a, 2.0, uop.Dtypes.ImageFloat32, "webgpu")
	out := x.Mul(two)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatalf("CreateSchedule returned 0 items for image elementwise n=%d", n)
	}
	return items[0]
}

// buildImageMatmul builds the schedule for an M×K @ K×N image matmul.
func buildImageMatmul(t *testing.T, M, K, N int64) []schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(65536)
	A := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.ImageFloat32, "webgpu")
	B := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.ImageFloat32, "webgpu")
	out := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatalf("CreateSchedule returned 0 items for image matmul")
	}
	return items
}

// TestWGSL_ImageSlotDispatch_StorePattern pins the vec4 slot-dispatch store
// shape for image-output kernels: one thread per vec4 slot, a 4-lane loop
// with per-lane masking, and a single full-slot store. The legacy per-lane
// cascade to storage (data0[_img_slot].x = ...) must be gone — it is what
// raced when the output row stride was not a multiple of 4.
func TestWGSL_ImageSlotDispatch_StorePattern(t *testing.T) {
	item := buildImageElementwise1D(t, 256)
	wgsl := codegen.RenderWGSL(item).WGSL
	for _, want := range []string{
		"var _img_out: vec4<f32> = vec4<f32>(0.0);",
		"for (var _img_lane: u32 = 0u; _img_lane < 4u; _img_lane++) {",
		"let _img_flat = gid_x * 4u + _img_lane;",
		"data0[gid_x] = _img_out;",
	} {
		if !strings.Contains(wgsl, want) {
			t.Errorf("image kernel WGSL missing slot-dispatch fragment %q\nfull shader:\n%s", want, wgsl)
		}
	}
	if strings.Contains(wgsl, "data0[_img_slot]") {
		t.Errorf("image kernel WGSL still uses the legacy per-lane storage cascade (race-prone)\nfull shader:\n%s", wgsl)
	}
	// Exactly one write to data0 — the whole-slot store.
	if n := strings.Count(wgsl, "data0["); n != 1 {
		t.Errorf("expected exactly 1 data0 store (whole-slot write), got %d\nfull shader:\n%s", n, wgsl)
	}
}

// TestWGSL_ImageSlotDispatch_TailMask pins the tail masking for every
// numel mod 4 residue: the slot-count guard and the per-lane flat bound must
// carry the exact literals. Residue 0 keeps the (statically always-true)
// lane guard for uniformity.
func TestWGSL_ImageSlotDispatch_TailMask(t *testing.T) {
	cases := []struct {
		n     int64
		slots int64
	}{
		{16, 4}, // residue 0
		{17, 5}, // residue 1
		{18, 5}, // residue 2
		{19, 5}, // residue 3
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("n_%d", tc.n), func(t *testing.T) {
			item := buildImageElementwise1D(t, tc.n)
			wgsl := codegen.RenderWGSL(item).WGSL
			slotGuard := fmt.Sprintf("if (gid_x >= %du) { return; }", tc.slots)
			laneGuard := fmt.Sprintf("if (_img_flat < %du) {", tc.n)
			if !strings.Contains(wgsl, slotGuard) {
				t.Errorf("missing slot guard %q\nfull shader:\n%s", slotGuard, wgsl)
			}
			if !strings.Contains(wgsl, laneGuard) {
				t.Errorf("missing lane tail mask %q\nfull shader:\n%s", laneGuard, wgsl)
			}
		})
	}
}

// TestWGSL_ImageSlotDispatch_DispatchGeometry pins the derived dispatch: the
// global thread count covers ceil(numel/4) slots with the 64-wide default
// workgroup, not one thread per logical element.
func TestWGSL_ImageSlotDispatch_DispatchGeometry(t *testing.T) {
	// 256 logical elements → 64 slots → exactly one 64-wide workgroup.
	res := codegen.RenderWGSL(buildImageElementwise1D(t, 256))
	if res.LocalSize != [3]int{64, 1, 1} {
		t.Errorf("LocalSize = %v, want [64 1 1]", res.LocalSize)
	}
	if res.WorkgroupCount != [3]int{1, 1, 1} {
		t.Errorf("WorkgroupCount = %v, want [1 1 1] (64 slots / 64 threads)", res.WorkgroupCount)
	}
	// 1030 logical elements → 258 slots → ceil(258/64) = 5 workgroups.
	res = codegen.RenderWGSL(buildImageElementwise1D(t, 1030))
	if res.WorkgroupCount != [3]int{5, 1, 1} {
		t.Errorf("WorkgroupCount = %v, want [5 1 1] (258 slots, 64-wide)", res.WorkgroupCount)
	}
}

// TestWGSL_ImageSlotDispatch_MatmulUnaligned pins the slot path on a reduce
// kernel whose output row stride is NOT a multiple of 4 (N=6): the reduce
// loop must sit inside the lane loop and the store must be the single
// whole-slot write. This is the configuration the legacy cascade raced on.
func TestWGSL_ImageSlotDispatch_MatmulUnaligned(t *testing.T) {
	items := buildImageMatmul(t, 6, 5, 6)
	wgsl := codegen.RenderWGSL(items[len(items)-1]).WGSL
	laneIdx := strings.Index(wgsl, "for (var _img_lane")
	accIdx := strings.Index(wgsl, "var acc0:")
	storeIdx := strings.Index(wgsl, "data0[gid_x] = _img_out;")
	if laneIdx < 0 || accIdx < 0 || storeIdx < 0 {
		t.Fatalf("missing lane loop / accumulator / slot store\nfull shader:\n%s", wgsl)
	}
	if laneIdx >= accIdx || accIdx >= storeIdx {
		t.Errorf("expected lane loop (%d) < reduce acc (%d) < slot store (%d)\nfull shader:\n%s",
			laneIdx, accIdx, storeIdx, wgsl)
	}
}

// TestWGSL_ImageSlotDispatch_NormalizeStable is the cross-arena byte-identity
// proof for the slot-dispatch path (SPEC §10): the new _img_* identifiers are
// fixed strings and r{N}/t{N} are covered by normalizeWGSL, so two arenas
// building the same unaligned image matmul must hash identically.
func TestWGSL_ImageSlotDispatch_NormalizeStable(t *testing.T) {
	build := func() string {
		items := buildImageMatmul(t, 6, 5, 6)
		return codegen.RenderWGSL(items[len(items)-1]).WGSL
	}
	w1, w2 := build(), build()
	h1, h2 := codegen.BeamWGSLHash(w1), codegen.BeamWGSLHash(w2)
	if h1 != h2 {
		t.Fatalf("normalizeWGSL not byte-stable across arenas for image slot dispatch: "+
			"hash1=%s hash2=%s\nshader1:\n%s\nshader2:\n%s", h1, h2, w1, w2)
	}
}

// TestWGSL_ImageSlotDispatch_NonImageUnchanged guards requirement 3: scalar
// f32 kernels must not pick up any slot-dispatch artifacts.
func TestWGSL_ImageSlotDispatch_NonImageUnchanged(t *testing.T) {
	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{17}, uop.Dtypes.Float32, "webgpu")
	two := tensor.ConstScalar(a, 2.0, uop.Dtypes.Float32, "webgpu")
	out := x.Mul(two)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	wgsl := codegen.RenderWGSL(items[0]).WGSL
	if strings.Contains(wgsl, "_img_") {
		t.Errorf("scalar f32 kernel contains _img_ slot-dispatch artifacts\nfull shader:\n%s", wgsl)
	}
}

// TestActionSpace_ImageKernelEmpty pins the BEAM exclusion: image-output
// kernels get an empty action space (every Opt would push the kernel off the
// deterministic slot dispatch and back onto the race-prone cascade), while
// the same kernel in scalar f32 form keeps a non-empty space — proving the
// filter keys on the output dtype, not on kernel shape.
func TestActionSpace_ImageKernelEmpty(t *testing.T) {
	items := buildImageMatmul(t, 16, 16, 16)
	if got := codegen.ActionSpace(items[len(items)-1].Ast); len(got) != 0 {
		t.Errorf("ActionSpace(image matmul) = %d actions, want 0", len(got))
	}

	a := uop.NewArena(65536)
	A := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
	out := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	scalarItems := schedule.CreateSchedule(sink, "webgpu")
	if got := codegen.ActionSpace(scalarItems[len(scalarItems)-1].Ast); len(got) == 0 {
		t.Errorf("ActionSpace(scalar f32 matmul) is empty — filter is over-broad")
	}
}

// TestLower_ImageKernel_HandOptPanics pins the fail-loud guard: hand-applying
// an Opt to an image-output kernel (bypassing the ActionSpace filter) must
// panic at Lower time rather than silently lowering through the race-prone
// legacy store path.
func TestLower_ImageKernel_HandOptPanics(t *testing.T) {
	item := buildImageElementwise1D(t, 256)
	opted := codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 0, Arg: 8})
	if opted.Index() == item.Ast.Index() {
		t.Fatalf("OptLocal did not transform the image kernel — panic guard untestable")
	}
	item.SetAst(opted)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Lower(image kernel with hand-applied OptLocal) did not panic")
		}
	}()
	codegen.Lower(item)
}
