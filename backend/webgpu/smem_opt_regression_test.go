package webgpu_test

// Permanent regression tests for the two harness defects that masked a broken
// var<workgroup> (shared-memory) path on the gogpu Metal stack:
//
//   Defect A (cache mask): schedule cacheStore pre-renders WGSL from the
//     un-opted Ast at CreateSchedule time; Run/Benchmark short-circuit on a
//     non-empty item.WGSL, so mutating item.Ast after scheduling executed the
//     identity kernel. Fixed by ExecItem.SetAst (codegen.ApplyOpts routes
//     through it); pinned by TestApplyOpts_InvalidatesPrerenderedWGSL.
//
//   Defect B (zero-input mask): opt-oracle tests ran with dev.Run(items, nil),
//     comparing all-zero outputs (0 == 0). Pinned by seededLeafInputs +
//     requireNonDegenerate across the B-series oracles, and by the
//     CPU-reference tests here.
//
// TestSmem_WorkgroupPassthrough and TestSmem_TiledMatmulVsCPU (tile/b3/b37
// subtests) FAILED while the upstream smem defect persisted (naga's Go MSL
// writer emitted threadgroup entry-point parameters but the Metal HAL never
// called setThreadgroupMemoryLength:atIndex:, so every smem read returned 0).
// The dependency fix (go.mod replace → naga-smem-fix) turned them green;
// they remain un-skip-gated as the permanent smem canary.

import (
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestSmem_WorkgroupPassthrough is the minimal var<workgroup> correctness
// probe: a 64-thread workgroup writes in[lid]+lid+1 into shared memory,
// barriers, and each thread reads its neighbor's slot. Any all-zero (or
// otherwise wrong) readback means shared memory is broken in the GPU stack.
func TestSmem_WorkgroupPassthrough(t *testing.T) {
	dev := requireDevice(t)
	const wgsl = `@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@group(0) @binding(1) var<storage, read> data1: array<f32>;

var<workgroup> sm0: array<f32, 64>;
@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  sm0[lid.x] = data1[lid.x] + f32(lid.x) + 1.0;
  workgroupBarrier();
  let j = (lid.x + 1u) % 64u;
  data0[lid.x] = sm0[j];
}
`
	item := schedule.ExecItem{
		WGSL:           wgsl,
		LocalSize:      [3]int{64, 1, 1},
		WorkgroupCount: [3]int{1, 1, 1},
		Bufs: []schedule.Buffer{
			{UOpIdx: 1, Size: 64, Shape: []int64{64}, DType: uop.Dtypes.Float32, Slot: -1},
			{UOpIdx: 2, Size: 64, Shape: []int64{64}, DType: uop.Dtypes.Float32, Slot: -1},
		},
	}
	in := lcgData(64, 0x5E1)
	res, err := dev.Run([]schedule.ExecItem{item}, map[uint32][]float32{2: in})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	out := res[1]
	if len(out) != 64 {
		t.Fatalf("missing output: %v", res)
	}
	// Metal compiles MSL with fast-math: `in[j] + f32(j) + 1.0` may be
	// reassociated as `in[j] + (f32(j) + 1.0)`, which rounds differently from
	// the CPU's left-to-right f32 evaluation by 1 ULP on a few lanes. A broken
	// smem path returns 0 (or garbage) - orders of magnitude beyond 1e-5 - so
	// the tolerance keeps full diagnostic power.
	bad := 0
	for i := 0; i < 64; i++ {
		j := (i + 1) % 64
		d := out[i] - (in[j] + float32(j) + 1)
		if d < -1e-5 || d > 1e-5 {
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("smem passthrough wrong for %d/64 lanes (out[0..4]=%v) - var<workgroup> broken in the GPU stack", bad, out[:4])
	}
}

// TestSmem_TiledMatmulVsCPU runs each tiled opt stack against a CPU float32
// reference with real (nonzero, seeded) inputs. The identity subtest proves
// the harness; tile/b3/b37 prove the OptTile-derived shared-memory kernels
// actually compute on the real GPU.
func TestSmem_TiledMatmulVsCPU(t *testing.T) {
	dev := requireDevice(t)
	const M, K, N = 64, 64, 64
	const TS = 16

	stacks := []struct {
		name string
		opts []codegen.Opt
	}{
		{"identity", nil},
		{"tile", []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptTile, Axis: 0, Arg: TS},
		}},
		{"b3", b3Opts(TS, 4, 4)},
		{"b37", b37Opts(TS, 4, 4, 4)},
	}

	ad := lcgData(M*K, 0xA0)
	bd := lcgData(K*N, 0xB0)
	ref := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float32
			for k := 0; k < K; k++ {
				s += ad[i*K+k] * bd[k*N+j]
			}
			ref[i*N+j] = s
		}
	}
	requireNonDegenerate(t, "CPU reference", ref)

	for _, tc := range stacks {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(65536)
			A := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.Float32, "webgpu")
			B := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.Float32, "webgpu")
			out := A.Matmul(B)
			items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
			for i := range items {
				items[i] = applyMatmulOptsBestEffort(items[i], tc.opts)
			}
			inputs := map[uint32][]float32{
				A.Node().Index(): ad,
				B.Node().Index(): bd,
			}
			res, err := dev.Run(items, inputs)
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			got := firstFinalOutput(t, items, res)
			if len(got) != M*N {
				t.Fatalf("output length %d, want %d", len(got), M*N)
			}
			var maxd float64
			for i := range ref {
				d := float64(got[i] - ref[i])
				if d < 0 {
					d = -d
				}
				if d > maxd {
					maxd = d
				}
			}
			t.Logf("%s: max-abs-diff vs CPU = %g (got[0]=%g ref[0]=%g)", tc.name, maxd, got[0], ref[0])
			// Tolerance precedent: TestValueOracle_MatMul (webgpu_test.go) accepts
			// matmul-vs-CPU-reference at 1e-2 absolute. Tiled stacks reassociate the
			// K accumulation, so exact-0 (the GPU-vs-GPU oracle bar) does not apply;
			// with inputs in (0,1] and K=64 the fp noise is orders below 1e-2.
			if maxd > 1e-2 {
				t.Errorf("%s computes a wrong matmul on the real GPU (max-abs-diff=%g)", tc.name, maxd)
			}
		})
	}
}

// TestApplyOpts_InvalidatesPrerenderedWGSL pins the Defect A fix without
// touching the GPU: CreateSchedule pre-renders identity WGSL into the item;
// ApplyOpts must clear it (so executors re-render from the opted Ast), the
// opted render must differ from the identity render, and two renders of the
// same opted Ast must agree (what Run/Benchmark will execute). A no-op opt
// sequence must keep the pre-render - the schedule-cache fast path.
func TestApplyOpts_InvalidatesPrerenderedWGSL(t *testing.T) {
	a := uop.NewArena(65536)
	A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	out := A.Matmul(B)
	items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
	item := items[len(items)-1]
	if item.WGSL == "" {
		t.Fatalf("expected cacheStore to pre-fill item.WGSL at CreateSchedule time")
	}
	identityWGSL := item.WGSL

	opted := codegen.ApplyOpts(item, b37Opts(16, 4, 4, 4))
	if opted.WGSL != "" {
		t.Fatalf("ApplyOpts left the pre-rendered WGSL in place - executors would run the identity kernel (Defect A)")
	}
	if opted.Ast.Index() == item.Ast.Index() {
		t.Fatalf("b37 opts did not transform the matmul Ast - test premise broken")
	}

	fresh := codegen.RenderWGSL(opted).WGSL
	if fresh == identityWGSL {
		t.Errorf("opted render identical to identity render - opts not reflected in WGSL")
	}
	// Lowering appends arena nodes, so a second render of the same Ast gets
	// different index-derived variable names; compare via the normalized hash
	// (the same normalization the beam SK-collision guard relies on).
	if again := codegen.RenderWGSL(opted).WGSL; codegen.BeamWGSLHash(again) != codegen.BeamWGSLHash(fresh) {
		t.Errorf("re-render of the same opted Ast differs structurally - executor would compile a different kernel")
	}

	// Cache fast path: a no-op sequence keeps the pre-rendered WGSL.
	ident := codegen.ApplyOpts(item, []codegen.Opt{{Kind: codegen.OptIdentity}})
	if ident.WGSL != identityWGSL {
		t.Errorf("no-op opt sequence cleared the pre-rendered WGSL - schedule-cache fast path lost")
	}

	// Zeroed-Ast items (schedule-cache hits release the arena reference) pass
	// through ApplyOpts untouched.
	zeroed := item
	zeroed.Ast = uop.UOp{}
	if got := codegen.ApplyOpts(zeroed, b37Opts(16, 4, 4, 4)); got.WGSL != identityWGSL || got.Ast.Valid() {
		t.Errorf("ApplyOpts on a zeroed-Ast item must be a no-op")
	}
}
