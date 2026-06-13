package webgpu_test

// OptVec4Load GPU verification: value oracles (bit-exact vs identity — the
// load width never changes the per-output FMA order), refusal-path identity,
// BEAM integration, and the slice timing table (Logf/Printf only; no perf
// gates per the timing-harness contract in b0_test.go).

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// tileVec4Opts is the minimal OptVec4Load stack (B2 tiled path + vec4 loads).
func tileVec4Opts(TS int) []codegen.Opt {
	return []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptTile, Axis: 0, Arg: TS},
		{Kind: codegen.OptVec4Load},
	}
}

// vec4MatmulOracle runs an M×K·K×N matmul twice — identity and opts — with
// identical seeded inputs and requires bit-exact agreement (max-abs-diff 0).
// requireApplied asserts the opt sequence actually transformed the kernel
// (guards against a silent refusal turning the oracle vacuous).
func vec4MatmulOracle(t *testing.T, dev *webgpu.Device, M, K, N int64, opts []codegen.Opt, seed uint64, requireApplied bool) {
	t.Helper()
	build := func() []schedule.ExecItem {
		a := uop.NewArena(1 << 17)
		A := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.Float32, "webgpu")
		return schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")
	}

	itemsDef := build()
	resDef, err := dev.Run(itemsDef, seededLeafInputs(itemsDef, seed))
	if err != nil {
		t.Fatalf("identity run: %v", err)
	}
	gotDef := firstFinalOutput(t, itemsDef, resDef)
	requireNonDegenerate(t, "identity output", gotDef)

	itemsOpt := build()
	applied := false
	for i := range itemsOpt {
		before := itemsOpt[i].Ast
		itemsOpt[i] = codegen.ApplyOpts(itemsOpt[i], opts)
		if itemsOpt[i].Ast.Index() != before.Index() {
			applied = true
		}
	}
	if requireApplied && !applied {
		t.Fatal("opt sequence did not transform any kernel — oracle would be vacuous")
	}
	resOpt, err := dev.Run(itemsOpt, seededLeafInputs(itemsOpt, seed))
	if err != nil {
		t.Fatalf("opted run: %v", err)
	}
	gotOpt := firstFinalOutput(t, itemsOpt, resOpt)

	if len(gotOpt) != len(gotDef) {
		t.Fatalf("length mismatch: def=%d opt=%d", len(gotDef), len(gotOpt))
	}
	var maxd float64
	var idx int
	for i := range gotDef {
		d := float64(gotOpt[i] - gotDef[i])
		if d < 0 {
			d = -d
		}
		if d > maxd {
			maxd = d
			idx = i
		}
	}
	t.Logf("matmul %dx%dx%d: max-abs-diff vs identity = %g", M, K, N, maxd)
	if maxd != 0 {
		t.Errorf("not bit-exact: max-abs-diff=%g at i=%d (def=%g opt=%g) — vec4 loads must not change FMA order",
			maxd, idx, gotDef[idx], gotOpt[idx])
	}
}

// TestVec4Load_ValueOracle_TileMatmul: OptTile+OptVec4Load (B2 path) across
// the briefed sizes. 1024³ doubles as the large-shape oracle for the timing
// configuration.
func TestVec4Load_ValueOracle_TileMatmul(t *testing.T) {
	dev := requireDevice(t)
	for _, N := range []int64{64, 256, 512, 1024} {
		t.Run(fmt.Sprintf("matmul_%d", N), func(t *testing.T) {
			if N >= 512 {
				skipIfSoftwareGPU(t, dev)
			}
			vec4MatmulOracle(t, dev, N, N, N, tileVec4Opts(16), 0x4C0+uint64(N), true)
		})
	}
}

// TestVec4Load_ValueOracle_Composition: vec4 loads composed under the B3
// (upcast) and B3.7 (full OptLocal+OptTile+OptUpcast+OptVectorize+OptVec4Load)
// lowering paths, plus alignment-edge shapes:
//   - M=17: M is NOT 4-aligned (only the stride-1 extents K and N are gated);
//     padded rows exercise the slot-load row mask.
//   - K=20 with TS=16: 4-aligned K that the tile sweep overshoots — the tail
//     vec4 slots are fully out-of-range and must be masked whole.
func TestVec4Load_ValueOracle_Composition(t *testing.T) {
	dev := requireDevice(t)

	b3v4 := append(b3Opts(16, 4, 4), codegen.Opt{Kind: codegen.OptVec4Load})
	b37v4 := append(b37Opts(16, 4, 4, 4), codegen.Opt{Kind: codegen.OptVec4Load})

	cases := []struct {
		name    string
		M, K, N int64
		opts    []codegen.Opt
	}{
		{"b3_upcast_128", 128, 128, 128, b3v4},
		{"b37_full_64", 64, 64, 64, b37v4},
		{"b37_full_256", 256, 256, 256, b37v4},
		{"tile_M17_rowmask", 17, 32, 32, tileVec4Opts(16)},
		{"tile_K20_tailslot", 64, 20, 64, tileVec4Opts(16)},
		{"b3_M17_K20", 17, 20, 64, b3v4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			vec4MatmulOracle(t, dev, tc.M, tc.K, tc.N, tc.opts, 0x4C1, true)
		})
	}
}

// TestVec4Load_Refusal_K17 pins the GPU-side refusal behavior: with K=17 the
// whole opt refuses (both-or-nothing) and the kernel still computes correctly
// through the scalar tiled path; the rendered WGSL carries no vec4 bindings.
func TestVec4Load_Refusal_K17(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(1 << 17)
	A := tensor.NewLeaf(a, []int64{64, 17}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{17, 64}, uop.Dtypes.Float32, "webgpu")
	items := schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")
	item := codegen.ApplyOpts(items[len(items)-1], tileVec4Opts(16))
	if got := codegen.Vec4LoadParams(item.Ast); len(got) != 0 {
		t.Fatalf("K=17 kernel has vec4-tagged params %v — refusal gate failed", got)
	}

	// Values still bit-exact vs identity through the scalar tiled fallback.
	vec4MatmulOracle(t, dev, 64, 17, 64, tileVec4Opts(16), 0x4C2, false)
}

// TestVec4Load_Refusal_ConvSpray sprays the vec4 stack across a conv schedule:
// no kernel is tilable-matmul shaped, every OptVec4Load refuses, and the
// output is unchanged. (Symbolic and image refusals are pinned without a GPU
// in codegen/optvec4load_test.go.)
func TestVec4Load_Refusal_ConvSpray(t *testing.T) {
	dev := requireDevice(t)

	build := func() []schedule.ExecItem {
		a := uop.NewArena(1 << 17)
		x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
		conv := nn.NewConv2d(a, 1, 1, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "webgpu")
		conv.Weight.Value = uniformData(9, 8)
		conv.Weight.Load(a)
		out := conv.Forward(x)
		return schedule.CreateSchedule(makeSink(a, out), "webgpu")
	}

	itemsDef := build()
	resDef, err := dev.Run(itemsDef, seededLeafInputs(itemsDef, 0x4C3))
	if err != nil {
		t.Fatalf("default run: %v", err)
	}
	gotDef := firstFinalOutput(t, itemsDef, resDef)
	requireNonDegenerate(t, "default conv output", gotDef)

	itemsOpt := build()
	for i := range itemsOpt {
		itemsOpt[i] = applyMatmulOptsBestEffort(itemsOpt[i], tileVec4Opts(16))
	}
	resOpt, err := dev.Run(itemsOpt, seededLeafInputs(itemsOpt, 0x4C3))
	if err != nil {
		t.Fatalf("sprayed run: %v", err)
	}
	gotOpt := firstFinalOutput(t, itemsOpt, resOpt)
	if !approxEq(gotOpt, gotDef, 0) {
		t.Errorf("conv output changed under sprayed vec4 stack: def=%v opt=%v", firstN(gotDef, 4), firstN(gotOpt, 4))
	}
}

// TestVec4Load_BeamSearch verifies OptVec4Load inside a real beam run on a
// 256³ matmul: the search completes, and the winning sequence — whatever it
// is — stays bit-exact vs identity (the beam value guard plus an external
// re-check here). Logs whether the winner picked OptVec4Load.
func TestVec4Load_BeamSearch(t *testing.T) {
	dev := requireDevice(t)
	skipIfSoftwareGPU(t, dev)
	codegen.BeamCacheReset()

	mk := func() []schedule.ExecItem {
		a := uop.NewArena(1 << 17)
		A := tensor.NewLeaf(a, []int64{256, 256}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{256, 256}, uop.Dtypes.Float32, "webgpu")
		return schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")
	}

	cfg := codegen.BeamConfig{Width: 4, MaxDepth: 6, Warmup: 2, Iters: 5}
	br := codegen.BeamSearch(dev, dev, mk()[0], cfg)
	t.Logf("beam winner: opts=%v  min=%.2fµs  base=%.2fµs  searched=%d",
		br.Opts, br.MinMicros, br.BaseMicros, br.Searched)
	hasVec4 := false
	for _, o := range br.Opts {
		if o.Kind == codegen.OptVec4Load {
			hasVec4 = true
		}
	}
	t.Logf("winner includes OptVec4Load: %v", hasVec4)

	if len(br.Opts) > 0 {
		vec4MatmulOracle(t, dev, 256, 256, 256, br.Opts, 0x4C4, true)
	}
}

// TestVec4Load_Timing — THE slice timing table (report-only, min-of-N):
// identity vs best-known stacks without OptVec4Load vs with it, 1024³ and
// 2048³. This is also the first REAL tiled-matmul baseline after the smem fix
// (all prior B-series numbers were identity-vs-identity artifacts).
func TestVec4Load_Timing(t *testing.T) {
	dev := requireDevice(t)
	skipIfSoftwareGPU(t, dev)

	const (
		warmup = 2
		iters  = 6
		TS     = 16
	)
	b3 := b3Opts(TS, 4, 4)
	b37 := b37Opts(TS, 4, 4, 4)
	configs := []struct {
		name string
		opts []codegen.Opt
	}{
		{"identity", nil},
		{"tile", tileVec4Opts(TS)[:3]},
		{"tile+vec4", tileVec4Opts(TS)},
		{"b3 (tile+upcast)", b3},
		{"b3+vec4", append(append([]codegen.Opt{}, b3...), codegen.Opt{Kind: codegen.OptVec4Load})},
		{"b37 (tile+upcast+vectorize)", b37},
		{"b37+vec4", append(append([]codegen.Opt{}, b37...), codegen.Opt{Kind: codegen.OptVec4Load})},
	}

	fmt.Printf("\n=== OptVec4Load TIMING TABLE — min-of-%d, warmup=%d, TS=%d MR=NR=4 W=4 ===\n", iters, warmup, TS)
	for _, N := range []int64{1024, 2048} {
		a := uop.NewArena(1 << 17)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		base := schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")[0]

		var idGflops float64
		for _, cfg := range configs {
			item := base
			if len(cfg.opts) > 0 {
				item = codegen.ApplyOpts(base, cfg.opts)
				if item.Ast.Index() == base.Ast.Index() {
					fmt.Printf("%4d³  %-28s  REFUSED (no-op)\n", N, cfg.name)
					continue
				}
			}
			res, err := dev.Benchmark(item, warmup, iters)
			if err != nil {
				fmt.Printf("%4d³  %-28s  FAILED: %v\n", N, cfg.name, err)
				continue
			}
			gflops := rebaselineGFLOPS(N, res.MinMicros)
			speedup := ""
			if cfg.name == "identity" {
				idGflops = gflops
			} else if idGflops > 0 {
				speedup = fmt.Sprintf("  %5.2fx vs identity", gflops/idGflops)
			}
			fmt.Printf("%4d³  %-28s  min=%10.2fµs  %8.2f GFLOP/s%s\n",
				N, cfg.name, res.MinMicros, gflops, speedup)
		}
		fmt.Println()
	}
}
