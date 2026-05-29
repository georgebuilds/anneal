package webgpu_test

// Diagnostic experiments — read-only timing tests, no IR changes, no new Opt kinds.
// Hypothesis inventory:
//   H1 – default 1D is near real Metal ceiling (auto-vectorized)
//   H2 – M3 f32 ALU peak is much lower than headline TFLOPs
//   H3 – smem→register load latency dominates after OptTile (B3 not fixing the right thing)
//
// Exp 1: workgroup-size sweep on default matmul 1024³ (OptLocal only, TS∈{4,8,16,32})
// Exp 2: pure-FMA throughput probe — no hot-loop global memory, ≈ 2*1024³ FLOPs
// Exp 3: OptTile-only at TS∈{8,16,32} on 1024³ (no OptUpcast)
// Exp 4: OptUpcast factor sweep (MR=NR∈{2,4}) on B2-tiled kernel; note below explains why
//         OptUpcast on the bare default (no OptTile) is not currently lowerer-supported.

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// diagItem builds a fresh 1024³ matmul ExecItem with no opts applied.
func diagItem() schedule.ExecItem {
	a := uop.NewArena(65536)
	A := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	return schedule.CreateSchedule(makeSink(a, C), "webgpu")[0]
}

// diagGFLOPS converts a min-latency µs measurement to GFLOP/s for an N³ matmul.
func diagGFLOPS(N int64, minMicros float64) float64 {
	return (2.0 * float64(N*N*N)) / (minMicros * 1e3)
}

// TestDiag_WorkgroupSweep (Exp 1) — measures 1024³ matmul at workgroup sizes
// WS = TS*TS (TS∈{4,8,16,32}) using two symmetric OptLocal passes. Tests whether
// occupancy / wavefront count is the performance lever.
func TestDiag_WorkgroupSweep(t *testing.T) {
	dev := requireDevice(t)
	const N = int64(1024)

	resBase, err := dev.Benchmark(diagItem(), 2, 5)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	gBase := diagGFLOPS(N, resBase.MinMicros)
	fmt.Printf("[Diag-1] Default 1D:         Min=%7.2fµs  %6.2f GFLOP/s\n",
		resBase.MinMicros, gBase)

	for _, TS := range []int{4, 8, 16, 32} {
		item := diagItem()
		item.Ast = codegen.ApplyOpts(item, []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		}).Ast
		res, err := dev.Benchmark(item, 2, 5)
		if err != nil {
			t.Fatalf("TS=%d: %v", TS, err)
		}
		g := diagGFLOPS(N, res.MinMicros)
		fmt.Printf("[Diag-1] WS=%4d (TS=%2d):    Min=%7.2fµs  %6.2f GFLOP/s  %.2fx\n",
			TS*TS, TS, res.MinMicros, g, g/gBase)
	}
}

// TestDiag_FMAProbe (Exp 2) — injects a raw WGSL kernel with no smem and no
// hot-loop global memory. Measures raw f32 ALU throughput as a ceiling proxy.
//
// Kernel: workgroup_size=(256,1,1), dispatch=(4096,1,1).
// Each thread: 4 independent f32 accumulators, 256 fma iterations = 2048 FMAs/thread.
// Total FLOPs ≈ 4 × 256 × 2 × 256×4096 = 2,147,483,648 (matches 2*1024³ matmul).
func TestDiag_FMAProbe(t *testing.T) {
	dev := requireDevice(t)

	const probeWGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx  = gid.x;
    let seed = inp[0u] + f32(idx % 7u) * 0.001f;
    var a0   = seed;
    var a1   = seed + 1.0f;
    var a2   = seed + 2.0f;
    var a3   = seed + 3.0f;
    let c    = 1.00001f;
    let d    = 0.00001f;
    for (var i: u32 = 0u; i < 256u; i++) {
        a0 = a0 * c + d;
        a1 = a1 * c + d;
        a2 = a2 * c + d;
        a3 = a3 * c + d;
    }
    out[idx] = a0 + a1 + a2 + a3;
}
`
	const (
		wgX     = 4096
		wgY     = 1
		threads = 256 * wgX // 1,048,576
	)
	const probeFlops = float64(4 * 256 * 2 * threads) // 2,147,483,648

	item := schedule.ExecItem{
		WGSL:           probeWGSL,
		LocalSize:      [3]int{256, 1, 1},
		WorkgroupCount: [3]int{wgX, wgY, 1},
		Bufs: []schedule.Buffer{
			{UOpIdx: 0, Size: threads, DType: uop.Dtypes.Float32},
			{UOpIdx: 1, Size: 1, DType: uop.Dtypes.Float32},
		},
	}
	res, err := dev.Benchmark(item, 2, 5)
	if err != nil {
		t.Fatalf("FMA probe: %v", err)
	}
	gflops := probeFlops / (res.MinMicros * 1e3)
	fmt.Printf("[Diag-2] Pure-FMA probe:     Min=%7.2fµs  %6.2f GFLOP/s  (FLOPs=%.3eF)\n",
		res.MinMicros, gflops, probeFlops)
}

// TestDiag_TileNoUpcast (Exp 3) — measures 1024³ matmul with OptTile only (no
// OptUpcast). TS∈{8,16,32}. Isolates smem-tiling benefit from register blocking.
func TestDiag_TileNoUpcast(t *testing.T) {
	dev := requireDevice(t)
	const N = int64(1024)

	resBase, err := dev.Benchmark(diagItem(), 2, 5)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	gBase := diagGFLOPS(N, resBase.MinMicros)
	fmt.Printf("[Diag-3] Default 1D:         Min=%7.2fµs  %6.2f GFLOP/s\n",
		resBase.MinMicros, gBase)

	for _, TS := range []int{8, 16, 32} {
		item := diagItem()
		item.Ast = codegen.ApplyOpts(item, []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptTile, Axis: 0, Arg: TS},
		}).Ast
		res, err := dev.Benchmark(item, 2, 5)
		if err != nil {
			t.Fatalf("TS=%d: %v", TS, err)
		}
		g := diagGFLOPS(N, res.MinMicros)
		wgs := (N / int64(TS)) * (N / int64(TS))
		fmt.Printf("[Diag-3] OptTile TS=%2d:      Min=%7.2fµs  %6.2f GFLOP/s  %.2fx  WGs=%d\n",
			TS, res.MinMicros, g, g/gBase, wgs)
	}
}

// TestDiag_UpcastFactorSweep (Exp 4) — sweeps OptUpcast factor (MR=NR∈{1,2,4}) on
// a B2-tiled (TS=16) base kernel. MR=NR=1 is pure B2 (no upcast); MR=NR=4 is B3.
//
// NOTE: OptUpcast on the bare default kernel (no OptTile, no OptLocal) is not
// supported by the current lowerer — AxisUpcast ranges outside the tiled-reduce
// path receive placeholder expression "0", so each thread computes only row M*F+0
// (not the full F-wide stripe). This would give silently wrong outputs. The useful
// diagnostic equivalent is the factor sweep below: it directly shows how the
// occupancy / workgroup-count tradeoff scales with the upcast factor.
func TestDiag_UpcastFactorSweep(t *testing.T) {
	dev := requireDevice(t)
	const (
		N  = int64(1024)
		TS = 16
	)

	// MR=NR=1 → B2 (no upcast)
	b2item := diagItem()
	b2item.Ast = codegen.ApplyOpts(b2item, []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptTile, Axis: 0, Arg: TS},
	}).Ast
	resB2, err := dev.Benchmark(b2item, 2, 5)
	if err != nil {
		t.Fatalf("B2 baseline: %v", err)
	}
	gB2 := diagGFLOPS(N, resB2.MinMicros)
	wgsB2 := (N / TS) * (N / TS)
	fmt.Printf("[Diag-4] B2 (MR=NR=1):       Min=%7.2fµs  %6.2f GFLOP/s  WGs=%d\n",
		resB2.MinMicros, gB2, wgsB2)

	for _, F := range []int{2, 4} {
		item := diagItem()
		item.Ast = codegen.ApplyOpts(item, []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptTile, Axis: 0, Arg: TS},
			{Kind: codegen.OptUpcast, Axis: 0, Arg: F}, // M_wg
			{Kind: codegen.OptUpcast, Axis: 1, Arg: F}, // N_wg
		}).Ast
		res, err := dev.Benchmark(item, 2, 5)
		if err != nil {
			t.Fatalf("MR=NR=%d: %v", F, err)
		}
		g := diagGFLOPS(N, res.MinMicros)
		wgs := (N / (int64(F) * TS)) * (N / (int64(F) * TS))
		fmt.Printf("[Diag-4] B3 (MR=NR=%d):       Min=%7.2fµs  %6.2f GFLOP/s  %.2fx  WGs=%d\n",
			F, res.MinMicros, g, g/gB2, wgs)
	}
}
