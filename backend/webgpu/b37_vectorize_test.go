package webgpu_test

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// b37Opts returns the canonical B3.7 pipeline: OptLocal×2 + OptTile + OptUpcast×2 + OptVectorize.
// axis=1 targets N_loc (the innermost N-local axis, stride-1 in B and C).
// After the two OptUpcasts, the eligible non-reduce, non-upcast-partnered ranges are:
//
//	idx=0: M_loc (AxisLocal, M direction)
//	idx=1: N_loc (AxisLocal, N direction ← stride-1 ✓)
func b37Opts(TS, MR, NR, W int) []codegen.Opt {
	return append(b3Opts(TS, MR, NR),
		codegen.Opt{Kind: codegen.OptVectorize, Axis: 1, Arg: W})
}

// TestB37_ValueOracle_VectorizeMatmul checks bit-exact agreement of
// OptTile+OptUpcast+OptVectorize against the default 1D path.
// Includes irregular shapes and a shape where the N dim is not divisible by 4.
func TestB37_ValueOracle_VectorizeMatmul(t *testing.T) {
	dev := requireDevice(t)

	tests := []struct {
		name       string
		M, N, K    int64
		TS, MR, NR int
		W          int
	}{
		// Regular shapes
		{"matmul_64x64x64_TS16_MR4_NR4_W4", 64, 64, 64, 16, 4, 4, 4},
		{"matmul_128x128x128_TS16_MR4_NR4_W4", 128, 128, 128, 16, 4, 4, 4},
		// Existing B3 irregular shapes — B37 must pass all of these too.
		{"matmul_irregular_M17_TS16_MR4_NR4_W4", 17, 32, 32, 16, 4, 4, 4},
		{"matmul_irregular_N30_TS16_MR4_NR4_W4", 32, 30, 32, 16, 4, 4, 4},
		{"matmul_irregular_M17N30K35_TS16_MR4_NR4_W4", 17, 30, 35, 16, 4, 4, 4},
		// Vector-non-multiple: N=17 is not divisible by W=4.
		// The per-component bounds check in the vec4 store path must handle the tail.
		{"matmul_vecnonmult_N17_TS16_MR4_NR4_W4", 32, 17, 32, 16, 4, 4, 4},
		{"matmul_vecnonmult_N30_TS16_MR4_NR4_W4", 64, 30, 64, 16, 4, 4, 4},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(65536)
			A := tensor.NewLeaf(a, []int64{tc.M, tc.K}, uop.Dtypes.Float32, "webgpu")
			B := tensor.NewLeaf(a, []int64{tc.K, tc.N}, uop.Dtypes.Float32, "webgpu")
			A.SetData(uniformData(int(tc.M*tc.K), 1))
			B.SetData(uniformData(int(tc.K*tc.N), 2))
			out := A.Matmul(B)
			itemsDef := schedule.CreateSchedule(makeSink(a, out), "webgpu")
			resDef, err := dev.Run(itemsDef, seededLeafInputs(itemsDef, 0xB37))
			if err != nil {
				t.Fatalf("Default run failed: %v", err)
			}
			gotDef := firstFinalOutput(t, itemsDef, resDef)
			requireNonDegenerate(t, "default output", gotDef)

			a2 := uop.NewArena(65536)
			A2 := tensor.NewLeaf(a2, []int64{tc.M, tc.K}, uop.Dtypes.Float32, "webgpu")
			B2 := tensor.NewLeaf(a2, []int64{tc.K, tc.N}, uop.Dtypes.Float32, "webgpu")
			A2.SetData(uniformData(int(tc.M*tc.K), 1))
			B2.SetData(uniformData(int(tc.K*tc.N), 2))
			out2 := A2.Matmul(B2)
			itemsOpt := schedule.CreateSchedule(makeSink(a2, out2), "webgpu")
			for i := range itemsOpt {
				itemsOpt[i] = applyMatmulOptsBestEffort(itemsOpt[i], b37Opts(tc.TS, tc.MR, tc.NR, tc.W))
			}
			resOpt, err := dev.Run(itemsOpt, seededLeafInputs(itemsOpt, 0xB37))
			if err != nil {
				t.Fatalf("Opt run failed: %v", err)
			}
			gotOpt := firstFinalOutput(t, itemsOpt, resOpt)

			if len(gotOpt) != len(gotDef) {
				t.Fatalf("length mismatch: def=%d opt=%d", len(gotDef), len(gotOpt))
			}
			if !approxEq(gotOpt, gotDef, 0) {
				var maxDiff float32
				var idx int
				for i := range gotDef {
					d := gotOpt[i] - gotDef[i]
					if d < 0 {
						d = -d
					}
					if d > maxDiff {
						maxDiff = d
						idx = i
					}
				}
				t.Fatalf("value mismatch at i=%d: def=%g opt=%g (max-abs-diff=%g)",
					idx, gotDef[idx], gotOpt[idx], maxDiff)
			}
		})
	}
}

// TestB37_ValueOracle_VectorizeMLPBackward checks MLP forward+backward grad values
// are bit-exact under OptTile+OptUpcast+OptVectorize. Non-matmul kernels (Mean, Sub)
// return unchanged from ApplyOpts (no eligible AxisLocal to vectorize).
func TestB37_ValueOracle_VectorizeMLPBackward(t *testing.T) {
	dev := requireDevice(t)

	build := func() (*tensor.Tensor, *tensor.Tensor, []schedule.ExecItem) {
		a := uop.NewArena(1 << 17)
		x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "webgpu")
		w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "webgpu")
		x.SetData(uniformData(16*32, 7))
		w.SetData(uniformData(32*64, 9))
		pred := x.Matmul(w)
		loss := pred.Sum(nil, false)
		grads := tensor.Backward(loss, []*tensor.Tensor{x, w})
		gx := grads[x]
		gw := grads[w]
		items := schedule.CreateSchedule(
			a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{gx.Node(), gw.Node()}, nil, nil),
			"webgpu",
		)
		return gx, gw, items
	}

	_, _, itemsDef := build()
	resDef, err := dev.Run(itemsDef, seededLeafInputs(itemsDef, 0xB371))
	if err != nil {
		t.Fatalf("Default run: %v", err)
	}
	foDef := finalOutputs(t, itemsDef, resDef)
	gxD, gwD := foDef[0], foDef[1]
	requireNonDegenerate(t, "default gx", gxD)
	requireNonDegenerate(t, "default gw", gwD)

	_, _, itemsOpt := build()
	for i := range itemsOpt {
		itemsOpt[i] = applyMatmulOptsBestEffort(itemsOpt[i], b37Opts(16, 4, 4, 4))
	}
	resOpt, err := dev.Run(itemsOpt, seededLeafInputs(itemsOpt, 0xB371))
	if err != nil {
		t.Fatalf("Opt run: %v", err)
	}
	foOpt := finalOutputs(t, itemsOpt, resOpt)
	gxO, gwO := foOpt[0], foOpt[1]

	if !approxEq(gxO, gxD, 0) {
		t.Errorf("gx mismatch (first 4): def=%v opt=%v", firstN(gxD, 4), firstN(gxO, 4))
	}
	if !approxEq(gwO, gwD, 0) {
		t.Errorf("gw mismatch (first 4): def=%v opt=%v", firstN(gwD, 4), firstN(gwO, 4))
	}
}

// TestB37_ValueOracle_VectorizeConv checks conv2d output is unchanged under b37Opts.
// Conv kernels don't match the tiled-matmul pattern: OptTile refuses (no
// Mul(Index, Index) reduce), OptLocal refuses its non-divisible L=16 split on
// the conv kernels' small axes (divisibility gate in applyLocal), and the
// spray helper skips OptUpcast/OptVectorize on untiled kernels.
func TestB37_ValueOracle_VectorizeConv(t *testing.T) {
	dev := requireDevice(t)

	build := func() (*tensor.Tensor, []schedule.ExecItem) {
		a := uop.NewArena(1 << 17)
		x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
		x.SetData(uniformData(64, 7))
		conv := nn.NewConv2d(a, 1, 1, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "webgpu")
		conv.Weight.Value = uniformData(9, 8)
		conv.Weight.Load(a)
		out := conv.Forward(x)
		items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
		return out, items
	}

	_, itemsDef := build()
	resDef, err := dev.Run(itemsDef, seededLeafInputs(itemsDef, 0xB372))
	if err != nil {
		t.Fatalf("Default run: %v", err)
	}
	gotDef := firstFinalOutput(t, itemsDef, resDef)
	requireNonDegenerate(t, "default conv output", gotDef)

	_, itemsOpt := build()
	for i := range itemsOpt {
		itemsOpt[i] = applyMatmulOptsBestEffort(itemsOpt[i], b37Opts(16, 4, 4, 4))
	}
	resOpt, err := dev.Run(itemsOpt, seededLeafInputs(itemsOpt, 0xB372))
	if err != nil {
		t.Fatalf("Opt run: %v", err)
	}
	gotOpt := firstFinalOutput(t, itemsOpt, resOpt)

	if !approxEq(gotOpt, gotDef, 0) {
		t.Errorf("conv mismatch (first 4): def=%v opt=%v", firstN(gotDef, 4), firstN(gotOpt, 4))
	}
}

// TestB37_ScheduleCache_HitCorrect verifies that b37Opts produces reproducible WGSL
// (structural cache key is stable across two independent renders).
func TestB37_ScheduleCache_HitCorrect(t *testing.T) {
	requireDevice(t)

	mk := func() schedule.ExecItem {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		items := schedule.CreateSchedule(makeSink(a, C), "webgpu")
		return items[0]
	}

	build := func() string {
		item := codegen.ApplyOpts(mk(), b37Opts(16, 4, 4, 4))
		return codegen.RenderWGSL(item).WGSL
	}

	w1 := build()
	w2 := build()
	if w1 != w2 {
		t.Fatalf("WGSL render not reproducible under b37Opts (cache key risk)")
	}
}

// TestB37_GeometryRegression_WorkgroupShrink locks the mechanism behind the
// measured b37 regression (OptVectorize ~109 GF/s vs b3 ~312 GF/s @1024³): the
// OptVectorize split shrinks the workgroup from 256 to 64 threads. applyVectorize
// splits the TS-wide N_loc AxisLocal into an outer AxisLocal of size TS/W plus an
// AxisVectorize inner of size W; the lowerer derives workgroup_size from AxisLocal
// range sizes only (the AxisVectorize inner is excluded), so workgroup_size.x
// drops 16 -> 4. Same output tile, same total ALU, 1/4 the threads -> the GPU
// core loses latency-hiding across the smem-barrier-bounded k-loop, and the
// "vec4" lowers to 4 scalar FMAs on Metal so nothing is bought back (SPEC §
// large-matmul). This is GPU-free (render-time only): it asserts the geometry,
// not the timing. A future "fix" that keeps OptVectorize while restoring 256
// threads must update this test deliberately. The durable win is OptVec4Load on
// the b3 stack (real 128-bit loads, 256 threads), not OptVectorize.
func TestB37_GeometryRegression_WorkgroupShrink(t *testing.T) {
	mk := func() schedule.ExecItem {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		return schedule.CreateSchedule(makeSink(a, C), "webgpu")[0]
	}

	const TS, MR, NR, W = 16, 4, 4, 4
	b3 := codegen.RenderWGSL(codegen.ApplyOpts(mk(), b3Opts(TS, MR, NR))).LocalSize
	b37 := codegen.RenderWGSL(codegen.ApplyOpts(mk(), b37Opts(TS, MR, NR, W))).LocalSize

	if b3 != [3]int{TS, TS, 1} {
		t.Errorf("b3 LocalSize = %v, want [%d %d 1] (256 threads)", b3, TS, TS)
	}
	if b37 != [3]int{TS / W, TS, 1} {
		t.Errorf("b37 LocalSize = %v, want [%d %d 1] (64 threads — the regression)", b37, TS/W, TS)
	}
	b3Threads := b3[0] * b3[1] * b3[2]
	b37Threads := b37[0] * b37[1] * b37[2]
	if b37Threads*4 != b3Threads {
		t.Errorf("expected b37 to have 1/4 of b3's threads (occupancy collapse); got b3=%d b37=%d", b3Threads, b37Threads)
	}
}

// TestB37_Timing_Matmul_Vectorize reports Min-of-N µs and GFLOP/s for
// default vs (OptTile+OptUpcast+OptVectorize) at 512³, 1024³, 2048³, 4096³.
// Acceptance grade: ≥1.5x at 2048³ (≥125 GFLOP/s). Below 1.5x is the honest
// finding that we are at the scalar-WGSL throughput ceiling; not a retune target.
func TestB37_Timing_Matmul_Vectorize(t *testing.T) {
	dev := requireDevice(t)
	skipIfSoftwareGPU(t, dev)

	const (
		warmup = 2
		iters  = 5
		TS     = 16
		MR, NR = 4, 4
		W      = 4
	)

	for _, N := range []int64{512, 1024, 2048, 4096} {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		base := schedule.CreateSchedule(makeSink(a, C), "webgpu")[0]

		resDef, err := dev.Benchmark(base, warmup, iters)
		if err != nil {
			t.Fatalf("default benchmark N=%d: %v", N, err)
		}
		gflopsDef := (2.0 * float64(N*N*N)) / (resDef.MinMicros * 1e3)
		fmt.Printf("Matmul %d³ (Default):                   Min=%8.2fµs  %7.2f GFLOP/s\n",
			N, resDef.MinMicros, gflopsDef)

		itemVec := codegen.ApplyOpts(base, b37Opts(TS, MR, NR, W))
		resVec, err := dev.Benchmark(itemVec, warmup, iters)
		if err != nil {
			t.Fatalf("vectorize benchmark N=%d: %v", N, err)
		}
		gflopsVec := (2.0 * float64(N*N*N)) / (resVec.MinMicros * 1e3)
		speedup := gflopsVec / gflopsDef
		fmt.Printf("Matmul %d³ (OptTile+OptUpcast+OptVec):  Min=%8.2fµs  %7.2f GFLOP/s  %.2fx\n",
			N, resVec.MinMicros, gflopsVec, speedup)

		if N == 2048 {
			if speedup < 1.5 {
				fmt.Printf("  [FINDING] 2048³ speedup %.2fx < 1.5x — expected: OptVectorize shrinks the workgroup 256->64 threads (occupancy collapse, see TestB37_GeometryRegression_WorkgroupShrink). Use OptVec4Load on the b3 stack instead.\n", speedup)
			} else {
				fmt.Printf("  [PASS] 2048³ speedup %.2fx >= 1.5x target\n", speedup)
			}
		}
	}
}
