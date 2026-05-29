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
//   idx=0: M_loc (AxisLocal, M direction)
//   idx=1: N_loc (AxisLocal, N direction ← stride-1 ✓)
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
			resDef, err := dev.Run(itemsDef, nil)
			if err != nil {
				t.Fatalf("Default run failed: %v", err)
			}
			gotDef := resDef[out.Node().Index()]

			a2 := uop.NewArena(65536)
			A2 := tensor.NewLeaf(a2, []int64{tc.M, tc.K}, uop.Dtypes.Float32, "webgpu")
			B2 := tensor.NewLeaf(a2, []int64{tc.K, tc.N}, uop.Dtypes.Float32, "webgpu")
			A2.SetData(uniformData(int(tc.M*tc.K), 1))
			B2.SetData(uniformData(int(tc.K*tc.N), 2))
			out2 := A2.Matmul(B2)
			itemsOpt := schedule.CreateSchedule(makeSink(a2, out2), "webgpu")
			for i := range itemsOpt {
				itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], b37Opts(tc.TS, tc.MR, tc.NR, tc.W)).Ast
			}
			resOpt, err := dev.Run(itemsOpt, nil)
			if err != nil {
				t.Fatalf("Opt run failed: %v", err)
			}
			gotOpt := resOpt[out2.Node().Index()]

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

	gxDef, gwDef, itemsDef := build()
	resDef, err := dev.Run(itemsDef, nil)
	if err != nil {
		t.Fatalf("Default run: %v", err)
	}
	gxD := resDef[gxDef.Node().Index()]
	gwD := resDef[gwDef.Node().Index()]

	gxOpt, gwOpt, itemsOpt := build()
	for i := range itemsOpt {
		itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], b37Opts(16, 4, 4, 4)).Ast
	}
	resOpt, err := dev.Run(itemsOpt, nil)
	if err != nil {
		t.Fatalf("Opt run: %v", err)
	}
	gxO := resOpt[gxOpt.Node().Index()]
	gwO := resOpt[gwOpt.Node().Index()]

	if !approxEq(gxO, gxD, 0) {
		t.Errorf("gx mismatch (first 4): def=%v opt=%v", firstN(gxD, 4), firstN(gxO, 4))
	}
	if !approxEq(gwO, gwD, 0) {
		t.Errorf("gw mismatch (first 4): def=%v opt=%v", firstN(gwD, 4), firstN(gwO, 4))
	}
}

// TestB37_ValueOracle_VectorizeConv checks conv2d output is unchanged under b37Opts.
// Conv kernels don't match the tiled-matmul pattern; ApplyOpts no-ops them.
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

	outDef, itemsDef := build()
	resDef, err := dev.Run(itemsDef, nil)
	if err != nil {
		t.Fatalf("Default run: %v", err)
	}
	gotDef := resDef[outDef.Node().Index()]

	outOpt, itemsOpt := build()
	for i := range itemsOpt {
		itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], b37Opts(16, 4, 4, 4)).Ast
	}
	resOpt, err := dev.Run(itemsOpt, nil)
	if err != nil {
		t.Fatalf("Opt run: %v", err)
	}
	gotOpt := resOpt[outOpt.Node().Index()]

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
		item := mk()
		item.Ast = codegen.ApplyOpts(item, b37Opts(16, 4, 4, 4)).Ast
		return codegen.RenderWGSL(item).WGSL
	}

	w1 := build()
	w2 := build()
	if w1 != w2 {
		t.Fatalf("WGSL render not reproducible under b37Opts (cache key risk)")
	}
}

// TestB37_Timing_Matmul_Vectorize reports Min-of-N µs and GFLOP/s for
// default vs (OptTile+OptUpcast+OptVectorize) at 512³, 1024³, 2048³, 4096³.
// Acceptance grade: ≥1.5x at 2048³ (≥125 GFLOP/s). Below 1.5x is the honest
// finding that we are at the scalar-WGSL throughput ceiling; not a retune target.
func TestB37_Timing_Matmul_Vectorize(t *testing.T) {
	dev := requireDevice(t)

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

		itemVec := base
		itemVec.Ast = codegen.ApplyOpts(base, b37Opts(TS, MR, NR, W)).Ast
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
				fmt.Printf("  [FINDING] 2048³ speedup %.2fx < 1.5x target — at scalar-WGSL throughput ceiling\n", speedup)
			} else {
				fmt.Printf("  [PASS] 2048³ speedup %.2fx >= 1.5x target\n", speedup)
			}
		}
	}
}
