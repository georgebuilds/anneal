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

// b3Opts returns the canonical OptLocal*2 + OptTile + OptUpcast*2 pipeline used
// by B3 register-blocking tests. axis=0 then axis=1 on the upcasts because after
// the first OptUpcast the M-Workgroup outer gets an AxisUpcast partner and is
// skipped; the next eligible non-Reduce range is M-Local (idx 0), so N-Workgroup
// lands at idx 1.
func b3Opts(TS, MR, NR int) []codegen.Opt {
	return []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
		{Kind: codegen.OptTile, Axis: 0, Arg: TS},
		{Kind: codegen.OptUpcast, Axis: 0, Arg: MR}, // M_wg
		{Kind: codegen.OptUpcast, Axis: 1, Arg: NR}, // N_wg
	}
}

// TestB3_ValueOracle_UpcastMatmul checks bit-exact agreement of OptTile+OptUpcast
// against the default 1D path across regular and upcast-boundary-irregular shapes.
func TestB3_ValueOracle_UpcastMatmul(t *testing.T) {
	dev := requireDevice(t)

	tests := []struct {
		name       string
		M, N, K    int64
		TS, MR, NR int
	}{
		{"matmul_64x64x64_TS16_MR4_NR4", 64, 64, 64, 16, 4, 4},
		{"matmul_128x128x128_TS16_MR4_NR4", 128, 128, 128, 16, 4, 4},
		{"matmul_irregular_M17_TS16_MR4_NR4", 17, 32, 32, 16, 4, 4},
		{"matmul_irregular_N30_TS16_MR4_NR4", 32, 30, 32, 16, 4, 4},
		{"matmul_irregular_M17N30_TS16_MR4_NR4", 17, 30, 35, 16, 4, 4},
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
				itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], b3Opts(tc.TS, tc.MR, tc.NR)).Ast
			}
			resOpt, err := dev.Run(itemsOpt, nil)
			if err != nil {
				t.Fatalf("Opt run failed: %v", err)
			}
			gotOpt := resOpt[out2.Node().Index()]

			if len(gotOpt) != len(gotDef) {
				t.Fatalf("length mismatch: def=%d opt=%d", len(gotDef), len(gotOpt))
			}
			// B3 acceptance bar: max-abs-diff == 0 (bit-exact).
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

// TestB3_ValueOracle_UpcastMLPBackward checks that the MLP forward+backward
// gradient values are bit-exact under OptTile+OptUpcast. The matmul subkernels
// are the only ones affected; non-matmul kernels (Mean/Sub) return-unchanged
// from ApplyOpts because they have no AxisLoop ranges of the targeted index.
func TestB3_ValueOracle_UpcastMLPBackward(t *testing.T) {
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
		itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], b3Opts(16, 4, 4)).Ast
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

// TestB3_ValueOracle_UpcastConv checks that conv2d output is bit-exact under
// b3Opts. Conv kernels do not match the matmul OptTile pattern (their Reduce
// element node is not a plain Mul of two 2-arg Index nodes), so OptTile returns
// the sink unchanged; OptLocal and OptUpcast also no-op when they can't find
// eligible ranges. The point: the schedule must remain correct end-to-end.
func TestB3_ValueOracle_UpcastConv(t *testing.T) {
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
		itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], b3Opts(16, 4, 4)).Ast
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

// TestB3_ScheduleCache_HitCorrect verifies that re-applying b3Opts on the same
// kernel structure produces the same WGSL byte-for-byte (kernel cache key is
// content-derived). Slice-1 watch item: kernel structure changed again under
// OptUpcast, the cache key must reflect that.
func TestB3_ScheduleCache_HitCorrect(t *testing.T) {
	requireDevice(t) // ensures backend is wired even though we only render

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
		item.Ast = codegen.ApplyOpts(item, b3Opts(16, 4, 4)).Ast
		return codegen.RenderWGSL(item).WGSL
	}

	w1 := build()
	w2 := build()
	if w1 != w2 {
		t.Fatalf("WGSL render not reproducible under b3Opts (cache key risk)")
	}
}

// TestB3_Timing_Matmul_Upcast reports min-of-N µs and GFLOP/s for default vs
// (OptTile+OptUpcast) at 512³ and 1024³. Honest acceptance: ≥2× at 1024³.
func TestB3_Timing_Matmul_Upcast(t *testing.T) {
	dev := requireDevice(t)

	for _, N := range []int64{512, 1024} {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		item := schedule.CreateSchedule(makeSink(a, C), "webgpu")[0]

		resDef, err := dev.Benchmark(item, 2, 5)
		if err != nil {
			t.Fatalf("default benchmark: %v", err)
		}
		gflopsDef := (2.0 * float64(N*N*N)) / (resDef.MinMicros * 1e3)
		fmt.Printf("Matmul %d³ (Default 1D): Min=%0.2fµs (%0.2f GFLOP/s)\n",
			N, resDef.MinMicros, gflopsDef)

		itemOpt := item
		itemOpt.Ast = codegen.ApplyOpts(item, b3Opts(16, 4, 4)).Ast
		resOpt, err := dev.Benchmark(itemOpt, 2, 5)
		if err != nil {
			t.Fatalf("opt benchmark: %v", err)
		}
		gflopsOpt := (2.0 * float64(N*N*N)) / (resOpt.MinMicros * 1e3)
		fmt.Printf("Matmul %d³ (OptTile+OptUpcast TS=16 MR=4 NR=4): Min=%0.2fµs (%0.2f GFLOP/s)\n",
			N, resOpt.MinMicros, gflopsOpt)
		if gflopsOpt > gflopsDef {
			fmt.Printf("  Speedup: %0.2fx\n", gflopsOpt/gflopsDef)
		} else {
			fmt.Printf("  No speedup. (Opt=%0.2f, Def=%0.2f GFLOP/s)\n", gflopsOpt, gflopsDef)
		}
	}
}

func firstN(s []float32, n int) []float32 {
	if len(s) < n {
		return s
	}
	return s[:n]
}
