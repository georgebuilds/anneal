package webgpu_test

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestB2_ValueOracle_TiledMatmul(t *testing.T) {
	dev := requireDevice(t)

	tests := []struct {
		name string
		M, N, K int64
		TS int
	}{
		{"matmul_16x16x16_TS8", 16, 16, 16, 8},
		{"matmul_16x16x16_TS16", 16, 16, 16, 16},
		{"matmul_32x32x32_TS16", 32, 32, 32, 16},
		{"matmul_irregular_TS16", 31, 33, 35, 16},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(65536)
			A := tensor.NewLeaf(a, []int64{tc.M, tc.K}, uop.Dtypes.Float32, "webgpu")
			B := tensor.NewLeaf(a, []int64{tc.K, tc.N}, uop.Dtypes.Float32, "webgpu")
			A.SetData(uniformData(int(tc.M*tc.K), 1))
			B.SetData(uniformData(int(tc.K*tc.N), 2))
			out := A.Matmul(B)

			// 1. Run default (1D)
			itemsDef := schedule.CreateSchedule(makeSink(a, out), "webgpu")
			resDef, err := dev.Run(itemsDef, nil)
			if err != nil {
				t.Fatalf("Default run failed: %v", err)
			}
			gotDef := resDef[out.Node().Index()]

			// 2. Run with OptTile + OptLocal
			a2 := uop.NewArena(65536)
			A2 := tensor.NewLeaf(a2, []int64{tc.M, tc.K}, uop.Dtypes.Float32, "webgpu")
			B2 := tensor.NewLeaf(a2, []int64{tc.K, tc.N}, uop.Dtypes.Float32, "webgpu")
			A2.SetData(uniformData(int(tc.M*tc.K), 1))
			B2.SetData(uniformData(int(tc.K*tc.N), 2))
			out2 := A2.Matmul(B2)
			
			itemsOpt := schedule.CreateSchedule(makeSink(a2, out2), "webgpu")
			for i := range itemsOpt {
				// Matmul kernel is usually the one with the Reduce
				itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], []codegen.Opt{
					{Kind: codegen.OptLocal, Axis: 0, Arg: tc.TS}, // M -> [M_wg, M_loc, N]
					{Kind: codegen.OptLocal, Axis: 0, Arg: tc.TS}, // N -> [M_wg, M_loc, N_wg, N_loc]
					{Kind: codegen.OptTile, Axis: 0, Arg: tc.TS},  // K (reduction axis 0)
				}).Ast
			}
			resOpt, err := dev.Run(itemsOpt, nil)
			if err != nil {
				t.Fatalf("Opt run failed: %v", err)
			}
			gotOpt := resOpt[out2.Node().Index()]

			if !approxEq(gotOpt, gotDef, 1e-5) {
				t.Errorf("Value mismatch with OptTile!\nDef[0:4]: %v\nOpt[0:4]: %v", gotDef[:4], gotOpt[:4])
			}
		})
	}
}

func TestB2_Timing_Matmul_Tiled(t *testing.T) {
	dev := requireDevice(t)

	for _, N := range []int64{512, 1024} {
		TS := 16
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		item := schedule.CreateSchedule(makeSink(a, C), "webgpu")[0]

		// 1. Benchmark default
		resDef, err := dev.Benchmark(item, 2, 5)
		if err != nil {
			t.Fatalf("Default benchmark failed: %v", err)
		}
		gflopsDef := (2.0 * float64(N*N*N)) / (resDef.MinMicros * 1e3)
		fmt.Printf("Matmul %dx%dx%d (Default 1D): Min=%0.2fµs (%0.2f GFLOP/s)\n", N, N, N, resDef.MinMicros, gflopsDef)

		// 2. Benchmark with OptTile
		itemOpt := item
		itemOpt.Ast = codegen.ApplyOpts(item, []codegen.Opt{
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			{Kind: codegen.OptTile, Axis: 0, Arg: TS},
		}).Ast
		resOpt, err := dev.Benchmark(itemOpt, 2, 5)
		if err != nil {
			t.Fatalf("Opt benchmark failed: %v", err)
		}
		gflopsOpt := (2.0 * float64(N*N*N)) / (resOpt.MinMicros * 1e3)
		fmt.Printf("Matmul %dx%dx%d (OptTile %d): Min=%0.2fµs (%0.2f GFLOP/s)\n", N, N, N, TS, resOpt.MinMicros, gflopsOpt)

		if gflopsOpt > gflopsDef {
			fmt.Printf("Speedup: %0.2fx\n", gflopsOpt/gflopsDef)
		} else {
			fmt.Printf("No speedup. (Opt: %0.2f GFLOP/s, Def: %0.2f GFLOP/s)\n", gflopsOpt, gflopsDef)
		}
		
		if N == 512 {
			fmt.Printf("WGSL for Tiled Matmul:\n%s\n", codegen.RenderWGSL(itemOpt).WGSL)
		}
	}
}
