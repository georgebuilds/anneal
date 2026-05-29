package webgpu_test

// Re-baseline experiment: >L2 matmul sizes on M3 GPU.
// M3 GPU L2 ≈ 12 MB; per-operand footprint: 512³=1MB, 1024³=4MB, 2048³=16MB, 4096³=64MB.
// Sizes 2048+ exceed L2 and expose true DRAM-bandwidth-limited behavior.
//
// Configs per size:
//   (a) Default 1D — no opts
//   (b) OptLocal alone — workgroup_size 16×16
//   (c) OptTile only — TS=16, no upcast
//   (d) OptTile + OptUpcast — TS=16, MR=NR=4 (B3 production config)
//
// A failure (resource limit, dispatch overflow) at any cell is reported as a
// finding without poisoning adjacent rows.

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func rebaselineGFLOPS(N int64, minMicros float64) float64 {
	return (2.0 * float64(N*N*N)) / (minMicros * 1e3)
}

// TestRebaseline_LargeMatmul measures 4 opt configs × 5 sizes.
func TestRebaseline_LargeMatmul(t *testing.T) {
	dev := requireDevice(t)

	const (
		warmup = 2
		iters  = 5
		TS     = 16
		MR, NR = 4, 4
	)

	type cell struct {
		minUs  float64
		gflops float64
		err    string
	}

	sizes := []int64{512, 1024, 2048, 4096}
	cfgNames := [4]string{
		"(a) Default1D   ",
		"(b) OptLocal    ",
		"(c) OptTileOnly ",
		"(d) OptTile+Upc ",
	}

	results := [4][4]cell{} // [sizeIdx][cfgIdx]

	for si, N := range sizes {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		C := A.Matmul(B)
		base := schedule.CreateSchedule(makeSink(a, C), "webgpu")[0]

		bench := func(item schedule.ExecItem) cell {
			res, err := dev.Benchmark(item, warmup, iters)
			if err != nil {
				return cell{err: err.Error()}
			}
			return cell{minUs: res.MinMicros, gflops: rebaselineGFLOPS(N, res.MinMicros)}
		}

		// (a) Default 1D
		results[si][0] = bench(base)

		// (b) OptLocal 16×16
		{
			item := base
			item.Ast = codegen.ApplyOpts(base, []codegen.Opt{
				{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
				{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
			}).Ast
			results[si][1] = bench(item)
		}

		// (c) OptTile only (TS=16)
		{
			item := base
			item.Ast = codegen.ApplyOpts(base, []codegen.Opt{
				{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
				{Kind: codegen.OptLocal, Axis: 0, Arg: TS},
				{Kind: codegen.OptTile, Axis: 0, Arg: TS},
			}).Ast
			results[si][2] = bench(item)
		}

		// (d) OptTile + OptUpcast TS=16 MR=NR=4
		{
			item := base
			item.Ast = codegen.ApplyOpts(base, b3Opts(TS, MR, NR)).Ast
			results[si][3] = bench(item)
		}
	}

	// Print table
	fmt.Printf("\n=== RE-BASELINE TABLE — M3 GPU, >L2 working-set sizes ===\n")
	fmt.Printf("L2 ≈ 12MB; per-operand: 512³=1MB 1024³=4MB 2048³=16MB 4096³=64MB\n")
	fmt.Printf("warmup=%d  iters=%d  TS=%d  MR=%d  NR=%d\n\n", warmup, iters, TS, MR, NR)
	fmt.Printf("%-18s  %6s  %10s  %8s  %6s\n", "Config", "N", "Min µs", "GFLOP/s", "vs(a)")
	fmt.Printf("%-18s  %6s  %10s  %8s  %6s\n",
		"------------------", "------", "----------", "--------", "------")

	for si, N := range sizes {
		def := results[si][0]
		for ci, name := range cfgNames {
			c := results[si][ci]
			if c.err != "" {
				fmt.Printf("%-18s  %6d  FAILED: %s\n", name, N, c.err)
				continue
			}
			speedup := "  —   "
			if ci > 0 && def.err == "" {
				speedup = fmt.Sprintf("%6.2fx", c.gflops/def.gflops)
			}
			fmt.Printf("%-18s  %6d  %10.2f  %8.2f  %s\n",
				name, N, c.minUs, c.gflops, speedup)
		}
		fmt.Println()
	}
}
