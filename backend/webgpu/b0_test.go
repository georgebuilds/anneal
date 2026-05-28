package webgpu_test

import (
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestB0_ApplyOpt_Identity_ValueOracle(t *testing.T) {
	dev := requireDevice(t)
	a := uop.NewArena(1024 * 1024)

	// Test with a matmul kernel
	M, K, N := int64(32), int64(32), int64(32)
	x := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.Float32, "webgpu")
	w := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.Float32, "webgpu")
	y := x.Matmul(w)

	items := makeSchedule(t, "webgpu", y)
	item := items[0]

	inputs := map[uint32][]float32{
		x.Node().Index(): make([]float32, M*K),
		w.Node().Index(): make([]float32, K*N),
	}
	for i := range inputs[x.Node().Index()] {
		inputs[x.Node().Index()][i] = float32(i)
	}
	for i := range inputs[w.Node().Index()] {
		inputs[w.Node().Index()][i] = float32(i) * 0.1
	}

	// 1. Without Opts
	outputs1 := runSchedule(t, dev, items, inputs)
	got1 := firstFinalOutput(t, items, outputs1)

	// 2. With OptIdentity
	opt := codegen.Opt{Kind: codegen.OptIdentity}
	appliedItem := codegen.ApplyOpts(item, []codegen.Opt{opt})
	appliedItems := []schedule.ExecItem{appliedItem}

	outputs2 := runSchedule(t, dev, appliedItems, inputs)
	got2 := firstFinalOutput(t, appliedItems, outputs2)

	// Value Oracle: max-abs-diff == 0
	if !approxEq(got1, got2, 0) {
		t.Fatalf("ApplyOpt(Identity) round-trip failed: outputs differ")
	}

	// Test with an elementwise kernel
	a2 := uop.NewArena(1024 * 1024)
	x2 := tensor.NewLeaf(a2, []int64{1024}, uop.Dtypes.Float32, "webgpu")
	y2 := x2.Exp2().Log2() // should be identity-ish
	items2 := makeSchedule(t, "webgpu", y2)
	item2 := items2[0]

	inputs2 := map[uint32][]float32{
		x2.Node().Index(): make([]float32, 1024),
	}
	for i := range inputs2[x2.Node().Index()] {
		inputs2[x2.Node().Index()][i] = float32(i)
	}

	outputs3 := runSchedule(t, dev, items2, inputs2)
	got3 := firstFinalOutput(t, items2, outputs3)

	appliedItem2 := codegen.ApplyOpts(item2, []codegen.Opt{opt})
	appliedItems2 := []schedule.ExecItem{appliedItem2}
	outputs4 := runSchedule(t, dev, appliedItems2, inputs2)
	got4 := firstFinalOutput(t, appliedItems2, outputs4)

	if !approxEq(got3, got4, 0) {
		t.Fatalf("ApplyOpt(Identity) elementwise failed: outputs differ")
	}
}

func TestB0_TimingHarness_Stability(t *testing.T) {
	dev := requireDevice(t)

	runBench := func(M, K, N int64) {
		a := uop.NewArena(1024 * 1024)
		x := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.Float32, "webgpu")
		w := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.Float32, "webgpu")
		y := x.Matmul(w)

		items := makeSchedule(t, "webgpu", y)
		item := items[0]

		res, err := dev.Benchmark(item, 10, 50)
		if err != nil {
			t.Fatalf("Benchmark failed: %v", err)
		}
		t.Logf("Matmul %dx%dx%d: min=%.2fµs, median=%.2fµs, max=%.2fµs, cv=%.4f",
			M, K, N, res.MinMicros, res.MedianMicros, res.MaxMicros, res.CV)

		if M >= 512 && res.CV > 0.05 {
			t.Errorf("Matmul %dx%dx%d: CV %.4f exceeds 5%% target", M, K, N, res.CV)
		}
	}

	runBench(64, 64, 64)
	runBench(512, 512, 512)
	runBench(1024, 1024, 1024)
}
