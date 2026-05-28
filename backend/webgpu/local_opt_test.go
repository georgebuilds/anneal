package webgpu_test

import (
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func makeSink(a *uop.Arena, t *tensor.Tensor) uop.UOp {
	return a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{t.Node()}, nil, nil)
}

// TestB1_ValueOracle_LocalOpt verifies that applying OptLocal to matmul, conv,
// and MLP produces results identical to the default 1D path.
func TestB1_ValueOracle_LocalOpt(t *testing.T) {
	dev := requireDevice(t)

	tests := []struct {
		name string
		fn   func(a *uop.Arena) *tensor.Tensor
		opts []codegen.Opt
	}{
		{
			name: "matmul_8x8_local8",
			fn: func(a *uop.Arena) *tensor.Tensor {
				A := tensor.NewLeaf(a, []int64{8, 8}, uop.Dtypes.Float32, "webgpu")
				B := tensor.NewLeaf(a, []int64{8, 8}, uop.Dtypes.Float32, "webgpu")
				A.SetData(uniformData(64, 1))
				B.SetData(uniformData(64, 2))
				return A.Matmul(B)
			},
			opts: []codegen.Opt{{Kind: codegen.OptLocal, Axis: 0, Arg: 8}},
		},
		{
			name: "matmul_16x16_local4x4",
			fn: func(a *uop.Arena) *tensor.Tensor {
				A := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
				B := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
				A.SetData(uniformData(256, 3))
				B.SetData(uniformData(256, 4))
				return A.Matmul(B)
			},
			opts: []codegen.Opt{
				{Kind: codegen.OptLocal, Axis: 0, Arg: 4},
				{Kind: codegen.OptLocal, Axis: 1, Arg: 4},
			},
		},
		{
			name: "mlp_fwd_bwd_grad",
			fn: func(a *uop.Arena) *tensor.Tensor {
				// [4, 4] @ [4, 8] -> [4, 8] -> sum -> scalar
				x := tensor.NewLeaf(a, []int64{4, 4}, uop.Dtypes.Float32, "webgpu")
				w := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
				x.SetData(uniformData(16, 5))
				w.SetData(uniformData(32, 6))
				pred := x.Matmul(w)
				loss := pred.Sum(nil, false)
				grads := tensor.Backward(loss, []*tensor.Tensor{w})
				return grads[w]
			},
			opts: []codegen.Opt{{Kind: codegen.OptLocal, Axis: 0, Arg: 4}},
		},
		{
			name: "conv_fwd",
			fn: func(a *uop.Arena) *tensor.Tensor {
				// [1, 1, 8, 8] conv [1, 1, 3, 3] -> [1, 1, 6, 6]
				x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
				x.SetData(uniformData(64, 7))
				conv := nn.NewConv2d(a, 1, 1, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "webgpu")
				conv.Weight.Value = uniformData(9, 8)
				conv.Weight.Load(a)
				return conv.Forward(x)
			},
			opts: []codegen.Opt{
				{Kind: codegen.OptLocal, Axis: 0, Arg: 2},
				{Kind: codegen.OptLocal, Axis: 1, Arg: 2},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(65536)
			out := tc.fn(a)

			// 1. Run default (1D)
			itemsDef := schedule.CreateSchedule(makeSink(a, out), "webgpu")
			resDef, err := dev.Run(itemsDef, nil)
			if err != nil {
				t.Fatalf("Default run failed: %v", err)
			}
			gotDef := resDef[out.Node().Index()]

			// 2. Run with OptLocal. Use a fresh arena to avoid schedule cache hit
			// which would return items with zeroed Ast.
			a2 := uop.NewArena(65536)
			out2 := tc.fn(a2)
			itemsOpt := schedule.CreateSchedule(makeSink(a2, out2), "webgpu")
			for i := range itemsOpt {
				itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], tc.opts).Ast
			}
			resOpt, err := dev.Run(itemsOpt, nil)
			if err != nil {
				t.Fatalf("Opt run failed: %v", err)
			}
			gotOpt := resOpt[out2.Node().Index()]

			if !approxEq(gotOpt, gotDef, 1e-5) {
				t.Errorf("Value mismatch with OptLocal!\nDef[0:4]: %v\nOpt[0:4]: %v", gotDef[:4], gotOpt[:4])
			}
		})
	}
}

// TestB1_Scalability_LargeGrid verifies that a kernel exceeding the 65535 workgroup
// limit (due to large output) now runs correctly via dimension spreading.
func TestB1_Scalability_LargeGrid(t *testing.T) {
	dev := requireDevice(t)

	// Workgroup size is 64 by default.
	// 65536 workgroups * 64 threads = 4,194,304 elements.
	// Let's use 8,000,000 elements to force spreading.
	N := int64(8000000)
	a := uop.NewArena(1024) // tiny arena, we just need the nodes
	x := tensor.NewLeaf(a, []int64{N}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()

	items := schedule.CreateSchedule(makeSink(a, y), "webgpu")
	if len(items) != 1 {
		t.Fatalf("Expected 1 item, got %d", len(items))
	}
	item := items[0]

	// Verify spreading in lowerer
	_, _, wc := codegen.Lower(item)
	t.Logf("Dispatch grid for N=%d: %v", N, wc)
	if wc[0] > 65535 {
		t.Errorf("Spreading failed: wc[0]=%d exceeds 65535", wc[0])
	}
	if wc[1] <= 1 && N > 65535*64 {
		t.Errorf("Spreading failed: wc[1]=%d for N=%d", wc[1], N)
	}

	tensor.DefaultExecutor = dev
	data := make([]float32, N)
	for i := range data {
		data[i] = 1.0
	}

	// Actually run a smaller but still spreading-triggering size if memory allows.
	x.SetData(data)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize failed: %v", err)
	}
	got := y.Data()
	if len(got) != int(N) {
		t.Fatalf("Output size mismatch: got %d, want %d", len(got), N)
	}
	// Check a few values across the grid
	indices := []int{0, 1000000, 4194304, 7999999}
	for _, idx := range indices {
		if got[idx] != 2.0 {
			t.Errorf("Value mismatch at index %d: got %f, want 2.0", idx, got[idx])
		}
	}
}

// TestB1_Timing_Matmul_Local verifies timing for a compute-dominated matmul.
func TestB1_Timing_Matmul_Local(t *testing.T) {
	dev := requireDevice(t)

	// Acceptance requirement: >= 512^3 matmul
	N := int64(512)
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
	t.Logf("Matmul %dx%d (Default 1D): Min=%0.2fµs", N, N, resDef.MinMicros)

	// 2. Benchmark with OptLocal (8,8)
	itemOpt := item
	itemOpt.Ast = codegen.ApplyOpts(item, []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 8},
		{Kind: codegen.OptLocal, Axis: 1, Arg: 8},
	}).Ast
	resOpt, err := dev.Benchmark(itemOpt, 2, 5)
	if err != nil {
		t.Fatalf("Opt benchmark failed: %v", err)
	}
	t.Logf("Matmul %dx%d (OptLocal 8x8): Min=%0.2fµs", N, N, resOpt.MinMicros)

	t.Logf("B1 Timing Report: Default vs Local dispatch. (Expect neutral-ish until tiling in B2)")
}

func uniformData(n int, seed float32) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(i) * 0.1 + seed
	}
	return d
}
