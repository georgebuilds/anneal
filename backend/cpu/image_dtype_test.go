package cpu_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestCPU_ImageDType_RoundTrip verifies that an image-storage dtype tensor
// round-trips through the CPU backend's Realize path. The vec4 packing is a
// GPU buffer concept; on CPU we use a plain flat f32 host slice (allocator
// routes IsImage → f32 storage), so a copy-by-multiply-by-1 must reproduce
// the input bit-exactly for every shape including non-vec4-aligned ones.
func TestCPU_ImageDType_RoundTrip(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	cases := []struct {
		name  string
		shape []int64
		data  []float32
	}{
		{"flat_4_aligned", []int64{4}, []float32{1, 2, 3, 4}},
		{"flat_17_tail", []int64{17}, func() []float32 {
			d := make([]float32, 17)
			for i := range d {
				d[i] = float32(i) * 0.5
			}
			return d
		}()},
		{"matrix_3x5", []int64{3, 5}, func() []float32 {
			d := make([]float32, 15)
			for i := range d {
				d[i] = float32(i) - 7
			}
			return d
		}()},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(1024)
			x := tensor.NewLeaf(a, tc.shape, uop.Dtypes.ImageFloat32, "cpu")
			x.SetData(tc.data)
			one := tensor.ConstScalar(a, 1.0, uop.Dtypes.ImageFloat32, "cpu")
			out := x.Mul(one)
			if err := tensor.Realize(out); err != nil {
				t.Fatalf("Realize: %v", err)
			}
			got := out.Data()
			if len(got) != len(tc.data) {
				t.Fatalf("output length: got %d want %d", len(got), len(tc.data))
			}
			var maxDiff float64
			for i := range tc.data {
				d := math.Abs(float64(got[i] - tc.data[i]))
				if d > maxDiff {
					maxDiff = d
				}
			}
			t.Logf("%s: max-abs-diff=%g", tc.name, maxDiff)
			if maxDiff != 0 {
				t.Fatalf("CPU image round-trip not bit-exact: max-abs-diff=%g", maxDiff)
			}
		})
	}
}

// TestCPU_ImageDType_AllocatorFloatStorage verifies the CPU allocator
// routes image dtypes to the f32 backing slice (the vec4 packing is a GPU
// concept). A Realize that writes into the buffer and reads it back must
// produce identical values; if the allocator returned an i32 buffer or
// errored, every downstream Realize would fail.
func TestCPU_ImageDType_AllocatorFloatStorage(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.ImageFloat32, "cpu")
	x.SetData([]float32{0.5, -0.5, 1, -1, 2, -2, 3.14, -3.14})
	zero := tensor.ConstScalar(a, 0.0, uop.Dtypes.ImageFloat32, "cpu")
	out := x.Add(zero)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	got := out.Data()
	want := []float32{0.5, -0.5, 1, -1, 2, -2, 3.14, -3.14}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("idx %d: got %v want %v", i, got[i], want[i])
		}
	}
}
