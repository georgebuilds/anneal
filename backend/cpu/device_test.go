package cpu_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestOpenClose(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	dev.Close()
}

// TestSimpleAddMul exercises the elementwise + reduce paths end-to-end via
// the tensor.Realize → schedule → cpu.Run pipeline.
func TestSimpleAddMul(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y.SetData([]float32{10, 20, 30, 40})

	out := x.Add(y).Mul(y) // (x+y) * y
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	got := out.Data()
	want := []float32{(1 + 10) * 10, (2 + 20) * 20, (3 + 30) * 30, (4 + 40) * 40}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-5 {
			t.Errorf("idx %d: got %f want %f (diff %g)", i, got[i], want[i], d)
		}
	}
}

// TestSumReduce exercises an OpReduce(Add) inside the body.
func TestSumReduce(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4, 5, 6})
	out := x.Sum(nil, false)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	got := out.Data()[0]
	want := float32(1 + 2 + 3 + 4 + 5 + 6)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("sum: got %f want %f", got, want)
	}
}

// TestMatmul covers the matmul reduce path the MLP relies on.
func TestMatmul(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4, 5, 6})
	w := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "cpu")
	w.SetData([]float32{1, 2, 3, 4, 5, 6})

	out := x.Matmul(w)
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	got := out.Data()
	// [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
	want := []float32{22, 28, 49, 64}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Errorf("idx %d: got %f want %f", i, got[i], want[i])
		}
	}
}
