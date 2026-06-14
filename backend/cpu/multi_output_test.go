package cpu_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestMultiOutputAttribution_IsomorphicKernels is the regression for the
// multi-output scramble bug: a single Realize over N>=3 INDEPENDENT but
// structurally isomorphic non-leaf tensors must assign each tensor its OWN
// result, not a sibling's.
//
// Each output is a [2,2]=[2,3]@[3,2] matmul — identical shape AND identical
// graph structure across all N, so the kernels are genuinely isomorphic and
// CANNOT be disambiguated by shape or structure. Only the input DATA differs.
// The old positional-zip assignOutputs matched final-output buffers to tensors
// by structural-key (schedule) order, which does not track caller order, so it
// silently handed each tensor a sibling's buffer.
//
// With the by-node-identity fix (CreateScheduleWithOutputs → assignOutputs),
// tensor[i] resolves to the buffer the scheduler attributed to ITS sink src.
func TestMultiOutputAttribution_IsomorphicKernels(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	const N = 4
	a := uop.NewArena(4096)

	// Distinct per-output input data so a swapped buffer is detectable.
	xData := make([][]float32, N)
	wData := make([][]float32, N)
	outs := make([]*tensor.Tensor, N)
	for k := 0; k < N; k++ {
		base := float32(k+1) * 10
		// x: [2,3]
		xData[k] = []float32{
			base + 1, base + 2, base + 3,
			base + 4, base + 5, base + 6,
		}
		// w: [3,2]
		wData[k] = []float32{
			base + 1, base + 2,
			base + 3, base + 4,
			base + 5, base + 6,
		}
		x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
		x.SetData(xData[k])
		w := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "cpu")
		w.SetData(wData[k])
		outs[k] = x.Matmul(w)
	}

	// Single batched Realize over all N isomorphic outputs.
	if err := tensor.Realize(outs...); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	// Reference: compute each [2,2] matmul independently in Go.
	ref := func(x, w []float32) []float32 {
		out := make([]float32, 4)
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				var s float32
				for p := 0; p < 3; p++ {
					s += x[i*3+p] * w[p*2+j]
				}
				out[i*2+j] = s
			}
		}
		return out
	}

	for k := 0; k < N; k++ {
		want := ref(xData[k], wData[k])
		got := outs[k].Data()
		if len(got) != len(want) {
			t.Fatalf("output %d: len got %d want %d", k, len(got), len(want))
		}
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-4 {
				t.Errorf("output %d idx %d: got %f want %f (diff %g) — buffer attribution scramble",
					k, i, got[i], want[i], d)
			}
		}
	}
}

// TestMultiOutputAttribution_DuplicateAndLeaf covers two edge cases of the
// by-identity attribution: the same non-leaf tensor passed twice (both copies
// must receive the correct result) and a leaf tensor mixed into the batch
// (its caller-provided data must be left untouched).
func TestMultiOutputAttribution_DuplicateAndLeaf(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer dev.Close()
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(4096)

	xA := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	xA.SetData([]float32{1, 2, 3, 4})
	yA := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	yA.SetData([]float32{10, 20, 30, 40})
	outA := xA.Add(yA) // {11,22,33,44}

	xB := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	xB.SetData([]float32{5, 6, 7, 8})
	yB := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	yB.SetData([]float32{100, 200, 300, 400})
	outB := xB.Add(yB) // {105,206,307,408}

	leaf := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	leaf.SetData([]float32{-1, -2, -3, -4})

	// Pass outA twice, plus outB, plus a bare leaf.
	if err := tensor.Realize(outA, leaf, outB, outA); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	checkEq := func(name string, got, want []float32) {
		t.Helper()
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-4 {
				t.Errorf("%s idx %d: got %f want %f", name, i, got[i], want[i])
			}
		}
	}
	checkEq("outA", outA.Data(), []float32{11, 22, 33, 44})
	checkEq("outB", outB.Data(), []float32{105, 206, 307, 408})
	checkEq("leaf", leaf.Data(), []float32{-1, -2, -3, -4})
}
