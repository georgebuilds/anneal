package tensor_test

// Regression test for the OpContiguous gradient rule. Contiguous() is a
// value-identity materialization barrier used to break kernel fusion (e.g. to
// satisfy the WebGPU 8-buffer-per-kernel cap). Its gradient is the identity;
// before the rule was added, Backward dead-ended at any Contiguous() on a
// gradient path and silently dropped the upstream gradient.

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestContiguousGradientPassesThrough(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	xData := []float32{0.5, -1.5, 2.0, 0.25}

	// loss = sum((x*x).Contiguous())  ->  d/dx = 2x
	grad := func(contig bool) []float32 {
		a := uop.NewArena(1 << 14)
		x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
		x.SetData(append([]float32(nil), xData...))
		sq := x.Mul(x)
		if contig {
			sq = sq.Contiguous()
		}
		loss := sq.Sum(nil, false)
		g := tensor.Backward(loss, []*tensor.Tensor{x})[x]
		if g == nil {
			t.Fatalf("no gradient (contig=%v) — Contiguous acted as a gradient barrier", contig)
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("realize: %v", err)
		}
		return append([]float32(nil), g.Data()...)
	}

	withC := grad(true)
	for i, v := range xData {
		want := 2 * v
		if math.Abs(float64(withC[i]-want)) > 1e-5 {
			t.Errorf("grad through Contiguous at [%d]: got %.5f, want %.5f (2x)", i, withC[i], want)
		}
	}
	// Contiguous must be gradient-transparent: identical to the non-contiguous graph.
	without := grad(false)
	for i := range withC {
		if math.Abs(float64(withC[i]-without[i])) > 1e-6 {
			t.Errorf("Contiguous changed the gradient at [%d]: %.6f vs %.6f", i, withC[i], without[i])
		}
	}
}
