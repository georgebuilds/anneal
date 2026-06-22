package tensor

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// Regression for the keepdim-reduce backward shape bug. A keepdim=true reduce
// whose output feeds another reduce was mis-derived by shapeOfNode (the kept
// size-1 dims were dropped, e.g. [2,1,1,1] -> [2]), so the outer reduce's gradient
// indexed its axis list out of range and panicked. ReduceArg now records keepdim
// and shapeOfNode honors it. This builds the backward graph (where the panic was)
// and checks the gradient has the input's shape.
func TestKeepdimReduceBackwardShape(t *testing.T) {
	a := uop.NewArena(1 << 20)
	x := NewLeaf(a, []int64{2, 3, 4, 4}, uop.Dtypes.Float32, "cpu")
	x.SetData(make([]float32, 2*3*4*4))

	m := x.Mul(x).Mean([]int{1, 2, 3}, true) // [2,1,1,1] keepdim output
	loss := m.Mean(nil, false)               // scalar; outer reduce over the kept dims

	grads := Backward(loss, []*Tensor{x})
	g := grads[x]
	if g == nil {
		t.Fatal("no gradient produced for x")
	}
	got := g.Shape()
	if len(got) != 4 || got[0] != 2 || got[1] != 3 || got[2] != 4 || got[3] != 4 {
		t.Fatalf("gradient shape %v, want [2 3 4 4]", got)
	}
}

// TestReduceKeepdimDistinctInterning confirms keepdim=true and keepdim=false
// produce different nodes with the right shapes. keepdim=true is a dropped reduce
// plus an explicit reshape, so its root node is the reshape (not the reduce).
func TestReduceKeepdimDistinctInterning(t *testing.T) {
	a := uop.NewArena(1 << 16)
	x := NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	x.SetData(make([]float32, 6))
	keep := x.Sum([]int{1}, true)  // [2,1]
	drop := x.Sum([]int{1}, false) // [2]
	if keep.node.Index() == drop.node.Index() {
		t.Fatal("keepdim=true and keepdim=false reduces interned to the same node")
	}
	if ks := keep.Shape(); len(ks) != 2 || ks[1] != 1 {
		t.Fatalf("keepdim sum shape %v, want [2 1]", ks)
	}
	if ds := drop.Shape(); len(ds) != 1 || ds[0] != 2 {
		t.Fatalf("non-keepdim sum shape %v, want [2]", ds)
	}
}
