package tensor_test

import (
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// shapeSintsConcrete extracts an []int64 from a fully concrete Sint slice.
// Panics if any dim is symbolic; intended for shape-equality assertions in
// fixtures with no symbolic dims.
func shapeSintsConcrete(t *testing.T, sh []shape.Sint) []int64 {
	t.Helper()
	out := make([]int64, len(sh))
	for i, s := range sh {
		v, ok := s.ConstValue()
		if !ok {
			t.Fatalf("shape dim %d is symbolic, expected concrete", i)
		}
		out[i] = v
	}
	return out
}

// ── Forward shape fixtures (design doc §9) ───────────────────────────────────

// Fixture 1: W=[8,4] gathered along axis 0 by idx=[3] -> [3,4].
func TestGatherShape_Axis0_Rank2(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "cpu")

	out := w.Gather(0, idx)
	assertOp(t, out, uop.OpGather)
	shapeEq(t, out.Shape(), []int64{3, 4})
	if out.DType() != uop.Dtypes.Float32 {
		t.Fatalf("dtype: got %s want Float32", out.DType())
	}
	if out.Node().Arg().(int64) != 0 {
		t.Fatalf("OpGather arg: got %v want int64(0)", out.Node().Arg())
	}
	if out.Node().NSrc() != 2 {
		t.Fatalf("OpGather should have 2 srcs (data, index); got %d", out.Node().NSrc())
	}
	if out.Node().Src(0) != w.Node() {
		t.Fatalf("OpGather src[0] should be the data buffer")
	}
}

// Fixture 2: W=[6,6] gathered along axis 1 by idx=[5,2] -> [6,5,2].
// Torch-gather: output shape = data shape with dim replaced by index shape.
func TestGatherShape_Axis1_Rank2_MultiDimIndex(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{6, 6}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{5, 2}, uop.Dtypes.Int32, "cpu")

	out := w.Gather(1, idx)
	shapeEq(t, out.Shape(), []int64{6, 5, 2})
	if out.Node().Arg().(int64) != 1 {
		t.Fatalf("OpGather arg: got %v want int64(1)", out.Node().Arg())
	}
}

// Fixture 3: GPT-2 embedding W=[50257,768], idx=[16] -> [16,768].
func TestGatherShape_GPT2Embedding(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{50257, 768}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Int32, "cpu")

	out := w.Gather(0, idx)
	shapeEq(t, out.Shape(), []int64{16, 768})
}

// Fixture 4: symbolic-batch index: W=[50257,768], idx=[n] with n in [1,32].
// Output shape should carry the symbolic dim through.
func TestGatherShape_SymbolicBatchIndex(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{50257, 768}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewSymbolicInput(a, "n", 1, 32, uop.Dtypes.Int32, "cpu")

	out := w.Gather(0, idx)
	outSh := out.ShapeSints()
	if len(outSh) != 2 {
		t.Fatalf("expected rank 2 output, got rank %d", len(outSh))
	}
	if _, ok := outSh[0].ConstValue(); ok {
		t.Fatalf("output dim 0 should be symbolic (carried from index)")
	}
	v, ok := outSh[1].ConstValue()
	if !ok || v != 768 {
		t.Fatalf("output dim 1: got %v ok=%v want 768", v, ok)
	}
}

// Fixture 5: every idx value identical (purely structural for forward;
// exercises the shape rule without any de-dup/sort).
func TestGatherShape_AdversarialDuplicates(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{5}, uop.Dtypes.Int32, "cpu") // all-2s in practice

	out := w.Gather(0, idx)
	shapeEq(t, out.Shape(), []int64{5, 4})
}

// ── Frontend edge cases ──────────────────────────────────────────────────────

func TestGather_NegativeDimWraps(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{6, 6}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{2}, uop.Dtypes.Int32, "cpu")

	out := w.Gather(-1, idx) // -1 wraps to dim 1
	shapeEq(t, out.Shape(), []int64{6, 2})
	if out.Node().Arg().(int64) != 1 {
		t.Fatalf("OpGather arg: got %v want int64(1)", out.Node().Arg())
	}
}

func TestGather_DimOutOfRangePanics(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{2}, uop.Dtypes.Int32, "cpu")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on dim=2 with rank=1; got none")
		}
	}()
	_ = w.Gather(2, idx)
}

func TestGather_RankZeroPanics(t *testing.T) {
	a := newArena()
	s := tensor.ConstScalar(a, 1.0, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Int32, "cpu")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic gathering on a scalar tensor; got none")
		}
	}()
	_ = s.Gather(0, idx)
}

func TestGather_FloatIndexPanics(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{4, 3}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{2}, uop.Dtypes.Float32, "cpu")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic for float-dtype index; got none")
		}
	}()
	_ = w.Gather(0, idx)
}

// Int64 indices should be auto-cast to Int32; the output graph should carry
// a Cast under the OpGather's index src.
func TestGather_Int64IndexCastToInt32(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int64, "cpu")

	out := w.Gather(0, idx)
	shapeEq(t, out.Shape(), []int64{3, 4})
	idxSrc := out.Node().Src(1)
	if idxSrc.Op() != uop.OpCast {
		t.Fatalf("expected Int64 index to be wrapped in OpCast; got %s", idxSrc.Op())
	}
	if idxSrc.DType() != uop.Dtypes.Int32 {
		t.Fatalf("cast target dtype: got %s want Int32", idxSrc.DType())
	}
}

// ── ShapeSints helper exercise ───────────────────────────────────────────────

func TestGatherShapeSints_ConcreteRoundtrip(t *testing.T) {
	a := newArena()
	w := tensor.NewLeaf(a, []int64{8, 4}, uop.Dtypes.Float32, "cpu")
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "cpu")
	out := w.Gather(0, idx)

	got := shapeSintsConcrete(t, out.ShapeSints())
	shapeEq(t, got, []int64{3, 4})
}

// Slice C lands forward lowering; the Slice B NYI panic has been removed.
// Realize-side correctness is exercised in gather_realize_test.go (GPU).
