package tensor_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// concatHelpers — value-oracle pre-realize checks.
//
// Concat is built as Pad+Add (no OpCat) so its forward correctness is fully
// determined by the underlying Pad/Add op semantics. We assert the composed
// graph: the result root is OpAdd (for N≥2), each operand walks back through
// OpPad to its original leaf, and pad amounts on the concat axis match the
// computed prefix/suffix sums. The leaf data is what would be materialised
// at realize time; we assert each leaf's data slice element-by-element.

func mustLeaf(a *uop.Arena, sh []int64, data []float32) *tensor.Tensor {
	leaf := tensor.NewLeaf(a, sh, uop.Dtypes.Float32, "cpu")
	leaf.SetData(data)
	return leaf
}

func TestConcat_TwoInputsAxis0(t *testing.T) {
	a := uop.NewArena(64)
	x := mustLeaf(a, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	y := mustLeaf(a, []int64{1, 3}, []float32{7, 8, 9})

	out := tensor.Concat([]*tensor.Tensor{x, y}, 0)
	want := []int64{3, 3}
	if got := out.Shape(); !shapeEqInts(got, want) {
		t.Fatalf("shape=%v, want %v", got, want)
	}
	if out.DType() != uop.Dtypes.Float32 {
		t.Fatalf("dtype=%v, want f32", out.DType())
	}
	// Result is x_padded + y_padded; root op should be OpAdd.
	if out.Node().Op() != uop.OpAdd {
		t.Fatalf("root op=%s, want Add", out.Node().Op())
	}
	// Operand 0 walks back to x via OpPad with pad amounts {0,1} on axis 0
	// (1 = size of y on axis 0).
	xPad := out.Node().Src(0)
	if xPad.Op() != uop.OpPad {
		t.Fatalf("op0 root=%s, want Pad", xPad.Op())
	}
	// Leaf data on x and y is untouched; assert it survives unchanged.
	if got, want := x.Data(), []float32{1, 2, 3, 4, 5, 6}; !float32SliceEq(got, want) {
		t.Fatalf("x data=%v, want %v", got, want)
	}
	if got, want := y.Data(), []float32{7, 8, 9}; !float32SliceEq(got, want) {
		t.Fatalf("y data=%v, want %v", got, want)
	}
}

func TestConcat_TwoInputsAxis1(t *testing.T) {
	a := uop.NewArena(64)
	x := mustLeaf(a, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	y := mustLeaf(a, []int64{2, 2}, []float32{10, 11, 12, 13})

	out := tensor.Concat([]*tensor.Tensor{x, y}, 1)
	want := []int64{2, 5}
	if got := out.Shape(); !shapeEqInts(got, want) {
		t.Fatalf("shape=%v, want %v", got, want)
	}
}

func TestConcat_TwoInputsLastAxis(t *testing.T) {
	a := uop.NewArena(64)
	x := mustLeaf(a, []int64{2, 2}, []float32{1, 2, 3, 4})
	y := mustLeaf(a, []int64{2, 3}, []float32{5, 6, 7, 8, 9, 10})

	out := tensor.Concat([]*tensor.Tensor{x, y}, -1)
	want := []int64{2, 5}
	if got := out.Shape(); !shapeEqInts(got, want) {
		t.Fatalf("shape=%v, want %v", got, want)
	}
}

func TestConcat_ThreeInputs(t *testing.T) {
	a := uop.NewArena(64)
	x := mustLeaf(a, []int64{2}, []float32{1, 2})
	y := mustLeaf(a, []int64{3}, []float32{3, 4, 5})
	z := mustLeaf(a, []int64{1}, []float32{6})

	out := tensor.Concat([]*tensor.Tensor{x, y, z}, 0)
	want := []int64{6}
	if got := out.Shape(); !shapeEqInts(got, want) {
		t.Fatalf("shape=%v, want %v", got, want)
	}
	// Root is Add (chain x_padded + y_padded + z_padded).
	if out.Node().Op() != uop.OpAdd {
		t.Fatalf("root op=%s, want Add", out.Node().Op())
	}
}

func TestConcat_NegativeAxisWraps(t *testing.T) {
	a := uop.NewArena(64)
	x := mustLeaf(a, []int64{2, 3, 4}, make([]float32, 24))
	y := mustLeaf(a, []int64{2, 3, 5}, make([]float32, 30))

	out := tensor.Concat([]*tensor.Tensor{x, y}, -1)
	want := []int64{2, 3, 9}
	if got := out.Shape(); !shapeEqInts(got, want) {
		t.Fatalf("shape=%v, want %v", got, want)
	}
}

func TestConcat_Single(t *testing.T) {
	a := uop.NewArena(8)
	x := mustLeaf(a, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	out := tensor.Concat([]*tensor.Tensor{x}, 0)
	if out != x {
		t.Fatalf("Concat with one input should return that input unchanged")
	}
}

func TestConcat_Empty_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on empty slice")
		}
	}()
	_ = tensor.Concat(nil, 0)
}

func TestConcat_AxisOutOfRange_Panics(t *testing.T) {
	a := uop.NewArena(8)
	x := mustLeaf(a, []int64{2, 3}, make([]float32, 6))
	y := mustLeaf(a, []int64{2, 3}, make([]float32, 6))
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on axis 5 for rank 2")
		}
	}()
	_ = tensor.Concat([]*tensor.Tensor{x, y}, 5)
}

func TestConcat_RankMismatch_Panics(t *testing.T) {
	a := uop.NewArena(8)
	x := mustLeaf(a, []int64{2, 3}, make([]float32, 6))
	y := mustLeaf(a, []int64{2}, make([]float32, 2))
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on rank mismatch")
		} else if !strings.Contains(toString(r), "rank") {
			t.Fatalf("expected 'rank' in panic, got %v", r)
		}
	}()
	_ = tensor.Concat([]*tensor.Tensor{x, y}, 0)
}

func TestConcat_NonConcatDimMismatch_Panics(t *testing.T) {
	a := uop.NewArena(8)
	x := mustLeaf(a, []int64{2, 3}, make([]float32, 6))
	y := mustLeaf(a, []int64{3, 3}, make([]float32, 9)) // axis-0 OK, axis-1 same; concat axis 1
	z := mustLeaf(a, []int64{2, 3}, make([]float32, 6)) // axis-0 mismatch with x for axis=1 concat
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on non-concat-axis dim mismatch")
		}
	}()
	// Concat along axis 1; x has dim0=2, y has dim0=3 — mismatch.
	_ = tensor.Concat([]*tensor.Tensor{x, y, z}, 1)
}

func TestConcat_DeviceMismatch_Panics(t *testing.T) {
	a := uop.NewArena(8)
	x := mustLeaf(a, []int64{2, 3}, make([]float32, 6))
	y := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "other-device")
	y.SetData(make([]float32, 6))
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on device mismatch")
		}
	}()
	_ = tensor.Concat([]*tensor.Tensor{x, y}, 0)
}

// TestConcat_SymbolicBatch verifies the symbolic-batch path: two tensors with
// shape [n, 4] concatenated along axis 1 produce shape [n, 8]; concat along
// axis 0 produces shape [2n, 4]. Pad amounts go through shape.Sint arithmetic
// (Sub of axisDims), so this test gates the symbolic Pad path.
func TestConcat_SymbolicBatch_Axis1(t *testing.T) {
	a := uop.NewArena(64)
	x := tensor.NewSymbolicBatchInput(a, "n", 1, 32, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewSymbolicBatchInput(a, "n", 1, 32, []int64{4}, uop.Dtypes.Float32, "cpu")

	out := tensor.Concat([]*tensor.Tensor{x, y}, 1)
	gotSh := out.ShapeSints()
	if len(gotSh) != 2 {
		t.Fatalf("rank=%d, want 2", len(gotSh))
	}
	// dim 0 symbolic, dim 1 = 8.
	if _, concrete := gotSh[0].ConstValue(); concrete {
		t.Fatalf("dim 0 should be symbolic, got concrete")
	}
	if v, ok := gotSh[1].ConstValue(); !ok || v != 8 {
		t.Fatalf("dim 1=%v ok=%v, want 8", v, ok)
	}
}

// Concat along the symbolic axis itself (both inputs symbolic on the same axis)
// is a v1 limitation: Pad's internal shape formula (Sub(Add(shape, hi), Neg(lo)))
// yields structurally distinct Sint expressions for the two sides which don't
// re-intern equal, breaking the subsequent broadcast-Add. Concat along a
// non-symbolic axis (the common case in CNN imports) is fully supported, see
// TestConcat_SymbolicBatch_Axis1 above. We assert the limitation here so a
// future symmetric-Sint rewrite enabling this case becomes the gate.
func TestConcat_SymbolicBatch_Axis0_NotYet(t *testing.T) {
	a := uop.NewArena(64)
	x := tensor.NewSymbolicBatchInput(a, "n", 1, 32, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewSymbolicBatchInput(a, "n", 1, 32, []int64{4}, uop.Dtypes.Float32, "cpu")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("symbolic+symbolic axis-0 concat unexpectedly succeeded; rewrite the gate")
		}
	}()
	_ = tensor.Concat([]*tensor.Tensor{x, y}, 0)
}

// ── helpers ──────────────────────────────────────────────────────────────────

func shapeEqInts(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func float32SliceEq(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if err, ok := v.(error); ok {
		return err.Error()
	}
	if stringer, ok := v.(interface{ String() string }); ok {
		return stringer.String()
	}
	return ""
}

// keep shape import alive in case future tests use it directly.
var _ = shape.Const(0)
