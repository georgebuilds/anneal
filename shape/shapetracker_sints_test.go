package shape

import "testing"

// sintSliceEq compares two []Sint slices by concrete value (all-concrete only).
func sintSliceEq(a []Sint, want []int64) bool {
	if len(a) != len(want) {
		return false
	}
	for i := range a {
		v, ok := a[i].ConstValue()
		if !ok || v != want[i] {
			return false
		}
	}
	return true
}

func TestNewShapeTrackerSints(t *testing.T) {
	st := NewShapeTrackerSints(AsSints([]int64{3, 4}))
	if !sintSliceEq(st.ShapeSints(), []int64{3, 4}) {
		t.Errorf("ShapeSints = %v", st.ShapeSints())
	}
	if !st.ActiveView().Contiguous {
		t.Error("fresh view should be contiguous")
	}
}

func TestReshapeSintsSameShape(t *testing.T) {
	st := NewShapeTrackerSints(AsSints([]int64{2, 6}))
	st2 := st.ReshapeSints(AsSints([]int64{2, 6}))
	if len(st2.Views) != len(st.Views) {
		t.Error("same-shape reshape should not push a view")
	}
}

func TestReshapeSintsReuseStrides(t *testing.T) {
	// Contiguous (2,6) → (12,): strides can be reused, no push.
	st := NewShapeTrackerSints(AsSints([]int64{2, 6}))
	st2 := st.ReshapeSints(AsSints([]int64{12}))
	if len(st2.Views) != 1 {
		t.Errorf("contiguous reshape should reuse view, got %d views", len(st2.Views))
	}
	if !sintSliceEq(st2.ShapeSints(), []int64{12}) {
		t.Errorf("shape = %v want [12]", st2.ShapeSints())
	}
}

func TestReshapeSintsPushNewView(t *testing.T) {
	// Pad to make non-contiguous, then a cross-boundary reshape forces a push.
	st := NewShapeTrackerSints(AsSints([]int64{6}))
	st = st.PadSints(AsMaskSint([][2]int64{{1, 1}})) // shape (8,), non-contiguous
	st2 := st.ReshapeSints(AsSints([]int64{4, 2}))
	if len(st2.Views) != 2 {
		t.Errorf("forced reshape should push, got %d views", len(st2.Views))
	}
	if !st2.ActiveView().Contiguous {
		t.Error("pushed view should be contiguous")
	}
}

func TestExpandSints(t *testing.T) {
	st := NewShapeTrackerSints(AsSints([]int64{1, 4}))
	st = st.ExpandSints(AsSints([]int64{3, 4}))
	if !sintSliceEq(st.ShapeSints(), []int64{3, 4}) {
		t.Errorf("expanded shape = %v want [3 4]", st.ShapeSints())
	}
	// Broadcast dim has stride 0.
	if cv(st.ActiveView().Strides[0]) != 0 {
		t.Error("broadcast dim should have stride 0")
	}
}

func TestPadSints(t *testing.T) {
	st := NewShapeTrackerSints(AsSints([]int64{4}))
	st = st.PadSints(AsMaskSint([][2]int64{{1, 2}})) // (1+4+2)=7
	if !sintSliceEq(st.ShapeSints(), []int64{7}) {
		t.Errorf("padded shape = %v want [7]", st.ShapeSints())
	}
	if st.ActiveView().Mask == nil {
		t.Error("pad should create a mask")
	}
}

func TestShrinkSints(t *testing.T) {
	st := NewShapeTrackerSints(AsSints([]int64{8}))
	st = st.ShrinkSints(AsMaskSint([][2]int64{{2, 6}})) // (4,)
	if !sintSliceEq(st.ShapeSints(), []int64{4}) {
		t.Errorf("shrunk shape = %v want [4]", st.ShapeSints())
	}
	// Offset should be 2 (skipped two elements of stride 1).
	if cv(st.ActiveView().Offset) != 2 {
		t.Errorf("shrink offset = %d want 2", cv(st.ActiveView().Offset))
	}
}

// ── flatindex.IndexExpr ─────────────────────────────────────────────────────

func TestIndexExprContiguous(t *testing.T) {
	v := NewContiguousView(AsSints([]int64{2, 3}))
	// row-major strides (3,1), offset 0.
	got := IndexExpr(v)
	want := "0 + i0*3 + i1*1"
	if got != want {
		t.Errorf("IndexExpr = %q want %q", got, want)
	}
}

func TestIndexExprNoStrides(t *testing.T) {
	// Scalar view: no strides → just the offset.
	v := NewView(nil, nil, Const(5), nil)
	if got := IndexExpr(v); got != "5" {
		t.Errorf("IndexExpr scalar = %q want \"5\"", got)
	}
}

func TestIndexExprNegativeAndZeroStride(t *testing.T) {
	// A view with offset, a zero stride (skipped), and a negative stride (flip).
	v := NewView(
		AsSints([]int64{2, 3, 4}),
		[]Sint{Const(0), Const(-6), Const(1)},
		Const(19),
		nil,
	)
	got := IndexExpr(v)
	want := "19 - i1*6 + i2*1"
	if got != want {
		t.Errorf("IndexExpr = %q want %q", got, want)
	}
}
