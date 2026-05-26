package shape

import "testing"

// shapeEq64 compares two []int64 slices — used for ShapeTracker.Shape() comparisons.
func shapeEq64(a, b []int64) bool {
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

func TestShapeTrackerBasic(t *testing.T) {
	st := NewShapeTracker([]int64{4, 5})
	if !shapeEq64(st.Shape(), []int64{4, 5}) {
		t.Fatalf("initial shape %v", st.Shape())
	}
	if !st.ActiveView().Contiguous {
		t.Fatal("initial view should be contiguous")
	}
}

func TestShapeTrackerReshapePush(t *testing.T) {
	// Non-contiguous view: reshape that can't reuse strides → new view pushed.
	base := NewShapeTracker([]int64{6})
	// Pad to (8,) making it non-contiguous.
	st := base.Pad([][2]int64{{1, 1}})
	// Reshape to (8,): same shape, no push.
	st2 := st.Reshape([]int64{8})
	if len(st2.Views) != len(st.Views) {
		t.Errorf("same-shape reshape should not push view, got %d views", len(st2.Views))
	}
}

func TestShapeTrackerViewStack(t *testing.T) {
	// A reshape that forces a new view increases the view stack length.
	st := NewShapeTracker([]int64{6})
	st = st.Pad([][2]int64{{1, 1}}) // shape=(8,), non-contiguous
	// Reshape to (4,2): mask (1,7) on shape (8,) → cross-boundary → push new view.
	st2 := st.Reshape([]int64{4, 2})
	if len(st2.Views) != 2 {
		t.Errorf("expected 2 views after forced push, got %d", len(st2.Views))
	}
	// The new view should be contiguous.
	if !st2.ActiveView().Contiguous {
		t.Error("pushed view should be contiguous")
	}
	if !shapeEq64(st2.Shape(), []int64{4, 2}) {
		t.Errorf("pushed view shape %v want [4 2]", st2.Shape())
	}
}

func TestShapeTrackerMovementChain(t *testing.T) {
	// transpose → shrink → flip → check FlatIndex
	st := NewShapeTracker([]int64{4, 6})
	st = st.Permute([]int{1, 0})                     // (6,4)
	st = st.Shrink([][2]int64{{1, 5}, {0, 4}})       // (4,4)
	st = st.Flip([]bool{false, true})                // flip dim-1

	v := st.ActiveView()
	// Original layout: row-major 4×6 (strides 6,1).
	// After permute: shape (6,4), strides (1,6).
	// After shrink with arg [(1,5),(0,4)]: offset = 1*1+0*6 = 1, shape (4,4), strides (1,6).
	// After flip dim-1: offset += (4-1)*6 = 18 → total 19, stride[1] = -6.
	if cv(v.Offset) != 19 {
		t.Errorf("chain offset=%d want 19", cv(v.Offset))
	}
	// Element (0,0) of final view = flat index 19 + 0*1 + 0*(-6) = 19.
	idx := FlatIndex(v, []int64{0, 0})
	if idx != 19 {
		t.Errorf("FlatIndex(0,0)=%d want 19", idx)
	}
	// Element (0,3) of final view = 19 + 0 + 3*(-6) = 19 - 18 = 1.
	idx = FlatIndex(v, []int64{0, 3})
	if idx != 1 {
		t.Errorf("FlatIndex(0,3)=%d want 1", idx)
	}
}

func TestShapeTrackerExpandReshape(t *testing.T) {
	// Broadcast then reshape: (1,4) → expand (3,4) → reshape (12,) should fail strides
	// (stride-0 mixes with real) → push new view.
	st := NewShapeTracker([]int64{1, 4})
	st = st.Expand([]int64{3, 4})
	st = st.Reshape([]int64{12})
	// Active view should be fresh contiguous for (12,).
	v := st.ActiveView()
	if !shapeEq64(AsInts(v.Shape), []int64{12}) {
		t.Fatalf("shape=%v want [12]", AsInts(v.Shape))
	}
	if !v.Contiguous {
		t.Error("pushed view should be contiguous")
	}
}

func TestShapeTrackerImmutable(t *testing.T) {
	// Verify that movement ops return new ShapeTackers without modifying the original.
	st1 := NewShapeTracker([]int64{3, 4})
	st2 := st1.Permute([]int{1, 0})
	if shapeEq64(st1.Shape(), st2.Shape()) {
		t.Error("permute should change shape")
	}
	if !shapeEq64(st1.Shape(), []int64{3, 4}) {
		t.Error("original ShapeTracker mutated")
	}
}
