package shape

// ShapeTracker is a stack of Views representing a tensor's accumulated movement-op history.
// The last View is the active one.  All methods return a new ShapeTracker; the original
// is not modified (value semantics, matching UOp immutability requirements).
type ShapeTracker struct {
	Views []View
}

// NewShapeTracker returns a ShapeTracker with a single contiguous View for shape.
func NewShapeTracker(shape []int64) ShapeTracker {
	return ShapeTracker{Views: []View{NewContiguousView(shape)}}
}

// ActiveView returns the last (active) view.
func (st ShapeTracker) ActiveView() View { return st.Views[len(st.Views)-1] }

// Shape returns the shape of the active view.
func (st ShapeTracker) Shape() []int64 { return st.ActiveView().Shape }

// withLastView returns a new ShapeTracker with the last view replaced by v.
func (st ShapeTracker) withLastView(v View) ShapeTracker {
	views := make([]View, len(st.Views))
	copy(views, st.Views)
	views[len(views)-1] = v
	return ShapeTracker{Views: views}
}

// withPushedView returns a new ShapeTracker with v appended as a new view.
func (st ShapeTracker) withPushedView(v View) ShapeTracker {
	views := make([]View, len(st.Views)+1)
	copy(views, st.Views)
	views[len(views)-1] = v
	return ShapeTracker{Views: views}
}

// ── movement ops ──────────────────────────────────────────────────────────────

// Reshape returns a ShapeTracker with newShape applied.
// If the active view can reuse its strides, the view is updated in place;
// otherwise a fresh contiguous view is pushed (rangeify dissolves the stack).
func (st ShapeTracker) Reshape(newShape []int64) ShapeTracker {
	if shapesEqual(st.Shape(), newShape) {
		return st
	}
	if v, ok := st.ActiveView().Reshape(newShape); ok {
		return st.withLastView(v)
	}
	return st.withPushedView(NewContiguousView(newShape))
}

// Expand broadcasts dimensions (see View.Expand).
func (st ShapeTracker) Expand(newShape []int64) ShapeTracker {
	return st.withLastView(st.ActiveView().Expand(newShape))
}

// Permute reorders dimensions (see View.Permute).
func (st ShapeTracker) Permute(order []int) ShapeTracker {
	return st.withLastView(st.ActiveView().Permute(order))
}

// Pad adds zero padding (see View.Pad).
func (st ShapeTracker) Pad(arg [][2]int64) ShapeTracker {
	return st.withLastView(st.ActiveView().Pad(arg))
}

// Shrink selects a sub-region (see View.Shrink).
func (st ShapeTracker) Shrink(arg [][2]int64) ShapeTracker {
	return st.withLastView(st.ActiveView().Shrink(arg))
}

// Flip reverses elements along specified axes (see View.Flip).
func (st ShapeTracker) Flip(axes []bool) ShapeTracker {
	return st.withLastView(st.ActiveView().Flip(axes))
}
