package shape

// Sint is the symbolic-integer seam. In v1 every Sint is a concrete int64 (ConstInt).
//
// Migration path to symbolic shapes (Phase 6+):
//   - SymInt { Node *uop.UOp } implements Sint
//   - View.Shape / View.Strides / View.Offset change from []int64 / int64
//     to []Sint / Sint once the rewrite engine can fold Sint arithmetic
//   - Movement-op signatures stay identical; callers that hold []int64 wrap with ConstInt
//   - ShapeTracker.FlatIndex returns a Sint (UOp expression) instead of an int64
//
// Do NOT add a SymInt implementation here until Phase 6 scheduler work begins.
type Sint interface {
	isSint()
	// ConstValue returns the concrete value and true if this Sint is a compile-time constant.
	ConstValue() (int64, bool)
}

// ConstInt is the v1 Sint: a concrete compile-time integer.
type ConstInt struct{ V int64 }

func (ConstInt) isSint()                     {}
func (c ConstInt) ConstValue() (int64, bool) { return c.V, true }
