package shape

import "fmt"

// FlatIndex computes the flat buffer index for the given per-dimension indices
// under the active view:
//
//	idx = offset + Σ indices[i] * strides[i]
//
// This is the expression rangeify will consume in Phase 7.  Call IsValid first
// to check mask membership; FlatIndex does not validate bounds.
func FlatIndex(v View, indices []int64) int64 {
	idx := v.Offset
	for i, ind := range indices {
		idx += ind * v.Strides[i]
	}
	return idx
}

// IsValid reports whether indices falls within the view's mask.
// If the view has no mask, all in-bounds indices are valid.
func IsValid(v View, indices []int64) bool {
	if v.Mask == nil {
		return true
	}
	for i, ind := range indices {
		if ind < v.Mask[i][0] || ind >= v.Mask[i][1] {
			return false
		}
	}
	return true
}

// IndexExpr returns the flat-index expression as a human-readable string,
// suitable for debug output and for the Phase-7 rangeify pass to validate its
// range-substitution output against.
//
// Format: "offset [+/-] i0*s0 [+/-] i1*s1 …"
func IndexExpr(v View) string {
	if len(v.Strides) == 0 {
		return fmt.Sprintf("%d", v.Offset)
	}
	s := fmt.Sprintf("%d", v.Offset)
	for i, st := range v.Strides {
		if st == 0 {
			continue
		}
		if st > 0 {
			s += fmt.Sprintf(" + i%d*%d", i, st)
		} else {
			s += fmt.Sprintf(" - i%d*%d", i, -st)
		}
	}
	return s
}
