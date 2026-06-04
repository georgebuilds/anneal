package uop

import "math"

// MulInt64Checked returns a*b and reports whether the product fits in int64.
//
// Use at every site where bound analysis, affine extraction, or stride
// accumulation multiplies int64 values whose product would otherwise wrap
// silently. Callers that own a "give-up" channel (Bounds.Valid=false,
// (nil, 0, false), etc.) should treat ok=false as "cannot represent" and
// return that channel. Callers that lack a give-up channel (stride emission,
// memory-address arithmetic) should panic on ok=false with a site-specific
// diagnostic, since a wrong product silently corrupts downstream codegen.
//
// Detection uses divide-back (p/b == a) with explicit MinInt64*-1 guarding,
// because Go's signed division panics on math.MinInt64 / -1.
func MulInt64Checked(a, b int64) (int64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if (a == math.MinInt64 && b == -1) || (b == math.MinInt64 && a == -1) {
		return 0, false
	}
	p := a * b
	if p/b != a {
		return 0, false
	}
	return p, true
}
