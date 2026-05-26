package rules

import "github.com/georgebuilds/anneal/uop"

// Bounds is an inclusive integer interval [Min, Max].
// Valid=false means bounds are unknown or not applicable (e.g. float dtype,
// unsupported op, or a node whose bounds cannot be statically determined).
type Bounds struct {
	Min, Max int64
	Valid    bool
}

// exact returns a Bounds for a single known value.
func exact(v int64) Bounds { return Bounds{v, v, true} }

// BoundsOf computes integer interval bounds for u via structural interval arithmetic.
// Only integer-dtype nodes produce valid bounds; floats return Bounds{Valid: false}.
//
// No SMT/Z3 is used (SPEC §6.4 invariant). Bounds compose through binary ALU ops
// by standard interval rules.
func BoundsOf(u uop.UOp) Bounds {
	if u.DType() == nil || u.DType().IsFloat() {
		return Bounds{}
	}

	switch u.Op() {
	case uop.OpConst:
		if i, ok := asInt(u.Arg()); ok {
			return exact(i)
		}
		if b, ok := u.Arg().(bool); ok {
			if b {
				return exact(1)
			}
			return exact(0)
		}
		return Bounds{}

	case uop.OpDefineVar:
		// DefineVar stores (name, min, max) but anneal v1 encodes min/max via src.
		// Two-src form: src[0]=lower Const, src[1]=upper Const (exclusive upper).
		if u.NSrc() == 2 {
			lo := BoundsOf(u.Src(0))
			hi := BoundsOf(u.Src(1))
			if lo.Valid && hi.Valid {
				return Bounds{lo.Min, hi.Max - 1, true}
			}
		}
		return Bounds{}

	case uop.OpRange:
		// Range(lower, upper) — GPU iteration range [lower, upper).
		if u.NSrc() == 2 {
			lo := BoundsOf(u.Src(0))
			hi := BoundsOf(u.Src(1))
			if lo.Valid && hi.Valid && hi.Max > lo.Min {
				return Bounds{lo.Min, hi.Max - 1, true}
			}
		}
		return Bounds{}
	}

	// Binary ALU ops — only for integer dtypes.
	if !u.DType().IsInt() && !u.DType().IsBool() {
		return Bounds{}
	}
	if u.NSrc() != 2 {
		return Bounds{}
	}

	s0, s1 := BoundsOf(u.Src(0)), BoundsOf(u.Src(1))
	if !s0.Valid || !s1.Valid {
		return Bounds{}
	}

	switch u.Op() {
	case uop.OpAdd:
		return Bounds{s0.Min + s1.Min, s0.Max + s1.Max, true}

	case uop.OpSub:
		return Bounds{s0.Min - s1.Max, s0.Max - s1.Min, true}

	case uop.OpMul:
		corners := [4]int64{
			s0.Min * s1.Min, s0.Min * s1.Max,
			s0.Max * s1.Min, s0.Max * s1.Max,
		}
		lo, hi := corners[0], corners[0]
		for _, c := range corners[1:] {
			if c < lo {
				lo = c
			}
			if c > hi {
				hi = c
			}
		}
		return Bounds{lo, hi, true}

	case uop.OpMax:
		lo := s0.Min
		if s1.Min > lo {
			lo = s1.Min
		}
		hi := s0.Max
		if s1.Max > hi {
			hi = s1.Max
		}
		return Bounds{lo, hi, true}

	case uop.OpIDiv:
		// Floor division; only tractable when divisor is sign-definite.
		if s1.Min*s1.Max > 0 {
			corners := [4]int64{
				floorDiv(s0.Min, s1.Min), floorDiv(s0.Min, s1.Max),
				floorDiv(s0.Max, s1.Min), floorDiv(s0.Max, s1.Max),
			}
			lo, hi := corners[0], corners[0]
			for _, c := range corners[1:] {
				if c < lo {
					lo = c
				}
				if c > hi {
					hi = c
				}
			}
			return Bounds{lo, hi, true}
		}
		return Bounds{}

	case uop.OpMod:
		// Floor modulo; tractable when divisor is a positive constant.
		if s1.Min == s1.Max && s1.Min > 0 {
			c := s1.Min
			if s0.Min/c == s0.Max/c {
				// Dividend stays within one modular period.
				return Bounds{floorMod(s0.Min, c), floorMod(s0.Max, c), true}
			}
			return Bounds{0, c - 1, true}
		}
		return Bounds{}

	case uop.OpCmpLt:
		// Result is bool (0 or 1).
		vmin := s0.Max < s1.Min // always true?
		vmax := s0.Min < s1.Max // ever true?
		b := func(v bool) int64 {
			if v {
				return 1
			}
			return 0
		}
		return Bounds{b(vmin), b(vmax), true}

	case uop.OpCmpNe:
		neverEqual := (s0.Max < s1.Min) || (s1.Max < s0.Min)
		alwaysEqual := s0.Min == s0.Max && s0.Min == s1.Min && s1.Min == s1.Max
		b := func(v bool) int64 {
			if v {
				return 1
			}
			return 0
		}
		return Bounds{b(neverEqual), b(!alwaysEqual), true}

	case uop.OpCmpEq:
		neverEqual := (s0.Max < s1.Min) || (s1.Max < s0.Min)
		alwaysEqual := s0.Min == s0.Max && s0.Min == s1.Min && s1.Min == s1.Max
		b := func(v bool) int64 {
			if v {
				return 1
			}
			return 0
		}
		return Bounds{b(alwaysEqual), b(!neverEqual), true}
	}

	return Bounds{}
}

// ── Canonicalization key ──────────────────────────────────────────────────────

// cmpUOp defines a stable total order on UOps for canonicalizing commutative
// operands. Returns negative if a < b, positive if a > b, zero if structurally
// equal. Mirrors tinygrad's UOp.tuplize comparison.
//
// Order: (op, dtype, arg, nsrc, src[0], src[1], …), with arena index as final
// tiebreak only for nodes that are structural twins (interning collapses them
// anyway). Depth-limited to 16; at the depth limit arena index is used directly.
func cmpUOp(a, b uop.UOp) int {
	return cmpUOpD(a, b, 16)
}

func cmpUOpD(a, b uop.UOp, depth int) int {
	if a == b {
		return 0
	}
	if depth == 0 {
		if a.Index() < b.Index() {
			return -1
		}
		if a.Index() > b.Index() {
			return 1
		}
		return 0
	}
	if a.Op() != b.Op() {
		if a.Op() < b.Op() {
			return -1
		}
		return 1
	}
	if c := uop.CmpDType(a.DType(), b.DType()); c != 0 {
		return c
	}
	if c := cmpAny(a.Arg(), b.Arg()); c != 0 {
		return c
	}
	if a.NSrc() != b.NSrc() {
		if a.NSrc() < b.NSrc() {
			return -1
		}
		return 1
	}
	for i := 0; i < a.NSrc(); i++ {
		if c := cmpUOpD(a.Src(i), b.Src(i), depth-1); c != 0 {
			return c
		}
	}
	if a.Index() < b.Index() {
		return -1
	}
	if a.Index() > b.Index() {
		return 1
	}
	return 0
}

// cmpAny imposes a total order on arg values of the same or different types.
// Type order: nil < bool < int64 < float64 < string.
func cmpAny(a, b any) int {
	ta, tb := typeTag(a), typeTag(b)
	if ta != tb {
		if ta < tb {
			return -1
		}
		return 1
	}
	switch av := a.(type) {
	case nil:
		return 0
	case bool:
		bv := b.(bool)
		if !av && bv {
			return -1
		}
		if av && !bv {
			return 1
		}
		return 0
	case int64:
		bv := b.(int64)
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
		return 0
	case float64:
		bv := b.(float64)
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
		return 0
	case string:
		bv := b.(string)
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
		return 0
	}
	return 0
}

func typeTag(a any) int {
	switch a.(type) {
	case nil:
		return 0
	case bool:
		return 1
	case int64:
		return 2
	case float64:
		return 3
	case string:
		return 4
	}
	return 5
}
