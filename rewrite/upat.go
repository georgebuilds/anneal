package rewrite

import (
	"math"

	"github.com/georgebuilds/anneal/uop"
)

// UPat is a pattern node in the pattern-match DSL.
// Construct via Pat, Pats, WildPat, or AnyPat; chain With* methods to add constraints.
//
// Top-level patterns (direct children of a PatternMatcher rule) must specify ops.
// Nested patterns may use WildPat or omit the op constraint.
type UPat struct {
	// op constraint: nil = match any op; non-nil = uop.Op must be in this slice
	ops []uop.Op
	// dtype constraint: nil = match any dtype
	matchDtypes []*uop.DType
	// arg constraint
	hasArgConstr bool
	arg          any
	// capture name; "" = no capture
	name string

	// src matching — exactly one mode active:
	srcWild     bool      // no constraint on srcs
	srcRepeat   *UPat     // all srcs must match this single pattern
	srcVariants [][]*UPat // ordered/permutation variants

	strictLen   bool
	requiredLen int

	// earlyReject: every op in this set must appear in the direct children of the
	// node being matched. Checked before the full recursive match. Populated from the
	// first src variant's single-op patterns.
	earlyReject map[uop.Op]struct{}

	// OR combinator
	isAny   bool
	anyAlts []*UPat
}

// Pat creates a UPat matching a single op with no other constraints.
func Pat(op uop.Op) *UPat {
	return &UPat{ops: []uop.Op{op}, srcWild: true, earlyReject: map[uop.Op]struct{}{}}
}

// Pats creates a UPat matching any of the given ops.
func Pats(ops ...uop.Op) *UPat {
	return &UPat{ops: append([]uop.Op(nil), ops...), srcWild: true, earlyReject: map[uop.Op]struct{}{}}
}

// WildPat creates a UPat with no op constraint (wildcard, matches any op).
func WildPat() *UPat {
	return &UPat{srcWild: true, earlyReject: map[uop.Op]struct{}{}}
}

// AnyPat creates a UPat that succeeds if any of the given alternatives match (logical OR).
// Each alternative should specify its own op constraint.
func AnyPat(alts ...*UPat) *UPat {
	return &UPat{isAny: true, anyAlts: alts, earlyReject: map[uop.Op]struct{}{}}
}

// WithName sets the capture name. The matched UOp is stored in the captures map under name.
// If the same name appears in multiple parts of a pattern, all must bind to the same UOp.
func (p *UPat) WithName(n string) *UPat { p.name = n; return p }

// WithDtype constrains the matched UOp's dtype to one of the given types.
func (p *UPat) WithDtype(dtypes ...*uop.DType) *UPat {
	p.matchDtypes = append(p.matchDtypes, dtypes...)
	return p
}

// WithArg constrains the matched UOp's arg to equal a. Pass nil to match UOps with nil arg.
func (p *UPat) WithArg(a any) *UPat { p.hasArgConstr = true; p.arg = a; return p }

// WithSrc constrains sources positionally: the UOp must have exactly len(srcs) sources,
// each matching the corresponding pattern.
func (p *UPat) WithSrc(srcs ...*UPat) *UPat {
	return p.applySrc([][]*UPat{append([]*UPat(nil), srcs...)}, len(srcs), true)
}

// WithCommSrc is like WithSrc but tries all permutations, enabling commutative matching.
// Use when the op is commutative and you want to match regardless of operand order.
func (p *UPat) WithCommSrc(srcs ...*UPat) *UPat {
	return p.applySrc(permutations(srcs), len(srcs), true)
}

// WithRepSrc matches all sources against a single repeated pattern (e.g. SINK's many inputs).
func (p *UPat) WithRepSrc(src *UPat) *UPat {
	p.srcWild = false
	p.srcRepeat = src
	p.srcVariants = nil
	p.strictLen = false
	p.requiredLen = 0
	p.earlyReject = map[uop.Op]struct{}{}
	if len(src.ops) == 1 {
		p.earlyReject[src.ops[0]] = struct{}{}
	}
	return p
}

// WithAnyLen relaxes the strict length check so WithSrc patterns act as prefix matches.
func (p *UPat) WithAnyLen() *UPat { p.strictLen = false; return p }

func (p *UPat) applySrc(variants [][]*UPat, reqLen int, strict bool) *UPat {
	p.srcWild = false
	p.srcRepeat = nil
	p.srcVariants = variants
	p.requiredLen = reqLen
	p.strictLen = strict
	p.earlyReject = map[uop.Op]struct{}{}
	if len(variants) > 0 {
		for _, sp := range variants[0] {
			if len(sp.ops) == 1 {
				p.earlyReject[sp.ops[0]] = struct{}{}
			}
		}
	}
	return p
}

// ── Match ─────────────────────────────────────────────────────────────────────

// Match attempts to match u against this pattern starting with the provided capture store.
// Returns all successful capture bindings (multiple entries arise from commutative src tries).
// The incoming store is never modified; each result map is a fresh copy.
func (p *UPat) Match(u uop.UOp, store map[string]uop.UOp) []map[string]uop.UOp {
	if p.isAny {
		var res []map[string]uop.UOp
		for _, alt := range p.anyAlts {
			res = append(res, alt.Match(u, copyStore(store))...)
		}
		return res
	}

	// op check
	if len(p.ops) > 0 && !hasOp(p.ops, u.Op()) {
		return nil
	}

	// name capture — if already bound, the bound UOp must match u
	curStore := store
	if p.name != "" {
		if existing, ok := store[p.name]; ok {
			if existing != u {
				return nil
			}
		} else {
			ns := copyStore(store)
			ns[p.name] = u
			curStore = ns
		}
	}

	// dtype check
	if len(p.matchDtypes) > 0 && !hasDtype(p.matchDtypes, u.DType()) {
		return nil
	}

	// arg check
	if p.hasArgConstr && !equalArgs(p.arg, u.Arg()) {
		return nil
	}

	// src wildcard
	if p.srcWild {
		return []map[string]uop.UOp{curStore}
	}

	// repeat src: all srcs must satisfy srcRepeat
	if p.srcRepeat != nil {
		stores := []map[string]uop.UOp{curStore}
		for i := 0; i < u.NSrc(); i++ {
			child := u.Src(i)
			var next []map[string]uop.UOp
			for _, s := range stores {
				next = append(next, p.srcRepeat.Match(child, s)...)
			}
			stores = next
			if len(stores) == 0 {
				return nil
			}
		}
		return stores
	}

	// ordered/permutation src
	nSrc := u.NSrc()
	if nSrc < p.requiredLen || (p.strictLen && nSrc != p.requiredLen) {
		return nil
	}

	var res []map[string]uop.UOp
	for _, variant := range p.srcVariants {
		stores := []map[string]uop.UOp{curStore}
		for i, pat := range variant {
			child := u.Src(i)
			var next []map[string]uop.UOp
			for _, s := range stores {
				next = append(next, pat.Match(child, s)...)
			}
			stores = next
			if len(stores) == 0 {
				break
			}
		}
		res = append(res, stores...)
	}
	return res
}

// ── internal helpers ──────────────────────────────────────────────────────────

func copyStore(s map[string]uop.UOp) map[string]uop.UOp {
	out := make(map[string]uop.UOp, len(s))
	for k, v := range s {
		out[k] = v
	}
	return out
}

func hasOp(ops []uop.Op, op uop.Op) bool {
	for _, o := range ops {
		if o == op {
			return true
		}
	}
	return false
}

func hasDtype(dtypes []*uop.DType, d *uop.DType) bool {
	for _, dt := range dtypes {
		if dt == d {
			return true
		}
	}
	return false
}

// equalArgs compares two arg values for pattern-matching purposes.
// NaN float64 values are equal when bits match (same constant = same node).
func equalArgs(a, b any) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	switch av := a.(type) {
	case int64:
		bv, ok := b.(int64)
		return ok && av == bv
	case float64:
		bv, ok := b.(float64)
		return ok && math.Float64bits(av) == math.Float64bits(bv)
	case bool:
		bv, ok := b.(bool)
		return ok && av == bv
	case string:
		bv, ok := b.(string)
		return ok && av == bv
	default:
		return false
	}
}

// permutations returns all orderings of srcs.
// If all elements are pointer-identical, a single variant suffices (idempotent case).
func permutations(srcs []*UPat) [][]*UPat {
	if len(srcs) <= 1 {
		return [][]*UPat{append([]*UPat(nil), srcs...)}
	}
	allSame := true
	for _, s := range srcs[1:] {
		if s != srcs[0] {
			allSame = false
			break
		}
	}
	if allSame {
		return [][]*UPat{append([]*UPat(nil), srcs...)}
	}
	var result [][]*UPat
	used := make([]bool, len(srcs))
	var perm []*UPat
	var gen func()
	gen = func() {
		if len(perm) == len(srcs) {
			result = append(result, append([]*UPat(nil), perm...))
			return
		}
		for i, s := range srcs {
			if used[i] {
				continue
			}
			used[i] = true
			perm = append(perm, s)
			gen()
			perm = perm[:len(perm)-1]
			used[i] = false
		}
	}
	gen()
	return result
}
