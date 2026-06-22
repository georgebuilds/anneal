package shape

import "github.com/georgebuilds/anneal/uop"

// Sint is the symbolic-integer seam: either a concrete ConstInt or a SymInt backed
// by a UOp node. Arithmetic (Add/Sub/Mul/Neg/IDiv/Mod) builds real UOp expressions
// exercised by every dynamic-batch test. The comparison functions Lt/Le/Eq
// deliberately panic for symbolic operands - they are the symbolic-comparison fence
// (SPEC §6.4): comparing two symbolic values arithmetically would require an SMT
// solver, which §10 forbids on the core indexing path. Do not silently enable these.
type Sint interface {
	isSint()
	ConstValue() (int64, bool)
}

// ConstInt is the concrete Sint: a compile-time integer.
type ConstInt struct{ V int64 }

func (ConstInt) isSint()                     {}
func (c ConstInt) ConstValue() (int64, bool) { return c.V, true }

// SymInt is a symbolic dimension backed by a UOp node.
// Arithmetic on SymInt builds UOp expression nodes in the same arena.
type SymInt struct{ Node uop.UOp }

func (SymInt) isSint()                   {}
func (SymInt) ConstValue() (int64, bool) { return 0, false }

// cv extracts the concrete int64 value from s.
// Panics if s is symbolic (unreachable until symbolic dims are bound and folded).
func cv(s Sint) int64 {
	v, ok := s.ConstValue()
	if !ok {
		panic("shape: symbolic Sint not yet bound; call BoundFold before extracting")
	}
	return v
}

// CV is the exported variant of cv for use from other packages.
func CV(s Sint) int64 { return cv(s) }

// Const wraps a literal int64 as a Sint.
func Const(v int64) Sint { return ConstInt{V: v} }

// SintFromShapeDim converts a uop.ShapeDim into a Sint by reconstructing the
// bound expression from its (VarName, Mul) encoding for symbolic dims, and
// returning a ConstInt for concrete dims. The symbolic path delegates to
// uop.RebuildSymBound, so interning aliases the rebuilt node to the original
// whenever the original was built in canonical orientation. Single source of
// truth for ShapeDim to Sint conversion across schedule/, tensor/gradient.go,
// and any other consumer.
func SintFromShapeDim(a *uop.Arena, d uop.ShapeDim) Sint {
	if !d.Sym {
		return ConstInt{V: d.V}
	}
	return SymInt{Node: uop.RebuildSymBound(a, d)}
}

// SintShapesEqual reports whether two Sint slices are structurally equal without
// forcing concretisation. ConstInt dims are compared by value; SymInt dims are
// compared by UOp identity (same arena index in the same arena).
func SintShapesEqual(a, b []Sint) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		switch av := a[i].(type) {
		case ConstInt:
			bv, ok := b[i].(ConstInt)
			if !ok || av.V != bv.V {
				return false
			}
		case SymInt:
			bv, ok := b[i].(SymInt)
			if !ok || av.Node != bv.Node {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// symArena returns the arena from the first SymInt operand.
// Caller must ensure at least one operand is SymInt.
func symArena(a, b Sint) *uop.Arena {
	if sym, ok := a.(SymInt); ok {
		return sym.Node.Arena()
	}
	if sym, ok := b.(SymInt); ok {
		return sym.Node.Arena()
	}
	panic("shape: symArena called without any SymInt operand")
}

// symArena1 is the unary variant of symArena.
func symArena1(a Sint) *uop.Arena {
	if sym, ok := a.(SymInt); ok {
		return sym.Node.Arena()
	}
	panic("shape: symArena1 called without a SymInt operand")
}

// toUOp converts a Sint to a UOp in arena ar.
// ConstInt values become OpConst nodes; SymInt values return their Node directly.
func toUOp(s Sint, ar *uop.Arena) uop.UOp {
	if sym, ok := s.(SymInt); ok {
		return sym.Node
	}
	v, _ := s.ConstValue()
	return ar.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil)
}

// Sint arithmetic - fast path for ConstInt×ConstInt, symbolic path builds UOp nodes.

func Add(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va + vb}
	}
	// Identity folds - x + 0 = x, 0 + x = x. Keeps Pad/Shrink intermediate
	// shapes canonical (otherwise Sub(Add(x, 0), 0) wraps a trivially-x
	// expression that ResolveLE can't see through via interval arithmetic).
	if oka && va == 0 {
		return b
	}
	if okb && vb == 0 {
		return a
	}
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func Sub(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va - vb}
	}
	if okb && vb == 0 {
		return a // x - 0 = x
	}
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func Mul(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va * vb}
	}
	// Identity / annihilator folds.
	if oka {
		if va == 0 {
			return ConstInt{V: 0}
		}
		if va == 1 {
			return b
		}
	}
	if okb {
		if vb == 0 {
			return ConstInt{V: 0}
		}
		if vb == 1 {
			return a
		}
	}
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func Neg(a Sint) Sint {
	if v, ok := a.ConstValue(); ok {
		return ConstInt{V: -v}
	}
	ar := symArena1(a)
	sym := a.(SymInt)
	return SymInt{Node: ar.New(uop.OpNeg, uop.Dtypes.Index, []uop.UOp{sym.Node}, nil, nil)}
}

func IDiv(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va / vb}
	}
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func Mod(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va % vb}
	}
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func SintMax(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		if va >= vb {
			return a
		}
		return b
	}
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpMax, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func SintMin(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		if va <= vb {
			return a
		}
		return b
	}
	// min(a,b) = where(a < b, a, b)
	ar := symArena(a, b)
	ua, ub := toUOp(a, ar), toUOp(b, ar)
	cond := ar.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{ua, ub}, nil, nil)
	return SymInt{Node: ar.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{cond, ua, ub}, nil, nil)}
}

// Sint comparisons - panic for symbolic operands (SPEC §6.4 fence).

func Eq(a, b Sint) bool        { return cv(a) == cv(b) }
func Lt(a, b Sint) bool        { return cv(a) < cv(b) }
func Le(a, b Sint) bool        { return cv(a) <= cv(b) }
func EqI(a Sint, b int64) bool { return cv(a) == b }

// ── Bounds predicates (Option B Slice 5) ──────────────────────────────────────
//
// ResolveNonNeg and ResolveLE answer "is this provably true?" by walking the
// underlying UOp expression and propagating integer intervals through the
// arithmetic ops introduced by SymInt construction (Add/Sub/Mul/Neg/IDiv/Mod
// over DefineVar/Const operands). Mirrors tinygrad's resolve() at Pad/Shrink
// validation sites: a "false" answer (provably-false OR unprovable) becomes
// an "invalid pad/shrink" error at the call site.
//
// The walker is intentionally a minimal subset of rewrite/rules.BoundsOf -
// shape/ cannot import rewrite/rules (cycle), and only DefineVar-based
// arithmetic appears in SymInt expressions. Bounds for DefineVar come from
// its (min, max) src Consts (matching DefineVar's construction at
// uop.Arena.DefineVar - exclusive upper).

type sintBounds struct {
	min, max int64
	valid    bool
}

// boundsOfUOp computes inclusive [min, max] for an integer UOp node built
// from the SymInt-arithmetic constructors. Unhandled ops return valid=false.
func boundsOfUOp(u uop.UOp) sintBounds {
	if !u.Valid() {
		return sintBounds{}
	}
	switch u.Op() {
	case uop.OpConst:
		v, ok := u.Arg().(int64)
		if !ok {
			return sintBounds{}
		}
		return sintBounds{v, v, true}
	case uop.OpDefineVar:
		// DefineVar(name, min, max+1) - exclusive upper in src[1].
		if u.NSrc() != 2 {
			return sintBounds{}
		}
		lo := boundsOfUOp(u.Src(0))
		hi := boundsOfUOp(u.Src(1))
		if !lo.valid || !hi.valid {
			return sintBounds{}
		}
		return sintBounds{lo.min, hi.max - 1, true}
	case uop.OpNeg:
		if u.NSrc() != 1 {
			return sintBounds{}
		}
		s := boundsOfUOp(u.Src(0))
		if !s.valid {
			return sintBounds{}
		}
		return sintBounds{-s.max, -s.min, true}
	}
	if u.NSrc() != 2 {
		return sintBounds{}
	}
	a := boundsOfUOp(u.Src(0))
	b := boundsOfUOp(u.Src(1))
	if !a.valid || !b.valid {
		return sintBounds{}
	}
	switch u.Op() {
	case uop.OpAdd:
		return sintBounds{a.min + b.min, a.max + b.max, true}
	case uop.OpSub:
		return sintBounds{a.min - b.max, a.max - b.min, true}
	case uop.OpMul:
		corners := [4]int64{a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max}
		lo, hi := corners[0], corners[0]
		for _, c := range corners[1:] {
			if c < lo {
				lo = c
			}
			if c > hi {
				hi = c
			}
		}
		return sintBounds{lo, hi, true}
	case uop.OpIDiv:
		if b.min*b.max > 0 {
			// Floor div with sign-definite divisor.
			fd := func(x, y int64) int64 {
				q := x / y
				if (x%y != 0) && ((x < 0) != (y < 0)) {
					q--
				}
				return q
			}
			corners := [4]int64{fd(a.min, b.min), fd(a.min, b.max), fd(a.max, b.min), fd(a.max, b.max)}
			lo, hi := corners[0], corners[0]
			for _, c := range corners[1:] {
				if c < lo {
					lo = c
				}
				if c > hi {
					hi = c
				}
			}
			return sintBounds{lo, hi, true}
		}
		return sintBounds{}
	case uop.OpMod:
		if b.min == b.max && b.min > 0 {
			return sintBounds{0, b.min - 1, true}
		}
		return sintBounds{}
	}
	return sintBounds{}
}

// ResolveNonNeg reports whether s is provably >= 0 without invoking the
// SymInt-comparison fence. Concrete Sints short-circuit by value; symbolic
// Sints consult interval bounds on the backing UOp tree. Returns false on
// "provably negative" OR "unprovable" - both fail the same validation site.
// Matches tinygrad's resolve(s >= 0) semantics at Pad / Shrink validation.
func ResolveNonNeg(s Sint) bool {
	if v, ok := s.ConstValue(); ok {
		return v >= 0
	}
	sym, ok := s.(SymInt)
	if !ok {
		return false
	}
	b := boundsOfUOp(sym.Node)
	return b.valid && b.min >= 0
}

// ResolveLE reports whether a <= b is provably true. Concrete operands use
// direct integer compare; identical SymInts return true (a <= a). For
// non-identical mixed/symbolic operands the predicate forms Sub(b, a) and
// checks that the result's min bound is >= 0. Returns false on unprovable.
//
// Why an identity short-circuit: boundsOfUOp uses interval arithmetic
// without dependency tracking, so Sub(n, n) lowers to [n.min - n.max,
// n.max - n.min] which can be negative - losing the trivially-provable
// a <= a relation. Comparing node identity recovers it without invoking
// a symbolic simplifier in the validation path.
func ResolveLE(a, b Sint) bool {
	av, aok := a.ConstValue()
	bv, bok := b.ConstValue()
	if aok && bok {
		return av <= bv
	}
	if SintEqual(a, b) {
		return true
	}
	// Need a SymInt to find an arena.
	if _, ok := a.(SymInt); !ok {
		if _, ok := b.(SymInt); !ok {
			return false
		}
	}
	diff := Sub(b, a)
	if v, ok := diff.ConstValue(); ok {
		return v >= 0
	}
	sym, ok := diff.(SymInt)
	if !ok {
		return false
	}
	bd := boundsOfUOp(sym.Node)
	return bd.valid && bd.min >= 0
}

// AsSints converts a concrete []int64 slice to []Sint.
func AsSints(ints []int64) []Sint {
	if ints == nil {
		return nil
	}
	out := make([]Sint, len(ints))
	for i, v := range ints {
		out[i] = ConstInt{V: v}
	}
	return out
}

// AsInts extracts concrete int64 values from a []Sint slice.
// Panics if any element is symbolic.
func AsInts(sints []Sint) []int64 {
	if sints == nil {
		return nil
	}
	out := make([]int64, len(sints))
	for i, s := range sints {
		out[i] = cv(s)
	}
	return out
}

// AsMaskSint converts a [][2]int64 mask to [][2]Sint.
func AsMaskSint(m [][2]int64) [][2]Sint {
	if m == nil {
		return nil
	}
	out := make([][2]Sint, len(m))
	for i, p := range m {
		out[i] = [2]Sint{ConstInt{V: p[0]}, ConstInt{V: p[1]}}
	}
	return out
}

// AsIntMask extracts concrete int64 values from a [][2]Sint mask.
// Panics if any element is symbolic.
func AsIntMask(m [][2]Sint) [][2]int64 {
	if m == nil {
		return nil
	}
	out := make([][2]int64, len(m))
	for i, p := range m {
		out[i] = [2]int64{cv(p[0]), cv(p[1])}
	}
	return out
}

// Product computes the product of all Sint values in dims.
func Product(dims []Sint) Sint {
	acc := int64(1)
	for _, d := range dims {
		acc *= cv(d)
	}
	return ConstInt{V: acc}
}

// SymbolicProduct computes the product of all Sint values in dims, building
// a SymInt expression for any symbolic operand. For all-concrete inputs this
// returns a ConstInt and is equivalent to Product. Used by reshape validation
// to compare symbolic sub-products without forcing concretisation.
//
// Size-1 concrete dims are dropped before multiplication so that products
// agree under the multiplicative identity (e.g. prod([n,4]) == prod([n,4,1])).
// Matches tinygrad's prod() over a Python list where concrete 1s don't
// generate distinct symbolic nodes.
func SymbolicProduct(dims []Sint) Sint {
	// Filter out concrete-1 dims.
	filtered := make([]Sint, 0, len(dims))
	for _, d := range dims {
		if v, ok := d.ConstValue(); ok && v == 1 {
			continue
		}
		filtered = append(filtered, d)
	}
	if len(filtered) == 0 {
		return ConstInt{V: 1}
	}
	acc := filtered[0]
	for _, d := range filtered[1:] {
		acc = Mul(acc, d)
	}
	return acc
}

// SintEqual reports whether a and b are structurally equal.
// Concrete ConstInts compare by value; SymInts compare by UOp identity (which,
// because ALU ops are interned, is structural equality on the bound expression).
// Used by reshape sub-product validation across symbolic shapes.
func SintEqual(a, b Sint) bool {
	return SintShapesEqual([]Sint{a}, []Sint{b})
}

// HasSymbolic reports whether any element of sh is a SymInt.
func HasSymbolic(sh []Sint) bool {
	for _, s := range sh {
		if _, ok := s.ConstValue(); !ok {
			return true
		}
	}
	return false
}
