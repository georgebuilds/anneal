package shape

import "github.com/georgebuilds/anneal/uop"

// Sint is the symbolic-integer seam. In slices 1–2 every Sint is a ConstInt.
// SymInt exists as a type-complete stub; its methods panic until slice 3.
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

// Sint arithmetic — fast path for ConstInt×ConstInt, symbolic path builds UOp nodes.

func Add(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va + vb}
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
	ar := symArena(a, b)
	return SymInt{Node: ar.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{toUOp(a, ar), toUOp(b, ar)}, nil, nil)}
}

func Mul(a, b Sint) Sint {
	va, oka := a.ConstValue()
	vb, okb := b.ConstValue()
	if oka && okb {
		return ConstInt{V: va * vb}
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

// Sint comparisons — panic for symbolic operands (not needed until slice 3b).

func Eq(a, b Sint) bool    { return cv(a) == cv(b) }
func Lt(a, b Sint) bool    { return cv(a) < cv(b) }
func Le(a, b Sint) bool    { return cv(a) <= cv(b) }
func EqI(a Sint, b int64) bool { return cv(a) == b }

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
