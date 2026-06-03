package tensor

import (
	"fmt"
	"sort"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// Variable is the ergonomic surface for a symbolic dimension. It wraps an
// interned OpDefineVar UOp and exposes (a) Sint() to compose into shape
// lists for the general symbolic-shape constructor and (b) Bind(val) to
// produce a binding entry usable with RealizeWithBinding.
//
// Construct via NewVariable(arena, name, min, max). Two Variables created
// with the same (name, min, max) in the same arena alias to the same
// underlying DefineVar (intern). Two Variables created with the same name
// but different bounds in the same arena panic — the downstream
// FindDefineVar lookup is name-keyed, so a same-name collision would
// silently route shape/binding to the wrong bounds.
//
// Idiom mirrors tinygrad's Variable("seq", 1, 1024).bind(64):
//
//	v := tensor.NewVariable(a, "seq", 1, 1024)
//	x := tensor.NewSymbolicShape(a, []shape.Sint{shape.Const(B), v.Sint(), shape.Const(D)},
//	    uop.Dtypes.Float32, "webgpu")
//	tensor.RealizeWithBinding(v.Bind(64), x)
type Variable struct {
	defVar uop.UOp
	name   string
	min    int64
	max    int64
}

// NewVariable creates (or aliases to the existing) DefineVar with name and
// inclusive bounds [min, max] in arena a. Panics if a DefineVar with the
// same name but different bounds already exists in the arena (same-name
// collision; SPEC §6.4 ShapeDim is name-keyed).
func NewVariable(a *uop.Arena, name string, min, max int64) Variable {
	if name == "" {
		panic("tensor: NewVariable: name must be non-empty")
	}
	if min > max {
		panic(fmt.Sprintf("tensor: NewVariable: min %d > max %d for %q", min, max, name))
	}
	if existing, ok := a.FindDefineVar(name); ok {
		// Decode the existing bounds (DefineVar.src = [Const(min), Const(max+1)]).
		exMin, exMax, ok := definedBounds(existing)
		if !ok {
			panic(fmt.Sprintf("tensor: NewVariable: existing DefineVar %q has unexpected bound structure", name))
		}
		if exMin != min || exMax != max {
			panic(fmt.Sprintf(
				"tensor: NewVariable: name %q already registered with bounds [%d,%d]; got [%d,%d]",
				name, exMin, exMax, min, max,
			))
		}
		return Variable{defVar: existing, name: name, min: min, max: max}
	}
	dv := a.DefineVar(name, min, max)
	return Variable{defVar: dv, name: name, min: min, max: max}
}

// definedBounds decodes the inclusive [min, max] from a DefineVar UOp whose
// src is [Const(min), Const(max+1)] (the DefineVar internal encoding;
// inclusive lower, exclusive upper).
func definedBounds(dv uop.UOp) (int64, int64, bool) {
	if dv.Op() != uop.OpDefineVar || dv.NSrc() != 2 {
		return 0, 0, false
	}
	loC, ok := dv.Src(0).Arg().(int64)
	if !ok {
		return 0, 0, false
	}
	hiC, ok := dv.Src(1).Arg().(int64)
	if !ok {
		return 0, 0, false
	}
	return loC, hiC - 1, true
}

// Name returns the variable's symbolic name.
func (v Variable) Name() string { return v.name }

// Min returns the inclusive lower bound.
func (v Variable) Min() int64 { return v.min }

// Max returns the inclusive upper bound.
func (v Variable) Max() int64 { return v.max }

// Node returns the underlying DefineVar UOp.
func (v Variable) Node() uop.UOp { return v.defVar }

// Sint returns a shape.Sint wrapping this variable's DefineVar, suitable
// for composition into the shape slice passed to NewSymbolicShape.
func (v Variable) Sint() shape.Sint { return shape.SymInt{Node: v.defVar} }

// Bind returns a single-entry binding map mapping this variable's name to
// val. Use shape.MergeBindings (or plain map composition) to combine
// multiple variable bindings before passing to RealizeWithBinding.
//
// Bind does not check val against [min, max] — RealizeWithBinding's
// dispatch path performs the bound check (returns an error if out of
// range). This keeps Bind allocation-free and side-effect-free.
func (v Variable) Bind(val int64) map[string]int64 {
	return map[string]int64{v.name: val}
}

// MergeBindings unions zero or more binding maps. Same key in two inputs
// must map to the same value, otherwise it panics. Returns nil for zero
// inputs (RealizeWithBinding accepts nil).
//
// Convenience for the multi-Variable idiom:
//
//	binding := tensor.MergeBindings(B.Bind(32), T.Bind(128))
//	tensor.RealizeWithBinding(binding, out)
func MergeBindings(maps ...map[string]int64) map[string]int64 {
	if len(maps) == 0 {
		return nil
	}
	out := make(map[string]int64)
	for _, m := range maps {
		for k, v := range m {
			if existing, ok := out[k]; ok && existing != v {
				panic(fmt.Sprintf("tensor: MergeBindings: conflicting bindings for %q: %d vs %d", k, existing, v))
			}
			out[k] = v
		}
	}
	return out
}

// NewSymbolicShape creates a leaf tensor backed by a BUFFER node whose
// shape may contain symbolic dimensions at any position. sh is a mixed
// slice of shape.Const (concrete) and Variable.Sint() (symbolic) entries.
//
// Compared to NewSymbolicInput (1D, single sym dim) and
// NewSymbolicBatchInput (sym only at the outermost dim), this constructor
// is the general path: sym dims may appear at any position, multiple sym
// dims may appear in the same shape, and concrete and symbolic dims may
// be freely interleaved.
//
// The ShapeSintArg encoding sets Sym=true, V=0, VarName=name, Mul=1 at
// every sym position (regardless of axis index). SPEC §10's V-on-symbolic
// invariant is enforced defensively at every dim.
//
// All distinct DefineVars referenced by sym dims are collected and stored
// as srcs of the BUFFER node in name-sorted order. The 1D-symbolic and
// outermost-sym-only encodings (src[0]=DefineVar, arg=nil or a
// single-DefineVar src plus ShapeSintArg) remain in their current form;
// NewSymbolicShape is additive.
//
// Use RealizeWithBinding to provide concrete values for every sym dim at
// dispatch time.
func NewSymbolicShape(a *uop.Arena, sh []shape.Sint, dtype *uop.DType, device string) *Tensor {
	if len(sh) == 0 {
		panic("tensor: NewSymbolicShape: shape must be non-empty (use NewLeaf for scalar)")
	}
	// Build the ShapeSintArg, enforcing the SPEC §10 V=0-on-sym invariant
	// at every dim. Collect DefineVar UOps in name-sorted order.
	arg := make(uop.ShapeSintArg, len(sh))
	defVarByName := make(map[string]uop.UOp)
	hasSym := false
	for i, s := range sh {
		switch sv := s.(type) {
		case shape.ConstInt:
			arg[i] = uop.ShapeDim{V: sv.V}
		case shape.SymInt:
			if sv.Node.Op() != uop.OpDefineVar {
				// Slice 4 ShapeDim encoding assumes a bare-DefineVar bound at
				// the tensor-surface construction site. Derived bounds (Mul,
				// Add) are introduced later by the scheduler; users should
				// pass a Variable directly, not a SymInt expression.
				panic(fmt.Sprintf(
					"tensor: NewSymbolicShape: sym dim %d must be a Variable (bare DefineVar); got op=%s",
					i, sv.Node.Op(),
				))
			}
			name := sv.Node.Arg().(uop.VarArg).Name
			arg[i] = uop.ShapeDim{Sym: true, VarName: name, Mul: 1}
			if arg[i].V != 0 {
				panic(fmt.Sprintf("uop: ShapeSintArg.V must be 0 when Sym=true (SPEC §10); got V=%d VarName=%q at dim %d", arg[i].V, arg[i].VarName, i))
			}
			defVarByName[name] = sv.Node
			hasSym = true
		default:
			panic(fmt.Sprintf("tensor: NewSymbolicShape: dim %d is not a ConstInt or SymInt", i))
		}
	}
	if !hasSym {
		// Fully concrete shape; route through NewLeaf for byte-identical
		// behaviour with the static path. The general constructor's
		// contract is "may contain sym dims", not "must".
		ints := make([]int64, len(sh))
		for i, s := range sh {
			v, _ := s.ConstValue()
			ints[i] = v
		}
		return NewLeaf(a, ints, dtype, device)
	}
	// Sort DefineVar names for stable src ordering across construction
	// orders. Same logical shape built in different orders produces the
	// same BUFFER node srcs (and thus the same arena intern).
	names := make([]string, 0, len(defVarByName))
	for n := range defVarByName {
		names = append(names, n)
	}
	sort.Strings(names)
	srcs := make([]uop.UOp, len(names))
	for i, n := range names {
		srcs[i] = defVarByName[n]
	}
	node := a.New(uop.OpBuffer, dtype, srcs, arg, nil)
	return &Tensor{node: node, st: shape.NewShapeTrackerSints(append([]shape.Sint{}, sh...)), dtype: dtype, device: device}
}
