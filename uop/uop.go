package uop

import (
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"
)

// Phase classifies the compilation pass that first constructed a UOp node.
// PhaseForward is the zero value so all arenas start in the forward phase.
type Phase uint8

const (
	PhaseForward  Phase = 0 // forward computation (default)
	PhaseBackward Phase = 1 // reverse-mode autodiff pass
)

func (p Phase) String() string {
	if p == PhaseBackward {
		return "backward"
	}
	return "forward"
}

// Arena holds all UOp nodes for one compilation unit (bounded by one realize boundary).
//
// Not safe for concurrent mutation; the single-threaded compile path in v1 makes
// per-operation locking unnecessary. Document any future multi-arena usage explicitly.
type Arena struct {
	nodes      []uopNode
	cache      map[uint64][]uint32  // hash → matching arena indices (separate chaining)
	leaves     map[uint32][]float32 // leaf data indexed by local UOp index; released with the arena
	provenance []Phase              // parallel to nodes; set once at first construction
	phase      Phase                // current build phase; new allocations inherit this
	Ext        any                  // arena-scoped extension slot; GC'd with the arena

	// realizeID is a process-unique, monotonic id (never reused, even after this
	// arena is GC'd) used to scope the executor's stateful realize buffer cache.
	// realizeGen bumps whenever a leaf's data changes (SetData/Load), so a cache
	// keyed by (realizeID, realizeGen) is invalidated on any input mutation.
	realizeID  uint64
	realizeGen uint64
}

var arenaIDCounter atomic.Uint64

// uopNode is the stored, immutable representation of one UOp.
type uopNode struct {
	op    Op
	dtype *DType
	src   []uint32 // arena indices of source nodes; nil for leaf ops
	arg   any
	tag   any
}

// NewArena returns an Arena pre-sized for capacity UOp nodes.
func NewArena(capacity int) *Arena {
	return &Arena{
		nodes:      make([]uopNode, 0, capacity),
		cache:      make(map[uint64][]uint32, capacity),
		leaves:     make(map[uint32][]float32),
		provenance: make([]Phase, 0, capacity),
		realizeID:  arenaIDCounter.Add(1),
	}
}

// RealizeID is a process-unique id for this arena, used to scope the executor's
// stateful realize buffer cache (never reused, so a fresh arena always
// invalidates a prior arena's cache).
func (a *Arena) RealizeID() uint64 { return a.realizeID }

// RealizeGen is the current input-data generation; it bumps on every leaf
// SetData/Load so a cache keyed by (RealizeID, RealizeGen) is invalidated when
// any input changes within the arena's lifetime.
func (a *Arena) RealizeGen() uint64 { return a.realizeGen }

// BumpRealizeGen records that a leaf's data changed, invalidating any cached
// realize buffers for this arena.
func (a *Arena) BumpRealizeGen() { a.realizeGen++ }

// Reset discards all UOp nodes and clears the intern cache.
// Every UOp handle previously issued by this arena becomes invalid after Reset -
// the arena resets at the realize boundary and nothing holds indices across it.
func (a *Arena) Reset() {
	a.nodes = a.nodes[:0]
	a.cache = make(map[uint64][]uint32, cap(a.nodes))
	a.leaves = make(map[uint32][]float32)
	a.provenance = a.provenance[:0]
	a.phase = PhaseForward
}

// SetLeaf stores float32 data for the leaf node at idx.
// Called by tensor.SetData; data lifetime is tied to the arena.
func (a *Arena) SetLeaf(idx uint32, data []float32) {
	a.leaves[idx] = data
}

// Leaf returns the data previously stored by SetLeaf, if any.
func (a *Arena) Leaf(idx uint32) ([]float32, bool) {
	v, ok := a.leaves[idx]
	return v, ok
}

// Len returns the number of UOp nodes currently allocated in the arena.
func (a *Arena) Len() int { return len(a.nodes) }

// bypassInternSet is the set of ops that carry intrinsic identity and must never dedup.
//
// In tinygrad, UNIQUE always receives a fresh counter arg so it never aliases through
// the intern cache. In Go we make this structural guarantee explicit: bypass ops always
// allocate a fresh slot regardless of field values. BUFFER is included because each
// buffer represents a distinct allocation even when same-sized, and LUNIQUE is the
// per-bufferize variant with the same guarantee.
var bypassInternSet = map[Op]bool{
	OpUnique:  true,
	OpLUnique: true,
	OpBuffer:  true,
	// OpRange nodes are unique loop variables; two ranges with the same
	// (ID, Size, Type) from different kernels or realize calls must not
	// alias. Without this, hash-consing would collapse them to the same
	// arena index, corrupting getFusedRanges sort order.
	OpRange:       true,
	OpDefineLocal: true,
}

// New constructs or retrieves an interned UOp in a.
//
// All elements of src must belong to a; passing a UOp from another arena panics.
// arg and tag must be nil or one of the supported types (int64, float64, bool, string).
// Passing an unsupported type panics at construction time, keeping type errors local.
//
// For ops in bypassInternSet, a fresh node is always allocated regardless of fields.
func (a *Arena) New(op Op, dtype *DType, src []UOp, arg, tag any) UOp {
	srcIdx := make([]uint32, len(src))
	for i, s := range src {
		if !s.Valid() {
			panic("uop: invalid (zero-value) UOp passed as src")
		}
		if s.a != a {
			panic("uop: src UOp belongs to a different arena")
		}
		srcIdx[i] = s.idx
	}

	node := uopNode{op: op, dtype: dtype, src: srcIdx, arg: arg, tag: tag}

	if bypassInternSet[op] {
		return a.allocFresh(node)
	}
	return a.intern(node)
}

// At returns the UOp at the given arena index. The caller must ensure idx is valid.
func (a *Arena) At(idx uint32) UOp { return UOp{a: a, idx: idx} }

// SetPhase sets the current construction phase used by all subsequent New calls
// and returns the previous phase so callers can restore it with defer.
// Cache-hit nodes are not affected - first-construction wins.
func (a *Arena) SetPhase(p Phase) Phase {
	prev := a.phase
	a.phase = p
	return prev
}

// Provenance returns the Phase that was active when the node at idx was first
// allocated. Panics if idx is out of range (same contract as At).
func (a *Arena) Provenance(idx uint32) Phase {
	return a.provenance[idx]
}

func (a *Arena) allocFresh(node uopNode) UOp {
	idx := uint32(len(a.nodes))
	a.nodes = append(a.nodes, node)
	a.provenance = append(a.provenance, a.phase)
	return UOp{a: a, idx: idx}
}

func (a *Arena) intern(node uopNode) UOp {
	h := hashNode(node)
	for _, idx := range a.cache[h] {
		if equalNodes(a.nodes[idx], node) {
			return UOp{a: a, idx: idx}
		}
	}
	u := a.allocFresh(node)
	a.cache[h] = append(a.cache[h], u.idx)
	return u
}

// ── UOp handle ────────────────────────────────────────────────────────────────

// UOp is a lightweight, comparable handle for a node in an Arena.
// The zero value is invalid; always construct via Arena.New.
//
// Within one arena, u1 == u2 iff they reference the same node - which, by the
// interning invariant, equals structural equality. This makes UOp safe as a map key.
type UOp struct {
	a   *Arena
	idx uint32
}

// Valid reports whether u refers to a live arena node (non-zero-value handle).
func (u UOp) Valid() bool { return u.a != nil }

func (u UOp) node() uopNode { return u.a.nodes[u.idx] }

// Op returns the operation code.
func (u UOp) Op() Op { return u.node().op }

// DType returns the output data type (Dtypes.Void for control ops).
func (u UOp) DType() *DType { return u.node().dtype }

// NSrc returns the number of source UOps.
func (u UOp) NSrc() int { return len(u.node().src) }

// Src returns the i-th source UOp. Panics if i is out of range.
func (u UOp) Src(i int) UOp { return UOp{a: u.a, idx: u.node().src[i]} }

// Arg returns the static metadata payload. Nil for most ops.
func (u UOp) Arg() any { return u.node().arg }

// Tag returns the lowering classification tag. Nil in most nodes.
func (u UOp) Tag() any { return u.node().tag }

// Arena returns the arena this UOp belongs to.
func (u UOp) Arena() *Arena { return u.a }

// Index returns the raw arena index, useful for serialization or debug output.
func (u UOp) Index() uint32 { return u.idx }

func (u UOp) String() string {
	if !u.Valid() {
		return "<invalid UOp>"
	}
	n := u.node()
	if n.arg == nil && n.tag == nil {
		return fmt.Sprintf("UOp(%s, %s, srcs=%d)", n.op, n.dtype, len(n.src))
	}
	return fmt.Sprintf("UOp(%s, %s, srcs=%d, arg=%v, tag=%v)", n.op, n.dtype, len(n.src), n.arg, n.tag)
}

// ── hashing and structural equality ──────────────────────────────────────────

// StructuralKeys computes a bottom-up structural content hash for every node
// currently in a, returned as a slice indexed by arena position.
//
// Unlike hashNode (the intern hash), which mixes in raw arena indices of
// children, StructuralKeys mixes in the structural keys of children.  Two
// structurally identical subgraphs built at different arena positions receive
// the same key.
//
// The arena construction invariant guarantees every src index is strictly less
// than the containing node's index, so a single forward pass suffices.
// No reflection is used; panics on unknown arg/tag types (same contract as hashNode).
func StructuralKeys(a *Arena) []uint64 {
	n := a.Len()
	keys := make([]uint64, n)
	const (
		offset uint64 = 14695981039346656037
		prime  uint64 = 1099511628211
	)
	mix := func(h, v uint64) uint64 { return (h ^ v) * prime }
	for i := 0; i < n; i++ {
		node := a.nodes[i]
		h := offset
		h = mix(h, uint64(node.op))
		h = mix(h, node.dtype.StructuralHash())
		h = mix(h, uint64(len(node.src)))
		for _, srcIdx := range node.src {
			h = mix(h, keys[srcIdx]) // structural key of child, NOT arena index
		}
		h = hashArg(h, node.arg, prime)
		h = hashArg(h, node.tag, prime)
		keys[i] = h
	}
	return keys
}

// hashNode computes an FNV-1a hash of all fields that participate in the intern key.
// No reflection is used; only types explicitly listed in hashArg are supported.
func hashNode(n uopNode) uint64 {
	const offset uint64 = 14695981039346656037
	const prime uint64 = 1099511628211
	mix := func(h, v uint64) uint64 { return (h ^ v) * prime }

	h := offset
	h = mix(h, uint64(n.op))
	// DType is interned, so pointer identity equals structural equality.
	// uintptr conversion is safe here: we consume the value immediately and do not
	// store it; the GC does not move objects in current Go implementations.
	h = mix(h, uint64(uintptr(unsafe.Pointer(n.dtype))))
	h = mix(h, uint64(len(n.src)))
	for _, idx := range n.src {
		h = mix(h, uint64(idx))
	}
	h = hashArg(h, n.arg, prime)
	h = hashArg(h, n.tag, prime)
	return h
}

// ReduceArg is the arg payload for OpReduceAxis nodes.
// Op is the reduction operation (e.g. OpAdd for sum, OpMax for max);
// Axes is the sorted list of dimensions being reduced.
type ReduceArg struct {
	Op   Op
	Axes []int
}

// AxisType classifies a RANGE loop axis.
type AxisType int8

const (
	AxisLoop      AxisType = 0 // standard forward iteration
	AxisReduce    AxisType = 1 // inner reduction axis (accumulate, not store)
	AxisWorkgroup AxisType = 2 // split-out workgroup dimension
	AxisLocal     AxisType = 3 // split-out local dimension
	AxisUpcast    AxisType = 4 // per-thread unrolled stripe; size = micro-tile factor
	AxisVectorize AxisType = 5 // SIMD-width unrolled stripe; size = vector width (e.g. 4 for vec4<f32>)
)

// RangeArg is the arg payload for OpRange nodes.
// ID is a scheduler-assigned counter that uniquely identifies this loop variable
// within a kernel; Type classifies the axis kind.
//
// The exclusive upper bound lives in src[0] as a UOp expression - a Const for
// concrete sizes, a DefineVar (or expression over DefineVars) for symbolic.
// This matches tinygrad's master rangeify representation. Variable-slot
// assignment for the WGSL params_n uniform is computed per-kernel at codegen
// time via VariablesOf; it is not stored on the range itself.
type RangeArg struct {
	ID   int
	Type AxisType
}

// RangeBound returns the exclusive upper bound UOp of an OpRange node.
// The lower bound is implicit zero.
func RangeBound(r UOp) UOp { return r.Src(0) }

// RangeIsSymbolic reports whether r's bound is not a compile-time constant.
func RangeIsSymbolic(r UOp) bool { return r.Src(0).Op() != OpConst }

// RangeSize returns the concrete exclusive upper bound. Panics if the bound
// is symbolic; callers must gate with RangeIsSymbolic first.
func RangeSize(r UOp) int64 {
	b := r.Src(0)
	if b.Op() != OpConst {
		panic(fmt.Sprintf("uop: RangeSize called on symbolic range (bound op = %s)", b.Op()))
	}
	return b.Arg().(int64)
}

// VariablesOf returns the DefineVar UOps reachable from root, sorted by
// VarArg.Name. Each DefineVar appears at most once. The traversal is the
// portable analogue of tinygrad's UOp.variables() and is the supported way
// for codegen to discover the symbolic variables of a kernel.
func VariablesOf(root UOp) []UOp {
	if !root.Valid() {
		return nil
	}
	a := root.Arena()
	seen := make(map[uint32]bool)
	var out []UOp
	type frame struct {
		u       UOp
		nextSrc int
	}
	stack := []frame{{root, 0}}
	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u
		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}
		pushed := false
		for f.nextSrc < u.NSrc() {
			ch := u.Src(f.nextSrc)
			f.nextSrc++
			if !seen[ch.Index()] {
				stack = append(stack, frame{ch, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			if u.Op() == OpDefineVar {
				out = append(out, u)
			}
			stack = stack[:len(stack)-1]
		}
	}
	// Sort by VarArg.Name for deterministic codegen-time slot assignment.
	sortVarsByName(a, out)
	return out
}

// sortVarsByName sorts DefineVar UOps in-place by their VarArg.Name field.
// Insertion sort: variable count per kernel is small (≤4 today; cap is a
// codegen concern, not a uop one).
func sortVarsByName(_ *Arena, vs []UOp) {
	for i := 1; i < len(vs); i++ {
		j := i
		for j > 0 && vs[j-1].Arg().(VarArg).Name > vs[j].Arg().(VarArg).Name {
			vs[j-1], vs[j] = vs[j], vs[j-1]
			j--
		}
	}
}

// BufferizeArg is the arg payload for OpBufferize nodes.
// Removable marks speculative (soft) realize points that may be elided by
// the cost pass; false marks hard boundaries that must materialize.
type BufferizeArg struct {
	Removable bool
}

// VarArg is the arg payload for OpDefineVar nodes.
// Name is the symbolic variable's human-readable identifier.
// Two DefineVars with the same name and bounds intern to one node;
// different names produce distinct nodes.
type VarArg struct{ Name string }

// ShapeDim is one element of a ShapeSintArg.
// Sym=false: V is a concrete dimension size (VarName, Mul are zero).
// Sym=true: V must be 0 (SPEC §10); VarName is the DefineVar's name and Mul is
// the per-dim multiplier - actual dim size = Mul * binding[VarName].
// Mul=1 encodes a bare DefineVar bound; Mul>1 encodes a derived bound such as
// Mul(DefineVar, Const). Identity is structural (name + multiplier), not arena
// position, so the encoding is portable across arenas (SPEC §10 - fix in
// Option B Slice 4 of the recurring identity-as-allocation-position bug class).
type ShapeDim struct {
	V       int64
	Sym     bool
	VarName string
	Mul     int64
}

// ShapeSintArg is the arg payload for OpReshape and OpExpand nodes whose shape
// contains at least one symbolic dimension. Concrete dims carry their size in V;
// symbolic dims set Sym=true and (VarName, Mul) describe the size as a multiple
// of a named DefineVar's value. This type supplements the plain []int64 arg
// used for fully-concrete shapes.
type ShapeSintArg []ShapeDim

// PadSintArg is the arg payload for OpPad nodes whose pad amounts contain at
// least one symbolic value. Each element is the (lo, hi) pair for one axis,
// encoded as ShapeDim so concrete and symbolic amounts share the structural-key
// machinery (mirrors ShapeSintArg from Slice 1 / 4). Supplements the plain
// [][2]int64 arg used for fully-concrete Pad.
//
// SPEC §10 V-on-symbolic-dim invariant applies per element: when Sym=true,
// V must be 0 and (VarName, Mul) describe the amount as Mul * binding[VarName].
type PadSintArg [][2]ShapeDim

// ShrinkSintArg is the arg payload for OpShrink nodes whose [lo, hi) bounds
// contain at least one symbolic value. Element semantics match PadSintArg.
type ShrinkSintArg [][2]ShapeDim

// AffineTerm is one term of an affine-sum bound expression: contributes
// Mul * binding[VarName] to the result.
type AffineTerm struct {
	Mul     int64
	VarName string
}

// BoundDim is one element of a BoundExprArg. Concrete dims set V (Affine=nil).
// Symbolic dims set Affine to the per-term decomposition; total bound is
// sum(Terms[i].Mul * binding[Terms[i].VarName]) + Offset. This supersedes
// the ShapeDim (VarName, Mul) encoding for the buffer-output case where the
// bound is an Add of distinct DefineVars (the Slice 4 carried debt).
type BoundDim struct {
	V      int64 // concrete dim size (when Terms empty and Sym=false)
	Sym    bool
	Terms  []AffineTerm
	Offset int64
}

// BoundExprArg is the arg payload for OpBuffer nodes whose symbolic-dim
// bounds exceed the ShapeSintArg single-term encoding. Each element resolves
// to a concrete dim size at dispatch time by evaluating its affine sum
// against the runtime binding map.
//
// Cross-arena structural identity is preserved by mixing (Terms, Offset)
// into hashArg/equalArg - same approach as ShapeSintArg uses for ShapeDim.
type BoundExprArg []BoundDim

// BoundToAffine decomposes a symbolic-dim bound expression UOp into an
// affine sum: bound_value == Sum(Mul[i] * binding[VarName[i]]) + Offset at
// runtime. Supports the shapes accepted by SymBoundFactor plus OpAdd of
// affine subexpressions (Option B Slice 5). Returns ok=false on unsupported
// shapes so the caller can route via SymBoundFactor for narrow encodings or
// surface STOP for richer ones.
//
// Supported recursive structure:
//   - OpConst                       → (no terms, Offset=Const)
//   - OpDefineVar                   → ({Mul:1, VarName:name}, Offset=0)
//   - OpMul(DefineVar, Const) etc.  → ({Mul:Const, VarName:name}, Offset=0)
//   - OpMul(Const, Const)           → (no terms, Offset=product)
//   - OpAdd(a, b)                   → merge(a.Terms, b.Terms), Offset+=
//   - OpSub(a, b)                   → merge(a.Terms, neg(b.Terms)), Offset-=
//
// Like-named terms accumulate (e.g. n+n → 2n); zero-coefficient terms drop.
func BoundToAffine(u UOp) (terms []AffineTerm, offset int64, ok bool) {
	switch u.Op() {
	case OpConst:
		v, ok2 := u.Arg().(int64)
		if !ok2 {
			return nil, 0, false
		}
		return nil, v, true
	case OpDefineVar:
		return []AffineTerm{{Mul: 1, VarName: u.Arg().(VarArg).Name}}, 0, true
	case OpMul:
		if u.NSrc() != 2 {
			return nil, 0, false
		}
		a0, a1 := u.Src(0), u.Src(1)
		// Const * X (or X * Const) where X is affine.
		var c int64
		var other UOp
		switch {
		case a0.Op() == OpConst:
			cv, isInt := a0.Arg().(int64)
			if !isInt {
				return nil, 0, false
			}
			c = cv
			other = a1
		case a1.Op() == OpConst:
			cv, isInt := a1.Arg().(int64)
			if !isInt {
				return nil, 0, false
			}
			c = cv
			other = a0
		default:
			return nil, 0, false
		}
		otherTerms, otherOffset, ok2 := BoundToAffine(other)
		if !ok2 {
			return nil, 0, false
		}
		// Scale all terms by c via MulInt64Checked so a silent wrap on a
		// pathological Const operand gives up cleanly (matches the existing
		// ok=false channel) instead of producing a wrong-but-valid affine
		// encoding downstream.
		scaled := make([]AffineTerm, 0, len(otherTerms))
		for _, t := range otherTerms {
			scaledMul, mulOK := MulInt64Checked(t.Mul, c)
			if !mulOK {
				return nil, 0, false
			}
			if scaledMul != 0 {
				scaled = append(scaled, AffineTerm{Mul: scaledMul, VarName: t.VarName})
			}
		}
		scaledOffset, offOK := MulInt64Checked(otherOffset, c)
		if !offOK {
			return nil, 0, false
		}
		return scaled, scaledOffset, true
	case OpAdd:
		if u.NSrc() != 2 {
			return nil, 0, false
		}
		t1, o1, ok1 := BoundToAffine(u.Src(0))
		t2, o2, ok2 := BoundToAffine(u.Src(1))
		if !ok1 || !ok2 {
			return nil, 0, false
		}
		return mergeAffineTerms(t1, t2), o1 + o2, true
	case OpSub:
		if u.NSrc() != 2 {
			return nil, 0, false
		}
		t1, o1, ok1 := BoundToAffine(u.Src(0))
		t2, o2, ok2 := BoundToAffine(u.Src(1))
		if !ok1 || !ok2 {
			return nil, 0, false
		}
		negT2 := make([]AffineTerm, 0, len(t2))
		for _, t := range t2 {
			if t.Mul != 0 {
				negT2 = append(negT2, AffineTerm{Mul: -t.Mul, VarName: t.VarName})
			}
		}
		return mergeAffineTerms(t1, negT2), o1 - o2, true
	}
	return nil, 0, false
}

// mergeAffineTerms combines two affine term slices, accumulating coefficients
// on identical VarNames and dropping zero-coefficient results. Output order
// follows append-order from a then b (insertion); like terms in b collapse
// onto matching entries in a. Deterministic for structural-key stability.
func mergeAffineTerms(a, b []AffineTerm) []AffineTerm {
	if len(a) == 0 {
		out := make([]AffineTerm, 0, len(b))
		for _, t := range b {
			if t.Mul != 0 {
				out = append(out, t)
			}
		}
		return out
	}
	out := make([]AffineTerm, len(a))
	copy(out, a)
	for _, t := range b {
		if t.Mul == 0 {
			continue
		}
		found := false
		for i := range out {
			if out[i].VarName == t.VarName {
				out[i].Mul += t.Mul
				found = true
				break
			}
		}
		if !found {
			out = append(out, t)
		}
	}
	// Drop zero-coefficient terms (may arise from accumulation).
	pruned := out[:0]
	for _, t := range out {
		if t.Mul != 0 {
			pruned = append(pruned, t)
		}
	}
	return pruned
}

// SymBoundFactor decomposes a symbolic-dim bound expression UOp into
// (VarName, Mul) such that bound_value == Mul * binding[VarName] at runtime.
// Supported shapes (Slice 3 surface):
//   - bare OpDefineVar             → (Name, 1)
//   - OpMul(OpDefineVar, OpConst)  → (Name, const)
//   - OpMul(OpConst, OpDefineVar)  → (Name, const)
//
// Panics on any other shape - richer bound expressions require widening
// the ShapeDim encoding (VarName + Mul) and this helper together.
func SymBoundFactor(u UOp) (varName string, mul int64) {
	switch u.Op() {
	case OpDefineVar:
		return u.Arg().(VarArg).Name, 1
	case OpMul:
		if u.NSrc() == 2 {
			a0, a1 := u.Src(0), u.Src(1)
			switch {
			case a0.Op() == OpDefineVar && a1.Op() == OpConst:
				return a0.Arg().(VarArg).Name, a1.Arg().(int64)
			case a0.Op() == OpConst && a1.Op() == OpDefineVar:
				return a1.Arg().(VarArg).Name, a0.Arg().(int64)
			}
		}
	}
	panic(fmt.Sprintf("uop: SymBoundFactor: unsupported bound shape (op=%s nsrc=%d)", u.Op(), u.NSrc()))
}

// FindDefineVar returns the arena's OpDefineVar UOp whose VarArg.Name equals
// name, or (UOp{}, false) if no such variable exists. Used by consumers that
// receive a ShapeDim with VarName set and need to recover the originating
// DefineVar node (e.g. shapeSintArgToSints rebuilds bound expression UOps
// from arena-portable VarName + Mul encoding). Linear scan over arena nodes;
// O(arena.Len()). Two arenas with the same logical graph will return UOps
// with the same VarArg.Name but distinct arena indices - that's the point.
func (a *Arena) FindDefineVar(name string) (UOp, bool) {
	for i := range a.nodes {
		n := &a.nodes[i]
		if n.op != OpDefineVar {
			continue
		}
		if va, ok := n.arg.(VarArg); ok && va.Name == name {
			return UOp{a: a, idx: uint32(i)}, true
		}
	}
	return UOp{}, false
}

// DefineVar creates (or retrieves interned) a symbolic variable with name
// and inclusive integer bounds [min, max]. The resulting UOp has dtype Index
// and two Const srcs encoding the exclusive-upper interval: src[0]=min,
// src[1]=max+1. SPEC §10 inclusive-bounds invariant: user-supplied (min, max)
// are inclusive; the internal +1 lets the renderer emit loops as the
// canonical exclusive-upper `r < bound`. BoundsOf and shape.boundsOfUOp
// unwrap the +1 so every user-facing consumer reads inclusive [Min, Max].
func (a *Arena) DefineVar(name string, min, max int64) UOp {
	minC := a.New(OpConst, Dtypes.Index, nil, min, nil)
	maxC := a.New(OpConst, Dtypes.Index, nil, max+1, nil)
	return a.New(OpDefineVar, Dtypes.Index, []UOp{minC, maxC}, VarArg{Name: name}, nil)
}

// Bind records that the DefineVar v has been given concrete value val at this
// dispatch. The fold rule (Bind (DefineVar)) → Const(val) collapses it; after
// GraphRewrite with Symbolic the result is a Const node.
func (a *Arena) Bind(v UOp, val int64) UOp {
	return a.New(OpBind, v.DType(), []UOp{v}, val, nil)
}

// hashArg mixes a typed arg/tag value into h.
// Each type is tagged with a discriminator to prevent cross-type collisions.
// Adding a new arg type requires entries in both hashArg and equalArg.
func hashArg(h uint64, a any, prime uint64) uint64 {
	mix := func(h, v uint64) uint64 { return (h ^ v) * prime }
	switch v := a.(type) {
	case nil:
		return mix(mix(h, 0), 0xdead_cafe)
	case int64:
		return mix(mix(h, 1), uint64(v))
	case float64:
		return mix(mix(h, 2), math.Float64bits(v))
	case bool:
		if v {
			return mix(mix(h, 3), 1)
		}
		return mix(mix(h, 3), 0)
	case string:
		h = mix(h, 4) // type discriminator
		for i := 0; i < len(v); i++ {
			h = mix(h, uint64(v[i]))
		}
		return h
	case []int64:
		h = mix(h, 5)
		h = mix(h, uint64(len(v)))
		for _, x := range v {
			h = mix(h, uint64(x))
		}
		return h
	case [][2]int64:
		h = mix(h, 6)
		h = mix(h, uint64(len(v)))
		for _, p := range v {
			h = mix(h, uint64(p[0]))
			h = mix(h, uint64(p[1]))
		}
		return h
	case ReduceArg:
		h = mix(h, 7)
		h = mix(h, uint64(v.Op))
		h = mix(h, uint64(len(v.Axes)))
		for _, ax := range v.Axes {
			h = mix(h, uint64(ax))
		}
		return h
	case RangeArg:
		h = mix(h, 8)
		h = mix(h, uint64(v.ID))
		h = mix(h, uint64(v.Type))
		return h
	case BufferizeArg:
		h = mix(h, 9)
		if v.Removable {
			return mix(h, 1)
		}
		return mix(h, 0)
	case Op:
		// kernel-level REDUCE carries the accumulation op as its arg
		h = mix(h, 10)
		return mix(h, uint64(v))
	case KernelInfo:
		h = mix(h, 11)
		return mix(h, uint64(v.NumParams))
	case VarArg:
		h = mix(h, 12)
		for i := 0; i < len(v.Name); i++ {
			h = mix(h, uint64(v.Name[i]))
		}
		return h
	case ShapeSintArg:
		h = mix(h, 13)
		h = mix(h, uint64(len(v)))
		for _, d := range v {
			h = mixShapeDim(h, d, mix)
		}
		return h
	case PadSintArg:
		h = mix(h, 14)
		h = mix(h, uint64(len(v)))
		for _, p := range v {
			h = mixShapeDim(h, p[0], mix)
			h = mixShapeDim(h, p[1], mix)
		}
		return h
	case ShrinkSintArg:
		h = mix(h, 15)
		h = mix(h, uint64(len(v)))
		for _, p := range v {
			h = mixShapeDim(h, p[0], mix)
			h = mixShapeDim(h, p[1], mix)
		}
		return h
	case BoundExprArg:
		h = mix(h, 16)
		h = mix(h, uint64(len(v)))
		for _, d := range v {
			if d.Sym {
				h = mix(h, 1)
				h = mix(h, uint64(len(d.Terms)))
				for _, t := range d.Terms {
					h = mix(h, uint64(t.Mul))
					for i := 0; i < len(t.VarName); i++ {
						h = mix(h, uint64(t.VarName[i]))
					}
					h = mix(h, uint64(len(t.VarName)))
				}
				h = mix(h, uint64(d.Offset))
			} else {
				h = mix(h, 0)
				h = mix(h, uint64(d.V))
			}
		}
		return h
	default:
		panic(fmt.Sprintf("uop: unsupported arg type %T; add it to hashArg and equalArg", a))
	}
}

// mixShapeDim mixes a single ShapeDim into the hash, mirroring the
// per-element logic from the ShapeSintArg case. Centralised so PadSintArg /
// ShrinkSintArg stay byte-identical with the established encoding (and so
// any future encoding tweak only changes one place).
func mixShapeDim(h uint64, d ShapeDim, mix func(uint64, uint64) uint64) uint64 {
	if d.Sym {
		h = mix(h, 1)
		for i := 0; i < len(d.VarName); i++ {
			h = mix(h, uint64(d.VarName[i]))
		}
		h = mix(h, uint64(len(d.VarName)))
		h = mix(h, uint64(d.Mul))
	} else {
		h = mix(h, 0)
		h = mix(h, uint64(d.V))
	}
	return h
}

// equalNodes reports whether two uopNodes are structurally equal.
// Called only when hashes match; must handle all field types correctly.
func equalNodes(a, b uopNode) bool {
	if a.op != b.op || a.dtype != b.dtype || len(a.src) != len(b.src) {
		return false
	}
	for i := range a.src {
		if a.src[i] != b.src[i] {
			return false
		}
	}
	return equalArg(a.arg, b.arg) && equalArg(a.tag, b.tag)
}

// equalArg reports whether two arg/tag values are equal under the intern semantics.
// NaN float64 values with identical bit patterns are considered equal (same constant).
func equalArg(a, b any) bool {
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
	case []int64:
		bv, ok := b.([]int64)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if av[i] != bv[i] {
				return false
			}
		}
		return true
	case [][2]int64:
		bv, ok := b.([][2]int64)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if av[i] != bv[i] {
				return false
			}
		}
		return true
	case ReduceArg:
		bv, ok := b.(ReduceArg)
		if !ok || av.Op != bv.Op || len(av.Axes) != len(bv.Axes) {
			return false
		}
		for i := range av.Axes {
			if av.Axes[i] != bv.Axes[i] {
				return false
			}
		}
		return true
	case RangeArg:
		bv, ok := b.(RangeArg)
		return ok && av == bv
	case BufferizeArg:
		bv, ok := b.(BufferizeArg)
		return ok && av == bv
	case Op:
		bv, ok := b.(Op)
		return ok && av == bv
	case KernelInfo:
		bv, ok := b.(KernelInfo)
		return ok && av == bv
	case VarArg:
		bv, ok := b.(VarArg)
		return ok && av == bv
	case ShapeSintArg:
		bv, ok := b.(ShapeSintArg)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if av[i] != bv[i] {
				return false
			}
		}
		return true
	case PadSintArg:
		bv, ok := b.(PadSintArg)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if av[i] != bv[i] {
				return false
			}
		}
		return true
	case ShrinkSintArg:
		bv, ok := b.(ShrinkSintArg)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if av[i] != bv[i] {
				return false
			}
		}
		return true
	case BoundExprArg:
		bv, ok := b.(BoundExprArg)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if av[i].V != bv[i].V || av[i].Sym != bv[i].Sym ||
				av[i].Offset != bv[i].Offset || len(av[i].Terms) != len(bv[i].Terms) {
				return false
			}
			for j := range av[i].Terms {
				if av[i].Terms[j] != bv[i].Terms[j] {
					return false
				}
			}
		}
		return true
	default:
		panic(fmt.Sprintf("uop: unsupported arg type %T; add it to hashArg and equalArg", a))
	}
}
