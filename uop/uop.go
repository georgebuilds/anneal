package uop

import (
	"fmt"
	"math"
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
}

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
	}
}

// Reset discards all UOp nodes and clears the intern cache.
// Every UOp handle previously issued by this arena becomes invalid after Reset —
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
	OpRange: true,
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
// Cache-hit nodes are not affected — first-construction wins.
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
// Within one arena, u1 == u2 iff they reference the same node — which, by the
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
)

// RangeArg is the arg payload for OpRange nodes.
// ID is a scheduler-assigned counter that uniquely identifies this loop variable
// within a kernel; Size is the exclusive upper bound ([0, Size)).
//
// Symbolic shapes spike (SLICE 1): when Symbolic=true the bound is not known at
// compile time. Size is ignored; the runtime value is read from the kernel's
// params_n storage buffer at slot SymParamIdx. The static path (Symbolic=false)
// is entirely unaffected — the bool zero-value keeps all existing kernels intact.
type RangeArg struct {
	ID          int
	Size        int64
	Type        AxisType
	Symbolic    bool   // true → read bound from params_n[SymParamIdx] at dispatch
	SymParamIdx int    // index into the per-kernel params_n buffer (only when Symbolic)
	VarName     string // DefineVar name for symbolic ranges; "" for static
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
// Sym=false: V is a concrete dimension size.
// Sym=true: VarIdx is the arena index of the corresponding DefineVar UOp.
type ShapeDim struct {
	V      int64
	Sym    bool
	VarIdx uint32
}

// ShapeSintArg is the arg payload for OpReshape and OpExpand nodes whose shape
// contains at least one symbolic dimension. Concrete dims carry their size in V;
// symbolic dims set Sym=true and VarIdx to the DefineVar UOp's arena index.
// This type supplements the plain []int64 arg used for fully-concrete shapes.
type ShapeSintArg []ShapeDim

// DefineVar creates (or retrieves interned) a symbolic variable with name
// and inclusive integer bounds [min, max]. The resulting UOp has dtype Index
// and two Const srcs encoding the exclusive interval: src[0]=min, src[1]=max+1.
// This encoding matches what BoundsOf expects: it returns [src[0], src[1].Max-1].
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
		h = mix(h, uint64(v.Size))
		h = mix(h, uint64(v.Type))
		if v.Symbolic {
			h = mix(h, 1)
		} else {
			h = mix(h, 0)
		}
		h = mix(h, uint64(v.SymParamIdx))
		for i := 0; i < len(v.VarName); i++ {
			h = mix(h, uint64(v.VarName[i]))
		}
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
			if d.Sym {
				h = mix(h, 1)
				h = mix(h, uint64(d.VarIdx))
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
	default:
		panic(fmt.Sprintf("uop: unsupported arg type %T; add it to hashArg and equalArg", a))
	}
}
