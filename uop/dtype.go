package uop

import (
	"fmt"
	"sync"
)

// AddrSpace is the memory address space of a pointer dtype.
type AddrSpace int8

const (
	Global AddrSpace = iota // GPU global memory (the default)
	Local                   // GPU shared / threadgroup memory
	Reg                     // register-level
)

func (a AddrSpace) String() string {
	switch a {
	case Global:
		return "Global"
	case Local:
		return "Local"
	case Reg:
		return "Reg"
	default:
		return fmt.Sprintf("AddrSpace(%d)", int(a))
	}
}

// DType is an interned, immutable data-type descriptor.
//
// All instances are obtained through the Dtypes singletons or the Vec / Ptr
// methods — never constructed directly. Because all instances are interned,
// pointer equality (==) is structural equality. *DType is therefore safe as a
// map key and as a field inside a struct used as a map key (the UOp interning
// key in a later step relies on this property).
type DType struct {
	priority int
	bitsize  int    // total bits: bitsize-per-lane × count for vector types
	name     string // device rendering name, e.g. "float", "signed char"
	fmt      string // struct-pack format char; "" if none
	count    int    // 1 = scalar dtype; >1 = vector lane count

	// scalar is nil for scalar dtypes; for vector dtypes it points to the
	// per-lane element dtype.
	scalar *DType

	// Pointer-type fields — only meaningful when isPtr is true.
	isPtr     bool
	base      *DType    // element dtype (what is being pointed to)
	addrSpace AddrSpace // address space of the pointer
	ptrVec    int       // ptr vectorization width (1 = non-vectorized)
	ptrSize   int       // element count of the pointed-to buffer; -1 = unbounded
}

// ── properties ────────────────────────────────────────────────────────────────

// ItemSize returns the total byte width: per-lane bytes × count.
func (d *DType) ItemSize() int { return (d.bitsize + 7) / 8 }

// BitSize returns the total bit width.
func (d *DType) BitSize() int { return d.bitsize }

// Name returns the device rendering name.
func (d *DType) Name() string { return d.name }

// Count returns the vector lane count (1 for scalar dtypes).
func (d *DType) Count() int { return d.count }

// IsPtr reports whether d is a pointer dtype.
func (d *DType) IsPtr() bool { return d.isPtr }

// AddrSpaceOf returns the address space (meaningful only for pointer dtypes).
func (d *DType) AddrSpaceOf() AddrSpace { return d.addrSpace }

// PtrSize returns the element-count bound on the pointed-to buffer (-1 = unbounded).
func (d *DType) PtrSize() int { return d.ptrSize }

// Scalar returns the per-lane element dtype for vector dtypes, or d itself for
// scalar and pointer dtypes.
func (d *DType) Scalar() *DType {
	if d.scalar != nil {
		return d.scalar
	}
	return d
}

// Base returns the element dtype for pointer dtypes, or d itself for non-pointer dtypes.
func (d *DType) Base() *DType {
	if d.isPtr {
		return d.base
	}
	return d
}

// ── type predicates ───────────────────────────────────────────────────────────

// IsFloat reports whether d (or its scalar element, for vectors) is a
// floating-point type.
func (d *DType) IsFloat() bool {
	s := d.Scalar()
	return s == Dtypes.Float16 || s == Dtypes.BFloat16 ||
		s == Dtypes.Float32 || s == Dtypes.Float64 ||
		s == Dtypes.FP8E4M3 || s == Dtypes.FP8E5M2
}

// IsInt reports whether d (or its scalar element) is an integer type
// (signed or unsigned, including the index dtype).
func (d *DType) IsInt() bool {
	s := d.Scalar()
	switch s {
	case Dtypes.Int8, Dtypes.UInt8, Dtypes.Int16, Dtypes.UInt16,
		Dtypes.Int32, Dtypes.UInt32, Dtypes.Int64, Dtypes.UInt64,
		Dtypes.Index:
		return true
	}
	return false
}

// IsUnsigned reports whether d (or its scalar element) is an unsigned integer type.
func (d *DType) IsUnsigned() bool {
	s := d.Scalar()
	switch s {
	case Dtypes.UInt8, Dtypes.UInt16, Dtypes.UInt32, Dtypes.UInt64:
		return true
	}
	return false
}

// IsBool reports whether d is the bool dtype.
func (d *DType) IsBool() bool { return d.Scalar() == Dtypes.Bool }

// ── construction ──────────────────────────────────────────────────────────────

// Vec returns a vector dtype with sz lanes of element type d.
// Returns d unchanged for sz == 1 or when d is Dtypes.Void.
// Panics if d is already a vector dtype.
func (d *DType) Vec(sz int) *DType {
	if sz == 1 || d == Dtypes.Void {
		return d
	}
	if d.count != 1 {
		panic(fmt.Sprintf("uop: cannot vectorize %s: already a vector (count=%d)", d, d.count))
	}
	return internDType(DType{
		priority: d.priority,
		bitsize:  d.bitsize * sz,
		name:     d.name,
		fmt:      "",
		count:    sz,
		scalar:   d,
	})
}

// Ptr returns a pointer dtype that points to elements of type d.
// Pass size = -1 for an unbounded pointer (the common case).
// Panics if d is already a pointer dtype.
func (d *DType) Ptr(size int, addrSpace AddrSpace) *DType {
	if d.isPtr {
		panic(fmt.Sprintf("uop: cannot make pointer to pointer dtype %s", d))
	}
	return internDType(DType{
		priority:  d.priority,
		bitsize:   d.bitsize,
		name:      d.name,
		fmt:       d.fmt,
		count:     d.count,
		scalar:    d.scalar,
		isPtr:     true,
		base:      d,
		addrSpace: addrSpace,
		ptrVec:    1,
		ptrSize:   size,
	})
}

// ── string representation ─────────────────────────────────────────────────────

// String returns a human-readable representation suitable for debugging and
// error messages.
func (d *DType) String() string {
	if d == nil {
		return "<nil dtype>"
	}
	baseName := d.name
	if d.scalar != nil {
		baseName = d.scalar.name
	}
	if d.isPtr {
		s := baseName + ".ptr"
		if d.ptrSize != -1 {
			s = fmt.Sprintf("%s(%d)", s, d.ptrSize)
		}
		if d.addrSpace != Global {
			s = fmt.Sprintf("%s[%s]", s, d.addrSpace)
		}
		if d.ptrVec != 1 {
			s = fmt.Sprintf("%s.vec(%d)", s, d.ptrVec)
		}
		return s
	}
	if d.count != 1 {
		return fmt.Sprintf("%s×%d", baseName, d.count)
	}
	return baseName
}

// ── interning ─────────────────────────────────────────────────────────────────

// dtypeKey is the equality key used by the intern cache.
// All fields are comparable; pointer fields compare by address, which is
// correct because pointers themselves are interned.
type dtypeKey struct {
	priority  int
	bitsize   int
	name      string
	fmt       string
	count     int
	scalar    *DType
	isPtr     bool
	base      *DType
	addrSpace AddrSpace
	ptrVec    int
	ptrSize   int
}

var (
	dtypeCacheMu sync.Mutex
	dtypeCache   = map[dtypeKey]*DType{}
)

// internDType returns the canonical pointer for d.
// Concurrent calls with an identical key return the same pointer.
func internDType(d DType) *DType {
	key := dtypeKey{
		d.priority, d.bitsize, d.name, d.fmt, d.count,
		d.scalar, d.isPtr, d.base, d.addrSpace, d.ptrVec, d.ptrSize,
	}
	dtypeCacheMu.Lock()
	defer dtypeCacheMu.Unlock()
	if p, ok := dtypeCache[key]; ok {
		return p
	}
	p := &d
	dtypeCache[key] = p
	return p
}

// newScalar creates and interns a scalar (count=1, non-ptr) DType.
func newScalar(priority, bitsize int, name, fmtStr string) *DType {
	return internDType(DType{
		priority: priority,
		bitsize:  bitsize,
		name:     name,
		fmt:      fmtStr,
		count:    1,
	})
}

// ── dtype singletons ──────────────────────────────────────────────────────────

// Dtypes provides named singleton instances for all built-in scalar dtypes.
// These are the canonical entry points for constructing dtypes; vector and
// pointer variants are obtained via the Vec and Ptr methods.
var Dtypes = struct {
	Void  *DType // no value; dtype of control ops
	Index *DType // platform-sized indexing integer (800-bit "priority" sentinel)

	Bool *DType

	Int8   *DType
	UInt8  *DType
	Int16  *DType
	UInt16 *DType
	Int32  *DType
	UInt32 *DType
	Int64  *DType
	UInt64 *DType

	// FP8 variants — present in the dtype system and promotion lattice;
	// runtime lowering is deferred (SPEC §1.3).
	FP8E4M3 *DType
	FP8E5M2 *DType

	Float16  *DType
	BFloat16 *DType
	Float32  *DType
	Float64  *DType

	// Convenience aliases matching tinygrad naming.
	Float  *DType // = Float32
	Half   *DType // = Float16
	Double *DType // = Float64
	Int    *DType // = Int32
	Char   *DType // = Int8
	Short  *DType // = Int16
	Long   *DType // = Int64
	UChar  *DType // = UInt8
	UShort *DType // = UInt16
	UInt   *DType // = UInt32
	ULong  *DType // = UInt64
}{
	Void:  newScalar(-1, 0, "void", ""),
	Index: newScalar(-1, 800, "index", ""),
	Bool:  newScalar(0, 1, "bool", "?"),

	// Names match C/WGSL rendering names used by tinygrad's codegen.
	Int8:   newScalar(1, 8, "signed char", "b"),
	UInt8:  newScalar(2, 8, "unsigned char", "B"),
	Int16:  newScalar(3, 16, "short", "h"),
	UInt16: newScalar(4, 16, "unsigned short", "H"),
	Int32:  newScalar(5, 32, "int", "i"),
	UInt32: newScalar(6, 32, "unsigned int", "I"),
	Int64:  newScalar(7, 64, "long", "q"),
	UInt64: newScalar(8, 64, "unsigned long", "Q"),

	FP8E4M3: newScalar(9, 8, "float8_e4m3", ""),
	FP8E5M2: newScalar(10, 8, "float8_e5m2", ""),

	Float16:  newScalar(11, 16, "half", "e"),
	BFloat16: newScalar(12, 16, "__bf16", ""),
	Float32:  newScalar(13, 32, "float", "f"),
	Float64:  newScalar(14, 64, "double", "d"),
}

func init() {
	// Aliases — set after primary singletons are initialised.
	Dtypes.Float = Dtypes.Float32
	Dtypes.Half = Dtypes.Float16
	Dtypes.Double = Dtypes.Float64
	Dtypes.Int = Dtypes.Int32
	Dtypes.Char = Dtypes.Int8
	Dtypes.Short = Dtypes.Int16
	Dtypes.Long = Dtypes.Int64
	Dtypes.UChar = Dtypes.UInt8
	Dtypes.UShort = Dtypes.UInt16
	Dtypes.UInt = Dtypes.UInt32
	Dtypes.ULong = Dtypes.UInt64

	buildPromoLattice()
}

// ── type promotion ────────────────────────────────────────────────────────────

// promoLattice maps each scalar dtype to its immediate promotion targets.
// Built in init() after the Dtypes singletons are available.
var promoLattice map[*DType][]*DType

func buildPromoLattice() {
	D := &Dtypes
	promoLattice = map[*DType][]*DType{
		D.Bool:     {D.Int8, D.UInt8},
		D.Int8:     {D.Int16},
		D.Int16:    {D.Int32},
		D.Int32:    {D.Int64},
		D.Int64:    {D.UInt64},
		D.UInt8:    {D.Int16, D.UInt16},
		D.UInt16:   {D.Int32, D.UInt32},
		D.UInt32:   {D.Int64, D.UInt64},
		D.UInt64:   {D.FP8E4M3, D.FP8E5M2},
		D.FP8E5M2:  {D.Float16, D.BFloat16},
		D.FP8E4M3:  {D.Float16, D.BFloat16},
		D.Float16:  {D.Float32},
		D.BFloat16: {D.Float32},
		D.Float32:  {D.Float64},
		// Float64 is the lattice top; no outgoing edges.
	}
}

// CmpDType returns -1, 0, or 1 for a total order on DType values.
//
// Order for scalar/vector dtypes: (isPtr, priority, bitsize, name, fmt, count).
// For pointer dtypes the additional fields (addrSpace, ptrVec, ptrSize, base)
// are compared in that order after the shared fields. nil is less than any
// non-nil dtype. Because DType is interned, a == b implies structural equality
// and returns 0 immediately without inspecting any fields.
func CmpDType(a, b *DType) int {
	if a == b {
		return 0
	}
	if a == nil {
		return -1
	}
	if b == nil {
		return 1
	}
	if a.isPtr != b.isPtr {
		if !a.isPtr {
			return -1
		}
		return 1
	}
	if a.priority != b.priority {
		if a.priority < b.priority {
			return -1
		}
		return 1
	}
	if a.bitsize != b.bitsize {
		if a.bitsize < b.bitsize {
			return -1
		}
		return 1
	}
	if a.name != b.name {
		if a.name < b.name {
			return -1
		}
		return 1
	}
	if a.fmt != b.fmt {
		if a.fmt < b.fmt {
			return -1
		}
		return 1
	}
	if a.count != b.count {
		if a.count < b.count {
			return -1
		}
		return 1
	}
	if a.isPtr {
		if a.addrSpace != b.addrSpace {
			if a.addrSpace < b.addrSpace {
				return -1
			}
			return 1
		}
		if a.ptrVec != b.ptrVec {
			if a.ptrVec < b.ptrVec {
				return -1
			}
			return 1
		}
		if a.ptrSize != b.ptrSize {
			if a.ptrSize < b.ptrSize {
				return -1
			}
			return 1
		}
		return CmpDType(a.base, b.base)
	}
	return 0
}

// ancestors returns the set of all dtypes reachable from d (inclusive) via the
// promotion lattice, i.e. all types that d can be promoted to.
func ancestors(d *DType) map[*DType]struct{} {
	result := map[*DType]struct{}{}
	queue := []*DType{d.Scalar()}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if _, seen := result[cur]; seen {
			continue
		}
		result[cur] = struct{}{}
		for _, parent := range promoLattice[cur] {
			queue = append(queue, parent)
		}
	}
	return result
}

// LeastUpperDType returns the smallest dtype that all of the given dtypes can
// be promoted to without loss of precision, mirroring tinygrad's
// least_upper_dtype.  Returns nil if ds is empty or if the intersection of
// ancestor sets is empty (which should not occur for well-formed inputs).
func LeastUpperDType(ds ...*DType) *DType {
	if len(ds) == 0 {
		return nil
	}
	common := ancestors(ds[0])
	for _, d := range ds[1:] {
		a := ancestors(d)
		for k := range common {
			if _, ok := a[k]; !ok {
				delete(common, k)
			}
		}
	}
	var best *DType
	for d := range common {
		if best == nil || CmpDType(d, best) < 0 {
			best = d
		}
	}
	return best
}
