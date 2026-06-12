package uop

import (
	"fmt"
	"math"
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

	// isImage marks an image-storage dtype: behaves identically to its scalar
	// peer (e.g. Float32) for ALU, promotion, autodiff, and comparisons; the
	// only effect is on the GPU storage layout (buffer is bound as
	// array<vec4<f32>> rather than array<f32>; logical element i maps to
	// buffer[i/4][i%4]). Storage layout choice, not a compute-level type.
	isImage bool
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
// floating-point type. Image-storage dtypes share their scalar peer's
// compute identity and are also counted as float.
func (d *DType) IsFloat() bool {
	s := d.Scalar()
	return s == Dtypes.Float16 || s == Dtypes.BFloat16 ||
		s == Dtypes.Float32 || s == Dtypes.Float64 ||
		s == Dtypes.FP8E4M3 || s == Dtypes.FP8E5M2 ||
		s == Dtypes.ImageFloat32
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

// IsImage reports whether d is an image-storage dtype. Image dtypes behave
// as their scalar peer for compute (ALU, autodiff, promotion) and only
// differ in GPU buffer layout (array<vec4<f32>> instead of array<f32>;
// logical element i lives at buffer[i/4][i%4]). The vec4 packing is in the
// storage layout, not in the dtype lane count, so IsImage is a per-dtype
// predicate distinct from Count > 1.
func (d *DType) IsImage() bool {
	if d == nil {
		return false
	}
	return d.isImage
}

// ── construction ──────────────────────────────────────────────────────────────

// Vec returns a vector dtype with sz lanes of element type d.
// Returns d unchanged for sz == 1 or when d is Dtypes.Void.
// Panics if d is already a vector dtype, or if d is an image dtype: the
// vec4 packing of an image lives in the storage layout, not the dtype lane
// count, so Vec on image dtypes is structurally meaningless.
func (d *DType) Vec(sz int) *DType {
	if sz == 1 || d == Dtypes.Void {
		return d
	}
	if d.isImage {
		panic(fmt.Sprintf("uop: cannot vectorize image dtype %s: vec4 packing is in storage layout, not dtype lane count", d))
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
	isImage   bool
}

var (
	dtypeCacheMu sync.Mutex
	dtypeCache   = map[dtypeKey]*DType{}
)

// internDType returns the canonical pointer for d.
// Concurrent calls with an identical key return the same pointer.
func internDType(d DType) *DType {
	key := dtypeKey{ //nolint:staticcheck // S1016: explicit field list intentional, dtype interning key composition stays visible at call site
		d.priority, d.bitsize, d.name, d.fmt, d.count,
		d.scalar, d.isPtr, d.base, d.addrSpace, d.ptrVec, d.ptrSize,
		d.isImage,
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

// newImageScalar creates and interns an image-storage scalar DType. The
// fields are intentionally identical to the matching newScalar call so the
// only structural difference is the isImage flag; the intern cache keys on
// the full struct so the two singletons are distinct.
func newImageScalar(priority, bitsize int, name, fmtStr string) *DType {
	return internDType(DType{
		priority: priority,
		bitsize:  bitsize,
		name:     name,
		fmt:      fmtStr,
		count:    1,
		isImage:  true,
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

	// FP8 variants (OCP FP8: e4m3fn and e5m2) — storage-only dtypes with f32
	// compute, on the bf16 decoded-storage scheme. Conversion helpers below;
	// WGSL store narrowing in codegen/wgsl.go (SPEC §11.3).
	FP8E4M3 *DType
	FP8E5M2 *DType

	Float16  *DType
	BFloat16 *DType
	Float32  *DType
	Float64  *DType

	// Image-storage variants — behave as their scalar peer for ALU, autodiff,
	// and promotion. The only difference is GPU buffer storage layout (the
	// buffer is bound as array<vec4<f32>> rather than array<f32>; logical
	// element i lives at buffer[i/4][i%4]). See DType.IsImage and SPEC §1.3.
	ImageFloat32 *DType

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

	// Image-storage dtype: same priority/bitsize/name as Float32 but with the
	// isImage flag set so the intern cache discriminates the two singletons.
	// All other fields match Float32 so the promotion lattice can treat them
	// as interchangeable peers (LeastUpperDType collapses to Float32).
	ImageFloat32: newImageScalar(13, 32, "float", "f"),
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

// StructuralHash returns a stable hash of d's field values, independent of the
// pointer's allocation address. Used by StructuralKeys so that cross-build node
// keys are deterministic. This must NOT replace hashNode, which correctly uses
// pointer identity for the intern hash (faster and correct within one process).
//
// The DType graph is acyclic and at most one level deep (vector's scalar →
// scalar dtype; pointer's base → non-pointer dtype), so the recursive calls
// always terminate without memoisation.
func (d *DType) StructuralHash() uint64 {
	const (
		offset      uint64 = 14695981039346656037
		prime       uint64 = 1099511628211
		nilSentinel uint64 = 0xcafebabe00000001
	)
	if d == nil {
		return nilSentinel
	}
	mix := func(h, v uint64) uint64 { return (h ^ v) * prime }
	h := offset
	h = mix(h, uint64(int64(d.priority))) // int64 cast preserves sign bits for -1
	h = mix(h, uint64(d.bitsize))
	for i := 0; i < len(d.name); i++ {
		h = mix(h, uint64(d.name[i]))
	}
	for i := 0; i < len(d.fmt); i++ {
		h = mix(h, uint64(d.fmt[i]))
	}
	h = mix(h, uint64(d.count))
	h = mix(h, d.scalar.StructuralHash()) // nil for scalar dtypes → nilSentinel
	if d.isPtr {
		h = mix(h, 1)
		h = mix(h, d.base.StructuralHash()) // base is always non-pointer
		h = mix(h, uint64(d.addrSpace))
		h = mix(h, uint64(d.ptrVec))
		h = mix(h, uint64(int64(d.ptrSize))) // -1 → ^uint64(0), distinct from 0
	} else {
		h = mix(h, 0)
	}
	// Image-storage flag: pure-field mix-in keeps StructuralHash a function
	// of dtype identity (SPEC §10 portability invariant). Two dtypes with
	// the same scalar identity but differing isImage must hash distinctly so
	// the cross-arena identity test catches a missed field.
	if d.isImage {
		h = mix(h, 1)
	} else {
		h = mix(h, 0)
	}
	return h
}

// ── conversion and quantization ───────────────────────────────────────────────

// Float32ToFloat16 converts a float32 to its nearest IEEE 754 half-precision bit pattern
// using round-to-nearest-even (RTNE) rounding.
//
// Implementation derived from the common bit-twiddling approach for IEEE 754
// narrowing, ensuring that NaN-ness is preserved and ties round to even.
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits >> 31)
	exp := int32((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	switch {
	case exp == 128: // NaN or Inf
		if frac != 0 {
			// NaN: preserve NaN-ness. Shift mantissa and ensure at least one bit
			// is set so it doesn't become Infinity.
			m16 := uint16(frac >> 13)
			if m16 == 0 {
				m16 = 1
			}
			return (sign << 15) | 0x7C00 | m16
		}
		return (sign << 15) | 0x7C00 // Inf
	case exp > 15:
		// Overflow → ±Inf in f16.
		return (sign << 15) | 0x7C00
	case exp < -25:
		// Underflow → ±zero. (Halfway point for RTNE to zero is 2^-25).
		return sign << 15
	case exp < -14:
		// Subnormal f16. Round with RTNE.
		// bit 23 is implicit 1.
		m := frac | 0x800000
		shift := uint32(-1 - exp) // 14 for e=-15, 23 for e=-24
		m_round := m >> (shift - 1)
		if (m_round&1 != 0) && (m_round&2 != 0 || (m&((1<<(shift-1))-1) != 0)) {
			m_round += 2
		}
		return (sign << 15) | uint16(m_round>>1)
	default:
		// Normal f16. Round with RTNE.
		m_round := frac >> 12
		if (m_round&1 != 0) && (m_round&2 != 0 || (frac&0xFFF != 0)) {
			m_round += 2
		}
		m16 := uint16(m_round >> 1)
		e16 := exp + 15
		if m16&0x400 != 0 {
			// Rounding overflowed into exponent.
			m16 &= 0x3FF
			e16++
		}
		if e16 > 30 {
			return (sign << 15) | 0x7C00 // Rounded to infinity
		}
		return (sign << 15) | (uint16(e16) << 10) | m16
	}
}

// Float32ToBFloat16 narrows a float32 to its nearest bfloat16 bit pattern
// using round-to-nearest-even (RTNE). NaN inputs are canonicalized to the
// bf16 quiet-NaN bit pattern (0x7FC0); the algorithm is the PyTorch /
// TensorFlow / Eigen reference (c10::detail::round_to_nearest_even).
//
// Tie-to-even is encoded by adding ((bits>>16)&1) + 0x7FFF as a rounding
// bias before truncating: when the discarded low 16 bits are exactly
// halfway (0x8000), the carry happens only if the kept LSB is already 1,
// rounding ties to the even neighbour. The NaN guard is required because
// the bias formula can shift small-mantissa NaNs into the all-zero
// mantissa pattern (Inf), so canonical qNaN is returned instead.
func Float32ToBFloat16(v float32) uint16 {
	if v != v {
		return 0x7FC0
	}
	u := math.Float32bits(v)
	bias := ((u >> 16) & 1) + 0x7FFF
	return uint16((u + bias) >> 16)
}

// BFloat16ToFloat32 widens a bfloat16 bit pattern to float32. bf16 is the
// high 16 bits of an f32 IEEE-754 encoding; the low 16 mantissa bits zero-
// fill, so this direction is lossless.
func BFloat16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// Float16ToFloat32 converts an IEEE 754 half-precision bit pattern to float32.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x3FF)
	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign // ±zero
		} else {
			// Subnormal f16: normalise into f32.
			exp32 := uint32(127 - 14)
			for frac&0x400 == 0 {
				frac <<= 1
				exp32--
			}
			frac &= 0x3FF
			bits = sign | (exp32 << 23) | (frac << 13)
		}
	case 31:
		// Inf or NaN.
		bits = sign | 0x7F800000 | (frac << 13)
	default:
		bits = sign | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}

// float32ToFP8 narrows a float32 to an 8-bit floating-point code with manBits
// mantissa bits and bias-biased exponent, using round-to-nearest-even (RTNE).
// It is the shared core of Float32ToFP8E4M3 and Float32ToFP8E5M2; the two
// formats differ only in their constants and overflow behaviour, which the
// callers apply via maxCode / overflowCode / nanCode:
//
//   - e4m3fn (OCP FP8): no infinities; finite overflow and ±Inf inputs both
//     saturate to ±448 (CUDA satfinite / Transformer Engine convention).
//   - e5m2 (IEEE-style): overflow rounds to ±Inf.
//
// The structure deliberately mirrors Float32ToFloat16 so the three narrowing
// paths stay reviewable side by side; the WGSL store helpers in codegen/wgsl.go
// mirror this algorithm instruction for instruction, which is what makes the
// GPU-vs-host storage comparison bit-exact rather than tolerance-based.
func float32ToFP8(f float32, manBits uint32, bias int32, maxCode, overflowCode, nanCode uint8) uint8 {
	bits := math.Float32bits(f)
	sign := uint8(bits>>31) << 7
	exp := int32((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF
	shift := 23 - manBits
	minExp := 1 - bias // smallest normal exponent

	switch {
	case exp == 128:
		if frac != 0 {
			return nanCode // NaN: canonical quiet NaN, sign dropped (matches Float32ToBFloat16)
		}
		return sign | overflowCode // ±Inf → format's overflow value (Inf for e5m2, NaN for e4m3fn)
	case exp < minExp:
		// Subnormal target. Total right shift grows as the exponent drops below
		// the normal range; past 24 the value is under half the smallest
		// subnormal and RTNE rounds to (signed) zero.
		t := shift + uint32(minExp-exp)
		if t > 24 {
			return sign
		}
		m := frac | 0x800000
		mRound := m >> (t - 1)
		if (mRound&1 != 0) && (mRound&2 != 0 || (m&((1<<(t-1))-1) != 0)) {
			mRound += 2
		}
		// A round-up past the subnormal range carries naturally into the
		// exponent field (min normal), same as the f16 subnormal path.
		return sign | uint8(mRound>>1)
	default:
		// Normal target. Round with RTNE; mantissa carry bumps the exponent.
		mRound := frac >> (shift - 1)
		if (mRound&1 != 0) && (mRound&2 != 0 || (frac&((1<<(shift-1))-1) != 0)) {
			mRound += 2
		}
		man := mRound >> 1
		eF := exp + bias
		if man&(1<<manBits) != 0 {
			man = 0
			eF++
		}
		code := eF<<manBits | int32(man)
		if code > int32(maxCode) {
			return sign | overflowCode
		}
		return sign | uint8(code)
	}
}

// Float32ToFP8E4M3 narrows a float32 to its nearest fp8 e4m3fn bit pattern
// (OCP FP8: bias 7, 3 mantissa bits, no infinities, NaN = S.1111.111, max
// finite ±448) using round-to-nearest-even. Finite overflow and ±Inf both
// saturate to ±448 — CUDA's satfinite conversion mode, which is what
// Transformer Engine and production fp8 training stacks use. (PyTorch's CPU
// converter instead maps overflow to NaN; saturation was chosen here as the
// deliberate, documented behaviour — see notes/fp8_preflight.md.)
func Float32ToFP8E4M3(f float32) uint8 {
	// maxCode 0x7E = 448; overflowCode 0x7E saturates; nanCode 0x7F canonical.
	return float32ToFP8(f, 3, 7, 0x7E, 0x7E, 0x7F)
}

// Float32ToFP8E5M2 narrows a float32 to its nearest fp8 e5m2 bit pattern
// (IEEE-style: bias 15, 2 mantissa bits, ±Inf and NaN, max finite ±57344)
// using round-to-nearest-even. Overflow rounds to ±Inf (0x7C), the IEEE
// behaviour for a format that represents infinity.
func Float32ToFP8E5M2(f float32) uint8 {
	// maxCode 0x7B = 57344; overflowCode 0x7C = Inf; nanCode 0x7E canonical.
	return float32ToFP8(f, 2, 15, 0x7B, 0x7C, 0x7E)
}

// FP8E4M3ToFloat32 widens an fp8 e4m3fn bit pattern to float32. Every e4m3fn
// value is exactly representable in float32, so this direction is lossless.
func FP8E4M3ToFloat32(b uint8) float32 {
	eF := int32(b>>3) & 0xF
	man := int32(b & 7)
	if eF == 0xF && man == 7 {
		return float32(math.NaN())
	}
	var v float64
	if eF == 0 {
		v = math.Ldexp(float64(man), -9) // subnormal: man * 2^(1-7-3)
	} else {
		v = math.Ldexp(1+float64(man)/8, int(eF-7))
	}
	if b&0x80 != 0 {
		v = -v
	}
	return float32(v)
}

// FP8E5M2ToFloat32 widens an fp8 e5m2 bit pattern to float32. Every e5m2
// value is exactly representable in float32, so this direction is lossless.
func FP8E5M2ToFloat32(b uint8) float32 {
	eF := int32(b>>2) & 0x1F
	man := int32(b & 3)
	sign := float64(1)
	if b&0x80 != 0 {
		sign = -1
	}
	if eF == 0x1F {
		if man == 0 {
			return float32(sign * math.Inf(1))
		}
		return float32(math.NaN())
	}
	var v float64
	if eF == 0 {
		v = math.Ldexp(float64(man), -16) // subnormal: man * 2^(1-15-2)
	} else {
		v = math.Ldexp(1+float64(man)/4, int(eF-15))
	}
	return float32(sign * v)
}

// Quantize returns v rounded to the nearest value representable in d.
// Float16, BFloat16, and the fp8 formats perform quantization; all other
// dtypes return v unchanged. All use round-to-nearest-even (RTNE) to match
// PyTorch and the rest of the ML ecosystem; storage and compute paths share
// these helpers.
func (d *DType) Quantize(v float32) float32 {
	s := d.Scalar()
	if s == Dtypes.Float16 {
		return Float16ToFloat32(Float32ToFloat16(v))
	}
	if s == Dtypes.BFloat16 {
		return BFloat16ToFloat32(Float32ToBFloat16(v))
	}
	if s == Dtypes.FP8E4M3 {
		return FP8E4M3ToFloat32(Float32ToFP8E4M3(v))
	}
	if s == Dtypes.FP8E5M2 {
		return FP8E5M2ToFloat32(Float32ToFP8E5M2(v))
	}
	return v
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

		// Image-storage variants share their compute identity with their
		// scalar peer. The edge Image→Float32 makes LeastUpperDType collapse
		// (Image, scalar) pairs to the scalar (storage choice, not compute
		// type). The (Image, Image) case is handled by LeastUpperDType's
		// "all inputs equal" early-exit so the result is Image, not Float32.
		D.ImageFloat32: {D.Float32},
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
		queue = append(queue, promoLattice[cur]...)
	}
	return result
}

// LeastUpperDType returns the smallest dtype that all of the given dtypes can
// be promoted to without loss of precision, mirroring tinygrad's
// least_upper_dtype.  Returns nil if ds is empty or if the intersection of
// ancestor sets is empty (which should not occur for well-formed inputs).
//
// Image-storage dtypes (DType.IsImage) are interchangeable with their scalar
// peer at the compute level: LeastUpperDType(ImageFloat32, Float32) == Float32.
// The image-vs-image case is special-cased so it collapses to the image
// dtype rather than its scalar peer (storage layout is preserved when every
// input agrees on it).
func LeastUpperDType(ds ...*DType) *DType {
	if len(ds) == 0 {
		return nil
	}
	// All-inputs-equal early exit preserves the image-vs-image case so
	// LeastUpperDType(ImageFloat32, ImageFloat32) == ImageFloat32 (the
	// promotion lattice itself routes Image→Scalar, which would otherwise
	// collapse the result to the scalar peer even when every input is image).
	allEqual := true
	for _, d := range ds[1:] {
		if d != ds[0] {
			allEqual = false
			break
		}
	}
	if allEqual {
		return ds[0]
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
	// The intersection never contains both ImageFloat32 and Float32 at the
	// same time: ImageFloat32's ancestor set is {Image, F32, F64} while
	// every non-image scalar's set tops out at {F32, F64}, so the (Image,
	// Scalar) tie-break is unreachable from the present lattice and the
	// pre-tie-break CmpDType total order is sufficient. If a future slice
	// adds image edges that COULD collide (e.g. ImageFloat16), reintroduce
	// the isImage tie-breaker here and prefer the non-image so the lattice
	// result stays the compute identity.
	var best *DType
	for d := range common {
		if best == nil || CmpDType(d, best) < 0 {
			best = d
		}
	}
	return best
}
