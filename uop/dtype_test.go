package uop_test

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── itemsize ──────────────────────────────────────────────────────────────────

func TestDTypeItemSize(t *testing.T) {
	tests := []struct {
		dt   *uop.DType
		want int
	}{
		{uop.Dtypes.Bool, 1},
		{uop.Dtypes.Int8, 1},
		{uop.Dtypes.UInt8, 1},
		{uop.Dtypes.Int16, 2},
		{uop.Dtypes.UInt16, 2},
		{uop.Dtypes.Int32, 4},
		{uop.Dtypes.UInt32, 4},
		{uop.Dtypes.Int64, 8},
		{uop.Dtypes.UInt64, 8},
		{uop.Dtypes.Float16, 2},
		{uop.Dtypes.BFloat16, 2},
		{uop.Dtypes.Float32, 4},
		{uop.Dtypes.Float64, 8},
		{uop.Dtypes.FP8E4M3, 1},
		{uop.Dtypes.FP8E5M2, 1},
	}
	for _, tc := range tests {
		if got := tc.dt.ItemSize(); got != tc.want {
			t.Errorf("%s.ItemSize() = %d, want %d", tc.dt, got, tc.want)
		}
	}
}

// ── aliases ───────────────────────────────────────────────────────────────────

func TestDTypeAliases(t *testing.T) {
	tests := []struct {
		alias    *uop.DType
		canon    *uop.DType
		aliasStr string
	}{
		{uop.Dtypes.Float, uop.Dtypes.Float32, "Float"},
		{uop.Dtypes.Half, uop.Dtypes.Float16, "Half"},
		{uop.Dtypes.Double, uop.Dtypes.Float64, "Double"},
		{uop.Dtypes.Int, uop.Dtypes.Int32, "Int"},
		{uop.Dtypes.Char, uop.Dtypes.Int8, "Char"},
		{uop.Dtypes.Short, uop.Dtypes.Int16, "Short"},
		{uop.Dtypes.Long, uop.Dtypes.Int64, "Long"},
		{uop.Dtypes.UChar, uop.Dtypes.UInt8, "UChar"},
		{uop.Dtypes.UShort, uop.Dtypes.UInt16, "UShort"},
		{uop.Dtypes.UInt, uop.Dtypes.UInt32, "UInt"},
		{uop.Dtypes.ULong, uop.Dtypes.UInt64, "ULong"},
	}
	for _, tc := range tests {
		if tc.alias != tc.canon {
			t.Errorf("Dtypes.%s is not the same pointer as its canonical dtype", tc.aliasStr)
		}
	}
}

// ── vector: Vec / Scalar roundtrip ───────────────────────────────────────────

func TestDTypeVec(t *testing.T) {
	tests := []struct {
		base  *uop.DType
		n     int
		items int
		bytes int
	}{
		{uop.Dtypes.Float32, 4, 4, 16},
		{uop.Dtypes.Int32, 4, 4, 16},
		{uop.Dtypes.Float16, 8, 8, 16},
		{uop.Dtypes.Bool, 8, 8, 1}, // bool is 1-bit per lane → 8 bits total → 1 byte
	}
	for _, tc := range tests {
		vec := tc.base.Vec(tc.n)
		if vec.Count() != tc.n {
			t.Errorf("%s.Vec(%d).Count() = %d, want %d", tc.base, tc.n, vec.Count(), tc.n)
		}
		if vec.ItemSize() != tc.bytes {
			t.Errorf("%s.Vec(%d).ItemSize() = %d, want %d", tc.base, tc.n, vec.ItemSize(), tc.bytes)
		}
		if vec.Scalar() != tc.base {
			t.Errorf("%s.Vec(%d).Scalar() != base dtype", tc.base, tc.n)
		}
	}
}

func TestDTypeVec1IsIdentity(t *testing.T) {
	d := uop.Dtypes.Float32
	if d.Vec(1) != d {
		t.Error("Vec(1) should return the dtype unchanged")
	}
}

func TestDTypeVecVoidIsIdentity(t *testing.T) {
	if uop.Dtypes.Void.Vec(4) != uop.Dtypes.Void {
		t.Error("Void.Vec(n) should return Void unchanged")
	}
}

// ── vector: interning ─────────────────────────────────────────────────────────

func TestDTypeVecInterning(t *testing.T) {
	a := uop.Dtypes.Float32.Vec(4)
	b := uop.Dtypes.Float32.Vec(4)
	if a != b {
		t.Error("two Vec(4) calls on the same base must return the same pointer")
	}
}

func TestDTypeVecDistinct(t *testing.T) {
	f4 := uop.Dtypes.Float32.Vec(4)
	f8 := uop.Dtypes.Float32.Vec(8)
	if f4 == f8 {
		t.Error("Vec(4) and Vec(8) must be distinct pointers")
	}
}

// ── pointer: Ptr / Base roundtrip ─────────────────────────────────────────────

func TestDTypePtr(t *testing.T) {
	tests := []struct {
		base      *uop.DType
		size      int
		addrSpace uop.AddrSpace
	}{
		{uop.Dtypes.Float32, -1, uop.Global},
		{uop.Dtypes.Int32, 1024, uop.Global},
		{uop.Dtypes.Float32, -1, uop.Local},
		{uop.Dtypes.Float32, -1, uop.Reg},
	}
	for _, tc := range tests {
		p := tc.base.Ptr(tc.size, tc.addrSpace)
		if !p.IsPtr() {
			t.Errorf("%s.Ptr().IsPtr() = false", tc.base)
		}
		if p.Base() != tc.base {
			t.Errorf("%s.Ptr().Base() != original base", tc.base)
		}
		if p.PtrSize() != tc.size {
			t.Errorf("%s.Ptr(%d, _).PtrSize() = %d, want %d", tc.base, tc.size, p.PtrSize(), tc.size)
		}
		if p.AddrSpaceOf() != tc.addrSpace {
			t.Errorf("%s.Ptr(_, %v).AddrSpaceOf() = %v, want %v",
				tc.base, tc.addrSpace, p.AddrSpaceOf(), tc.addrSpace)
		}
	}
}

func TestDTypePtrInterning(t *testing.T) {
	a := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	b := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	if a != b {
		t.Error("identical Ptr() calls must return the same pointer")
	}
}

func TestDTypePtrAddrSpaceDistinct(t *testing.T) {
	global := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	local := uop.Dtypes.Float32.Ptr(-1, uop.Local)
	if global == local {
		t.Error("Global and Local ptr variants must be distinct")
	}
}

func TestDTypePtrBaseIsNotPtr(t *testing.T) {
	p := uop.Dtypes.Float32.Ptr(-1, uop.Global)
	if p.Base().IsPtr() {
		t.Error("Ptr().Base() must not be a pointer dtype")
	}
}

// ── map-key comparability ─────────────────────────────────────────────────────

// TestDTypeAsMapKey verifies that *DType is usable as a map key and that
// interned identity equality is correct.
func TestDTypeAsMapKey(t *testing.T) {
	m := map[*uop.DType]string{}
	m[uop.Dtypes.Float32] = "f32"
	m[uop.Dtypes.Int32] = "i32"

	if m[uop.Dtypes.Float32] != "f32" {
		t.Error("Dtypes.Float32 lookup failed")
	}
	// Vec and Ptr types as keys
	v := uop.Dtypes.Float32.Vec(4)
	m[v] = "f32x4"
	if m[uop.Dtypes.Float32.Vec(4)] != "f32x4" {
		t.Error("Vec(4) map lookup failed — interning broken")
	}
}

// ── type predicates ───────────────────────────────────────────────────────────

func TestIsFloat(t *testing.T) {
	floats := []*uop.DType{
		uop.Dtypes.Float16, uop.Dtypes.BFloat16, uop.Dtypes.Float32, uop.Dtypes.Float64,
		uop.Dtypes.FP8E4M3, uop.Dtypes.FP8E5M2,
	}
	for _, d := range floats {
		if !d.IsFloat() {
			t.Errorf("%s.IsFloat() = false, want true", d)
		}
		// vector of float should also be float
		if !d.Vec(4).IsFloat() {
			t.Errorf("%s.Vec(4).IsFloat() = false, want true", d)
		}
	}
	nonFloats := []*uop.DType{
		uop.Dtypes.Bool, uop.Dtypes.Int32, uop.Dtypes.UInt8,
	}
	for _, d := range nonFloats {
		if d.IsFloat() {
			t.Errorf("%s.IsFloat() = true, want false", d)
		}
	}
}

func TestIsInt(t *testing.T) {
	ints := []*uop.DType{
		uop.Dtypes.Int8, uop.Dtypes.UInt8, uop.Dtypes.Int16, uop.Dtypes.UInt16,
		uop.Dtypes.Int32, uop.Dtypes.UInt32, uop.Dtypes.Int64, uop.Dtypes.UInt64,
		uop.Dtypes.Index,
	}
	for _, d := range ints {
		if !d.IsInt() {
			t.Errorf("%s.IsInt() = false, want true", d)
		}
	}
	nonInts := []*uop.DType{uop.Dtypes.Float32, uop.Dtypes.Bool}
	for _, d := range nonInts {
		if d.IsInt() {
			t.Errorf("%s.IsInt() = true, want false", d)
		}
	}
}

func TestIsUnsigned(t *testing.T) {
	uints := []*uop.DType{uop.Dtypes.UInt8, uop.Dtypes.UInt16, uop.Dtypes.UInt32, uop.Dtypes.UInt64}
	for _, d := range uints {
		if !d.IsUnsigned() {
			t.Errorf("%s.IsUnsigned() = false, want true", d)
		}
	}
	signed := []*uop.DType{uop.Dtypes.Int8, uop.Dtypes.Int32, uop.Dtypes.Float32, uop.Dtypes.Bool}
	for _, d := range signed {
		if d.IsUnsigned() {
			t.Errorf("%s.IsUnsigned() = true, want false", d)
		}
	}
}

func TestIsBool(t *testing.T) {
	if !uop.Dtypes.Bool.IsBool() {
		t.Error("Bool.IsBool() = false")
	}
	if uop.Dtypes.Int32.IsBool() {
		t.Error("Int32.IsBool() = true")
	}
	if !uop.Dtypes.Bool.Vec(4).IsBool() {
		t.Error("Bool.Vec(4).IsBool() = false")
	}
}

// ── LeastUpperDType ───────────────────────────────────────────────────────────

func TestLeastUpperDType(t *testing.T) {
	tests := []struct {
		name    string
		inputs  []*uop.DType
		want    *uop.DType
	}{
		{"same", []*uop.DType{uop.Dtypes.Float32, uop.Dtypes.Float32}, uop.Dtypes.Float32},
		{"bool+int8→int8", []*uop.DType{uop.Dtypes.Bool, uop.Dtypes.Int8}, uop.Dtypes.Int8},
		{"int32+float32→float32", []*uop.DType{uop.Dtypes.Int32, uop.Dtypes.Float32}, uop.Dtypes.Float32},
		{"float32+float64→float64", []*uop.DType{uop.Dtypes.Float32, uop.Dtypes.Float64}, uop.Dtypes.Float64},
		{"int8+uint8→int16", []*uop.DType{uop.Dtypes.Int8, uop.Dtypes.UInt8}, uop.Dtypes.Int16},
		{"int32+int64→int64", []*uop.DType{uop.Dtypes.Int32, uop.Dtypes.Int64}, uop.Dtypes.Int64},
		{"single float32", []*uop.DType{uop.Dtypes.Float32}, uop.Dtypes.Float32},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := uop.LeastUpperDType(tc.inputs...)
			if got != tc.want {
				t.Errorf("LeastUpperDType(%v) = %s, want %s", tc.inputs, got, tc.want)
			}
		})
	}
}

// ── StructuralHash ────────────────────────────────────────────────────────────

// TestDTypeStructuralHashStable verifies that the same dtype always returns the
// same StructuralHash value and that a nil *DType returns the sentinel.
func TestDTypeStructuralHashStable(t *testing.T) {
	dtypes := []*uop.DType{
		uop.Dtypes.Void, uop.Dtypes.Bool,
		uop.Dtypes.Int8, uop.Dtypes.UInt8, uop.Dtypes.Int16, uop.Dtypes.UInt16,
		uop.Dtypes.Int32, uop.Dtypes.UInt32, uop.Dtypes.Int64, uop.Dtypes.UInt64,
		uop.Dtypes.Float16, uop.Dtypes.BFloat16, uop.Dtypes.Float32, uop.Dtypes.Float64,
		uop.Dtypes.FP8E4M3, uop.Dtypes.FP8E5M2, uop.Dtypes.Index,
		uop.Dtypes.Float32.Vec(4),
		uop.Dtypes.Float32.Ptr(-1, uop.Global),
		uop.Dtypes.Float32.Ptr(-1, uop.Local),
	}
	for _, d := range dtypes {
		h1 := d.StructuralHash()
		h2 := d.StructuralHash()
		if h1 != h2 {
			t.Errorf("%s.StructuralHash() not stable: %016x != %016x", d, h1, h2)
		}
	}
	// nil sentinel is distinct from any real dtype hash.
	var nilDT *uop.DType
	nilH := nilDT.StructuralHash()
	for _, d := range dtypes {
		if d.StructuralHash() == nilH {
			t.Errorf("%s.StructuralHash() == nil sentinel %016x", d, nilH)
		}
	}
}

// TestDTypeStructuralHashUnique verifies that all canonical scalar/vector/pointer
// dtypes used in anneal produce distinct hash values.
func TestDTypeStructuralHashUnique(t *testing.T) {
	dtypes := []*uop.DType{
		uop.Dtypes.Void, uop.Dtypes.Bool,
		uop.Dtypes.Int8, uop.Dtypes.UInt8, uop.Dtypes.Int16, uop.Dtypes.UInt16,
		uop.Dtypes.Int32, uop.Dtypes.UInt32, uop.Dtypes.Int64, uop.Dtypes.UInt64,
		uop.Dtypes.Float16, uop.Dtypes.BFloat16, uop.Dtypes.Float32, uop.Dtypes.Float64,
		uop.Dtypes.FP8E4M3, uop.Dtypes.FP8E5M2, uop.Dtypes.Index,
		// vector variants
		uop.Dtypes.Float32.Vec(2), uop.Dtypes.Float32.Vec(4), uop.Dtypes.Float32.Vec(8),
		uop.Dtypes.Int32.Vec(4),
		// pointer variants
		uop.Dtypes.Float32.Ptr(-1, uop.Global),
		uop.Dtypes.Float32.Ptr(-1, uop.Local),
		uop.Dtypes.Float32.Ptr(-1, uop.Reg),
		uop.Dtypes.Float32.Ptr(1024, uop.Global),
		uop.Dtypes.Int32.Ptr(-1, uop.Global),
	}
	seen := make(map[uint64]*uop.DType, len(dtypes))
	for _, d := range dtypes {
		h := d.StructuralHash()
		if prev, collision := seen[h]; collision {
			t.Errorf("hash collision: %s and %s both hash to %016x", prev, d, h)
		}
		seen[h] = d
	}
}

// TestDTypeStructuralHashGolden pins the exact StructuralHash values for key
// dtypes. A stable golden value is the definitive proof that the hash is derived
// from DType field values alone, not from the pointer's allocation address: any
// address-based component would change across runs, invalidating the constant.
//
// If the DType field layout or StructuralHash algorithm intentionally changes,
// update these constants and document the reason.
func TestDTypeStructuralHashGolden(t *testing.T) {
	cases := []struct {
		name   string
		dt     *uop.DType
		golden uint64
	}{
		{"Float32", uop.Dtypes.Float32, 0x7f4278a474769c36},
		{"Int32", uop.Dtypes.Int32, 0xfee909559c15d11a},
		{"Float16", uop.Dtypes.Float16, 0x07091dd970d2d7c0},
		{"Bool", uop.Dtypes.Bool, 0x7e387016041cc497},
		{"Index", uop.Dtypes.Index, 0x7a131b2190c6940c},
		{"Float32×4", uop.Dtypes.Float32.Vec(4), 0xc512b320f01638d4},
		{"Float32.ptr(Global)", uop.Dtypes.Float32.Ptr(-1, uop.Global), 0x743ef066b1770e27},
	}
	for _, tc := range cases {
		h := tc.dt.StructuralHash()
		if h != tc.golden {
			t.Errorf("%s: StructuralHash()=0x%016x, want golden 0x%016x — "+
				"algorithm or field layout changed, or hash is no longer address-independent",
				tc.name, h, tc.golden)
		}
	}
}

// TestStructuralKeysGolden pins the StructuralKeys value for a simple node to
// prove cross-build stability: a const(1.0 f32) node always produces the same
// key regardless of which address Float32 was allocated at in this process.
func TestStructuralKeysGolden(t *testing.T) {
	a := uop.NewArena(4)
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	keys := uop.StructuralKeys(a)

	const golden uint64 = 0xd580b1967dd050b5
	got := keys[c.Index()]
	t.Logf("const(1.0 f32) structural key: 0x%016x (golden: 0x%016x)", got, golden)
	if got != golden {
		t.Errorf("StructuralKeys: 0x%016x != golden 0x%016x — "+
			"dtype address-dependence reintroduced, or hash algorithm changed",
			got, golden)
	}
}

// ── String ────────────────────────────────────────────────────────────────────

func TestDTypeString(t *testing.T) {
	tests := []struct {
		dt       *uop.DType
		contains string
	}{
		{uop.Dtypes.Float32, "float"},
		{uop.Dtypes.Int32, "int"},
		{uop.Dtypes.Bool, "bool"},
		{uop.Dtypes.Float32.Vec(4), "float"},
		{uop.Dtypes.Float32.Ptr(-1, uop.Global), "float"},
		{uop.Dtypes.Float32.Ptr(-1, uop.Local), "Local"},
	}
	for _, tc := range tests {
		s := tc.dt.String()
		if s == "" {
			t.Errorf("%T.String() returned empty string", tc.dt)
		}
		if tc.contains != "" {
			found := false
			for i := 0; i+len(tc.contains) <= len(s); i++ {
				if s[i:i+len(tc.contains)] == tc.contains {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%v.String() = %q, want substring %q", tc.dt, s, tc.contains)
			}
		}
	}
}
