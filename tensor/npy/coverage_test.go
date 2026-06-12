package npy

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// buildNPY constructs a minimal v1.0 .npy byte buffer for the given descr,
// fortran flag, shape, and raw payload. It mirrors the on-disk format closely
// enough to drive the in-memory byte parser.
func buildNPY(descr string, fortran bool, shape string, payload []byte) []byte {
	fo := "False"
	if fortran {
		fo = "True"
	}
	hdr := "{'descr': '" + descr + "', 'fortran_order': " + fo + ", 'shape': " + shape + ", }"
	// Pad header so that 10 + len(hdr) is a multiple of 64, ending in '\n'.
	total := 10 + len(hdr) + 1
	if rem := total % 64; rem != 0 {
		hdr += strings.Repeat(" ", 64-rem)
	}
	hdr += "\n"

	out := make([]byte, 0, 10+len(hdr)+len(payload))
	out = append(out, 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0)
	lenB := make([]byte, 2)
	binary.LittleEndian.PutUint16(lenB, uint16(len(hdr)))
	out = append(out, lenB...)
	out = append(out, []byte(hdr)...)
	out = append(out, payload...)
	return out
}

// buildNPYv2 constructs a v2.0 header (4-byte header length) so the v2 branch
// and uint32Le helper are exercised.
func buildNPYv2(descr, shape string, payload []byte) []byte {
	hdr := "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }"
	total := 12 + len(hdr) + 1
	if rem := total % 64; rem != 0 {
		hdr += strings.Repeat(" ", 64-rem)
	}
	hdr += "\n"

	out := make([]byte, 0, 12+len(hdr)+len(payload))
	out = append(out, 0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0)
	lenB := make([]byte, 4)
	binary.LittleEndian.PutUint32(lenB, uint32(len(hdr)))
	out = append(out, lenB...)
	out = append(out, []byte(hdr)...)
	out = append(out, payload...)
	return out
}

func f32Payload(vals []float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// ── happy paths through parseNPYBytes (covers uint16Le / uint32Le) ───────────

func TestParseNPYBytes_F32_V1(t *testing.T) {
	data := buildNPY("<f4", false, "(2, 2)", f32Payload([]float32{1, 2, 3, 4}))
	descr, shape, f32, err := parseNPYBytes(data)
	if err != nil {
		t.Fatalf("parseNPYBytes: %v", err)
	}
	if descr != "<f4" {
		t.Errorf("descr = %q", descr)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Errorf("shape = %v", shape)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if f32[i] != want[i] {
			t.Errorf("data[%d] = %v want %v", i, f32[i], want[i])
		}
	}
}

func TestParseNPYBytes_V2_Header(t *testing.T) {
	data := buildNPYv2("<f4", "(3,)", f32Payload([]float32{5, 6, 7}))
	_, shape, f32, err := parseNPYBytes(data)
	if err != nil {
		t.Fatalf("parseNPYBytes v2: %v", err)
	}
	if len(shape) != 1 || shape[0] != 3 {
		t.Errorf("shape = %v", shape)
	}
	if f32[0] != 5 || f32[2] != 7 {
		t.Errorf("data = %v", f32)
	}
}

func TestUint32Le(t *testing.T) {
	if got := uint32Le([]byte{0x01, 0x02, 0x03, 0x04}); got != 0x04030201 {
		t.Errorf("uint32Le = %#x", got)
	}
}

func TestMin(t *testing.T) {
	if min(3, 7) != 3 {
		t.Error("min(3,7) != 3")
	}
	if min(9, 2) != 2 {
		t.Error("min(9,2) != 2")
	}
}

// ── ReadBytes fortran transposition via byte path ────────────────────────────

func TestReadBytes_Fortran(t *testing.T) {
	// Fortran column-major [[0,1,2],[3,4,5]] is stored [0,3,1,4,2,5].
	payload := f32Payload([]float32{0, 3, 1, 4, 2, 5})
	data := buildNPY("<f4", true, "(2, 3)", payload)
	e, err := ReadBytes(data)
	if err != nil {
		t.Fatalf("ReadBytes: %v", err)
	}
	want := []float32{0, 1, 2, 3, 4, 5}
	for i := range want {
		if e.Data[i] != want[i] {
			t.Errorf("data[%d] = %v want %v", i, e.Data[i], want[i])
		}
	}
}

// ── scalar shape () ──────────────────────────────────────────────────────────

func TestReadBytes_Scalar(t *testing.T) {
	data := buildNPY("<f4", false, "()", f32Payload([]float32{42}))
	e, err := ReadBytes(data)
	if err != nil {
		t.Fatalf("ReadBytes scalar: %v", err)
	}
	if len(e.Shape) != 0 {
		t.Errorf("scalar shape = %v, want []", e.Shape)
	}
	if len(e.Data) != 1 || e.Data[0] != 42 {
		t.Errorf("scalar data = %v", e.Data)
	}
}

// ── error paths in parseNPYBytes ─────────────────────────────────────────────

func TestParseNPYBytes_Errors(t *testing.T) {
	good := buildNPY("<f4", false, "(2,)", f32Payload([]float32{1, 2}))

	cases := []struct {
		name string
		data []byte
		frag string
	}{
		{"too-short", []byte{0x93, 'N'}, "too short"},
		{"bad-magic", append([]byte{0x00}, good[1:]...), "bad magic"},
		{"unsupported-version", func() []byte {
			d := append([]byte(nil), good...)
			d[6] = 9
			return d
		}(), "unsupported format version"},
		{"v2-short", []byte{0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0, 1, 0}, "too short for v2"},
		{"header-overflow", func() []byte {
			d := append([]byte(nil), good...)
			// Inflate declared header length far beyond the file.
			binary.LittleEndian.PutUint16(d[8:10], 60000)
			return d
		}(), "exceeds file length"},
		{"truncated-payload", buildNPY("<f4", false, "(8,)", f32Payload([]float32{1, 2})), "truncated"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, _, _, err := parseNPYBytes(c.data)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", c.frag)
			}
			if !strings.Contains(err.Error(), c.frag) {
				t.Fatalf("error = %v, want fragment %q", err, c.frag)
			}
		})
	}
}

// ── parseDescr branches ──────────────────────────────────────────────────────

func TestParseDescr(t *testing.T) {
	cases := []struct {
		descr     string
		order     byte
		typeChar  byte
		itemSize  int
		expectErr bool
	}{
		{"<f4", '<', 'f', 4, false},
		{">i8", '>', 'i', 8, false},
		{"|b1", '|', 'b', 1, false},
		{"=u2", '=', 'u', 2, false},
		{"f4", '|', 'f', 4, false}, // no byte-order prefix → default '|'
		{"", 0, 0, 0, true},        // empty
		{"<f", 0, 0, 0, true},      // too short after prefix
		{"<fx", 0, 0, 0, true},     // non-numeric size
	}
	for _, c := range cases {
		t.Run(c.descr, func(t *testing.T) {
			order, tc, sz, err := parseDescr(c.descr)
			if c.expectErr {
				if err == nil {
					t.Fatalf("expected error for %q", c.descr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseDescr(%q): %v", c.descr, err)
			}
			if order != c.order || tc != c.typeChar || sz != c.itemSize {
				t.Errorf("parseDescr(%q) = (%c,%c,%d), want (%c,%c,%d)",
					c.descr, order, tc, sz, c.order, c.typeChar, c.itemSize)
			}
		})
	}
}

// ── header field error paths ─────────────────────────────────────────────────

func TestHdrFields_Errors(t *testing.T) {
	if _, err := hdrStringField("{}", "descr"); err == nil {
		t.Error("missing descr key should error")
	}
	if _, err := hdrStringField("{'descr' 4}", "descr"); err == nil {
		t.Error("missing colon should error")
	}
	if _, err := hdrStringField("{'descr': 4}", "descr"); err == nil {
		t.Error("non-quoted value should error")
	}
	if _, err := hdrStringField("{'descr': '<f4}", "descr"); err == nil {
		t.Error("unterminated string should error")
	}

	if _, err := hdrBoolField("{}", "fortran_order"); err == nil {
		t.Error("missing bool key should error")
	}
	if _, err := hdrBoolField("{'fortran_order' True}", "fortran_order"); err == nil {
		t.Error("missing colon should error")
	}
	if _, err := hdrBoolField("{'fortran_order': Maybe}", "fortran_order"); err == nil {
		t.Error("invalid bool literal should error")
	}
	if v, err := hdrBoolField("{'fortran_order': True}", "fortran_order"); err != nil || !v {
		t.Errorf("True parse: v=%v err=%v", v, err)
	}

	if _, err := hdrShapeField("{}"); err == nil {
		t.Error("missing shape key should error")
	}
	if _, err := hdrShapeField("{'shape' (2,)}"); err == nil {
		t.Error("missing colon should error")
	}
	if _, err := hdrShapeField("{'shape': [2]}"); err == nil {
		t.Error("non-paren shape should error")
	}
	if _, err := hdrShapeField("{'shape': (2, 3}"); err == nil {
		t.Error("unterminated shape tuple should error")
	}
	if _, err := hdrShapeField("{'shape': (2, x)}"); err == nil {
		t.Error("invalid dimension should error")
	}
	sh, err := hdrShapeField("{'shape': (5,)}")
	if err != nil || len(sh) != 1 || sh[0] != 5 {
		t.Errorf("single-element tuple: %v err=%v", sh, err)
	}
}

// ── convertToFloat32 dtype matrix and overflow/unsupported errors ────────────

func TestConvertToFloat32_Dtypes(t *testing.T) {
	// f64 little-endian.
	f64 := make([]byte, 16)
	binary.LittleEndian.PutUint64(f64[0:], math.Float64bits(1.5))
	binary.LittleEndian.PutUint64(f64[8:], math.Float64bits(-2.5))
	if out, err := convertToFloat32(f64, '<', 'f', 8, 2); err != nil || out[0] != 1.5 || out[1] != -2.5 {
		t.Errorf("f64: out=%v err=%v", out, err)
	}

	// int16, uint16.
	i16 := make([]byte, 4)
	neg3 := int16(-3)
	binary.LittleEndian.PutUint16(i16[0:], uint16(neg3))
	binary.LittleEndian.PutUint16(i16[2:], 7)
	if out, err := convertToFloat32(i16, '<', 'i', 2, 2); err != nil || out[0] != -3 || out[1] != 7 {
		t.Errorf("i16: out=%v err=%v", out, err)
	}
	u16 := make([]byte, 2)
	binary.LittleEndian.PutUint16(u16, 65535)
	if out, err := convertToFloat32(u16, '<', 'u', 2, 1); err != nil || out[0] != 65535 {
		t.Errorf("u16: out=%v err=%v", out, err)
	}

	// int32, uint32 in range.
	i32 := make([]byte, 4)
	neg100 := int32(-100)
	binary.LittleEndian.PutUint32(i32, uint32(neg100))
	if out, err := convertToFloat32(i32, '<', 'i', 4, 1); err != nil || out[0] != -100 {
		t.Errorf("i32: out=%v err=%v", out, err)
	}
	u32 := make([]byte, 4)
	binary.LittleEndian.PutUint32(u32, 100)
	if out, err := convertToFloat32(u32, '<', 'u', 4, 1); err != nil || out[0] != 100 {
		t.Errorf("u32: out=%v err=%v", out, err)
	}

	// uint8 by value.
	if out, err := convertToFloat32([]byte{0, 200}, '|', 'u', 1, 2); err != nil || out[1] != 200 {
		t.Errorf("u8: out=%v err=%v", out, err)
	}

	// big-endian float32.
	be := make([]byte, 4)
	binary.BigEndian.PutUint32(be, math.Float32bits(3.25))
	if out, err := convertToFloat32(be, '>', 'f', 4, 1); err != nil || out[0] != 3.25 {
		t.Errorf("be f32: out=%v err=%v", out, err)
	}
}

func TestConvertToFloat32_Errors(t *testing.T) {
	// int32 overflow (> 2^24).
	i32 := make([]byte, 4)
	binary.LittleEndian.PutUint32(i32, uint32(int32(1<<24+1)))
	if _, err := convertToFloat32(i32, '<', 'i', 4, 1); err == nil {
		t.Error("int32 overflow should error")
	}
	// int64 overflow.
	i64 := make([]byte, 8)
	binary.LittleEndian.PutUint64(i64, uint64(int64(1<<24+1)))
	if _, err := convertToFloat32(i64, '<', 'i', 8, 1); err == nil {
		t.Error("int64 overflow should error")
	}
	// uint32 overflow.
	u32 := make([]byte, 4)
	binary.LittleEndian.PutUint32(u32, uint32(1<<24+1))
	if _, err := convertToFloat32(u32, '<', 'u', 4, 1); err == nil {
		t.Error("uint32 overflow should error")
	}
	// uint64 overflow.
	u64 := make([]byte, 8)
	binary.LittleEndian.PutUint64(u64, uint64(1<<24+1))
	if _, err := convertToFloat32(u64, '<', 'u', 8, 1); err == nil {
		t.Error("uint64 overflow should error")
	}
	// unsupported float size.
	if _, err := convertToFloat32(make([]byte, 2), '<', 'f', 2, 1); err == nil {
		t.Error("float16 should be unsupported")
	}
	// unsupported int / uint size.
	if _, err := convertToFloat32(make([]byte, 16), '<', 'i', 16, 1); err == nil {
		t.Error("int128 should be unsupported")
	}
	if _, err := convertToFloat32(make([]byte, 16), '<', 'u', 16, 1); err == nil {
		t.Error("uint128 should be unsupported")
	}
	// bool wrong size.
	if _, err := convertToFloat32(make([]byte, 2), '|', 'b', 2, 1); err == nil {
		t.Error("bool size != 1 should error")
	}
	// complex.
	if _, err := convertToFloat32(make([]byte, 8), '<', 'c', 8, 1); err == nil {
		t.Error("complex should be unsupported")
	}
	// unknown kind.
	if _, err := convertToFloat32(make([]byte, 4), '<', 'z', 4, 1); err == nil {
		t.Error("unknown kind should be unsupported")
	}
	// bool true → 1.0.
	if out, err := convertToFloat32([]byte{0, 1}, '|', 'b', 1, 2); err != nil || out[0] != 0 || out[1] != 1 {
		t.Errorf("bool: out=%v err=%v", out, err)
	}
}
