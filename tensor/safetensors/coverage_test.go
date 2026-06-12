package safetensors

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTemp writes b to a fresh temp file and returns its path.
func writeTemp(t *testing.T, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "blob.safetensors")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatalf("writeTemp: %v", err)
	}
	return p
}

// buildST assembles a minimal in-memory safetensors file from a JSON header
// string and a raw data buffer.
func buildST(hdr string, data []byte) []byte {
	hdrB := []byte(hdr)
	if rem := len(hdrB) % 8; rem != 0 {
		pad := make([]byte, 8-rem)
		for i := range pad {
			pad[i] = ' '
		}
		hdrB = append(hdrB, pad...)
	}
	out := make([]byte, 8+len(hdrB)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(hdrB)))
	copy(out[8:], hdrB)
	copy(out[8+len(hdrB):], data)
	return out
}

// ── decodeToFloat32 dtype matrix (via in-memory ReadBytes) ───────────────────

func TestDecode_AllDtypes(t *testing.T) {
	cases := []struct {
		dtype string
		raw   []byte
		n     int
		want  []float32
	}{
		{"F32", f32LE([]float32{1.5, -2.5}), 2, []float32{1.5, -2.5}},
		{"F64", f64LE([]float64{3.5, -4.5}), 2, []float32{3.5, -4.5}},
		{"I8", []byte{0xFB, 0x7F}, 2, []float32{-5, 127}}, // -5, 127
		{"U8", []byte{0x00, 0xFF}, 2, []float32{0, 255}},
		{"I16", u16LE([]uint16{u16of(-3), 7}), 2, []float32{-3, 7}},
		{"U16", u16LE([]uint16{0, 65535}), 2, []float32{0, 65535}},
		{"I32", u32LE([]uint32{u32of(-100), 100}), 2, []float32{-100, 100}},
		{"U32", u32LE([]uint32{0, 1000}), 2, []float32{0, 1000}},
		{"I64", u64LE([]uint64{u64of(-7), 9}), 2, []float32{-7, 9}},
		{"U64", u64LE([]uint64{0, 12345}), 2, []float32{0, 12345}},
		{"BOOL", []byte{0, 1, 5}, 3, []float32{0, 1, 1}},
	}
	for _, c := range cases {
		t.Run(c.dtype, func(t *testing.T) {
			out, err := decodeToFloat32(c.raw, c.dtype, []int64{int64(c.n)}, "t")
			if err != nil {
				t.Fatalf("decode %s: %v", c.dtype, err)
			}
			for i := range c.want {
				if out[i] != c.want[i] {
					t.Errorf("%s[%d] = %v, want %v", c.dtype, i, out[i], c.want[i])
				}
			}
		})
	}
}

func TestDecode_Errors(t *testing.T) {
	// Byte-length mismatch.
	if _, err := decodeToFloat32([]byte{1, 2, 3}, "F32", []int64{1}, "t"); err == nil {
		t.Error("length mismatch should error")
	}
	// Unsupported dtype (item size unknown).
	if _, err := decodeToFloat32([]byte{1}, "F8_E4M3", []int64{1}, "t"); err == nil {
		t.Error("unsupported dtype should error")
	}
	// Overflow paths.
	if _, err := decodeToFloat32(u32LE([]uint32{1<<24 + 1}), "I32", []int64{1}, "t"); err == nil {
		t.Error("I32 overflow should error")
	}
	if _, err := decodeToFloat32(u64LE([]uint64{1<<24 + 1}), "I64", []int64{1}, "t"); err == nil {
		t.Error("I64 overflow should error")
	}
	if _, err := decodeToFloat32(u32LE([]uint32{1<<24 + 1}), "U32", []int64{1}, "t"); err == nil {
		t.Error("U32 overflow should error")
	}
	if _, err := decodeToFloat32(u64LE([]uint64{1<<24 + 1}), "U64", []int64{1}, "t"); err == nil {
		t.Error("U64 overflow should error")
	}
}

func TestDtypeItemSize(t *testing.T) {
	cases := map[string]int{
		"BOOL": 1, "I8": 1, "U8": 1,
		"F16": 2, "BF16": 2, "I16": 2, "U16": 2,
		"F32": 4, "I32": 4, "U32": 4,
		"F64": 8, "I64": 8, "U64": 8,
	}
	for dt, want := range cases {
		got, err := dtypeItemSize(dt)
		if err != nil || got != want {
			t.Errorf("dtypeItemSize(%s) = %d,%v; want %d", dt, got, err, want)
		}
	}
	if _, err := dtypeItemSize("NOPE"); err == nil {
		t.Error("unknown dtype should error")
	}
}

// ── f16ToF32 edge cases: subnormal, NaN, ±Inf, ±0 ────────────────────────────

func TestF16ToF32_Edges(t *testing.T) {
	// Smallest positive subnormal f16: bits 0x0001 = 2^-24.
	if got := f16ToF32(0x0001); got != float32(math.Ldexp(1, -24)) {
		t.Errorf("subnormal min: got %v want %v", got, math.Ldexp(1, -24))
	}
	// A larger subnormal.
	if got := f16ToF32(0x03FF); got <= 0 {
		t.Errorf("max subnormal should be positive, got %v", got)
	}
	// NaN: exponent all ones, nonzero mantissa.
	if got := f16ToF32(0x7E00); !math.IsNaN(float64(got)) {
		t.Errorf("0x7E00 should be NaN, got %v", got)
	}
	// +Inf / -Inf.
	if got := f16ToF32(0x7C00); !math.IsInf(float64(got), 1) {
		t.Errorf("0x7C00 should be +Inf, got %v", got)
	}
	if got := f16ToF32(0xFC00); !math.IsInf(float64(got), -1) {
		t.Errorf("0xFC00 should be -Inf, got %v", got)
	}
	// -0.0.
	if got := f16ToF32(0x8000); math.Float32bits(got) != 0x80000000 {
		t.Errorf("0x8000 should be -0.0, got bits %#x", math.Float32bits(got))
	}
}

func TestBF16ToF32(t *testing.T) {
	// 0x3F80 << 16 = 0x3F800000 = 1.0.
	if got := bf16ToF32(0x3F80); got != 1.0 {
		t.Errorf("bf16 1.0: got %v", got)
	}
	if got := bf16ToF32(0xBF80); got != -1.0 {
		t.Errorf("bf16 -1.0: got %v", got)
	}
}

func TestShapesEqual(t *testing.T) {
	if !shapesEqual([]int64{2, 3}, []int64{2, 3}) {
		t.Error("equal shapes reported unequal")
	}
	if shapesEqual([]int64{2, 3}, []int64{2}) {
		t.Error("different-rank shapes reported equal")
	}
	if shapesEqual([]int64{2, 3}, []int64{2, 4}) {
		t.Error("different shapes reported equal")
	}
}

// ── ReadBytes byte-level error paths ─────────────────────────────────────────

func TestReadBytes_Errors(t *testing.T) {
	// Too short (< 8 bytes).
	if _, err := ReadBytes([]byte{1, 2, 3}); err == nil {
		t.Error("short input should error")
	}
	// Header length exceeds file.
	bad := make([]byte, 8)
	binary.LittleEndian.PutUint64(bad, 9999)
	if _, err := ReadBytes(bad); err == nil {
		t.Error("oversized header length should error")
	}
	// Malformed JSON header.
	if _, err := ReadBytes(buildST(`not json`, nil)); err == nil {
		t.Error("bad JSON should error")
	}
	// data_offsets out of range.
	hdr := `{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,9999]}}`
	if _, err := ReadBytes(buildST(hdr, f32LE([]float32{1}))); err == nil {
		t.Error("out-of-range offsets should error")
	}
	// Unsupported dtype propagates from decode.
	hdrU := `{"t":{"dtype":"F8_E5M2","shape":[1],"data_offsets":[0,1]}}`
	if _, err := ReadBytes(buildST(hdrU, []byte{0})); err == nil {
		t.Error("unsupported dtype should error")
	}
}

// ── ReadBytes happy path with __metadata__ skipped & ordered keys ────────────

func TestReadBytes_MetadataSkipAndOrder(t *testing.T) {
	// Two tensors plus a metadata block. b's data follows a's.
	hdr := `{"__metadata__":{"k":"v"},` +
		`"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},` +
		`"b":{"dtype":"I8","shape":[2],"data_offsets":[8,10]}}`
	data := append(f32LE([]float32{1, 2}), 0x03, 0x04)
	entries, err := ReadBytes(buildST(hdr, data))
	if err != nil {
		t.Fatalf("ReadBytes: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2 (metadata skipped)", len(entries))
	}
	// orderedTopKeys should preserve a-before-b.
	if entries[0].Name != "a" || entries[1].Name != "b" {
		t.Errorf("order = %s,%s; want a,b", entries[0].Name, entries[1].Name)
	}
	if entries[0].Data[1] != 2 || entries[1].Data[0] != 3 {
		t.Errorf("data = %v / %v", entries[0].Data, entries[1].Data)
	}
}

func TestOrderedTopKeys(t *testing.T) {
	keys := orderedTopKeys([]byte(`{"first":1,"second":[1,2],"third":{"nested":9}}`))
	want := []string{"first", "second", "third"}
	if strings.Join(keys, ",") != strings.Join(want, ",") {
		t.Errorf("keys = %v, want %v", keys, want)
	}
	// Escaped quote inside a key/value must not break the walker.
	keys2 := orderedTopKeys([]byte(`{"a\"b":1,"c":2}`))
	if len(keys2) != 2 {
		t.Errorf("escaped-quote keys = %v, want 2 entries", keys2)
	}
}

// ── LoadTensors error paths via temp files ───────────────────────────────────

func TestLoadTensors_Errors(t *testing.T) {
	// Nonexistent file.
	if _, err := LoadTensors(t.TempDir() + "/missing.safetensors"); err == nil {
		t.Error("missing file should error")
	}
	// Malformed header JSON.
	bad := buildST(`xx`, nil)
	p := writeTemp(t, bad)
	if _, err := LoadTensors(p); err == nil {
		t.Error("bad JSON file should error")
	}
	// out-of-range offsets.
	hdr := `{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,9999]}}`
	p2 := writeTemp(t, buildST(hdr, f32LE([]float32{1})))
	if _, err := LoadTensors(p2); err == nil {
		t.Error("out-of-range offsets should error")
	}
}

// ── small byte helpers ───────────────────────────────────────────────────────

func f32LE(vals []float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func f64LE(vals []float64) []byte {
	b := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
	}
	return b
}

// u16of / u32of / u64of reinterpret a negative int as two's-complement
// unsigned, avoiding the "constant overflows" vet error on direct casts.
func u16of(v int16) uint16 { return uint16(v) }
func u32of(v int32) uint32 { return uint32(v) }
func u64of(v int64) uint64 { return uint64(v) }

func u16LE(vals []uint16) []byte {
	b := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[i*2:], v)
	}
	return b
}

func u32LE(vals []uint32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}

func u64LE(vals []uint64) []byte {
	b := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[i*8:], v)
	}
	return b
}
