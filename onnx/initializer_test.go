package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// approxEq32 compares two float32 values within tol; NaN matches NaN.
func approxEq32(a, b, tol float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func sliceApproxEq32(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if !approxEq32(got[i], want[i], tol) {
			t.Errorf("[%d]: got=%v want=%v (tol=%v)", i, got[i], want[i], tol)
		}
	}
}

// raw encoders for each width.
func rawFromUint16s(vs []uint16) []byte {
	raw := make([]byte, len(vs)*2)
	for i, v := range vs {
		binary.LittleEndian.PutUint16(raw[i*2:], v)
	}
	return raw
}

func rawFromUint32s(vs []uint32) []byte {
	raw := make([]byte, len(vs)*4)
	for i, v := range vs {
		binary.LittleEndian.PutUint32(raw[i*4:], v)
	}
	return raw
}

func rawFromUint64s(vs []uint64) []byte {
	raw := make([]byte, len(vs)*8)
	for i, v := range vs {
		binary.LittleEndian.PutUint64(raw[i*8:], v)
	}
	return raw
}

// ── FLOAT raw_data ────────────────────────────────────────────────────────────

func TestTensorFromProto_FLOAT_RawData_RoundTrip(t *testing.T) {
	vals := []float32{1.0, -2.5, 3.25, 0.0, -0.0, 4.5}
	bits := make([]uint32, len(vals))
	for i, v := range vals {
		bits[i] = math.Float32bits(v)
	}
	tp := &onnxpb.TensorProto{
		Name:     "f",
		Dims:     []int64{int64(len(vals))},
		DataType: int32(onnxpb.TensorProto_FLOAT),
		RawData:  rawFromUint32s(bits),
	}
	arena := uop.NewArena(8)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Float32 {
		t.Errorf("dtype=%v, want f32", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), vals, 0)
}

// FLOAT via float_data (typed field).
func TestTensorFromProto_FLOAT_FloatData(t *testing.T) {
	vals := []float32{1, 2, 3, 4}
	tp := &onnxpb.TensorProto{
		Name:      "f",
		Dims:      []int64{2, 2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: vals,
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	sliceApproxEq32(t, tt.Data(), vals, 0)
}

// ── FLOAT16 ───────────────────────────────────────────────────────────────────

func TestTensorFromProto_FLOAT16_RoundTrip(t *testing.T) {
	// Pick values exactly representable in f16: 1.0, -2.0, 0.5.
	// f16 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800.
	bits := []uint16{0x3C00, 0xC000, 0x3800}
	want := []float32{1.0, -2.0, 0.5}
	tp := &onnxpb.TensorProto{
		Name:     "h",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_FLOAT16),
		RawData:  rawFromUint16s(bits),
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Float16 {
		t.Errorf("dtype=%v, want f16", tt.DType())
	}
	// SetData re-quantises through f16 — the values we picked are exactly
	// representable so the round-trip is bit-exact.
	sliceApproxEq32(t, tt.Data(), want, 0)
}

func TestTensorFromProto_FLOAT16_Int32Data(t *testing.T) {
	// f16 bits stored in low 16 of int32_data, per spec.
	tp := &onnxpb.TensorProto{
		Name:      "h",
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_FLOAT16),
		Int32Data: []int32{0x3C00, 0x3800}, // 1.0, 0.5
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	sliceApproxEq32(t, tt.Data(), []float32{1.0, 0.5}, 0)
}

// TestFloat16Bits_Edges decodes +Inf, -Inf, NaN, subnormal directly through
// the package-private float16Bits helper. We assert the *bit-level decoder*
// behavior here so initializer round-trip tests below can rely on it.
func TestFloat16Bits_Edges(t *testing.T) {
	cases := []struct {
		name string
		bits uint16
		want func(float32) bool
	}{
		{"PosInf", 0x7C00, func(v float32) bool { return math.IsInf(float64(v), +1) }},
		{"NegInf", 0xFC00, func(v float32) bool { return math.IsInf(float64(v), -1) }},
		{"NaN", 0x7E00, func(v float32) bool { return math.IsNaN(float64(v)) }},
		{"PosZero", 0x0000, func(v float32) bool { return v == 0 && math.Signbit(float64(v)) == false }},
		{"NegZero", 0x8000, func(v float32) bool { return v == 0 && math.Signbit(float64(v)) == true }},
		{"Subnormal_min", 0x0001, func(v float32) bool {
			// Smallest positive subnormal f16: 2^-24.
			return approxEq32(v, float32(math.Ldexp(1, -24)), 1e-12)
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := float16Bits(c.bits)
			if !c.want(got) {
				t.Errorf("float16Bits(0x%04x)=%v fails predicate", c.bits, got)
			}
		})
	}
}

// ── BFLOAT16 ──────────────────────────────────────────────────────────────────

func TestTensorFromProto_BFLOAT16_RoundTrip(t *testing.T) {
	// bf16 = upper 16 bits of f32. 1.0=0x3F80, -2.0=0xC000, 0.5=0x3F00.
	bits := []uint16{0x3F80, 0xC000, 0x3F00}
	want := []float32{1.0, -2.0, 0.5}
	tp := &onnxpb.TensorProto{
		Name:     "b",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_BFLOAT16),
		RawData:  rawFromUint16s(bits),
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.BFloat16 {
		t.Errorf("dtype=%v, want bf16", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), want, 0)
}

func TestTensorFromProto_BFLOAT16_Int32Data(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:      "b",
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_BFLOAT16),
		Int32Data: []int32{0x3F80, 0x3F00}, // 1.0, 0.5
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	sliceApproxEq32(t, tt.Data(), []float32{1.0, 0.5}, 0)
}

func TestBfloat16Bits_Edges(t *testing.T) {
	cases := []struct {
		name string
		bits uint16
		want func(float32) bool
	}{
		{"PosInf", 0x7F80, func(v float32) bool { return math.IsInf(float64(v), +1) }},
		{"NegInf", 0xFF80, func(v float32) bool { return math.IsInf(float64(v), -1) }},
		{"NaN", 0x7FC0, func(v float32) bool { return math.IsNaN(float64(v)) }},
		{"PosZero", 0x0000, func(v float32) bool { return v == 0 && !math.Signbit(float64(v)) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := bfloat16Bits(c.bits)
			if !c.want(got) {
				t.Errorf("bfloat16Bits(0x%04x)=%v fails predicate", c.bits, got)
			}
		})
	}
}

// ── DOUBLE downcast + warning ─────────────────────────────────────────────────

func TestTensorFromProto_DOUBLE_RawData_DowncastsAndWarns(t *testing.T) {
	vals := []float64{1.5, -2.25, 3.0}
	bits := make([]uint64, len(vals))
	for i, v := range vals {
		bits[i] = math.Float64bits(v)
	}
	tp := &onnxpb.TensorProto{
		Name:     "d",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_DOUBLE),
		RawData:  rawFromUint64s(bits),
	}

	var captured []string
	restore := withWarnCapture(&captured)
	defer restore()

	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Float32 {
		t.Errorf("dtype=%v, want Float32 (downcast)", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), []float32{1.5, -2.25, 3.0}, 0)

	if len(captured) != 1 {
		t.Fatalf("warn callback invoked %d times, want 1; captured=%v", len(captured), captured)
	}
	if !strings.Contains(captured[0], "DOUBLE") {
		t.Errorf("warn %q does not mention DOUBLE", captured[0])
	}
	if !strings.Contains(captured[0], "f32") {
		t.Errorf("warn %q does not mention f32", captured[0])
	}
}

func TestTensorFromProto_DOUBLE_DoubleData(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:       "d",
		Dims:       []int64{3},
		DataType:   int32(onnxpb.TensorProto_DOUBLE),
		DoubleData: []float64{0.1, -0.2, 0.3},
	}
	var captured []string
	restore := withWarnCapture(&captured)
	defer restore()

	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	// f32 downcast: tolerate the rounding for 0.1, 0.2, 0.3.
	sliceApproxEq32(t, tt.Data(), []float32{0.1, -0.2, 0.3}, 1e-7)
	if len(captured) != 1 {
		t.Errorf("warn count=%d, want 1", len(captured))
	}
}

// ── INT8/INT16/INT32 raw_data round-trip + sign extension ─────────────────────

func TestTensorFromProto_INT8_RawData_SignExtension(t *testing.T) {
	// 0xFF as INT8 → -1 ; 0x80 → -128 ; 0x7F → 127.
	tp := &onnxpb.TensorProto{
		Name:     "i8",
		Dims:     []int64{4},
		DataType: int32(onnxpb.TensorProto_INT8),
		RawData:  []byte{0xFF, 0x80, 0x7F, 0x00},
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Int8 {
		t.Errorf("dtype=%v, want Int8", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), []float32{-1, -128, 127, 0}, 0)
}

func TestTensorFromProto_INT16_RawData_SignExtension(t *testing.T) {
	// 0xFFFF → -1 ; 0x8000 → -32768 ; 0x7FFF → 32767.
	tp := &onnxpb.TensorProto{
		Name:     "i16",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_INT16),
		RawData:  rawFromUint16s([]uint16{0xFFFF, 0x8000, 0x7FFF}),
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Int16 {
		t.Errorf("dtype=%v, want Int16", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), []float32{-1, -32768, 32767}, 0)
}

func TestTensorFromProto_INT32_RawData_SignExtension(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "i32",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_INT32),
		RawData:  rawFromUint32s([]uint32{0xFFFFFFFF, 0x80000000, 0x7FFFFFFF}),
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Int32 {
		t.Errorf("dtype=%v, want Int32", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), []float32{-1, -2147483648, 2147483647}, 0)
}

// ── UINT8/UINT16/UINT32 raw_data ──────────────────────────────────────────────

func TestTensorFromProto_UINT8_RawData(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "u8",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_UINT8),
		RawData:  []byte{0x00, 0xFF, 0x80},
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.UInt8 {
		t.Errorf("dtype=%v, want UInt8", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), []float32{0, 255, 128}, 0)
}

func TestTensorFromProto_UINT16_RawData(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "u16",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_UINT16),
		RawData:  rawFromUint16s([]uint16{0, 65535, 32768}),
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.UInt16 {
		t.Errorf("dtype=%v, want UInt16", tt.DType())
	}
	sliceApproxEq32(t, tt.Data(), []float32{0, 65535, 32768}, 0)
}

func TestTensorFromProto_UINT32_RawData(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "u32",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_UINT32),
		RawData:  rawFromUint32s([]uint32{0, 0xFFFFFFFF, 0x80000000}),
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.UInt32 {
		t.Errorf("dtype=%v, want UInt32", tt.DType())
	}
	// Note: 0xFFFFFFFF = 4294967295, but float32 cannot represent it exactly;
	// the nearest representable is 4.2949673e9. We allow that rounding.
	sliceApproxEq32(t, tt.Data(), []float32{0, 4294967295.0, 2147483648.0}, 256)
}

// ── BOOL ──────────────────────────────────────────────────────────────────────

func TestTensorFromProto_BOOL_RawData(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "b",
		Dims:     []int64{4},
		DataType: int32(onnxpb.TensorProto_BOOL),
		RawData:  []byte{0x00, 0x01, 0xFF, 0x00}, // false/true/true/false (any non-zero is true)
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Bool {
		t.Errorf("dtype=%v, want Bool", tt.DType())
	}
	// BOOL is treated as a 1-byte unsigned int by decodeRawIntWidth, so the
	// raw byte values come through as float32.
	sliceApproxEq32(t, tt.Data(), []float32{0, 1, 255, 0}, 0)
}

// ── INT64 downcast + overflow trap ────────────────────────────────────────────

func TestTensorFromProto_INT64_RawData_RoundTrip(t *testing.T) {
	vals := []int64{-100, -1, 0, 1, 100, math.MinInt32, math.MaxInt32}
	tp := &onnxpb.TensorProto{
		Name:     "i64",
		Dims:     []int64{int64(len(vals))},
		DataType: int32(onnxpb.TensorProto_INT64),
		RawData: func() []byte {
			out := make([]byte, len(vals)*8)
			for i, v := range vals {
				binary.LittleEndian.PutUint64(out[i*8:], uint64(v))
			}
			return out
		}(),
	}
	arena := uop.NewArena(8)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if tt.DType() != uop.Dtypes.Int32 {
		t.Errorf("dtype=%v, want Int32 (i64 downcast)", tt.DType())
	}
	want := []float32{-100, -1, 0, 1, 100, math.MinInt32, math.MaxInt32}
	// float32 can't represent ±2,147,483,647 exactly; tolerate the rounding.
	sliceApproxEq32(t, tt.Data(), want, 256)
}

func TestTensorFromProto_INT64_RawData_OverflowTrap(t *testing.T) {
	// MaxInt32 + 1 = 2147483648 overflows.
	tp := &onnxpb.TensorProto{
		Name:     "i64ovf",
		Dims:     []int64{1},
		DataType: int32(onnxpb.TensorProto_INT64),
		RawData: func() []byte {
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, uint64(int64(math.MaxInt32)+1))
			return b
		}(),
	}
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil {
		t.Fatalf("expected overflow error, got nil")
	}
	if !strings.Contains(err.Error(), "overflow") {
		t.Errorf("error %q missing 'overflow'", err.Error())
	}
	// And the lower bound trap.
	tp2 := &onnxpb.TensorProto{
		Name:     "i64ovf2",
		Dims:     []int64{1},
		DataType: int32(onnxpb.TensorProto_INT64),
		RawData: func() []byte {
			b := make([]byte, 8)
			v := int64(math.MinInt32) - 1
			binary.LittleEndian.PutUint64(b, uint64(v))
			return b
		}(),
	}
	_, err = tensorFromProto(arena, tp2, "test")
	if err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Errorf("expected lower-bound overflow error, got %v", err)
	}
}

func TestTensorFromProto_INT64_Int64Data_OverflowTrap(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "i64",
		Dims:     []int64{1},
		DataType: int32(onnxpb.TensorProto_INT64),
		Int64Data: []int64{int64(math.MaxInt32) + 1},
	}
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Errorf("expected overflow error, got %v", err)
	}
}

func TestTensorFromProto_INT64_Int64Data_RoundTrip(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:      "i64",
		Dims:      []int64{3},
		DataType:  int32(onnxpb.TensorProto_INT64),
		Int64Data: []int64{-7, 0, 7},
	}
	arena := uop.NewArena(2)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	sliceApproxEq32(t, tt.Data(), []float32{-7, 0, 7}, 0)
}

// ── UINT64: documented production discrepancy ─────────────────────────────────

// TestTensorFromProto_UINT64_IsRejectedAtDtypeGate pins the observed behavior:
// initializer.go's onnxDType() table omits UINT64, so tensorFromProto rejects
// any UINT64 TensorProto with "unsupported dtype 13" — even though
// decodeTensorData has a complete UINT64 branch (raw_data + uint64_data) that
// is therefore dead code. This test exists so the inconsistency is loudly
// visible; do NOT change the assertion without fixing the dtype gate.
func TestTensorFromProto_UINT64_IsRejectedAtDtypeGate(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:       "u64",
		Dims:       []int64{2},
		DataType:   int32(onnxpb.TensorProto_UINT64),
		Uint64Data: []uint64{42, 99},
	}
	arena := uop.NewArena(4)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil {
		t.Fatalf("UINT64 unexpectedly accepted; if you added it to onnxDType, update this test")
	}
	if !strings.Contains(err.Error(), "unsupported dtype") {
		t.Errorf("error %q missing 'unsupported dtype'", err.Error())
	}
}

// ── INT32 via int32_data typed field ──────────────────────────────────────────

func TestTensorFromProto_INT32_Int32Data(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:      "i32",
		Dims:      []int64{4},
		DataType:  int32(onnxpb.TensorProto_INT32),
		Int32Data: []int32{-100, -1, 0, 100},
	}
	arena := uop.NewArena(4)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	sliceApproxEq32(t, tt.Data(), []float32{-100, -1, 0, 100}, 0)
}

// ── unsupported dtypes ────────────────────────────────────────────────────────

func TestTensorFromProto_UnsupportedDtypes(t *testing.T) {
	cases := []struct {
		name string
		dt   onnxpb.TensorProto_DataType
	}{
		{"UNDEFINED", onnxpb.TensorProto_UNDEFINED},
		{"STRING", onnxpb.TensorProto_STRING},
		{"COMPLEX64", onnxpb.TensorProto_COMPLEX64},
		{"COMPLEX128", onnxpb.TensorProto_COMPLEX128},
		{"FLOAT8E4M3FN", onnxpb.TensorProto_FLOAT8E4M3FN},
		{"FLOAT8E4M3FNUZ", onnxpb.TensorProto_FLOAT8E4M3FNUZ},
		{"FLOAT8E5M2", onnxpb.TensorProto_FLOAT8E5M2},
		{"FLOAT8E5M2FNUZ", onnxpb.TensorProto_FLOAT8E5M2FNUZ},
		{"UINT4", onnxpb.TensorProto_UINT4},
		{"INT4", onnxpb.TensorProto_INT4},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tp := &onnxpb.TensorProto{
				Name:     "u_" + c.name,
				Dims:     []int64{1},
				DataType: int32(c.dt),
				RawData:  []byte{0, 0, 0, 0, 0, 0, 0, 0},
			}
			arena := uop.NewArena(2)
			_, err := tensorFromProto(arena, tp, "test")
			if err == nil {
				t.Fatalf("expected unsupported dtype error for %s, got nil", c.name)
			}
			if !strings.Contains(err.Error(), "unsupported dtype") {
				t.Errorf("error %q missing 'unsupported dtype'", err.Error())
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("%d", int32(c.dt))) {
				t.Errorf("error %q missing the unsupported dtype number %d", err.Error(), int32(c.dt))
			}
		})
	}
}

// ── nil and external_data and negative dims ───────────────────────────────────

func TestTensorFromProto_NilProto(t *testing.T) {
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, nil, "test")
	if err == nil {
		t.Fatalf("expected error on nil TensorProto, got nil")
	}
	if !strings.Contains(err.Error(), "nil TensorProto") {
		t.Errorf("error %q missing 'nil TensorProto'", err.Error())
	}
}

func TestTensorFromProto_ExternalData(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:         "ext",
		Dims:         []int64{1},
		DataType:     int32(onnxpb.TensorProto_FLOAT),
		DataLocation: onnxpb.TensorProto_EXTERNAL,
	}
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil {
		t.Fatalf("expected error on EXTERNAL data, got nil")
	}
	if !strings.Contains(err.Error(), "external-data") {
		t.Errorf("error %q missing 'external-data'", err.Error())
	}
}

func TestTensorFromProto_NegativeDim(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:      "neg",
		Dims:      []int64{-1, 2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2},
	}
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil {
		t.Fatalf("expected error on negative dim, got nil")
	}
	if !strings.Contains(err.Error(), "negative dim") {
		t.Errorf("error %q missing 'negative dim'", err.Error())
	}
}

// ── empty raw_data ────────────────────────────────────────────────────────────

func TestTensorFromProto_EmptyTensor(t *testing.T) {
	// Dims=[0] with empty raw_data: zero elements, decode succeeds.
	tp := &onnxpb.TensorProto{
		Name:     "empty",
		Dims:     []int64{0},
		DataType: int32(onnxpb.TensorProto_FLOAT),
		RawData:  nil,
	}
	arena := uop.NewArena(2)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	if got := tt.Data(); len(got) != 0 {
		t.Errorf("empty tensor Data length=%d, want 0", len(got))
	}
}

// ── length mismatch errors ────────────────────────────────────────────────────

func TestTensorFromProto_LengthMismatch_Float(t *testing.T) {
	// Dims claim 3 elements but only 2 floats in float_data.
	tp := &onnxpb.TensorProto{
		Name:      "lm",
		Dims:      []int64{3},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2},
	}
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil {
		t.Fatalf("expected length mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "elem count") {
		t.Errorf("error %q missing 'elem count'", err.Error())
	}
}

func TestTensorFromProto_LengthMismatch_RawFloat(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "lm",
		Dims:     []int64{3},
		DataType: int32(onnxpb.TensorProto_FLOAT),
		RawData:  []byte{0, 0, 0, 0}, // 4 bytes = 1 float, want 3
	}
	arena := uop.NewArena(2)
	_, err := tensorFromProto(arena, tp, "test")
	if err == nil {
		t.Fatalf("expected length mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "raw_data length") {
		t.Errorf("error %q missing 'raw_data length'", err.Error())
	}
}

// ── endian-decode determinism ─────────────────────────────────────────────────

// TestTensorFromProto_LittleEndianDecode emits raw bytes for the value
// 0x01020304 in INT32 little-endian. On a little-endian host an unsafe.Slice
// cast would happen to agree with binary.LittleEndian; on a big-endian host
// the unsafe cast would produce 0x04030201 instead. The package decodes
// explicitly via binary.LittleEndian, so this test pins the contract: a
// well-known little-endian byte sequence always decodes to the same integer.
func TestTensorFromProto_LittleEndianDecode(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Name:     "le",
		Dims:     []int64{1},
		DataType: int32(onnxpb.TensorProto_INT32),
		RawData:  []byte{0x04, 0x03, 0x02, 0x01}, // little-endian 0x01020304
	}
	arena := uop.NewArena(2)
	tt, err := tensorFromProto(arena, tp, "test")
	if err != nil {
		t.Fatalf("tensorFromProto: %v", err)
	}
	got := tt.Data()
	if len(got) != 1 || got[0] != float32(0x01020304) {
		t.Errorf("LE decode: got=%v, want [%v]", got, float32(0x01020304))
	}
}

// ── initializer hash key determinism ──────────────────────────────────────────

// TestInitializerHashKey_StructuralIdentity verifies the hash key depends on
// dtype + dims + payload but NOT on the name field. This is the contract that
// Runner.Import relies on for interning. Already partly exercised by
// TestInitializerInterning (which goes through Import); here we attack the
// pure helper.
func TestInitializerHashKey_StructuralIdentity(t *testing.T) {
	a := &onnxpb.TensorProto{
		Name:      "a",
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2},
	}
	b := &onnxpb.TensorProto{
		Name:      "b", // different name
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2},
	}
	ha := initializerHashKey(a)
	hb := initializerHashKey(b)
	if ha != hb {
		t.Errorf("hash a=%x != b=%x; different names should not affect hash", ha, hb)
	}

	// Different dtype → different hash.
	c := &onnxpb.TensorProto{
		Name:      "a",
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_DOUBLE),
		FloatData: []float32{1, 2},
	}
	if initializerHashKey(c) == ha {
		t.Errorf("hash matches across dtype change")
	}

	// Different dims → different hash.
	d := &onnxpb.TensorProto{
		Name:      "a",
		Dims:      []int64{1, 2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2},
	}
	if initializerHashKey(d) == ha {
		t.Errorf("hash matches across dims change")
	}

	// Different payload → different hash.
	e := &onnxpb.TensorProto{
		Name:      "a",
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 99},
	}
	if initializerHashKey(e) == ha {
		t.Errorf("hash matches across payload change")
	}
}

// TestInitializerHashKey_RawVsTypedCover ensures hashing covers RawData, the
// Int32Data path, Int64Data, DoubleData, and Uint64Data fields. The contract
// says structurally equivalent inputs must alias even when carried in
// different storage fields IF the encoded bytes match; we don't test that
// equivalence — we test that the typed fields are mixed in at all, by
// observing different values change the hash.
func TestInitializerHashKey_TypedFieldsAffectHash(t *testing.T) {
	base := func() *onnxpb.TensorProto {
		return &onnxpb.TensorProto{
			Dims:     []int64{1},
			DataType: int32(onnxpb.TensorProto_INT32),
		}
	}

	// Int32Data sensitivity.
	a, b := base(), base()
	a.Int32Data = []int32{1}
	b.Int32Data = []int32{2}
	if initializerHashKey(a) == initializerHashKey(b) {
		t.Errorf("hash insensitive to Int32Data change")
	}

	// Int64Data sensitivity.
	c, d := base(), base()
	c.DataType = int32(onnxpb.TensorProto_INT64)
	d.DataType = int32(onnxpb.TensorProto_INT64)
	c.Int64Data = []int64{1}
	d.Int64Data = []int64{2}
	if initializerHashKey(c) == initializerHashKey(d) {
		t.Errorf("hash insensitive to Int64Data change")
	}

	// DoubleData sensitivity.
	e, f := base(), base()
	e.DataType = int32(onnxpb.TensorProto_DOUBLE)
	f.DataType = int32(onnxpb.TensorProto_DOUBLE)
	e.DoubleData = []float64{1.0}
	f.DoubleData = []float64{2.0}
	if initializerHashKey(e) == initializerHashKey(f) {
		t.Errorf("hash insensitive to DoubleData change")
	}

	// Uint64Data sensitivity.
	g, h := base(), base()
	g.DataType = int32(onnxpb.TensorProto_UINT64)
	h.DataType = int32(onnxpb.TensorProto_UINT64)
	g.Uint64Data = []uint64{1}
	h.Uint64Data = []uint64{2}
	if initializerHashKey(g) == initializerHashKey(h) {
		t.Errorf("hash insensitive to Uint64Data change")
	}
}

// ── onnxDType matrix ──────────────────────────────────────────────────────────

func TestOnnxDType_AllSupported(t *testing.T) {
	cases := []struct {
		name string
		in   onnxpb.TensorProto_DataType
		want *uop.DType
		dc   bool
	}{
		{"FLOAT", onnxpb.TensorProto_FLOAT, uop.Dtypes.Float32, false},
		{"FLOAT16", onnxpb.TensorProto_FLOAT16, uop.Dtypes.Float16, false},
		{"BFLOAT16", onnxpb.TensorProto_BFLOAT16, uop.Dtypes.BFloat16, false},
		{"DOUBLE", onnxpb.TensorProto_DOUBLE, uop.Dtypes.Float32, true},
		{"INT8", onnxpb.TensorProto_INT8, uop.Dtypes.Int8, false},
		{"INT16", onnxpb.TensorProto_INT16, uop.Dtypes.Int16, false},
		{"INT32", onnxpb.TensorProto_INT32, uop.Dtypes.Int32, false},
		{"INT64", onnxpb.TensorProto_INT64, uop.Dtypes.Int32, true},
		{"UINT8", onnxpb.TensorProto_UINT8, uop.Dtypes.UInt8, false},
		{"UINT16", onnxpb.TensorProto_UINT16, uop.Dtypes.UInt16, false},
		{"UINT32", onnxpb.TensorProto_UINT32, uop.Dtypes.UInt32, false},
		{"BOOL", onnxpb.TensorProto_BOOL, uop.Dtypes.Bool, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dt, _, dc, ok := onnxDType(int32(c.in))
			if !ok {
				t.Fatalf("ok=false for %s, want true", c.name)
			}
			if dt != c.want {
				t.Errorf("dtype=%v, want %v", dt, c.want)
			}
			if dc != c.dc {
				t.Errorf("downcast=%v, want %v", dc, c.dc)
			}
		})
	}
}

func TestOnnxDType_Unsupported(t *testing.T) {
	for _, dt := range []onnxpb.TensorProto_DataType{
		onnxpb.TensorProto_UNDEFINED,
		onnxpb.TensorProto_STRING,
		onnxpb.TensorProto_COMPLEX64,
		onnxpb.TensorProto_COMPLEX128,
		onnxpb.TensorProto_FLOAT8E4M3FN,
		onnxpb.TensorProto_FLOAT8E4M3FNUZ,
		onnxpb.TensorProto_FLOAT8E5M2,
		onnxpb.TensorProto_FLOAT8E5M2FNUZ,
		onnxpb.TensorProto_UINT4,
		onnxpb.TensorProto_INT4,
		onnxpb.TensorProto_UINT64, // covered by raw decode but onnxDType maps it? — see test below
	} {
		t.Run(dt.String(), func(t *testing.T) {
			_, _, _, ok := onnxDType(int32(dt))
			// UINT64 is not in the onnxDType table; the raw decode path
			// handles it via decodeTensorData. Document & assert.
			if dt == onnxpb.TensorProto_UINT64 {
				if ok {
					t.Errorf("onnxDType reports UINT64 supported, but the table omits it (decode handles it directly)")
				}
				return
			}
			if ok {
				t.Errorf("onnxDType reports %s as supported", dt)
			}
		})
	}
}

// isSignedDType: the helper is used internally; sanity-test it.
func TestIsSignedDType(t *testing.T) {
	signed := []onnxpb.TensorProto_DataType{
		onnxpb.TensorProto_INT8, onnxpb.TensorProto_INT16,
		onnxpb.TensorProto_INT32, onnxpb.TensorProto_INT64,
	}
	unsigned := []onnxpb.TensorProto_DataType{
		onnxpb.TensorProto_UINT8, onnxpb.TensorProto_UINT16,
		onnxpb.TensorProto_UINT32, onnxpb.TensorProto_BOOL,
		onnxpb.TensorProto_FLOAT,
	}
	for _, dt := range signed {
		if !isSignedDType(dt) {
			t.Errorf("isSignedDType(%s)=false, want true", dt)
		}
	}
	for _, dt := range unsigned {
		if isSignedDType(dt) {
			t.Errorf("isSignedDType(%s)=true, want false", dt)
		}
	}
}
