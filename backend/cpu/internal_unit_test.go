package cpu

// Internal unit tests for Slice 2: newBuffer dtype routing, the Buffer
// byte boundary (Write/Read little-endian contract shared with WebGPU),
// and the integer-comparison helper. These cover branches the E2E
// tensor-pipeline tests cannot reach (error paths, the CmpNe arm).

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

func TestNewBufferDtypeRouting(t *testing.T) {
	cases := []struct {
		name    string
		dt      *uop.DType
		wantF32 bool
		wantI32 bool
		wantErr bool
	}{
		{"nil→f32", nil, true, false, false},
		{"f32", uop.Dtypes.Float32, true, false, false},
		{"image→f32", uop.Dtypes.ImageFloat32, true, false, false},
		{"f16→quantized f32", uop.Dtypes.Float16, true, false, false},
		{"bf16→quantized f32", uop.Dtypes.BFloat16, true, false, false},
		{"e4m3→quantized f32", uop.Dtypes.FP8E4M3, true, false, false},
		{"e5m2→quantized f32", uop.Dtypes.FP8E5M2, true, false, false},
		{"i32", uop.Dtypes.Int32, false, true, false},
		{"u32", uop.Dtypes.UInt32, false, true, false},
		{"f64 unsupported", uop.Dtypes.Float64, false, false, true},
		{"i64 unsupported", uop.Dtypes.Int64, false, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := newBuffer(8, tc.dt)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("want error, got buffer %+v", b)
				}
				return
			}
			if err != nil {
				t.Fatalf("newBuffer: %v", err)
			}
			if (b.asF32() != nil) != tc.wantF32 || (b.asI32() != nil) != tc.wantI32 {
				t.Errorf("storage routing: f32=%v i32=%v, want f32=%v i32=%v",
					b.asF32() != nil, b.asI32() != nil, tc.wantF32, tc.wantI32)
			}
		})
	}

	if _, err := newBuffer(-1, uop.Dtypes.Float32); err == nil {
		t.Error("negative elems must error")
	}
	b, err := newBuffer(0, uop.Dtypes.Float32)
	if err != nil || b.Size() != 1 {
		t.Errorf("zero elems must floor to 1 (got size=%d err=%v)", b.Size(), err)
	}
}

func TestBufferWriteReadRoundTrip(t *testing.T) {
	t.Run("f32", func(t *testing.T) {
		b, err := newBuffer(3, uop.Dtypes.Float32)
		if err != nil {
			t.Fatal(err)
		}
		vals := []float32{1.5, -2.25, math.Float32frombits(0x7FC00000)} // incl. qNaN bits
		raw := make([]byte, 12)
		for i, v := range vals {
			putLE32(raw[i*4:], math.Float32bits(v))
		}
		if err := b.Write(raw); err != nil {
			t.Fatalf("Write: %v", err)
		}
		got, err := b.Read()
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		for i := range raw {
			if got[i] != raw[i] {
				t.Fatalf("byte %d: got %#x want %#x", i, got[i], raw[i])
			}
		}
		if b.DType().Scalar() != uop.Dtypes.Float32 {
			t.Errorf("DType: got %v", b.DType())
		}
	})

	t.Run("i32", func(t *testing.T) {
		b, err := newBuffer(2, uop.Dtypes.Int32)
		if err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, 8)
		putLE32(raw[0:], uint32(0x80000001)) // negative int32 bit pattern
		putLE32(raw[4:], 42)
		if err := b.Write(raw); err != nil {
			t.Fatalf("Write: %v", err)
		}
		if got := b.asI32(); got[0] != int32(-2147483647) || got[1] != 42 {
			t.Fatalf("i32 decode: got %v", got)
		}
		out, err := b.Read()
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		for i := range raw {
			if out[i] != raw[i] {
				t.Fatalf("byte %d: got %#x want %#x", i, out[i], raw[i])
			}
		}
	})

	t.Run("errors", func(t *testing.T) {
		b, _ := newBuffer(4, uop.Dtypes.Float32)
		if err := b.Write(make([]byte, 3)); err == nil {
			t.Error("short f32 write must error")
		}
		bi, _ := newBuffer(4, uop.Dtypes.Int32)
		if err := bi.Write(make([]byte, 3)); err == nil {
			t.Error("short i32 write must error")
		}
		b.Release()
		if err := b.Write(make([]byte, 16)); err == nil {
			t.Error("released-buffer write must error")
		}
		if _, err := b.Read(); err == nil {
			t.Error("released-buffer read must error")
		}
		if b.Size() != 0 {
			t.Errorf("released size: %d", b.Size())
		}
	})
}

func TestOutputAsF32(t *testing.T) {
	// f32: verbatim copy.
	bf, _ := newBuffer(3, uop.Dtypes.Float32)
	copy(bf.asF32(), []float32{1.5, -2, 3})
	got := outputAsF32(bf, 3)
	if got[0] != 1.5 || got[1] != -2 || got[2] != 3 {
		t.Errorf("f32 output: %v", got)
	}

	// i32: bit-reinterpret (GPU readback parity, NOT numeric cast).
	bi, _ := newBuffer(2, uop.Dtypes.Int32)
	copy(bi.asI32(), []int32{7, -1})
	got = outputAsF32(bi, 2)
	if math.Float32bits(got[0]) != 7 || math.Float32bits(got[1]) != 0xFFFFFFFF {
		t.Errorf("i32 output must be bit patterns: bits=[%#x %#x]",
			math.Float32bits(got[0]), math.Float32bits(got[1]))
	}

	// Released buffer: nil.
	bf.Release()
	if outputAsF32(bf, 1) != nil {
		t.Error("released buffer must yield nil")
	}
}

func TestIntCmpHolds(t *testing.T) {
	cases := []struct {
		op   uop.Op
		a, b int64
		want bool
	}{
		{uop.OpCmpLt, 1, 2, true},
		{uop.OpCmpLt, 2, 2, false},
		{uop.OpCmpNe, 1, 2, true},
		{uop.OpCmpNe, 2, 2, false},
		{uop.OpCmpEq, 2, 2, true},
		{uop.OpCmpEq, 1, 2, false},
	}
	for _, tc := range cases {
		if got := intCmpHolds(tc.op, tc.a, tc.b); got != tc.want {
			t.Errorf("intCmpHolds(%v, %d, %d) = %v, want %v", tc.op, tc.a, tc.b, got, tc.want)
		}
	}
}

// putLE32 writes a uint32 little-endian (avoids importing encoding/binary
// in the test for two call sites).
func putLE32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}
