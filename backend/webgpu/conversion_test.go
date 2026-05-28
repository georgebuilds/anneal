package webgpu

import (
	"math"
	"testing"
)

func TestFloat32ToFloat16_Adversarial(t *testing.T) {
	tests := []struct {
		name string
		in   uint32 // f32 bit pattern
		want uint16 // expected f16 bit pattern
	}{
		// Same cases as uop to confirm identical behavior/bugs
		{"positive zero", 0x00000000, 0x0000},
		{"negative zero", 0x80000000, 0x8000},
		{"one", 0x3F800000, 0x3C00},
		{"negative one", 0xBF800000, 0xBC00},
		{"max finite", 0x477FE000, 0x7BFF},
		{"min normal", 0x38800000, 0x0400},

		// 2. Rounding (Expected Round-to-Nearest-Even)
		{"1.0 + 1/2048 (round down)", 0x3F801000, 0x3C00},
		{"1.0 + 1/1024 (exact)", 0x3F802000, 0x3C01},
		{"1.0 + 1.5/1024 (half-way, round up to even)", 0x3F803000, 0x3C02},
		{"1.0 + 2.5/1024 (half-way, round down to even)", 0x3F805000, 0x3C02},

		// 3. Subnormals
		{"min subnormal", 0x33800000, 0x0001},
		{"underflow to zero", 0x33000000, 0x0000},

		// 4. Overflow
		{"above max finite (rounds up to Inf)", 0x477FF000, 0x7C00},
		{"large overflow", 0x501502F9, 0x7C00},

		// 5. Special values
		{"positive infinity", 0x7F800000, 0x7C00},
		{"quiet NaN", 0x7FC00000, 0x7E00},
		{"signaling NaN", 0x7F800001, 0x7C01},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := math.Float32frombits(tc.in)
			got := float32ToFloat16(in)
			if got != tc.want {
				t.Errorf("float32ToFloat16(0x%08x) = 0x%04x, want 0x%04x", tc.in, got, tc.want)
			}
		})
	}
}

func TestFloat16ToFloat32_Adversarial(t *testing.T) {
	tests := []struct {
		name string
		in   uint16 // f16 bit pattern
		want uint32 // expected f32 bit pattern
	}{
		{"one", 0x3C00, 0x3F800000},
		{"min subnormal", 0x0001, 0x33800000},
		{"NaN", 0x7C01, 0x7F802000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := float16ToFloat32(tc.in)
			gotBits := math.Float32bits(got)
			if tc.name == "NaN" {
				if !math.IsNaN(float64(got)) {
					t.Errorf("float16ToFloat32(0x%04x) = %g, want NaN", tc.in, got)
				}
			} else if gotBits != tc.want {
				t.Errorf("float16ToFloat32(0x%04x) = 0x%08x, want 0x%08x", tc.in, gotBits, tc.want)
			}
		})
	}
}

func TestBulkConversion_Adversarial(t *testing.T) {
	data := []float32{0.0, 1.0, 2.0, 65504.0}
	// Expected f16 bit patterns (LE):
	// 0.0 -> 0x0000 -> 0x00, 0x00
	// 1.0 -> 0x3C00 -> 0x00, 0x3C
	// 2.0 -> 0x4000 -> 0x00, 0x40
	// 65504.0 -> 0x7BFF -> 0xFF, 0x7B
	wantBytes := []byte{0x00, 0x00, 0x00, 0x3C, 0x00, 0x40, 0xFF, 0x7B}

	t.Run("float32sToF16Bytes", func(t *testing.T) {
		got := float32sToF16Bytes(data)
		if len(got) != len(wantBytes) {
			t.Fatalf("length mismatch: got %d, want %d", len(got), len(wantBytes))
		}
		for i := range got {
			if got[i] != wantBytes[i] {
				t.Errorf("byte %d: got 0x%02x, want 0x%02x", i, got[i], wantBytes[i])
			}
		}
	})

	t.Run("bytesToFloat32s (f32)", func(t *testing.T) {
		// Test standard f32 bytes to f32s
		f32Data := []float32{1.0, 2.0}
		// 1.0 -> 0x3F800000 -> 0x00, 0x00, 0x80, 0x3F
		// 2.0 -> 0x40000000 -> 0x00, 0x00, 0x00, 0x40
		f32Bytes := []byte{0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40}
		got := bytesToFloat32s(f32Bytes)
		if len(got) != len(f32Data) {
			t.Fatalf("length mismatch: got %d, want %d", len(got), len(f32Data))
		}
		for i := range got {
			if got[i] != f32Data[i] {
				t.Errorf("float %d: got %g, want %g", i, got[i], f32Data[i])
			}
		}
	})
}
