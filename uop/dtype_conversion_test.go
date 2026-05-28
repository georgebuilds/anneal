package uop

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
		// 1. Exact representable values
		{"positive zero", 0x00000000, 0x0000},
		{"negative zero", 0x80000000, 0x8000},
		{"one", 0x3F800000, 0x3C00},
		{"negative one", 0xBF800000, 0xBC00},
		{"two", 0x40000000, 0x4000},
		{"half", 0x3F000000, 0x3800},
		{"quarter", 0x3E800000, 0x3400},
		{"max finite", 0x477FE000, 0x7BFF},
		{"min normal", 0x38800000, 0x0400},

		// 2. Rounding (Expected Round-to-Nearest-Even)
		{"1.0 + 1/2048 (round down)", 0x3F801000, 0x3C00},
		{"1.0 + 1/1024 (exact)", 0x3F802000, 0x3C01},
		{"1.0 + 1.5/1024 (half-way, round up to even)", 0x3F803000, 0x3C02},
		{"1.0 + 2.5/1024 (half-way, round down to even)", 0x3F805000, 0x3C02},

		// 3. Subnormals
		{"min subnormal", 0x33800000, 0x0001}, // 2^-24
		{"mid subnormal", 0x35800000, 0x0010}, // 2^-20
		{"just above subnormal threshold", 0x38802000, 0x0401},
		{"just below subnormal threshold", 0x387FE000, 0x03FF}, // 2^-14 - epsilon
		{"underflow to zero", 0x33000000, 0x0000},

		// 4. Overflow
		{"above max finite (rounds up to Inf)", 0x477FF000, 0x7C00},
		{"large overflow", 0x501502F9, 0x7C00},
		{"negative overflow", 0xD01502F9, 0xFC00},

		// 5. Special values
		{"positive infinity", 0x7F800000, 0x7C00},
		{"negative infinity", 0xFF800000, 0xFC00},
		{"quiet NaN", 0x7FC00000, 0x7E00},
		{"signaling NaN", 0x7F800001, 0x7C01},

		// 6. Slice 4 oracle value
		{"1.0 + 1/256 (exact in f16)", 0x3F808000, 0x3C04},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := math.Float32frombits(tc.in)
			got := Float32ToFloat16(in)
			if got != tc.want {
				t.Errorf("Float32ToFloat16(0x%08x [%g]) = 0x%04x, want 0x%04x", tc.in, in, got, tc.want)
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
		{"positive zero", 0x0000, 0x00000000},
		{"negative zero", 0x8000, 0x80000000},
		{"one", 0x3C00, 0x3F800000},
		{"negative one", 0xBC00, 0xBF800000},
		{"max finite", 0x7BFF, 0x477FE000},
		{"min normal", 0x0400, 0x38800000},
		{"min subnormal", 0x0001, 0x33800000},
		{"positive infinity", 0x7C00, 0x7F800000},
		{"negative infinity", 0xFC00, 0xFF800000},
		// NaN bit patterns may differ but NaN-ness must be preserved.
		// We'll check bits for our specific implementation's behavior.
		{"NaN", 0x7C01, 0x7F802000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Float16ToFloat32(tc.in)
			gotBits := math.Float32bits(got)
			if tc.name == "NaN" {
				if !math.IsNaN(float64(got)) {
					t.Errorf("Float16ToFloat32(0x%04x) = %g, want NaN", tc.in, got)
				}
			} else {
				if gotBits != tc.want {
					t.Errorf("Float16ToFloat32(0x%04x) = 0x%08x, want 0x%08x", tc.in, gotBits, tc.want)
				}
			}
		})
	}
}

func TestQuantize_Adversarial(t *testing.T) {
	tests := []struct {
		name  string
		dtype *DType
		in    float32
		want  float32
	}{
		{"f32 identity", Dtypes.Float32, 1.2345, 1.2345},
		{"f16 quantization", Dtypes.Float16, 1.0 + 1.5/1024, 1.0 + 2.0/1024}, // RTNE
		{"bf16 quantization", Dtypes.BFloat16, 1.0 + 1.0/256, 1.0},           // Truncation
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.dtype.Quantize(tc.in)
			if tc.dtype.Scalar() == Dtypes.Float16 {
				// For f16 we check bits of result
				gotBits := math.Float32bits(got)
				wantBits := math.Float32bits(tc.want)
				if gotBits != wantBits {
					t.Errorf("Quantize(%g, f16) = 0x%08x, want 0x%08x", tc.in, gotBits, wantBits)
				}
			} else if got != tc.want {
				t.Errorf("Quantize(%g, %v) = %g, want %g", tc.in, tc.dtype, got, tc.want)
			}
		})
	}
}
