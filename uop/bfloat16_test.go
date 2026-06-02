package uop

import (
	"math"
	"testing"
)

// TestFloat32ToBFloat16_RTNE verifies that f32 -> bf16 narrowing implements
// round-to-nearest-even per the PyTorch / TensorFlow / Eigen reference
// (c10::detail::round_to_nearest_even). Cases cover:
//
//   - exactly-representable bf16 grid points (identity)
//   - non-halfway round-down and round-up
//   - exact halfways that tie to even (LSB=0 keeps, LSB=1 carries)
//   - sign preservation (positive vs negative same magnitude)
//   - +/-Inf passing through the bias formula unchanged
//   - NaN canonicalization (any NaN -> 0x7FC0)
//   - negative zero preservation
//   - subnormal f32 rounding into bf16 subnormal / zero
//   - large finite f32 rounding into bf16 +Inf when overflow is the
//     nearest representable value
func TestFloat32ToBFloat16_RTNE(t *testing.T) {
	cases := []struct {
		name string
		in   float32
		want uint16
	}{
		{"+0.0 identity", 0, 0x0000},
		{"-0.0 preserves sign", math.Float32frombits(0x80000000), 0x8000},
		{"1.0 identity", 1.0, 0x3F80},
		{"-1.0 identity", -1.0, 0xBF80},
		{"+Inf passes through", float32(math.Inf(1)), 0x7F80},
		{"-Inf passes through", float32(math.Inf(-1)), 0xFF80},

		// 1.0 + 1/256 = 1.00390625 is exactly halfway between 1.0 and
		// 1.0078125 (the next bf16 grid point). 1.0's bf16 mantissa LSB
		// is 0 (even), so RTNE rounds to 1.0.
		{"tie-to-even rounds down (kept LSB=0)", math.Float32frombits(0x3F808000), 0x3F80},

		// 1.0 + 3/256 in bf16: 1.0 + 1/128 = 0x3F81. The bf16 grid step
		// above 0x3F81 is 0x3F82. 0x3F818000 sits exactly between 0x3F81
		// and 0x3F82; 0x3F81's mantissa LSB is 1 (odd), so RTNE rounds
		// up to 0x3F82.
		{"tie-to-even rounds up (kept LSB=1)", math.Float32frombits(0x3F818000), 0x3F82},

		// Non-halfway round-down: low 16 bits 0x7FFF < 0x8000, so the
		// (u + bias) carry never propagates into the kept LSB.
		{"non-halfway rounds down", math.Float32frombits(0x3F807FFF), 0x3F80},

		// Non-halfway round-up: low 16 bits 0x8001 > 0x8000, so the
		// (u + bias) carry always propagates.
		{"non-halfway rounds up", math.Float32frombits(0x3F808001), 0x3F81},

		// NaN: any qNaN / sNaN must canonicalize to 0x7FC0.
		{"qNaN canonicalizes", math.Float32frombits(0x7FC00000), 0x7FC0},
		{"sNaN canonicalizes", math.Float32frombits(0x7F800001), 0x7FC0},
		{"-NaN canonicalizes", math.Float32frombits(0xFFC00000), 0x7FC0},

		// Subnormal f32 with magnitude < ~1.18e-38: bias formula rounds
		// the tiny value to bf16 +0.
		{"subnormal f32 rounds to +0", math.Float32frombits(0x00000001), 0x0000},

		// Exactly-half above bf16 max (0x7F7F): rounding-overflow
		// produces +Inf per IEEE semantics (the canonical Eigen test).
		{"finite rounds up to +Inf", math.Float32frombits(0x7F7F8000), 0x7F80},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Float32ToBFloat16(c.in)
			if got != c.want {
				t.Fatalf("Float32ToBFloat16(0x%08X) = 0x%04X, want 0x%04X",
					math.Float32bits(c.in), got, c.want)
			}
		})
	}
}

// TestBFloat16ToFloat32_BitsExact widens known bf16 patterns and verifies the
// IEEE-754 reconstruction. Roundtrip with Float32ToBFloat16 must be the
// identity for any representable bf16 value (the bf16 grid is closed under
// the upper-16-bit projection).
func TestBFloat16ToFloat32_BitsExact(t *testing.T) {
	cases := []struct {
		bits uint16
		want uint32
	}{
		{0x0000, 0x00000000},
		{0x8000, 0x80000000},
		{0x3F80, 0x3F800000},
		{0xBF80, 0xBF800000},
		{0x7F80, 0x7F800000},
		{0xFF80, 0xFF800000},
		{0x7FC0, 0x7FC00000},
	}
	for _, c := range cases {
		got := math.Float32bits(BFloat16ToFloat32(c.bits))
		if got != c.want {
			t.Errorf("BFloat16ToFloat32(0x%04X) bits=0x%08X want 0x%08X",
				c.bits, got, c.want)
		}
	}
}

// TestBFloat16Roundtrip_IdentityOnGrid sweeps every finite bf16 pattern,
// widens to f32, narrows back, and asserts bit equality. Confirms RTNE is
// the identity on grid points. Exponent 0xFF is skipped because NaN patterns
// canonicalize to 0x7FC0 (not a roundtrip) and Inf is checked explicitly in
// TestFloat32ToBFloat16_RTNE.
func TestBFloat16Roundtrip_IdentityOnGrid(t *testing.T) {
	for hi := uint32(0); hi < 0x10000; hi++ {
		if (hi & 0x7F80) == 0x7F80 {
			continue
		}
		f := BFloat16ToFloat32(uint16(hi))
		got := Float32ToBFloat16(f)
		if got != uint16(hi) {
			t.Fatalf("grid roundtrip 0x%04X -> f32=0x%08X -> 0x%04X",
				hi, math.Float32bits(f), got)
		}
	}
}
