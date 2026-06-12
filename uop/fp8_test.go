package uop

import (
	"math"
	"sort"
	"testing"
)

// fp8Format bundles the per-format helpers so the exhaustive tests run
// identically over e4m3fn and e5m2.
type fp8Format struct {
	name      string
	encode    func(float32) uint8
	decode    func(uint8) float32
	dtype     *DType
	nanCode   uint8
	isNaNCode func(uint8) bool
	maxFinite float32
	saturates bool // e4m3fn saturates finite overflow; e5m2 rounds to Inf
}

func fp8Formats() []fp8Format {
	return []fp8Format{
		{
			name:    "e4m3fn",
			encode:  Float32ToFP8E4M3,
			decode:  FP8E4M3ToFloat32,
			dtype:   Dtypes.FP8E4M3,
			nanCode: 0x7F,
			isNaNCode: func(b uint8) bool {
				return b&0x7F == 0x7F // S.1111.111
			},
			maxFinite: 448,
			saturates: true,
		},
		{
			name:    "e5m2",
			encode:  Float32ToFP8E5M2,
			decode:  FP8E5M2ToFloat32,
			dtype:   Dtypes.FP8E5M2,
			nanCode: 0x7E,
			isNaNCode: func(b uint8) bool {
				return b&0x7C == 0x7C && b&0x03 != 0 // S.11111.MM with MM != 0
			},
			maxFinite: 57344,
			saturates: false,
		},
	}
}

// TestFP8RoundTripExhaustive decodes every one of the 256 bit patterns per
// format and re-encodes the result. Finite codes must round-trip to the
// identical code (decode is exact in f32, and an exact grid point must
// encode to itself); NaN codes canonicalize; Inf codes (e5m2 only)
// round-trip exactly.
func TestFP8RoundTripExhaustive(t *testing.T) {
	for _, f := range fp8Formats() {
		t.Run(f.name, func(t *testing.T) {
			for code := 0; code < 256; code++ {
				b := uint8(code)
				v := f.decode(b)
				got := f.encode(v)
				switch {
				case f.isNaNCode(b):
					if got != f.nanCode {
						t.Errorf("code 0x%02X (NaN) re-encoded to 0x%02X, want canonical 0x%02X", b, got, f.nanCode)
					}
				case b == 0x80: // -0 round-trips with sign
					if got != 0x80 {
						t.Errorf("-0 (0x80) re-encoded to 0x%02X", got)
					}
				default:
					if got != b {
						t.Errorf("code 0x%02X (%g) re-encoded to 0x%02X (%g)", b, v, got, f.decode(got))
					}
				}
			}
		})
	}
}

// fp8FiniteValues returns the sorted, deduplicated finite values of a format
// (positive and negative), used as the reference grid for RTNE checks.
func fp8FiniteValues(f fp8Format) []float64 {
	seen := map[float64]bool{}
	var vals []float64
	for code := 0; code < 256; code++ {
		v := float64(f.decode(uint8(code)))
		if math.IsNaN(v) || math.IsInf(v, 0) || seen[v] {
			continue
		}
		seen[v] = true
		vals = append(vals, v)
	}
	sort.Float64s(vals)
	return vals
}

// fp8NearestRTNE returns the format's correctly-rounded encoding of x by
// brute-force search over the value grid: nearest finite value, ties to the
// even mantissa, overflow per format (saturate for e4m3fn, Inf for e5m2).
// This is the independent oracle the bit-twiddling encoder is checked against.
func fp8NearestRTNE(f fp8Format, x float64, grid []float64) float32 {
	maxF := float64(f.maxFinite)
	// Overflow: beyond the midpoint between maxFinite and the next would-be
	// grid point (maxFinite * (1 + step/2) where step is one mantissa ulp).
	manSteps := 8.0 // e4m3: 2^3
	if !f.saturates {
		manSteps = 4.0 // e5m2: 2^2
	}
	overflowAt := maxF * (1 + 1/(2*manSteps)) // midpoint to the next binade step
	if math.Abs(x) >= overflowAt {
		if f.saturates {
			return float32(math.Copysign(maxF, x))
		}
		return float32(math.Copysign(math.Inf(1), x))
	}
	// Nearest grid value; on an exact tie pick the one whose mantissa is even.
	i := sort.SearchFloat64s(grid, x)
	cand := []float64{}
	if i < len(grid) {
		cand = append(cand, grid[i])
	}
	if i > 0 {
		cand = append(cand, grid[i-1])
	}
	best := cand[0]
	for _, c := range cand[1:] {
		dc, db := math.Abs(c-x), math.Abs(best-x)
		switch {
		case dc < db:
			best = c
		case dc == db && c != best:
			// Tie: even mantissa wins. The even-mantissa member of an adjacent
			// pair is the one whose encoding has LSB 0.
			if f.encodeLSB(c) == 0 {
				best = c
			}
		}
	}
	if best == 0 {
		// The grid dedups ±0 into one entry; IEEE underflow preserves the
		// input's sign, so the oracle must too.
		return float32(math.Copysign(0, x))
	}
	return float32(best)
}

// encodeLSB returns the mantissa LSB of the format's encoding of an exact
// grid value v (used only for tie resolution in the reference oracle; the
// encode of an exact grid point is exact by TestFP8RoundTripExhaustive).
func (f fp8Format) encodeLSB(v float64) uint8 {
	return f.encode(float32(v)) & 1
}

// TestFP8EncodeRTNEReference sweeps each format's full grid plus the exact
// midpoints between adjacent grid values and near-midpoint perturbations,
// checking the bit-twiddling encoder against the brute-force RTNE oracle.
func TestFP8EncodeRTNEReference(t *testing.T) {
	for _, f := range fp8Formats() {
		t.Run(f.name, func(t *testing.T) {
			grid := fp8FiniteValues(f)
			// The encoder consumes float32, so perturbations must be one
			// float32 ulp, not one float64 ulp — a float64-only nudge
			// vanishes under the f32 rounding of the input and the encoder
			// would (correctly) still see an exact tie.
			check := func(x float32) {
				t.Helper()
				want := fp8NearestRTNE(f, float64(x), grid)
				got := f.decode(f.encode(x))
				// Compare as bits so ±0 are distinguished.
				if math.Float32bits(want) != math.Float32bits(got) {
					t.Errorf("encode(%g): got %g, want %g", x, got, want)
				}
			}
			for i, v := range grid {
				check(float32(v))
				if i+1 < len(grid) {
					// Midpoints between adjacent fp8 grid values are dyadic
					// rationals exactly representable in float32, so this is
					// a true RTNE tie.
					mid := float32((v + grid[i+1]) / 2)
					check(mid)
					check(math.Nextafter32(mid, float32(v)))
					check(math.Nextafter32(mid, float32(grid[i+1])))
				}
			}
		})
	}
}

// TestFP8Specials pins the special-value behaviour: NaN canonicalization,
// Inf handling per format, signed zero, saturation vs round-to-Inf, and the
// subnormal underflow boundary (half the smallest subnormal ties to zero).
func TestFP8Specials(t *testing.T) {
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	nan := float32(math.NaN())

	t.Run("e4m3fn", func(t *testing.T) {
		cases := []struct {
			name string
			in   float32
			want uint8
		}{
			{"NaN canonicalizes", nan, 0x7F},
			{"+Inf saturates to 448 (satfinite)", posInf, 0x7E},
			{"-Inf saturates to -448 (satfinite)", negInf, 0xFE},
			{"+0", 0, 0x00},
			{"-0 preserves sign", math.Float32frombits(0x80000000), 0x80},
			{"max finite 448", 448, 0x7E},
			{"finite overflow saturates", 1e10, 0x7E},
			{"negative overflow saturates", -1e10, 0xFE},
			{"just above 448 rounds back (RTNE)", 460, 0x7E},
			{"min subnormal 2^-9", float32(math.Ldexp(1, -9)), 0x01},
			{"half min subnormal ties to zero", float32(math.Ldexp(1, -10)), 0x00},
			{"under half min subnormal flushes", float32(math.Ldexp(1, -11)), 0x00},
			{"just above half rounds to min subnormal", float32(math.Ldexp(1.1, -10)), 0x01},
		}
		for _, c := range cases {
			if got := Float32ToFP8E4M3(c.in); got != c.want {
				t.Errorf("%s: Float32ToFP8E4M3(%g) = 0x%02X, want 0x%02X", c.name, c.in, got, c.want)
			}
		}
	})

	t.Run("e5m2", func(t *testing.T) {
		cases := []struct {
			name string
			in   float32
			want uint8
		}{
			{"NaN canonicalizes", nan, 0x7E},
			{"+Inf passes through", posInf, 0x7C},
			{"-Inf passes through", negInf, 0xFC},
			{"+0", 0, 0x00},
			{"-0 preserves sign", math.Float32frombits(0x80000000), 0x80},
			{"max finite 57344", 57344, 0x7B},
			{"finite overflow rounds to +Inf", 1e10, 0x7C},
			{"negative overflow rounds to -Inf", -1e10, 0xFC},
			{"min subnormal 2^-16", float32(math.Ldexp(1, -16)), 0x01},
			{"half min subnormal ties to zero", float32(math.Ldexp(1, -17)), 0x00},
		}
		for _, c := range cases {
			if got := Float32ToFP8E5M2(c.in); got != c.want {
				t.Errorf("%s: Float32ToFP8E5M2(%g) = 0x%02X, want 0x%02X", c.name, c.in, got, c.want)
			}
		}
	})
}

// TestFP8Quantize verifies the DType.Quantize dispatch reaches the fp8
// round-trips and leaves wider dtypes untouched.
func TestFP8Quantize(t *testing.T) {
	// 3.7 is not on either fp8 grid: e4m3fn steps by 0.25 in [2,4)
	// (3.5, 3.75); e5m2 steps by 0.5 (3.5, 4.0) and 3.7 is nearer 3.5.
	if got := Dtypes.FP8E4M3.Quantize(3.7); got != 3.75 {
		t.Errorf("FP8E4M3.Quantize(3.7) = %g, want 3.75", got)
	}
	if got := Dtypes.FP8E5M2.Quantize(3.7); got != 3.5 {
		t.Errorf("FP8E5M2.Quantize(3.7) = %g, want 3.5", got)
	}
	if got := Dtypes.Float32.Quantize(3.7); got != 3.7 {
		t.Errorf("Float32.Quantize(3.7) = %g, want 3.7 unchanged", got)
	}
}
