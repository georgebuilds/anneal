package uop

import (
	"math"
	"testing"
)

func TestMulInt64Checked(t *testing.T) {
	cases := []struct {
		name   string
		a, b   int64
		wantP  int64
		wantOK bool
	}{
		{"zero left", 0, math.MaxInt64, 0, true},
		{"zero right", math.MaxInt64, 0, 0, true},
		{"both zero", 0, 0, 0, true},
		{"one times one", 1, 1, 1, true},
		{"small positive", 12345, 67890, 12345 * 67890, true},
		{"small negative", -12345, 67890, -12345 * 67890, true},
		{"both negative", -12345, -67890, 12345 * 67890, true},
		{"maxint times one", math.MaxInt64, 1, math.MaxInt64, true},
		{"minint times one", math.MinInt64, 1, math.MinInt64, true},
		{"minint times neg one overflow", math.MinInt64, -1, 0, false},
		{"neg one times minint overflow", -1, math.MinInt64, 0, false},
		{"maxint times two overflow", math.MaxInt64, 2, 0, false},
		{"two times maxint overflow", 2, math.MaxInt64, 0, false},
		{"large positive overflow", 1 << 40, 1 << 30, 0, false},
		{"large positive no overflow", 1 << 30, 1 << 30, 1 << 60, true},
		{"large negative overflow", -(1 << 40), 1 << 30, 0, false},
		{"sqrt-maxint squared", 3037000499, 3037000499, 3037000499 * 3037000499, true},
		{"just over sqrt-maxint squared", 3037000500, 3037000500, 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := MulInt64Checked(c.a, c.b)
			if ok != c.wantOK {
				t.Fatalf("MulInt64Checked(%d, %d) ok=%v want=%v", c.a, c.b, ok, c.wantOK)
			}
			if ok && got != c.wantP {
				t.Fatalf("MulInt64Checked(%d, %d) = %d want %d", c.a, c.b, got, c.wantP)
			}
		})
	}
}
