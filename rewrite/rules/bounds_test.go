package rules_test

import (
	"testing"

	"github.com/georgebuilds/anneal/rewrite/rules"
	"github.com/georgebuilds/anneal/uop"
)

// ── local helpers ─────────────────────────────────────────────────────────────

// dv builds a two-src DefineVar [lo, hi).  BoundsOf returns {lo, hi-1}.
func dv(a *uop.Arena, lo, hi int64) uop.UOp {
	return a.New(uop.OpDefineVar, uop.Dtypes.Int32, []uop.UOp{ci(a, lo), ci(a, hi)}, "x", nil)
}

// rng builds a two-src Range [lo, hi).  BoundsOf returns {lo, hi-1}.
func rng(a *uop.Arena, lo, hi int64) uop.UOp {
	return a.New(uop.OpRange, uop.Dtypes.Int32, []uop.UOp{ci(a, lo), ci(a, hi)}, nil, nil)
}

// cmpeq builds a CmpEq bool node.
func cmpeq(a *uop.Arena, x, y uop.UOp) uop.UOp {
	return bop(a, uop.OpCmpEq, uop.Dtypes.Bool, x, y)
}

func checkBounds(t *testing.T, got rules.Bounds, valid bool, min, max int64) {
	t.Helper()
	if got.Valid != valid {
		t.Errorf("Valid: got %v, want %v (bounds=%+v)", got.Valid, valid, got)
		return
	}
	if valid && (got.Min != min || got.Max != max) {
		t.Errorf("got [%d,%d], want [%d,%d]", got.Min, got.Max, min, max)
	}
}

// ── TestBoundsOfLeaves ────────────────────────────────────────────────────────

func TestBoundsOfLeaves(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		valid bool
		min   int64
		max   int64
	}{
		{"const int positive", func(a *uop.Arena) uop.UOp { return ci(a, 42) }, true, 42, 42},
		{"const int negative", func(a *uop.Arena) uop.UOp { return ci(a, -7) }, true, -7, -7},
		{"const int zero", func(a *uop.Arena) uop.UOp { return ci(a, 0) }, true, 0, 0},
		{"const bool true", func(a *uop.Arena) uop.UOp { return cb(a, true) }, true, 1, 1},
		{"const bool false", func(a *uop.Arena) uop.UOp { return cb(a, false) }, true, 0, 0},
		{"const float invalid", func(a *uop.Arena) uop.UOp { return cf(a, 3.14) }, false, 0, 0},
		{"define_var [1,10)", func(a *uop.Arena) uop.UOp { return dv(a, 1, 10) }, true, 1, 9},
		{"define_var [0,1)", func(a *uop.Arena) uop.UOp { return dv(a, 0, 1) }, true, 0, 0},
		{"define_var negative range [-5,3)", func(a *uop.Arena) uop.UOp { return dv(a, -5, 3) }, true, -5, 2},
		{"range [0,8)", func(a *uop.Arena) uop.UOp { return rng(a, 0, 8) }, true, 0, 7},
		{"range [3,7)", func(a *uop.Arena) uop.UOp { return rng(a, 3, 7) }, true, 3, 6},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			checkBounds(t, rules.BoundsOf(tc.build(a)), tc.valid, tc.min, tc.max)
		})
	}
}

// ── TestBoundsOfSub ───────────────────────────────────────────────────────────

func TestBoundsOfSub(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		min   int64
		max   int64
	}{
		// [5,10] - [1,3] = [2,9]
		{"positive", func(a *uop.Arena) uop.UOp { return sub(a, dv(a, 5, 11), dv(a, 1, 4)) }, 2, 9},
		// [2,4] - [3,7] = [-5,1]
		{"yields negative", func(a *uop.Arena) uop.UOp { return sub(a, dv(a, 2, 5), dv(a, 3, 8)) }, -5, 1},
		// [0,0] - [0,0] = [0,0]
		{"zero minus zero", func(a *uop.Arena) uop.UOp { return sub(a, ci(a, 0), ci(a, 0)) }, 0, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			checkBounds(t, rules.BoundsOf(tc.build(a)), true, tc.min, tc.max)
		})
	}
}

// ── TestBoundsOfMulSignRegimes ────────────────────────────────────────────────

func TestBoundsOfMulSignRegimes(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		min   int64
		max   int64
	}{
		// [2,5] * [3,7] = [6,35]
		{"pos * pos", func(a *uop.Arena) uop.UOp { return mul(a, dv(a, 2, 6), dv(a, 3, 8)) }, 6, 35},
		// [-5,-2] * [-4,-2] = [4,20]
		{"neg * neg", func(a *uop.Arena) uop.UOp { return mul(a, dv(a, -5, -1), dv(a, -4, -1)) }, 4, 20},
		// [-4,-2] * [2,4] = [-16,-4]
		{"neg * pos", func(a *uop.Arena) uop.UOp { return mul(a, dv(a, -4, -1), dv(a, 2, 5)) }, -16, -4},
		// [-2,3] * [1,2] = [-4,6]
		{"mixed * pos", func(a *uop.Arena) uop.UOp { return mul(a, dv(a, -2, 4), dv(a, 1, 3)) }, -4, 6},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			checkBounds(t, rules.BoundsOf(tc.build(a)), true, tc.min, tc.max)
		})
	}
}

// ── TestBoundsOfMax ───────────────────────────────────────────────────────────

func TestBoundsOfMax(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		min   int64
		max   int64
	}{
		// max([1,3], [5,8]) = [5,8]
		{"lo range wins entirely", func(a *uop.Arena) uop.UOp {
			return bop(a, uop.OpMax, uop.Dtypes.Int32, dv(a, 1, 4), dv(a, 5, 9))
		}, 5, 8},
		// max([2,6], [4,8]) = [4,8]
		{"overlapping", func(a *uop.Arena) uop.UOp {
			return bop(a, uop.OpMax, uop.Dtypes.Int32, dv(a, 2, 7), dv(a, 4, 9))
		}, 4, 8},
		// max([-3,5], [0,2]) = [0,5]
		{"one crosses zero", func(a *uop.Arena) uop.UOp {
			return bop(a, uop.OpMax, uop.Dtypes.Int32, dv(a, -3, 6), dv(a, 0, 3))
		}, 0, 5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			checkBounds(t, rules.BoundsOf(tc.build(a)), true, tc.min, tc.max)
		})
	}
}

// ── TestBoundsOfInvalidPropagation ───────────────────────────────────────────

func TestBoundsOfInvalidPropagation(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
	}{
		// OpDefineVar with no srcs → bounds unknown
		{"define_var no srcs", func(a *uop.Arena) uop.UOp {
			return a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "y", nil)
		}},
		// add(opaque_var, const) → left operand invalid → result invalid
		{"add with opaque operand", func(a *uop.Arena) uop.UOp {
			opaque := a.New(uop.OpDefineVar, uop.Dtypes.Int32, nil, "z", nil)
			return add(a, opaque, ci(a, 5))
		}},
		// Float dtype immediately returns invalid regardless of op
		{"float dtype node", func(a *uop.Arena) uop.UOp {
			return cf(a, 1.0)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			got := rules.BoundsOf(tc.build(a))
			if got.Valid {
				t.Errorf("expected invalid bounds, got %+v", got)
			}
		})
	}
}

// ── TestBoundsOfIDiv ─────────────────────────────────────────────────────────

func TestBoundsOfIDiv(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		valid bool
		min   int64
		max   int64
	}{
		// [2,5] / 2 = [1,2] (floor div)
		{"pos / pos const", func(a *uop.Arena) uop.UOp {
			return idiv(a, dv(a, 2, 6), ci(a, 2))
		}, true, 1, 2},
		// [-7,0] / 3 = [-3,0] (floor div: floorDiv(-7,3)=-3, floorDiv(0,3)=0)
		{"neg range / pos const", func(a *uop.Arena) uop.UOp {
			return idiv(a, dv(a, -7, 1), ci(a, 3))
		}, true, -3, 0},
		// [-5,5] / [-1,1] → divisor crosses zero → invalid
		{"divisor crosses zero", func(a *uop.Arena) uop.UOp {
			return idiv(a, dv(a, -5, 6), dv(a, -1, 2))
		}, false, 0, 0},
		// [6,12] / [2,3] = [2,6] (all corners: 6/2=3, 6/3=2, 12/2=6, 12/3=4)
		{"pos range / pos range", func(a *uop.Arena) uop.UOp {
			return idiv(a, dv(a, 6, 13), dv(a, 2, 4))
		}, true, 2, 6},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			checkBounds(t, rules.BoundsOf(tc.build(a)), tc.valid, tc.min, tc.max)
		})
	}
}

// ── TestBoundsOfMod ───────────────────────────────────────────────────────────

// BUG: bounds.go line 134 uses Go truncating division (s0.Min/c == s0.Max/c) instead
// of floorDiv for the "same modular period" check.  For ranges that straddle a
// period boundary while having the same truncated quotient (e.g. [-3,3] mod 4),
// the check incorrectly reports "same period" and returns a wrong lower bound:
// it returns {1,3} instead of the correct {0,3} because floorMod(0,4)=0 is
// reachable (x=0) but the truncated-div check misses the period crossing.
//
// The cases marked with "BUG" below will FAIL until line 134 is corrected to:
//
//	if floorDiv(s0.Min, c) == floorDiv(s0.Max, c) {
func TestBoundsOfMod(t *testing.T) {
	tests := []struct {
		name    string
		build   func(*uop.Arena) uop.UOp
		valid   bool
		min     int64
		max     int64
		skipMsg string // non-empty marks a known bug; keep want intact for when the fix lands
	}{
		// [2,5] mod 10 — same period (both quotients 0): {2,5}
		{"same period positive", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, 2, 6), ci(a, 10))
		}, true, 2, 5, ""},

		// [0,16] mod 8 — wraps: {0,7}
		{"wrapping positive", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, 0, 17), ci(a, 8))
		}, true, 0, 7, ""},

		// [-3,3] mod 4: correct={0,3}, buggy code returns {1,3}         ← BUG
		// Truncating: (-3)/4=0, 3/4=0 → "same period" → {floorMod(-3,4),floorMod(3,4)}={1,3}
		// Floor div:  floorDiv(-3,4)=-1 ≠ floorDiv(3,4)=0 → wrapping → {0,3}
		{"zero-crossing period boundary", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, -3, 4), ci(a, 4))
		}, true, 0, 3, ""},

		// [-1,1] mod 2: correct={0,1}, buggy code returns {1,1}          ← BUG
		// Truncating: (-1)/2=0, 1/2=0 → "same period" → {floorMod(-1,2),floorMod(1,2)}={1,1}
		// Floor div:  floorDiv(-1,2)=-1 ≠ floorDiv(1,2)=0 → wrapping → {0,1}
		{"minimal crossing case", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, -1, 2), ci(a, 2))
		}, true, 0, 1, ""},

		// non-const divisor → invalid
		{"non-const divisor", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, 0, 10), dv(a, 2, 5))
		}, false, 0, 0, ""},

		// negative divisor → invalid (s1.Min <= 0)
		{"negative divisor", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, 0, 10), ci(a, -3))
		}, false, 0, 0, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			got := rules.BoundsOf(tc.build(a))
			if tc.skipMsg != "" {
				t.Skip(tc.skipMsg)
			}
			if !tc.valid {
				if got.Valid {
					t.Errorf("expected invalid, got %+v", got)
				}
				return
			}
			if !got.Valid {
				t.Errorf("expected valid bounds [%d,%d], got invalid", tc.min, tc.max)
				return
			}
			if got.Min != tc.min || got.Max != tc.max {
				t.Errorf("got [%d,%d], want [%d,%d]", got.Min, got.Max, tc.min, tc.max)
			}
		})
	}
}

// ── TestBoundsOfCmp ───────────────────────────────────────────────────────────

func TestBoundsOfCmp(t *testing.T) {
	type tc struct {
		name  string
		build func(*uop.Arena) uop.UOp
		min   int64 // 0 or 1 (bool result)
		max   int64
	}
	runTable := func(t *testing.T, tests []tc) {
		t.Helper()
		for _, c := range tests {
			t.Run(c.name, func(t *testing.T) {
				a := arena()
				checkBounds(t, rules.BoundsOf(c.build(a)), true, c.min, c.max)
			})
		}
	}

	t.Run("CmpLt", func(t *testing.T) {
		runTable(t, []tc{
			// [0,4] < 10 → always true
			{"always true", func(a *uop.Arena) uop.UOp {
				return cmplt(a, dv(a, 0, 5), ci(a, 10))
			}, 1, 1},
			// [5,9] < 3 → always false
			{"always false", func(a *uop.Arena) uop.UOp {
				return cmplt(a, dv(a, 5, 10), ci(a, 3))
			}, 0, 0},
			// [3,7] < [5,9] → sometimes (vmin=7<5=false, vmax=3<9=true)
			{"unknown", func(a *uop.Arena) uop.UOp {
				return cmplt(a, dv(a, 3, 8), dv(a, 5, 10))
			}, 0, 1},
		})
	})

	t.Run("CmpNe", func(t *testing.T) {
		runTable(t, []tc{
			// 5 != 5 → always false
			{"always false equal consts", func(a *uop.Arena) uop.UOp {
				return cmpne(a, ci(a, 5), ci(a, 5))
			}, 0, 0},
			// [0,4] != [6,9] → always true (disjoint)
			{"always true disjoint", func(a *uop.Arena) uop.UOp {
				return cmpne(a, dv(a, 0, 5), dv(a, 6, 10))
			}, 1, 1},
			// [2,5] != [4,7] → sometimes
			{"unknown overlapping", func(a *uop.Arena) uop.UOp {
				return cmpne(a, dv(a, 2, 6), dv(a, 4, 8))
			}, 0, 1},
		})
	})

	t.Run("CmpEq", func(t *testing.T) {
		runTable(t, []tc{
			// 7 == 7 → always true
			{"always true equal consts", func(a *uop.Arena) uop.UOp {
				return cmpeq(a, ci(a, 7), ci(a, 7))
			}, 1, 1},
			// [0,4] == [6,9] → always false (disjoint)
			{"always false disjoint", func(a *uop.Arena) uop.UOp {
				return cmpeq(a, dv(a, 0, 5), dv(a, 6, 10))
			}, 0, 0},
			// [2,5] == [4,7] → sometimes
			{"unknown overlapping", func(a *uop.Arena) uop.UOp {
				return cmpeq(a, dv(a, 2, 6), dv(a, 4, 8))
			}, 0, 1},
		})
	})
}

// ── TestBoundsOfCastInvalid ───────────────────────────────────────────────────

func TestBoundsOfCastInvalid(t *testing.T) {
	// OpCast is unary (NSrc=1); BoundsOf falls through binary-op guard → invalid.
	a := arena()
	castNode := a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{ci(a, 3)}, nil, nil)
	got := rules.BoundsOf(castNode)
	if got.Valid {
		t.Errorf("BoundsOf(Cast) should be invalid, got %+v", got)
	}
}

// ── TestBoundsOfComposed ──────────────────────────────────────────────────────

func TestBoundsOfComposed(t *testing.T) {
	// add(mul(DefineVar([1,1024]), 4), 3) → [7, 4099]
	a := arena()
	v := dv(a, 1, 1025)        // [1,1024]
	m := mul(a, v, ci(a, 4))   // [4,4096]
	result := add(a, m, ci(a, 3)) // [7,4099]
	checkBounds(t, rules.BoundsOf(result), true, 7, 4099)
}
