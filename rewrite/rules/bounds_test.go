package rules_test

import (
	"testing"

	"github.com/georgebuilds/anneal/rewrite/rules"
	"github.com/georgebuilds/anneal/uop"
)

// ── local helpers ─────────────────────────────────────────────────────────────

// Convention note — dv vs Arena.DefineVar:
//
//   - dv (below) stores raw lo/hi src Consts and yields a *half-open* [lo, hi)
//     interval; BoundsOf returns {lo, hi-1}. Suitable for unit-testing
//     BoundsOf's interval arithmetic primitive in isolation without going
//     through the public constructor.
//   - Arena.DefineVar(name, min, max) stores src[1] as max+1 internally and
//     presents the user-facing *inclusive* interval [min, max]; BoundsOf
//     unwraps the +1. Mirrors tinygrad's DefineVar semantics — use this path
//     when verifying tinygrad-equivalence at the public-API surface.
//
// Both feed identical math into BoundsOf's ALU cases; only the leaf intervals
// they expose differ. See TestBoundsOfFiveOps_Slice6 (dv path) and
// TestBoundsOfFiveOps_Slice6_DefineVar (public API path) for paired oracles.

// dv builds a two-src DefineVar [lo, hi).  BoundsOf returns {lo, hi-1}.
func dv(a *uop.Arena, lo, hi int64) uop.UOp {
	return a.New(uop.OpDefineVar, uop.Dtypes.Int32, []uop.UOp{ci(a, lo), ci(a, hi)}, "x", nil)
}

// rng builds a one-src Range [0, hi).  BoundsOf returns {0, hi-1}.
func rng(a *uop.Arena, hi int64) uop.UOp {
	return a.New(uop.OpRange, uop.Dtypes.Int32, []uop.UOp{ci(a, hi)}, nil, nil)
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
		{"range [0,8)", func(a *uop.Arena) uop.UOp { return rng(a, 8) }, true, 0, 7},
		{"range [0,1)", func(a *uop.Arena) uop.UOp { return rng(a, 1) }, true, 0, 0},
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

// ── TestBoundsOfMulOverflowGivesUp ────────────────────────────────────────────
//
// QA-5 regression: a corner-product overflow inside OpMul bounds analysis
// must return Bounds{Valid:false} rather than a silently-wrong interval.
// Two paired DefineVars whose max values multiply past int64 exercise the
// guard.
func TestBoundsOfMulOverflowGivesUp(t *testing.T) {
	a := arena()
	hi := int64(1) << 40
	got := rules.BoundsOf(mul(a, dv(a, 0, hi), dv(a, 0, hi)))
	if got.Valid {
		t.Fatalf("expected Valid=false on overflowing mul, got %+v", got)
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

		// [-3,3] mod 4: {0,3}. Regression guard for the floor-vs-truncating-div
		// period check: floorDiv(-3,4)=-1 ≠ floorDiv(3,4)=0 → wrapping → {0,3}.
		// Truncating div would produce (-3)/4=0, 3/4=0 → "same period" → {1,3}.
		{"zero-crossing period boundary", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, -3, 4), ci(a, 4))
		}, true, 0, 3, ""},

		// [-1,1] mod 2: {0,1}. Minimal regression guard for the same floorDiv check.
		// floorDiv(-1,2)=-1 ≠ floorDiv(1,2)=0 → wrapping → {0,1}.
		// Truncating div would give (-1)/2=0, 1/2=0 → "same period" → {1,1}.
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
	v := dv(a, 1, 1025)           // [1,1024]
	m := mul(a, v, ci(a, 4))      // [4,4096]
	result := add(a, m, ci(a, 3)) // [7,4099]
	checkBounds(t, rules.BoundsOf(result), true, 7, 4099)
}

// ── TestBoundsOfFiveOps_Slice6 ────────────────────────────────────────────────
//
// Option B Slice 6 — vmax arithmetic oracle. Confirms BoundsOf's
// implementations of Add/Sub/Mul/IDiv/Mod match tinygrad master's _min_max
// semantics on the five spec-named ALU ops. Anneal's results are the values
// of record; if a tinygrad-compatible workload ever diverges, this is the
// table to update.
func TestBoundsOfFiveOps_Slice6(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		min   int64
		max   int64
	}{
		// Add(a, b): {a.min+b.min, a.max+b.max}.
		{"Add bounded both", func(a *uop.Arena) uop.UOp {
			x := dv(a, 0, 6)  // [0,5]
			y := dv(a, 2, 11) // [2,10]
			return add(a, x, y)
		}, 2, 15},
		// Sub(a, b): {a.min-b.max, a.max-b.min}.
		{"Sub bounded both", func(a *uop.Arena) uop.UOp {
			x := dv(a, 0, 11) // [0,10]
			y := dv(a, 2, 6)  // [2,5]
			return sub(a, x, y)
		}, -5, 8},
		// Mul(a, b) non-negative range matrix: 4-corner gives {min*min, max*max}.
		{"Mul [0,5]*[0,4]", func(a *uop.Arena) uop.UOp {
			return mul(a, dv(a, 0, 6), dv(a, 0, 5))
		}, 0, 20},
		{"Mul [1,3]*[2,4]", func(a *uop.Arena) uop.UOp {
			return mul(a, dv(a, 1, 4), dv(a, 2, 5))
		}, 2, 12},
		// Symbolic × concrete (the common scaling case).
		{"Mul var*Const(4)", func(a *uop.Arena) uop.UOp {
			return mul(a, dv(a, 0, 100), ci(a, 4))
		}, 0, 396},
		// IDiv(a, Const(c)): floor-div bounds on sign-definite divisor.
		{"IDiv [10,21]/Const(5)", func(a *uop.Arena) uop.UOp {
			return idiv(a, dv(a, 10, 22), ci(a, 5))
		}, 2, 4},
		// Mod(a, Const(c)): wrapping range → [0, c-1]; same-period → exact.
		{"Mod [0,17]%Const(8) wrap", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, 0, 18), ci(a, 8))
		}, 0, 7},
		{"Mod [2,5]%Const(10) same period", func(a *uop.Arena) uop.UOp {
			return mod(a, dv(a, 2, 6), ci(a, 10))
		}, 2, 5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			got := rules.BoundsOf(tc.build(a))
			checkBounds(t, got, true, tc.min, tc.max)
		})
	}
}

// ── TestBoundsOfFiveOps_Slice6_DefineVar ──────────────────────────────────────
//
// Companion oracle to TestBoundsOfFiveOps_Slice6 that exercises the same ALU
// shapes through the public Arena.DefineVar(name, min, max) API instead of the
// half-open dv test helper. DefineVar is inclusive (src[1]=max+1 internally;
// BoundsOf unwraps), so the same numeric inputs yield slightly wider intervals
// — e.g. Mul(DefineVar(0,100), Const(4)) is {0,400} here vs {0,396} under dv.
//
// Expected values match tinygrad master's _min_max under inclusive semantics
// (reference: raw.githubusercontent.com/tinygrad/tinygrad/9d9151a2/tinygrad/
// uop/ops.py). Divergence on any case would indicate a real bug in BoundsOf
// or Arena.DefineVar's encoding.
func TestBoundsOfFiveOps_Slice6_DefineVar(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		min   int64
		max   int64
	}{
		// Add(a, b): {a.min+b.min, a.max+b.max} — [0,6]+[2,11] = [2,17].
		{"Add bounded both", func(a *uop.Arena) uop.UOp {
			return add(a, a.DefineVar("x", 0, 6), a.DefineVar("y", 2, 11))
		}, 2, 17},
		// Sub(a, b): {a.min-b.max, a.max-b.min} — [0,11]-[2,6] = [-6,9].
		{"Sub bounded both", func(a *uop.Arena) uop.UOp {
			return sub(a, a.DefineVar("x", 0, 11), a.DefineVar("y", 2, 6))
		}, -6, 9},
		// Mul non-negative ranges: 4-corner reduces to {min*min, max*max}.
		{"Mul [0,6]*[0,5]", func(a *uop.Arena) uop.UOp {
			return mul(a, a.DefineVar("x", 0, 6), a.DefineVar("y", 0, 5))
		}, 0, 30},
		{"Mul [1,4]*[2,5]", func(a *uop.Arena) uop.UOp {
			return mul(a, a.DefineVar("x", 1, 4), a.DefineVar("y", 2, 5))
		}, 2, 20},
		// Symbolic × concrete (the prompt's reference shift case).
		{"Mul var*Const(4)", func(a *uop.Arena) uop.UOp {
			return mul(a, a.DefineVar("x", 0, 100), ci(a, 4))
		}, 0, 400},
		// IDiv by positive const: floor-div corner bounds — [10,22]/5 = [2,4].
		{"IDiv [10,22]/Const(5)", func(a *uop.Arena) uop.UOp {
			return idiv(a, a.DefineVar("x", 10, 22), ci(a, 5))
		}, 2, 4},
		// Mod wrapping: floorDiv(0,8)=0 ≠ floorDiv(18,8)=2 → [0, c-1].
		{"Mod [0,18]%Const(8) wrap", func(a *uop.Arena) uop.UOp {
			return mod(a, a.DefineVar("x", 0, 18), ci(a, 8))
		}, 0, 7},
		// Mod same period: floorDiv(2,10)=floorDiv(6,10)=0 → [2,6].
		{"Mod [2,6]%Const(10) same period", func(a *uop.Arena) uop.UOp {
			return mod(a, a.DefineVar("x", 2, 6), ci(a, 10))
		}, 2, 6},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			got := rules.BoundsOf(tc.build(a))
			checkBounds(t, got, true, tc.min, tc.max)
		})
	}
}

// ── TestIndexDtypeForBound ────────────────────────────────────────────────────
//
// Option B Slice 6 — INV-B (vmax-driven int dtype upcast). IndexDtypeForBound
// is the single source of truth for "is i32 enough for this index expression?"
// WebGPU downgrades the Int64 result to i32 at emission time and acknowledges
// via a WGSL comment; a future backend would honor Int64 unchanged.
func TestIndexDtypeForBound(t *testing.T) {
	tests := []struct {
		name  string
		build func(*uop.Arena) uop.UOp
		want  *uop.DType
	}{
		{"small constant fits i32", func(a *uop.Arena) uop.UOp {
			return ci(a, 1024)
		}, uop.Dtypes.Int32},
		{"DefineVar small bounds fits i32", func(a *uop.Arena) uop.UOp {
			return dv(a, 0, 4096) // BoundsOf = {0, 4095}
		}, uop.Dtypes.Int32},
		{"DefineVar near int32 max fits i32", func(a *uop.Arena) uop.UOp {
			return dv(a, 0, 2_000_000_000)
		}, uop.Dtypes.Int32},
		{"Mul overflows int32 — needs i64", func(a *uop.Arena) uop.UOp {
			// var in [0, 1_000_000]; Mul by 4000 → vmax = 4_000_000_000 > MaxInt32 (~2.1e9).
			v := dv(a, 0, 1_000_001) // BoundsOf = {0, 1_000_000}
			return mul(a, v, ci(a, 4000))
		}, uop.Dtypes.Int64},
		{"unprovable bound conservatively i64", func(a *uop.Arena) uop.UOp {
			// OpCast is not modeled by BoundsOf (returns Valid=false) → i64 fallback.
			return a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{ci(a, 5)}, nil, nil)
		}, uop.Dtypes.Int64},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := arena()
			got := rules.IndexDtypeForBound(tc.build(a))
			if got != tc.want {
				t.Errorf("IndexDtypeForBound = %v, want %v", got, tc.want)
			}
		})
	}
}
