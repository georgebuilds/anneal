package rewrite_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/uop"
)

// ── NewPatternMatcher panic paths ─────────────────────────────────────────────

// TestNewPatternMatcherTopLevelNoOpPanics covers the panic when a top-level
// (non-Any) pattern omits its op constraint.
func TestNewPatternMatcherTopLevelNoOpPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for op-less top-level UPat")
		}
	}()
	rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.WildPat(), // no op constraint
			Fn:  func(map[string]uop.UOp, any) (uop.UOp, bool) { return uop.UOp{}, false },
		},
	})
}

// TestNewPatternMatcherAnyAltNoOpPanics covers the panic when an AnyPat
// alternative omits its op constraint.
func TestNewPatternMatcherAnyAltNoOpPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for op-less AnyPat alternative")
		}
	}()
	rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.AnyPat(rewrite.Pat(uop.OpAdd), rewrite.WildPat()),
			Fn:  func(map[string]uop.UOp, any) (uop.UOp, bool) { return uop.UOp{}, false },
		},
	})
}

// ── WithAnyLen (prefix matching) ──────────────────────────────────────────────

// TestWithAnyLenPrefixMatch verifies WithAnyLen relaxes the strict length check
// so a 2-source pattern matches a 3-source node (prefix match).
func TestWithAnyLenPrefixMatch(t *testing.T) {
	a := newArena()
	x := constN(a, 1)
	y := constN(a, 2)
	z := constN(a, 3)
	// A 3-source node (use OpWhere-style: any op accepting 3 srcs). OpMulAcc takes 3.
	node := a.New(uop.OpMulAcc, uop.Dtypes.Int32, []uop.UOp{x, y, z}, nil, nil)

	// Pattern wants only the first two srcs; WithAnyLen permits extra trailing srcs.
	fired := false
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpMulAcc).WithSrc(
				rewrite.Pat(uop.OpConst).WithName("a"),
				rewrite.Pat(uop.OpConst).WithName("b"),
			).WithAnyLen(),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				fired = true
				return c["a"], true
			},
		},
	})
	res := rewrite.GraphRewrite(node, pm)
	if !fired {
		t.Fatal("WithAnyLen prefix pattern did not fire on a longer node")
	}
	if res != x {
		t.Errorf("rewrite result = %d, want %d (x)", res.Index(), x.Index())
	}
}

// TestWithoutAnyLenStrictRejectsLonger confirms the default (strict) behavior
// rejects a longer node, isolating WithAnyLen's effect.
func TestWithoutAnyLenStrictRejectsLonger(t *testing.T) {
	a := newArena()
	x := constN(a, 1)
	y := constN(a, 2)
	z := constN(a, 3)
	node := a.New(uop.OpMulAcc, uop.Dtypes.Int32, []uop.UOp{x, y, z}, nil, nil)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpMulAcc).WithSrc(
				rewrite.Pat(uop.OpConst).WithName("a"),
				rewrite.Pat(uop.OpConst).WithName("b"),
			), // strict: requires exactly 2 srcs
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return c["a"], true
			},
		},
	})
	res := rewrite.GraphRewrite(node, pm)
	if res != node {
		t.Errorf("strict pattern wrongly fired on 3-src node: got %d", res.Index())
	}
}

// ── equalArgs scalar branches via WithArg ─────────────────────────────────────

// TestWithArgFloat64 covers equalArgs' float64 branch (bit comparison).
func TestWithArgFloat64(t *testing.T) {
	a := newArena()
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3.5), nil)
	matched := matchArgConst(t, a, c, uop.OpConst, float64(3.5))
	if !matched {
		t.Error("float64 arg constraint did not match equal value")
	}
	if matchArgConst(t, a, c, uop.OpConst, float64(2.0)) {
		t.Error("float64 arg constraint matched a different value")
	}
}

// TestWithArgFloat64NaN covers the NaN-bits-equal path of equalArgs.
func TestWithArgFloat64NaN(t *testing.T) {
	a := newArena()
	nan := math.NaN()
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, nan, nil)
	if !matchArgConst(t, a, c, uop.OpConst, nan) {
		t.Error("NaN arg constraint did not match NaN with identical bits")
	}
}

// TestWithArgBool covers equalArgs' bool branch.
func TestWithArgBool(t *testing.T) {
	a := newArena()
	c := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	if !matchArgConst(t, a, c, uop.OpConst, true) {
		t.Error("bool arg constraint did not match")
	}
	if matchArgConst(t, a, c, uop.OpConst, false) {
		t.Error("bool arg constraint matched opposite value")
	}
}

// TestWithArgString covers equalArgs' string branch.
func TestWithArgString(t *testing.T) {
	a := newArena()
	c := a.New(uop.OpDefineVar, uop.Dtypes.Index, []uop.UOp{
		a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil),
		a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(10), nil),
	}, uop.VarArg{Name: "n"}, nil)
	// VarArg is not a string; equalArgs treats unknown types as unequal (default
	// branch). Confirm a string constraint does NOT match a VarArg node.
	if matchArgConst(t, a, c, uop.OpDefineVar, "n") {
		t.Error("string constraint should not match a VarArg-typed arg")
	}
}

// TestWithArgNilMismatch covers equalArgs' nil-vs-nonnil branch: a pattern with
// WithArg(nil) must not match a node carrying a non-nil arg.
func TestWithArgNilMismatch(t *testing.T) {
	a := newArena()
	c := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(5), nil)
	if matchArgConst(t, a, c, uop.OpConst, nil) {
		t.Error("WithArg(nil) wrongly matched a node with non-nil arg")
	}
}

// TestWithArgTypeMismatch covers equalArgs' type-mismatch within a scalar branch
// (e.g. int64 constraint vs float64 arg).
func TestWithArgTypeMismatch(t *testing.T) {
	a := newArena()
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(5), nil)
	if matchArgConst(t, a, c, uop.OpConst, int64(5)) {
		t.Error("int64 constraint wrongly matched a float64 arg")
	}
}

// matchArgConst builds a one-rule matcher with an op+arg constraint and reports
// whether it fires on node n. The handler is a no-op identity that signals match.
func matchArgConst(t *testing.T, a *uop.Arena, n uop.UOp, op uop.Op, arg any) bool {
	t.Helper()
	fired := false
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(op).WithArg(arg),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				fired = true
				return n, false // don't actually rewrite; just record the match
			},
		},
	})
	rewrite.GraphRewrite(n, pm)
	return fired
}

// ── permutations: all-identical fast path ─────────────────────────────────────

// TestCommSrcAllIdenticalSinglePass exercises permutations' all-same shortcut: a
// commutative pattern whose two src patterns are the SAME *UPat pointer yields a
// single variant, not a factorial blowup, yet still matches.
func TestCommSrcAllIdentical(t *testing.T) {
	a := newArena()
	x := constN(a, 4)
	sum := addN(a, x, x) // Add(4, 4)
	same := rewrite.Pat(uop.OpConst).WithName("c")
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithCommSrc(same, same),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return c["c"], true
			},
		},
	})
	res := rewrite.GraphRewrite(sum, pm)
	if res != x {
		t.Errorf("identical-src comm pattern: got %d, want %d", res.Index(), x.Index())
	}
}

// ── bpm cycle guard ───────────────────────────────────────────────────────────

// TestBPMCycleGuard exercises the seen-set cycle guard in the bpm fixpoint loop:
// a bpm that maps Const(0)→Const(1) and Const(1)→Const(0) would loop forever
// without the guard. The loop must terminate and produce a finite result.
func TestBPMCycleGuard(t *testing.T) {
	a := newArena()
	zero := constN(a, 0)

	// bpm flips 0<->1 endlessly.
	bpm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return constN(a, 1), true
			},
		},
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(1)),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return constN(a, 0), true
			},
		},
	})

	// Must terminate. Result is whichever value the guard halts on (0 or 1).
	res := rewrite.GraphRewrite(zero, emptyPM(), rewrite.WithBPM(bpm))
	got := res.Arg().(int64)
	if got != 0 && got != 1 {
		t.Errorf("cycle-guard result arg = %d, want 0 or 1", got)
	}
}
