package rewrite_test

import (
	"testing"

	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/uop"
)

// ── test helpers ──────────────────────────────────────────────────────────────

func newArena() *uop.Arena { return uop.NewArena(64) }

func constN(a *uop.Arena, n int64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Int32, nil, n, nil)
}

func addN(a *uop.Arena, x, y uop.UOp) uop.UOp {
	return a.New(uop.OpAdd, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
}

func mulN(a *uop.Arena, x, y uop.UOp) uop.UOp {
	return a.New(uop.OpMul, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
}

// identity returns a PM that never rewrites (useful as an argument placeholder).
func emptyPM() *rewrite.PatternMatcher {
	return rewrite.NewPatternMatcher(nil)
}

// ── basic single-rule rewrite: x+0 → x ───────────────────────────────────────

func TestSingleRuleRewrite(t *testing.T) {
	a := newArena()
	x := constN(a, 7)
	zero := constN(a, 0)
	sum := addN(a, x, zero) // Add(7, 0)

	// Rule: Add(x, Const(0)) → x
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return c["x"], true
			},
		},
	})

	result := rewrite.GraphRewrite(sum, pm)
	if result != x {
		t.Errorf("x+0 rewrite: got index %d, want %d", result.Index(), x.Index())
	}
}

// ── commutative match: 0+x and x+0 both rewrite to x ─────────────────────────

func TestCommutativeMatch(t *testing.T) {
	a := newArena()
	x := constN(a, 5)
	zero := constN(a, 0)

	// Rule: Add(a, b) where one is Const(0), commutative → return the non-zero
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithCommSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return c["x"], true
			},
		},
	})

	tests := []struct {
		name string
		node uop.UOp
	}{
		{"x+0", addN(a, x, zero)},
		{"0+x", addN(a, zero, x)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := rewrite.GraphRewrite(tc.node, pm)
			if result != x {
				t.Errorf("%s: got %v, want %v", tc.name, result.Index(), x.Index())
			}
		})
	}
}

// ── named-capture binding ─────────────────────────────────────────────────────

// TestNamedCaptureBinding verifies that the two captured operands are the expected nodes.
func TestNamedCaptureBinding(t *testing.T) {
	a := newArena()
	left := constN(a, 10)
	right := constN(a, 20)
	sum := addN(a, left, right)

	var gotLeft, gotRight uop.UOp
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat().WithName("l"),
				rewrite.WildPat().WithName("r"),
			),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				gotLeft, gotRight = c["l"], c["r"]
				return uop.UOp{}, false // reject to record captures without rewriting
			},
		},
	})

	rewrite.GraphRewrite(sum, pm)
	if gotLeft != left {
		t.Errorf("capture 'l' = %v, want %v", gotLeft.Index(), left.Index())
	}
	if gotRight != right {
		t.Errorf("capture 'r' = %v, want %v", gotRight.Index(), right.Index())
	}
}

// TestSameNameCapture verifies that two uses of the same capture name enforce equality.
func TestSameNameCapture(t *testing.T) {
	a := newArena()
	x := constN(a, 3)
	y := constN(a, 4)

	// Rule: Add(x, x) → x  (both srcs must be the same node)
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.WildPat().WithName("x"), // same name → must be same UOp
			),
			Fn: func(c map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return c["x"], true
			},
		},
	})

	sameArgs := addN(a, x, x)   // Add(x, x) — should fire
	diffArgs := addN(a, x, y)   // Add(x, y) — should NOT fire

	if result := rewrite.GraphRewrite(sameArgs, pm); result != x {
		t.Errorf("Add(x,x): expected x (%d), got %d", x.Index(), result.Index())
	}
	if result := rewrite.GraphRewrite(diffArgs, pm); result != diffArgs {
		t.Errorf("Add(x,y): expected no rewrite, got %d", result.Index())
	}
}

// ── multi-rule ordering: earlier rules win ────────────────────────────────────

func TestMultiRuleOrdering(t *testing.T) {
	a := newArena()
	x := constN(a, 99)
	zero := constN(a, 0)
	sum := addN(a, x, zero)

	winner := constN(a, 1)
	loser := constN(a, 2)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat(),
				rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return winner, true },
		},
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat(),
				rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return loser, true },
		},
	})

	result := rewrite.GraphRewrite(sum, pm)
	if result != winner {
		t.Errorf("first rule should win; got %d, want %d", result.Index(), winner.Index())
	}
}

// ── early-reject optimization ─────────────────────────────────────────────────

func TestEarlyReject(t *testing.T) {
	a := newArena()
	x := constN(a, 5)
	y := constN(a, 6)
	sumXY := addN(a, x, y)         // Add(Const, Const) — Const IS in src ops
	sumXX := addN(a, x, addN(a, x, x)) // Add(Const, Add) — Mul is NOT in src ops

	callCount := 0
	// This rule's earlyReject will contain OpMul because the second src is Pat(OpMul).
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat(),
				rewrite.Pat(uop.OpMul), // earlyReject[OpMul]
			),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) {
				callCount++
				return uop.UOp{}, false
			},
		},
	})

	// Neither node has a Mul child, so the rule should be early-rejected.
	rewrite.GraphRewrite(sumXY, pm)
	rewrite.GraphRewrite(sumXX, pm)
	if callCount != 0 {
		t.Errorf("handler called %d times; earlyReject should have blocked all attempts", callCount)
	}

	// Now build a node with a Mul child — handler should be reached.
	mul := mulN(a, x, y)
	sumWithMul := addN(a, x, mul) // Add(Const, Mul) — Mul IS in src ops
	rewrite.GraphRewrite(sumWithMul, pm)
	if callCount == 0 {
		t.Errorf("handler never called for Add(Const, Mul); earlyReject too aggressive")
	}
}

// ── pm chain convergence: replacement nodes are re-processed ──────────────────

// TestPMChainConvergence verifies that when pm rewrites A→B and B→C, a single
// GraphRewrite call converges to C. The replacement is pushed back onto the stack
// and re-processed, giving a chain effect without an explicit outer fixpoint loop.
func TestPMChainConvergence(t *testing.T) {
	a := newArena()
	c0 := constN(a, 0)
	c1 := constN(a, 1)
	c2 := constN(a, 2)

	// Two rules: Const(0)→Const(1), Const(1)→Const(2). Chain converges to Const(2).
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return c1, true },
		},
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(1)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return c2, true },
		},
	})

	result := rewrite.GraphRewrite(c0, pm)
	if result != c2 {
		t.Errorf("chain convergence: got %d, want %d", result.Index(), c2.Index())
	}
}

// ── bpm per-node fixpoint convergence ────────────────────────────────────────

// TestBPMFixpointConvergence verifies that bpm's per-node fixpoint loop applies
// rules repeatedly until stable, so Const(0) converges to Const(3) in one
// GraphRewrite call via the pre-order fixpoint without touching pm at all.
func TestBPMFixpointConvergence(t *testing.T) {
	a := newArena()
	c0 := constN(a, 0)
	c1 := constN(a, 1)
	c2 := constN(a, 2)
	c3 := constN(a, 3)

	bpm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return c1, true },
		},
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(1)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return c2, true },
		},
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(2)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return c3, true },
		},
	})

	result := rewrite.GraphRewrite(c0, emptyPM(), rewrite.WithBPM(bpm))
	if result != c3 {
		t.Errorf("bpm fixpoint: got Const(%v), want Const(3)", result.Arg())
	}
}

// ── deep graph: iterative driver handles arbitrary depth ─────────────────────

// TestDeepGraphNoStackOverflow builds a linear chain of 50 000 nodes and verifies
// GraphRewrite completes and returns the correct root. A naive recursive DFS would
// blow Python's default stack (depth 1000) and can exhaust large stacks in practice.
func TestDeepGraphNoStackOverflow(t *testing.T) {
	const depth = 50_000
	a := uop.NewArena(depth + 1)
	leaf := constN(a, 0)
	cur := leaf
	for i := 0; i < depth; i++ {
		// Each node is Add(prev, leaf) — a right-leaning chain depth nodes deep.
		cur = addN(a, cur, leaf)
	}
	root := cur

	result := rewrite.GraphRewrite(root, emptyPM())
	if result != root {
		t.Errorf("deep graph: result changed when no rules apply")
	}
}

// ── unchanged subtrees are shared by index ───────────────────────────────────

// TestSubtreeSharingAfterRewrite builds a DAG where one shared node (c) appears
// in two branches. After rewriting the branches (Add(c,0)→c, Mul(c,1)→c), the
// rebuilt root should have both sources pointing to c with the same arena index.
func TestSubtreeSharingAfterRewrite(t *testing.T) {
	a := newArena()
	c := constN(a, 5)
	zero := constN(a, 0)
	one := constN(a, 1)

	left := addN(a, c, zero) // Add(c, 0) → c
	right := mulN(a, c, one) // Mul(c, 1) → c
	root := addN(a, left, right)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			),
			Fn: func(cap map[string]uop.UOp, _ any) (uop.UOp, bool) { return cap["x"], true },
		},
		{
			Pat: rewrite.Pat(uop.OpMul).WithSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.Pat(uop.OpConst).WithArg(int64(1)),
			),
			Fn: func(cap map[string]uop.UOp, _ any) (uop.UOp, bool) { return cap["x"], true },
		},
	})

	result := rewrite.GraphRewrite(root, pm)

	// result should be Add(c, c)
	if result.Op() != uop.OpAdd {
		t.Fatalf("root op = %v, want Add", result.Op())
	}
	s0, s1 := result.Src(0), result.Src(1)
	if s0 != c {
		t.Errorf("result.Src(0) = index %d, want %d (c)", s0.Index(), c.Index())
	}
	if s1 != c {
		t.Errorf("result.Src(1) = index %d, want %d (c)", s1.Index(), c.Index())
	}
	// Both sources must be the same interned node.
	if s0 != s1 {
		t.Errorf("result.Src(0) != result.Src(1): shared subtree not shared by index")
	}
}

// ── PatternMatcher.Add composition ───────────────────────────────────────────

func TestPatternMatcherAdd(t *testing.T) {
	a := newArena()
	x := constN(a, 7)
	zero := constN(a, 0)
	sentinel := constN(a, 99)

	pm1 := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd).WithSrc(rewrite.WildPat(), rewrite.Pat(uop.OpConst).WithArg(int64(0))),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return sentinel, true },
		},
	})
	pm2 := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpAdd),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return x, true },
		},
	})

	combined := pm1.Add(pm2)
	sum := addN(a, x, zero)

	result := rewrite.GraphRewrite(sum, combined)
	// pm1's more-specific rule comes first, so it should fire (returns sentinel).
	if result != sentinel {
		t.Errorf("Add composition: pm1 should win; got %d want %d", result.Index(), sentinel.Index())
	}
}

// ── WildPat ───────────────────────────────────────────────────────────────────

func TestWildPatMatchesAnyOp(t *testing.T) {
	a := newArena()
	x := constN(a, 1)
	y := constN(a, 2)
	add := addN(a, x, y)
	mul := mulN(a, x, y)
	sentinel := constN(a, 0)

	// Rule: WildPat with src constraint — matches any op that has two Const children.
	// Use it at the outer level via a Pats(OpAdd, OpMul) rule instead (WildPat can't
	// be a top-level pattern; must specify ops).
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pats(uop.OpAdd, uop.OpMul).WithSrc(
				rewrite.WildPat(), rewrite.WildPat(),
			),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return sentinel, true },
		},
	})

	for _, node := range []uop.UOp{add, mul} {
		result := rewrite.GraphRewrite(node, pm)
		if result != sentinel {
			t.Errorf("Pats(Add,Mul) did not match %v", node.Op())
		}
	}
}

// ── arg constraint ────────────────────────────────────────────────────────────

func TestArgConstraint(t *testing.T) {
	a := newArena()
	c5 := constN(a, 5)
	c6 := constN(a, 6)
	target := constN(a, 99)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(5)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return target, true },
		},
	})

	if r := rewrite.GraphRewrite(c5, pm); r != target {
		t.Errorf("Const(5) rule: want target, got index %d", r.Index())
	}
	if r := rewrite.GraphRewrite(c6, pm); r != c6 {
		t.Errorf("Const(6) should not match Const(5) rule")
	}
}

// ── dtype constraint ──────────────────────────────────────────────────────────

func TestDtypeConstraint(t *testing.T) {
	a := newArena()
	i32 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	i64 := a.New(uop.OpConst, uop.Dtypes.Int64, nil, int64(1), nil)
	target := constN(a, 99)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst).WithDtype(uop.Dtypes.Int32),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return target, true },
		},
	})

	if r := rewrite.GraphRewrite(i32, pm); r != target {
		t.Errorf("Int32 Const should match dtype constraint")
	}
	if r := rewrite.GraphRewrite(i64, pm); r != i64 {
		t.Errorf("Int64 Const should not match Int32 dtype constraint")
	}
}

// ── AnyPat combinator ─────────────────────────────────────────────────────────

func TestAnyPat(t *testing.T) {
	a := newArena()
	add := addN(a, constN(a, 1), constN(a, 2))
	mul := mulN(a, constN(a, 3), constN(a, 4))
	target := constN(a, 0)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.AnyPat(rewrite.Pat(uop.OpAdd), rewrite.Pat(uop.OpMul)),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return target, true },
		},
	})

	for _, node := range []uop.UOp{add, mul} {
		if r := rewrite.GraphRewrite(node, pm); r != target {
			t.Errorf("AnyPat did not match %v", node.Op())
		}
	}
}

// ── WithRepSrc (repeat src pattern) ──────────────────────────────────────────

func TestRepSrc(t *testing.T) {
	a := newArena()
	c1 := constN(a, 1)
	c2 := constN(a, 2)
	c3 := constN(a, 3)
	// Build a Sink with three Const children.
	sink3 := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{c1, c2, c3}, nil, nil)
	// Build a Sink with two Const and one Add child.
	add := addN(a, c1, c2)
	sinkMixed := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{c1, c2, add}, nil, nil)

	target := constN(a, 99)
	hitCount := 0

	// Rule: Sink where ALL children are Const → rewrite.
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpSink).WithRepSrc(rewrite.Pat(uop.OpConst)),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) {
				hitCount++
				return target, true
			},
		},
	})

	if r := rewrite.GraphRewrite(sink3, pm); r != target {
		t.Errorf("Sink(Const×3): expected target, got %v", r.Op())
	}
	if r := rewrite.GraphRewrite(sinkMixed, pm); r == target {
		t.Errorf("Sink(Const,Const,Add): should not match all-Const rep pattern")
	}
	if hitCount != 1 {
		t.Errorf("rep-src rule fired %d times, want exactly 1", hitCount)
	}
}

// ── ctx passthrough ───────────────────────────────────────────────────────────

func TestCtxPassthrough(t *testing.T) {
	a := newArena()
	c := constN(a, 1)
	target := constN(a, 42)

	var gotCtx any
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst),
			Fn: func(_ map[string]uop.UOp, ctx any) (uop.UOp, bool) {
				gotCtx = ctx
				return target, true
			},
		},
	})

	rewrite.GraphRewrite(c, pm, rewrite.WithCtx("hello"))
	if gotCtx != "hello" {
		t.Errorf("ctx = %v, want \"hello\"", gotCtx)
	}
}

// ── leaf nodes (no sources) rewrite correctly ─────────────────────────────────

func TestLeafNodeRewrite(t *testing.T) {
	a := newArena()
	leaf := constN(a, 7)
	target := constN(a, 0)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return target, true },
		},
	})

	result := rewrite.GraphRewrite(leaf, pm)
	if result != target {
		t.Errorf("leaf rewrite: got %d, want %d", result.Index(), target.Index())
	}
}

// ── no-match passthrough ──────────────────────────────────────────────────────

func TestNoMatchPassthrough(t *testing.T) {
	a := newArena()
	x := constN(a, 1)
	y := constN(a, 2)
	sum := addN(a, x, y)

	// PM with a rule that cannot match this graph.
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpMul),
			Fn:  func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) { return uop.UOp{}, false },
		},
	})

	result := rewrite.GraphRewrite(sum, pm)
	if result != sum {
		t.Errorf("no-match: graph changed when no rule fires")
	}
}

// ── DAG sharing: shared subtree processed once ───────────────────────────────

// TestSharedSubtreeProcessedOnce verifies that a shared node at the bottom of a
// diamond DAG is only rewritten once, not once per parent.
func TestSharedSubtreeProcessedOnce(t *testing.T) {
	a := newArena()
	base := constN(a, 0)
	left := addN(a, base, constN(a, 1))
	right := addN(a, base, constN(a, 2))
	root := addN(a, left, right)

	rewrites := 0
	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpConst).WithArg(int64(0)),
			Fn: func(_ map[string]uop.UOp, _ any) (uop.UOp, bool) {
				rewrites++
				return constN(a, 100), true
			},
		},
	})

	rewrite.GraphRewrite(root, pm)
	if rewrites != 1 {
		t.Errorf("shared Const(0) was rewritten %d times, want exactly 1", rewrites)
	}
}

// ── diamond: shared node ITSELF rewrites, both parents must land on the new index ──

// TestDiamondSharedNodeRewrites is the core property Phase 6 autodiff depends on.
// Every forward node consumed by both forward and backward is a diamond: two parents,
// one shared child. Here the shared child c = Mul(d,1) rewrites to d; p1 and p2 both
// hold c. After graph_rewrite both parents must reference d — not the stale c index.
// TestSubtreeSharingAfterRewrite does NOT cover this: it collapses parents to c but
// never rewrites c itself.
func TestDiamondSharedNodeRewrites(t *testing.T) {
	a := newArena()
	d := constN(a, 5)
	c := mulN(a, d, constN(a, 1)) // c = Mul(d,1) → d
	p1 := addN(a, c, constN(a, 2))
	p2 := addN(a, c, constN(a, 3))
	root := addN(a, p1, p2)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpMul).WithSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.Pat(uop.OpConst).WithArg(int64(1)),
			),
			Fn: func(cap map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return cap["x"], true
			},
		},
	})

	result := rewrite.GraphRewrite(root, pm)

	if result.NSrc() != 2 {
		t.Fatalf("root NSrc = %d, want 2", result.NSrc())
	}
	rp1, rp2 := result.Src(0), result.Src(1)

	// Both rebuilt parents must point to d at their first src.
	if rp1.NSrc() < 1 || rp1.Src(0) != d {
		t.Errorf("p1'.Src(0) idx=%d, want d idx=%d", rp1.Src(0).Index(), d.Index())
	}
	if rp2.NSrc() < 1 || rp2.Src(0) != d {
		t.Errorf("p2'.Src(0) idx=%d, want d idx=%d", rp2.Src(0).Index(), d.Index())
	}
	// The two diamond paths must converge to the exact same node by index.
	if rp1.Src(0) != rp2.Src(0) {
		t.Errorf("diamond paths diverge: p1'.Src(0) idx=%d, p2'.Src(0) idx=%d",
			rp1.Src(0).Index(), rp2.Src(0).Index())
	}
	// Stale c index must not appear anywhere in the output.
	staleC := c.Index()
	for _, u := range []uop.UOp{result, rp1, rp2, rp1.Src(0), rp2.Src(0)} {
		if u.Index() == staleC {
			t.Errorf("stale c index %d found in output (op=%v)", staleC, u.Op())
		}
	}
}

// TestDiamondLongShortPath is the ordering variant that catches the stale-child bug.
// The shared node is reachable via a 1-hop path (root.Src(0) = shared directly) AND
// a 200-hop spine (root.Src(1) → chain → shared at the bottom). The driver must write
// replace[shared] before any parent rebuilds its srcs — regardless of which DFS path
// reaches shared first. If the replace memo is applied stale, the long path produces a
// different index than the short path.
func TestDiamondLongShortPath(t *testing.T) {
	const spineDepth = 200
	a := uop.NewArena(spineDepth + 10)

	d := constN(a, 5)
	shared := mulN(a, d, constN(a, 1)) // Mul(d,1) → d; this is the shared node
	extra := constN(a, 42)

	// spine[0] = Add(shared, extra), spine[i] = Add(spine[i-1], extra).
	cur := shared
	for range spineDepth {
		cur = addN(a, cur, extra)
	}
	deepSpine := cur // 200 Add levels; Src(0)^200 reaches shared

	// root.Src(0) = shared  (short path, depth 1)
	// root.Src(1) = deepSpine  (long path, depth 201 to reach shared)
	root := addN(a, shared, deepSpine)

	pm := rewrite.NewPatternMatcher([]rewrite.Rule{
		{
			Pat: rewrite.Pat(uop.OpMul).WithSrc(
				rewrite.WildPat().WithName("x"),
				rewrite.Pat(uop.OpConst).WithArg(int64(1)),
			),
			Fn: func(cap map[string]uop.UOp, _ any) (uop.UOp, bool) {
				return cap["x"], true
			},
		},
	})

	result := rewrite.GraphRewrite(root, pm)

	// Short path: result.Src(0) = shared → d.
	shortEnd := result.Src(0)
	if shortEnd != d {
		t.Errorf("short path: result.Src(0) idx=%d, want d idx=%d", shortEnd.Index(), d.Index())
	}

	// Long path: descend Src(0) through each spine level.
	// After spineDepth descents through Add nodes we reach the rewritten shared = d.
	longEnd := result.Src(1)
	for i := range spineDepth {
		if longEnd.Op() != uop.OpAdd {
			t.Fatalf("long path depth %d: op=%v, want Add", i, longEnd.Op())
		}
		longEnd = longEnd.Src(0)
	}
	if longEnd != d {
		t.Errorf("long path bottom idx=%d, want d idx=%d", longEnd.Index(), d.Index())
	}

	// Both paths must arrive at the same node by index.
	if shortEnd != longEnd {
		t.Errorf("short path (idx=%d) and long path (idx=%d) reach different nodes — replace memo stale",
			shortEnd.Index(), longEnd.Index())
	}

	// Stale shared index must not appear in the output.
	stale := shared.Index()
	if shortEnd.Index() == stale || longEnd.Index() == stale {
		t.Errorf("stale shared index %d found in output", stale)
	}
}
