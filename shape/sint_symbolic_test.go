package shape_test

// TestSymbolicOracle — SLICE 3a proof test.
//
// Proves that symbolic Sint arithmetic builds correct UOp expressions, that
// GraphRewrite+Symbolic folds them to the same result as concrete arithmetic,
// and that BoundsOf returns intervals that contain all concrete values.
//
// Four proofs:
//  1. Symbolic-equals-concrete oracle (the core correctness proof)
//  2. Bounds containment and tightness
//  3. DefineVar / Bind interning invariants
//  4. Static path still passes (via go test ./...)

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/rewrite"
	"github.com/georgebuilds/anneal/rewrite/rules"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// pointSubst replaces one specific UOp node (by arena index) with a replacement.
// Used as a BPM to substitute DefineVar → Const(val) before folding.
type pointSubst struct {
	target uint32
	repl   uop.UOp
}

func (ps *pointSubst) Rewrite(u uop.UOp, _ any) (uop.UOp, bool) {
	if u.Index() == ps.target {
		return ps.repl, true
	}
	return uop.UOp{}, false
}

// foldAt evaluates expr symbolically: substitutes varNode → Const(val)
// and folds the resulting expression using the Symbolic rule set.
func foldAt(expr, varNode uop.UOp, val int64) uop.UOp {
	a := expr.Arena()
	constVal := a.New(uop.OpConst, varNode.DType(), nil, val, nil)
	bpm := &pointSubst{target: varNode.Index(), repl: constVal}
	return rewrite.GraphRewrite(expr, rules.Symbolic, rewrite.WithBPM(bpm))
}

// foldAt2 substitutes two variables simultaneously.
func foldAt2(expr, v0, v1 uop.UOp, val0, val1 int64) uop.UOp {
	a := expr.Arena()
	c0 := a.New(uop.OpConst, v0.DType(), nil, val0, nil)
	c1 := a.New(uop.OpConst, v1.DType(), nil, val1, nil)
	type two struct{ ps pointSubst }
	bpm := &twoSubst{
		t0: v0.Index(), r0: c0,
		t1: v1.Index(), r1: c1,
	}
	return rewrite.GraphRewrite(expr, rules.Symbolic, rewrite.WithBPM(bpm))
}

type twoSubst struct {
	t0, t1 uint32
	r0, r1 uop.UOp
}

func (ts *twoSubst) Rewrite(u uop.UOp, _ any) (uop.UOp, bool) {
	if u.Index() == ts.t0 {
		return ts.r0, true
	}
	if u.Index() == ts.t1 {
		return ts.r1, true
	}
	return uop.UOp{}, false
}

// constI64 extracts the int64 from a folded-to-Const UOp.
// Reports failure if the node is not a Const with an int64 arg.
func constI64(u uop.UOp) (int64, bool) {
	if u.Op() != uop.OpConst {
		return 0, false
	}
	v, ok := u.Arg().(int64)
	return v, ok
}

// ── Proof 1: symbolic-equals-concrete oracle ──────────────────────────────────

func TestSymbolicEqualsConcreteOracle(t *testing.T) {
	// Test values: chosen to span small, medium, and large; prime-free so
	// products don't accidentally cancel each other's errors.
	bindVals := []int64{1, 7, 8, 64, 1000}

	type row struct {
		expr   string
		sym    int64
		conc   int64
		match  bool
	}

	// Each case: build the expression symbolically, evaluate at each binding,
	// compare with the concrete expression evaluated directly.
	cases := []struct {
		name     string
		buildSym func(a *uop.Arena, n uop.UOp) uop.UOp
		buildConc func(a *uop.Arena, val int64) uop.UOp
	}{
		{
			// (n * 4) + 3
			name: "n*4+3",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
				c3 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
				mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
				return a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, c3}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, val*4+3, nil)
			},
		},
		{
			// n + 0  →  n  (identity rule)
			name: "n+0=n",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c0 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
				return a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{n, c0}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, val, nil)
			},
		},
		{
			// n * 1  →  n  (identity rule)
			name: "n*1=n",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
				return a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c1}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, val, nil)
			},
		},
		{
			// n * 0  →  0  (absorbing element)
			name: "n*0=0",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c0 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
				return a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c0}, nil, nil)
			},
			buildConc: func(a *uop.Arena, _ int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
			},
		},
		{
			// n / 1  →  n  (identity rule)
			name: "n/1=n",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
				return a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{n, c1}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, val, nil)
			},
		},
		{
			// (n * 8) / 8  →  n  (fold: n*8 constant, then divide)
			name: "(n*8)/8",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c8 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(8), nil)
				mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c8}, nil, nil)
				return a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{mul, c8}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, val, nil)
			},
		},
		{
			// (n * 3) % 3  →  0  (self-mod)
			name: "(n*3)%3",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c3 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
				mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c3}, nil, nil)
				return a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{mul, c3}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, (val*3)%3, nil)
			},
		},
		{
			// n - n  →  0  (self-cancellation)
			name: "n-n=0",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				return a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{n, n}, nil, nil)
			},
			buildConc: func(a *uop.Arena, _ int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
			},
		},
		{
			// n * 4 + n  →  n * 5  (fold arithmetic)
			name: "n*4+n=n*5",
			buildSym: func(a *uop.Arena, n uop.UOp) uop.UOp {
				c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
				mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
				return a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, n}, nil, nil)
			},
			buildConc: func(a *uop.Arena, val int64) uop.UOp {
				return a.New(uop.OpConst, uop.Dtypes.Index, nil, val*4+val, nil)
			},
		},
	}

	allMatch := true
	t.Logf("%-16s  %8s  %12s  %12s  %s", "expr", "val", "sym-folded", "concrete", "match")
	t.Logf("%s", "---------------------------------------------------------------")

	for _, tc := range cases {
		for _, val := range bindVals {
			a := uop.NewArena(64)
			n := a.DefineVar("n", 1, 1024)

			// Build symbolic expression and fold at val.
			symExpr := tc.buildSym(a, n)
			folded := foldAt(symExpr, n, val)
			symVal, symOK := constI64(folded)

			// Build concrete expression (directly as Const).
			concExpr := tc.buildConc(a, val)
			// Fold the concrete expression too (for cases like n*0 that might not be Const yet).
			foldedConc := rewrite.GraphRewrite(concExpr, rules.Symbolic)
			concVal, concOK := constI64(foldedConc)

			match := symOK && concOK && symVal == concVal
			status := "OK"
			if !match {
				status = fmt.Sprintf("FAIL(symOK=%v concOK=%v sym=%d conc=%d)", symOK, concOK, symVal, concVal)
				allMatch = false
			}
			t.Logf("%-16s  %8d  %12d  %12d  %s", tc.name, val, symVal, concVal, status)
		}
	}

	if !allMatch {
		t.Fatal("symbolic-equals-concrete oracle: some rows did not match (see table above)")
	}
}

// TestSymbolicOracleTwoVars proves two-variable expressions (n+m)*2 and (n*s)/s.
func TestSymbolicOracleTwoVars(t *testing.T) {
	type pair struct{ n, m int64 }
	pairs := []pair{{1, 2}, {8, 3}, {64, 7}, {100, 50}}

	cases := []struct {
		name      string
		buildSym  func(a *uop.Arena, n, m uop.UOp) uop.UOp
		buildConc func(n, m int64) int64
	}{
		{
			name: "(n+m)*2",
			buildSym: func(a *uop.Arena, n, m uop.UOp) uop.UOp {
				c2 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
				sum := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{n, m}, nil, nil)
				return a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{sum, c2}, nil, nil)
			},
			buildConc: func(n, m int64) int64 { return (n + m) * 2 },
		},
		{
			name: "(n*s)/s (stride cancel)",
			buildSym: func(a *uop.Arena, n, s uop.UOp) uop.UOp {
				mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, s}, nil, nil)
				return a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{mul, s}, nil, nil)
			},
			buildConc: func(n, s int64) int64 { return (n * s) / s },
		},
	}

	allMatch := true
	t.Logf("%-24s  %6s  %6s  %12s  %12s  %s", "expr", "n", "m/s", "sym", "conc", "match")
	t.Logf("%s", "-------------------------------------------------------------------")

	for _, tc := range cases {
		for _, p := range pairs {
			a := uop.NewArena(64)
			n := a.DefineVar("n", 1, 10000)
			m := a.DefineVar("m", 1, 10000)

			symExpr := tc.buildSym(a, n, m)
			folded := foldAt2(symExpr, n, m, p.n, p.m)
			symVal, symOK := constI64(folded)

			conc := tc.buildConc(p.n, p.m)
			match := symOK && symVal == conc
			status := "OK"
			if !match {
				status = fmt.Sprintf("FAIL(ok=%v sym=%d conc=%d)", symOK, symVal, conc)
				allMatch = false
			}
			t.Logf("%-24s  %6d  %6d  %12d  %12d  %s", tc.name, p.n, p.m, symVal, conc, status)
		}
	}

	if !allMatch {
		t.Fatal("two-variable oracle: some rows did not match")
	}
}

// ── Proof 2: bounds containment and tightness ─────────────────────────────────

func TestSymbolicBoundsContainment(t *testing.T) {
	a := uop.NewArena(64)
	// n ∈ [1, 1024] (inclusive)
	n := a.DefineVar("n", 1, 1024)

	bindVals := []int64{1, 7, 8, 64, 1000}

	type exprCase struct {
		name    string
		build   func() uop.UOp
		expMin  int64
		expMax  int64
	}

	c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	c3 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
	c8 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(8), nil)

	cases := []exprCase{
		{
			name:   "n",
			build:  func() uop.UOp { return n },
			expMin: 1, expMax: 1024,
		},
		{
			name: "n*4",
			build: func() uop.UOp {
				return a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
			},
			expMin: 4, expMax: 4096,
		},
		{
			name: "n*4+3",
			build: func() uop.UOp {
				mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
				return a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, c3}, nil, nil)
			},
			expMin: 7, expMax: 4099,
		},
		{
			name: "n*8",
			build: func() uop.UOp {
				return a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c8}, nil, nil)
			},
			expMin: 8, expMax: 8192,
		},
	}

	allOK := true
	t.Logf("%-12s  %8s  %8s  %8s  %8s  %8s  %s",
		"expr", "bMin", "bMax", "obsMin", "obsMax", "contained", "tight")
	t.Logf("%s", "--------------------------------------------------------------------")

	for _, tc := range cases {
		expr := tc.build()
		b := rules.BoundsOf(expr)

		if !b.Valid {
			t.Errorf("expr %s: BoundsOf returned invalid", tc.name)
			allOK = false
			continue
		}

		// Verify expected tight bounds.
		if b.Min != tc.expMin || b.Max != tc.expMax {
			t.Errorf("expr %s: bounds [%d,%d] want [%d,%d]", tc.name, b.Min, b.Max, tc.expMin, tc.expMax)
			allOK = false
		}

		// Verify containment: every concrete value falls within [b.Min, b.Max].
		var obsMin, obsMax int64 = 1<<62, -1 << 62
		for _, val := range bindVals {
			folded := foldAt(expr, n, val)
			conc, ok := constI64(folded)
			if !ok {
				t.Errorf("expr %s at val=%d: fold did not produce Const", tc.name, val)
				allOK = false
				continue
			}
			if conc < b.Min || conc > b.Max {
				t.Errorf("expr %s at val=%d: concrete=%d not in bounds [%d,%d]",
					tc.name, val, conc, b.Min, b.Max)
				allOK = false
			}
			if conc < obsMin {
				obsMin = conc
			}
			if conc > obsMax {
				obsMax = conc
			}
		}

		tight := obsMin == b.Min && obsMax == b.Max
		tightStr := "yes"
		if !tight {
			tightStr = fmt.Sprintf("no(obs=[%d,%d])", obsMin, obsMax)
		}
		contained := allOK
		_ = contained
		t.Logf("%-12s  %8d  %8d  %8d  %8d  %8v  %s",
			tc.name, b.Min, b.Max, obsMin, obsMax, true, tightStr)
	}

	if !allOK {
		t.Fatal("bounds containment check failed (see above)")
	}
}

// ── Proof 3: interning of DefineVar and Bind ──────────────────────────────────

func TestDefineVarInterning(t *testing.T) {
	a := uop.NewArena(64)

	// Same name and bounds → same node (interned).
	n1 := a.DefineVar("n", 1, 1024)
	n2 := a.DefineVar("n", 1, 1024)
	if n1 != n2 {
		t.Errorf("DefineVar interning FAIL: same name+bounds should intern to one node (idx %d vs %d)",
			n1.Index(), n2.Index())
	} else {
		t.Logf("DefineVar('n',1,1024): idx=%d  [interned correctly]", n1.Index())
	}

	// Different names → different nodes.
	m := a.DefineVar("m", 1, 1024)
	if n1 == m {
		t.Errorf("DefineVar interning FAIL: 'n' and 'm' should be distinct nodes")
	} else {
		t.Logf("DefineVar('m',1,1024): idx=%d  [distinct from n, correct]", m.Index())
	}

	// Different bounds → different nodes.
	nWide := a.DefineVar("n", 1, 2048)
	if n1 == nWide {
		t.Errorf("DefineVar interning FAIL: same name, different bounds should be distinct")
	} else {
		t.Logf("DefineVar('n',1,2048): idx=%d  [distinct from 'n'[1,1024], correct]", nWide.Index())
	}

	// Bind(n, 8) folds to Const(8) after GraphRewrite.
	bindN := a.Bind(n1, int64(8))
	t.Logf("Bind(n,8): idx=%d (before fold)", bindN.Index())

	folded := rewrite.GraphRewrite(bindN, rules.Symbolic)
	val, ok := constI64(folded)
	if !ok || val != 8 {
		t.Errorf("Bind(n,8) fold FAIL: got op=%v arg=%v, want Const(8)", folded.Op(), folded.Arg())
	} else {
		t.Logf("Bind(n,8) → Const(%d): idx=%d  [fold correct]", val, folded.Index())
	}

	// Two Bind nodes for same (var, val) must intern to one node.
	b1 := a.Bind(n1, int64(8))
	b2 := a.Bind(n1, int64(8))
	if b1 != b2 {
		t.Errorf("Bind interning FAIL: Bind(n,8) twice should be same node (idx %d vs %d)",
			b1.Index(), b2.Index())
	} else {
		t.Logf("Bind(n,8) twice: idx=%d  [interned correctly]", b1.Index())
	}

	// Bind with different values must be distinct.
	b8 := a.Bind(n1, int64(8))
	b64 := a.Bind(n1, int64(64))
	if b8 == b64 {
		t.Errorf("Bind interning FAIL: Bind(n,8) and Bind(n,64) must be distinct nodes")
	} else {
		t.Logf("Bind(n,8) idx=%d  Bind(n,64) idx=%d  [distinct, correct]", b8.Index(), b64.Index())
	}
}

// ── Proof: identity rules fire for symbolic DefineVar ─────────────────────────

func TestIdentityRulesFireForDefineVar(t *testing.T) {
	a := uop.NewArena(32)
	n := a.DefineVar("n", 1, 1024)

	check := func(name string, expr uop.UOp) {
		folded := rewrite.GraphRewrite(expr, rules.Symbolic)
		// After folding the identity, the result must be the DefineVar node itself.
		if folded != n {
			t.Errorf("%s: identity did not fold to n; got op=%v idx=%d", name, folded.Op(), folded.Index())
		} else {
			t.Logf("%s → DefineVar(n)  OK", name)
		}
	}

	c0 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	c1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)

	check("n + 0", a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{n, c0}, nil, nil))
	check("n * 1", a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c1}, nil, nil))
	check("n / 1", a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{n, c1}, nil, nil))
	check("n - 0", a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{n, c0}, nil, nil))

	// n * 0 → Const(0, Index)
	cZero := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	n0 := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c0}, nil, nil)
	folded0 := rewrite.GraphRewrite(n0, rules.Symbolic)
	v, ok := constI64(folded0)
	if !ok || v != 0 {
		t.Errorf("n*0: expected Const(0) got op=%v arg=%v", folded0.Op(), folded0.Arg())
	} else {
		t.Logf("n * 0 → Const(0)  OK")
	}
	_ = cZero

	// n - n → Const(0)
	subSelf := a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{n, n}, nil, nil)
	foldedSub := rewrite.GraphRewrite(subSelf, rules.Symbolic)
	vs, oks := constI64(foldedSub)
	if !oks || vs != 0 {
		t.Errorf("n-n: expected Const(0) got op=%v arg=%v", foldedSub.Op(), foldedSub.Arg())
	} else {
		t.Logf("n - n → Const(0)  OK")
	}
}
