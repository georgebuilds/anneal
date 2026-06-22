package webgpu

// Slice 7c - direct verification of the executor's symbolic dispatch-thread
// computation.
//
// The 7b end-to-end C1/C2/C3/C4 tests prove the *dispatch result* is correct
// (max-abs-diff = 0 across the whole output), but they don't isolate the
// per-call value coming out of symElemCount for the output buffer - that's
// what `outElems` becomes at executor.go:929, which determines the workgroup
// count. This file pins that value directly for each C-case so any regression
// that breaks the multi-dim-correct path is caught even when it would otherwise
// be masked by the `if outElems == 0 { outElems = n }` SymVars[0] fallback.
//
// Audit context (see notes/slice7c_audit.md): the fallback never fires on
// C1-C4 today because symElemCount returns a strictly positive total for the
// output buffers under the test bindings. These tests assert the *non-zero
// total* directly, so a future change that silently regressed symElemCount
// to 0 for these shapes (and would thus reroute through the positional
// SymVars[0] fallback) would fail here.

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// Output buffers constructed below mirror exactly the buildC{1,2,3,4}Kernel
// helpers in slice7b_ccase_test.go (which lives in package webgpu_test). They
// are duplicated here because slice7b's helpers aren't exported and live in a
// different package; the shape of the duplication is intentionally minimal -
// only the output buffer (Bufs[0]) is materialised, since symElemCount only
// reads buf.Shape + per-dim sym metadata.

func c1OutBuf(varN, varM string) schedule.Buffer {
	return schedule.Buffer{
		UOpIdx:    0,
		Shape:     []int64{0, 0},
		DType:     uop.Dtypes.Float32,
		Slot:      -1,
		SymDimMul: []int64{1, 1},
		SymDimVar: []string{varN, varM},
	}
}

func c2OutBuf(varN string) schedule.Buffer {
	return schedule.Buffer{
		UOpIdx:    0,
		Shape:     []int64{4, 0},
		DType:     uop.Dtypes.Float32,
		Slot:      -1,
		SymDimMul: []int64{1},
		SymDimVar: []string{varN},
	}
}

func c3OutBuf(varN, varK string) schedule.Buffer {
	return schedule.Buffer{
		UOpIdx: 0,
		Shape:  []int64{0, 0},
		DType:  uop.Dtypes.Float32,
		Slot:   -1,
		SymDimAffine: []schedule.SymDimAffineEntry{
			{Terms: []uop.AffineTerm{{Mul: 1, VarName: varN}}, Offset: 0},
			{Terms: []uop.AffineTerm{{Mul: 1, VarName: varK}}, Offset: 4},
		},
	}
}

func c4OutBuf(varN string) schedule.Buffer {
	return schedule.Buffer{
		UOpIdx:    0,
		Shape:     []int64{4, 0},
		DType:     uop.Dtypes.Float32,
		Slot:      -1,
		SymDimMul: []int64{1},
		SymDimVar: []string{varN},
	}
}

// TestSymElemCountCCases pins the dispatch-thread count for each C-case
// output buffer. These are the values that the executor feeds into
// `wgs := ceil(outElems / LocalSize[0])` at executor.go:933.
func TestSymElemCountCCases(t *testing.T) {
	cases := []struct {
		name    string
		buf     schedule.Buffer
		binding map[string]int64
		symVars []string
		want    int64
	}{
		{
			name:    "C1 n=3 m=5 -> 15",
			buf:     c1OutBuf("cn", "cm"),
			binding: map[string]int64{"cn": 3, "cm": 5},
			symVars: []string{"cm", "cn"}, // name-sorted
			want:    15,
		},
		{
			name:    "C1 n=5 m=3 -> 15 (commuted binding)",
			buf:     c1OutBuf("cn", "cm"),
			binding: map[string]int64{"cn": 5, "cm": 3},
			symVars: []string{"cm", "cn"},
			want:    15,
		},
		{
			name:    "C1 n=7 m=11 -> 77",
			buf:     c1OutBuf("cn", "cm"),
			binding: map[string]int64{"cn": 7, "cm": 11},
			symVars: []string{"cm", "cn"},
			want:    77,
		},
		{
			name:    "C2 n=3 -> 12 (sym inner)",
			buf:     c2OutBuf("cn2"),
			binding: map[string]int64{"cn2": 3},
			symVars: []string{"cn2"},
			want:    12,
		},
		{
			name:    "C2 n=11 -> 44",
			buf:     c2OutBuf("cn2"),
			binding: map[string]int64{"cn2": 11},
			symVars: []string{"cn2"},
			want:    44,
		},
		{
			name:    "C3 n=3 k=2 -> 18 (Affine inner = k+4)",
			buf:     c3OutBuf("c3n", "c3k"),
			binding: map[string]int64{"c3n": 3, "c3k": 2},
			symVars: []string{"c3k", "c3n"},
			want:    18,
		},
		{
			name:    "C3 n=5 k=1 -> 25",
			buf:     c3OutBuf("c3n", "c3k"),
			binding: map[string]int64{"c3n": 5, "c3k": 1},
			symVars: []string{"c3k", "c3n"},
			want:    25,
		},
		{
			name:    "C3 n=8 k=4 -> 64",
			buf:     c3OutBuf("c3n", "c3k"),
			binding: map[string]int64{"c3n": 8, "c3k": 4},
			symVars: []string{"c3k", "c3n"},
			want:    64,
		},
		{
			name:    "C4 n=3 -> 12 (permute -> sym inner)",
			buf:     c4OutBuf("c4n"),
			binding: map[string]int64{"c4n": 3},
			symVars: []string{"c4n"},
			want:    12,
		},
		{
			name:    "C4 n=5 -> 20",
			buf:     c4OutBuf("c4n"),
			binding: map[string]int64{"c4n": 5},
			symVars: []string{"c4n"},
			want:    20,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := symElemCount(c.buf, c.binding, c.symVars)
			if got != c.want {
				t.Fatalf("symElemCount=%d want %d", got, c.want)
			}
			if got == 0 {
				t.Fatalf("symElemCount returned 0 - would route through the "+
					"SymVars[0] fallback at executor.go:931; want %d", c.want)
			}
		})
	}
}

// TestSymElemCountAdaptsToBinding asserts that for the same buffer (no
// recompile), changing the binding produces the matching total. This is the
// "compile-once, dispatch-many" property: symElemCount is the per-dispatch
// resolver and must walk binding fresh each call.
func TestSymElemCountAdaptsToBinding(t *testing.T) {
	t.Run("C1 [n,m] binding sweep", func(t *testing.T) {
		buf := c1OutBuf("cn", "cm")
		symVars := []string{"cm", "cn"}
		bindings := []struct {
			n, m, want int64
		}{
			{3, 5, 15},
			{7, 1, 7},
			{1, 11, 11},
			{13, 13, 169},
		}
		for _, b := range bindings {
			binding := map[string]int64{"cn": b.n, "cm": b.m}
			got := symElemCount(buf, binding, symVars)
			if got != b.want {
				t.Fatalf("C1 n=%d m=%d: symElemCount=%d want %d", b.n, b.m, got, b.want)
			}
		}
	})

	t.Run("C3 [n, 4+k] binding sweep", func(t *testing.T) {
		buf := c3OutBuf("c3n", "c3k")
		symVars := []string{"c3k", "c3n"}
		bindings := []struct {
			n, k, want int64
		}{
			{3, 2, 18},
			{5, 0, 20}, // k=0: affine still gives offset 4 → n*4
			{1, 7, 11}, // n=1: 1*(4+7) = 11
			{10, 10, 140},
		}
		for _, b := range bindings {
			binding := map[string]int64{"c3n": b.n, "c3k": b.k}
			got := symElemCount(buf, binding, symVars)
			if got != b.want {
				t.Fatalf("C3 n=%d k=%d: symElemCount=%d want %d", b.n, b.k, got, b.want)
			}
		}
	})
}

// TestSymElemCountFallbackNeverFiresOnCCases asserts that, for every C-case
// output buffer at every test binding used by slice7b_ccase_test.go, the
// `outElems == 0` fallback at executor.go:930 does not engage. This is the
// audit's load-bearing claim: the 7b C-cases pass because symElemCount
// returns the correct multi-dim total directly, not because the positional
// SymVars[0] fallback happens to coincide with a single-dim case.
func TestSymElemCountFallbackNeverFiresOnCCases(t *testing.T) {
	// C1: TestC1Elementwise uses cases {3,5}, {5,3}, {7,11}.
	c1cases := []struct{ n, m int64 }{{3, 5}, {5, 3}, {7, 11}}
	c1buf := c1OutBuf("cn", "cm")
	c1syms := []string{"cm", "cn"}
	for _, c := range c1cases {
		binding := map[string]int64{"cn": c.n, "cm": c.m}
		if got := symElemCount(c1buf, binding, c1syms); got == 0 {
			t.Errorf("C1 n=%d m=%d: symElemCount=0 (would fall back to SymVars[0])", c.n, c.m)
		}
	}

	// C2: TestC2ElementwiseSymInner uses n ∈ {3, 5, 7, 11}.
	c2buf := c2OutBuf("cn2")
	c2syms := []string{"cn2"}
	for _, n := range []int64{3, 5, 7, 11} {
		binding := map[string]int64{"cn2": n}
		if got := symElemCount(c2buf, binding, c2syms); got == 0 {
			t.Errorf("C2 n=%d: symElemCount=0 (would fall back to SymVars[0])", n)
		}
	}

	// C3: TestC3PadInner uses cases {3,2}, {5,1}, {8,4}.
	c3cases := []struct{ n, k int64 }{{3, 2}, {5, 1}, {8, 4}}
	c3buf := c3OutBuf("c3n", "c3k")
	c3syms := []string{"c3k", "c3n"}
	for _, c := range c3cases {
		binding := map[string]int64{"c3n": c.n, "c3k": c.k}
		if got := symElemCount(c3buf, binding, c3syms); got == 0 {
			t.Errorf("C3 n=%d k=%d: symElemCount=0 (would fall back to SymVars[0])", c.n, c.k)
		}
	}

	// C4: TestC4PermutePlus uses n ∈ {3, 5}.
	c4buf := c4OutBuf("c4n")
	c4syms := []string{"c4n"}
	for _, n := range []int64{3, 5} {
		binding := map[string]int64{"c4n": n}
		if got := symElemCount(c4buf, binding, c4syms); got == 0 {
			t.Errorf("C4 n=%d: symElemCount=0 (would fall back to SymVars[0])", n)
		}
	}
}
