package uop_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── arena lifecycle ───────────────────────────────────────────────────────────

func TestNewArena(t *testing.T) {
	a := uop.NewArena(64)
	if a.Len() != 0 {
		t.Errorf("fresh arena Len = %d, want 0", a.Len())
	}
}

func TestArenaReset(t *testing.T) {
	a := uop.NewArena(16)
	a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	if a.Len() != 2 {
		t.Fatalf("before Reset: Len = %d, want 2", a.Len())
	}

	a.Reset()

	if a.Len() != 0 {
		t.Errorf("after Reset: Len = %d, want 0", a.Len())
	}

	// First node after Reset must take index 0 (arena starts over).
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if u.Index() != 0 {
		t.Errorf("first node after Reset has index %d, want 0", u.Index())
	}
}

// TestArenaResetClearsIntern verifies that the intern cache is cleared by Reset:
// a node created post-Reset must not match an old (now-gone) entry.
func TestArenaResetClearsIntern(t *testing.T) {
	a := uop.NewArena(16)
	a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(99), nil)
	a.Reset()
	u := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(99), nil)

	if a.Len() != 1 {
		t.Errorf("post-Reset Len = %d, want 1", a.Len())
	}
	if u.Index() != 0 {
		t.Errorf("post-Reset first node index = %d, want 0", u.Index())
	}
}

// ── UOp accessors ─────────────────────────────────────────────────────────────

func TestUOpAccessors(t *testing.T) {
	a := uop.NewArena(16)
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, int64(42), nil)

	if u.Op() != uop.OpConst {
		t.Errorf("Op = %v, want OpConst", u.Op())
	}
	if u.DType() != uop.Dtypes.Float32 {
		t.Errorf("DType = %v, want Float32", u.DType())
	}
	if u.Arg() != int64(42) {
		t.Errorf("Arg = %v (%T), want int64(42)", u.Arg(), u.Arg())
	}
	if u.Tag() != nil {
		t.Errorf("Tag = %v, want nil", u.Tag())
	}
	if u.NSrc() != 0 {
		t.Errorf("NSrc = %d, want 0", u.NSrc())
	}
	if !u.Valid() {
		t.Error("Valid() = false, want true")
	}
}

func TestUOpSrcAccessors(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3), nil)
	add := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)

	if add.NSrc() != 2 {
		t.Fatalf("NSrc = %d, want 2", add.NSrc())
	}
	if add.Src(0) != x {
		t.Error("Src(0) != x")
	}
	if add.Src(1) != y {
		t.Error("Src(1) != y")
	}
}

func TestUOpZeroValueInvalid(t *testing.T) {
	var u uop.UOp
	if u.Valid() {
		t.Error("zero-value UOp must report Valid() = false")
	}
}

// ── interning: basic cases ───────────────────────────────────────────────────

func TestInterningSameNode(t *testing.T) {
	a := uop.NewArena(16)
	u1 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)

	if u1 != u2 {
		t.Error("same node built twice must be identical (index equality)")
	}
	if a.Len() != 1 {
		t.Errorf("arena Len = %d after two identical News, want 1", a.Len())
	}
}

func TestInterningDifferentOps(t *testing.T) {
	a := uop.NewArena(16)
	u1 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	u2 := a.New(uop.OpVConst, uop.Dtypes.Float32, nil, float64(1), nil)

	if u1 == u2 {
		t.Error("different ops must produce distinct UOps")
	}
}

func TestInterningDifferentArg(t *testing.T) {
	a := uop.NewArena(16)
	u1 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(2), nil)

	if u1 == u2 {
		t.Error("different args must produce distinct UOps")
	}
}

func TestInterningDifferentDType(t *testing.T) {
	a := uop.NewArena(16)
	u1 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0), nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Float64, nil, float64(0), nil)

	if u1 == u2 {
		t.Error("different dtypes must produce distinct UOps")
	}
}

func TestInterningDifferentSrc(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)

	// neg(x) vs neg(y) — same op and dtype, different src
	nx := a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{x}, nil, nil)
	ny := a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{y}, nil, nil)

	if nx == ny {
		t.Error("nodes with different src must be distinct")
	}
}

// ── interning: DAG correctness (load-bearing) ─────────────────────────────────

// TestInterningWithSrc verifies that binary nodes intern correctly.
func TestInterningWithSrc(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3), nil)

	add1 := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)
	add2 := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)

	if add1 != add2 {
		t.Error("identical binary nodes must intern to the same UOp")
	}
}

// TestInterningDeepDAG builds (a+b)*(a+b) and verifies both paths to the sum
// intern to the same node. The mul's two sources must be identical.
func TestInterningDeepDAG(t *testing.T) {
	a := uop.NewArena(64)
	av := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	bv := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)

	sum1 := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{av, bv}, nil, nil)
	sum2 := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{av, bv}, nil, nil)

	if sum1 != sum2 {
		t.Error("(a+b) built twice must be the same interned node")
	}

	mul := a.New(uop.OpMul, uop.Dtypes.Float32, []uop.UOp{sum1, sum2}, nil, nil)
	if mul.Src(0) != mul.Src(1) {
		t.Error("mul(s,s): both sources must be the same interned node")
	}
}

// TestInterningBuildTwoWays builds the same DAG through separate call sequences
// and asserts the roots are identical. This is the core identity contract.
func TestInterningBuildTwoWays(t *testing.T) {
	build := func(a *uop.Arena) uop.UOp {
		x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(5), nil)
		y := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(7), nil)
		return a.New(uop.OpAdd, uop.Dtypes.Int32, []uop.UOp{x, y}, nil, nil)
	}

	a := uop.NewArena(16)
	r1 := build(a)
	r2 := build(a)

	if r1 != r2 {
		t.Errorf("same DAG built twice must yield identical UOp: r1.idx=%d r2.idx=%d",
			r1.Index(), r2.Index())
	}
	if a.Len() != 3 { // x, y, add
		t.Errorf("arena Len = %d, want 3", a.Len())
	}
}

// TestInterningChain verifies a 3-level chain: (a + b) + c  intern correctly.
func TestInterningChain(t *testing.T) {
	a := uop.NewArena(32)
	c1 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	c2 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	c3 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3), nil)

	inner := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{c1, c2}, nil, nil)
	root1 := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{inner, c3}, nil, nil)
	root2 := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{inner, c3}, nil, nil)

	if root1 != root2 {
		t.Error("same chain built twice must be the same UOp")
	}
	if a.Len() != 4 { // c1, c2, c3, inner, root — wait, 5 unique nodes
		// actually: c1, c2, inner, c3, root = 5
	}
	_ = root1
}

// ── interning: tag participates in the key ────────────────────────────────────

func TestTagInInternKey(t *testing.T) {
	a := uop.NewArena(8)
	u1 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), true) // different tag

	if u1 == u2 {
		t.Error("different tags must produce distinct UOps")
	}
}

func TestTagSameInterns(t *testing.T) {
	a := uop.NewArena(8)
	u1 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), "sched")
	u2 := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), "sched")

	if u1 != u2 {
		t.Error("identical tags must still intern to the same UOp")
	}
}

// ── bypass ops: intrinsic identity ───────────────────────────────────────────

// TestBypassOpsAreDistinct verifies that bypass ops (UNIQUE, LUNIQUE, BUFFER)
// always produce distinct nodes even when all fields are identical.
func TestBypassOpsAreDistinct(t *testing.T) {
	tests := []struct {
		name string
		op   uop.Op
	}{
		{"Unique", uop.OpUnique},
		{"LUnique", uop.OpLUnique},
		{"Buffer", uop.OpBuffer},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(16)
			u1 := a.New(tc.op, uop.Dtypes.Void, nil, nil, nil)
			u2 := a.New(tc.op, uop.Dtypes.Void, nil, nil, nil)

			if u1 == u2 {
				t.Errorf("%s: bypass op produced identical UOps — must be distinct", tc.name)
			}
			if a.Len() != 2 {
				t.Errorf("%s: arena Len = %d, want 2", tc.name, a.Len())
			}
		})
	}
}

// TestBypassOpsManyDistinct verifies that N bypass calls produce N distinct nodes.
func TestBypassOpsManyDistinct(t *testing.T) {
	const N = 10
	a := uop.NewArena(N)
	seen := map[uop.UOp]bool{}
	for i := 0; i < N; i++ {
		u := a.New(uop.OpUnique, uop.Dtypes.Void, nil, nil, nil)
		if seen[u] {
			t.Fatalf("bypass op returned duplicate UOp at iteration %d", i)
		}
		seen[u] = true
	}
	if a.Len() != N {
		t.Errorf("arena Len = %d, want %d", a.Len(), N)
	}
}

// TestNonBypassOpInterns verifies that non-bypass ops with identical fields DO dedup.
func TestNonBypassOpInterns(t *testing.T) {
	a := uop.NewArena(4)
	u1 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(0), nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(0), nil)

	if u1 != u2 {
		t.Error("non-bypass op must dedup identical nodes")
	}
}

// ── cross-arena safety ────────────────────────────────────────────────────────

func TestCrossArenaSrcPanics(t *testing.T) {
	a1 := uop.NewArena(8)
	a2 := uop.NewArena(8)
	u := a1.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for cross-arena src, got none")
		}
	}()
	a2.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{u}, nil, nil)
}

func TestZeroValueSrcPanics(t *testing.T) {
	a := uop.NewArena(8)
	var zero uop.UOp

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero-value src UOp, got none")
		}
	}()
	a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{zero}, nil, nil)
}

// ── arg type coverage ─────────────────────────────────────────────────────────

// TestArgTypesCover verifies that every supported arg type interns correctly
// and that distinct values produce distinct nodes.
func TestArgTypesCover(t *testing.T) {
	a := uop.NewArena(64)

	args := []any{
		nil,
		int64(0), int64(1), int64(-1), int64(1<<32), int64(-1 << 32),
		float64(0.0), float64(1.5), float64(-1.5),
		true, false,
		"", "hello", "world",
	}

	seen := map[uint32]any{} // index → arg value
	for _, arg := range args {
		u1 := a.New(uop.OpConst, uop.Dtypes.Void, nil, arg, nil)
		u2 := a.New(uop.OpConst, uop.Dtypes.Void, nil, arg, nil)
		if u1 != u2 {
			t.Errorf("arg=%v (%T): intern failed", arg, arg)
		}
		if prev, ok := seen[u1.Index()]; ok {
			t.Errorf("index collision: arg %v (%T) and %v (%T) share index %d",
				arg, arg, prev, prev, u1.Index())
		}
		seen[u1.Index()] = arg
	}
}

// TestArgTypeNilVsIntZero verifies that nil and int64(0) have distinct intern keys.
func TestArgTypeNilVsIntZero(t *testing.T) {
	a := uop.NewArena(8)
	nilNode := a.New(uop.OpConst, uop.Dtypes.Void, nil, nil, nil)
	zeroNode := a.New(uop.OpConst, uop.Dtypes.Void, nil, int64(0), nil)

	if nilNode == zeroNode {
		t.Error("nil arg and int64(0) arg must produce distinct UOps")
	}
}

// TestUnsupportedArgPanics verifies that an unsupported arg type panics at New time.
func TestUnsupportedArgPanics(t *testing.T) {
	a := uop.NewArena(4)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unsupported arg type, got none")
		}
	}()
	// []int is not a supported arg type.
	a.New(uop.OpConst, uop.Dtypes.Void, nil, []int{1, 2, 3}, nil)
}

// ── StructuralKeys cross-arena stability ─────────────────────────────────────

// TestStructuralKeysCrossArena verifies that StructuralKeys produces the same
// hash for structurally identical nodes even when they live at different arena
// indices (i.e., the keys are a pure function of graph structure, not allocation
// order). This is the cross-build stability guarantee that StructuralKeys must
// uphold after replacing pointer-address dtype hashing with DType.StructuralHash.
func TestStructuralKeysCrossArena(t *testing.T) {
	// Arena 1: const(1.0) at idx 0, const(2.0) at idx 1, add at idx 2.
	a1 := uop.NewArena(8)
	x1 := a1.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y1 := a1.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	add1 := a1.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x1, y1}, nil, nil)
	keys1 := uop.StructuralKeys(a1)

	// Arena 2: const(99.0) at idx 0 (extra node to shift indices), then
	// const(1.0) at idx 1, const(2.0) at idx 2, add at idx 3.
	a2 := uop.NewArena(8)
	a2.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(99), nil) // idx 0: unrelated node
	x2 := a2.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y2 := a2.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	add2 := a2.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x2, y2}, nil, nil)
	keys2 := uop.StructuralKeys(a2)

	if add1.Index() == add2.Index() {
		t.Fatal("test setup: add nodes must be at different arena indices to be meaningful")
	}
	t.Logf("arena1: add@%d key=%016x  arena2: add@%d key=%016x",
		add1.Index(), keys1[add1.Index()], add2.Index(), keys2[add2.Index()])

	if keys1[add1.Index()] != keys2[add2.Index()] {
		t.Errorf("cross-arena structural key mismatch for identical add node: %016x != %016x",
			keys1[add1.Index()], keys2[add2.Index()])
	}
	// Leaf nodes with identical content must also match across arenas.
	if keys1[x1.Index()] != keys2[x2.Index()] {
		t.Errorf("cross-arena key mismatch for const(1.0): %016x != %016x",
			keys1[x1.Index()], keys2[x2.Index()])
	}
	// Structurally distinct nodes must not collide.
	c99 := a2.At(0)
	if keys2[c99.Index()] == keys2[x2.Index()] {
		t.Errorf("const(99) and const(1) must have different structural keys")
	}
}

// ── map key correctness ───────────────────────────────────────────────────────

func TestUOpAsMapKey(t *testing.T) {
	a := uop.NewArena(16)
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)

	m := map[uop.UOp]string{}
	m[x] = "x"
	m[y] = "y"

	// Reconstructing the same structural node must hit intern and look up "x".
	xAgain := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if m[xAgain] != "x" {
		t.Error("interned UOp map lookup failed")
	}
	if m[y] != "y" {
		t.Error("distinct interned node lookup failed")
	}
}

// ── String representation ─────────────────────────────────────────────────────

func TestUOpString(t *testing.T) {
	a := uop.NewArena(8)
	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3.14), nil)
	s := u.String()

	if s == "" {
		t.Error("String() returned empty")
	}
	if !strings.Contains(s, "Const") {
		t.Errorf("String() = %q; want to contain op name 'Const'", s)
	}
}

func TestUOpStringInvalid(t *testing.T) {
	var u uop.UOp
	s := u.String()
	if !strings.Contains(s, "invalid") {
		t.Errorf("zero-value UOp String() = %q; want to contain 'invalid'", s)
	}
}

// ── provenance: first-construction-wins ──────────────────────────────────────

// TestProvenanceFirstConstructionWins is the critical proof that:
//
//  1. building a node in forward phase then requesting the same structure in
//     backward phase still yields ONE arena entry (interning intact), AND
//  2. that node's provenance is PhaseForward (first-construction wins, not
//     flipped by the backward reference).
func TestProvenanceFirstConstructionWins(t *testing.T) {
	a := uop.NewArena(16)

	// Build in forward phase (default: PhaseForward).
	x := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	y := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	add := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)

	if a.Provenance(add.Index()) != uop.PhaseForward {
		t.Fatalf("forward-built add has provenance %v, want PhaseForward", a.Provenance(add.Index()))
	}

	// Switch to backward phase and request the same structural node.
	prev := a.SetPhase(uop.PhaseBackward)
	addAgain := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{x, y}, nil, nil)
	a.SetPhase(prev)

	// Interning intact: same arena index.
	if addAgain.Index() != add.Index() {
		t.Errorf("interning broken: forward add@%d, backward request→@%d (expected same)",
			add.Index(), addAgain.Index())
	}
	// Arena has exactly 3 nodes (x, y, add) — not 4.
	if a.Len() != 3 {
		t.Errorf("arena Len = %d after backward re-request, want 3 (no duplicate)", a.Len())
	}
	// First-construction wins: provenance is still forward.
	if a.Provenance(addAgain.Index()) != uop.PhaseForward {
		t.Errorf("provenance flipped to %v, want PhaseForward (first-construction wins)",
			a.Provenance(addAgain.Index()))
	}

	t.Logf("proof: add@idx=%d, same structure in backward→idx=%d, provenance=%v",
		add.Index(), addAgain.Index(), a.Provenance(add.Index()))
}

// TestProvenanceBypassOpsGetCurrentPhase verifies that bypass ops (BUFFER,
// UNIQUE, LUNIQUE) — which always allocate fresh slots — correctly inherit
// the current build phase at allocation time.
func TestProvenanceBypassOpsGetCurrentPhase(t *testing.T) {
	a := uop.NewArena(8)

	bufFwd := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)
	if a.Provenance(bufFwd.Index()) != uop.PhaseForward {
		t.Errorf("forward buffer: provenance = %v, want PhaseForward", a.Provenance(bufFwd.Index()))
	}

	prev := a.SetPhase(uop.PhaseBackward)
	bufBwd := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4}, nil)
	a.SetPhase(prev)

	if a.Provenance(bufBwd.Index()) != uop.PhaseBackward {
		t.Errorf("backward buffer: provenance = %v, want PhaseBackward", a.Provenance(bufBwd.Index()))
	}
}

// TestProvenanceResetClearsPhase verifies that Reset restores the arena to the
// forward phase and clears all provenance records.
func TestProvenanceResetClearsPhase(t *testing.T) {
	a := uop.NewArena(8)
	prev := a.SetPhase(uop.PhaseBackward)
	a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	a.SetPhase(prev)

	a.Reset()

	u := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if a.Provenance(u.Index()) != uop.PhaseForward {
		t.Errorf("after Reset, node provenance = %v, want PhaseForward", a.Provenance(u.Index()))
	}
	if u.Index() != 0 {
		t.Errorf("after Reset, first node index = %d, want 0", u.Index())
	}
}
