package uop_test

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// definevar_bind_test.go — adversarial coverage for Arena.DefineVar and Arena.Bind.
//
// These wrap the OpDefineVar + OpBind construction used by the Option-A symbolic
// path (NewSymbolicBatchInput + RealizeWithBinding). Production exercises them
// end-to-end, but they have no direct adversarial coverage; this file pins:
//
//   - DefineVar interns by (name, min, max). Different bounds → distinct nodes;
//     identical bounds + identical name → same node.
//   - DefineVar's dtype is Index, its src holds two Const bounds [min, max+1].
//   - Bind interns by (DefineVar, val). Same val on the same var → same node;
//     different val → distinct.
//   - Bind reuses the DefineVar dtype.
//
// Recurring-bug-class adjacency (B1, OpRange): a future intern-set change near
// these constructors would silently regress symbolic shapes and we'd discover
// it the slow way (a gradient mysteriously zeroing out weeks later).

// ── DefineVar: identity & structure ──────────────────────────────────────────

func TestDefineVarStructure(t *testing.T) {
	a := uop.NewArena(16)
	v := a.DefineVar("batch", 1, 64)

	if v.Op() != uop.OpDefineVar {
		t.Errorf("Op = %v, want OpDefineVar", v.Op())
	}
	if v.DType() != uop.Dtypes.Index {
		t.Errorf("DType = %v, want Index", v.DType())
	}
	if v.NSrc() != 2 {
		t.Fatalf("NSrc = %d, want 2 (min, max+1)", v.NSrc())
	}

	// Per uop.go:347-349: src[0]=min, src[1]=max+1 (exclusive upper).
	minC, maxC := v.Src(0), v.Src(1)
	if minC.Op() != uop.OpConst {
		t.Errorf("src[0].Op = %v, want OpConst", minC.Op())
	}
	if minC.Arg() != int64(1) {
		t.Errorf("src[0].Arg = %v, want int64(1) (min)", minC.Arg())
	}
	if maxC.Arg() != int64(65) {
		t.Errorf("src[1].Arg = %v, want int64(65) (max+1, exclusive)", maxC.Arg())
	}

	// VarArg.Name is the identifier.
	arg, ok := v.Arg().(uop.VarArg)
	if !ok {
		t.Fatalf("Arg type = %T, want uop.VarArg", v.Arg())
	}
	if arg.Name != "batch" {
		t.Errorf("VarArg.Name = %q, want \"batch\"", arg.Name)
	}
}

// TestDefineVarInternsByName verifies the documented contract (uop.go:323-324):
// "Two DefineVars with the same name and bounds intern to one node; different
// names produce distinct nodes."
func TestDefineVarInternsByName(t *testing.T) {
	a := uop.NewArena(16)
	v1 := a.DefineVar("batch", 1, 64)
	v2 := a.DefineVar("batch", 1, 64)

	if v1 != v2 {
		t.Errorf("DefineVar with identical (name, min, max) did not intern: v1.idx=%d v2.idx=%d",
			v1.Index(), v2.Index())
	}
	// Arena should hold exactly 3 nodes: the two Const bounds + the DefineVar.
	if a.Len() != 3 {
		t.Errorf("arena Len = %d, want 3 (Const(min), Const(max+1), DefineVar)", a.Len())
	}
}

func TestDefineVarDifferentNameDistinct(t *testing.T) {
	a := uop.NewArena(16)
	v1 := a.DefineVar("batch", 1, 64)
	v2 := a.DefineVar("seq", 1, 64)

	if v1 == v2 {
		t.Error("DefineVar with different name aliased — name dropped from intern key")
	}
}

func TestDefineVarDifferentBoundsDistinct(t *testing.T) {
	cases := []struct {
		name     string
		a1, a2   int64 // min
		b1, b2   int64 // max
		distinct bool
	}{
		{"min differs", 1, 2, 64, 64, true},
		{"max differs", 1, 1, 64, 128, true},
		{"both differ", 1, 2, 64, 128, true},
		{"identical", 1, 1, 64, 64, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(16)
			v1 := a.DefineVar("batch", tc.a1, tc.b1)
			v2 := a.DefineVar("batch", tc.a2, tc.b2)
			got := v1 != v2
			if got != tc.distinct {
				t.Errorf("DefineVar distinct = %v, want %v (a1=%d b1=%d a2=%d b2=%d)",
					got, tc.distinct, tc.a1, tc.b1, tc.a2, tc.b2)
			}
		})
	}
}

// TestDefineVarMaxPlusOneEncoding pins the [min, max+1) exclusive-upper contract.
// If anyone "fixes" DefineVar to store max directly, downstream BoundsOf returns
// off-by-one bounds; gradient symbolic axes would over- or under-run by 1 element.
func TestDefineVarMaxPlusOneEncoding(t *testing.T) {
	a := uop.NewArena(8)
	v := a.DefineVar("n", 0, 9) // [0, 9] inclusive → src[1] should be 10

	if v.Src(1).Arg() != int64(10) {
		t.Errorf("max+1 encoding broken: src[1].Arg = %v, want int64(10) for max=9",
			v.Src(1).Arg())
	}
}

// TestDefineVarReusedBoundsConsts verifies that two DefineVars sharing identical
// bounds also share the underlying Const bound nodes (interning).
func TestDefineVarReusedBoundsConsts(t *testing.T) {
	a := uop.NewArena(16)
	v1 := a.DefineVar("a", 0, 9)
	v2 := a.DefineVar("b", 0, 9) // different name, identical bounds

	if v1 == v2 {
		t.Fatal("test setup: different names must produce distinct DefineVars")
	}
	if v1.Src(0) != v2.Src(0) {
		t.Error("Const(min) not shared across DefineVars with same min — Const interning broken")
	}
	if v1.Src(1) != v2.Src(1) {
		t.Error("Const(max+1) not shared across DefineVars with same max — Const interning broken")
	}
}

// ── Bind: identity & structure ───────────────────────────────────────────────

func TestBindStructure(t *testing.T) {
	a := uop.NewArena(16)
	v := a.DefineVar("batch", 1, 64)
	b := a.Bind(v, 32)

	if b.Op() != uop.OpBind {
		t.Errorf("Op = %v, want OpBind", b.Op())
	}
	if b.DType() != v.DType() {
		t.Errorf("DType = %v, want %v (reuse DefineVar dtype)", b.DType(), v.DType())
	}
	if b.NSrc() != 1 {
		t.Fatalf("NSrc = %d, want 1", b.NSrc())
	}
	if b.Src(0) != v {
		t.Error("Bind.Src(0) is not the DefineVar")
	}
	if b.Arg() != int64(32) {
		t.Errorf("Arg = %v, want int64(32)", b.Arg())
	}
}

func TestBindInternsByValue(t *testing.T) {
	a := uop.NewArena(16)
	v := a.DefineVar("batch", 1, 64)
	b1 := a.Bind(v, 32)
	b2 := a.Bind(v, 32)

	if b1 != b2 {
		t.Errorf("Bind(v,32) repeated did not intern: b1.idx=%d b2.idx=%d", b1.Index(), b2.Index())
	}
}

func TestBindDifferentValuesDistinct(t *testing.T) {
	a := uop.NewArena(16)
	v := a.DefineVar("batch", 1, 64)
	b1 := a.Bind(v, 32)
	b2 := a.Bind(v, 33)

	if b1 == b2 {
		t.Error("Bind with different concrete values aliased — val dropped from intern key")
	}
}

func TestBindDifferentVarsDistinct(t *testing.T) {
	a := uop.NewArena(16)
	v1 := a.DefineVar("batch", 1, 64)
	v2 := a.DefineVar("seq", 1, 64)
	b1 := a.Bind(v1, 32)
	b2 := a.Bind(v2, 32)

	if b1 == b2 {
		t.Error("Bind on different DefineVars aliased — src not part of intern key")
	}
}

// TestBindIsInternedNotBypassed pins that OpBind goes through normal interning
// (not bypassed like OpRange/OpBuffer). If a future refactor adds OpBind to
// bypassInternSet, every Bind call would allocate a fresh slot — bloating the
// arena and breaking equality lookups for binding maps keyed by the Bind UOp.
func TestBindIsInternedNotBypassed(t *testing.T) {
	a := uop.NewArena(16)
	v := a.DefineVar("batch", 1, 64)
	pre := a.Len()
	a.Bind(v, 5)
	a.Bind(v, 5)
	a.Bind(v, 5)

	added := a.Len() - pre
	if added != 1 {
		t.Errorf("3 identical Bind calls added %d nodes, want 1 (OpBind must intern)", added)
	}
}

// TestDefineVarStructuralKeysCrossArena verifies that two arenas with the same
// DefineVar (same name, bounds) produce the same StructuralKey — necessary for
// cross-arena cache lookups (schedule cache + BEAM disk cache).
func TestDefineVarStructuralKeysCrossArena(t *testing.T) {
	a1 := uop.NewArena(8)
	v1 := a1.DefineVar("batch", 1, 64)
	keys1 := uop.StructuralKeys(a1)

	a2 := uop.NewArena(8)
	// Insert an unrelated node first to shift indices, proving the key is
	// content-only and does not depend on arena position.
	a2.New(uop.OpConst, uop.Dtypes.Index, nil, int64(999), nil)
	v2 := a2.DefineVar("batch", 1, 64)
	keys2 := uop.StructuralKeys(a2)

	if v1.Index() == v2.Index() {
		t.Fatal("test setup: DefineVars must land at different indices to make the cross-arena test meaningful")
	}
	if keys1[v1.Index()] != keys2[v2.Index()] {
		t.Errorf("DefineVar StructuralKey mismatch across arenas: %016x vs %016x",
			keys1[v1.Index()], keys2[v2.Index()])
	}
}

// TestDefineVarStructuralKeysNameMatters: two arenas, one with name="batch"
// and one with name="seq" but identical bounds, must produce different
// StructuralKeys. If they collide, the schedule cache would mistake a
// "batch"-parameterised kernel for a "seq"-parameterised one and vice versa.
func TestDefineVarStructuralKeysNameMatters(t *testing.T) {
	a1 := uop.NewArena(8)
	v1 := a1.DefineVar("batch", 1, 64)
	keys1 := uop.StructuralKeys(a1)

	a2 := uop.NewArena(8)
	v2 := a2.DefineVar("seq", 1, 64)
	keys2 := uop.StructuralKeys(a2)

	if keys1[v1.Index()] == keys2[v2.Index()] {
		t.Errorf("DefineVar StructuralKey collision: name=\"batch\" and name=\"seq\" with same bounds "+
			"share key %016x — VarArg.Name dropped from StructuralKeys", keys1[v1.Index()])
	}
}

// TestDefineVarProvenance: DefineVar built in backward phase must record
// backward provenance. Necessary for gradient/symbolic interaction.
func TestDefineVarProvenance(t *testing.T) {
	a := uop.NewArena(8)
	prev := a.SetPhase(uop.PhaseBackward)
	v := a.DefineVar("n", 0, 9)
	a.SetPhase(prev)

	if a.Provenance(v.Index()) != uop.PhaseBackward {
		t.Errorf("DefineVar provenance = %v, want PhaseBackward", a.Provenance(v.Index()))
	}
}

// TestDefineVarTrivialBounds: a [k, k] singleton interval is legal and the
// encoding gives src[1].Arg = k+1, so src[0] != src[1] when k != -1. (Pinning
// because tinygrad has historically rejected degenerate intervals; ours does not.)
func TestDefineVarTrivialBounds(t *testing.T) {
	a := uop.NewArena(8)
	v := a.DefineVar("x", 5, 5) // [5, 6) — singleton
	if v.Src(0).Arg() != int64(5) {
		t.Errorf("singleton min = %v, want 5", v.Src(0).Arg())
	}
	if v.Src(1).Arg() != int64(6) {
		t.Errorf("singleton max+1 = %v, want 6", v.Src(1).Arg())
	}
}
