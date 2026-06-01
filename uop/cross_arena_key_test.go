package uop_test

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// cross_arena_key_test.go — proves the Option B Slice 4 hygiene: ShapeSintArg
// no longer encodes symbolic dims by raw arena index (the recurring §10 bug
// class of construction-order identity), so the StructuralKey of a sink built
// in two distinct arenas from the same logical graph is byte-identical.

// buildGraph constructs a small symbolic graph in a fresh arena and returns
// the sink UOp. The graph is shape [n, 4] where n is a DefineVar with the
// same bounds in every call. Two calls to buildGraph in two distinct arenas
// produce structurally identical sinks even though their arena indices differ.
func buildGraph(varName string) uop.UOp {
	a := uop.NewArena(8)
	defVar := a.DefineVar(varName, 1, 64)
	arg := uop.ShapeSintArg{
		{Sym: true, VarName: varName, Mul: 1},
		{V: 4},
	}
	return a.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{defVar}, arg, nil)
}

// TestCrossArenaStructuralKey proves the Slice 4 hygiene: two arenas
// constructing the same logical graph produce byte-equal structural keys.
// Pre-Slice 4, ShapeSintArg.VarIdx (raw arena index) leaked construction
// order into the structural hash; this test would have failed because the
// DefineVar's arena index can vary across arenas. Post-fix, identity is
// keyed on VarArg.Name (a user-supplied label) so the hash is portable.
func TestCrossArenaStructuralKey(t *testing.T) {
	sink1 := buildGraph("n")
	sink2 := buildGraph("n")

	keys1 := uop.StructuralKeys(sink1.Arena())
	keys2 := uop.StructuralKeys(sink2.Arena())

	k1 := keys1[sink1.Index()]
	k2 := keys2[sink2.Index()]

	t.Logf("arena 1 sink key: %#x", k1)
	t.Logf("arena 2 sink key: %#x", k2)

	if k1 != k2 {
		t.Fatalf("cross-arena structural keys diverged: arena1=%#x arena2=%#x — "+
			"ShapeSintArg encoding leaked construction-order identity (SPEC §10 bug class)", k1, k2)
	}
}

// TestCrossArenaStructuralKeyDifferentVarNamesDistinct sanity-checks that the
// VarName actually discriminates: two graphs that differ ONLY in DefineVar name
// must produce DIFFERENT structural keys (otherwise the new encoding has dropped
// the variable's identity altogether).
func TestCrossArenaStructuralKeyDifferentVarNamesDistinct(t *testing.T) {
	sinkN := buildGraph("n")
	sinkM := buildGraph("m")

	keys1 := uop.StructuralKeys(sinkN.Arena())
	keys2 := uop.StructuralKeys(sinkM.Arena())

	kN := keys1[sinkN.Index()]
	kM := keys2[sinkM.Index()]

	if kN == kM {
		t.Fatalf("structural keys for VarName=%q and %q collided (%#x) — "+
			"the variable's name dropped from the structural key", "n", "m", kN)
	}
}

// TestSameArenaStructuralKeyStable verifies Slice 1 invariance: in one arena,
// the structural key of a logical sink does not depend on construction order
// of unrelated nodes that precede it.
func TestSameArenaStructuralKeyStable(t *testing.T) {
	// Arena A: build only the sink.
	a1 := uop.NewArena(8)
	defVar1 := a1.DefineVar("n", 1, 64)
	arg1 := uop.ShapeSintArg{
		{Sym: true, VarName: "n", Mul: 1},
		{V: 4},
	}
	sink1 := a1.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{defVar1}, arg1, nil)

	// Arena B: prepend an unrelated allocation that shifts arena indices,
	// then build the same sink.
	a2 := uop.NewArena(8)
	_ = a2.New(uop.OpConst, uop.Dtypes.Int64, nil, int64(7), nil) // unrelated
	_ = a2.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3.14), nil)
	defVar2 := a2.DefineVar("n", 1, 64)
	arg2 := uop.ShapeSintArg{
		{Sym: true, VarName: "n", Mul: 1},
		{V: 4},
	}
	sink2 := a2.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{defVar2}, arg2, nil)

	k1 := uop.StructuralKeys(a1)[sink1.Index()]
	k2 := uop.StructuralKeys(a2)[sink2.Index()]

	if k1 != k2 {
		t.Fatalf("same-logical-sink keys differ across construction-order shifts: %#x vs %#x", k1, k2)
	}
}

// TestCrossArenaStructuralKeyMulDistinct checks Mul discrimination: two graphs
// differing only in the per-dim multiplier (n vs 4n) must hash distinctly.
func TestCrossArenaStructuralKeyMulDistinct(t *testing.T) {
	a := uop.NewArena(8)
	defVar := a.DefineVar("n", 1, 64)
	argBare := uop.ShapeSintArg{{Sym: true, VarName: "n", Mul: 1}}
	argMul4 := uop.ShapeSintArg{{Sym: true, VarName: "n", Mul: 4}}
	sinkBare := a.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{defVar}, argBare, nil)
	sinkMul4 := a.New(uop.OpBuffer, uop.Dtypes.Float32, []uop.UOp{defVar}, argMul4, nil)

	keys := uop.StructuralKeys(a)
	if keys[sinkBare.Index()] == keys[sinkMul4.Index()] {
		t.Fatalf("Mul=1 and Mul=4 produced identical structural keys — " +
			"multiplier dropped from the structural key")
	}
}
