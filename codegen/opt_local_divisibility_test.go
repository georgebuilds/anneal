package codegen_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// elemwiseItem builds a [6, 6] elementwise add kernel — the minimal stand-in
// for the conv-spray shape class (6-extent axes, no tilable Mul(Index, Index)
// reduce) that exposed the OptLocal non-divisible padding bug.
func elemwiseItem(t *testing.T) schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(1 << 16)
	A := tensor.NewLeaf(a, []int64{6, 6}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{6, 6}, uop.Dtypes.Float32, "webgpu")
	C := A.Add(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items")
	}
	return items[0]
}

// matmulItem builds an M×K · K×N matmul kernel.
func matmulItem(t *testing.T, M, K, N int64) schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(1 << 16)
	A := tensor.NewLeaf(a, []int64{M, K}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{K, N}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items")
	}
	return items[0]
}

// TestApplyLocal_RefusesNonDivisibleOnNonTilable pins the divisibility gate:
// OptLocal with L ∤ S on a kernel WITHOUT the tilable Mul(Index, Index)
// reduce shape must refuse (return the sink unchanged — the applyTile
// inapplicability convention). The static store path indexes the output with
// the padded ceil(S/L)*L strides and has no tail mask, so applying the split
// would scatter values into a wrong layout (the conv-spray bug: L∈{4,16} on
// 6-extent axes → 20-28/36 elements wrong).
func TestApplyLocal_RefusesNonDivisibleOnNonTilable(t *testing.T) {
	for _, L := range []int{4, 16} {
		item := elemwiseItem(t)
		opted := codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 0, Arg: L})
		if opted.Index() != item.Ast.Index() {
			t.Errorf("OptLocal(axis=0, L=%d) on a 6-extent non-tilable kernel must refuse; Ast was transformed", L)
		}
	}
	// Divisible splits still apply.
	for _, L := range []int{2, 3, 6} {
		item := elemwiseItem(t)
		opted := codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 0, Arg: L})
		if opted.Index() == item.Ast.Index() {
			t.Errorf("OptLocal(axis=0, L=%d) divides the 6-extent axis and must apply; Ast was unchanged", L)
		}
	}
}

// TestApplyLocal_AllowsNonDivisibleOnTilableMatmul pins the escape valve: a
// kernel with the tilable Mul(Index, Index) reduce shape may take a
// non-divisible split, because the sanctioned OptLocal→OptTile composition
// lowers through emitTiledReduce, which masks every load and store by the
// real operand extents (proven bit-exact by the B2/B3/B37 irregular-shape
// value oracles in backend/webgpu).
func TestApplyLocal_AllowsNonDivisibleOnTilableMatmul(t *testing.T) {
	item := matmulItem(t, 17, 32, 32)
	opted := codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 0, Arg: 16})
	if opted.Index() == item.Ast.Index() {
		t.Fatalf("OptLocal(axis=0, L=16) on M=17 matmul must apply (tile-masked store path rescues the padding); Ast was unchanged")
	}
}

// TestActionSpace_OnlyDivisibleLocal verifies BEAM's probe never proposes a
// non-divisible OptLocal split. On tilable matmuls applyLocal ALLOWS the
// padding split (for the hand-composed tile pipeline), but BEAM cannot
// guarantee OptTile lands in a later ply — an un-tiled padded kernel panics
// at the unmasked static store — so the action space pre-filters it.
func TestActionSpace_OnlyDivisibleLocal(t *testing.T) {
	// M=17, N=30, K=35: no beamLocalArgs entry (8/16/32) divides 17 or 30.
	actions := codegen.ActionSpace(matmulItem(t, 17, 35, 30).Ast)
	for _, act := range actions {
		if act.Kind == codegen.OptLocal {
			t.Errorf("ActionSpace proposed OptLocal %+v on a 17×30 output — no beamLocalArgs value divides either extent", act)
		}
	}
	// 6-extent elementwise: 8/16/32 never divide 6 → no OptLocal proposals.
	actions = codegen.ActionSpace(elemwiseItem(t).Ast)
	for _, act := range actions {
		if act.Kind == codegen.OptLocal {
			t.Errorf("ActionSpace proposed OptLocal %+v on a [6, 6] elementwise kernel", act)
		}
	}
	// 64³ matmul: 8/16/32 all divide 64 → OptLocal must still be proposed.
	found := false
	for _, act := range codegen.ActionSpace(matmulItem(t, 64, 64, 64).Ast) {
		if act.Kind == codegen.OptLocal {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("ActionSpace proposed no OptLocal for a 64³ matmul — divisibility pre-filter is over-rejecting")
	}
}

// TestLower_PanicsOnPaddedUnmaskedStaticStore pins the fail-loud backstop for
// the residual hole the apply-time gate cannot close: OptLocal(L ∤ S) on a
// tilable matmul is allowed (escape valve above), but if no OptTile follows,
// the kernel lowers through the UNMASKED static store with padded strides.
// The lowerer detects the padding (loop-range product != output element
// count) and panics instead of emitting a silently-wrong kernel.
func TestLower_PanicsOnPaddedUnmaskedStaticStore(t *testing.T) {
	item := matmulItem(t, 17, 32, 32)
	opted := codegen.ApplyOpts(item, []codegen.Opt{{Kind: codegen.OptLocal, Axis: 0, Arg: 16}})

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic from rendering an OptLocal-padded kernel without a tile-masked store, got none")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		for _, want := range []string{"padded dispatch space", "OptLocal"} {
			if !strings.Contains(msg, want) {
				t.Errorf("panic message missing %q\nfull message: %s", want, msg)
			}
		}
	}()

	codegen.RenderWGSL(opted)
	t.Fatal("RenderWGSL returned; expected panic")
}
