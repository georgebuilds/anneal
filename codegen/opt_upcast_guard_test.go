package codegen_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestOptUpcastGuard_PanicsOnNonTiledMatmul exercises the fail-loud assertion
// added for the carried-debt "non-matmul UPCAST silently broken" issue. A
// matmul kernel without OptTile applied still has an eligible parallel range
// for OptUpcast to grab, but the lowerer's AxisUpcast handling is only wired
// inside emitTiledReduce. Pre-guard, applyUpcast happily mutated the AST and
// the resulting kernel silently wrote only lane 0 of each factor-wide stripe;
// post-guard, applyUpcast refuses with a diagnostic.
func TestOptUpcastGuard_PanicsOnNonTiledMatmul(t *testing.T) {
	a := uop.NewArena(1 << 16)
	A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items")
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic from OptUpcast on non-tiled kernel, got none")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		for _, want := range []string{"OptUpcast", "OptTile", "AxisUpcast", "silently drops lanes"} {
			if !strings.Contains(msg, want) {
				t.Errorf("panic message missing %q\nfull message: %s", want, msg)
			}
		}
	}()

	codegen.ApplyOpts(items[0], []codegen.Opt{{Kind: codegen.OptUpcast, Axis: 0, Arg: 4}})
	t.Fatal("ApplyOpts returned; expected panic")
}

// TestOptUpcastGuard_AllowedAfterOptTile confirms the canonical
// OptTile then OptUpcast composition still works under the new guard.
func TestOptUpcastGuard_AllowedAfterOptTile(t *testing.T) {
	a := uop.NewArena(1 << 16)
	A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items")
	}

	out := codegen.ApplyOpts(items[0], []codegen.Opt{
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptLocal, Axis: 0, Arg: 16},
		{Kind: codegen.OptTile, Axis: 0, Arg: 16},
		{Kind: codegen.OptUpcast, Axis: 0, Arg: 4},
	})
	if !out.Ast.Valid() {
		t.Fatal("expected valid Ast after Tile then Upcast pipeline")
	}
}

// TestActionSpace_NoUpcastWithoutTile verifies that BEAM's probe never
// generates an OptUpcast action for a kernel that lacks an OptTile-tagged
// reduce. Without this pre-filter, BEAM would crash on applyUpcast's guard.
func TestActionSpace_NoUpcastWithoutTile(t *testing.T) {
	a := uop.NewArena(1 << 16)
	A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("schedule produced 0 items")
	}

	actions := codegen.ActionSpace(items[0].Ast)
	for _, act := range actions {
		if act.Kind == codegen.OptUpcast {
			t.Errorf("ActionSpace returned OptUpcast for a kernel without OptTile: %+v", act)
		}
	}
}
