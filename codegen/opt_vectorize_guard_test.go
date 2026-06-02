package codegen_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestOptVectorizeGuard_PanicsOnNonTiledMatmul mirrors the OptUpcast guard
// test for OptVectorize. AxisVectorize lowering relies on the vecTileActive
// machinery set inside emitTiledReduce; without an OptTile-tagged OpReduce,
// the vec4 store path silently drops 3 of 4 lanes. Post-guard, applyVectorize
// refuses with a diagnostic naming OptTile and the silent-drop failure mode.
func TestOptVectorizeGuard_PanicsOnNonTiledMatmul(t *testing.T) {
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
			t.Fatalf("expected panic from OptVectorize on non-tiled kernel, got none")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		for _, want := range []string{"OptVectorize", "OptTile", "AxisVectorize", "silently drops lanes"} {
			if !strings.Contains(msg, want) {
				t.Errorf("panic message missing %q\nfull message: %s", want, msg)
			}
		}
	}()

	codegen.ApplyOpts(items[0], []codegen.Opt{{Kind: codegen.OptVectorize, Axis: 0, Arg: 4}})
	t.Fatal("ApplyOpts returned; expected panic")
}

// TestOptVectorizeGuard_AllowedAfterFullB37Pipeline confirms the canonical
// OptLocal x2 then OptTile then OptUpcast x2 then OptVectorize composition
// still works under the new guard.
func TestOptVectorizeGuard_AllowedAfterFullB37Pipeline(t *testing.T) {
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
		{Kind: codegen.OptUpcast, Axis: 1, Arg: 4},
		{Kind: codegen.OptVectorize, Axis: 1, Arg: 4},
	})
	if !out.Ast.Valid() {
		t.Fatal("expected valid Ast after Tile + Upcast + Vectorize pipeline")
	}
}

// TestActionSpace_NoVectorizeWithoutTile verifies BEAM's probe never generates
// an OptVectorize action for a kernel that lacks an OptTile-tagged reduce.
// Without this pre-filter, BEAM would crash on applyVectorize's guard.
func TestActionSpace_NoVectorizeWithoutTile(t *testing.T) {
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
		if act.Kind == codegen.OptVectorize {
			t.Errorf("ActionSpace returned OptVectorize for a kernel without OptTile: %+v", act)
		}
	}
}
