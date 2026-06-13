package schedule_test

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestExecItem_SetAst pins the Defect A seam contract: swapping the Ast must
// clear every render-derived field (WGSL, LocalSize, WorkgroupCount,
// SymDispatch) so executors re-render, while setting the same Ast must keep
// the pre-rendered fast path intact.
func TestExecItem_SetAst(t *testing.T) {
	a := uop.NewArena(256)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := x.Exp2()
	z := y.Log2()

	item := schedule.ExecItem{
		Ast:            y.Node(),
		WGSL:           "pre-rendered",
		LocalSize:      [3]int{64, 1, 1},
		WorkgroupCount: [3]int{2, 1, 1},
		SymDispatch:    [3]schedule.DimDispatch{{Const: 7}},
	}

	// Same Ast: no-op, render-derived fields preserved.
	item.SetAst(y.Node())
	if item.WGSL != "pre-rendered" || item.LocalSize != [3]int{64, 1, 1} {
		t.Fatalf("SetAst(same ast) must keep the pre-rendered fast path: %+v", item)
	}

	// New Ast: every render-derived field invalidated.
	item.SetAst(z.Node())
	if item.Ast != z.Node() {
		t.Fatalf("SetAst did not swap the Ast")
	}
	if item.WGSL != "" {
		t.Errorf("SetAst left stale WGSL %q", item.WGSL)
	}
	if item.LocalSize != [3]int{} || item.WorkgroupCount != [3]int{} {
		t.Errorf("SetAst left stale dispatch sizing: ls=%v wc=%v", item.LocalSize, item.WorkgroupCount)
	}
	if item.SymDispatch[0].Const != 0 || item.SymDispatch[0].SymBounds != nil {
		t.Errorf("SetAst left stale SymDispatch: %+v", item.SymDispatch)
	}
}
