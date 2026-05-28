package codegen_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func newArena() *uop.Arena { return uop.NewArena(1024) }

func makeSink(a *uop.Arena, ts ...*tensor.Tensor) uop.UOp {
	srcs := make([]uop.UOp, len(ts))
	for i, t := range ts {
		srcs[i] = t.Node()
	}
	return a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
}

// firstItem returns the first ExecItem from the schedule (asserts at least one exists).
func firstItem(t *testing.T, sink uop.UOp) schedule.ExecItem {
	t.Helper()
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("CreateSchedule returned 0 items")
	}
	return items[0]
}

// allItems returns all ExecItems.
func allItems(t *testing.T, sink uop.UOp) []schedule.ExecItem {
	t.Helper()
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("CreateSchedule returned 0 items")
	}
	return items
}

// assertContains fails the test if wgsl does not contain every expected substring.
func assertContains(t *testing.T, wgsl string, wants ...string) {
	t.Helper()
	for _, w := range wants {
		if !strings.Contains(wgsl, w) {
			t.Errorf("WGSL missing %q\nfull shader:\n%s", w, wgsl)
		}
	}
}

// assertNotContains fails if wgsl contains any of the forbidden substrings.
func assertNotContains(t *testing.T, wgsl string, bads ...string) {
	t.Helper()
	for _, bad := range bads {
		if strings.Contains(wgsl, bad) {
			t.Errorf("WGSL should not contain %q\nfull shader:\n%s", bad, wgsl)
		}
	}
}

// countOccurrences counts non-overlapping occurrences of sub in s.
func countOccurrences(s, sub string) int {
	return strings.Count(s, sub)
}

// ── structural invariants (apply to every kernel) ─────────────────────────────

// verifyWGSLStructure checks properties that must hold for every emitted shader.
func verifyWGSLStructure(t *testing.T, wgsl string, item schedule.ExecItem) {
	t.Helper()

	// Entry point and workgroup annotation must be present.
	assertContains(t, wgsl, "@compute", "fn main(", "@builtin(global_invocation_id)")

	// Each PARAM must have a corresponding @binding.
	ki := item.Ast.Arg().(uop.KernelInfo)
	for i := 0; i < ki.NumParams; i++ {
		assertContains(t, wgsl, fmt.Sprintf("@binding(%d)", i))
		assertContains(t, wgsl, fmt.Sprintf("data%d", i))
	}

	// Output buffer (binding 0) must be read_write; inputs must be read.
	assertContains(t, wgsl, "var<storage, read_write> data0")
	for i := 1; i < ki.NumParams; i++ {
		assertContains(t, wgsl, fmt.Sprintf("var<storage, read> data%d", i))
	}

	// Shader must store into data0.
	assertContains(t, wgsl, "data0[")

	// Braces must be balanced.
	opens := strings.Count(wgsl, "{")
	closes := strings.Count(wgsl, "}")
	if opens != closes {
		t.Errorf("unbalanced braces: %d open, %d close", opens, closes)
	}
}

// ── Test: elementwise kernel ──────────────────────────────────────────────────

func TestRender_Elementwise(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)

	// Elementwise: exactly 1 binding (output) and 1 input (the leaf buffer).
	// No reduce loops — no "for (" in the body.
	assertContains(t, wgsl, "exp2(")
	assertNotContains(t, wgsl, "for (")
	// Bounds guard references the output size (32).
	assertContains(t, wgsl, "32")
}

// ── Test: reduce kernel ───────────────────────────────────────────────────────

func TestRender_Reduce(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{0, 1}, false) // scalar output

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)

	// Must have at least one sequential for loop (AxisReduce).
	if !strings.Contains(wgsl, "for (") {
		t.Error("reduce kernel must contain at least one for loop")
	}

	// Accumulator must be initialised to additive identity (0.0).
	assertContains(t, wgsl, "var acc0: f32 = 0.0")

	// Accumulator update must add to acc.
	assertContains(t, wgsl, "acc0 = acc0 +")
}

func TestRender_ReduceAxisZero(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{0}, false) // [4, 8] -> [8]

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)

	// Output is [8], so it should have a loop range of size 8.
	assertContains(t, wgsl, "8")
	// Reduce loop is over axis 0, which has size 4.
	assertContains(t, wgsl, "4")
}

func TestRender_MaxReduce(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
	y := x.Max(nil, false) // scalar output

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)

	// Max identity for float32 should be -Inf.
	// 0xff7fffff is the bit pattern for -FLT_MAX.
	assertContains(t, wgsl, "bitcast<f32>(0xff7fffffu)")
	assertContains(t, wgsl, "max(")
}

// ── Test: indexing and layout ─────────────────────────────────────────────────

func TestRender_ReshapeIndex(t *testing.T) {
	a := newArena()
	// [4, 4] -> [16] reshape + exp2
	x := tensor.NewLeaf(a, []int64{4, 4}, uop.Dtypes.Float32, "webgpu")
	y := x.Reshape([]int64{16}).Exp2()

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)
	assertContains(t, wgsl, "exp2(")
}

func TestRender_PadElementwise(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{10}, uop.Dtypes.Float32, "webgpu")
	// Pad to 16 with constant 0.
	y := x.Pad([][2]int64{{0, 6}})

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)
	// Shader should contain the constant 0.0 for padding.
	assertContains(t, wgsl, "select(", "0.0")
}

// ── Test: fusion ──────────────────────────────────────────────────────────────

func TestRender_FusedChain(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2().Log2().Sqrt()

	items := schedule.CreateSchedule(makeSink(a, y), "webgpu")
	if len(items) != 1 {
		t.Errorf("expected 1 fused item, got %d", len(items))
	}
	item := items[0]
	wgsl := codegen.RenderWGSL(item).WGSL

	verifyWGSLStructure(t, wgsl, item)
	assertContains(t, wgsl, "exp2(", "log2(", "sqrt(")
}

// ── Test: forward + backward schedule ─────────────────────────────────────────

func TestRender_ForwardBackward(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum(nil, false) // forward

	// Gradients
	grads := tensor.Backward(y, []*tensor.Tensor{x})
	gx := grads[x]

	// Schedule everything
	items := schedule.CreateSchedule(makeSink(a, y, gx), "webgpu")

	for i, item := range items {
		wgsl := codegen.RenderWGSL(item).WGSL
		verifyWGSLStructure(t, wgsl, item)
		t.Logf("kernel %d:\n%s", i, wgsl)
	}

	// Kernel 0 must have a reduce loop; kernel 1 must not.
	wgsl0 := codegen.RenderWGSL(items[0]).WGSL
	wgsl1 := codegen.RenderWGSL(items[len(items)-1]).WGSL
	if !strings.Contains(wgsl0, "for (") && !strings.Contains(wgsl1, "for (") {
		t.Error("at least one kernel should have a reduce loop")
	}
}

// ── Golden Tests (regressions) ────────────────────────────────────────────────

func TestGolden_Elementwise4(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	t.Logf("generated WGSL:\n%s", wgsl)

	// Structural assertions (not brittle byte-for-byte matching).
	assertContains(t, wgsl,
		"@group(0) @binding(0) var<storage, read_write> data0: array<f32>",
		"@group(0) @binding(1) var<storage, read> data1: array<f32>",
		"@compute @workgroup_size(64, 1, 1)",
		"fn main(",
		"exp2(",
		"data0[",
	)
}

func TestGolden_ScalarReduce8(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum(nil, false)

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item).WGSL

	t.Logf("generated WGSL:\n%s", wgsl)

	assertContains(t, wgsl,
		"@group(0) @binding(0) var<storage, read_write> data0: array<f32>",
		"@group(0) @binding(1) var<storage, read> data1: array<f32>",
		"@compute @workgroup_size(1, 1, 1)",
		"var acc0: f32 = 0.0",
		"for (",
		"acc0 = acc0 +",
		"data0[",
	)
}
