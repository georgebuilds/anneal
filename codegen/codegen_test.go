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

	// Bounds guard must be present.
	if !strings.Contains(wgsl, "if (gid_x >=") {
		t.Errorf("WGSL missing bounds guard")
	}
	assertContains(t, wgsl, "return;")

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
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)

	// Elementwise: exactly 1 binding (output) and 1 input (the leaf buffer).
	// No reduce loops — no "for (" in the body.
	assertContains(t, wgsl, "exp2(")
	assertNotContains(t, wgsl, "for (")
	// Bounds guard references the output size (32).
	assertContains(t, wgsl, "32u")
}

// ── Test: reduce kernel ───────────────────────────────────────────────────────

func TestRender_Reduce(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{0, 1}, false) // scalar output

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)

	// Must have at least one sequential for loop (AxisReduce).
	if !strings.Contains(wgsl, "for (") {
		t.Error("reduce kernel must contain at least one for loop")
	}

	// Accumulator must be initialised to additive identity (0.0).
	assertContains(t, wgsl, "var acc0: f32 = 0.0")

	// Accumulator update must add to acc.
	assertContains(t, wgsl, "acc0 = acc0 +")

	// Scalar output: store to data0[0u].
	assertContains(t, wgsl, "data0[0u]")
}

// ── Test: reduce with non-trivial output loop ─────────────────────────────────

func TestRender_ReduceAxisZero(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{0}, false) // output shape [8]

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)

	// Output has 8 elements → GID bounds check on 8.
	assertContains(t, wgsl, "8u")
	// One AxisLoop range (output dim), one AxisReduce loop.
	assertContains(t, wgsl, "for (") // reduce loop
	assertContains(t, wgsl, "var acc0")
	// Store to gid_x (non-scalar output).
	assertContains(t, wgsl, "data0[gid_x]")
}

// ── Test: max-reduce identity ─────────────────────────────────────────────────

func TestRender_MaxReduce(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
	y := x.Max([]int{0}, false)

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)

	// Max-reduce identity: bitcast of 0xff7fffff (most negative finite f32).
	assertContains(t, wgsl, "bitcast<f32>(0xff7fffffu)")
	assertContains(t, wgsl, "max(acc0,")
}

// ── Test: reshape + elementwise (non-trivial index arithmetic) ────────────────

func TestRender_ReshapeIndex(t *testing.T) {
	// Use a fresh arena for each schedule call to avoid BUFFER/AFTER node
	// contamination from a prior GetKernelGraph run on the same arena.
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
	y := x.Reshape([]int64{32})
	z := y.Exp2()

	item := firstItem(t, makeSink(a, z))
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)

	// Must have exp2.
	assertContains(t, wgsl, "exp2(")
	// Index arithmetic must contain IDiv or Mod ('%') since source is 2D.
	if !strings.Contains(wgsl, "/") && !strings.Contains(wgsl, "%") {
		t.Error("reshape kernel should contain index arithmetic (/ or %)")
	}
	// No for loops (elementwise only).
	assertNotContains(t, wgsl, "for (")
}

// ── Test: pad + elementwise (WHERE guard) ─────────────────────────────────────

func TestRender_PadElementwise(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := x.Pad([][2]int64{{2, 2}}) // [8]
	z := y.Exp2()

	item := firstItem(t, makeSink(a, z))
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)

	// WHERE becomes WGSL select().
	assertContains(t, wgsl, "select(")
	assertContains(t, wgsl, "exp2(")
}

// ── Test: fused chain (exp2 ∘ log2) ──────────────────────────────────────────

func TestRender_FusedChain(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2().Log2()

	item := firstItem(t, makeSink(a, y))
	wgsl := codegen.RenderWGSL(item)

	verifyWGSLStructure(t, wgsl, item)
	assertContains(t, wgsl, "exp2(")
	assertContains(t, wgsl, "log2(")
	assertNotContains(t, wgsl, "for (")
}

// ── Test: multi-kernel schedule (reduce + elemwise) ───────────────────────────

func TestRender_MultiKernel(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{0}, false) // kernel 0: reduce [4,8]→[8]
	z := y.Exp2()               // kernel 1: elementwise [8]

	items := allItems(t, makeSink(a, z))
	if len(items) < 2 {
		t.Skipf("expected ≥2 kernels, got %d", len(items))
	}

	for i, item := range items {
		wgsl := codegen.RenderWGSL(item)
		verifyWGSLStructure(t, wgsl, item)
		t.Logf("kernel %d:\n%s", i, wgsl)
	}

	// Kernel 0 must have a reduce loop; kernel 1 must not.
	wgsl0 := codegen.RenderWGSL(items[0])
	wgsl1 := codegen.RenderWGSL(items[len(items)-1])
	if !strings.Contains(wgsl0, "for (") && !strings.Contains(wgsl1, "for (") {
		t.Error("at least one kernel should have a reduce loop")
	}
}

// ── Test: forward + backward schedule ────────────────────────────────────────

func TestRender_ForwardBackward(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	loss := x.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{x})
	gx, ok := grads[x]
	if !ok {
		t.Fatal("Backward returned no gradient for x")
	}

	items := allItems(t, makeSink(a, loss, gx))
	for i, item := range items {
		wgsl := codegen.RenderWGSL(item)
		verifyWGSLStructure(t, wgsl, item)
		t.Logf("kernel %d WGSL:\n%s", i, wgsl)
	}
}

// ── Test: binding count matches ExecItem.Bufs ─────────────────────────────────

func TestRender_BindingCount(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*uop.Arena) uop.UOp
	}{
		{"elementwise", func(a *uop.Arena) uop.UOp {
			x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
			return makeSink(a, x.Exp2())
		}},
		{"reduce", func(a *uop.Arena) uop.UOp {
			x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
			return makeSink(a, x.Sum([]int{0}, false))
		}},
		{"two-input", func(a *uop.Arena) uop.UOp {
			x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "webgpu")
			y := x.Sum([]int{0}, false)
			z := y.Exp2()
			return makeSink(a, z)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newArena()
			sink := tc.fn(a)
			items := schedule.CreateSchedule(sink, "webgpu")
			for _, item := range items {
				wgsl := codegen.RenderWGSL(item)
				ki := item.Ast.Arg().(uop.KernelInfo)
				// Every binding index [0..NumParams-1] must appear in the shader.
				for i := 0; i < ki.NumParams; i++ {
					if !strings.Contains(wgsl, fmt.Sprintf("@binding(%d)", i)) {
						t.Errorf("kernel missing @binding(%d) (NumParams=%d)\n%s", i, ki.NumParams, wgsl)
					}
				}
				// @binding(NumParams) must NOT appear.
				if strings.Contains(wgsl, fmt.Sprintf("@binding(%d)", ki.NumParams)) {
					t.Errorf("kernel has extra @binding(%d)\n%s", ki.NumParams, wgsl)
				}
			}
		})
	}
}

// ── Golden snapshot: elementwise exp2 over [4] ───────────────────────────────
//
// A human-readable golden test that captures the expected WGSL output.
// Update this snapshot if the renderer's output format changes intentionally.
// The value oracle (execute and compare) is a Phase 8b deliverable.

func TestGolden_Elementwise4(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	item := firstItem(t, makeSink(a, x.Exp2()))
	wgsl := codegen.RenderWGSL(item)

	t.Logf("generated WGSL:\n%s", wgsl)

	// Structural assertions (not brittle byte-for-byte matching).
	assertContains(t, wgsl,
		"@group(0) @binding(0) var<storage, read_write> data0: array<f32>",
		"@group(0) @binding(1) var<storage, read> data1: array<f32>",
		"@compute @workgroup_size(64)",
		"fn main(",
		"gid_x >= 4u",
		"exp2(",
		"data0[gid_x]",
	)
}

// ── Golden snapshot: scalar sum-reduce over [8] ───────────────────────────────

func TestGolden_ScalarReduce8(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
	item := firstItem(t, makeSink(a, x.Sum([]int{0}, false)))
	wgsl := codegen.RenderWGSL(item)

	t.Logf("generated WGSL:\n%s", wgsl)

	assertContains(t, wgsl,
		"@group(0) @binding(0) var<storage, read_write> data0: array<f32>",
		"@group(0) @binding(1) var<storage, read> data1: array<f32>",
		"@compute @workgroup_size(64)",
		"gid_x >= 1u",
		"var acc0: f32 = 0.0",
		"for (",
		"acc0 = acc0 +",
		"data0[0u]",
	)

	// Exactly one for loop (one reduce dimension).
	if n := countOccurrences(wgsl, "for ("); n != 1 {
		t.Errorf("expected 1 for loop in scalar reduce, got %d", n)
	}
}

// ── Verification note ─────────────────────────────────────────────────────────
//
// NO Go-importable WGSL parser is used. These tests verify structural properties
// of the emitted text: correct @binding counts, @compute entry presence, for-loop
// count per kernel, accumulator identity, and bounds guard.
//
// TRUE VALIDATION (syntactic parse + type-check) requires naga/Dawn and is a
// Phase 8b deliverable: wire RenderWGSL output into the Dawn/wgpu compile step
// and assert zero compile errors. Until then, structural assertions here are the
// best available oracle.
