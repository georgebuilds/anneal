package webgpu_test

// Multi-dim symbolic dispatch - regression test for OptLocal on a [n, m]
// elementwise kernel. After this slice the lowerer no longer collapses sym
// kernels into a single (dim=0, level=0) group; each axis lands in its own
// dispatch dim and the executor computes per-dim workgroup counts from the
// binding via SymKernelHandle.SymDispatch.
//
// The test is the value-oracle for "LOCAL works on multi-dim sym":
//   1. Build a [n, m] sym elementwise kernel.
//   2. Apply OptLocal(axis=1, L=8) - the inner axis, multi-of-L bindings only.
//   3. Run on the real GPU under several (n, m) bindings.
//   4. Compare to a CPU oracle; expect max-abs-diff 0.
//
// This is the permanent successor to codegen/slice7d_local_probe_test.go.

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/uop"
)

func TestOptLocalMultiDimSym(t *testing.T) {
	dev := requireDevice(t)

	// L=8 must divide m for LOCAL to be correct: the workgroup-bound stride
	// `(m + L - 1) / L * L` differs from m when L∤m, and the store index
	// uses the padded stride. Same constraint as the static-LOCAL path.
	const L = 8
	cases := []struct{ n, m int64 }{
		{3, 8},
		{2, 16},
		{4, 24},
	}

	for _, c := range cases {
		a := uop.NewArena(256)
		item := buildC1Kernel(a, "mn", "mm")

		// Apply LOCAL on the inner axis (axisIdx=1 = rM = "mm").
		item.SetAst(codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 1, Arg: L}))

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("OptLocal multi-dim n=%d m=%d compile: %v", c.n, c.m, err)
		}
		defer handle.Release()

		inA := make([]float32, c.n*c.m)
		inB := make([]float32, c.n*c.m)
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < c.m; j++ {
				inA[i*c.m+j] = float32(i*100 + j)
				inB[i*c.m+j] = float32(i*7 + j*13)
			}
		}

		binding := map[string]int64{"mn": c.n, "mm": c.m}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, c.n*c.m, [][]float32{inA, inB})
		if err != nil {
			t.Fatalf("OptLocal multi-dim n=%d m=%d dispatch: %v", c.n, c.m, err)
		}

		maxErr := 0.0
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < c.m; j++ {
				want := inA[i*c.m+j] + inB[i*c.m+j]
				got := out[i*c.m+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}
		t.Logf("OptLocal multi-dim [n=%d, m=%d, L=%d]  max-abs-diff=%.3e", c.n, c.m, L, maxErr)
		if maxErr != 0 {
			t.Errorf("OptLocal multi-dim [n=%d, m=%d] max-abs-diff=%g (expect 0)", c.n, c.m, maxErr)
		}
	}
}

// TestOptLocalMultiDimSymWGSL spot-checks the rendered WGSL after LOCAL on a
// multi-dim sym kernel: each axis gets its own per-axis guard, the inner
// (LOCAL'd) axis decomposes into workgroup + local components, and the
// store index uses the padded (workgroup × L) stride.
func TestOptLocalMultiDimSymWGSL(t *testing.T) {
	a := uop.NewArena(128)
	item := buildC1Kernel(a, "mn", "mm")
	item.SetAst(codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 1, Arg: 8}))

	wgsl := codegen.RenderWGSL(item).WGSL

	// The LOCAL'd inner axis should yield three let-bindings (wg, local, outer n).
	requireSubstring(t, wgsl, "let r1: i32 = i32(flat_wid_x);")
	requireSubstring(t, wgsl, "let r2: i32 = i32(lid.x % 8u);")
	requireSubstring(t, wgsl, "let r0: i32 = i32(flat_gid_y);")
	// Per-axis guards on the workgroup half (uses the ceil-div sym bound),
	// the local half (literal 8), and the outer n axis (sym bound n1).
	requireSubstring(t, wgsl, "if (r1 >= i32(((params_n.n0 + 7u) / 8u)))")
	requireSubstring(t, wgsl, "if (r2 >= 8) { return; }")
	requireSubstring(t, wgsl, "if (r0 >= i32(params_n.n1))")
}

// TestPerfMultiDimSymLocal characterizes the wall-clock cost of LOCAL on a
// representative multi-dim sym kernel: [n, m] elementwise add. Reports the
// min-of-N wall time for the baseline (no opt) and the LOCAL'd kernel, plus
// a honest "dispatch-bound vs compute-bound" verdict. Per SPEC §10's
// timing-harness contract, min-of-N is the only signal usable for pass/fail;
// this test logs and never errors so GPU contention can't flap.
func TestPerfMultiDimSymLocal(t *testing.T) {
	dev := requireDevice(t)

	// Larger m so the kernel is dispatch-bound (per-thread work is just an add,
	// memory bandwidth dominates). n=64, m=2048 → 131072 outputs.
	const n, m = int64(64), int64(2048)
	const L = 64 // multiple of m → divisibility constraint satisfied

	inA := make([]float32, n*m)
	inB := make([]float32, n*m)
	for i := range inA {
		inA[i] = float32(i)
		inB[i] = float32(2 * i)
	}
	binding := map[string]int64{"pn": n, "pm": m}

	timeMin := func(label string, applyLocal bool) time.Duration {
		a := uop.NewArena(256)
		item := buildC1Kernel(a, "pn", "pm")
		if applyLocal {
			item.SetAst(codegen.ApplyOpt(item.Ast, codegen.Opt{Kind: codegen.OptLocal, Axis: 1, Arg: L}))
		}
		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("%s compile: %v", label, err)
		}
		defer handle.Release()

		const warmup, iters = 3, 20
		for i := 0; i < warmup; i++ {
			if _, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, n*m, [][]float32{inA, inB}); err != nil {
				t.Fatalf("%s warmup: %v", label, err)
			}
		}
		minDur := time.Hour
		for i := 0; i < iters; i++ {
			start := time.Now()
			if _, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, n*m, [][]float32{inA, inB}); err != nil {
				t.Fatalf("%s iter %d: %v", label, i, err)
			}
			if d := time.Since(start); d < minDur {
				minDur = d
			}
		}
		t.Logf("%s [n=%d, m=%d, total=%d, iters=%d]  min=%v", label, n, m, n*m, iters, minDur)
		return minDur
	}

	tBase := timeMin("baseline (no opt)", false)
	tLocal := timeMin(fmt.Sprintf("OptLocal(axis=1, L=%d)", L), true)

	fmt.Printf("=== MULTI-DIM SYM LOCAL PERF [n=%d, m=%d, L=%d] ===\n", n, m, L)
	fmt.Printf("  baseline min:  %v\n", tBase)
	fmt.Printf("  local   min:   %v\n", tLocal)
	if tLocal < tBase {
		fmt.Printf("  speedup:       %.2fx\n", float64(tBase)/float64(tLocal))
	} else {
		fmt.Printf("  slowdown:      %.2fx (LOCAL did not help)\n", float64(tLocal)/float64(tBase))
	}
	// Buffer-allocation overhead dominates a single DispatchSymKernelWithBinding
	// call (the path re-creates GPU buffers per dispatch), so the absolute numbers
	// also include that bookkeeping. Both legs pay the same overhead - the
	// relative comparison still reflects the GPU dispatch cost.
}

func requireSubstring(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Errorf("expected WGSL to contain %q\n--- WGSL ---\n%s", needle, haystack)
	}
}
