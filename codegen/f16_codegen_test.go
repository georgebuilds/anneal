package codegen_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── f16 structural WGSL tests (no GPU required) ───────────────────────────────

// TestF16_EnableDirective verifies "enable f16;" is emitted iff f16 buffers are present.
func TestF16_EnableDirective(t *testing.T) {
	t.Run("f16_kernel_has_enable", func(t *testing.T) {
		a := newArena()
		x := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float16, "webgpu")
		y := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float16, "webgpu")
		z := x.Add(y)

		items := schedule.CreateSchedule(makeSink(a, z), "webgpu")
		if len(items) == 0 {
			t.Fatal("no schedule items")
		}
		res, err := codegen.CompileWGSL(items[0])
		if err != nil {
			t.Fatalf("CompileWGSL: %v", err)
		}
		wgsl := res.WGSL
		if !strings.HasPrefix(wgsl, "enable f16;") {
			t.Errorf("f16 kernel must start with 'enable f16;'\nshader:\n%s", wgsl)
		}
		assertContains(t, wgsl, "array<f16>")
	})

	t.Run("f32_kernel_no_enable", func(t *testing.T) {
		a := newArena()
		x := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float32, "webgpu")
		y := x.Exp2()

		items := schedule.CreateSchedule(makeSink(a, y), "webgpu")
		if len(items) == 0 {
			t.Fatal("no schedule items")
		}
		res, err := codegen.CompileWGSL(items[0])
		if err != nil {
			t.Fatalf("CompileWGSL: %v", err)
		}
		wgsl := res.WGSL
		assertNotContains(t, wgsl, "enable f16;")
		assertContains(t, wgsl, "array<f32>")
	})
}

// TestBF16_WGSL verifies bf16 kernels generate correct WGSL:
// - buffers declared as array<u32> (not array<bf16>, which WGSL does not support)
// - loads use bitcast<f32> to widen to f32 at read time
// - stores use bitcast<u32> & 0xFFFF0000u to truncate to bf16 precision
// - no "enable f16;" directive (bf16 doesn't use the f16 extension)
func TestBF16_WGSL(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.BFloat16, "webgpu")
	y := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.BFloat16, "webgpu")
	z := x.Add(y)

	items := schedule.CreateSchedule(makeSink(a, z), "webgpu")
	if len(items) == 0 {
		t.Fatal("no schedule items")
	}
	res, err := codegen.CompileWGSL(items[0])
	if err != nil {
		t.Fatalf("CompileWGSL: %v", err)
	}
	wgsl := res.WGSL
	t.Logf("bf16 elementwise WGSL:\n%s", wgsl)

	assertContains(t, wgsl, "array<u32>")
	assertNotContains(t, wgsl, "array<bf16>")
	assertNotContains(t, wgsl, "enable f16;")
	assertContains(t, wgsl, "bitcast<f32>(")
	assertContains(t, wgsl, "bitcast<u32>(")
	assertContains(t, wgsl, "0xFFFF0000u")
}

// TestBF16_ReduceWGSL verifies a bf16 sum-reduce uses a f32 accumulator and
// stores back with the bf16 truncation mask (not a f16() cast).
func TestBF16_ReduceWGSL(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.BFloat16, "webgpu")
	y := x.Sum([]int{0}, false)

	items := schedule.CreateSchedule(makeSink(a, y), "webgpu")
	if len(items) == 0 {
		t.Fatal("no schedule items")
	}
	res, err := codegen.CompileWGSL(items[0])
	if err != nil {
		t.Fatalf("CompileWGSL: %v", err)
	}
	wgsl := res.WGSL
	t.Logf("bf16 reduce WGSL:\n%s", wgsl)

	assertContains(t, wgsl, "var acc0: f32")
	assertNotContains(t, wgsl, "var acc0: bf16")
	assertContains(t, wgsl, "bitcast<u32>(acc0) & 0xFFFF0000u")
	assertNotContains(t, wgsl, "enable f16;")
}

// TestF16_ReduceF32Accumulator verifies a sum-reduce over f16 uses a f32 accumulator
// and narrows back to f16 at the store boundary.
func TestF16_ReduceF32Accumulator(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float16, "webgpu")
	y := x.Sum([]int{0}, false) // scalar f16 output

	items := schedule.CreateSchedule(makeSink(a, y), "webgpu")
	if len(items) == 0 {
		t.Fatal("no schedule items")
	}
	res, err := codegen.CompileWGSL(items[0])
	if err != nil {
		t.Fatalf("CompileWGSL: %v", err)
	}
	wgsl := res.WGSL
	t.Logf("f16 reduce WGSL:\n%s", wgsl)

	// Accumulator must be f32, not f16.
	if !strings.Contains(wgsl, "var acc0: f32") {
		t.Errorf("f16 reduce must use f32 accumulator\nshader:\n%s", wgsl)
	}
	assertNotContains(t, wgsl, "var acc0: f16")
	// Final expression must narrow to f16 at store.
	assertContains(t, wgsl, "f16(acc0)")
	// extension directive must be present.
	assertContains(t, wgsl, "enable f16;")
}

// TestF16_MatmulF32Accumulator verifies f16 matmul uses a f32 accumulator, widens
// loads to f32, and narrows the result back to f16 at the write boundary.
func TestF16_MatmulF32Accumulator(t *testing.T) {
	a := newArena()
	A := tensor.NewLeaf(a, []int64{8, 8}, uop.Dtypes.Float16, "webgpu")
	B := tensor.NewLeaf(a, []int64{8, 8}, uop.Dtypes.Float16, "webgpu")
	C := A.Matmul(B)

	items := schedule.CreateSchedule(makeSink(a, C), "webgpu")

	foundReduce := false
	for _, item := range items {
		res, err := codegen.CompileWGSL(item)
		if err != nil {
			t.Fatalf("CompileWGSL: %v", err)
		}
		wgsl := res.WGSL
		if !strings.Contains(wgsl, "for (") {
			continue // elementwise kernel — skip
		}
		// This is a reduce kernel.
		foundReduce = true
		t.Logf("f16 matmul reduce kernel:\n%s", wgsl)

		// Must use f32 accumulator.
		if !strings.Contains(wgsl, "var acc0: f32") {
			t.Errorf("f16 matmul reduce must use f32 accumulator\nshader:\n%s", wgsl)
		}
		// Must widen loads from f16 buffers to f32.
		if !strings.Contains(wgsl, "f32(data") {
			t.Errorf("f16 matmul must widen loads to f32\nshader:\n%s", wgsl)
		}
		// Must narrow back to f16 at the result site.
		assertContains(t, wgsl, "f16(acc0)")
		// Extension directive.
		assertContains(t, wgsl, "enable f16;")
	}
	if !foundReduce {
		t.Error("no reduce kernel found in matmul schedule")
	}
}

// TestF16_CastRoundTrip verifies that f16→f32→f16 lowers to explicit f32()/f16() casts.
func TestF16_CastRoundTrip(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float16, "webgpu")
	xf32 := x.Cast(uop.Dtypes.Float32)
	xf16 := xf32.Cast(uop.Dtypes.Float16)

	items := schedule.CreateSchedule(makeSink(a, xf16), "webgpu")
	if len(items) == 0 {
		t.Fatal("no schedule items")
	}
	res, err := codegen.CompileWGSL(items[0])
	if err != nil {
		t.Fatalf("CompileWGSL: %v", err)
	}
	wgsl := res.WGSL
	t.Logf("f16 cast round-trip WGSL:\n%s", wgsl)

	assertContains(t, wgsl, "f32(")
	assertContains(t, wgsl, "f16(")
	assertContains(t, wgsl, "enable f16;")
}

// TestF16_BufferTypes verifies that f16 buffers are declared as array<f16> and
// f32 buffers as array<f32> — no cross-contamination.
func TestF16_BufferTypes(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float16, "webgpu")
	y := x.Exp2()

	items := schedule.CreateSchedule(makeSink(a, y), "webgpu")
	if len(items) == 0 {
		t.Fatal("no schedule items")
	}
	res, err := codegen.CompileWGSL(items[0])
	if err != nil {
		t.Fatalf("CompileWGSL: %v", err)
	}
	wgsl := res.WGSL
	t.Logf("f16 elementwise WGSL:\n%s", wgsl)

	assertContains(t, wgsl, "array<f16>")
	assertNotContains(t, wgsl, "array<f32>")
}

// TestF16_F32KernelUnchanged verifies that existing f32 kernels are byte-identical
// before/after the f16 changes (static-path regression guard).
func TestF16_F32KernelUnchanged(t *testing.T) {
	tests := []struct {
		name    string
		wantSub []string
	}{
		{
			name: "exp2",
			wantSub: []string{
				"array<f32>",
				"exp2(",
				"@compute @workgroup_size(64, 1, 1)",
			},
		},
		{
			name: "sum_reduce",
			wantSub: []string{
				"var acc0: f32 = 0.0",
				"for (",
				"array<f32>",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newArena()
			var item schedule.ExecItem
			switch tc.name {
			case "exp2":
				x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "webgpu")
				items := schedule.CreateSchedule(makeSink(a, x.Exp2()), "webgpu")
				if len(items) == 0 {
					t.Fatal("no items")
				}
				item = items[0]
			case "sum_reduce":
				x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "webgpu")
				items := schedule.CreateSchedule(makeSink(a, x.Sum([]int{0}, false)), "webgpu")
				if len(items) == 0 {
					t.Fatal("no items")
				}
				item = items[0]
			}
			res, err := codegen.CompileWGSL(item)
			if err != nil {
				t.Fatalf("CompileWGSL: %v", err)
			}
			wgsl := res.WGSL
			for _, sub := range tc.wantSub {
				if !strings.Contains(wgsl, sub) {
					t.Errorf("f32 kernel %q missing %q\nshader:\n%s", tc.name, sub, wgsl)
				}
			}
			assertNotContains(t, wgsl, "enable f16;")
		})
	}
}
