package cpu_test

// CPU backend Slice 2 test suite - full UOp coverage beyond the Slice 1
// MLP/conv surface, per notes/cpu_slice2_preflight.md:
//
//   - pad / shrink / permute / expand movement (rangeify-dissolved index
//     arithmetic, incl. the OpAnd validity-mask chain pad emits)
//   - non-contiguous (mid-axis) reductions, sum and max
//   - gather forward and scatter-add backward (collision fixture), riding
//     the tensor-layer host-sort preprocessing
//   - f16 / bf16 / fp8 quantized-f32 storage: quantize on upload and at the
//     kernel store boundary, asserted BIT-EXACT against the uop.DType.Quantize
//     oracle (the same algorithm the GPU's store helpers mirror)
//   - int32 leaf upload (bits-in-f32 contract, gather's idx path)

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func useCPU(t *testing.T) {
	t.Helper()
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() {
		tensor.DefaultExecutor = prev
		dev.Close()
	})
}

func i32Bits(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

func wantF32s(t *testing.T, got, want []float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len got %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("%s[%d]: got %v want %v", label, i, got[i], want[i])
		}
	}
}

func realizeOrFatal(t *testing.T, out *tensor.Tensor, label string) []float32 {
	t.Helper()
	if err := tensor.Realize(out); err != nil {
		t.Fatalf("%s Realize: %v", label, err)
	}
	return out.Data()
}

// ── Movement ops ──────────────────────────────────────────────────────────────

// TestCPU_Pad exercises the OpAnd validity-mask chain that pad lowers to
// (schedule/index.go) - the op Slice 1 lacked. Pads on both dims so the
// mask is a two-term conjunction.
func TestCPU_Pad(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{2, 2}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	got := realizeOrFatal(t, x.Pad([][2]int64{{1, 0}, {0, 1}}).Contiguous(), "pad")
	wantF32s(t, got, []float32{0, 0, 0, 1, 2, 0, 3, 4, 0}, "pad")
}

// TestCPU_PadBothSides pads before and after on both dims (four mask terms).
func TestCPU_PadBothSides(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{1, 2}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{5, 6})
	got := realizeOrFatal(t, x.Pad([][2]int64{{1, 1}, {1, 1}}).Contiguous(), "pad2")
	wantF32s(t, got, []float32{
		0, 0, 0, 0,
		0, 5, 6, 0,
		0, 0, 0, 0,
	}, "pad2")
}

func TestCPU_Shrink(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{3, 3}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	got := realizeOrFatal(t, x.Shrink([][2]int64{{1, 3}, {0, 2}}).Contiguous(), "shrink")
	wantF32s(t, got, []float32{4, 5, 7, 8}, "shrink")
}

func TestCPU_PermuteExpand(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4, 5, 6})
	got := realizeOrFatal(t, x.Permute([]int{1, 0}).Contiguous(), "permute")
	wantF32s(t, got, []float32{1, 4, 2, 5, 3, 6}, "permute")

	b := uop.NewArena(1024)
	y := tensor.NewLeaf(b, []int64{1, 3}, uop.Dtypes.Float32, "cpu")
	y.SetData([]float32{7, 8, 9})
	got2 := realizeOrFatal(t, y.Expand([]int64{2, 3}).Contiguous(), "expand")
	wantF32s(t, got2, []float32{7, 8, 9, 7, 8, 9}, "expand")
}

// TestCPU_PadShrinkCompose chains pad → shrink → permute in one graph so the
// dissolved index arithmetic composes across movement ops.
func TestCPU_PadShrinkCompose(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(2048)
	x := tensor.NewLeaf(a, []int64{2, 2}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	// pad → [3,3] {{0,0,0},{1,2,0},{3,4,0}}; shrink rows 1:3 cols 0:2 →
	// {{1,2},{3,4}}; permute → {{1,3},{2,4}}.
	out := x.Pad([][2]int64{{1, 0}, {0, 1}}).
		Shrink([][2]int64{{1, 3}, {0, 2}}).
		Permute([]int{1, 0}).Contiguous()
	got := realizeOrFatal(t, out, "compose")
	wantF32s(t, got, []float32{1, 3, 2, 4}, "compose")
}

// ── Non-contiguous-axis reductions ────────────────────────────────────────────

func TestCPU_MidAxisSumReduce(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{2, 3, 4}, uop.Dtypes.Float32, "cpu")
	d := make([]float32, 24)
	for i := range d {
		d[i] = float32(i)
	}
	x.SetData(d)
	got := realizeOrFatal(t, x.Sum([]int{1}, false), "midsum")
	want := make([]float32, 8)
	for i := 0; i < 2; i++ {
		for k := 0; k < 4; k++ {
			var s float32
			for j := 0; j < 3; j++ {
				s += d[i*12+j*4+k]
			}
			want[i*4+k] = s
		}
	}
	wantF32s(t, got, want, "midsum")
}

func TestCPU_MidAxisMaxReduce(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{2, 3, 2}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{
		1, -2, 9, 0, 3, 4, // batch 0: max over axis1 = {9, 4}
		-5, -6, -1, -9, -3, -7, // batch 1: max = {-1, -6}
	})
	got := realizeOrFatal(t, x.Max([]int{1}, false), "midmax")
	wantF32s(t, got, []float32{9, 4, -1, -6}, "midmax")
}

// ── Gather / scatter ──────────────────────────────────────────────────────────

func TestCPU_GatherForward(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(2048)
	w := tensor.NewLeaf(a, []int64{4, 3}, uop.Dtypes.Float32, "cpu")
	w.SetData([]float32{0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32})
	idx := tensor.NewLeaf(a, []int64{2}, uop.Dtypes.Int32, "cpu")
	idx.SetData(i32Bits([]int32{3, 1}))
	got := realizeOrFatal(t, w.Gather(0, idx), "gather")
	wantF32s(t, got, []float32{30, 31, 32, 10, 11, 12}, "gather")
}

// TestCPU_GatherBackward runs the scatter-add adjoint with an index
// collision; loss = sum(W.Gather(0, idx) * dY) makes the gather adjoint
// exactly dY (the fixture pattern from tensor/gather_backward_test.go).
func TestCPU_GatherBackward(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(4096)
	const V, D = 4, 2
	w := tensor.NewLeaf(a, []int64{V, D}, uop.Dtypes.Float32, "cpu")
	w.SetData(make([]float32, V*D))
	idx := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Int32, "cpu")
	idx.SetData(i32Bits([]int32{2, 0, 2})) // collision on row 2
	dy := tensor.NewLeaf(a, []int64{3, D}, uop.Dtypes.Float32, "cpu")
	dy.SetData([]float32{1, 2, 3, 4, 5, 6})

	loss := w.Gather(0, idx).Mul(dy).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{w})
	dW := grads[w]
	if dW == nil {
		t.Fatal("no gradient for W")
	}
	got := realizeOrFatal(t, dW, "scatter dW")
	// dW[2] = dY[0]+dY[2] = (6,8); dW[0] = dY[1] = (3,4).
	wantF32s(t, got, []float32{3, 4, 0, 0, 6, 8, 0, 0}, "scatter dW")
}

// ── Narrow-dtype storage (f16 / bf16 / fp8) ──────────────────────────────────

var narrowDtypes = []struct {
	name string
	dt   *uop.DType
}{
	{"f16", uop.Dtypes.Float16},
	{"bf16", uop.Dtypes.BFloat16},
	{"e4m3", uop.Dtypes.FP8E4M3},
	{"e5m2", uop.Dtypes.FP8E5M2},
}

// TestCPU_NarrowDtypeAdd asserts z = x + y over narrow-dtype buffers is
// BIT-EXACT against Q(Q(a)+Q(b)) - the contract shared with the GPU store
// helpers. For a single binary op this holds for all four dtypes (one f32
// add of on-grid values is exact, then one RTNE narrowing).
func TestCPU_NarrowDtypeAdd(t *testing.T) {
	for _, tc := range narrowDtypes {
		t.Run(tc.name, func(t *testing.T) {
			useCPU(t)
			aVals := []float32{1.5, -2.25, 0.1, 100, 0.001, -300}
			bVals := []float32{0.5, 1.25, 0.2, 300, 0.002, -300}

			a := uop.NewArena(1024)
			x := tensor.NewLeaf(a, []int64{6}, tc.dt, "cpu")
			x.SetData(append([]float32{}, aVals...))
			y := tensor.NewLeaf(a, []int64{6}, tc.dt, "cpu")
			y.SetData(append([]float32{}, bVals...))
			got := realizeOrFatal(t, x.Add(y), "narrow add")

			q := tc.dt.Quantize
			for i := range got {
				want := q(q(aVals[i]) + q(bVals[i]))
				if math.Float32bits(got[i]) != math.Float32bits(want) {
					t.Errorf("%s add[%d]: got %v want %v (bit-exact)", tc.name, i, got[i], want)
				}
			}
		})
	}
}

// TestCPU_NarrowDtypeSumReduce uses small-integer inputs (exact in f32/f64
// under any accumulation order) so the reduce result is deterministic:
// quantize(sum of quantized inputs), bit-exact.
func TestCPU_NarrowDtypeSumReduce(t *testing.T) {
	for _, tc := range narrowDtypes {
		t.Run(tc.name, func(t *testing.T) {
			useCPU(t)
			const n = 64
			vals := make([]float32, n)
			var sum float32
			for i := range vals {
				vals[i] = float32(i%9 - 4) // integers in [-4,4], exact on all grids
				sum += vals[i]
			}
			a := uop.NewArena(1024)
			x := tensor.NewLeaf(a, []int64{n}, tc.dt, "cpu")
			x.SetData(append([]float32{}, vals...))
			got := realizeOrFatal(t, x.Sum(nil, false), "narrow sum")
			want := tc.dt.Quantize(sum)
			if len(got) != 1 || math.Float32bits(got[0]) != math.Float32bits(want) {
				t.Errorf("%s sum: got %v want [%v] (bit-exact)", tc.name, got, want)
			}
		})
	}
}

// TestCPU_NarrowDtypeRoundTrip casts narrow → f32 → narrow over grid-exact
// values (incl. min/max subnormals); RTNE is the identity on the grid.
func TestCPU_NarrowDtypeRoundTrip(t *testing.T) {
	cases := map[string][]float32{
		"f16":  {0, 1, -1, 0.5, -3.5, 65504, 5.960464477539063e-08},
		"bf16": {0, 1, -1, 0.5, -3.5, 0.015625, 128},
		"e4m3": {0, 1, -1, 0.5, -3.5, 448, 0.001953125, 0.013671875},
		"e5m2": {0, 1, -1, 0.5, -3.5, 57344, 0.0000152587890625},
	}
	for _, tc := range narrowDtypes {
		t.Run(tc.name, func(t *testing.T) {
			useCPU(t)
			vals := cases[tc.name]
			a := uop.NewArena(1024)
			x := tensor.NewLeaf(a, []int64{int64(len(vals))}, tc.dt, "cpu")
			x.SetData(append([]float32{}, vals...))
			got := realizeOrFatal(t, x.Cast(uop.Dtypes.Float32).Cast(tc.dt), "roundtrip")
			for i, v := range got {
				want := tc.dt.Quantize(vals[i])
				if math.Float32bits(v) != math.Float32bits(want) {
					t.Errorf("%s roundtrip[%d]: got %v want %v", tc.name, i, v, want)
				}
			}
		})
	}
}

// TestCPU_NarrowDtypeMatmulBF16 runs a small bf16 matmul: f32/f64 compute
// with one narrowing at the store. Small-integer inputs keep every
// intermediate exact, so the result is Q(exact product sums), bit-exact.
func TestCPU_NarrowDtypeMatmulBF16(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(2048)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.BFloat16, "cpu")
	x.SetData([]float32{1, 2, 3, 4, 5, 6})
	w := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.BFloat16, "cpu")
	w.SetData([]float32{1, 2, 3, 4, 5, 6})
	got := realizeOrFatal(t, x.Matmul(w), "bf16 matmul")
	want := []float32{22, 28, 49, 64} // exact ints; Q(bf16) identity ≤ 256
	for i := range want {
		q := uop.Dtypes.BFloat16.Quantize(want[i])
		if math.Float32bits(got[i]) != math.Float32bits(q) {
			t.Errorf("bf16 matmul[%d]: got %v want %v", i, got[i], q)
		}
	}
}

// ── Unsupported paths stay fail-loud ─────────────────────────────────────────

// TestCPU_UnsupportedDtypeFailsLoud pins the allocator's error contract for
// dtypes outside the supported set (here: f64 buffers).
func TestCPU_UnsupportedDtypeFailsLoud(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float64, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	err := tensor.Realize(x.Add(x))
	if err == nil {
		t.Fatal("expected unsupported-dtype error for f64, got nil")
	}
	t.Logf("fail-loud (correct): %v", err)
}

// TestCPU_UnimplementedOpFailsLoud runs an op outside the interpreter's
// coverage (Sin) through the full tensor → schedule → cpu.Run pipeline and
// pins the wrapped "not yet implemented" error contract.
func TestCPU_UnimplementedOpFailsLoud(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	err := tensor.Realize(x.Sin())
	if err == nil {
		t.Fatal("expected unimplemented-op error for Sin, got nil")
	}
	t.Logf("fail-loud (correct): %v", err)
}

// TestCPU_SymbolicFailsLoud pins the static-only contract: a schedule with
// symbolic kernels must error cleanly, not run with garbage bounds.
func TestCPU_SymbolicFailsLoud(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(2048)
	x := tensor.NewSymbolicBatchInput(a, "n", 1, 8, []int64{3}, uop.Dtypes.Float32, "cpu")
	err := tensor.Realize(x.Add(x))
	if err == nil {
		t.Fatal("expected symbolic-kernel error, got nil")
	}
	t.Logf("fail-loud (correct): %v", err)
}

// TestCPU_OversizedUploadClamps pins the upload clamp: SetData longer than
// the leaf's shape must not overrun the buffer (mirrors WebGPU, which writes
// at most the allocated byte size). Exercises the clamp on both the i32 and
// narrow-float upload branches.
func TestCPU_OversizedUploadClamps(t *testing.T) {
	useCPU(t)
	a := uop.NewArena(2048)
	w := tensor.NewLeaf(a, []int64{4, 2}, uop.Dtypes.Float32, "cpu")
	w.SetData([]float32{0, 1, 10, 11, 20, 21, 30, 31})
	idx := tensor.NewLeaf(a, []int64{2}, uop.Dtypes.Int32, "cpu")
	idx.SetData(i32Bits([]int32{3, 1, 0, 0})) // 4 values into a [2] leaf
	got := realizeOrFatal(t, w.Gather(0, idx), "clamped gather")
	wantF32s(t, got, []float32{30, 31, 10, 11}, "clamped gather")

	b := uop.NewArena(1024)
	x := tensor.NewLeaf(b, []int64{2}, uop.Dtypes.BFloat16, "cpu")
	x.SetData([]float32{1, 2, 3, 4}) // 4 values into a [2] leaf
	got2 := realizeOrFatal(t, x.Add(x), "clamped bf16")
	wantF32s(t, got2, []float32{2, 4}, "clamped bf16")
}

// TestCPU_EmptyRun pins Run's nil-schedule early return.
func TestCPU_EmptyRun(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	out, err := dev.Run(nil, nil)
	if err != nil || out != nil {
		t.Errorf("empty Run: out=%v err=%v, want nil/nil", out, err)
	}
}
