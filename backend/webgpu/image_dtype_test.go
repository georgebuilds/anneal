package webgpu_test

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// buildImageInputs returns the inputs map for a single Mul-by-const kernel
// over an image-typed leaf.
func buildImageInputs(x *tensor.Tensor, data []float32) map[uint32][]float32 {
	return map[uint32][]float32{
		x.Node().Index(): data,
	}
}

// TestImageDType_ValueOracle_Matmul checks bit-exact agreement between a
// matmul over ImageFloat32 operands and the same matmul over Float32
// operands. Image storage is a layout choice for the GPU buffer (vec4
// packing); the compute path stays f32, so a correctly-implemented codegen
// must produce numerically identical results - max-abs-diff == 0.
//
// Vec4 slot dispatch (this slice) lifted the former SHAPE CONSTRAINT: image
// kernels now dispatch one thread per vec4 OUTPUT SLOT (four logical outputs
// per thread, whole slot written once), so output row strides that are NOT a
// multiple of 4 - where adjacent rows share a vec4 slot and the old per-lane
// store cascade raced under naga's Metal lowering - are bit-exact and
// REQUIRED to pass here.
//
// Test shapes:
//   - 64x64x64: aligned to vec4 in every dim (the canonical aligned case)
//   - 128x128x128: larger aligned baseline
//   - 17x32: M not divisible by 4 but N stride is - aligned-stride case
//   - 64x30: N=30 row stride - the documented race configuration, now required
//   - 33x17: N=17 row stride, odd everything
//   - 17x17x17: M, N, K all unaligned
func TestImageDType_ValueOracle_Matmul(t *testing.T) {
	dev := requireDevice(t)

	tests := []struct {
		name    string
		M, N, K int64
	}{
		{"matmul_64x64x64", 64, 64, 64},
		{"matmul_128x128x128", 128, 128, 128},
		{"matmul_irregular_M17_Nalign32", 17, 32, 32},
		{"matmul_unaligned_N30", 64, 30, 32},
		{"matmul_unaligned_N17", 33, 17, 32},
		{"matmul_unaligned_M17_N17_K17", 17, 17, 17},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Scalar f32 reference.
			aRef := uop.NewArena(65536)
			Aref := tensor.NewLeaf(aRef, []int64{tc.M, tc.K}, uop.Dtypes.Float32, "webgpu")
			Bref := tensor.NewLeaf(aRef, []int64{tc.K, tc.N}, uop.Dtypes.Float32, "webgpu")
			aData := uniformData(int(tc.M*tc.K), 1)
			bData := uniformData(int(tc.K*tc.N), 2)
			outRef := Aref.Matmul(Bref)
			itemsRef := schedule.CreateSchedule(makeSink(aRef, outRef), "webgpu")
			inputsRef := map[uint32][]float32{
				Aref.Node().Index(): aData,
				Bref.Node().Index(): bData,
			}
			resRef, err := dev.Run(itemsRef, inputsRef)
			if err != nil {
				t.Fatalf("scalar reference run failed: %v", err)
			}
			gotRef := firstFinalOutput(t, itemsRef, resRef)

			// ImageFloat32 candidate.
			aImg := uop.NewArena(65536)
			Aimg := tensor.NewLeaf(aImg, []int64{tc.M, tc.K}, uop.Dtypes.ImageFloat32, "webgpu")
			Bimg := tensor.NewLeaf(aImg, []int64{tc.K, tc.N}, uop.Dtypes.ImageFloat32, "webgpu")
			outImg := Aimg.Matmul(Bimg)
			itemsImg := schedule.CreateSchedule(makeSink(aImg, outImg), "webgpu")
			inputsImg := map[uint32][]float32{
				Aimg.Node().Index(): aData,
				Bimg.Node().Index(): bData,
			}
			resImg, err := dev.Run(itemsImg, inputsImg)
			if err != nil {
				t.Fatalf("image candidate run failed: %v", err)
			}
			gotImg := firstFinalOutput(t, itemsImg, resImg)

			if len(gotImg) != len(gotRef) {
				t.Fatalf("length mismatch: ref=%d img=%d", len(gotRef), len(gotImg))
			}
			var maxDiff float32
			var idx int
			for i := range gotRef {
				d := gotImg[i] - gotRef[i]
				if d < 0 {
					d = -d
				}
				if d > maxDiff {
					maxDiff = d
					idx = i
				}
			}
			t.Logf("%s: max-abs-diff=%g (at i=%d, ref=%g img=%g, n=%d)",
				tc.name, maxDiff, idx, gotRef[idx], gotImg[idx], len(gotRef))
			if maxDiff != 0 {
				t.Fatalf("image matmul not bit-exact vs scalar: max-abs-diff=%g (at i=%d)",
					maxDiff, idx)
			}
		})
	}
}

// TestImageDType_RoundTrip_Elementwise pins the load-store path: a copy-by-
// multiply-by-1.0 of an image-typed input must round-trip exactly. This is
// the smallest possible exercise of both the image-load and image-store
// codegen paths on the real GPU, and any tail-padding or vec4-component
// addressing bug would show up as a non-zero diff.
func TestImageDType_RoundTrip_Elementwise(t *testing.T) {
	dev := requireDevice(t)

	cases := []struct {
		name string
		n    int
	}{
		{"n_16_aligned", 16},
		{"n_64_aligned", 64},
		{"n_17_tail", 17}, // 5 vec4 slots; last has 1 active + 3 padding
		{"n_18_tail", 18},
		{"n_19_tail", 19},
		{"n_30_tail", 30}, // numel % 4 == 2 at elementwise scale
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := uop.NewArena(4096)
			x := tensor.NewLeaf(a, []int64{int64(tc.n)}, uop.Dtypes.ImageFloat32, "webgpu")
			data := uniformData(tc.n, 3)
			one := tensor.ConstScalar(a, 1.0, uop.Dtypes.ImageFloat32, "webgpu")
			out := x.Mul(one)
			items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
			res, err := dev.Run(items, buildImageInputs(x, data))
			if err != nil {
				t.Fatalf("run failed: %v", err)
			}
			got := firstFinalOutput(t, items, res)
			if len(got) != tc.n {
				t.Fatalf("output length: got %d want %d", len(got), tc.n)
			}
			var maxDiff float32
			for i := range data {
				d := got[i] - data[i]
				if d < 0 {
					d = -d
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
			t.Logf("%s: max-abs-diff=%g", tc.name, maxDiff)
			if maxDiff != 0 {
				t.Fatalf("image round-trip not bit-exact: max-abs-diff=%g", maxDiff)
			}
		})
	}
}

// TestImageDType_RoundTrip_Elementwise2D_UnalignedRows pins the slot-dispatch
// tail handling when vec4 slots straddle ROW boundaries: a [5,6] tensor has
// row stride 6, so slot 1 covers row-0 elements 4..5 plus row-1 elements
// 0..1. Under slot dispatch one thread owns the whole slot, so the straddle
// must round-trip exactly; under the old per-element dispatch this was the
// race configuration.
func TestImageDType_RoundTrip_Elementwise2D_UnalignedRows(t *testing.T) {
	dev := requireDevice(t)
	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{5, 6}, uop.Dtypes.ImageFloat32, "webgpu")
	data := uniformData(30, 7)
	one := tensor.ConstScalar(a, 1.0, uop.Dtypes.ImageFloat32, "webgpu")
	out := x.Mul(one)
	items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
	res, err := dev.Run(items, buildImageInputs(x, data))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	got := firstFinalOutput(t, items, res)
	if len(got) != 30 {
		t.Fatalf("output length: got %d want 30", len(got))
	}
	var maxDiff float32
	for i := range data {
		d := got[i] - data[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("elementwise_5x6: max-abs-diff=%g", maxDiff)
	if maxDiff != 0 {
		t.Fatalf("2D unaligned-row image round-trip not bit-exact: max-abs-diff=%g", maxDiff)
	}
}

// TestImageDType_ValueOracle_MatmulEpilogue checks the epilogue-fusion
// interaction (schedule Pass 5, removeBufferize): matmul + bias add over
// ImageFloat32 with an unaligned output row stride (N=30) must stay
// bit-exact vs the same graph in scalar Float32. If the add fuses into the
// image-output matmul kernel, the fused body is evaluated per lane inside
// the slot loop; if it stays a separate kernel, that kernel is itself an
// unaligned image-output elementwise. Either schedule must agree with the
// reference exactly.
func TestImageDType_ValueOracle_MatmulEpilogue(t *testing.T) {
	dev := requireDevice(t)
	const M, K, N = 17, 16, 30

	run := func(dt *uop.DType) []float32 {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{M, K}, dt, "webgpu")
		B := tensor.NewLeaf(a, []int64{K, N}, dt, "webgpu")
		C := tensor.NewLeaf(a, []int64{M, N}, dt, "webgpu")
		out := A.Matmul(B).Add(C)
		items := schedule.CreateSchedule(makeSink(a, out), "webgpu")
		inputs := map[uint32][]float32{
			A.Node().Index(): uniformData(M*K, 1),
			B.Node().Index(): uniformData(K*N, 2),
			C.Node().Index(): uniformData(M*N, 3),
		}
		res, err := dev.Run(items, inputs)
		if err != nil {
			t.Fatalf("run failed (dtype %v): %v", dt, err)
		}
		return firstFinalOutput(t, items, res)
	}

	gotRef := run(uop.Dtypes.Float32)
	gotImg := run(uop.Dtypes.ImageFloat32)
	if len(gotImg) != len(gotRef) {
		t.Fatalf("length mismatch: ref=%d img=%d", len(gotRef), len(gotImg))
	}
	var maxDiff float32
	var idx int
	for i := range gotRef {
		d := gotImg[i] - gotRef[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
			idx = i
		}
	}
	t.Logf("matmul_epilogue_17x16x30: max-abs-diff=%g (at i=%d)", maxDiff, idx)
	if maxDiff != 0 {
		t.Fatalf("image matmul+epilogue not bit-exact vs scalar: max-abs-diff=%g (at i=%d, ref=%g img=%g)",
			maxDiff, idx, gotRef[idx], gotImg[idx])
	}
}
