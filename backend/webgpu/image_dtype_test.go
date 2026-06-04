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
// must produce numerically identical results — max-abs-diff == 0.
//
// SHAPE CONSTRAINT (image dtype slice 1): the OUTPUT row stride must be a
// multiple of 4. The flat-1D vec4 packing places adjacent rows on the same
// vec4 slot when row_stride %% 4 != 0; concurrent threads then race to
// write different lanes of the same slot (naga lowers per-component
// storage writes as RMW). A schedule that tiles by vec4 (one thread per
// slot, computing 4 outputs) would lift this constraint; that lives in a
// future slice. M=17 N=32 stays bit-exact because the output row stride
// (32) is vec4-aligned even though M is not.
//
// Test shapes:
//   - 64x64x64: aligned to vec4 in every dim (the canonical aligned case)
//   - 128x128x128: larger aligned baseline
//   - 17x32: M not divisible by 4 but N stride is — still bit-exact
func TestImageDType_ValueOracle_Matmul(t *testing.T) {
	dev := requireDevice(t)

	tests := []struct {
		name    string
		M, N, K int64
	}{
		{"matmul_64x64x64", 64, 64, 64},
		{"matmul_128x128x128", 128, 128, 128},
		{"matmul_irregular_M17_Nalign32", 17, 32, 32},
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
