package webgpu_test

// Option B Slice 6 — INV-A inclusive-bounds edge-value test.
//
// Declares a DefineVar with (min, max) inclusive bounds and binds the var
// to each endpoint (n=min and n=max). Both bindings must produce bit-exact
// results vs CPU reference. Today this is untested at the edge: existing
// dynamic-batch tests bind values strictly below max, so an off-by-one in
// any consumer that read max exclusively would not surface. This test
// closes that gap.
//
// SPEC §10: "DefineVar bounds are inclusive on both ends."

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestSlice6_InclusiveBoundsEdgeValue(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const device = "webgpu"
	const minBound = int64(1)
	const maxBound = int64(7)

	a := uop.NewArena(128)
	ta := tensor.NewSymbolicInput(a, "n", minBound, maxBound, uop.Dtypes.Float32, device)
	tb := tensor.NewSymbolicInput(a, "n", minBound, maxBound, uop.Dtypes.Float32, device)

	check := func(t *testing.T, nBound int64) {
		t.Helper()
		aData := make([]float32, nBound)
		bData := make([]float32, nBound)
		for i := range aData {
			aData[i] = float32(i)*0.5 + 0.25
			bData[i] = float32(i)*1.5 + 0.75
		}
		ta.SetData(aData)
		tb.SetData(bData)

		tc := ta.Add(tb)
		if err := tensor.RealizeWithBinding(map[string]int64{"n": nBound}, tc); err != nil {
			t.Fatalf("RealizeWithBinding(n=%d): %v", nBound, err)
		}
		got := tc.Data()
		if int64(len(got)) != nBound {
			t.Fatalf("n=%d output length = %d, want %d", nBound, len(got), nBound)
		}
		var maxErr float64
		for i := range got {
			want := aData[i] + bData[i]
			if e := math.Abs(float64(got[i] - want)); e > maxErr {
				maxErr = e
			}
		}
		if maxErr != 0 {
			t.Errorf("FAIL: n=%d max abs diff %g != 0 (inclusive-bounds edge regression)", nBound, maxErr)
		}
		t.Logf("n=%d (binding=%d): max abs diff %g (want 0)", nBound, nBound, maxErr)
	}

	t.Run("max_inclusive", func(t *testing.T) { check(t, maxBound) })
	t.Run("min_inclusive", func(t *testing.T) { check(t, minBound) })
}
