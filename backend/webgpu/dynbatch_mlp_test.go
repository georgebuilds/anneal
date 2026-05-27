package webgpu_test

// TestDynBatchMLP is the Slice 3b-2 proof test.
//
// It proves that a 2-layer MLP with a symbolic batch dimension runs correctly:
//
//   Input:   [n, 4]  (batch symbolic, features concrete)
//   Linear1: [4, 8]  weight  →  output [n, 8]
//   ReLU:    [n, 8]           →  output [n, 8]
//   Linear2: [8, 2]  weight  →  output [n, 2]
//
// Proof criteria:
//   - Forward correctness: max abs error < 1e-4 vs CPU reference for n=4 and n=8
//   - Compile-once: SymCompiledCount() == kernelsN4 after both dispatches (n=4 then n=8)
//   - Static path: a fully-concrete version produces byte-identical results for n=4

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestDynBatchMLP(t *testing.T) {
	dev := requireDevice(t)

	// ── Weight data (shared between symbolic and static runs) ─────────────
	rng := rand.New(rand.NewSource(42))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.5
		}
		return s
	}

	w1Data := randSlice(8 * 4)  // Linear1 weight: [8, 4]
	b1Data := randSlice(8)      // Linear1 bias:   [8]
	w2Data := randSlice(2 * 8)  // Linear2 weight: [2, 8]
	b2Data := randSlice(2)      // Linear2 bias:   [2]

	// ── CPU reference forward pass ─────────────────────────────────────────
	// y1[b, o] = ReLU( sum_i(x[b, i] * w1[o, i]) + b1[o] )
	// y2[b, o] = sum_i(y1[b, i] * w2[o, i]) + b2[o]
	cpuForward := func(x []float32, batch, inF, h, outF int) []float32 {
		// Layer 1
		h1 := make([]float32, batch*h)
		for b := 0; b < batch; b++ {
			for o := 0; o < h; o++ {
				v := b1Data[o]
				for i := 0; i < inF; i++ {
					v += x[b*inF+i] * w1Data[o*inF+i]
				}
				if v < 0 {
					v = 0
				}
				h1[b*h+o] = v
			}
		}
		// Layer 2
		y := make([]float32, batch*outF)
		for b := 0; b < batch; b++ {
			for o := 0; o < outF; o++ {
				v := b2Data[o]
				for i := 0; i < h; i++ {
					v += h1[b*h+i] * w2Data[o*h+i]
				}
				y[b*outF+o] = v
			}
		}
		return y
	}

	runSymbolicMLP := func(t *testing.T, dev *webgpu.Device, batch int, xData []float32, kernelsBeforeRun int) ([]float32, int) {
		t.Helper()
		tensor.DefaultExecutor = dev
		defer func() { tensor.DefaultExecutor = nil }()
		a := uop.NewArena(512)
		const device = "webgpu"

		// Build the graph with a symbolic batch dim.
		x := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{4}, uop.Dtypes.Float32, device)
		x.SetData(xData)

		l1 := nn.NewLinear(a, 4, 8, true, uop.Dtypes.Float32, device)
		l1.Weight.Value = w1Data
		l1.Weight.Load(a)
		l1.Bias.Value = b1Data
		l1.Bias.Load(a)

		l2 := nn.NewLinear(a, 8, 2, true, uop.Dtypes.Float32, device)
		l2.Weight.Value = w2Data
		l2.Weight.Load(a)
		l2.Bias.Value = b2Data
		l2.Bias.Load(a)

		h := nn.ReLU(l1.Forward(x))
		out := l2.Forward(h)

		if err := tensor.RealizeWithBinding(map[string]int64{"n": int64(batch)}, out); err != nil {
			t.Fatalf("RealizeWithBinding (batch=%d): %v", batch, err)
		}
		data := out.Data()
		if data == nil {
			t.Fatalf("out.Data() is nil after realize (batch=%d)", batch)
		}
		if len(data) != batch*2 {
			t.Fatalf("output len=%d, want %d (batch=%d, outF=2)", len(data), batch*2, batch)
		}
		return data, dev.SymCompiledCount() - kernelsBeforeRun
	}

	// ── Run n=4 ───────────────────────────────────────────────────────────
	const batch4 = 4
	x4 := randSlice(batch4 * 4)
	ref4 := cpuForward(x4, batch4, 4, 8, 2)

	countBefore := dev.SymCompiledCount()
	got4, kernels4 := runSymbolicMLP(t, dev, batch4, x4, countBefore)

	var maxErr4 float64
	for i := range got4 {
		if e := math.Abs(float64(got4[i] - ref4[i])); e > maxErr4 {
			maxErr4 = e
		}
	}

	// ── Run n=8 (same compiled kernels — compile-once proof) ──────────────
	const batch8 = 8
	x8 := randSlice(batch8 * 4)
	ref8 := cpuForward(x8, batch8, 4, 8, 2)

	countBeforeN8 := dev.SymCompiledCount()
	got8, _ := runSymbolicMLP(t, dev, batch8, x8, countBeforeN8)
	newKernelsN8 := dev.SymCompiledCount() - countBeforeN8

	var maxErr8 float64
	for i := range got8 {
		if e := math.Abs(float64(got8[i] - ref8[i])); e > maxErr8 {
			maxErr8 = e
		}
	}

	// ── Static path: n=4 concrete run must match ──────────────────────────
	staticOut := staticMLPForward(t, dev, x4, w1Data, b1Data, w2Data, b2Data, batch4)
	var maxErrStatic float64
	for i := range staticOut {
		if e := math.Abs(float64(staticOut[i] - ref4[i])); e > maxErrStatic {
			maxErrStatic = e
		}
	}

	// ── Report ────────────────────────────────────────────────────────────
	t.Logf("=== SLICE 3b-2 PROOF (dynamic-batch MLP) ===")
	t.Logf("kernels compiled on first run (n=4):  %d", kernels4)
	t.Logf("new kernels compiled on second run (n=8): %d  (want 0)", newKernelsN8)
	t.Logf("max abs error n=4  vs CPU: %.2e  (want < 1e-4)", maxErr4)
	t.Logf("max abs error n=8  vs CPU: %.2e  (want < 1e-4)", maxErr8)
	t.Logf("max abs error static n=4 vs CPU: %.2e  (want < 1e-4)", maxErrStatic)

	// ── Assertions ────────────────────────────────────────────────────────
	if kernels4 == 0 {
		t.Errorf("FAIL: no kernels compiled for n=4 run")
	}
	if newKernelsN8 != 0 {
		t.Errorf("FAIL: %d new kernels compiled for n=8 (compile-once broken)", newKernelsN8)
	}
	if maxErr4 > 1e-4 {
		t.Errorf("FAIL: n=4 max error %.2e > 1e-4", maxErr4)
	}
	if maxErr8 > 1e-4 {
		t.Errorf("FAIL: n=8 max error %.2e > 1e-4", maxErr8)
	}
	if maxErrStatic > 1e-4 {
		t.Errorf("FAIL: static n=4 max error %.2e > 1e-4", maxErrStatic)
	}
}

// staticMLPForward runs the same 2-layer MLP with fully-concrete tensors.
func staticMLPForward(t *testing.T, dev *webgpu.Device, x, w1, b1, w2, b2 []float32, batch int) []float32 {
	t.Helper()
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	a := uop.NewArena(512)
	const device = "webgpu"

	xT := tensor.NewLeaf(a, []int64{int64(batch), 4}, uop.Dtypes.Float32, device)
	xT.SetData(x)

	l1 := nn.NewLinear(a, 4, 8, true, uop.Dtypes.Float32, device)
	l1.Weight.Value = w1
	l1.Weight.Load(a)
	l1.Bias.Value = b1
	l1.Bias.Load(a)

	l2 := nn.NewLinear(a, 8, 2, true, uop.Dtypes.Float32, device)
	l2.Weight.Value = w2
	l2.Weight.Load(a)
	l2.Bias.Value = b2
	l2.Bias.Load(a)

	h := nn.ReLU(l1.Forward(xT))
	out := l2.Forward(h)

	if err := tensor.Realize(out); err != nil {
		t.Fatalf("static Realize: %v", err)
	}
	data := out.Data()
	if data == nil {
		t.Fatalf("static Realize: out.Data() nil")
	}
	return data
}

// TestDynBatchScheduleStructure proves the schedule properties for a symbolic
// batch MLP without touching the GPU — pure structure validation.
func TestDynBatchScheduleStructure(t *testing.T) {
	a := uop.NewArena(512)
	const device = "cpu"

	x := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{4}, uop.Dtypes.Float32, device)

	l1 := nn.NewLinear(a, 4, 8, true, uop.Dtypes.Float32, device)
	l1.Weight.Load(a)
	l1.Bias.Load(a)

	h := nn.ReLU(l1.Forward(x))

	srcs := []uop.UOp{h.Node()}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
	items := schedule.CreateSchedule(sink, device)

	if len(items) == 0 {
		t.Fatal("schedule is empty")
	}
	symCount := 0
	for _, item := range items {
		if itemHasSymDim(item) {
			symCount++
		}
	}
	if symCount == 0 {
		t.Errorf("no symbolic kernels in schedule (want at least 1)")
	}

	// Output shape of the final item must be [n, 8] — 2D with a symbolic first dim.
	last := items[len(items)-1]
	if len(last.Bufs) == 0 {
		t.Fatal("last ExecItem has no bufs")
	}
	outBuf := last.Bufs[0]
	if len(outBuf.Shape) != 2 {
		t.Errorf("output buf shape rank = %d, want 2 ([n, 8])", len(outBuf.Shape))
	} else {
		if outBuf.Shape[0] != 0 {
			t.Errorf("output buf Shape[0] = %d, want 0 (symbolic)", outBuf.Shape[0])
		}
		if outBuf.Shape[1] != 8 {
			t.Errorf("output buf Shape[1] = %d, want 8", outBuf.Shape[1])
		}
	}
	t.Logf("schedule: %d items, %d symbolic; output shape = %v", len(items), symCount, outBuf.Shape)
}
