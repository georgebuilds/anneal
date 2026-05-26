package webgpu_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// requireDevice opens a GPU device or skips the test with a clear message.
func requireDevice(t *testing.T) *webgpu.Device {
	t.Helper()
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU device available (%v) — build with CGO_ENABLED=0 and ensure Metal/Vulkan is present", err)
	}
	t.Cleanup(dev.Close)
	return dev
}

// approxEq reports whether two float32 slices are element-wise equal within tol.
func approxEq(got, want []float32, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			return false
		}
	}
	return true
}

// makeSchedule constructs a schedule from tensor nodes.
func makeSchedule(t *testing.T, device string, outputs ...*tensor.Tensor) []schedule.ExecItem {
	t.Helper()
	a := outputs[0].Arena()
	srcs := make([]uop.UOp, len(outputs))
	for i, ten := range outputs {
		srcs[i] = ten.Node()
	}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
	return schedule.CreateSchedule(sink, device)
}

// runSchedule executes items on the GPU and returns final output buffers.
func runSchedule(t *testing.T, dev *webgpu.Device, items []schedule.ExecItem, inputs map[uint32][]float32) map[uint32][]float32 {
	t.Helper()
	outputs, err := dev.Run(items, inputs)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	return outputs
}

// firstFinalOutput returns the data for the first final output buffer (in schedule order).
func firstFinalOutput(t *testing.T, items []schedule.ExecItem, outputs map[uint32][]float32) []float32 {
	t.Helper()
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}
	for _, item := range items {
		uid := item.Bufs[0].UOpIdx
		if !readByAny[uid] {
			data, ok := outputs[uid]
			if !ok {
				t.Fatalf("final output buffer %d missing from outputs map", uid)
			}
			return data
		}
	}
	t.Fatal("no final output buffer found in schedule")
	return nil
}

// ── Test: elementwise exp2 ────────────────────────────────────────────────────

func TestValueOracle_ElementwiseExp2(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()

	items := makeSchedule(t, "webgpu", y)
	inputs := map[uint32][]float32{
		x.Node().Index(): {0, 1, 2, 3},
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	want := []float32{1, 2, 4, 8}
	if !approxEq(got, want, 1e-5) {
		t.Fatalf("exp2([0,1,2,3]): got %v, want %v", got, want)
	}
	t.Logf("adapter: %s", dev.AdapterName())
}

// ── Test: elementwise add ─────────────────────────────────────────────────────

func TestValueOracle_ElementwiseAdd(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	z := x.Add(y)

	items := makeSchedule(t, "webgpu", z)
	inputs := map[uint32][]float32{
		x.Node().Index(): {1, 2, 3, 4},
		y.Node().Index(): {10, 20, 30, 40},
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	want := []float32{11, 22, 33, 44}
	if !approxEq(got, want, 1e-5) {
		t.Fatalf("add: got %v, want %v", got, want)
	}
}

// ── Test: scalar reduce (sum over all elements) ───────────────────────────────

func TestValueOracle_ScalarReduce(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum(nil, false) // sum all → scalar

	items := makeSchedule(t, "webgpu", y)
	inputs := map[uint32][]float32{
		x.Node().Index(): {1, 2, 3, 4, 5, 6, 7, 8},
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	if len(got) != 1 {
		t.Fatalf("sum: expected scalar output, got %d elements", len(got))
	}
	want := float32(36)
	if math.Abs(float64(got[0]-want)) > 1e-3 {
		t.Fatalf("sum([1..8]): got %v, want %v", got[0], want)
	}
}

// ── Test: axis-0 reduce (sum along rows) ─────────────────────────────────────

func TestValueOracle_AxisReduce(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{4, 4}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{0}, false) // sum along axis 0: [4,4] → [4]

	items := makeSchedule(t, "webgpu", y)
	data := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	inputs := map[uint32][]float32{x.Node().Index(): data}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	want := []float32{28, 32, 36, 40} // column sums
	if !approxEq(got, want, 1e-3) {
		t.Fatalf("sum(axis=0): got %v, want %v", got, want)
	}
}

// ── Test: max-reduce with all-negative input (identity-element correctness) ───
//
// If max-reduce uses 0 as identity (wrong) instead of -FLT_MAX (correct),
// all-negative inputs will return 0. This test catches that class of bug.
func TestValueOracle_MaxReduceNegativeInput(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := x.Max(nil, false) // global max → scalar

	items := makeSchedule(t, "webgpu", y)
	inputs := map[uint32][]float32{
		x.Node().Index(): {-5, -3, -1, -4}, // all negative
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	if len(got) != 1 {
		t.Fatalf("max: expected scalar, got %d elems", len(got))
	}
	want := float32(-1)
	if math.Abs(float64(got[0]-want)) > 1e-5 {
		t.Fatalf("max([-5,-3,-1,-4]): got %v, want %v (identity-element bug?)", got[0], want)
	}
}

// ── Test: fused exp2 + log2 chain ─────────────────────────────────────────────

func TestValueOracle_FusedChain(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	// log2(exp2(x)) == x for all finite x
	y := x.Exp2().Log2()

	items := makeSchedule(t, "webgpu", y)
	xdata := []float32{0.5, 1.0, 1.5, 2.0}
	inputs := map[uint32][]float32{x.Node().Index(): xdata}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	if !approxEq(got, xdata, 1e-5) {
		t.Fatalf("log2(exp2(x)): got %v, want %v", got, xdata)
	}
}

// ── Test: matrix multiplication [2,3] @ [3,2] → [2,2] ───────────────────────

func TestValueOracle_MatMul(t *testing.T) {
	dev := requireDevice(t)

	a := uop.NewArena(8192)
	A := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "webgpu")
	// C[i,j] = sum_k A[i,k] * B[k,j]
	// Broadcast: A[i,k] → [2,3,2], B[k,j] → [2,3,2], multiply, sum axis 1.
	Aexp := A.Reshape([]int64{2, 3, 1}).Expand([]int64{2, 3, 2})
	Bexp := B.Reshape([]int64{1, 3, 2}).Expand([]int64{2, 3, 2})
	C := Aexp.Mul(Bexp).Sum([]int{1}, false) // [2,3,2] → [2,2]

	items := makeSchedule(t, "webgpu", C)
	Adata := []float32{1, 2, 3, 4, 5, 6}   // [[1,2,3],[4,5,6]]
	Bdata := []float32{7, 8, 9, 10, 11, 12} // [[7,8],[9,10],[11,12]]
	inputs := map[uint32][]float32{
		A.Node().Index(): Adata,
		B.Node().Index(): Bdata,
	}
	outputs := runSchedule(t, dev, items, inputs)
	got := firstFinalOutput(t, items, outputs)
	// C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
	//   = [[58, 64], [139, 154]]
	want := []float32{58, 64, 139, 154}
	if !approxEq(got, want, 1e-2) {
		t.Fatalf("matmul: got %v, want %v", got, want)
	}
}

// ── Test: tensor.Realize() end-to-end integration ────────────────────────────

func TestRealize_Integration(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev

	a := uop.NewArena(4096)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	x.SetData([]float32{1, 2, 3, 4})
	y := x.Exp2()

	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	got := y.Data()
	want := []float32{2, 4, 8, 16}
	if !approxEq(got, want, 1e-5) {
		t.Fatalf("Realize(exp2([1,2,3,4])): got %v, want %v", got, want)
	}
}

// ── Test: MSE loss forward pass ───────────────────────────────────────────────
//
// Validates GPU forward values against hand-computed reference.
// This is the key proof that phases 7a+7b+8a+8b are all correct end-to-end.
func TestValueOracle_MSEForward(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev

	a := uop.NewArena(16384)
	// 1-layer linear: y = x @ W   x:[2,3], W:[3,2] → y:[2,2]
	// loss = mean((y - target)^2)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "webgpu")
	W := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "webgpu")
	target := tensor.NewLeaf(a, []int64{2, 2}, uop.Dtypes.Float32, "webgpu")

	x.SetData([]float32{1, 2, 3, 4, 5, 6})
	W.SetData([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	target.SetData([]float32{1.0, 2.0, 3.0, 4.0})

	xexp := x.Reshape([]int64{2, 3, 1}).Expand([]int64{2, 3, 2})
	Wexp := W.Reshape([]int64{1, 3, 2}).Expand([]int64{2, 3, 2})
	y := xexp.Mul(Wexp).Sum([]int{1}, false) // [2,2]

	diff := y.Sub(target)
	sq := diff.Mul(diff)
	loss := sq.Mean(nil, false)

	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("Realize: %v", err)
	}

	gotLoss := loss.Data()
	if len(gotLoss) != 1 {
		t.Fatalf("loss should be scalar, got %d elements", len(gotLoss))
	}

	// Reference:
	// y = [[1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6], [4*0.1+5*0.3+6*0.5, 4*0.2+5*0.4+6*0.6]]
	//   = [[2.2, 2.8], [4.9, 6.4]]
	// diff = [[1.2, 0.8], [1.9, 2.4]]
	// sq   = [[1.44, 0.64], [3.61, 5.76]]
	// mean = 11.45/4 = 2.8625
	wantLoss := float32(2.8625)
	if math.Abs(float64(gotLoss[0]-wantLoss)) > 1e-3 {
		t.Fatalf("MSE loss: GPU=%v, want=%v", gotLoss[0], wantLoss)
	}
	t.Logf("MSE loss: GPU=%.6f  expected=%.6f  ✓", gotLoss[0], wantLoss)
}

// ── Test: backward pass gradient value oracle ─────────────────────────────────
//
// Checks that GPU-executed gradient matches the analytical result.
// Proves the rewrite-based autodiff + scheduler + codegen + GPU pipeline is correct.
func TestValueOracle_BackwardGradient(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev

	a := uop.NewArena(8192)
	// f(x) = sum(exp2(x))  →  df/dx[i] = ln(2) * exp2(x[i])
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	x.SetData([]float32{0, 1, 2, 3})
	y := x.Exp2().Sum(nil, false) // scalar

	grads := tensor.Backward(y, []*tensor.Tensor{x})
	gx, ok := grads[x]
	if !ok {
		t.Fatal("gradient for x not in backward result")
	}
	if err := tensor.Realize(gx); err != nil {
		t.Fatalf("Realize(grad): %v", err)
	}

	gotGrad := gx.Data()
	// d/dx[i] sum(exp2(x)) = ln2 * exp2(x[i])
	ln2 := float32(math.Ln2)
	want := []float32{
		ln2 * 1, // exp2(0)=1
		ln2 * 2, // exp2(1)=2
		ln2 * 4, // exp2(2)=4
		ln2 * 8, // exp2(3)=8
	}
	if !approxEq(gotGrad, want, 1e-4) {
		t.Fatalf("grad of sum(exp2(x)): GPU=%v, want=%v", gotGrad, want)
	}
	t.Logf("grad: GPU=%v  expected=%v  ✓", gotGrad, want)
}

// ── Test: MSE backward FD oracle (matmul gradient path) ──────────────────────
//
// This is the real Phase 8b gate: prove that the matmul-gradient path
// (broadcast-mul-reduce backward) executes correctly on the GPU.
//
// The existing TestValueOracle_BackwardGradient uses sum(exp2(x)) to avoid
// matrix ops. This test uses a 1-layer linear (x@W) + MSE loss and checks
// the GPU-computed gradients of the loss w.r.t. x and W against a central
// finite-difference approximation of the same gradients.
func TestValueOracle_MSEBackwardFD(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev

	xData := []float32{1, 2, 3, 4, 5, 6}
	wData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	tgtData := []float32{1.0, 2.0, 3.0, 4.0}

	// Build forward graph and compute gradients via Phase 6 backward pass.
	a := uop.NewArena(65536)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "webgpu")
	W := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "webgpu")
	tgt := tensor.NewLeaf(a, []int64{2, 2}, uop.Dtypes.Float32, "webgpu")

	x.SetData(append([]float32{}, xData...))
	W.SetData(append([]float32{}, wData...))
	tgt.SetData(append([]float32{}, tgtData...))

	xexp := x.Reshape([]int64{2, 3, 1}).Expand([]int64{2, 3, 2})
	Wexp := W.Reshape([]int64{1, 3, 2}).Expand([]int64{2, 3, 2})
	y := xexp.Mul(Wexp).Sum([]int{1}, false) // [2,2]
	diff := y.Sub(tgt)
	sq := diff.Mul(diff)
	loss := sq.Mean(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{x, W})
	gx, okX := grads[x]
	gW, okW := grads[W]
	if !okX {
		t.Fatal("no gradient for x in Backward result")
	}
	if !okW {
		t.Fatal("no gradient for W in Backward result")
	}

	if err := tensor.Realize(gx); err != nil {
		t.Fatalf("Realize(gx): %v", err)
	}
	if err := tensor.Realize(gW); err != nil {
		t.Fatalf("Realize(gW): %v", err)
	}

	gpuGx := gx.Data()
	gpuGW := gW.Data()

	// Central finite-difference reference on GPU.
	eps := float32(1e-3)

	fdGx := make([]float32, len(xData))
	for i := range xData {
		xp := append([]float32{}, xData...)
		xp[i] += eps
		lossP := mseForwardLoss(t, dev, xp, wData, tgtData)

		xm := append([]float32{}, xData...)
		xm[i] -= eps
		lossM := mseForwardLoss(t, dev, xm, wData, tgtData)

		fdGx[i] = (lossP - lossM) / (2 * eps)
	}

	fdGW := make([]float32, len(wData))
	for i := range wData {
		wp := append([]float32{}, wData...)
		wp[i] += eps
		lossP := mseForwardLoss(t, dev, xData, wp, tgtData)

		wm := append([]float32{}, wData...)
		wm[i] -= eps
		lossM := mseForwardLoss(t, dev, xData, wm, tgtData)

		fdGW[i] = (lossP - lossM) / (2 * eps)
	}

	// Tolerance: 1e-2 is appropriate for float32 central differences over this
	// graph depth. Analytical reference: gx≈[0.14,0.34,0.54,0.335,0.765,1.195],
	// gW≈[4.4,5.2,5.95,6.8,7.5,8.4].
	tol := 1e-2
	if !approxEq(gpuGx, fdGx, tol) {
		t.Fatalf("gx mismatch — matmul backward incorrect:\n  GPU: %v\n   FD: %v", gpuGx, fdGx)
	}
	if !approxEq(gpuGW, fdGW, tol) {
		t.Fatalf("gW mismatch — matmul backward incorrect:\n  GPU: %v\n   FD: %v", gpuGW, fdGW)
	}
	t.Logf("gx: GPU=%v  FD=%v  ✓", gpuGx, fdGx)
	t.Logf("gW: GPU=%v  FD=%v  ✓", gpuGW, fdGW)
}

// mseForwardLoss builds a fresh arena, runs the 1-layer-linear + MSE forward
// pass on GPU, and returns the scalar loss. Used only for the FD oracle.
func mseForwardLoss(t *testing.T, dev *webgpu.Device, xData, wData, tgtData []float32) float32 {
	t.Helper()
	a := uop.NewArena(8192)
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "webgpu")
	W := tensor.NewLeaf(a, []int64{3, 2}, uop.Dtypes.Float32, "webgpu")
	tgt := tensor.NewLeaf(a, []int64{2, 2}, uop.Dtypes.Float32, "webgpu")

	x.SetData(append([]float32{}, xData...))
	W.SetData(append([]float32{}, wData...))
	tgt.SetData(append([]float32{}, tgtData...))

	xexp := x.Reshape([]int64{2, 3, 1}).Expand([]int64{2, 3, 2})
	Wexp := W.Reshape([]int64{1, 3, 2}).Expand([]int64{2, 3, 2})
	y := xexp.Mul(Wexp).Sum([]int{1}, false)
	diff := y.Sub(tgt)
	sq := diff.Mul(diff)
	loss := sq.Mean(nil, false)

	if err := tensor.Realize(loss); err != nil {
		t.Fatalf("mseForwardLoss Realize: %v", err)
	}
	return loss.Data()[0]
}
