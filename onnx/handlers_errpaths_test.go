package onnx

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// errCtx builds a HandlerCtx with the given node attrs and inputs. A fresh
// arena is allocated per call so tests are independent.
func errCtx(inputs []Value, attrs map[string]Attr) *HandlerCtx {
	return &HandlerCtx{
		Arena:  uop.NewArena(64),
		Device: "test",
		Node:   &Node{Attrs: attrs},
		Inputs: inputs,
		Opset:  13,
	}
}

// floatLeaf builds a device-tier float32 leaf Value of the given shape.
func floatLeaf(t *testing.T, arena *uop.Arena, sh []int64) Value {
	t.Helper()
	x := tensor.NewLeaf(arena, sh, uop.Dtypes.Float32, "test")
	return Device(x)
}

// intDeviceLeaf builds a device-tier int32 leaf carrying the given data. Used
// to exercise the device-tier branches of resolveShapeInput / asHostIntVec.
func intDeviceLeaf(t *testing.T, arena *uop.Arena, vals []float32) Value {
	t.Helper()
	x := tensor.NewLeaf(arena, []int64{int64(len(vals))}, uop.Dtypes.Int32, "test")
	x.SetData(vals)
	return Device(x)
}

// ── oneTensorInput / twoTensorInputs error branches ──────────────────────────

func TestOneTensorInput_Errors(t *testing.T) {
	// no inputs
	if _, err := oneTensorInput(errCtx(nil, nil), "Sqrt"); err == nil {
		t.Fatal("want error for zero inputs")
	}
	// non-device input
	if _, err := oneTensorInput(errCtx([]Value{HostInt64(3)}, nil), "Sqrt"); err == nil {
		t.Fatal("want error for non-device input")
	}
}

func TestTwoTensorInputs_Errors(t *testing.T) {
	// too few inputs
	if _, _, err := twoTensorInputs(errCtx([]Value{HostInt64(1)}, nil), "Add"); err == nil {
		t.Fatal("want error for one input")
	}
	// second input non-device
	arena := uop.NewArena(8)
	in := []Value{floatLeaf(t, arena, []int64{2}), HostInt64(1)}
	if _, _, err := twoTensorInputs(errCtx(in, nil), "Add"); err == nil {
		t.Fatal("want error for non-device second input")
	}
}

// The elementwise handlers all funnel through the two helpers; assert each
// reports the error rather than panicking on a host input.
func TestElementwiseHandlers_BadInputs(t *testing.T) {
	bad1 := []Value{HostInt64(1)}
	one := []func(*HandlerCtx) ([]Value, error){
		handleSqrt, handleNeg, handleTanh, handleSigmoid, handleRelu,
		handleClip, handleErf,
	}
	for i, h := range one {
		if _, err := h(errCtx(bad1, nil)); err == nil {
			t.Errorf("one-input handler %d: want error on host input", i)
		}
	}
	bad2 := []Value{HostInt64(1)}
	two := []func(*HandlerCtx) ([]Value, error){
		handleAdd, handleSub, handleMul, handleDiv, handlePow, handleEqual,
		handleLess, handleGreater, handleLessOrEqual, handleGreaterOrEqual,
	}
	for i, h := range two {
		if _, err := h(errCtx(bad2, nil)); err == nil {
			t.Errorf("two-input handler %d: want error on single host input", i)
		}
	}
}

func TestHandleCast_UnsupportedDtype(t *testing.T) {
	arena := uop.NewArena(8)
	ctx := errCtx([]Value{floatLeaf(t, arena, []int64{2})},
		map[string]Attr{"to": {Kind: AttrInt, I: 99}})
	if _, err := handleCast(ctx); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("want unsupported-dtype error, got %v", err)
	}
}

// ── handleWhere ──────────────────────────────────────────────────────────────

func TestHandleWhere_Errors(t *testing.T) {
	// too few inputs
	if _, err := handleWhere(errCtx([]Value{HostInt64(1), HostInt64(2)}, nil)); err == nil {
		t.Fatal("want error for 2 inputs")
	}
	// a host input among the three
	arena := uop.NewArena(16)
	in := []Value{floatLeaf(t, arena, []int64{2}), floatLeaf(t, arena, []int64{2}), HostInt64(0)}
	if _, err := handleWhere(errCtx(in, nil)); err == nil {
		t.Fatal("want error for host input")
	}
}

func TestHandleWhere_CastsNonBoolCond(t *testing.T) {
	arena := uop.NewArena(32)
	cond := floatLeaf(t, arena, []int64{2}) // f32 cond → handler casts to bool
	x := floatLeaf(t, arena, []int64{2})
	y := floatLeaf(t, arena, []int64{2})
	out, err := handleWhere(errCtx([]Value{cond, x, y}, nil))
	if err != nil {
		t.Fatalf("handleWhere err=%v", err)
	}
	if len(out) != 1 || !out[0].IsDevice() {
		t.Fatalf("want single device output, got %v", out)
	}
}

// ── handleMaxPool guard rails ────────────────────────────────────────────────

func TestHandleMaxPool_Errors(t *testing.T) {
	arena := uop.NewArena(64)
	x4 := floatLeaf(t, arena, []int64{1, 1, 4, 4})
	ks := map[string]Attr{"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}}}

	cases := []struct {
		name   string
		inputs []Value
		attrs  map[string]Attr
		sub    string
	}{
		{"non-device", []Value{HostInt64(1)}, ks, "tensor"},
		{"rank3", []Value{floatLeaf(t, arena, []int64{1, 4, 4})}, ks, "rank"},
		{"auto_pad", []Value{x4}, map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"auto_pad":     {Kind: AttrString, S: "SAME_UPPER"}}, "auto_pad"},
		{"ceil_mode", []Value{x4}, map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"ceil_mode":    {Kind: AttrInt, I: 1}}, "ceil_mode"},
		{"storage_order", []Value{x4}, map[string]Attr{
			"kernel_shape":  {Kind: AttrInts, Is: []int64{2, 2}},
			"storage_order": {Kind: AttrInt, I: 1}}, "storage_order"},
		{"dilations", []Value{x4}, map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"dilations":    {Kind: AttrInts, Is: []int64{2, 2}}}, "dilations"},
		{"bad_kernel", []Value{x4}, map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2}}}, "kernel_shape"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := handleMaxPool(errCtx(c.inputs, c.attrs))
			if err == nil || !strings.Contains(err.Error(), c.sub) {
				t.Fatalf("err=%v, want substring %q", err, c.sub)
			}
		})
	}
}

func TestHandleMaxPool_AsymmetricPads(t *testing.T) {
	arena := uop.NewArena(128)
	x4 := floatLeaf(t, arena, []int64{1, 1, 4, 4})
	ctx := errCtx([]Value{x4}, map[string]Attr{
		"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
		"strides":      {Kind: AttrInts, Is: []int64{2, 2}},
		"pads":         {Kind: AttrInts, Is: []int64{1, 0, 0, 1}},
	})
	out, err := handleMaxPool(ctx)
	if err != nil {
		t.Fatalf("handleMaxPool err=%v", err)
	}
	if len(out) != 1 || !out[0].IsDevice() {
		t.Fatalf("want device output")
	}
}

func TestHandleGlobalAveragePool_Errors(t *testing.T) {
	if _, err := handleGlobalAveragePool(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for non-device input")
	}
	arena := uop.NewArena(16)
	x := floatLeaf(t, arena, []int64{2, 3}) // rank 2 < 3
	if _, err := handleGlobalAveragePool(errCtx([]Value{x}, nil)); err == nil {
		t.Fatal("want error for rank<3 input")
	}
}

// ── handleSoftmax error / axis branches ──────────────────────────────────────

func TestHandleSoftmax_Errors(t *testing.T) {
	// non-device
	if _, err := handleSoftmax(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for non-device input")
	}
	arena := uop.NewArena(16)
	// rank 0 scalar
	scalar := Device(tensor.NewLeaf(arena, []int64{}, uop.Dtypes.Float32, "test"))
	if _, err := handleSoftmax(errCtx([]Value{scalar}, nil)); err == nil {
		t.Fatal("want error for rank-0 input")
	}
	// axis out of range
	x := floatLeaf(t, arena, []int64{2, 3})
	ctx := errCtx([]Value{x}, map[string]Attr{"axis": {Kind: AttrInt, I: 5}})
	if _, err := handleSoftmax(ctx); err == nil {
		t.Fatal("want error for out-of-range axis")
	}
}

func TestHandleSoftmax_Opset12FlattenPath(t *testing.T) {
	arena := uop.NewArena(64)
	x := floatLeaf(t, arena, []int64{2, 3, 4})
	ctx := errCtx([]Value{x}, map[string]Attr{"axis": {Kind: AttrInt, I: 1}})
	ctx.Opset = 11 // < 13 → flatten branch
	out, err := handleSoftmax(ctx)
	if err != nil {
		t.Fatalf("softmax opset11 err=%v", err)
	}
	if len(out) != 1 || !out[0].IsDevice() {
		t.Fatalf("want device output")
	}
	// shape preserved.
	assertShape(t, out[0].Tensor(), []int64{2, 3, 4})
}

func TestHandleSoftmax_NegativeAxis(t *testing.T) {
	arena := uop.NewArena(64)
	x := floatLeaf(t, arena, []int64{2, 3})
	ctx := errCtx([]Value{x}, map[string]Attr{"axis": {Kind: AttrInt, I: -1}})
	out, err := handleSoftmax(ctx)
	if err != nil {
		t.Fatalf("softmax neg-axis err=%v", err)
	}
	assertShape(t, out[0].Tensor(), []int64{2, 3})
}

// ── handleReduceMin (and the reduce family error path) ───────────────────────

func TestHandleReduceMin_NonDevice(t *testing.T) {
	if _, err := handleReduceMin(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for non-device input")
	}
}

func TestHandleReduceMin_KeepdimsFalse(t *testing.T) {
	arena := uop.NewArena(64)
	x := floatLeaf(t, arena, []int64{2, 3})
	ctx := errCtx([]Value{x}, map[string]Attr{
		"axes":     {Kind: AttrInts, Is: []int64{1}},
		"keepdims": {Kind: AttrInt, I: 0},
	})
	out, err := handleReduceMin(ctx)
	if err != nil {
		t.Fatalf("reduceMin err=%v", err)
	}
	assertShape(t, out[0].Tensor(), []int64{2})
}

// ── handleExpand / broadcastTargetSints ──────────────────────────────────────

func TestHandleExpand_Errors(t *testing.T) {
	// too few inputs
	if _, err := handleExpand(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for one input")
	}
	// data not device
	in := []Value{HostInt64(1), HostInts([]int64{2, 3})}
	if _, err := handleExpand(errCtx(in, nil)); err == nil {
		t.Fatal("want error for non-device data")
	}
}

func TestHandleExpand_BroadcastShape(t *testing.T) {
	arena := uop.NewArena(64)
	x := Device(tensor.NewLeaf(arena, []int64{3, 1}, uop.Dtypes.Float32, "test"))
	target := HostInts([]int64{2, 1, 6})
	out, err := handleExpand(errCtx([]Value{x, target}, nil))
	if err != nil {
		t.Fatalf("expand err=%v", err)
	}
	// numpy broadcast of [3,1] and [2,1,6] = [2,3,6].
	assertShape(t, out[0].Tensor(), []int64{2, 3, 6})
}

// ── asHostIntVec ─────────────────────────────────────────────────────────────

func TestAsHostIntVec(t *testing.T) {
	if got, err := asHostIntVec(HostInt64(5)); err != nil || !int64SliceEq(got, []int64{5}) {
		t.Fatalf("scalar: got=%v err=%v", got, err)
	}
	if got, err := asHostIntVec(HostInts([]int64{1, 2})); err != nil || !int64SliceEq(got, []int64{1, 2}) {
		t.Fatalf("vec: got=%v err=%v", got, err)
	}
	// device int leaf with data
	arena := uop.NewArena(16)
	if got, err := asHostIntVec(intDeviceLeaf(t, arena, []float32{3, 4, 5})); err != nil || !int64SliceEq(got, []int64{3, 4, 5}) {
		t.Fatalf("device int leaf: got=%v err=%v", got, err)
	}
	// device non-int leaf
	if _, err := asHostIntVec(floatLeaf(t, arena, []int64{2})); err == nil {
		t.Fatal("want error for non-int device tensor")
	}
	// device int leaf with no host data
	noData := Device(tensor.NewLeaf(arena, []int64{2}, uop.Dtypes.Int32, "test"))
	if _, err := asHostIntVec(noData); err == nil {
		t.Fatal("want error for device tensor without host data")
	}
	// unsupported kind
	if _, err := asHostIntVec(HostFloat64(1.0)); err == nil {
		t.Fatal("want error for float kind")
	}
}

// ── resolveShapeInput device + error branches ────────────────────────────────

func TestResolveShapeInput(t *testing.T) {
	arena := uop.NewArena(16)
	// device int leaf
	sh, err := resolveShapeInput(intDeviceLeaf(t, arena, []float32{2, 3}))
	if err != nil {
		t.Fatalf("device int leaf err=%v", err)
	}
	if len(sh) != 2 {
		t.Fatalf("len=%d, want 2", len(sh))
	}
	if c, ok := sh[0].ConstValue(); !ok || c != 2 {
		t.Errorf("sh[0]=%v ok=%v, want 2", c, ok)
	}
	// device float leaf → non-int error
	if _, err := resolveShapeInput(floatLeaf(t, arena, []int64{2})); err == nil {
		t.Fatal("want error for non-int device tensor")
	}
	// device int leaf without host data
	noData := Device(tensor.NewLeaf(arena, []int64{2}, uop.Dtypes.Int32, "test"))
	if _, err := resolveShapeInput(noData); err == nil {
		t.Fatal("want error for device tensor without host data")
	}
	// unsupported kind: neither host nor device (zero Value).
	if _, err := resolveShapeInput(Value{}); err == nil {
		t.Fatal("want error for unsupported kind")
	}
}

// ── handleConstant / handleConstantOfShape / handleIdentity error paths ──────

func TestHandleConstant_MissingValue(t *testing.T) {
	if _, err := handleConstant(errCtx(nil, map[string]Attr{})); err == nil {
		t.Fatal("want error for missing value attr")
	}
}

func TestHandleIdentity_Errors(t *testing.T) {
	if _, err := handleIdentity(errCtx(nil, nil)); err == nil {
		t.Fatal("want error for zero inputs")
	}
	if _, err := handleIdentity(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for non-device input")
	}
}

func TestHandleConstantOfShape_Errors(t *testing.T) {
	// no inputs
	if _, err := handleConstantOfShape(errCtx(nil, nil)); err == nil {
		t.Fatal("want error for zero inputs")
	}
	// bad shape input (float device tensor)
	arena := uop.NewArena(16)
	bad := floatLeaf(t, arena, []int64{2})
	if _, err := handleConstantOfShape(errCtx([]Value{bad}, nil)); err == nil {
		t.Fatal("want error for non-int shape input")
	}
}

// ── reduce family (Sum/Mean/Max) error + axes-input branches ─────────────────

func TestReduceHandlers_NonDevice(t *testing.T) {
	for name, h := range map[string]func(*HandlerCtx) ([]Value, error){
		"ReduceSum":  handleReduceSum,
		"ReduceMean": handleReduceMean,
		"ReduceMax":  handleReduceMax,
	} {
		if _, err := h(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
			t.Errorf("%s: want error for non-device input", name)
		}
	}
}

func TestReduceHandlers_BadAxesInput(t *testing.T) {
	arena := uop.NewArena(32)
	x := floatLeaf(t, arena, []int64{2, 3})
	// axes input is a float device tensor → asHostIntVec errors.
	badAxes := floatLeaf(t, arena, []int64{1})
	ctx := errCtx([]Value{x, badAxes}, nil)
	if _, err := handleReduceSum(ctx); err == nil {
		t.Fatal("want error for non-int axes input")
	}
}

func TestReduceHandlers_AxesInputPath(t *testing.T) {
	arena := uop.NewArena(64)
	x := floatLeaf(t, arena, []int64{2, 3})
	axes := HostInts([]int64{1}) // opset-13 axes-as-input
	ctx := errCtx([]Value{x, axes}, map[string]Attr{"keepdims": {Kind: AttrInt, I: 0}})
	out, err := handleReduceMean(ctx)
	if err != nil {
		t.Fatalf("reduceMean axes-input err=%v", err)
	}
	assertShape(t, out[0].Tensor(), []int64{2})
}

// ── handleMatMul ─────────────────────────────────────────────────────────────

func TestHandleMatMul_Error(t *testing.T) {
	if _, err := handleMatMul(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for single host input")
	}
}

func TestHandleMatMul_OK(t *testing.T) {
	arena := uop.NewArena(64)
	a := floatLeaf(t, arena, []int64{2, 3})
	b := floatLeaf(t, arena, []int64{3, 4})
	out, err := handleMatMul(errCtx([]Value{a, b}, nil))
	if err != nil {
		t.Fatalf("matmul err=%v", err)
	}
	assertShape(t, out[0].Tensor(), []int64{2, 4})
}

// ── handleGather (device-tier) ───────────────────────────────────────────────

func TestHandleGather_Errors(t *testing.T) {
	// too few inputs
	if _, err := handleGather(errCtx([]Value{HostInt64(1)}, nil)); err == nil {
		t.Fatal("want error for one input")
	}
	// host inputs (this entry point requires device tensors)
	in := []Value{HostInts([]int64{1, 2}), HostInt64(0)}
	if _, err := handleGather(errCtx(in, nil)); err == nil {
		t.Fatal("want error for non-device inputs")
	}
}

func TestHandleGather_OK(t *testing.T) {
	arena := uop.NewArena(64)
	data := floatLeaf(t, arena, []int64{4, 2})
	idx := Device(tensor.NewLeaf(arena, []int64{3}, uop.Dtypes.Int32, "test"))
	ctx := errCtx([]Value{data, idx}, map[string]Attr{"axis": {Kind: AttrInt, I: 0}})
	out, err := handleGather(ctx)
	if err != nil {
		t.Fatalf("gather err=%v", err)
	}
	assertShape(t, out[0].Tensor(), []int64{3, 2})
}

// ── hostDiv divide-by-zero guard ─────────────────────────────────────────────

func TestHostDiv_ByZero(t *testing.T) {
	// hostDiv returns 0 for division by zero rather than panicking.
	v, err := hostDiv(&Node{}, []Value{HostInt64(10), HostInt64(0)}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if v.Int64() != 0 {
		t.Errorf("Div by zero=%d, want 0", v.Int64())
	}
}
