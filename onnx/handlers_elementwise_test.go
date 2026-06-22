package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// elementwiseRig runs a single binary-op model on x=initializer, y=initializer
// and returns the output tensor. Values flow as leaf data; the handler's job
// is to emit the right primitive - we assert the root op and that the two
// operand leaves carry the input data unmodified (value oracle: the output
// computation is fully determined by (rootOp, leaf0, leaf1)).
func elementwiseRig(t *testing.T, opType string, xData, yData []float32, opset int64) *tensor.Tensor {
	t.Helper()
	if opset == 0 {
		opset = 13
	}
	b := &singleNodeBuilder{
		opType: opType,
		opset:  opset,
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{int64(len(xData))}},
			{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{int64(len(yData))}},
		},
		outputs: []nameInfo{{Name: "z", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{int64(len(xData))}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{int64(len(xData))}, xData),
			makeFloatInitializerForTests("y", []int64{int64(len(yData))}, yData),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	return outs["z"]
}

func TestHandleAdd(t *testing.T) {
	z := elementwiseRig(t, "Add", []float32{1, 2, 3, 4}, []float32{10, 20, 30, 40}, 13)
	if z.Node().Op() != uop.OpAdd {
		t.Fatalf("root op=%s, want Add", z.Node().Op())
	}
	// Operands must be the original leaves with intact data.
	a := z.Node().Src(0)
	b := z.Node().Src(1)
	aData, _ := a.Arena().Leaf(a.Index())
	bData, _ := b.Arena().Leaf(b.Index())
	wantA := []float32{1, 2, 3, 4}
	wantB := []float32{10, 20, 30, 40}
	for i := range wantA {
		if aData[i] != wantA[i] {
			t.Errorf("x[%d]=%v, want %v", i, aData[i], wantA[i])
		}
		if bData[i] != wantB[i] {
			t.Errorf("y[%d]=%v, want %v", i, bData[i], wantB[i])
		}
	}
}

func TestHandleSub(t *testing.T) {
	z := elementwiseRig(t, "Sub", []float32{10, 20, 30}, []float32{1, 2, 3}, 13)
	if z.Node().Op() != uop.OpSub {
		t.Fatalf("root op=%s, want Sub", z.Node().Op())
	}
}

func TestHandleMul(t *testing.T) {
	z := elementwiseRig(t, "Mul", []float32{2, 3, 4}, []float32{5, 6, 7}, 13)
	if z.Node().Op() != uop.OpMul {
		t.Fatalf("root op=%s, want Mul", z.Node().Op())
	}
}

func TestHandleDiv(t *testing.T) {
	z := elementwiseRig(t, "Div", []float32{10, 20, 30}, []float32{2, 4, 5}, 13)
	// Float Div composes as Mul(x, Reciprocal(y)); root op is Mul.
	if z.Node().Op() != uop.OpMul {
		t.Fatalf("root op=%s, want Mul (Div is x * recip(y) for float)", z.Node().Op())
	}
}

func TestHandlePow(t *testing.T) {
	z := elementwiseRig(t, "Pow", []float32{2, 3, 4}, []float32{2, 2, 2}, 13)
	// Pow = exp2(log2(x) * y) → root = Exp2
	if z.Node().Op() != uop.OpExp2 {
		t.Fatalf("root op=%s, want Exp2 (Pow expansion)", z.Node().Op())
	}
}

func TestHandleSqrt(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Sqrt",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{3}, []float32{4, 9, 16})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpSqrt {
		t.Fatalf("root op=%s, want Sqrt", y.Node().Op())
	}
}

func TestHandleNeg(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Neg",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{3}, []float32{1, -2, 3})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpNeg {
		t.Fatalf("root op=%s, want Neg", y.Node().Op())
	}
	// Walk to leaf and verify data.
	src := y.Node().Src(0)
	d, _ := src.Arena().Leaf(src.Index())
	want := []float32{1, -2, 3}
	for i := range want {
		if d[i] != want[i] {
			t.Errorf("x[%d]=%v, want %v", i, d[i], want[i])
		}
	}
}

func TestHandleRelu_Composition(t *testing.T) {
	// Relu composes as Maximum(x, 0). Root op is Max.
	b := &singleNodeBuilder{
		opType:       "Relu",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{4}, []float32{-1, 0, 1, 2})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpMax {
		t.Fatalf("root op=%s, want Max (Relu=Max(x,0))", y.Node().Op())
	}
}

func TestHandleSigmoid_Composition(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Sigmoid",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2}, []float32{0, 1})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	// Sigmoid = 1 / (1 + exp(-x)). Div composes as Mul(1, Recip(1+exp(-x))) → root = Mul.
	if y.Node().Op() != uop.OpMul {
		t.Fatalf("root op=%s, want Mul (Sigmoid via Div)", y.Node().Op())
	}
}

func TestHandleTanh_Composition(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Tanh",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2}, []float32{0, 1})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	// Tanh = 2*sigmoid(2x) - 1 → root = Sub
	if y.Node().Op() != uop.OpSub {
		t.Fatalf("root op=%s, want Sub (Tanh expansion ends in -1)", y.Node().Op())
	}
}

func TestHandleClip_Opset11(t *testing.T) {
	// min/max are inputs in opset 11+.
	b := &singleNodeBuilder{
		opType: "Clip",
		opset:  13,
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}},
			{Name: "min", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{}},
			{Name: "max", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{4}, []float32{-2, 0, 2, 7}),
			makeFloatInitializerForTests("min", []int64{}, []float32{0}),
			makeFloatInitializerForTests("max", []int64{}, []float32{5}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{4})
}

func TestHandleClip_Opset10_Attrs(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Clip",
		opset:        10,
		attrs:        map[string]Attr{"min": {Kind: AttrFloat, F: 0}, "max": {Kind: AttrFloat, F: 5}},
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{4}, []float32{-1, 0, 3, 9})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{4})
}

func TestHandleCast_F32ToF16(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Cast",
		attrs:  map[string]Attr{"to": {Kind: AttrInt, I: int64(onnxpb.TensorProto_FLOAT16)}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT16, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2}, []float32{1, 2})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpCast {
		t.Fatalf("root op=%s, want Cast", y.Node().Op())
	}
	if y.DType() != uop.Dtypes.Float16 {
		t.Fatalf("dtype=%v, want f16", y.DType())
	}
}

func TestHandleEqual(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Equal",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}},
			{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}},
		},
		outputs: []nameInfo{{Name: "z", DType: onnxpb.TensorProto_BOOL, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{3}, []float32{1, 2, 3}),
			makeFloatInitializerForTests("y", []int64{3}, []float32{1, 0, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	z := outs["z"]
	if z.Node().Op() != uop.OpCmpEq {
		t.Fatalf("root op=%s, want CmpEq", z.Node().Op())
	}
	if z.DType() != uop.Dtypes.Bool {
		t.Fatalf("dtype=%v, want bool", z.DType())
	}
}

// ── Value-oracle tests (cpuEval) ─────────────────────────────────────────────
//
// These run cpuEval on the handler-produced graph and compare against a
// hand-computed expected vector. atol = 1e-5 for f32 ops; 1e-3 for ops
// composed via exp/log expansions (Pow, Tanh, Sigmoid).

func TestHandleAdd_Value(t *testing.T) {
	z := elementwiseRig(t, "Add", []float32{1, 2, 3, 4}, []float32{10, 20, 30, 40}, 13)
	got, _, err := cpuEval(z)
	if err != nil {
		t.Fatalf("cpuEval: %v", err)
	}
	want := []float32{11, 22, 33, 44}
	if !allClose(got, want, 1e-5) {
		t.Errorf("Add value = %v, want %v", got, want)
	}
}

func TestHandleSub_Value(t *testing.T) {
	z := elementwiseRig(t, "Sub", []float32{10, 20, 30}, []float32{1, 2, 3}, 13)
	got, _, _ := cpuEval(z)
	if !allClose(got, []float32{9, 18, 27}, 1e-5) {
		t.Errorf("Sub value = %v, want [9 18 27]", got)
	}
}

func TestHandleMul_Value(t *testing.T) {
	z := elementwiseRig(t, "Mul", []float32{2, 3, 4}, []float32{5, 6, 7}, 13)
	got, _, _ := cpuEval(z)
	if !allClose(got, []float32{10, 18, 28}, 1e-5) {
		t.Errorf("Mul value = %v, want [10 18 28]", got)
	}
}

func TestHandleDiv_Value(t *testing.T) {
	z := elementwiseRig(t, "Div", []float32{10, 20, 30}, []float32{2, 4, 5}, 13)
	got, _, _ := cpuEval(z)
	if !allClose(got, []float32{5, 5, 6}, 1e-4) {
		t.Errorf("Div value = %v, want [5 5 6]", got)
	}
}

func TestHandleSqrt_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Sqrt",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{3}, []float32{4, 9, 16})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{2, 3, 4}, 1e-5) {
		t.Errorf("Sqrt value = %v, want [2 3 4]", got)
	}
}

func TestHandleNeg_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Neg",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{3}, []float32{1, -2, 3})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{-1, 2, -3}, 1e-5) {
		t.Errorf("Neg value = %v, want [-1 2 -3]", got)
	}
}

func TestHandleRelu_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType:       "Relu",
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{4}, []float32{-1, 0, 1, 2})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{0, 0, 1, 2}, 1e-5) {
		t.Errorf("Relu value = %v, want [0 0 1 2]", got)
	}
}

func TestHandleClip_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Clip",
		opset:  13,
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}},
			{Name: "min", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{}},
			{Name: "max", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{4}, []float32{-2, 0, 2, 7}),
			makeFloatInitializerForTests("min", []int64{}, []float32{0}),
			makeFloatInitializerForTests("max", []int64{}, []float32{5}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{0, 0, 2, 5}, 1e-5) {
		t.Errorf("Clip value = %v, want [0 0 2 5]", got)
	}
}

func TestHandleEqual_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Equal",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}},
			{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}},
		},
		outputs: []nameInfo{{Name: "z", DType: onnxpb.TensorProto_BOOL, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{3}, []float32{1, 2, 3}),
			makeFloatInitializerForTests("y", []int64{3}, []float32{1, 0, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["z"])
	want := []float32{1, 0, 1}
	if !allClose(got, want, 0) {
		t.Errorf("Equal value = %v, want %v", got, want)
	}
}

// Suppress unused imports if a future refactor drops them.
var _ = tensor.NewLeaf
