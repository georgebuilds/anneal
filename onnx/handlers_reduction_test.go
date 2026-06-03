package onnx

import (
	"math"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/uop"
)

func TestHandleReduceSum_AxisAttrOpset12(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceSum",
		opset:  12,
		attrs: map[string]Attr{
			"axes":     {Kind: AttrInts, Is: []int64{1}},
			"keepdims": {Kind: AttrInt, I: 0},
		},
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpReduceAxis {
		t.Fatalf("root op=%s, want ReduceAxis", y.Node().Op())
	}
	assertShape(t, y, []int64{2})
}

func TestHandleReduceSum_AxisInputOpset13(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceSum",
		opset:  13,
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "axes", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		attrs:   map[string]Attr{"keepdims": {Kind: AttrInt, I: 0}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("axes", []int64{1}, []int64{1}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{2})
}

func TestHandleReduceMean_KeepDimsDefault(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceMean",
		opset:  12,
		attrs:  map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{1}}}, // keepdims default = 1
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{2, 1})
}

func TestHandleReduceMax(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceMax",
		opset:  12,
		attrs:  map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{0}}, "keepdims": {Kind: AttrInt, I: 0}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{3, 2}, []float32{1, 2, 3, 4, 5, 6})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpReduceAxis {
		t.Fatalf("root op=%s, want ReduceAxis", y.Node().Op())
	}
	assertShape(t, y, []int64{2})
}

// TestHandleReduceSum_F16F32Accumulator confirms the f32-accumulator pattern:
// for an f16 input, the reduction is wrapped in upcast→Sum→downcast. We assert
// the output dtype matches the input dtype (no leaking f32) and the graph
// terminates with a Cast back to f16.
func TestHandleReduceSum_F16F32Accumulator(t *testing.T) {
	// Encode f16 via int32_data (per ONNX spec for FLOAT16 storage).
	f16Vals := []int32{
		int32(f32ToF16Bits(1.0)),
		int32(f32ToF16Bits(2.0)),
		int32(f32ToF16Bits(3.0)),
		int32(f32ToF16Bits(4.0)),
	}
	xInit := &onnxpb.TensorProto{
		Name:      "x",
		Dims:      []int64{2, 2},
		DataType:  int32(onnxpb.TensorProto_FLOAT16),
		Int32Data: f16Vals,
	}
	b := &singleNodeBuilder{
		opType:       "ReduceSum",
		opset:        12,
		attrs:        map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{1}}, "keepdims": {Kind: AttrInt, I: 0}},
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT16, Dims: []int64{2, 2}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT16, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{xInit},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.DType() != uop.Dtypes.Float16 {
		t.Fatalf("output dtype=%v, want f16 (accumulator should down-cast)", y.DType())
	}
	// Root should be Cast (back to f16); its src should be the reduce.
	if y.Node().Op() != uop.OpCast {
		t.Fatalf("root op=%s, want Cast (f16 accumulator pattern)", y.Node().Op())
	}
	red := y.Node().Src(0)
	if red.Op() != uop.OpReduceAxis {
		t.Fatalf("expected ReduceAxis under Cast, got %s", red.Op())
	}
}

// ── Value-oracle tests ────────────────────────────────────────────────────────

func TestHandleReduceSum_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceSum",
		opset:  12,
		attrs: map[string]Attr{
			"axes":     {Kind: AttrInts, Is: []int64{1}},
			"keepdims": {Kind: AttrInt, I: 0},
		},
		inputs:       []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, err := cpuEval(outs["y"])
	if err != nil {
		t.Fatalf("cpuEval: %v", err)
	}
	want := []float32{6, 15} // row sums
	if !allClose(got, want, 1e-5) {
		t.Errorf("ReduceSum value = %v, want %v", got, want)
	}
}

func TestHandleReduceMean_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceMean",
		opset:  12,
		attrs:  map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{1}}, "keepdims": {Kind: AttrInt, I: 0}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	want := []float32{2, 5}
	if !allClose(got, want, 1e-5) {
		t.Errorf("ReduceMean value = %v, want %v", got, want)
	}
}

func TestHandleReduceMax_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "ReduceMax",
		opset:  12,
		attrs:  map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{0}}, "keepdims": {Kind: AttrInt, I: 0}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		initializers: []*onnxpb.TensorProto{makeFloatInitializerForTests("x", []int64{3, 2}, []float32{1, 2, 3, 4, 5, 6})},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	want := []float32{5, 6}
	if !allClose(got, want, 1e-5) {
		t.Errorf("ReduceMax value = %v, want %v", got, want)
	}
}

// f32ToF16Bits encodes a float32 as IEEE-754 binary16 (truncating).
func f32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 0x1)
	exp := int32((bits >> 23) & 0xff)
	mant := bits & 0x7fffff
	if exp == 0 && mant == 0 {
		return sign << 15
	}
	newExp := exp - 127 + 15
	if newExp <= 0 {
		return sign << 15
	}
	if newExp >= 31 {
		return (sign << 15) | (31 << 10)
	}
	return (sign << 15) | (uint16(newExp) << 10) | uint16(mant>>13)
}
