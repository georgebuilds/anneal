package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

func TestHandleConv_2DSymmetricPads(t *testing.T) {
	// Input: 1x1x4x4. Weight 1x1x3x3 (no bias). pads=[1,1,1,1], stride=1.
	// Output: 1x1x4x4.
	xVals := make([]float32, 16)
	for i := range xVals {
		xVals[i] = float32(i + 1)
	}
	wVals := make([]float32, 9)
	for i := range wVals {
		wVals[i] = 1.0
	}
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
			"strides":      {Kind: AttrInts, Is: []int64{1, 1}},
			"pads":         {Kind: AttrInts, Is: []int64{1, 1, 1, 1}},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
			makeFloatInitializerForTests("w", []int64{1, 1, 3, 3}, wVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{1, 1, 4, 4})
}

func TestHandleConv_2DAsymmetricPads(t *testing.T) {
	// Asymmetric pads forces the handler to emit explicit Pad-then-Conv.
	xVals := make([]float32, 16)
	wVals := make([]float32, 9)
	for i := range wVals {
		wVals[i] = 1.0
	}
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
			"strides":      {Kind: AttrInts, Is: []int64{1, 1}},
			"pads":         {Kind: AttrInts, Is: []int64{1, 0, 0, 1}},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
			makeFloatInitializerForTests("w", []int64{1, 1, 3, 3}, wVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 1, 3, 3})
}

func TestHandleConv_1DLifted(t *testing.T) {
	// 1-D conv: input [1, 1, 5], weight [1, 1, 3], stride=1, pad=0. Output [1, 1, 3].
	xVals := []float32{1, 2, 3, 4, 5}
	wVals := []float32{1, 1, 1}
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3}},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 5}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 5}, xVals),
			makeFloatInitializerForTests("w", []int64{1, 1, 3}, wVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 1, 3})
}

func TestHandleConv_WithBias(t *testing.T) {
	xVals := make([]float32, 16)
	wVals := make([]float32, 9)
	for i := range wVals {
		wVals[i] = 1.0
	}
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}},
			{Name: "bias", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
			makeFloatInitializerForTests("w", []int64{1, 1, 3, 3}, wVals),
			makeFloatInitializerForTests("bias", []int64{1}, []float32{2.5}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 1, 2, 2})
}

func TestHandleConv_GroupGT1Rejected(t *testing.T) {
	xVals := make([]float32, 16)
	wVals := make([]float32, 9)
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
			"group":        {Kind: AttrInt, I: 2},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 4, 4}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1, 3, 3}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 2, 4, 4}, append(xVals, xVals...)),
			makeFloatInitializerForTests("w", []int64{2, 1, 3, 3}, append(wVals, wVals...)),
		},
	}
	_ = runSingleNodeExpectError(t, b.build(t), nil, "group", "not supported")
}

func TestHandleConv_Value(t *testing.T) {
	// Simple 1x1x3x3 input * 1x1x2x2 weight, no padding, stride=1, no bias.
	// Input: [[1,2,3],[4,5,6],[7,8,9]]; weight: [[1,0],[0,1]] (identity-style diagonal).
	// Output [1,1,2,2]: [1+5, 2+6, 4+8, 5+9] = [6, 8, 12, 14].
	xVals := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	wVals := []float32{1, 0, 0, 1}
	bld := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 2, 2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 3, 3}, xVals),
			makeFloatInitializerForTests("w", []int64{1, 1, 2, 2}, wVals),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	got, _, err := cpuEval(outs["y"])
	if err != nil {
		t.Fatalf("cpuEval: %v", err)
	}
	want := []float32{6, 8, 12, 14}
	if !allClose(got, want, 1e-4) {
		t.Errorf("Conv value = %v, want %v", got, want)
	}
}

func TestHandleConv_AutoPadRejected(t *testing.T) {
	xVals := make([]float32, 16)
	wVals := make([]float32, 9)
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
			"auto_pad":     {Kind: AttrString, S: "SAME_UPPER"},
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}},
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
			makeFloatInitializerForTests("w", []int64{1, 1, 3, 3}, wVals),
		},
	}
	_ = runSingleNodeExpectError(t, b.build(t), nil, "auto_pad")
}
