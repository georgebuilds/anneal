package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

func TestHandleMaxPool_Stride1(t *testing.T) {
	xVals := make([]float32, 16)
	for i := range xVals {
		xVals[i] = float32(i + 1)
	}
	b := &singleNodeBuilder{
		opType: "MaxPool",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"strides":      {Kind: AttrInts, Is: []int64{2, 2}},
		},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 1, 2, 2})
}

func TestHandleMaxPool_CeilModeRejected(t *testing.T) {
	xVals := make([]float32, 16)
	b := &singleNodeBuilder{
		opType: "MaxPool",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"ceil_mode":    {Kind: AttrInt, I: 1},
		},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 3, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
		},
	}
	runSingleNodeExpectError(t, b.build(t), nil, "ceil_mode")
}

func TestHandleMaxPool_AutoPadRejected(t *testing.T) {
	xVals := make([]float32, 16)
	b := &singleNodeBuilder{
		opType: "MaxPool",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"auto_pad":     {Kind: AttrString, S: "SAME_UPPER"},
		},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
		},
	}
	runSingleNodeExpectError(t, b.build(t), nil, "auto_pad")
}

func TestHandleMaxPool_Value(t *testing.T) {
	// 4x4 grid of 1..16; 2x2 kernel, stride 2 → each output cell is the max
	// of a 2x2 quadrant: [6, 8, 14, 16].
	xVals := make([]float32, 16)
	for i := range xVals {
		xVals[i] = float32(i + 1)
	}
	b := &singleNodeBuilder{
		opType: "MaxPool",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
			"strides":      {Kind: AttrInts, Is: []int64{2, 2}},
		},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, err := cpuEval(outs["y"])
	if err != nil {
		t.Fatalf("cpuEval: %v", err)
	}
	want := []float32{6, 8, 14, 16}
	if !allClose(got, want, 1e-5) {
		t.Errorf("MaxPool value = %v, want %v", got, want)
	}
}

func TestHandleGlobalAveragePool_Value(t *testing.T) {
	// 1..16 → mean = (1+16)*16/2/16 = 8.5
	xVals := make([]float32, 16)
	for i := range xVals {
		xVals[i] = float32(i + 1)
	}
	b := &singleNodeBuilder{
		opType:  "GlobalAveragePool",
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 1, 1}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{8.5}, 1e-5) {
		t.Errorf("GlobalAveragePool value = %v, want [8.5]", got)
	}
}

func TestHandleGlobalAveragePool(t *testing.T) {
	xVals := make([]float32, 16)
	for i := range xVals {
		xVals[i] = float32(i + 1)
	}
	b := &singleNodeBuilder{
		opType:  "GlobalAveragePool",
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 1, 1}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, xVals),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 1, 1, 1})
}
