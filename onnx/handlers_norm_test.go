package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

func TestHandleBatchNormalization_Inference(t *testing.T) {
	// X [1, 2, 2, 2], scale [2], B [2], mean [2], var [2].
	xVals := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	scale := []float32{1, 2}
	bias := []float32{0, 1}
	mean := []float32{0, 0}
	variance := []float32{1, 1}
	b := &singleNodeBuilder{
		opType: "BatchNormalization",
		attrs:  map[string]Attr{"epsilon": {Kind: AttrFloat, F: 1e-5}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}},
			{Name: "scale", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "mean", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "var", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 2, 2, 2}, xVals),
			makeFloatInitializerForTests("scale", []int64{2}, scale),
			makeFloatInitializerForTests("B", []int64{2}, bias),
			makeFloatInitializerForTests("mean", []int64{2}, mean),
			makeFloatInitializerForTests("var", []int64{2}, variance),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 2, 2, 2})
}

func TestHandleBatchNormalization_Value(t *testing.T) {
	// X [1, 2, 1, 1] = [[3], [5]]; scale=[2,4]; B=[1,2]; mean=[3,5]; var=[1,1].
	// y_c = ((x - 3 or 5) / sqrt(1 + 1e-5)) * (2 or 4) + (1 or 2) ≈ 0*2+1=1, 0*4+2=2.
	xVals := []float32{3, 5}
	b := &singleNodeBuilder{
		opType: "BatchNormalization",
		attrs:  map[string]Attr{"epsilon": {Kind: AttrFloat, F: 1e-5}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 1, 1}},
			{Name: "scale", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "mean", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "var", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 1, 1}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 2, 1, 1}, xVals),
			makeFloatInitializerForTests("scale", []int64{2}, []float32{2, 4}),
			makeFloatInitializerForTests("B", []int64{2}, []float32{1, 2}),
			makeFloatInitializerForTests("mean", []int64{2}, []float32{3, 5}),
			makeFloatInitializerForTests("var", []int64{2}, []float32{1, 1}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{1, 2}, 1e-4) {
		t.Errorf("BatchNorm value = %v, want [1 2]", got)
	}
}

func TestHandleBatchNormalization_TrainingModeRejected(t *testing.T) {
	xVals := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := &singleNodeBuilder{
		opType: "BatchNormalization",
		attrs:  map[string]Attr{"training_mode": {Kind: AttrInt, I: 1}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}},
			{Name: "scale", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "mean", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
			{Name: "var", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 2, 2, 2}, xVals),
			makeFloatInitializerForTests("scale", []int64{2}, []float32{1, 1}),
			makeFloatInitializerForTests("B", []int64{2}, []float32{0, 0}),
			makeFloatInitializerForTests("mean", []int64{2}, []float32{0, 0}),
			makeFloatInitializerForTests("var", []int64{2}, []float32{1, 1}),
		},
	}
	_ = runSingleNodeExpectError(t, b.build(t), nil, "training_mode")
}
