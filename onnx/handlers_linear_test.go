package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

func TestHandleGemm_Default(t *testing.T) {
	// A [2,3] x B [3,4] + C [4] → out [2,4].
	a := []float32{1, 2, 3, 4, 5, 6}
	bw := make([]float32, 12)
	for i := range bw {
		bw[i] = float32(i + 1)
	}
	c := []float32{1, 2, 3, 4}
	bld := &singleNodeBuilder{
		opType: "Gemm",
		inputs: []nameInfo{
			{Name: "A", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 4}},
			{Name: "C", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}},
		},
		outputs: []nameInfo{{Name: "Y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("A", []int64{2, 3}, a),
			makeFloatInitializerForTests("B", []int64{3, 4}, bw),
			makeFloatInitializerForTests("C", []int64{4}, c),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	assertShape(t, outs["Y"], []int64{2, 4})
}

func TestHandleGemm_TransAB(t *testing.T) {
	// transA, transB: A [3,2]^T @ B [4,3]^T → [2,4].
	a := []float32{1, 2, 3, 4, 5, 6}
	bw := make([]float32, 12)
	bld := &singleNodeBuilder{
		opType: "Gemm",
		attrs: map[string]Attr{
			"transA": {Kind: AttrInt, I: 1},
			"transB": {Kind: AttrInt, I: 1},
		},
		inputs: []nameInfo{
			{Name: "A", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4, 3}},
		},
		outputs: []nameInfo{{Name: "Y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("A", []int64{3, 2}, a),
			makeFloatInitializerForTests("B", []int64{4, 3}, bw),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	assertShape(t, outs["Y"], []int64{2, 4})
}

func TestHandleGemm_AlphaBeta(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	bw := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	c := []float32{1, 1, 1, 1}
	bld := &singleNodeBuilder{
		opType: "Gemm",
		attrs: map[string]Attr{
			"alpha": {Kind: AttrFloat, F: 2.0},
			"beta":  {Kind: AttrFloat, F: 0.5},
		},
		inputs: []nameInfo{
			{Name: "A", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 4}},
			{Name: "C", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}},
		},
		outputs: []nameInfo{{Name: "Y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 4}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("A", []int64{2, 3}, a),
			makeFloatInitializerForTests("B", []int64{3, 4}, bw),
			makeFloatInitializerForTests("C", []int64{4}, c),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	assertShape(t, outs["Y"], []int64{2, 4})
}

func TestHandleGemm_Value(t *testing.T) {
	// A=[[1,2],[3,4]], B=[[5,6],[7,8]], C=[1,1]
	// Y = A@B + C = [[5+14, 6+16]+[1,1], [15+28, 18+32]+[1,1]] = [[19+1, 22+1],[43+1,50+1]] = [[20,23],[44,51]]
	bld := &singleNodeBuilder{
		opType: "Gemm",
		inputs: []nameInfo{
			{Name: "A", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}},
			{Name: "C", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "Y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("A", []int64{2, 2}, []float32{1, 2, 3, 4}),
			makeFloatInitializerForTests("B", []int64{2, 2}, []float32{5, 6, 7, 8}),
			makeFloatInitializerForTests("C", []int64{2}, []float32{1, 1}),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	got, _, _ := cpuEval(outs["Y"])
	want := []float32{20, 23, 44, 51}
	if !allClose(got, want, 1e-5) {
		t.Errorf("Gemm value = %v, want %v", got, want)
	}
}

func TestHandleMatMul_Value(t *testing.T) {
	bld := &singleNodeBuilder{
		opType: "MatMul",
		inputs: []nameInfo{
			{Name: "A", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}},
		},
		outputs: []nameInfo{{Name: "Y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("A", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeFloatInitializerForTests("B", []int64{3, 2}, []float32{7, 8, 9, 10, 11, 12}),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	got, _, _ := cpuEval(outs["Y"])
	// A@B: row0=[1*7+2*9+3*11, 1*8+2*10+3*12]=[58, 64]; row1=[4*7+5*9+6*11, 4*8+5*10+6*12]=[139, 154]
	want := []float32{58, 64, 139, 154}
	if !allClose(got, want, 1e-4) {
		t.Errorf("MatMul value = %v, want %v", got, want)
	}
}

func TestHandleMatMul(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	bw := []float32{1, 2, 3, 4, 5, 6}
	bld := &singleNodeBuilder{
		opType: "MatMul",
		inputs: []nameInfo{
			{Name: "A", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}},
		},
		outputs: []nameInfo{{Name: "Y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("A", []int64{2, 3}, a),
			makeFloatInitializerForTests("B", []int64{3, 2}, bw),
		},
	}
	_, outs := runSingleNode(t, bld.build(t), nil)
	assertShape(t, outs["Y"], []int64{2, 2})
}
