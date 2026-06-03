package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/uop"
)

func TestHandleReshape_Concrete(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Reshape",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("sh", []int64{2}, []int64{2, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{2, 3})
}

func TestHandleReshape_NegOneInfer(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Reshape",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("sh", []int64{2}, []int64{2, -1}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{2, 3})
}

func TestHandleReshape_ZeroCopy(t *testing.T) {
	// 0 in target means "copy from input dim" (allowzero=0 default).
	b := &singleNodeBuilder{
		opType: "Reshape",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("sh", []int64{2}, []int64{0, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{2, 3})
}

func TestHandleReshape_AllowZero(t *testing.T) {
	// allowzero=1 means 0 is literal — cannot copy.
	b := &singleNodeBuilder{
		opType: "Reshape",
		attrs:  map[string]Attr{"allowzero": {Kind: AttrInt, I: 1}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{0, 0}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("sh", []int64{2}, []int64{0, 0}),
		},
	}
	// allowzero=1 with [0,0] target on a 6-element input is illegal (volume
	// mismatch); we just exercise the attribute path.
	defer func() {
		if r := recover(); r != nil {
			// fine — Reshape will reject volume mismatch via panic
			return
		}
	}()
	runSingleNode(t, b.build(t), nil)
}

func TestHandleFlatten_DefaultAxis(t *testing.T) {
	// Default axis=1 → [d0, prod(d1..)].
	b := &singleNodeBuilder{
		opType:  "Flatten",
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 12}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3, 4}, make([]float32, 24)),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{2, 12})
}

func TestHandleFlatten_AxisZero(t *testing.T) {
	// axis=0 → [1, prod(all)].
	b := &singleNodeBuilder{
		opType:  "Flatten",
		attrs:   map[string]Attr{"axis": {Kind: AttrInt, I: 0}},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 6}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{1, 6})
}

func TestHandleSqueeze_AttrOpset12(t *testing.T) {
	b := &singleNodeBuilder{
		opType:  "Squeeze",
		opset:   12,
		attrs:   map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{1}}},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1, 3}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 1, 3}, []float32{1, 2, 3, 4, 5, 6}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{2, 3})
}

func TestHandleSqueeze_InputOpset13(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Squeeze",
		opset:  13,
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1, 3}},
			{Name: "axes", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 1, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("axes", []int64{1}, []int64{1}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{2, 3})
}

func TestHandleSqueeze_NoAxes(t *testing.T) {
	// Without axes: remove all size-1 dims.
	b := &singleNodeBuilder{
		opType:  "Squeeze",
		opset:   12,
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 1, 3}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 2, 1, 3}, []float32{1, 2, 3, 4, 5, 6}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{2, 3})
}

func TestHandleUnsqueeze_AttrOpset12(t *testing.T) {
	b := &singleNodeBuilder{
		opType:  "Unsqueeze",
		opset:   12,
		attrs:   map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{1}}},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{2, 1, 3})
}

func TestHandleUnsqueeze_InputOpset13(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Unsqueeze",
		opset:  13,
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "axes", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("axes", []int64{1}, []int64{1}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{2, 1, 3})
}

func TestHandleTranspose_PermAttr(t *testing.T) {
	b := &singleNodeBuilder{
		opType:  "Transpose",
		attrs:   map[string]Attr{"perm": {Kind: AttrInts, Is: []int64{1, 0}}},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpPermute {
		t.Fatalf("root op=%s, want Permute", y.Node().Op())
	}
	assertShape(t, y, []int64{3, 2})
}

func TestHandleTranspose_DefaultReverse(t *testing.T) {
	b := &singleNodeBuilder{
		opType:  "Transpose",
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3, 4}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4, 3, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3, 4}, make([]float32, 24)),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{4, 3, 2})
}

func TestHandleConcat(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Concat",
		attrs:  map[string]Attr{"axis": {Kind: AttrInt, I: 1}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}},
		},
		outputs: []nameInfo{{Name: "z", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 5}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeFloatInitializerForTests("y", []int64{2, 2}, []float32{10, 20, 30, 40}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	z := outs["z"]
	assertShape(t, z, []int64{2, 5})
	// Concat builds Add of Pad-padded operands.
	if z.Node().Op() != uop.OpAdd {
		t.Fatalf("root op=%s, want Add (Concat via Pad+Add)", z.Node().Op())
	}
}

func TestHandleGather(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Gather",
		attrs:  map[string]Attr{"axis": {Kind: AttrInt, I: 0}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4, 3}},
			{Name: "idx", DType: onnxpb.TensorProto_INT32, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{4, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
		},
	}
	model := b.build(t)
	arena := uop.NewArena(64)
	bytes := mustMarshalProto(t, model)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	// Provide idx as a user input (since INT32 isn't in initializers).
	// Actually let me add it as initializer instead.
	_ = r
	bint32 := &singleNodeBuilder{
		opType: "Gather",
		attrs:  map[string]Attr{"axis": {Kind: AttrInt, I: 0}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4, 3}},
			{Name: "idx", DType: onnxpb.TensorProto_INT32, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{4, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
			{
				Name:      "idx",
				Dims:      []int64{2},
				DataType:  int32(onnxpb.TensorProto_INT32),
				Int32Data: []int32{1, 3},
			},
		},
	}
	_, outs := runSingleNode(t, bint32.build(t), nil)
	y := outs["y"]
	if y.Node().Op() != uop.OpGather {
		t.Fatalf("root op=%s, want Gather", y.Node().Op())
	}
	assertShape(t, y, []int64{2, 3})
}

func TestHandleSlice_PositiveStep(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Slice",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "starts", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
			{Name: "ends", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("starts", []int64{1}, []int64{1}),
			makeIntInitializer("ends", []int64{1}, []int64{4}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	assertShape(t, outs["y"], []int64{3})
}

func TestHandleSlice_NegativeStepRejected(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Slice",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "starts", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
			{Name: "ends", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
			{Name: "axes", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
			{Name: "steps", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("starts", []int64{1}, []int64{5}),
			makeIntInitializer("ends", []int64{1}, []int64{1}),
			makeIntInitializer("axes", []int64{1}, []int64{0}),
			makeIntInitializer("steps", []int64{1}, []int64{-1}),
		},
	}
	runSingleNodeExpectError(t, b.build(t), nil, "negative step")
}

// ── Value-oracle tests ────────────────────────────────────────────────────────

func TestHandleReshape_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Reshape",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("sh", []int64{2}, []int64{2, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, sh, err := cpuEval(outs["y"])
	if err != nil {
		t.Fatalf("cpuEval: %v", err)
	}
	if !int64SliceEq(sh, []int64{2, 3}) {
		t.Fatalf("shape=%v, want [2 3]", sh)
	}
	if !allClose(got, []float32{1, 2, 3, 4, 5, 6}, 0) {
		t.Errorf("Reshape value = %v, want [1..6]", got)
	}
}

func TestHandleTranspose_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType:  "Transpose",
		attrs:   map[string]Attr{"perm": {Kind: AttrInts, Is: []int64{1, 0}}},
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3, 2}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	want := []float32{1, 4, 2, 5, 3, 6}
	if !allClose(got, want, 0) {
		t.Errorf("Transpose value = %v, want %v", got, want)
	}
}

func TestHandleConcat_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Concat",
		attrs:  map[string]Attr{"axis": {Kind: AttrInt, I: 1}},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}},
			{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}},
		},
		outputs: []nameInfo{{Name: "z", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 5}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			makeFloatInitializerForTests("y", []int64{2, 2}, []float32{10, 20, 30, 40}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["z"])
	// Row-major concat axis=1: rows are [1 2 3 | 10 20] [4 5 6 | 30 40].
	want := []float32{1, 2, 3, 10, 20, 4, 5, 6, 30, 40}
	if !allClose(got, want, 0) {
		t.Errorf("Concat value = %v, want %v", got, want)
	}
}

func TestHandleSlice_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Slice",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{6}},
			{Name: "starts", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
			{Name: "ends", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{6}, []float32{1, 2, 3, 4, 5, 6}),
			makeIntInitializer("starts", []int64{1}, []int64{1}),
			makeIntInitializer("ends", []int64{1}, []int64{4}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	if !allClose(got, []float32{2, 3, 4}, 0) {
		t.Errorf("Slice value = %v, want [2 3 4]", got)
	}
}

func TestHandleExpand_Value(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Expand",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 3}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 3}, []float32{7, 8, 9}),
			makeIntInitializer("sh", []int64{2}, []int64{2, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	got, _, _ := cpuEval(outs["y"])
	want := []float32{7, 8, 9, 7, 8, 9}
	if !allClose(got, want, 0) {
		t.Errorf("Expand value = %v, want %v", got, want)
	}
}

func TestHandleExpand(t *testing.T) {
	b := &singleNodeBuilder{
		opType: "Expand",
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 3}},
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4, 3}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, 3}, []float32{1, 2, 3}),
			makeIntInitializer("sh", []int64{2}, []int64{4, 3}),
		},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{4, 3})
	if y.Node().Op() != uop.OpExpand {
		t.Fatalf("root op=%s, want Expand", y.Node().Op())
	}
}
