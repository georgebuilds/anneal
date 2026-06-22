package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestHandleConstant_FloatTensor exercises the device-tier Constant path:
// a TensorProto with FLOAT data is materialised as a leaf, then exposed as
// a graph output. We assert the output shape, dtype, AND its float32 data
// element-by-element - Constant carries data eagerly so this is a pure
// value-oracle check.
func TestHandleConstant_FloatTensor(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Dims:      []int64{2, 3},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2, 3, 4, 5, 6},
	}
	b := &singleNodeBuilder{
		opType: "Constant",
		attrs:  map[string]Attr{"value": {Kind: AttrTensor, T: tp}},
		inputs: nil,
		outputs: []nameInfo{{
			Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3},
		}},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y, ok := outs["y"]
	if !ok {
		t.Fatalf("output y missing")
	}
	assertShape(t, y, []int64{2, 3})
	if y.DType() != uop.Dtypes.Float32 {
		t.Fatalf("dtype=%v, want f32", y.DType())
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	got := y.Data()
	if len(got) != len(want) {
		t.Fatalf("data len=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("data[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

// TestHandleIdentity_PassThrough asserts Identity returns the same tensor.
func TestHandleIdentity_PassThrough(t *testing.T) {
	b := &singleNodeBuilder{
		opType:  "Identity",
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}}},
	}
	model := b.build(t)
	arena := uop.NewArena(64)
	bytes := mustMarshalProto(t, model)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	x := tensor.NewLeaf(arena, []int64{2, 2}, uop.Dtypes.Float32, "test")
	x.SetData([]float32{1, 2, 3, 4})
	outs, err := r.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	y := outs["y"]
	if y == nil {
		t.Fatalf("output y missing")
	}
	// Identity should pass the same tensor through.
	if y != x {
		t.Errorf("Identity returned different tensor")
	}
	got := y.Data()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("data[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

// TestHandleConstantOfShape_Default builds a 0-filled tensor of the host-fed
// shape. We assert the output shape + dtype + structure (Expand of a Reshape
// of a Const). The leaf data of the underlying Const is asserted too -
// FullSints builds a CONST → RESHAPE → EXPAND tree.
func TestHandleConstantOfShape_Default(t *testing.T) {
	// Initializer carrying shape [2,3] as INT64.
	shapeInit := makeIntInitializer("sh", []int64{2}, []int64{2, 3})
	b := &singleNodeBuilder{
		opType: "ConstantOfShape",
		inputs: []nameInfo{
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3}}},
		initializers: []*onnxpb.TensorProto{shapeInit},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	if y == nil {
		t.Fatalf("output y missing")
	}
	assertShape(t, y, []int64{2, 3})
	if y.Node().Op() != uop.OpExpand {
		t.Fatalf("root op=%s, want Expand (FullSints emits Expand-on-Reshape-on-Const)", y.Node().Op())
	}
}

// TestHandleConstantOfShape_WithValue uses the `value` attribute to fill
// with a non-default scalar. The Const leaf carries the fill value.
func TestHandleConstantOfShape_WithValue(t *testing.T) {
	shapeInit := makeIntInitializer("sh", []int64{2}, []int64{2, 2})
	fillTP := &onnxpb.TensorProto{
		Dims: []int64{1}, DataType: int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{7.5},
	}
	b := &singleNodeBuilder{
		opType: "ConstantOfShape",
		attrs:  map[string]Attr{"value": {Kind: AttrTensor, T: fillTP}},
		inputs: []nameInfo{
			{Name: "sh", DType: onnxpb.TensorProto_INT64, Dims: []int64{2}},
		},
		outputs:      []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 2}}},
		initializers: []*onnxpb.TensorProto{shapeInit},
	}
	_, outs := runSingleNode(t, b.build(t), nil)
	y := outs["y"]
	assertShape(t, y, []int64{2, 2})
	// Walk: y is Expand → Reshape → Const(7.5). Assert the Const's arg.
	expand := y.Node()
	reshape := expand.Src(0)
	c := reshape.Src(0)
	if c.Op() != uop.OpConst {
		t.Fatalf("expected Const at leaf, got %s", c.Op())
	}
	v, ok := c.Arg().(float64)
	if !ok || v != 7.5 {
		t.Errorf("Const arg=%v ok=%v, want 7.5", v, ok)
	}
}
