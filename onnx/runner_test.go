package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// makeFloatTensor builds a TensorProto initializer using float_data.
func makeFloatTensor(name string, dims []int64, vals []float32) *onnxpb.TensorProto {
	return &onnxpb.TensorProto{
		Name:      name,
		Dims:      dims,
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: vals,
	}
}

// makeRawTensor builds a TensorProto with a typed int data type encoded as
// little-endian raw_data. width is the element width in bytes.
func makeRawTensor(name string, dims []int64, dt onnxpb.TensorProto_DataType, vals []int64, width int) *onnxpb.TensorProto {
	raw := make([]byte, len(vals)*width)
	switch width {
	case 1:
		for i, v := range vals {
			raw[i] = byte(v)
		}
	case 2:
		for i, v := range vals {
			binary.LittleEndian.PutUint16(raw[i*2:], uint16(v))
		}
	case 4:
		for i, v := range vals {
			binary.LittleEndian.PutUint32(raw[i*4:], uint32(v))
		}
	case 8:
		for i, v := range vals {
			binary.LittleEndian.PutUint64(raw[i*8:], uint64(v))
		}
	}
	return &onnxpb.TensorProto{
		Name:     name,
		Dims:     dims,
		DataType: int32(dt),
		RawData:  raw,
	}
}

// makeDoubleRawTensor builds a DOUBLE TensorProto whose raw_data carries
// IEEE-754 binary64 little-endian.
func makeDoubleRawTensor(name string, dims []int64, vals []float64) *onnxpb.TensorProto {
	raw := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	return &onnxpb.TensorProto{
		Name:     name,
		Dims:     dims,
		DataType: int32(onnxpb.TensorProto_DOUBLE),
		RawData:  raw,
	}
}

// makeValueInfo constructs a graph input/output entry.
func makeValueInfo(name string, dt onnxpb.TensorProto_DataType, dims []int64) *onnxpb.ValueInfoProto {
	shape := &onnxpb.TensorShapeProto{}
	for _, d := range dims {
		shape.Dim = append(shape.Dim, &onnxpb.TensorShapeProto_Dimension{
			Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: d},
		})
	}
	return &onnxpb.ValueInfoProto{
		Name: name,
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: int32(dt),
					Shape:    shape,
				},
			},
		},
	}
}

func mustMarshal(t *testing.T, m proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(m)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	return b
}

// withWarnCapture swaps Warn for a capture function; the returned defer-able
// restores the original.
func withWarnCapture(captured *[]string) func() {
	prev := Warn
	Warn = func(format string, args ...any) {
		*captured = append(*captured, fmt.Sprintf(format, args...))
	}
	return func() { Warn = prev }
}

// newDummyTensor allocates a tiny float32 leaf so input resolution can
// succeed during dispatch tests that don't care about the input contents.
func newDummyTensor(arena *uop.Arena) *tensor.Tensor {
	t := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
	t.SetData([]float32{0})
	return t
}

// TestImportZeroNodeIdentity exercises the foundation: a model with one
// initializer and one graph output whose name matches the initializer's
// name. Run() should return the initializer's tensor.
func TestImportZeroNodeIdentity(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "tiny",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("w", onnxpb.TensorProto_FLOAT, []int64{2, 3})},
			Initializer: []*onnxpb.TensorProto{
				makeFloatTensor("w", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
			},
		},
	}
	bytes := mustMarshal(t, model)

	arena := uop.NewArena(64)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}

	if r.Opset() != 13 {
		t.Fatalf("opset=%d, want 13", r.Opset())
	}

	w, ok := r.Initializers()["w"]
	if !ok {
		t.Fatalf("initializer %q missing from runner.initializers", "w")
	}
	if !w.IsDevice() {
		t.Fatalf("initializer %q is not a device tensor (kind=%d)", "w", w.Kind)
	}
	wt := w.Tensor()
	if got, want := wt.Shape(), []int64{2, 3}; !shapeEq(got, want) {
		t.Fatalf("initializer shape=%v, want %v", got, want)
	}
	if wt.DType() != uop.Dtypes.Float32 {
		t.Fatalf("initializer dtype=%v, want f32", wt.DType())
	}
	if data := wt.Data(); !float32SliceEq(data, []float32{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("initializer data=%v, want [1..6]", data)
	}

	// Run with no inputs needed (the input isn't consumed). Output "w"
	// resolves directly from the initializer state.
	out, err := r.Run(nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if _, ok := out["w"]; !ok {
		t.Fatalf("Run output missing %q (got %v)", "w", keys(out))
	}
}

// TestRunUnknownOpType walks a graph containing a node whose OpType has no
// registered handler and asserts the dispatch error is descriptive.
func TestRunUnknownOpType(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "with_unknown",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("y", onnxpb.TensorProto_FLOAT, []int64{1})},
			Node: []*onnxpb.NodeProto{
				{
					Name:   "n1",
					OpType: "XXX",
					Input:  []string{"x"},
					Output: []string{"y"},
				},
			},
		},
	}
	bytes := mustMarshal(t, model)

	arena := uop.NewArena(64)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}

	// Seed input "x" with a dummy tensor so input resolution doesn't trip
	// before dispatch.
	xt := newDummyTensor(arena)
	_, err = r.Run(map[string]*tensor.Tensor{"x": xt})
	if err == nil {
		t.Fatalf("expected unsupported op error, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported op") {
		t.Fatalf("error %q does not contain 'unsupported op'", err.Error())
	}
	if !strings.Contains(err.Error(), `"XXX"`) {
		t.Fatalf("error %q does not name op XXX", err.Error())
	}
}

// TestImportIntInitializers checks INT8/INT16/INT32 initializers decode into
// the right dtype with the right values.
func TestImportIntInitializers(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "ints",
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("dummy", onnxpb.TensorProto_INT32, []int64{1})},
			Initializer: []*onnxpb.TensorProto{
				makeRawTensor("i8", []int64{4}, onnxpb.TensorProto_INT8, []int64{-1, 0, 1, 127}, 1),
				makeRawTensor("i16", []int64{3}, onnxpb.TensorProto_INT16, []int64{-32768, 0, 32767}, 2),
				makeRawTensor("i32", []int64{2}, onnxpb.TensorProto_INT32, []int64{-100, 100}, 4),
			},
		},
	}
	bytes := mustMarshal(t, model)

	arena := uop.NewArena(64)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}

	cases := []struct {
		name  string
		dtype *uop.DType
		shape []int64
		vals  []float32
	}{
		{"i8", uop.Dtypes.Int8, []int64{4}, []float32{-1, 0, 1, 127}},
		{"i16", uop.Dtypes.Int16, []int64{3}, []float32{-32768, 0, 32767}},
		{"i32", uop.Dtypes.Int32, []int64{2}, []float32{-100, 100}},
	}
	for _, c := range cases {
		v, ok := r.Initializers()[c.name]
		if !ok {
			t.Errorf("initializer %q missing", c.name)
			continue
		}
		tw := v.Tensor()
		if tw.DType() != c.dtype {
			t.Errorf("%s: dtype=%v, want %v", c.name, tw.DType(), c.dtype)
		}
		if !shapeEq(tw.Shape(), c.shape) {
			t.Errorf("%s: shape=%v, want %v", c.name, tw.Shape(), c.shape)
		}
		if !float32SliceEq(tw.Data(), c.vals) {
			t.Errorf("%s: data=%v, want %v", c.name, tw.Data(), c.vals)
		}
	}
}

// TestImportDoubleInitializerWarns checks a DOUBLE initializer is cast to f32
// and a warning is emitted.
func TestImportDoubleInitializerWarns(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "doubles",
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("dummy", onnxpb.TensorProto_FLOAT, []int64{1})},
			Initializer: []*onnxpb.TensorProto{
				makeDoubleRawTensor("d", []int64{3}, []float64{1.5, -2.25, 3.0}),
			},
		},
	}
	bytes := mustMarshal(t, model)

	var captured []string
	restore := withWarnCapture(&captured)
	defer restore()

	arena := uop.NewArena(64)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}

	v, ok := r.Initializers()["d"]
	if !ok {
		t.Fatalf("initializer %q missing", "d")
	}
	tw := v.Tensor()
	if tw.DType() != uop.Dtypes.Float32 {
		t.Fatalf("DOUBLE initializer dtype=%v, want f32 downcast", tw.DType())
	}
	want := []float32{1.5, -2.25, 3.0}
	if !float32SliceEq(tw.Data(), want) {
		t.Fatalf("DOUBLE downcast data=%v, want %v", tw.Data(), want)
	}

	matched := false
	for _, s := range captured {
		if strings.Contains(s, "DOUBLE") && strings.Contains(s, "f32") {
			matched = true
			break
		}
	}
	if !matched {
		t.Fatalf("no DOUBLE downcast warning captured; warnings=%v", captured)
	}
}

// TestOpsetTooOld verifies opset < 10 is a hard error.
func TestOpsetTooOld(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 6,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 9},
		},
		Graph: &onnxpb.GraphProto{Name: "old"},
	}
	bytes := mustMarshal(t, model)
	arena := uop.NewArena(8)
	_, err := Import(bytes, arena, "test")
	if err == nil {
		t.Fatalf("expected error for opset 9, got nil")
	}
	if !strings.Contains(err.Error(), "opset") {
		t.Fatalf("error %q does not mention opset", err.Error())
	}
}

// TestInitializerInterning checks that two TensorProto initializers with
// identical (dtype, dims, raw payload) intern to the same arena leaf.
func TestInitializerInterning(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "interning",
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("dummy", onnxpb.TensorProto_FLOAT, []int64{1})},
			Initializer: []*onnxpb.TensorProto{
				makeFloatTensor("a", []int64{2}, []float32{1, 2}),
				makeFloatTensor("b", []int64{2}, []float32{1, 2}),
			},
		},
	}
	bytes := mustMarshal(t, model)
	arena := uop.NewArena(16)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	va, ok := r.Initializers()["a"]
	if !ok {
		t.Fatalf("missing a")
	}
	vb, ok := r.Initializers()["b"]
	if !ok {
		t.Fatalf("missing b")
	}
	if va.Tensor().Node().Index() != vb.Tensor().Node().Index() {
		t.Fatalf("interning failed: a node=%d != b node=%d",
			va.Tensor().Node().Index(), vb.Tensor().Node().Index())
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func shapeEq(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func float32SliceEq(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func keys(m map[string]*tensor.Tensor) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
