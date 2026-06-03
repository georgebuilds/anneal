package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// singleNodeModel builds a one-node ModelProto for handler-level tests.
// inputs[i] = (name, dtype, shape); outputs[j] = (name, dtype, shape).
// node has opType, attrs, declared inputs/outputs aligned with the slices.
//
// initializers seeds extra TensorProto initializers (e.g. weights). For
// handlers that need typed-host shape inputs, register them via initializers
// keyed by name so the runner can resolve them.
type singleNodeBuilder struct {
	opType       string
	opset        int64
	attrs        map[string]Attr
	inputs       []nameInfo
	outputs      []nameInfo
	initializers []*onnxpb.TensorProto
	domain       string
}

type nameInfo struct {
	Name  string
	DType onnxpb.TensorProto_DataType
	Dims  []int64
}

func (b *singleNodeBuilder) build(t *testing.T) *onnxpb.ModelProto {
	t.Helper()
	opset := b.opset
	if opset == 0 {
		opset = 13
	}

	// Convert attrs map to a slice of AttributeProto. (We construct a fresh
	// model and proto.Marshal it, so the Attrs map's encoded form must be
	// re-derived. Easier: rebuild AttributeProto from each Attr.)
	attrPbs := make([]*onnxpb.AttributeProto, 0, len(b.attrs))
	for name, a := range b.attrs {
		switch a.Kind {
		case AttrInt:
			attrPbs = append(attrPbs, &onnxpb.AttributeProto{
				Name: name, Type: onnxpb.AttributeProto_INT, I: a.I,
			})
		case AttrInts:
			attrPbs = append(attrPbs, &onnxpb.AttributeProto{
				Name: name, Type: onnxpb.AttributeProto_INTS, Ints: append([]int64{}, a.Is...),
			})
		case AttrFloat:
			attrPbs = append(attrPbs, &onnxpb.AttributeProto{
				Name: name, Type: onnxpb.AttributeProto_FLOAT, F: float32(a.F),
			})
		case AttrFloats:
			fs := make([]float32, len(a.Fs))
			for i, v := range a.Fs {
				fs[i] = float32(v)
			}
			attrPbs = append(attrPbs, &onnxpb.AttributeProto{
				Name: name, Type: onnxpb.AttributeProto_FLOATS, Floats: fs,
			})
		case AttrString:
			attrPbs = append(attrPbs, &onnxpb.AttributeProto{
				Name: name, Type: onnxpb.AttributeProto_STRING, S: []byte(a.S),
			})
		case AttrTensor:
			attrPbs = append(attrPbs, &onnxpb.AttributeProto{
				Name: name, Type: onnxpb.AttributeProto_TENSOR, T: a.T,
			})
		}
	}

	inputNames := make([]string, len(b.inputs))
	inputVI := make([]*onnxpb.ValueInfoProto, len(b.inputs))
	for i, in := range b.inputs {
		inputNames[i] = in.Name
		inputVI[i] = makeVI(in.Name, in.DType, in.Dims)
	}
	outputNames := make([]string, len(b.outputs))
	outputVI := make([]*onnxpb.ValueInfoProto, len(b.outputs))
	for i, out := range b.outputs {
		outputNames[i] = out.Name
		outputVI[i] = makeVI(out.Name, out.DType, out.Dims)
	}

	return &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: opset},
		},
		Graph: &onnxpb.GraphProto{
			Name:        "single-node",
			Input:       inputVI,
			Output:      outputVI,
			Initializer: b.initializers,
			Node: []*onnxpb.NodeProto{
				{
					Name:      "n1",
					OpType:    b.opType,
					Domain:    b.domain,
					Input:     inputNames,
					Output:    outputNames,
					Attribute: attrPbs,
				},
			},
		},
	}
}

func makeVI(name string, dt onnxpb.TensorProto_DataType, dims []int64) *onnxpb.ValueInfoProto {
	sh := &onnxpb.TensorShapeProto{}
	for _, d := range dims {
		sh.Dim = append(sh.Dim, &onnxpb.TensorShapeProto_Dimension{
			Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: d},
		})
	}
	return &onnxpb.ValueInfoProto{
		Name: name,
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: int32(dt),
					Shape:    sh,
				},
			},
		},
	}
}

// runSingleNode imports the model and runs it with `inputs` keyed by name.
// Returns the named output map (the runner-built device tensors).
func runSingleNode(t *testing.T, model *onnxpb.ModelProto, inputs map[string]*tensor.Tensor) (*Runner, map[string]*tensor.Tensor) {
	t.Helper()
	arena := uop.NewArena(256)
	bytes := mustMarshalProto(t, model)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	out, err := r.Run(inputs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return r, out
}

// runSingleNodeExpectError imports and runs, asserting the run errors and the
// error message contains every substring in wantSubs. Returns the error.
func runSingleNodeExpectError(t *testing.T, model *onnxpb.ModelProto, inputs map[string]*tensor.Tensor, wantSubs ...string) error {
	t.Helper()
	arena := uop.NewArena(64)
	bytes := mustMarshalProto(t, model)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		// Import may itself be the error path (e.g. bad attr decoding).
		for _, sub := range wantSubs {
			if !contains(err.Error(), sub) {
				t.Fatalf("Import error %q missing %q", err.Error(), sub)
			}
		}
		return err
	}
	_, err = r.Run(inputs)
	if err == nil {
		t.Fatalf("expected error containing %v, got nil", wantSubs)
	}
	for _, sub := range wantSubs {
		if !contains(err.Error(), sub) {
			t.Fatalf("error %q missing %q", err.Error(), sub)
		}
	}
	return err
}

func mustMarshalProto(t *testing.T, m *onnxpb.ModelProto) []byte {
	t.Helper()
	return mustMarshal(t, m)
}

func contains(haystack, needle string) bool {
	if needle == "" {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

// makeLeaf builds a tensor leaf with float32 data on the test arena.
func makeLeaf(arena *uop.Arena, sh []int64, data []float32) *tensor.Tensor {
	t := tensor.NewLeaf(arena, sh, uop.Dtypes.Float32, "test")
	t.SetData(data)
	return t
}

// makeIntInitializer builds an INT64 raw_data initializer with given values.
// Used to register host-tier shape inputs in single-node models.
func makeIntInitializer(name string, dims []int64, vals []int64) *onnxpb.TensorProto {
	raw := make([]byte, len(vals)*8)
	for i, v := range vals {
		u := uint64(v)
		raw[i*8] = byte(u)
		raw[i*8+1] = byte(u >> 8)
		raw[i*8+2] = byte(u >> 16)
		raw[i*8+3] = byte(u >> 24)
		raw[i*8+4] = byte(u >> 32)
		raw[i*8+5] = byte(u >> 40)
		raw[i*8+6] = byte(u >> 48)
		raw[i*8+7] = byte(u >> 56)
	}
	return &onnxpb.TensorProto{
		Name:     name,
		Dims:     dims,
		DataType: int32(onnxpb.TensorProto_INT64),
		RawData:  raw,
	}
}

// makeFloatInitializerForTests builds a FLOAT initializer.
func makeFloatInitializerForTests(name string, dims []int64, vals []float32) *onnxpb.TensorProto {
	return &onnxpb.TensorProto{
		Name:      name,
		Dims:      dims,
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: vals,
	}
}

// assertShape compares a tensor's shape against want.
func assertShape(t *testing.T, got *tensor.Tensor, want []int64) {
	t.Helper()
	gs := got.Shape()
	if len(gs) != len(want) {
		t.Fatalf("rank=%d, want %d (shape=%v)", len(gs), len(want), gs)
	}
	for i := range gs {
		if gs[i] != want[i] {
			t.Fatalf("dim %d=%d, want %d (shape=%v want %v)", i, gs[i], want[i], gs, want)
		}
	}
}

// assertShapeSints compares a tensor's symbolic shape rank.
func assertSints(t *testing.T, got []shape.Sint, want []shape.Sint) {
	t.Helper()
	if !shape.SintShapesEqual(got, want) {
		t.Fatalf("Sint shape mismatch")
	}
}
