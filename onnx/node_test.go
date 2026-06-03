package onnx

import (
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

// ── Attr default-return accessor matrix ───────────────────────────────────────

// TestAttr_Int_HitAndDefault verifies Int() returns the payload when Kind=AttrInt
// and returns def on every other kind.
func TestAttr_Int_HitAndDefault(t *testing.T) {
	hit := Attr{Kind: AttrInt, I: 42}
	if got := hit.Int(-1); got != 42 {
		t.Errorf("Int=%d, want 42", got)
	}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrInts, Is: []int64{1}},
		{Kind: AttrFloat, F: 1.0},
		{Kind: AttrFloats, Fs: []float64{1.0}},
		{Kind: AttrString, S: "x"},
		{Kind: AttrStrings, Ss: []string{"x"}},
		{Kind: AttrTensor, T: &onnxpb.TensorProto{}},
		{Kind: AttrGraph, G: &onnxpb.GraphProto{}},
	}
	for _, a := range wrong {
		if got := a.Int(99); got != 99 {
			t.Errorf("Int on kind=%d returned %d, want default 99", a.Kind, got)
		}
	}
}

func TestAttr_Ints_HitAndDefault(t *testing.T) {
	src := []int64{1, 2, 3}
	hit := Attr{Kind: AttrInts, Is: src}
	got := hit.Ints(nil)
	if len(got) != len(src) {
		t.Fatalf("Ints length=%d, want %d", len(got), len(src))
	}
	for i := range src {
		if got[i] != src[i] {
			t.Errorf("Ints[%d]=%d, want %d", i, got[i], src[i])
		}
	}
	def := []int64{9, 9}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrInt, I: 1},
		{Kind: AttrString, S: "x"},
		{Kind: AttrTensor, T: &onnxpb.TensorProto{}},
	}
	for _, a := range wrong {
		got := a.Ints(def)
		if len(got) != len(def) || got[0] != 9 {
			t.Errorf("Ints on kind=%d did not return default", a.Kind)
		}
	}
}

func TestAttr_Float_HitAndDefault(t *testing.T) {
	hit := Attr{Kind: AttrFloat, F: 2.5}
	if got := hit.Float(-1); got != 2.5 {
		t.Errorf("Float=%v, want 2.5", got)
	}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrInt, I: 1},
		{Kind: AttrFloats, Fs: []float64{1.0}},
	}
	for _, a := range wrong {
		if got := a.Float(7.7); got != 7.7 {
			t.Errorf("Float on kind=%d returned %v, want default 7.7", a.Kind, got)
		}
	}
}

func TestAttr_Floats_HitAndDefault(t *testing.T) {
	src := []float64{1.0, 2.0, 3.0}
	hit := Attr{Kind: AttrFloats, Fs: src}
	got := hit.Floats(nil)
	if len(got) != len(src) {
		t.Fatalf("Floats length=%d, want %d", len(got), len(src))
	}
	for i := range src {
		if got[i] != src[i] {
			t.Errorf("Floats[%d]=%v, want %v", i, got[i], src[i])
		}
	}
	def := []float64{9.9}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrFloat, F: 1.0},
	}
	for _, a := range wrong {
		if got := a.Floats(def); len(got) != 1 || got[0] != 9.9 {
			t.Errorf("Floats on kind=%d did not return default", a.Kind)
		}
	}
}

func TestAttr_String_HitAndDefault(t *testing.T) {
	hit := Attr{Kind: AttrString, S: "hello"}
	if got := hit.String("def"); got != "hello" {
		t.Errorf("String=%q, want hello", got)
	}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrInt, I: 1},
		{Kind: AttrStrings, Ss: []string{"x"}},
	}
	for _, a := range wrong {
		if got := a.String("DEF"); got != "DEF" {
			t.Errorf("String on kind=%d returned %q, want default DEF", a.Kind, got)
		}
	}
}

func TestAttr_Strings_HitAndDefault(t *testing.T) {
	src := []string{"a", "b"}
	hit := Attr{Kind: AttrStrings, Ss: src}
	got := hit.Strings(nil)
	if len(got) != len(src) || got[0] != "a" || got[1] != "b" {
		t.Errorf("Strings=%v, want %v", got, src)
	}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrString, S: "x"},
	}
	def := []string{"d"}
	for _, a := range wrong {
		if got := a.Strings(def); len(got) != 1 || got[0] != "d" {
			t.Errorf("Strings on kind=%d did not return default", a.Kind)
		}
	}
}

func TestAttr_Tensor_HitAndNil(t *testing.T) {
	tp := &onnxpb.TensorProto{Name: "t"}
	hit := Attr{Kind: AttrTensor, T: tp}
	if got := hit.Tensor(); got != tp {
		t.Errorf("Tensor returned %p, want %p", got, tp)
	}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrInt, I: 1},
		{Kind: AttrGraph, G: &onnxpb.GraphProto{}},
	}
	for _, a := range wrong {
		if got := a.Tensor(); got != nil {
			t.Errorf("Tensor on kind=%d returned non-nil %p, want nil", a.Kind, got)
		}
	}
}

func TestAttr_Graph_HitAndNil(t *testing.T) {
	gp := &onnxpb.GraphProto{Name: "g"}
	hit := Attr{Kind: AttrGraph, G: gp}
	if got := hit.Graph(); got != gp {
		t.Errorf("Graph returned %p, want %p", got, gp)
	}
	wrong := []Attr{
		{Kind: AttrUnset},
		{Kind: AttrTensor, T: &onnxpb.TensorProto{}},
	}
	for _, a := range wrong {
		if got := a.Graph(); got != nil {
			t.Errorf("Graph on kind=%d returned non-nil %p, want nil", a.Kind, got)
		}
	}
}

// ── lowerAttr table ───────────────────────────────────────────────────────────

func TestLowerAttr_INT(t *testing.T) {
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_INT, I: 17}
	a := lowerAttr(pb)
	if a.Kind != AttrInt {
		t.Fatalf("Kind=%d, want AttrInt", a.Kind)
	}
	if a.I != 17 {
		t.Errorf("I=%d, want 17", a.I)
	}
}

func TestLowerAttr_INTS_CopiesSlice(t *testing.T) {
	src := []int64{1, 2, 3}
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_INTS, Ints: src}
	a := lowerAttr(pb)
	if a.Kind != AttrInts {
		t.Fatalf("Kind=%d, want AttrInts", a.Kind)
	}
	if len(a.Is) != 3 || a.Is[0] != 1 || a.Is[1] != 2 || a.Is[2] != 3 {
		t.Errorf("Is=%v, want [1 2 3]", a.Is)
	}
	// Mutate source: the lowered slice must not see the change (copy invariant).
	src[0] = 999
	if a.Is[0] == 999 {
		t.Errorf("lowered Is aliases the pb slice (mutation leaked)")
	}
}

func TestLowerAttr_FLOAT(t *testing.T) {
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_FLOAT, F: 1.5}
	a := lowerAttr(pb)
	if a.Kind != AttrFloat {
		t.Fatalf("Kind=%d, want AttrFloat", a.Kind)
	}
	if a.F != 1.5 {
		t.Errorf("F=%v, want 1.5", a.F)
	}
}

func TestLowerAttr_FLOATS(t *testing.T) {
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_FLOATS, Floats: []float32{1.0, 2.0, 3.5}}
	a := lowerAttr(pb)
	if a.Kind != AttrFloats {
		t.Fatalf("Kind=%d, want AttrFloats", a.Kind)
	}
	want := []float64{1.0, 2.0, 3.5}
	if len(a.Fs) != len(want) {
		t.Fatalf("Fs length=%d, want %d", len(a.Fs), len(want))
	}
	for i := range want {
		if a.Fs[i] != want[i] {
			t.Errorf("Fs[%d]=%v, want %v", i, a.Fs[i], want[i])
		}
	}
}

func TestLowerAttr_STRING(t *testing.T) {
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_STRING, S: []byte("relu")}
	a := lowerAttr(pb)
	if a.Kind != AttrString {
		t.Fatalf("Kind=%d, want AttrString", a.Kind)
	}
	if a.S != "relu" {
		t.Errorf("S=%q, want relu", a.S)
	}
}

func TestLowerAttr_STRINGS(t *testing.T) {
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_STRINGS, Strings: [][]byte{[]byte("a"), []byte("bc")}}
	a := lowerAttr(pb)
	if a.Kind != AttrStrings {
		t.Fatalf("Kind=%d, want AttrStrings", a.Kind)
	}
	if len(a.Ss) != 2 || a.Ss[0] != "a" || a.Ss[1] != "bc" {
		t.Errorf("Ss=%v, want [a bc]", a.Ss)
	}
}

func TestLowerAttr_TENSOR(t *testing.T) {
	tp := &onnxpb.TensorProto{Name: "k"}
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_TENSOR, T: tp}
	a := lowerAttr(pb)
	if a.Kind != AttrTensor {
		t.Fatalf("Kind=%d, want AttrTensor", a.Kind)
	}
	if a.T != tp {
		t.Errorf("T not aliased to source TensorProto")
	}
}

func TestLowerAttr_GRAPH(t *testing.T) {
	gp := &onnxpb.GraphProto{Name: "g"}
	pb := &onnxpb.AttributeProto{Name: "x", Type: onnxpb.AttributeProto_GRAPH, G: gp}
	a := lowerAttr(pb)
	if a.Kind != AttrGraph {
		t.Fatalf("Kind=%d, want AttrGraph", a.Kind)
	}
	if a.G != gp {
		t.Errorf("G not aliased to source GraphProto")
	}
}

// TestLowerAttr_Unsupported verifies SPARSE_TENSOR / TENSORS / GRAPHS /
// TYPE_PROTO / UNDEFINED lower to AttrUnset (the documented sentinel).
func TestLowerAttr_Unsupported(t *testing.T) {
	cases := []onnxpb.AttributeProto_AttributeType{
		onnxpb.AttributeProto_UNDEFINED,
		onnxpb.AttributeProto_SPARSE_TENSOR,
		onnxpb.AttributeProto_TENSORS,
		onnxpb.AttributeProto_GRAPHS,
		onnxpb.AttributeProto_TYPE_PROTO,
		onnxpb.AttributeProto_SPARSE_TENSORS,
		onnxpb.AttributeProto_TYPE_PROTOS,
	}
	for _, ty := range cases {
		t.Run(ty.String(), func(t *testing.T) {
			pb := &onnxpb.AttributeProto{Name: "x", Type: ty}
			a := lowerAttr(pb)
			if a.Kind != AttrUnset {
				t.Errorf("unsupported attr type %s: Kind=%d, want AttrUnset", ty, a.Kind)
			}
		})
	}
}

// ── lowerNode ─────────────────────────────────────────────────────────────────

func TestLowerNode_HappyPath(t *testing.T) {
	pb := &onnxpb.NodeProto{
		Name:   "conv1",
		OpType: "Conv",
		Domain: "ai.onnx",
		Input:  []string{"x", "w", "b"},
		Output: []string{"y"},
		Attribute: []*onnxpb.AttributeProto{
			{Name: "kernel_shape", Type: onnxpb.AttributeProto_INTS, Ints: []int64{3, 3}},
			{Name: "pads", Type: onnxpb.AttributeProto_INTS, Ints: []int64{1, 1, 1, 1}},
			{Name: "strides", Type: onnxpb.AttributeProto_INTS, Ints: []int64{1, 1}},
			{Name: "auto_pad", Type: onnxpb.AttributeProto_STRING, S: []byte("NOTSET")},
		},
	}
	n := lowerNode(pb)
	if n.OpType != "Conv" {
		t.Errorf("OpType=%q, want Conv", n.OpType)
	}
	if n.Domain != "ai.onnx" {
		t.Errorf("Domain=%q, want ai.onnx", n.Domain)
	}
	if n.Name != "conv1" {
		t.Errorf("Name=%q, want conv1", n.Name)
	}
	if len(n.Inputs) != 3 || n.Inputs[0] != "x" || n.Inputs[1] != "w" || n.Inputs[2] != "b" {
		t.Errorf("Inputs=%v, want [x w b]", n.Inputs)
	}
	if len(n.Outputs) != 1 || n.Outputs[0] != "y" {
		t.Errorf("Outputs=%v, want [y]", n.Outputs)
	}
	if len(n.Attrs) != 4 {
		t.Fatalf("Attrs length=%d, want 4", len(n.Attrs))
	}
	if ks := n.Attrs["kernel_shape"]; ks.Kind != AttrInts || len(ks.Is) != 2 || ks.Is[0] != 3 || ks.Is[1] != 3 {
		t.Errorf("kernel_shape attr=%+v, want AttrInts [3 3]", ks)
	}
	if ap := n.Attrs["auto_pad"]; ap.Kind != AttrString || ap.S != "NOTSET" {
		t.Errorf("auto_pad attr=%+v, want AttrString NOTSET", ap)
	}
	// Mutate pb input slice — lowered Inputs must not change.
	pb.Input[0] = "ZZZ"
	if n.Inputs[0] == "ZZZ" {
		t.Errorf("lowered Inputs aliases pb.Input (mutation leaked)")
	}
}

func TestLowerNode_EmptyEverything(t *testing.T) {
	pb := &onnxpb.NodeProto{OpType: "Relu"}
	n := lowerNode(pb)
	if n.OpType != "Relu" {
		t.Errorf("OpType=%q, want Relu", n.OpType)
	}
	if len(n.Inputs) != 0 {
		t.Errorf("Inputs not empty: %v", n.Inputs)
	}
	if len(n.Outputs) != 0 {
		t.Errorf("Outputs not empty: %v", n.Outputs)
	}
	if n.Attrs == nil {
		t.Errorf("Attrs is nil, want non-nil empty map")
	}
	if len(n.Attrs) != 0 {
		t.Errorf("Attrs not empty: %v", n.Attrs)
	}
}
