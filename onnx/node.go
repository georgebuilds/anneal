package onnx

import (
	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

// Node is anneal's lowered form of an ONNX NodeProto. Once construction is
// complete we drop the protobuf representation entirely; the lowered Node form
// is what the dispatcher and op handlers see. This matches the implementation
// plan §2 ("discard the protobuf structs immediately; do not retain them past
// construction") and is the API surface duck-typed by handlers in handlers.go.
type Node struct {
	OpType  string          // e.g. "Conv", "Relu", "Reshape"
	Domain  string          // "" or "ai.onnx" for the standard domain
	Inputs  []string        // input names; "" denotes an absent optional input
	Outputs []string        // output names
	Attrs   map[string]Attr // by attribute name
	Name    string          // optional node name; surfaces in error messages
}

// AttrKind discriminates the (effectively-union) Attr payload.
type AttrKind int

const (
	AttrUnset AttrKind = iota
	AttrInt
	AttrInts
	AttrFloat
	AttrFloats
	AttrString
	AttrStrings
	AttrTensor
	AttrGraph
)

// Attr is the lowered attribute carrier. Exactly one payload field is
// meaningful for a given Kind; the typed accessors (Int / Ints / Float / ...)
// return a default when the attribute is absent (zero Kind), which is the
// idiomatic "duck-typed" path used by handlers. See implementation plan §4.
type Attr struct {
	Kind AttrKind

	I  int64
	Is []int64
	F  float64
	Fs []float64
	S  string
	Ss []string

	// T holds an attribute-tensor (e.g. Constant's value attribute). We keep
	// it by reference because the rest of the protobuf is dropped early; this
	// pointer is the only retained pb backreference and it lives for the
	// lifetime of the Runner.
	T *onnxpb.TensorProto

	// G holds an attribute-subgraph (e.g. If/Loop's body). Control flow is a
	// non-goal for v1; the slot exists so we can recognise these and reject
	// them at dispatch time.
	G *onnxpb.GraphProto
}

// Int returns the attribute's int value, or def when the attribute is absent.
// Phase 1.A intentionally returns def on type mismatch as well so that
// optional-with-default semantics are uniform.
func (a Attr) Int(def int64) int64 {
	if a.Kind == AttrInt {
		return a.I
	}
	return def
}

// Ints returns the attribute's int vector, or def when the attribute is
// absent or of a different kind.
func (a Attr) Ints(def []int64) []int64 {
	if a.Kind == AttrInts {
		return a.Is
	}
	return def
}

// Float returns the attribute's float value, or def when absent.
func (a Attr) Float(def float64) float64 {
	if a.Kind == AttrFloat {
		return a.F
	}
	return def
}

// Floats returns the attribute's float vector, or def when absent.
func (a Attr) Floats(def []float64) []float64 {
	if a.Kind == AttrFloats {
		return a.Fs
	}
	return def
}

// String returns the attribute's string value, or def when absent.
func (a Attr) String(def string) string {
	if a.Kind == AttrString {
		return a.S
	}
	return def
}

// Strings returns the attribute's string vector, or def when absent.
func (a Attr) Strings(def []string) []string {
	if a.Kind == AttrStrings {
		return a.Ss
	}
	return def
}

// Tensor returns the attribute's TensorProto, or nil when absent.
func (a Attr) Tensor() *onnxpb.TensorProto {
	if a.Kind == AttrTensor {
		return a.T
	}
	return nil
}

// Graph returns the attribute's GraphProto, or nil when absent.
func (a Attr) Graph() *onnxpb.GraphProto {
	if a.Kind == AttrGraph {
		return a.G
	}
	return nil
}

// lowerAttr converts a protobuf AttributeProto into anneal's lowered Attr.
// The protobuf type tag is the source of truth (per IR_VERSION ≥ 0.0.2; see
// the AttributeProto.Type comment in the .proto file).
func lowerAttr(pb *onnxpb.AttributeProto) Attr {
	switch pb.GetType() {
	case onnxpb.AttributeProto_INT:
		return Attr{Kind: AttrInt, I: pb.GetI()}
	case onnxpb.AttributeProto_INTS:
		// Copy the slice so we can drop the pb safely.
		out := make([]int64, len(pb.GetInts()))
		copy(out, pb.GetInts())
		return Attr{Kind: AttrInts, Is: out}
	case onnxpb.AttributeProto_FLOAT:
		return Attr{Kind: AttrFloat, F: float64(pb.GetF())}
	case onnxpb.AttributeProto_FLOATS:
		src := pb.GetFloats()
		out := make([]float64, len(src))
		for i, v := range src {
			out[i] = float64(v)
		}
		return Attr{Kind: AttrFloats, Fs: out}
	case onnxpb.AttributeProto_STRING:
		return Attr{Kind: AttrString, S: string(pb.GetS())}
	case onnxpb.AttributeProto_STRINGS:
		src := pb.GetStrings()
		out := make([]string, len(src))
		for i, v := range src {
			out[i] = string(v)
		}
		return Attr{Kind: AttrStrings, Ss: out}
	case onnxpb.AttributeProto_TENSOR:
		return Attr{Kind: AttrTensor, T: pb.GetT()}
	case onnxpb.AttributeProto_GRAPH:
		return Attr{Kind: AttrGraph, G: pb.GetG()}
	}
	return Attr{Kind: AttrUnset}
}

// lowerNode converts an ONNX NodeProto into anneal's lowered Node form.
func lowerNode(pb *onnxpb.NodeProto) *Node {
	inputs := make([]string, len(pb.GetInput()))
	copy(inputs, pb.GetInput())
	outputs := make([]string, len(pb.GetOutput()))
	copy(outputs, pb.GetOutput())

	attrs := make(map[string]Attr, len(pb.GetAttribute()))
	for _, a := range pb.GetAttribute() {
		attrs[a.GetName()] = lowerAttr(a)
	}
	return &Node{
		OpType:  pb.GetOpType(),
		Domain:  pb.GetDomain(),
		Inputs:  inputs,
		Outputs: outputs,
		Attrs:   attrs,
		Name:    pb.GetName(),
	}
}
