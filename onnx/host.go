package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
)

// HostState carries host-tier values during a Run. It is a tiny integer
// interpreter for the shape subgraph: Shape, Size, ConstantOfShape's shape
// arg, Reshape's target shape, Slice's starts/ends/axes/steps, etc.
type HostState struct {
	// values is the per-Run host-tier state, keyed by ONNX value name.
	values map[string]Value
}

// NewHostState returns an empty host state.
func NewHostState() *HostState {
	return &HostState{values: make(map[string]Value)}
}

// Get returns the host value for name, or (zero, false) if absent.
func (s *HostState) Get(name string) (Value, bool) {
	v, ok := s.values[name]
	return v, ok
}

// Set stores a host value under name.
func (s *HostState) Set(name string, v Value) { s.values[name] = v }

// HostOpHandler is the signature for host-tier op implementations.
// Inputs are the resolved Values (in node.Inputs order) so handlers see the
// same view the device-tier handlers see; the HostState is provided for ops
// that may need to write side-channel host values (none currently). Returning
// a Value sets it under node.Outputs[0] in both host and device state maps
// (the runner reconciles).
type HostOpHandler func(node *Node, inputs []Value, st *HostState) (Value, error)

// hostOps is the host-tier op table.
var hostOps = map[string]HostOpHandler{}

// RegisterHostOp installs a host-tier op handler. Panics on double-register
// to surface accidental duplicate registrations.
func RegisterHostOp(name string, h HostOpHandler) {
	if _, ok := hostOps[name]; ok {
		panic(fmt.Sprintf("onnx: host op %q already registered", name))
	}
	hostOps[name] = h
}

// IsHostOp reports whether name has a host-tier handler registered.
func IsHostOp(name string) bool {
	_, ok := hostOps[name]
	return ok
}

// evalHost dispatches a node to its host-tier handler. Returns an error if
// the op is not implemented.
func evalHost(node *Node, inputs []Value, st *HostState) (Value, error) {
	h, ok := hostOps[node.OpType]
	if !ok {
		return Value{}, fmt.Errorf("onnx: host op %q not implemented", node.OpType)
	}
	return h(node, inputs, st)
}

// ErrFallThroughToDevice is the sentinel a host op returns to ask the runner
// to dispatch to the device-tier handler instead. Used by ops that are
// host-evaluable for some attribute shapes (e.g. Constant with integer
// tensors) but device-tier for others (e.g. Constant with float tensors).
var ErrFallThroughToDevice = fmt.Errorf("onnx: fall through to device handler")

// isHostInput reports whether v is the kind of input a host-tier op can
// consume directly. Shape/Size accept Device tensors (they read the shape,
// not the data); all other host ops require host-tier inputs.
func isHostInput(opType string, v Value) bool {
	if v.IsHost() {
		return true
	}
	// Shape, Size, and Identity (when passing through a device tensor)
	// are host ops that can be evaluated when the input is device-tier
	// — we read v.Tensor().ShapeSints() without realising. Other host ops
	// (host arithmetic, host Concat, Range, ConstantOfShape) require host
	// inputs.
	switch opType {
	case "Shape", "Size":
		return v.IsDevice()
	}
	return false
}

// hostInputsOK reports whether every input is host-evaluable for the given op.
// Empty / unset inputs (optional absent) are considered OK; the handler is
// responsible for validating presence.
func hostInputsOK(opType string, inputs []Value) bool {
	for _, v := range inputs {
		if v.Kind == KindUnset {
			continue
		}
		if !isHostInput(opType, v) {
			return false
		}
	}
	return true
}

// shapeSintsAsInts attempts to convert a []shape.Sint to []int64. Returns
// (nil, false) if any element is symbolic.
func shapeSintsAsInts(sh []shape.Sint) ([]int64, bool) {
	out := make([]int64, len(sh))
	for i, s := range sh {
		v, ok := s.ConstValue()
		if !ok {
			return nil, false
		}
		out[i] = v
	}
	return out, true
}
