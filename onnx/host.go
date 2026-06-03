package onnx

import (
	"fmt"
)

// HostState carries host-tier values during a Run. It is a tiny integer
// interpreter for the shape subgraph: Shape, Size, ConstantOfShape's shape
// arg, Reshape's target shape, Slice's starts/ends/axes/steps, etc.
//
// Phase 1.A lays out the dispatch surface; concrete host ops will be added in
// Phase 1.B alongside the device-tier handlers. Until then, every Eval call
// returns a "host op not implemented" error so unknown shape-side paths fail
// loudly rather than silently producing wrong shapes.
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
// Phase 1.B will populate the table below; the dispatch loop in runner.go
// already routes ops that resolve to all-host inputs through evalHost.
type HostOpHandler func(node *Node, st *HostState) (Value, error)

// hostOps is the host-tier op table. Empty in Phase 1.A; handlers register at
// init time once they exist.
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
// the op is not implemented; the dispatcher in runner.go uses that as the
// signal to fall through to the device-tier handler table.
func evalHost(node *Node, st *HostState) (Value, error) {
	h, ok := hostOps[node.OpType]
	if !ok {
		return Value{}, fmt.Errorf("onnx: host op %q not implemented", node.OpType)
	}
	return h(node, st)
}
