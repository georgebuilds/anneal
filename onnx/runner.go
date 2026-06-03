package onnx

import (
	"fmt"
	"os"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
	"google.golang.org/protobuf/proto"
)

// ValueInfo describes an ONNX graph input or output: its name, its anneal
// dtype, and its shape with any symbolic dims preserved as shape.SymInt.
type ValueInfo struct {
	Name  string
	Shape []shape.Sint
	DType *uop.DType
}

// HandlerCtx is the argument record passed to every device-tier handler.
// Handlers see the lowered Node (their OpType, attrs, in/out names), the
// arena to construct UOps on, the resolved inputs (Values), the resolved
// primary opset, and the device tag.
type HandlerCtx struct {
	Arena  *uop.Arena
	Device string
	Node   *Node
	Inputs []Value
	Opset  int64
}

// Handler is the signature for device-tier op implementations. Returning
// fewer or more outputs than node.Outputs is permitted: the dispatcher will
// reconcile by name, padding with the last-declared output for variadics
// (Phase 1.B concern; in 1.A we assume one-to-one mapping).
type Handler func(ctx *HandlerCtx) ([]Value, error)

// Runner is the importer's runtime object: it holds the arena, the lowered
// node list, the interned initializers, the resolved opset, and the handler
// registry. Construction parses and lowers; Run() walks the lowered graph.
type Runner struct {
	arena  *uop.Arena
	device string

	// graph is held only for the lifetime of Import so we can resolve graph
	// inputs/outputs; we clear the field at the end of Import to release
	// the rest of the protobuf payload to GC.
	graph *onnxpb.GraphProto

	nodes        []*Node
	initializers map[string]Value
	inputs       []ValueInfo
	outputs      []ValueInfo
	opset        int64

	handlers map[string]Handler
}

// Arena returns the runner's arena.
func (r *Runner) Arena() *uop.Arena { return r.arena }

// Device returns the device tag (currently a placeholder; v1 is single-device).
func (r *Runner) Device() string { return r.device }

// Opset returns the resolved primary ai.onnx opset version.
func (r *Runner) Opset() int64 { return r.opset }

// Nodes returns the lowered node list. Exposed for diagnostic / introspection
// use; do not mutate.
func (r *Runner) Nodes() []*Node { return r.nodes }

// Inputs returns the graph input descriptors.
func (r *Runner) Inputs() []ValueInfo { return r.inputs }

// Outputs returns the graph output descriptors.
func (r *Runner) Outputs() []ValueInfo { return r.outputs }

// Initializers returns the interned initializer table by name.
func (r *Runner) Initializers() map[string]Value { return r.initializers }

// RegisterHandler installs a device-tier op handler. Replaces any prior
// registration silently; this is the v1-friendly knob for tests and the
// canonical handlers in handlers.go.
func (r *Runner) RegisterHandler(opType string, h Handler) {
	r.handlers[opType] = h
}

// ImportFile parses a model from disk. Convenience wrapper over Import.
func ImportFile(path string, arena *uop.Arena, device string) (*Runner, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: read %q: %w", path, err)
	}
	return Import(data, arena, device)
}

// Import parses the ONNX protobuf bytes and constructs a Runner. The model's
// initializers are interned into the arena; the rest of the protobuf is
// dropped before this function returns.
//
// Opset resolution (plan §4): we pick the entry in opset_import whose Domain
// is "" or "ai.onnx" — that is the primary opset version. opset < 10 is a
// hard error (plan §4 cutoff); opset > 17 emits a warning (we don't refuse,
// but in-window correctness is what is exercised by tests).
func Import(modelBytes []byte, arena *uop.Arena, device string) (*Runner, error) {
	if arena == nil {
		return nil, fmt.Errorf("onnx.Import: nil arena")
	}

	var model onnxpb.ModelProto
	if err := proto.Unmarshal(modelBytes, &model); err != nil {
		return nil, fmt.Errorf("onnx: unmarshal ModelProto: %w", err)
	}
	if model.GetGraph() == nil {
		return nil, fmt.Errorf("onnx: model has no graph")
	}

	r := &Runner{
		arena:        arena,
		device:       device,
		graph:        model.GetGraph(),
		initializers: make(map[string]Value),
		handlers:     make(map[string]Handler),
	}

	// Resolve primary opset.
	opset, err := resolvePrimaryOpset(model.GetOpsetImport())
	if err != nil {
		return nil, err
	}
	r.opset = opset

	// Intern initializers (structural identity: identical TensorProtos share
	// the same arena leaf).
	type internEntry struct {
		v Value
	}
	internCache := make(map[[32]byte]internEntry)
	for _, tp := range r.graph.GetInitializer() {
		key := initializerHashKey(tp)
		if hit, ok := internCache[key]; ok {
			r.initializers[tp.GetName()] = hit.v
			continue
		}
		t, terr := tensorFromProto(arena, tp, device)
		if terr != nil {
			return nil, terr
		}
		v := Device(t)
		internCache[key] = internEntry{v: v}
		r.initializers[tp.GetName()] = v
	}

	// Lower nodes.
	r.nodes = make([]*Node, len(r.graph.GetNode()))
	for i, npb := range r.graph.GetNode() {
		r.nodes[i] = lowerNode(npb)
	}

	// Capture input / output descriptors (with symbolic dims preserved as
	// fresh DefineVar UOps, but only the *first* time a given dim_param
	// name is seen — same-name dim_params unify into a single Variable).
	r.inputs, err = lowerValueInfos(arena, r.graph.GetInput())
	if err != nil {
		return nil, fmt.Errorf("onnx: lowering graph inputs: %w", err)
	}
	r.outputs, err = lowerValueInfos(arena, r.graph.GetOutput())
	if err != nil {
		return nil, fmt.Errorf("onnx: lowering graph outputs: %w", err)
	}

	// Drop the protobuf payload. node-level Attr.T pointers continue to
	// reference TensorProtos in the now-unreferenced pb tree; that is fine,
	// the pb message will stay alive only as long as those pointers do.
	r.graph = nil

	// Install canonical handlers (Phase 1.B will populate this).
	RegisterAll(r)

	return r, nil
}

// resolvePrimaryOpset finds the OperatorSetIdProto with empty / "ai.onnx"
// domain and returns its version. Errors when the model has none, or when
// the version is < 10 (plan §4 cutoff). Warns when > 17.
func resolvePrimaryOpset(imports []*onnxpb.OperatorSetIdProto) (int64, error) {
	for _, im := range imports {
		if im.GetDomain() == "" || im.GetDomain() == "ai.onnx" {
			v := im.GetVersion()
			if v < 10 {
				return 0, fmt.Errorf("onnx: opset %d is below the supported floor (10)", v)
			}
			if v > 17 {
				Warn("opset %d is above the tested ceiling (17); proceeding without guarantees", v)
			}
			return v, nil
		}
	}
	return 0, fmt.Errorf("onnx: no ai.onnx opset_import found")
}

// lowerValueInfos translates ONNX ValueInfoProto entries into anneal-side
// descriptors. dim_param names lift to shape.SymInt backed by fresh
// DefineVar UOps; dim_value entries lift to shape.Const. Same-name
// dim_params seen earlier in the slice unify into the same DefineVar (the
// arena's FindDefineVar cache is the source of truth).
//
// Default symbolic bounds match the importer plan (min=1, max=4096). These
// can be overridden by handlers (Phase 1.B+) when a model carries explicit
// metadata; the defaults preserve correctness for v1.
func lowerValueInfos(arena *uop.Arena, vis []*onnxpb.ValueInfoProto) ([]ValueInfo, error) {
	const (
		symMin = int64(1)
		symMax = int64(4096)
	)
	out := make([]ValueInfo, 0, len(vis))
	for _, vi := range vis {
		var (
			sh []shape.Sint
			dt *uop.DType
		)
		if tt := vi.GetType().GetTensorType(); tt != nil {
			annealDT, _, _, ok := onnxDType(tt.GetElemType())
			if !ok {
				return nil, fmt.Errorf("onnx: input/output %q has unsupported elem_type %d", vi.GetName(), tt.GetElemType())
			}
			dt = annealDT
			if sp := tt.GetShape(); sp != nil {
				sh = make([]shape.Sint, 0, len(sp.GetDim()))
				for _, d := range sp.GetDim() {
					switch d.GetValue().(type) {
					case *onnxpb.TensorShapeProto_Dimension_DimValue:
						sh = append(sh, shape.Const(d.GetDimValue()))
					case *onnxpb.TensorShapeProto_Dimension_DimParam:
						name := d.GetDimParam()
						var v uop.UOp
						if existing, ok := arena.FindDefineVar(name); ok {
							v = existing
						} else {
							v = arena.DefineVar(name, symMin, symMax)
						}
						sh = append(sh, shape.SymInt{Node: v})
					default:
						// Rank-only dim with no value or param: treat as
						// a fresh symbolic dim with a synthetic name.
						name := fmt.Sprintf("%s_dim%d", vi.GetName(), len(sh))
						v := arena.DefineVar(name, symMin, symMax)
						sh = append(sh, shape.SymInt{Node: v})
					}
				}
			}
		}
		out = append(out, ValueInfo{Name: vi.GetName(), Shape: sh, DType: dt})
	}
	return out, nil
}

// Run walks the lowered node list once, threading values through a per-Run
// state map seeded with the initializers and the supplied named inputs.
// Outputs are returned by name; missing graph outputs become an error.
//
// Dispatch order (per node):
//  1. Resolve each named input from the state map. An unresolved input is
//     an error (we don't attempt forward references).
//  2. If the op is host-evaluable AND every input is a HostValue, run the
//     host interpreter.
//  3. Otherwise look up the device-tier handler. If none, return a
//     descriptive "unsupported op" error (plan §2: punt loudly).
//
// The "unsupported op" error path is the load-bearing site that Phase 1.A
// is gating on: it proves the dispatch loop walks the graph and surfaces
// missing handlers as a clean failure rather than a silent wrong output.
func (r *Runner) Run(named map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	state := make(map[string]Value, len(r.initializers)+len(named))
	for k, v := range r.initializers {
		state[k] = v
	}
	for k, t := range named {
		state[k] = Device(t)
	}

	host := NewHostState()

	for _, node := range r.nodes {
		// Resolve inputs.
		inputs := make([]Value, len(node.Inputs))
		allHost := true
		for i, name := range node.Inputs {
			if name == "" {
				// Optional absent input. Leave as zero Value; handlers
				// must check IsUnset themselves.
				allHost = false
				continue
			}
			v, ok := state[name]
			if !ok {
				// Try host state (some host ops emit into it directly).
				if hv, hok := host.Get(name); hok {
					v = hv
					ok = true
				}
			}
			if !ok {
				return nil, fmt.Errorf("onnx: node %q (op %q): unresolved input %q",
					node.Name, node.OpType, name)
			}
			inputs[i] = v
			if !v.IsHost() {
				allHost = false
			}
		}

		// Host-tier dispatch.
		if IsHostOp(node.OpType) && allHost {
			out, err := evalHost(node, host)
			if err != nil {
				return nil, fmt.Errorf("onnx: host eval of %q (node %q): %w",
					node.OpType, node.Name, err)
			}
			if len(node.Outputs) > 0 {
				name := node.Outputs[0]
				host.Set(name, out)
				state[name] = out
			}
			continue
		}

		// Device-tier dispatch.
		h, ok := r.handlers[node.OpType]
		if !ok {
			return nil, fmt.Errorf("onnx: unsupported op %q (node %q, inputs=%v, outputs=%v)",
				node.OpType, node.Name, node.Inputs, node.Outputs)
		}
		ctx := &HandlerCtx{
			Arena:  r.arena,
			Device: r.device,
			Node:   node,
			Inputs: inputs,
			Opset:  r.opset,
		}
		outs, err := h(ctx)
		if err != nil {
			return nil, fmt.Errorf("onnx: handler for %q (node %q): %w",
				node.OpType, node.Name, err)
		}
		if len(outs) > len(node.Outputs) {
			return nil, fmt.Errorf("onnx: handler %q produced %d outputs, declared %d",
				node.OpType, len(outs), len(node.Outputs))
		}
		for i, v := range outs {
			state[node.Outputs[i]] = v
		}
	}

	// Materialise graph outputs.
	out := make(map[string]*tensor.Tensor, len(r.outputs))
	for _, oi := range r.outputs {
		v, ok := state[oi.Name]
		if !ok {
			return nil, fmt.Errorf("onnx: graph output %q never assigned", oi.Name)
		}
		if !v.IsDevice() {
			return nil, fmt.Errorf("onnx: graph output %q is host-tier, want device tensor", oi.Name)
		}
		out[oi.Name] = v.Tensor()
	}
	return out, nil
}
