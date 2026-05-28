// Package viz implements the web-based UOp graph visualizer. It runs the real
// anneal compiler (frontend + rewrite engine + scheduler) compiled Go→WASM and
// renders the actual UOp graph in the browser — never a mock.
package viz

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// Node class constants (provenance).
const (
	ClassForward  = "forward"  // forward pass — teal
	ClassBackward = "backward" // backward pass — ember
)

// Node kind constants (structural shape hint for the renderer).
const (
	KindDefault = "default" // circle — regular computation
	KindLeaf    = "leaf"    // rounded rect — OpBuffer input/parameter
	KindReduce  = "reduce"  // diamond — OpReduceAxis (real kernel boundary in v1)
	KindSink    = "sink"    // hexagon — OpSink aggregation point
)

// GraphData is the serializable UOp graph for the browser renderer.
type GraphData struct {
	Name  string     `json:"name"`
	Nodes []NodeData `json:"nodes"`
	Edges []EdgeData `json:"edges"`
	Stats GraphStats `json:"stats"`
}

// NodeData describes one node in the visualization.
type NodeData struct {
	ID           uint32  `json:"id"`
	Op           string  `json:"op"`
	DType        string  `json:"dtype"`
	Shape        []int64 `json:"shape,omitempty"`
	Class        string  `json:"class"` // ClassForward or ClassBackward
	Kind         string  `json:"kind"`  // KindDefault, KindLeaf, KindReduce, KindSink
	Label        string  `json:"label"`
	Arg          string  `json:"arg,omitempty"`
	GradRule     string  `json:"gradRule,omitempty"`
	GradFiredSeq int     `json:"gradFiredSeq,omitempty"`
}

// EdgeData is a directed edge from source to consumer in the DAG.
type EdgeData struct {
	From uint32 `json:"from"`
	To   uint32 `json:"to"`
}

// GraphStats holds compilation summary statistics.
type GraphStats struct {
	FwdNodes int `json:"fwdNodes"`
	BwdNodes int `json:"bwdNodes"`
	Kernels  int `json:"kernels"`
	AllNodes int `json:"allNodes"`
}

// ToJSON serializes g to canonical JSON bytes.
func (g *GraphData) ToJSON() ([]byte, error) {
	return json.Marshal(g)
}

// BuildGraph builds the complete forward + backward UOp graph for the named
// example, runs the real rangeify scheduler to determine kernel count, and
// returns a serializable GraphData suitable for the browser renderer.
//
// DD2 compliance: the real anneal compiler (frontend + rewrite + scheduler)
// runs in this call. No GPU execution is performed — graph construction and
// scheduling are pure-Go graph operations that run identically on native and
// WASM targets.
//
// Cross-build render-stability: node IDs are arena indices, which are
// construction-order-dependent per SPEC §1.3 / AGENTS.md. This renders one
// compilation faithfully. The visualizer makes no cross-build comparisons and
// does not assume render-stability across reloads — this is the documented safe
// v1 posture.
func BuildGraph(name string) (*GraphData, error) {
	ex, err := examples.Get(name)
	if err != nil {
		return nil, err
	}
	result, err := ex.Build("webgpu")
	if err != nil {
		return nil, fmt.Errorf("viz: build %q: %w", name, err)
	}

	a := result.Arena
	out := result.Output

	// Reduce output to a scalar so Backward has a single root adjoint.
	loss := out.Sum(nil, false)

	// Run the real backward traversal (tensor/gradient.go typed traversal —
	// NOT a PatternMatcher ruleset, per SPEC §5). Each new UOp node built
	// inside Backward is stamped PhaseBackward by the arena's phase field;
	// forward nodes reused via interning keep their original PhaseForward.
	var grads map[*tensor.Tensor]*tensor.Tensor
	var trace *tensor.GradTrace
	if len(result.Leaves) > 0 {
		grads, trace = tensor.BackwardWithTrace(loss, result.Leaves)
	}

	// Collect all output roots: the loss and each gradient tensor.
	// These are the "live outputs" that a training step would realize.
	vizRoots := make([]uop.UOp, 0, 1+len(grads))
	vizRoots = append(vizRoots, loss.Node())
	for _, g := range grads {
		vizRoots = append(vizRoots, g.Node())
	}

	// Build the scheduler SINK over ALL outputs (forward loss + all gradients)
	// so CreateSchedule sees the complete forward+backward computation and
	// reports the true kernel count for a full training step.
	schedSink := a.New(uop.OpSink, uop.Dtypes.Void, vizRoots, nil, nil)

	// Run the real scheduler (passes 1–10, no GPU execution).
	items := schedule.CreateSchedule(schedSink, "webgpu")
	kernelCount := len(items)

	// Walk all nodes reachable from the viz roots, producing a topological
	// (sources-before-consumers) order. Scheduler-internal nodes (at indices
	// >= bwdEnd) are not reachable from any viz root and are excluded.
	topo := topoSortMultiRoot(vizRoots)

	nodeSet := make(map[uint32]bool, len(topo))
	for _, u := range topo {
		nodeSet[u.Index()] = true
	}

	var nodes []NodeData
	var edges []EdgeData
	fwdCount, bwdCount := 0, 0

	type ruleInfo struct {
		rule string
		seq  int
	}
	attribution := make(map[uint32]ruleInfo)
	if trace != nil {
		for _, ev := range trace.Events {
			for _, idx := range ev.ProducedIdx {
				if idx != tensor.TraceSentinel {
					if _, exists := attribution[idx]; !exists {
						attribution[idx] = ruleInfo{ev.ForwardOp.String(), ev.Seq}
					}
				}
			}
		}
	}

	for _, u := range topo {
		// Classify using durable per-node provenance stamped at construction time.
		// Scheduler nodes are not reachable from vizRoots and do not appear here.
		class := ClassForward
		if a.Provenance(u.Index()) == uop.PhaseBackward {
			class = ClassBackward
		}

		if class == ClassForward {
			fwdCount++
		} else {
			bwdCount++
		}

		firedSeq := -1
		var rule string
		if class == ClassBackward {
			if info, ok := attribution[u.Index()]; ok {
				rule = info.rule
				firedSeq = info.seq
			}
		}

		nd := NodeData{
			ID:           u.Index(),
			Op:           u.Op().String(),
			DType:        dtypeStr(u.DType()),
			Shape:        bufShape(u),
			Class:        class,
			Kind:         kindOf(u, a),
			Label:        nodeLabel(u),
			Arg:          argStr(u),
			GradRule:     rule,
			GradFiredSeq: firedSeq,
		}
		nodes = append(nodes, nd)

		for i := 0; i < u.NSrc(); i++ {
			src := u.Src(i)
			if nodeSet[src.Index()] {
				edges = append(edges, EdgeData{From: src.Index(), To: u.Index()})
			}
		}
	}

	sortEdges(edges)

	return &GraphData{
		Name:  name,
		Nodes: nodes,
		Edges: edges,
		Stats: GraphStats{
			FwdNodes: fwdCount,
			BwdNodes: bwdCount,
			Kernels:  kernelCount,
			AllNodes: len(nodes),
		},
	}, nil
}

// topoSortMultiRoot returns the DFS post-order (sources before consumers) of
// all nodes reachable from any of the given roots, without duplication.
// Uses an iterative stack to handle graphs of arbitrary depth safely.
func topoSortMultiRoot(roots []uop.UOp) []uop.UOp {
	seen := make(map[uint32]bool)
	var order []uop.UOp

	type frame struct {
		u    uop.UOp
		next int
	}

	// Seed the stack with all roots (reversed so the first root is processed first).
	stack := make([]frame, 0, len(roots)*4)
	for i := len(roots) - 1; i >= 0; i-- {
		stack = append(stack, frame{roots[i], 0})
	}

	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u
		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}
		pushed := false
		for f.next < u.NSrc() {
			ch := u.Src(f.next)
			f.next++
			if !seen[ch.Index()] {
				stack = append(stack, frame{ch, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			order = append(order, u)
			stack = stack[:len(stack)-1]
		}
	}
	return order
}

// kindOf returns the display kind for a UOp node.
func kindOf(u uop.UOp, a *uop.Arena) string {
	switch u.Op() {
	case uop.OpSink:
		return KindSink
	case uop.OpReduceAxis:
		// ReduceAxis nodes are hard kernel boundaries in v1 (removeBufferize is a
		// no-op stub — every reduce materialises to a buffer). Gold diamond shape.
		return KindReduce
	case uop.OpBuffer:
		return KindLeaf
	}
	return KindDefault
}

func dtypeStr(dt *uop.DType) string {
	if dt == nil || dt == uop.Dtypes.Void {
		return "void"
	}
	switch dt {
	case uop.Dtypes.Float32:
		return "f32"
	case uop.Dtypes.Float16:
		return "f16"
	case uop.Dtypes.Float64:
		return "f64"
	case uop.Dtypes.Int32:
		return "i32"
	case uop.Dtypes.UInt32:
		return "u32"
	case uop.Dtypes.Bool:
		return "bool"
	case uop.Dtypes.Index:
		return "idx"
	}
	return dt.String()
}

func argStr(u uop.UOp) string {
	switch v := u.Arg().(type) {
	case int64:
		return fmt.Sprintf("%d", v)
	case float64:
		return fmt.Sprintf("%g", v)
	case bool:
		if v {
			return "true"
		}
		return "false"
	case string:
		if len(v) > 16 {
			return v[:16] + "…"
		}
		return v
	case []int64:
		return fmt.Sprintf("%v", v)
	case uop.ReduceArg:
		return fmt.Sprintf("%s%v", v.Op, v.Axes)
	case uop.RangeArg:
		t := "loop"
		if v.Type == uop.AxisReduce {
			t = "red"
		}
		return fmt.Sprintf("[0,%d)%s", v.Size, t)
	}
	return ""
}

func nodeLabel(u uop.UOp) string {
	switch u.Op() {
	case uop.OpBuffer:
		sh := bufShape(u)
		if len(sh) == 0 {
			return "Buffer"
		}
		return fmt.Sprintf("Buffer%v", sh)
	case uop.OpConst:
		a := argStr(u)
		if a == "" {
			return "Const"
		}
		return "Const(" + a + ")"
	case uop.OpReduceAxis:
		ra, _ := u.Arg().(uop.ReduceArg)
		return ra.Op.String() + "-reduce"
	case uop.OpSink:
		return "Sink"
	}
	return u.Op().String()
}

func bufShape(u uop.UOp) []int64 {
	switch v := u.Arg().(type) {
	case []int64:
		return v
	case int64:
		return []int64{v}
	}
	return nil
}

func sortEdges(edges []EdgeData) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
}
