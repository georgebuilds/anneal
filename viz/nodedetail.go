// nodedetail.go — node-inspector backend (W4).
//
// BuildNodeDetail(graphId, nodeId) walks the same compile path BuildGraph
// uses, then resolves one node from the rendered union graph. The studio's
// visualize view embeds the existing viz artifact verbatim; clicking a node
// in the iframe posts {type:"nodeClick", graphId, nodeId} to the parent, the
// parent calls annealNodeDetail across the worker, and the drawer renders
// the returned NodeDetail JSON. DD2: the data is real-compiler output, never
// a hand-curated label.
//
// Source file:line info is not carried by the UOp arena today, so the
// SourceFile / SourceLine fields are always omitted by this implementation.
// They stay on the JSON contract so a future frontend trace-info hook can
// fill them without breaking the studio's drawer renderer.

package viz

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/georgebuilds/anneal/uop"
)

// NodeDetail is the JSON payload returned by annealNodeDetail. Mirrors the
// contract in notes/anneal_web_spec.md §4 / §5.2.
type NodeDetail struct {
	GraphID    string        `json:"graph_id"`
	NodeID     string        `json:"node_id"`
	Op         string        `json:"op"`
	DType      string        `json:"dtype"`
	Shape      []string      `json:"shape"`         // serialized shape.Sint values
	Phase      string        `json:"phase"`         // forward / backward / fused
	Parents    []ParentChild `json:"parents"`       // upstream nodes
	Children   []ParentChild `json:"children"`      // downstream nodes
	Arg        string        `json:"arg,omitempty"` // const value, axis, var name, etc.
	SourceFile string        `json:"source_file,omitempty"`
	SourceLine int           `json:"source_line,omitempty"`
}

// ParentChild is one entry in NodeDetail.Parents / .Children. The Label is
// the same human-friendly form the graph node renders (Buffer[..], Const(..),
// Add etc.); the ID is the string form ("n<index>") the iframe uses.
type ParentChild struct {
	ID    string `json:"id"`
	Op    string `json:"op"`
	Label string `json:"label"`
}

// ToJSON serializes nd to canonical JSON bytes.
func (nd *NodeDetail) ToJSON() ([]byte, error) { return json.Marshal(nd) }

// BuildNodeDetail builds the named example, runs forward + backward, finds
// the node by its viz-side ID ("n<index>"), and returns a NodeDetail with
// op / dtype / shape / phase / parents / children / arg populated.
//
// Errors are surfaced verbatim: unknown graphId returns the example registry
// error; an unknown node returns an explicit "node not found" message; a
// malformed nodeId returns a parse error.
//
// The walk reuses BuildGraph so the produced parent/child sets exactly match
// what the visualize view shows the user. Edges outside the rendered union
// (scheduler-internal nodes) are not reported as parents or children even if
// they exist in the arena — this keeps the inspector consistent with what is
// visible on screen.
func BuildNodeDetail(graphID, nodeID string) (*NodeDetail, error) {
	if graphID == "" {
		return nil, fmt.Errorf("viz: empty graphId")
	}
	if nodeID == "" {
		return nil, fmt.Errorf("viz: empty nodeId")
	}

	idx, err := parseNodeID(nodeID)
	if err != nil {
		return nil, err
	}

	// Build the same union graph the visualize view renders. Anything not
	// reachable from a viz root is excluded (the user can not click it).
	g, err := BuildGraph(graphID)
	if err != nil {
		return nil, err
	}

	// Index nodes by ID for O(1) lookup, and build a children index from
	// the rendered edges so we never report a child that is not visible.
	nodesByID := make(map[uint32]*NodeData, len(g.Nodes))
	for i := range g.Nodes {
		nodesByID[g.Nodes[i].ID] = &g.Nodes[i]
	}

	target, ok := nodesByID[idx]
	if !ok {
		return nil, fmt.Errorf("viz: node %q not found in graph %q", nodeID, graphID)
	}

	// Children index: for each node, the set of downstream nodes in the
	// rendered DAG (rather than walking the arena from the consumer side,
	// which would include scheduler-internal sinks).
	childrenByID := make(map[uint32][]uint32, len(g.Nodes))
	for _, e := range g.Edges {
		childrenByID[e.From] = append(childrenByID[e.From], e.To)
	}

	// Parents come from the edge list filtered to "edges ending on idx";
	// children from the children index. Both sets preserve the union-graph
	// topological order (parents in source-before-sink order, children in
	// the same).
	parents := make([]ParentChild, 0, 4)
	for _, e := range g.Edges {
		if e.To != idx {
			continue
		}
		p, ok := nodesByID[e.From]
		if !ok {
			continue
		}
		parents = append(parents, ParentChild{
			ID:    formatNodeID(p.ID),
			Op:    p.Op,
			Label: p.Label,
		})
	}
	childIDs := childrenByID[idx]
	children := make([]ParentChild, 0, len(childIDs))
	for _, cid := range childIDs {
		c, ok := nodesByID[cid]
		if !ok {
			continue
		}
		children = append(children, ParentChild{
			ID:    formatNodeID(c.ID),
			Op:    c.Op,
			Label: c.Label,
		})
	}

	return &NodeDetail{
		GraphID:  graphID,
		NodeID:   nodeID,
		Op:       target.Op,
		DType:    target.DType,
		Shape:    shapeStrings(target.Shape),
		Phase:    target.Class, // forward / backward (fused emitted by future kernel-aware path)
		Parents:  parents,
		Children: children,
		Arg:      target.Arg,
	}, nil
}

// parseNodeID parses a viz-format node ID ("n<index>") into a uint32. Accepts
// either "n<index>" or a bare numeric string so old harnesses keep working.
func parseNodeID(s string) (uint32, error) {
	t := strings.TrimPrefix(s, "n")
	v, err := strconv.ParseUint(t, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("viz: parse nodeId %q: %w", s, err)
	}
	return uint32(v), nil
}

// formatNodeID is the inverse of parseNodeID.
func formatNodeID(idx uint32) string { return fmt.Sprintf("n%d", idx) }

// shapeStrings renders each dim as a string. The shape carried on NodeData
// is []int64 (concrete dims); a future symbolic-aware path will widen to
// shape.Sint and call its String() form. For now we stringify each int64
// directly so the JSON contract is shape: ["8", "2"] not [8, 2].
//
// Returning [] for an empty / nil shape keeps the JSON shape stable (an empty
// array, not null).
func shapeStrings(shape []int64) []string {
	out := make([]string, 0, len(shape))
	for _, d := range shape {
		out = append(out, strconv.FormatInt(d, 10))
	}
	return out
}

// formatPhaseForUOp resolves a UOp's per-construction Phase to the JSON
// "forward" / "backward" enumeration the studio's drawer renders. Reserved
// for a future caller that wants to attribute by the arena directly rather
// than the rendered NodeData.Class field.
//
//nolint:unused // forward-compatible helper; exercised by Phase audits.
func formatPhaseForUOp(a *uop.Arena, idx uint32) string {
	if a == nil || idx >= uint32(a.Len()) {
		return ClassForward
	}
	if a.Provenance(idx) == uop.PhaseBackward {
		return ClassBackward
	}
	return ClassForward
}
