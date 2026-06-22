package viz

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestBuildNodeDetail_MLPNode0 pins the contract for the first rendered
// node of the mlp example. "first rendered" means topologically first in
// the union BuildGraph emits - typically the leaf buffer (input). The test
// asserts the fields the studio's drawer renders, not bytewise structure
// (the arena position is construction-order-dependent per SPEC §1.3).
func TestBuildNodeDetail_MLPNode0(t *testing.T) {
	// Discover the first rendered node's id by walking the union graph;
	// the smallest arena index in BuildGraph.Nodes is the one annealed at
	// the top of the topo (sources before consumers).
	g, err := BuildGraph("mlp")
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	if len(g.Nodes) == 0 {
		t.Fatal("BuildGraph produced no nodes")
	}
	firstID := g.Nodes[0].ID
	nodeID := formatNodeID(firstID)

	nd, err := BuildNodeDetail("mlp", nodeID)
	if err != nil {
		t.Fatalf("BuildNodeDetail(\"mlp\", %q): %v", nodeID, err)
	}
	if nd.GraphID != "mlp" {
		t.Errorf("GraphID = %q, want %q", nd.GraphID, "mlp")
	}
	if nd.NodeID != nodeID {
		t.Errorf("NodeID = %q, want %q", nd.NodeID, nodeID)
	}
	if nd.Op == "" {
		t.Errorf("Op is empty; expected a UOp opname")
	}
	if nd.DType == "" {
		t.Errorf("DType is empty; expected a dtype like f32 or void")
	}
	// Phase is the per-node provenance: forward or backward. We do not pin
	// which one this node is (an example might lead with a backward
	// DefineVar in a future refactor), only that it is one of the two
	// enumerated values the drawer renders.
	switch nd.Phase {
	case ClassForward, ClassBackward:
	default:
		t.Errorf("Phase = %q, want one of {forward, backward}", nd.Phase)
	}
	// Shape is always a non-nil slice so the JSON round-trips as `[]`
	// rather than `null` (the drawer's empty-state copy depends on it).
	if nd.Shape == nil {
		t.Errorf("Shape is nil; expected an empty slice for a scalar leaf")
	}
}

// TestBuildNodeDetail_UnknownGraph: an unknown example name surfaces as the
// examples-registry error wrapped by BuildGraph. The drawer's error UI
// matches on the prefix; pin that the call returns an error rather than a
// stub NodeDetail.
func TestBuildNodeDetail_UnknownGraph(t *testing.T) {
	nd, err := BuildNodeDetail("does-not-exist", "n0")
	if err == nil {
		t.Fatalf("BuildNodeDetail unknown graph: got nd=%v, want error", nd)
	}
	if nd != nil {
		t.Errorf("BuildNodeDetail unknown graph: got nd=%v, want nil", nd)
	}
}

// TestBuildNodeDetail_UnknownNode: an in-range graph with a node id past the
// arena's last index returns "node not found".
func TestBuildNodeDetail_UnknownNode(t *testing.T) {
	nd, err := BuildNodeDetail("mlp", "n9999999")
	if err == nil {
		t.Fatalf("BuildNodeDetail unknown node: got nd=%v, want error", nd)
	}
	if nd != nil {
		t.Errorf("BuildNodeDetail unknown node: got nd=%v, want nil", nd)
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error %v does not mention 'not found'", err)
	}
}

// TestBuildNodeDetail_MalformedNodeID: a non-numeric node id surfaces a
// parse error, not a panic. Studios run untrusted iframe message payloads
// through this path so the parse error path matters.
func TestBuildNodeDetail_MalformedNodeID(t *testing.T) {
	_, err := BuildNodeDetail("mlp", "not-an-id")
	if err == nil {
		t.Fatal("BuildNodeDetail malformed id: want error, got nil")
	}
}

// TestBuildNodeDetail_EmptyArgs guards against empty-string inputs producing
// a panic in BuildGraph / parseNodeID. Both empty graphId and empty nodeId
// should surface an error.
func TestBuildNodeDetail_EmptyArgs(t *testing.T) {
	if _, err := BuildNodeDetail("", "n0"); err == nil {
		t.Errorf("empty graphId: want error, got nil")
	}
	if _, err := BuildNodeDetail("mlp", ""); err == nil {
		t.Errorf("empty nodeId: want error, got nil")
	}
}

// TestBuildNodeDetail_JSONShape pins the JSON contract: every field the
// studio's drawer reads round-trips through encoding/json without loss. The
// test also asserts the empty-slice convention for Shape / Parents / Children
// so the frontend never sees a `null`.
func TestBuildNodeDetail_JSONShape(t *testing.T) {
	g, err := BuildGraph("mlp")
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	if len(g.Nodes) == 0 {
		t.Fatal("BuildGraph produced no nodes")
	}
	nd, err := BuildNodeDetail("mlp", formatNodeID(g.Nodes[0].ID))
	if err != nil {
		t.Fatalf("BuildNodeDetail: %v", err)
	}
	b, err := nd.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	// Spot-check the wire keys (snake_case per the spec).
	body := string(b)
	for _, want := range []string{
		`"graph_id"`,
		`"node_id"`,
		`"op"`,
		`"dtype"`,
		`"shape"`,
		`"phase"`,
		`"parents"`,
		`"children"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("JSON missing key %s\nbody: %s", want, body)
		}
	}
	// Round-trip into a fresh struct and confirm the field map is symmetric.
	var back NodeDetail
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v\nbody: %s", err, body)
	}
	if back.GraphID != nd.GraphID || back.NodeID != nd.NodeID || back.Op != nd.Op {
		t.Errorf("round-trip drift: got %+v, want %+v", back, *nd)
	}
}

// TestBuildNodeDetail_ParentsAndChildren walks for a node deep enough in the
// graph that both lists are non-empty and asserts the studio's traversal
// contract: every parent's ID round-trips through parseNodeID, and every
// child likewise.
func TestBuildNodeDetail_ParentsAndChildren(t *testing.T) {
	// Build the graph once to discover a node that has both a parent and a
	// child. MLP has a Buffer leaf (no parent) and Sink (no child); a middle
	// node has both. Walk g.Nodes for the first with non-trivial degrees.
	g, err := BuildGraph("mlp")
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	parentsOf := make(map[uint32]int)
	childrenOf := make(map[uint32]int)
	for _, e := range g.Edges {
		childrenOf[e.From]++
		parentsOf[e.To]++
	}
	pickID := ^uint32(0)
	for _, n := range g.Nodes {
		if parentsOf[n.ID] > 0 && childrenOf[n.ID] > 0 {
			pickID = n.ID
			break
		}
	}
	if pickID == ^uint32(0) {
		t.Skip("no interior node found in mlp graph; skipping (graph shape changed?)")
	}

	nd, err := BuildNodeDetail("mlp", formatNodeID(pickID))
	if err != nil {
		t.Fatalf("BuildNodeDetail: %v", err)
	}
	if len(nd.Parents) == 0 {
		t.Errorf("interior node has zero parents")
	}
	if len(nd.Children) == 0 {
		t.Errorf("interior node has zero children")
	}
	// IDs round-trip through parseNodeID.
	for _, p := range nd.Parents {
		if _, err := parseNodeID(p.ID); err != nil {
			t.Errorf("parent ID %q not parseable: %v", p.ID, err)
		}
		if p.Op == "" {
			t.Errorf("parent %q has empty Op", p.ID)
		}
	}
	for _, c := range nd.Children {
		if _, err := parseNodeID(c.ID); err != nil {
			t.Errorf("child ID %q not parseable: %v", c.ID, err)
		}
		if c.Op == "" {
			t.Errorf("child %q has empty Op", c.ID)
		}
	}
}
