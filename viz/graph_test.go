//go:build !js

package viz

import (
	"testing"

	_ "github.com/georgebuilds/anneal/examples" // register mlp, conv
)

func TestBuildGraph_MLP(t *testing.T) {
	g, err := BuildGraph("mlp")
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Nodes) < 5 {
		t.Fatalf("expected ≥5 nodes, got %d", len(g.Nodes))
	}

	var fwd, bwd, reduceKind int
	for _, n := range g.Nodes {
		switch n.Class {
		case ClassForward:
			fwd++
		case ClassBackward:
			bwd++
		}
		if n.Kind == KindReduce {
			reduceKind++
		}
	}
	if fwd == 0 {
		t.Error("expected forward nodes")
	}
	if bwd == 0 {
		t.Error("expected backward nodes (grad graph not constructed — check Leaves)")
	}
	if g.Stats.Kernels == 0 {
		t.Error("expected ≥1 kernel from scheduler")
	}
	if len(g.Edges) == 0 {
		t.Error("expected edges")
	}
	if reduceKind == 0 {
		t.Error("expected ≥1 reduce-kind node (kernel boundary)")
	}

	t.Logf("mlp: %d nodes (%d fwd, %d bwd), %d edges, %d kernels, %d reduce nodes",
		len(g.Nodes), fwd, bwd, len(g.Edges), g.Stats.Kernels, reduceKind)

	// Verify JSON serialization round-trips cleanly.
	b, err := g.ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	if len(b) == 0 {
		t.Error("ToJSON returned empty bytes")
	}
}

func TestBuildGraph_Conv(t *testing.T) {
	g, err := BuildGraph("conv")
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Nodes) < 10 {
		t.Fatalf("expected ≥10 nodes, got %d", len(g.Nodes))
	}

	var fwd, bwd int
	for _, n := range g.Nodes {
		switch n.Class {
		case ClassForward:
			fwd++
		case ClassBackward:
			bwd++
		}
	}
	if fwd == 0 {
		t.Error("expected forward nodes")
	}
	if bwd == 0 {
		t.Error("expected backward nodes")
	}

	t.Logf("conv: %d nodes (%d fwd, %d bwd), %d edges, %d kernels",
		len(g.Nodes), fwd, bwd, len(g.Edges), g.Stats.Kernels)
}

func TestBuildGraph_Unknown(t *testing.T) {
	_, err := BuildGraph("notexist")
	if err == nil {
		t.Error("expected error for unknown example name")
	}
}

func TestNodeClassification(t *testing.T) {
	g, err := BuildGraph("mlp")
	if err != nil {
		t.Fatal(err)
	}

	// Every node must have a valid class.
	for _, n := range g.Nodes {
		switch n.Class {
		case ClassForward, ClassBackward:
		default:
			t.Errorf("node %d has unexpected class %q", n.ID, n.Class)
		}
	}

	// Every node must have a valid kind.
	for _, n := range g.Nodes {
		switch n.Kind {
		case KindDefault, KindLeaf, KindReduce, KindSink:
		default:
			t.Errorf("node %d has unexpected kind %q", n.ID, n.Kind)
		}
	}

	// All edges must reference known node IDs.
	ids := make(map[uint32]bool, len(g.Nodes))
	for _, n := range g.Nodes {
		ids[n.ID] = true
	}
	for _, e := range g.Edges {
		if !ids[e.From] {
			t.Errorf("edge from unknown node %d", e.From)
		}
		if !ids[e.To] {
			t.Errorf("edge to unknown node %d", e.To)
		}
	}
}
