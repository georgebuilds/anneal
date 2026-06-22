//go:build !js

package viz

import (
	"testing"

	_ "github.com/georgebuilds/anneal/examples" // register mlp, conv
)

func TestBuildTimeline_MLP(t *testing.T) {
	tl, err := BuildTimeline("mlp")
	if err != nil {
		t.Fatal(err)
	}
	if len(tl.Stages) != 3 {
		t.Fatalf("want 3 stages, got %d", len(tl.Stages))
	}
	wantIDs := []string{"forward", "gradient", "scheduled"}
	for i, st := range tl.Stages {
		if st.ID != wantIDs[i] {
			t.Errorf("stage %d id = %q, want %q", i, st.ID, wantIDs[i])
		}
		if len(st.Overrides) == 0 {
			t.Errorf("stage %q has no overrides", st.ID)
		}
	}

	// Stage 0 (forward) must have no backward overrides.
	for id, ov := range tl.Stages[0].Overrides {
		if ov.Class == ClassBackward {
			t.Errorf("forward stage classified node %d as backward", id)
		}
	}
	// Stage 1 (gradient) must surface ≥1 backward override.
	bwd := 0
	for _, ov := range tl.Stages[1].Overrides {
		if ov.Class == ClassBackward {
			bwd++
		}
	}
	if bwd == 0 {
		t.Error("gradient stage has zero backward overrides")
	}

	// Stage 0 (forward) is a strict subset of stage 1 (gradient).
	if len(tl.Stages[0].Overrides) >= len(tl.Stages[1].Overrides) {
		t.Errorf("forward stage (%d nodes) should be smaller than gradient stage (%d nodes)",
			len(tl.Stages[0].Overrides), len(tl.Stages[1].Overrides))
	}

	// Stage 2 (scheduled) must report ≥1 kernel.
	if tl.Stages[2].Stats.Kernels == 0 {
		t.Error("scheduled stage reports 0 kernels")
	}
	// Stage 2 must promote ≥1 reduce node back to KindReduce (kernel boundary).
	reduces := 0
	for _, ov := range tl.Stages[2].Overrides {
		if ov.Kind == KindReduce {
			reduces++
		}
	}
	if reduces == 0 {
		t.Error("scheduled stage has no KindReduce nodes - expected ≥1 kernel boundary")
	}

	// Union nodes/edges must be non-empty and self-consistent.
	if len(tl.Nodes) == 0 {
		t.Error("timeline has no union nodes")
	}
	ids := make(map[uint32]bool, len(tl.Nodes))
	for _, n := range tl.Nodes {
		ids[n.ID] = true
	}
	for _, e := range tl.Edges {
		if !ids[e.From] || !ids[e.To] {
			t.Errorf("edge %d→%d references missing union node", e.From, e.To)
		}
	}

	t.Logf("mlp timeline: %d union nodes, %d edges, stages: %d→%d→%d(k=%d, fused=%d)",
		len(tl.Nodes), len(tl.Edges),
		len(tl.Stages[0].Overrides),
		len(tl.Stages[1].Overrides),
		len(tl.Stages[2].Overrides), tl.Stages[2].Stats.Kernels, tl.Stages[2].Stats.Fused,
	)
}

func TestBuildTimeline_Conv(t *testing.T) {
	tl, err := BuildTimeline("conv")
	if err != nil {
		t.Fatal(err)
	}
	if len(tl.Stages) != 3 {
		t.Fatalf("want 3 stages, got %d", len(tl.Stages))
	}
	if tl.Stages[2].Stats.Kernels == 0 {
		t.Error("conv scheduled stage reports 0 kernels")
	}
	t.Logf("conv timeline: %d union nodes, %d edges, stages: %d→%d→%d(k=%d, fused=%d)",
		len(tl.Nodes), len(tl.Edges),
		len(tl.Stages[0].Overrides),
		len(tl.Stages[1].Overrides),
		len(tl.Stages[2].Overrides), tl.Stages[2].Stats.Kernels, tl.Stages[2].Stats.Fused,
	)
}

func TestBuildTimeline_UnknownExample(t *testing.T) {
	_, err := BuildTimeline("notexist")
	if err == nil {
		t.Error("expected error for unknown example")
	}
}
