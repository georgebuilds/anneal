package tui

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestTokenSnapshotJSONRoundtrip pins the JSON tags by encoding a populated
// value and decoding it back. Drift in a tag would break the SSE wire
// contract used by /sse/generate; the studio reads these keys by string.
func TestTokenSnapshotJSONRoundtrip(t *testing.T) {
	yes := true
	in := TokenSnapshot{
		Step:                 3,
		MaxTokens:            32,
		TokenID:              17,
		TokenText:            " the",
		LogitArgmax:          17,
		LogitSummary:         "max=4.21 at idx 17",
		WallMs:               142,
		Phase:                PhaseTraining,
		RefMatch:             &yes,
		DispatchCount:        88,
		LastKernelID:         "k_attn_qkv",
		LastDispatchWasFused: true,
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// Pin every wire key the studio consumes.
	for _, key := range []string{
		`"step":3`,
		`"max_tokens":32`,
		`"token_id":17`,
		`"token_text":" the"`,
		`"logit_argmax":17`,
		`"logit_summary":"max=4.21 at idx 17"`,
		`"wall_ms":142`,
		`"phase":"training"`,
		`"ref_match":true`,
		`"dispatch_count":88`,
		`"last_kernel_id":"k_attn_qkv"`,
		`"last_dispatch_was_fused":true`,
	} {
		if !strings.Contains(string(b), key) {
			t.Errorf("marshal missing %s; got %s", key, string(b))
		}
	}

	var out TokenSnapshot
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.Step != in.Step || out.TokenID != in.TokenID || out.TokenText != in.TokenText {
		t.Errorf("roundtrip mismatch: %+v vs %+v", out, in)
	}
	if out.RefMatch == nil || *out.RefMatch != true {
		t.Errorf("RefMatch lost in roundtrip: got %v", out.RefMatch)
	}
	if out.LastKernelID != in.LastKernelID {
		t.Errorf("LastKernelID roundtrip: got %q, want %q", out.LastKernelID, in.LastKernelID)
	}
}

// TestTokenSnapshotRefMatchPointerSemantics pins that a nil RefMatch is
// omitted from the JSON (omitempty) - the studio reads "field absent" as
// "no oracle was configured", distinct from "oracle disagreed".
func TestTokenSnapshotRefMatchPointerSemantics(t *testing.T) {
	in := TokenSnapshot{
		Step:      0,
		TokenID:   1,
		TokenText: "a",
		Phase:     PhaseTraining,
		// RefMatch left nil.
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "ref_match") {
		t.Errorf("nil RefMatch should be omitted from JSON; got %s", string(b))
	}
	// And explicitly false should be present.
	no := false
	in.RefMatch = &no
	b2, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal (false): %v", err)
	}
	if !strings.Contains(string(b2), `"ref_match":false`) {
		t.Errorf("explicit false RefMatch should be present; got %s", string(b2))
	}
}

// TestTokenSnapshotPhaseEnum pins that the Phase field uses the same wire
// vocabulary as Snapshot - the studio routes on these strings.
func TestTokenSnapshotPhaseEnum(t *testing.T) {
	for _, tc := range []struct {
		phase Phase
		want  string
	}{
		{PhaseInit, `"phase":"init"`},
		{PhaseTraining, `"phase":"training"`},
		{PhaseDone, `"phase":"done"`},
		{PhaseError, `"phase":"error"`},
	} {
		in := TokenSnapshot{Phase: tc.phase}
		b, err := json.Marshal(in)
		if err != nil {
			t.Fatalf("marshal %v: %v", tc.phase, err)
		}
		if !strings.Contains(string(b), tc.want) {
			t.Errorf("phase %v: marshal missing %s; got %s", tc.phase, tc.want, string(b))
		}
	}
}
