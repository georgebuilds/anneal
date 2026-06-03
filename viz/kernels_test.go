// Tests for viz.BuildKernels — the WASM-buildable path that drives the W2
// kernels view. These tests run on native (no //go:build !js tag) so the
// same exercise applies in the WASM environment via direct Go calls (the
// kernel set is structurally the same in both targets — DD2).

package viz

import (
	"encoding/json"
	"strings"
	"testing"

	_ "github.com/georgebuilds/anneal/examples" // register mlp, conv, dynmlp
)

// TestBuildKernels_MLP pins the basic shape of the kernel set: model name
// echoed back, at least one kernel, every kernel has non-empty WGSL and a
// monotonically-numbered id.
func TestBuildKernels_MLP(t *testing.T) {
	k, err := BuildKernels("mlp")
	if err != nil {
		t.Fatal(err)
	}
	if k.Model != "mlp" {
		t.Errorf("Model = %q, want %q", k.Model, "mlp")
	}
	if len(k.Kernels) == 0 {
		t.Fatalf("expected >= 1 kernel for mlp, got 0")
	}
	for i, kr := range k.Kernels {
		if kr.WGSL == "" {
			t.Errorf("kernel %d has empty WGSL", i)
		}
		if kr.ID == "" {
			t.Errorf("kernel %d has empty id", i)
		}
		// op_count derived from the AST; for any non-trivial kernel it is
		// > 0 (every kernel performs at least one elementwise operation).
		if kr.OpCount == 0 {
			t.Errorf("kernel %d (%s) has op_count=0", i, kr.ID)
		}
		// Output buffer count is always exactly 1 (Bufs[0] in codegen).
		if kr.BuffersOut != 1 {
			t.Errorf("kernel %d (%s) buffers_out = %d, want 1", i, kr.ID, kr.BuffersOut)
		}
		// At least one fusion span (the prologue always counts).
		if len(kr.FusionSpans) == 0 {
			t.Errorf("kernel %d (%s) has no fusion spans", i, kr.ID)
		}
	}

	// Round-trip JSON marshal and assert documented top-level keys.
	b, err := k.ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"model"`) ||
		!strings.Contains(string(b), `"kernels"`) ||
		!strings.Contains(string(b), `"wgsl"`) ||
		!strings.Contains(string(b), `"fusion_spans"`) {
		t.Errorf("kernel JSON missing documented keys: %s", string(b)[:min(400, len(b))])
	}
}

// TestBuildKernels_Conv exercises a deeper model (more kernels + reduce-axis
// boundary). Pins that the conv graph also returns a sensible kernel set.
func TestBuildKernels_Conv(t *testing.T) {
	k, err := BuildKernels("conv")
	if err != nil {
		t.Fatal(err)
	}
	if len(k.Kernels) == 0 {
		t.Fatal("expected >= 1 kernel for conv")
	}
	for _, kr := range k.Kernels {
		if kr.WGSL == "" {
			t.Errorf("kernel %s has empty WGSL", kr.ID)
		}
	}
}

// TestBuildKernels_Unknown pins the error path.
func TestBuildKernels_Unknown(t *testing.T) {
	if _, err := BuildKernels("notexist"); err == nil {
		t.Error("expected error for unknown example")
	}
}

// TestBuildKernels_FusionSpansCoverWGSL pins that the spans cover every line
// of the rendered WGSL (no gaps, no overlap). This is the property the
// studio's gutter renderer relies on.
func TestBuildKernels_FusionSpansCoverWGSL(t *testing.T) {
	k, err := BuildKernels("mlp")
	if err != nil {
		t.Fatal(err)
	}
	for _, kr := range k.Kernels {
		lines := strings.Split(strings.TrimRight(kr.WGSL, "\n"), "\n")
		nLines := len(lines)
		if len(kr.FusionSpans) == 0 {
			t.Errorf("kernel %s has no spans", kr.ID)
			continue
		}
		if kr.FusionSpans[0].StartLine != 1 {
			t.Errorf("kernel %s first span starts at %d, want 1",
				kr.ID, kr.FusionSpans[0].StartLine)
		}
		last := kr.FusionSpans[len(kr.FusionSpans)-1]
		if last.EndLine != nLines {
			t.Errorf("kernel %s last span ends at %d, want %d",
				kr.ID, last.EndLine, nLines)
		}
		// Adjacent spans must connect with no gap and no overlap.
		for i := 1; i < len(kr.FusionSpans); i++ {
			prev := kr.FusionSpans[i-1]
			cur := kr.FusionSpans[i]
			if cur.StartLine != prev.EndLine+1 {
				t.Errorf("kernel %s span gap/overlap between %v and %v",
					kr.ID, prev, cur)
			}
			if cur.Label == prev.Label {
				t.Errorf("kernel %s adjacent spans share label %q (should be coalesced): %v / %v",
					kr.ID, cur.Label, prev, cur)
			}
		}
		// Every span label is one of the documented set.
		for _, sp := range kr.FusionSpans {
			switch sp.Label {
			case FusionLabelForward, FusionLabelBackward, FusionLabelFused:
			default:
				t.Errorf("kernel %s span has unexpected label %q", kr.ID, sp.Label)
			}
		}
	}
}

// TestBuildKernels_JSONShape pins the JSON field names so the JS renderer can
// rely on them without an out-of-band schema.
func TestBuildKernels_JSONShape(t *testing.T) {
	k, err := BuildKernels("mlp")
	if err != nil {
		t.Fatal(err)
	}
	b, err := k.ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(b, &parsed); err != nil {
		t.Fatalf("kernel set JSON does not parse: %v", err)
	}
	if _, ok := parsed["model"]; !ok {
		t.Error("missing top-level 'model'")
	}
	kernels, ok := parsed["kernels"].([]any)
	if !ok || len(kernels) == 0 {
		t.Fatalf("missing or empty 'kernels' array")
	}
	first, _ := kernels[0].(map[string]any)
	for _, key := range []string{
		"id", "op_count", "buffers_in", "buffers_out", "shape", "wgsl", "fusion_spans",
	} {
		if _, ok := first[key]; !ok {
			t.Errorf("first kernel missing key %q", key)
		}
	}
	spans, ok := first["fusion_spans"].([]any)
	if !ok || len(spans) == 0 {
		t.Fatalf("missing or empty fusion_spans on first kernel")
	}
	span0, _ := spans[0].(map[string]any)
	for _, key := range []string{"start_line", "end_line", "label"} {
		if _, ok := span0[key]; !ok {
			t.Errorf("fusion span missing key %q", key)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
