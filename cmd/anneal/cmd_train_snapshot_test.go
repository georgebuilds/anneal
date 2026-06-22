package main

import (
	"bytes"
	"regexp"
	"strings"
	"testing"

	// Register examples via init().
	_ "github.com/georgebuilds/anneal/examples"
)

// TestPlainTrainLineFormat verifies the plain-path output format pinned by
// spec §11.5. The line shape is "step %d: loss=%.6f\n" and must not drift
// under the W5 snapshot-channel refactor. This is the bundle-format and the
// regex-greppable contract CI scripts depend on.
func TestPlainTrainLineFormat(t *testing.T) {
	requireGPU(t)

	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--steps=3", "--log-every=1", "mlp"}, &buf)
	if code != 0 {
		t.Fatalf("trainCmdW exited %d, want 0; output:\n%s", code, buf.String())
	}
	out := buf.String()

	// The output must contain a "step N: loss=F\n" line for each logged step
	// (including step 0 from the initial-loss probe).
	stepLine := regexp.MustCompile(`(?m)^step \d+: loss=\d+\.\d{6}$`)
	matches := stepLine.FindAllString(out, -1)
	// MLP logs step 0 plus steps 1..3 = 4 lines.
	if len(matches) < 4 {
		t.Errorf("expected at least 4 loss lines matching %q, got %d:\n%s",
			stepLine.String(), len(matches), out)
	}

	// The header and footer lines must be present unchanged.
	for _, want := range []string{
		"training mlp",
		"device: ",
		"steps: 3 ",
		"done - 3 steps",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("plain output missing %q\n%s", want, out)
		}
	}
}
