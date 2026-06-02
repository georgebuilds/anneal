package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestGPT2CmdNoSubcommand verifies the gpt2 verb dispatcher prints usage
// and returns a non-zero exit code when called with no subcommand.
func TestGPT2CmdNoSubcommand(t *testing.T) {
	var buf bytes.Buffer
	rc := gpt2CmdW(nil, &buf)
	if rc == 0 {
		t.Errorf("gpt2 with no args should exit non-zero, got %d", rc)
	}
	if !strings.Contains(buf.String(), "usage: anneal gpt2") {
		t.Errorf("gpt2 with no args should print usage, got:\n%s", buf.String())
	}
}

// TestGPT2CmdUnknownSubcommand verifies an unknown subcommand prints a
// "unknown subcommand" error and the usage text.
func TestGPT2CmdUnknownSubcommand(t *testing.T) {
	var buf bytes.Buffer
	rc := gpt2CmdW([]string{"frobnicate"}, &buf)
	if rc == 0 {
		t.Errorf("gpt2 frobnicate should exit non-zero, got %d", rc)
	}
	out := buf.String()
	if !strings.Contains(out, "unknown subcommand") {
		t.Errorf("expected 'unknown subcommand' in output, got:\n%s", out)
	}
	if !strings.Contains(out, "usage: anneal gpt2") {
		t.Errorf("expected usage text after unknown subcommand, got:\n%s", out)
	}
}

// TestGPT2CmdHelpExitsZero exercises the help alias paths (-h, --help, help)
// and ensures they print usage with a zero exit code.
func TestGPT2CmdHelpExitsZero(t *testing.T) {
	for _, arg := range []string{"-h", "--help", "help"} {
		var buf bytes.Buffer
		rc := gpt2CmdW([]string{arg}, &buf)
		if rc != 0 {
			t.Errorf("gpt2 %s: rc=%d, want 0", arg, rc)
		}
		if !strings.Contains(buf.String(), "usage: anneal gpt2") {
			t.Errorf("gpt2 %s: missing usage text, got:\n%s", arg, buf.String())
		}
	}
}

// TestGPT2SampleCmdNoPrompt verifies that `anneal gpt2 sample` (no prompt)
// prints the sample-usage line and returns non-zero. This is the offline
// smoke gate: it must not hit the network and must not require a GPU.
func TestGPT2SampleCmdNoPrompt(t *testing.T) {
	t.Setenv("ANNEAL_OFFLINE", "1")
	var buf bytes.Buffer
	rc := gpt2SampleCmdW(nil, &buf)
	if rc == 0 {
		t.Errorf("gpt2 sample with no prompt should exit non-zero, got %d", rc)
	}
	if !strings.Contains(buf.String(), "usage: anneal gpt2 sample") {
		t.Errorf("expected sample usage line, got:\n%s", buf.String())
	}
}

// TestIsOfflineMissingAssetMatches sanity-checks the helper that turns the
// asset-cache error into a "fetch manually" hint trigger.
func TestIsOfflineMissingAssetMatches(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"ANNEAL_OFFLINE=1: asset not in cache at /foo; fetch manually from http://...", true},
		{"some other error", false},
		{"", false},
	}
	for _, c := range cases {
		var err error
		if c.in != "" {
			err = errString(c.in)
		}
		got := isOfflineMissingAsset(err)
		if got != c.want {
			t.Errorf("isOfflineMissingAsset(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// errString is a tiny error type for table-driven tests without pulling in
// fmt.Errorf at the call site.
type errString string

func (e errString) Error() string { return string(e) }
