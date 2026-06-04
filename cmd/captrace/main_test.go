package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunDefaultName(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run(nil, &stdout, &stderr); code != 0 {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr not empty: %q", stderr.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("stdout is not valid JSON: %v", err)
	}
	if _, ok := payload["stages"]; !ok {
		t.Errorf("payload missing stages key: %v", payload)
	}
}

func TestRunExplicitName(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"conv"}, &stdout, &stderr); code != 0 {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("stdout is not valid JSON: %v", err)
	}
}

func TestRunUnknownExample(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"nope-not-real"}, &stdout, &stderr)
	if code != 1 {
		t.Errorf("run() = %d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout should be empty on error, got %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "captrace:") {
		t.Errorf("stderr missing captrace prefix: %q", stderr.String())
	}
}
