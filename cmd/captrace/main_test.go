package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

// TestMainWrapper exercises the thin main() wrapper in-process by stubbing the
// os.Exit and timelineJSON seams. No GPU and no real compilation are needed.
func TestMainWrapper(t *testing.T) {
	origExit := osExit
	origArgs := os.Args
	origSeam := timelineJSON
	t.Cleanup(func() {
		osExit = origExit
		os.Args = origArgs
		timelineJSON = origSeam
	})

	var gotCode int
	exited := false
	osExit = func(code int) {
		gotCode = code
		exited = true
	}
	os.Args = []string{"captrace", "stub-example"}
	timelineJSON = func(name string) ([]byte, error) {
		if name != "stub-example" {
			t.Errorf("main() passed name %q, want stub-example", name)
		}
		return []byte(`{"stages":[]}`), nil
	}

	main()

	if !exited {
		t.Fatal("main() did not call osExit")
	}
	if gotCode != 0 {
		t.Errorf("main() exit code = %d, want 0", gotCode)
	}
}

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

// TestRunSerializationError forces the timelineJSON seam to fail, exercising
// the error branch of run without needing a GPU or a real compilation.
func TestRunSerializationError(t *testing.T) {
	orig := timelineJSON
	t.Cleanup(func() { timelineJSON = orig })

	timelineJSON = func(name string) ([]byte, error) {
		return nil, errors.New("boom: json marshal failed")
	}

	var stdout, stderr bytes.Buffer
	code := run([]string{"mlp"}, &stdout, &stderr)
	if code != 1 {
		t.Errorf("run() = %d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout should be empty on error, got %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "boom: json marshal failed") {
		t.Errorf("stderr missing forced error: %q", stderr.String())
	}
	if !strings.Contains(stderr.String(), "captrace:") {
		t.Errorf("stderr missing captrace prefix: %q", stderr.String())
	}
}

// TestRunSeamSuccess exercises the success path through the seam with a stub,
// keeping the test fast and GPU-free.
func TestRunSeamSuccess(t *testing.T) {
	orig := timelineJSON
	t.Cleanup(func() { timelineJSON = orig })

	timelineJSON = func(name string) ([]byte, error) {
		if name != "stub-example" {
			t.Errorf("seam got name %q, want stub-example", name)
		}
		return []byte(`{"ok":true}`), nil
	}

	var stdout, stderr bytes.Buffer
	if code := run([]string{"stub-example"}, &stdout, &stderr); code != 0 {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	if got := stdout.String(); got != `{"ok":true}` {
		t.Errorf("stdout = %q, want stub payload", got)
	}
}
