// Additional white-box coverage for cmd/anneal CPU-only paths: command
// dispatch (run), per-verb help text, usage/error branches that return before
// any GPU is opened, and the small display helpers (dtTypeName, bufferShape,
// kernelType, countFusedOps, detectBackend). None of these tests require a
// WebGPU adapter - every assertion is on textual output or pure-function
// return values reachable on CPU.

package main

import (
	"bytes"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/examples" // register mlp, conv, dynmlp via init
	"github.com/georgebuilds/anneal/uop"
)

// ── run() dispatcher (main.go) ────────────────────────────────────────────────

// TestRunDispatch_NoArgs prints top-level usage and returns 0.
func TestRunDispatch_NoArgs(t *testing.T) {
	if rc := run(nil); rc != 0 {
		t.Errorf("run(nil) = %d, want 0", rc)
	}
}

// TestRunDispatch_Version covers both spellings of the version verb.
func TestRunDispatch_Version(t *testing.T) {
	for _, v := range []string{"version", "--version"} {
		if rc := run([]string{v}); rc != 0 {
			t.Errorf("run([%q]) = %d, want 0", v, rc)
		}
	}
}

// TestRunDispatch_HelpTop covers the bare help / -h / --help paths (no verb).
func TestRunDispatch_HelpTop(t *testing.T) {
	for _, v := range []string{"help", "-h", "--help"} {
		if rc := run([]string{v}); rc != 0 {
			t.Errorf("run([%q]) = %d, want 0", v, rc)
		}
	}
}

// TestRunDispatch_HelpVerb routes `help <verb>` through verbHelp for every
// known verb (all return 0) and an unknown verb (returns 1).
func TestRunDispatch_HelpVerb(t *testing.T) {
	known := []string{"run", "train", "gpt2", "graph", "kernels", "explain", "web", "viz", "doctor"}
	for _, v := range known {
		if rc := run([]string{"help", v}); rc != 0 {
			t.Errorf("run([help %q]) = %d, want 0", v, rc)
		}
	}
	if rc := run([]string{"help", "not-a-real-verb"}); rc != 1 {
		t.Errorf("run([help not-a-real-verb]) = %d, want 1", rc)
	}
}

// TestRunDispatch_Unknown covers the default branch (unknown command).
func TestRunDispatch_Unknown(t *testing.T) {
	if rc := run([]string{"frobnicate"}); rc != 1 {
		t.Errorf("run([frobnicate]) = %d, want 1", rc)
	}
}

// TestRunDispatch_CPUSubcommands routes the CPU-only verbs through run() with
// a bad model argument so each handler exits non-zero before opening a GPU.
// This covers the run/graph/kernels/explain dispatch arms in run().
func TestRunDispatch_CPUSubcommands(t *testing.T) {
	for _, verb := range []string{"graph", "kernels", "explain", "run"} {
		if rc := run([]string{verb, "definitely-not-a-model-xyz"}); rc != 1 {
			t.Errorf("run([%q bad-model]) = %d, want 1", verb, rc)
		}
	}
}

// ── verbHelp() default arm ─────────────────────────────────────────────────────

// TestVerbHelpUnknown exercises verbHelp's default arm directly.
func TestVerbHelpUnknown(t *testing.T) {
	if rc := verbHelp("nope-xyz"); rc != 1 {
		t.Errorf("verbHelp(nope-xyz) = %d, want 1", rc)
	}
}

// ── usage / flag-error branches (no GPU) ───────────────────────────────────────

// TestUsageBranches_NoArgs verifies the four "no positional arg" usage paths
// (run/graph/kernels/explain) print a usage line and return 1 without a GPU.
func TestUsageBranches_NoArgs(t *testing.T) {
	cases := []struct {
		name   string
		fn     func([]string, *bytes.Buffer) int
		expect string
	}{
		{"run", func(a []string, b *bytes.Buffer) int { return runCmdW(a, b) }, "usage: anneal run"},
		{"graph", func(a []string, b *bytes.Buffer) int { return graphCmdW(a, b) }, "usage: anneal graph"},
		{"kernels", func(a []string, b *bytes.Buffer) int { return kernelsCmdW(a, b) }, "usage: anneal kernels"},
		{"explain", func(a []string, b *bytes.Buffer) int { return explainCmdW(a, b) }, "usage: anneal explain"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var buf bytes.Buffer
			if rc := c.fn(nil, &buf); rc != 1 {
				t.Fatalf("%s no-args rc = %d, want 1; out: %s", c.name, rc, buf.String())
			}
			if !strings.Contains(buf.String(), c.expect) {
				t.Errorf("%s no-args output missing %q; got: %s", c.name, c.expect, buf.String())
			}
		})
	}
}

// TestFlagErrorBranches verifies the parseFlags error branch in each CPU
// handler: an unknown flag must make the handler print the error and return 1
// before reaching any GPU code.
func TestFlagErrorBranches(t *testing.T) {
	cases := map[string]func([]string, *bytes.Buffer) int{
		"run":     func(a []string, b *bytes.Buffer) int { return runCmdW(a, b) },
		"graph":   func(a []string, b *bytes.Buffer) int { return graphCmdW(a, b) },
		"kernels": func(a []string, b *bytes.Buffer) int { return kernelsCmdW(a, b) },
		"explain": func(a []string, b *bytes.Buffer) int { return explainCmdW(a, b) },
	}
	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			if rc := fn([]string{"--no-such-flag"}, &buf); rc != 1 {
				t.Errorf("%s --no-such-flag rc = %d, want 1; out: %s", name, rc, buf.String())
			}
		})
	}
}

// TestBadModelBranches verifies the examples.Get error branch (unknown model)
// in run/graph/kernels - these return before any GPU is opened.
func TestBadModelBranches(t *testing.T) {
	cases := map[string]func([]string, *bytes.Buffer) int{
		"run":     func(a []string, b *bytes.Buffer) int { return runCmdW(a, b) },
		"graph":   func(a []string, b *bytes.Buffer) int { return graphCmdW(a, b) },
		"kernels": func(a []string, b *bytes.Buffer) int { return kernelsCmdW(a, b) },
	}
	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			if rc := fn([]string{"no-such-model-xyz"}, &buf); rc != 1 {
				t.Errorf("%s bad-model rc = %d, want 1; out: %s", name, rc, buf.String())
			}
		})
	}
}

// ── display helpers ────────────────────────────────────────────────────────────

// TestDtTypeName exercises every mapped dtype branch plus nil and a fallback.
func TestDtTypeName(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want string
	}{
		{uop.Dtypes.Float32, "f32"},
		{uop.Dtypes.Float16, "f16"},
		{uop.Dtypes.Float64, "f64"},
		{uop.Dtypes.Int32, "i32"},
		{uop.Dtypes.UInt32, "u32"},
		{uop.Dtypes.Int64, "i64"},
		{uop.Dtypes.UInt64, "u64"},
		{uop.Dtypes.Int8, "i8"},
		{uop.Dtypes.UInt8, "u8"},
		{uop.Dtypes.Bool, "bool"},
		{uop.Dtypes.Index, "index"},
		{uop.Dtypes.Void, "void"},
	}
	for _, c := range cases {
		if got := dtTypeName(c.dt); got != c.want {
			t.Errorf("dtTypeName(%v) = %q, want %q", c.dt, got, c.want)
		}
	}
	if got := dtTypeName(nil); got != "?" {
		t.Errorf("dtTypeName(nil) = %q, want ?", got)
	}
}

// TestBufferShape covers the []int64, int64, and default (nil) arms.
func TestBufferShape(t *testing.T) {
	a := uop.NewArena(16)
	multi := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 3}, nil)
	if got := bufferShape(multi); len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("bufferShape([]int64{2,3}) = %v, want [2 3]", got)
	}

	scalar := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, int64(7), nil)
	if got := bufferShape(scalar); len(got) != 1 || got[0] != 7 {
		t.Errorf("bufferShape(int64(7)) = %v, want [7]", got)
	}

	none := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, "unexpected-arg-type", nil)
	if got := bufferShape(none); got != nil {
		t.Errorf("bufferShape(string arg) = %v, want nil", got)
	}
}

// TestKernelType covers both reduction and elementwise classification.
func TestKernelType(t *testing.T) {
	if got := kernelType("let x = 1.0;\nfor (var i = 0u; i < 4u; i++) {}"); got != "reduction" {
		t.Errorf("kernelType(loop) = %q, want reduction", got)
	}
	if got := kernelType("let x = a + b;"); got != "elementwise" {
		t.Errorf("kernelType(no loop) = %q, want elementwise", got)
	}
}

// TestCountFusedOps counts let-t bindings.
func TestCountFusedOps(t *testing.T) {
	if got := countFusedOps("let t0 = a;\nlet t1 = b;\nlet z = c;"); got != 2 {
		t.Errorf("countFusedOps = %d, want 2", got)
	}
	if got := countFusedOps("no bindings here"); got != 0 {
		t.Errorf("countFusedOps(none) = %d, want 0", got)
	}
}

// TestDetectBackend asserts the host-OS arm returns one of the known labels.
func TestDetectBackend(t *testing.T) {
	got := detectBackend()
	switch runtime.GOOS {
	case "darwin":
		if got != "Metal" {
			t.Errorf("detectBackend on darwin = %q, want Metal", got)
		}
	case "linux":
		if got != "Vulkan" {
			t.Errorf("detectBackend on linux = %q, want Vulkan", got)
		}
	case "windows":
		if got != "D3D12" {
			t.Errorf("detectBackend on windows = %q, want D3D12", got)
		}
	default:
		if got != "unknown" {
			t.Errorf("detectBackend = %q, want unknown", got)
		}
	}
}

// TestJSONString covers the SSE error-payload JSON quoter.
func TestJSONString(t *testing.T) {
	got := jsonString("a\"b\nc")
	// json.Marshal escapes the quote and newline and wraps in double quotes.
	if !strings.HasPrefix(got, "\"") || !strings.HasSuffix(got, "\"") {
		t.Errorf("jsonString result not quoted: %q", got)
	}
	if !strings.Contains(got, "\\\"") || !strings.Contains(got, "\\n") {
		t.Errorf("jsonString did not escape quote/newline: %q", got)
	}
}

// ── train helpers (CPU-only) ───────────────────────────────────────────────────

// TestIsTerminalWriter covers the non-*os.File and *os.File (non-TTY) branches.
// A bytes.Buffer is not a file; an *os.File backed by a regular temp file is
// not a character device. The interactive-TTY branch is not exercisable in a
// test harness and is intentionally left out.
func TestIsTerminalWriter(t *testing.T) {
	if isTerminalWriter(&bytes.Buffer{}) {
		t.Error("isTerminalWriter(buffer) = true, want false")
	}
	f, err := os.CreateTemp(t.TempDir(), "tw")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if isTerminalWriter(f) {
		t.Error("isTerminalWriter(regular file) = true, want false")
	}
}

// TestMaybeOpenBundle_Disabled returns nil when bundling is off.
func TestMaybeOpenBundle_Disabled(t *testing.T) {
	var buf bytes.Buffer
	bw := maybeOpenBundle(&buf, false, "mlp", "adapter", "Metal", "webgpu", examples.TrainConfig{})
	if bw != nil {
		t.Errorf("maybeOpenBundle(enable=false) = %v, want nil", bw)
	}
}

// TestMaybeOpenBundle_BadRoot covers the EnvOrDefault error branch: a relative
// ANNEAL_RUN_DIR is rejected, the handler prints a skip line and returns nil.
func TestMaybeOpenBundle_BadRoot(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", "relative/not/absolute")
	var buf bytes.Buffer
	bw := maybeOpenBundle(&buf, true, "mlp", "adapter", "Metal", "webgpu", examples.TrainConfig{})
	if bw != nil {
		t.Errorf("maybeOpenBundle(bad root) = %v, want nil", bw)
	}
	if !strings.Contains(buf.String(), "bundle: skip") {
		t.Errorf("expected 'bundle: skip' line; got: %s", buf.String())
	}
}

// TestMaybeOpenBundle_Success drives the full success path: an absolute temp
// ANNEAL_RUN_DIR lets NewWriter/SetProvenance/WriteConfig run, returning a
// non-nil writer (covering the provenance + config arms on CPU).
func TestMaybeOpenBundle_Success(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	var buf bytes.Buffer
	cfg := examples.TrainConfig{Steps: 3, LR: 0.05, LogEvery: 1, Batch: 4}
	bw := maybeOpenBundle(&buf, true, "mlp", "adapter", "Metal", "webgpu", cfg)
	if bw == nil {
		t.Fatalf("maybeOpenBundle(good root) = nil, want writer; out: %s", buf.String())
	}
}

// ── canonicalOpName extra coverage ─────────────────────────────────────────────

// TestCanonicalOpName hits a mapped entry and the passthrough fallback.
func TestCanonicalOpName(t *testing.T) {
	if got := canonicalOpName("min"); got != "Min" {
		t.Errorf("canonicalOpName(min) = %q, want Min", got)
	}
	if got := canonicalOpName("erf"); got != "Erf" {
		t.Errorf("canonicalOpName(erf) = %q, want Erf", got)
	}
	if got := canonicalOpName("unmapped-op"); got != "unmapped-op" {
		t.Errorf("canonicalOpName(unmapped) = %q, want passthrough", got)
	}
}
