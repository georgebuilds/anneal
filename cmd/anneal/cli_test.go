package main

import (
	"bytes"
	"runtime"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/tensor"

	// Register examples via init().
	_ "github.com/georgebuilds/anneal/examples"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// requireGPU skips the test if no GPU is available and sets up the WebGPU
// executor for the duration of the test.
func requireGPU(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("no GPU: %v", err)
	}
	t.Cleanup(func() {
		tensor.DefaultExecutor = nil
		dev.Close()
	})
	tensor.DefaultExecutor = dev
}

// ── sentence-case tests ───────────────────────────────────────────────────────

// TestHelpSentenceCase verifies that the top-level usage text uses sentence case
// and does not contain Title Case headings or ALL CAPS labels.
func TestHelpSentenceCase(t *testing.T) {
	// Check the canonical usage text directly.
	usage := usageText

	// Must NOT contain Title Case command headers.
	badPhrases := []string{
		"Commands:", "Flags:", "Usage:", "Run ", "Train ", "Graph ",
		"COMMANDS", "FLAGS", "USAGE",
	}
	for _, phrase := range badPhrases {
		if strings.Contains(usage, phrase) {
			t.Errorf("usage text contains title/uppercase phrase %q (want sentence case)", phrase)
		}
	}

	// Must contain sentence-case section headers.
	goodPhrases := []string{"commands:", "flags (global):"}
	for _, phrase := range goodPhrases {
		if !strings.Contains(usage, phrase) {
			t.Errorf("usage text missing expected sentence-case phrase %q", phrase)
		}
	}
}

// ── NO_COLOR tests ────────────────────────────────────────────────────────────

// TestNoColor verifies that bold() returns the unmodified string when NO_COLOR is set.
func TestNoColor(t *testing.T) {
	t.Setenv("NO_COLOR", "1")

	s := "hello"
	got := bold(s)
	if got != s {
		t.Errorf("bold(%q) with NO_COLOR=1 = %q, want %q (no ANSI escapes)", s, got, s)
	}
	if strings.Contains(got, "\033") {
		t.Errorf("bold(%q) with NO_COLOR=1 contains ANSI escape codes", s)
	}
}

// ── env alias tests ───────────────────────────────────────────────────────────

// TestEnvAliases verifies that VIZ=1 sets viz, DEBUG=2 sets debug=2,
// and that explicit flags override env aliases.
func TestEnvAliases(t *testing.T) {
	t.Run("VIZ=1 sets viz", func(t *testing.T) {
		t.Setenv("VIZ", "1")
		t.Setenv("DEBUG", "")
		flags, _, err := parseFlags("test", []string{})
		if err != nil {
			t.Fatal(err)
		}
		if !flags.viz {
			t.Error("VIZ=1 should set viz=true")
		}
	})

	t.Run("DEBUG=2 sets debug", func(t *testing.T) {
		t.Setenv("VIZ", "")
		t.Setenv("DEBUG", "2")
		flags, _, err := parseFlags("test", []string{})
		if err != nil {
			t.Fatal(err)
		}
		if flags.debug != 2 {
			t.Errorf("DEBUG=2 should set debug=2, got %d", flags.debug)
		}
	})

	t.Run("explicit --viz overrides VIZ env", func(t *testing.T) {
		t.Setenv("VIZ", "0")
		flags, _, err := parseFlags("test", []string{"--viz"})
		if err != nil {
			t.Fatal(err)
		}
		if !flags.viz {
			t.Error("explicit --viz should set viz=true regardless of VIZ env")
		}
	})

	t.Run("explicit --debug overrides DEBUG env", func(t *testing.T) {
		t.Setenv("DEBUG", "3")
		flags, _, err := parseFlags("test", []string{"--debug=1"})
		if err != nil {
			t.Fatal(err)
		}
		if flags.debug != 1 {
			t.Errorf("explicit --debug=1 should win over DEBUG=3, got %d", flags.debug)
		}
	})
}

// ── error message tests ───────────────────────────────────────────────────────

// TestNoAdapterErrorMessage verifies the canonical no-adapter error message.
func TestNoAdapterErrorMessage(t *testing.T) {
	msg := noAdapterError()

	required := []string{
		"no WebGPU adapter found",
		"Metal",
		"Vulkan",
		"D3D12",
		"anneal doctor",
	}
	for _, phrase := range required {
		if !strings.Contains(msg, phrase) {
			t.Errorf("noAdapterError() missing %q", phrase)
		}
	}
}

// TestDoctorErrorShape verifies the doctor-specific failure message.
func TestDoctorErrorShape(t *testing.T) {
	msg := doctorFailureMsg()

	required := []string{
		"no WebGPU adapter found",
		"Metal",
		"Vulkan",
		"D3D12",
		"macOS",
		"Linux",
		"Windows",
	}
	for _, phrase := range required {
		if !strings.Contains(msg, phrase) {
			t.Errorf("doctorFailureMsg() missing %q", phrase)
		}
	}

	// Must NOT end with "run anneal doctor" since we ARE doctor.
	if strings.Contains(msg, "run 'anneal doctor'") {
		t.Error("doctorFailureMsg() must not say 'run anneal doctor' (we are already in doctor)")
	}
}

// ── explain tests ─────────────────────────────────────────────────────────────

// TestExplainAdd verifies explain for the "add" op.
func TestExplainAdd(t *testing.T) {
	var buf bytes.Buffer
	code := explainCmdW([]string{"add"}, &buf)
	if code != 0 {
		t.Fatalf("explainCmdW(add) exited %d, want 0; output: %s", code, buf.String())
	}
	out := buf.String()

	required := []string{
		"x + 0",
		"∂(a+b)",
	}
	for _, phrase := range required {
		if !strings.Contains(out, phrase) {
			t.Errorf("explain add: output missing %q\nfull output:\n%s", phrase, out)
		}
	}
}

// TestExplainMatmul verifies explain for matmul.backward.
func TestExplainMatmul(t *testing.T) {
	var buf bytes.Buffer
	code := explainCmdW([]string{"matmul.backward"}, &buf)
	if code != 0 {
		t.Fatalf("explainCmdW(matmul.backward) exited %d, want 0", code)
	}
	out := buf.String()

	required := []string{
		"ReduceAxis",
		"decompos",
	}
	for _, phrase := range required {
		if !strings.Contains(out, phrase) {
			t.Errorf("explain matmul.backward: output missing %q\nfull output:\n%s", phrase, out)
		}
	}
}

// TestExplainSymbolic verifies explain for the "symbolic" keyword.
func TestExplainSymbolic(t *testing.T) {
	var buf bytes.Buffer
	code := explainCmdW([]string{"symbolic"}, &buf)
	if code != 0 {
		t.Fatalf("explainCmdW(symbolic) exited %d, want 0", code)
	}
	out := buf.String()

	if !strings.Contains(out, "12") {
		t.Errorf("explain symbolic: output should mention 12 rule groups\nfull output:\n%s", out)
	}
}

// TestExplainUnknownOp verifies that an unknown op returns exit code 1 and
// includes the queried op name in the error.
func TestExplainUnknownOp(t *testing.T) {
	var buf bytes.Buffer
	code := explainCmdW([]string{"fakeopxyz"}, &buf)
	if code == 0 {
		t.Fatal("explainCmdW(fakeopxyz) exited 0, want 1")
	}
	out := buf.String()
	if !strings.Contains(out, "fakeopxyz") {
		t.Errorf("explain unknown op: output should mention the queried op; got:\n%s", out)
	}
}

// ── graph tests ───────────────────────────────────────────────────────────────

// TestGraphMLP verifies the graph command for the MLP example.
func TestGraphMLP(t *testing.T) {
	var buf bytes.Buffer
	code := graphCmdW([]string{"mlp"}, &buf)
	if code != 0 {
		t.Fatalf("graphCmdW(mlp) exited %d, want 0; output: %s", code, buf.String())
	}
	out := buf.String()

	required := []string{
		"Buffer",
		"ReduceAxis",
		"UOp",
	}
	for _, phrase := range required {
		if !strings.Contains(out, phrase) {
			t.Errorf("graph mlp: output missing %q\nfull output:\n%s", phrase, out)
		}
	}
}

// TestGraphSentenceCase verifies the graph command header uses sentence case.
func TestGraphSentenceCase(t *testing.T) {
	var buf bytes.Buffer
	graphCmdW([]string{"mlp"}, &buf)
	out := buf.String()

	// The header must say "graph:" not "Graph:", "device:" not "Device:"
	badPhrases := []string{"Graph:", "Device:", "Nodes:"}
	for _, phrase := range badPhrases {
		if strings.Contains(out, phrase) {
			t.Errorf("graph output contains title-case phrase %q (want sentence case)", phrase)
		}
	}
}

// ── kernels tests ─────────────────────────────────────────────────────────────

// TestKernelsMLP verifies the kernels command for the MLP example.
func TestKernelsMLP(t *testing.T) {
	var buf bytes.Buffer
	code := kernelsCmdW([]string{"mlp"}, &buf)
	if code != 0 {
		t.Fatalf("kernelsCmdW(mlp) exited %d, want 0; output: %s", code, buf.String())
	}
	out := buf.String()

	required := []string{
		"@compute",
		"fn main",
		"kernel 0",
	}
	for _, phrase := range required {
		if !strings.Contains(out, phrase) {
			t.Errorf("kernels mlp: output missing %q\nfull output:\n%s", phrase, out)
		}
	}
}

// ── flag parsing tests ────────────────────────────────────────────────────────

// TestFlagParsing verifies parseFlags handles --device=cpu, --debug=2, --viz.
func TestFlagParsing(t *testing.T) {
	t.Run("device flag", func(t *testing.T) {
		flags, rest, err := parseFlags("test", []string{"--device=cpu", "arg1"})
		if err != nil {
			t.Fatal(err)
		}
		if flags.device != "cpu" {
			t.Errorf("device = %q, want %q", flags.device, "cpu")
		}
		if len(rest) != 1 || rest[0] != "arg1" {
			t.Errorf("rest = %v, want [arg1]", rest)
		}
	})

	t.Run("debug flag", func(t *testing.T) {
		flags, _, err := parseFlags("test", []string{"--debug=2"})
		if err != nil {
			t.Fatal(err)
		}
		if flags.debug != 2 {
			t.Errorf("debug = %d, want 2", flags.debug)
		}
	})

	t.Run("viz flag", func(t *testing.T) {
		flags, _, err := parseFlags("test", []string{"--viz"})
		if err != nil {
			t.Fatal(err)
		}
		if !flags.viz {
			t.Error("viz should be true when --viz is passed")
		}
	})

	t.Run("unknown flag returns error", func(t *testing.T) {
		_, _, err := parseFlags("test", []string{"--unknown-flag"})
		if err == nil {
			t.Error("expected error for unknown flag, got nil")
		}
	})

	t.Run("defaults", func(t *testing.T) {
		flags, _, err := parseFlags("test", []string{})
		if err != nil {
			t.Fatal(err)
		}
		if flags.device != "webgpu" {
			t.Errorf("default device = %q, want webgpu", flags.device)
		}
		if flags.debug != 0 {
			t.Errorf("default debug = %d, want 0", flags.debug)
		}
		if flags.viz {
			t.Error("default viz should be false")
		}
	})
}

// ── GPU-dependent tests ───────────────────────────────────────────────────────

// TestRunMLP runs the MLP example end-to-end on GPU.
func TestRunMLP(t *testing.T) {
	requireGPU(t)

	var buf bytes.Buffer
	code := runCmdW([]string{"mlp"}, &buf)
	if code != 0 {
		t.Fatalf("runCmdW(mlp) exited %d, want 0; output: %s", code, buf.String())
	}
	out := buf.String()

	// Output should contain shape info and prediction values.
	if !strings.Contains(out, "shape") {
		t.Errorf("run mlp: output missing 'shape'; got:\n%s", out)
	}
}

// TestTrainMLP runs 5 training steps of the MLP example.
func TestTrainMLP(t *testing.T) {
	requireGPU(t)

	var buf bytes.Buffer
	code := trainCmdW([]string{"--steps=5", "--log-every=1", "mlp"}, &buf)
	if code != 0 {
		t.Fatalf("trainCmdW(mlp --steps=5) exited %d, want 0; output: %s", code, buf.String())
	}
	out := buf.String()

	// Should have at least 5 loss lines (step 0 + steps 1-5).
	lossLines := 0
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "loss=") {
			lossLines++
		}
	}
	if lossLines < 5 {
		t.Errorf("train mlp --steps=5: expected >= 5 loss lines, got %d\noutput:\n%s", lossLines, out)
	}
}

// TestDoctorSuccess runs the doctor command when a GPU is available.
func TestDoctorSuccess(t *testing.T) {
	requireGPU(t)

	var buf bytes.Buffer
	code := doctorCmdW([]string{}, &buf)
	if code != 0 {
		t.Fatalf("doctorCmdW exited %d, want 0; output: %s", code, buf.String())
	}
	out := buf.String()

	if !strings.Contains(out, "ready") {
		t.Errorf("doctor output should contain 'ready'; got:\n%s", out)
	}
}
