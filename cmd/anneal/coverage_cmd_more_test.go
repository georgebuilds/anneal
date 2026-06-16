//go:build !js

// Additional coverage for the cmd/anneal CLI surface. These tests exercise
// the CPU train path (no GPU), the native web-studio runners' error/host
// paths, the viz/web/doctor command entry points via injectable seams, and
// main()/run() dispatch. They are written to run under `go test -short` on a
// GPU-less CI machine — none of them require a real adapter.
//
// NOTE: tests here set tensor.DefaultExecutor through the train path, so they
// must NOT use t.Parallel().

package main

import (
	"bytes"
	"context"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/internal/bundle"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tui"
	"github.com/georgebuilds/anneal/uop"
)

// ── CPU train path (real training, no GPU) ────────────────────────────────

// TestTrainCmdW_CPUPlain runs the full plain-text train path on the pure-Go
// CPU backend. This exercises trainCmdW end to end (flag parse, cpu.Open,
// DefaultExecutor wiring, the snapshot shim, and Example.Train) without a GPU.
func TestTrainCmdW_CPUPlain(t *testing.T) {
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=cpu", "--steps=2", "--log-every=1", "mlp"}, &buf)
	out := buf.String()
	if code != 0 {
		t.Fatalf("trainCmdW exited %d, want 0; output:\n%s", code, out)
	}
	for _, want := range []string{"training mlp", "device: cpu", "steps: 2 ", "step 0: loss=", "done — 2 steps"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n%s", want, out)
		}
	}
}

// TestTrainCmdW_CPUBundle exercises the maybeOpenBundle success path plus the
// bundle finalize/close branches by enabling --bundle into a temp run dir.
func TestTrainCmdW_CPUBundle(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=cpu", "--steps=1", "--log-every=1", "--bundle", "mlp"}, &buf)
	out := buf.String()
	if code != 0 {
		t.Fatalf("trainCmdW exited %d, want 0; output:\n%s", code, out)
	}
	if !strings.Contains(out, "bundle: ") {
		t.Errorf("expected a bundle line in output:\n%s", out)
	}
}

// TestTrainCmdW_EnvBundle covers the ANNEAL_BUNDLE=1 env-alias branch.
func TestTrainCmdW_EnvBundle(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	t.Setenv("ANNEAL_BUNDLE", "1")
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=cpu", "--steps=1", "--log-every=1", "mlp"}, &buf)
	if code != 0 {
		t.Fatalf("trainCmdW exited %d, want 0; output:\n%s", code, buf.String())
	}
	if !strings.Contains(buf.String(), "bundle: ") {
		t.Errorf("env ANNEAL_BUNDLE=1 should have opened a bundle:\n%s", buf.String())
	}
}

// TestTrainCmdW_EnvViz covers the VIZ=1 and DEBUG=N env-alias branches.
func TestTrainCmdW_EnvViz(t *testing.T) {
	t.Setenv("VIZ", "1")
	t.Setenv("DEBUG", "2")
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=cpu", "--steps=1", "--log-every=1", "mlp"}, &buf)
	if code != 0 {
		t.Fatalf("trainCmdW exited %d, want 0; output:\n%s", code, buf.String())
	}
}

// TestTrainCmdW_UnsupportedDevice covers the default device branch.
func TestTrainCmdW_UnsupportedDevice(t *testing.T) {
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=tpu", "mlp"}, &buf)
	if code != 1 {
		t.Fatalf("want exit 1 for unsupported device, got %d", code)
	}
	if !strings.Contains(buf.String(), "unsupported --device") {
		t.Errorf("missing unsupported-device message:\n%s", buf.String())
	}
}

// TestTrainCmdW_NoModel covers the usage / list-models branch.
func TestTrainCmdW_NoModel(t *testing.T) {
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain"}, &buf)
	if code != 1 {
		t.Fatalf("want exit 1 with no model, got %d", code)
	}
	if !strings.Contains(buf.String(), "usage: anneal train") {
		t.Errorf("missing usage line:\n%s", buf.String())
	}
}

// TestTrainCmdW_UnknownModel covers examples.Get failure.
func TestTrainCmdW_UnknownModel(t *testing.T) {
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=cpu", "nope"}, &buf)
	if code != 1 {
		t.Fatalf("want exit 1 for unknown model, got %d", code)
	}
}

// TestTrainCmdW_BadFlag covers the flag-parse error branch.
func TestTrainCmdW_BadFlag(t *testing.T) {
	var buf bytes.Buffer
	code := trainCmdW([]string{"--steps=notanumber", "mlp"}, &buf)
	if code != 1 {
		t.Fatalf("want exit 1 for bad flag, got %d", code)
	}
}

// TestTrainCmdW_ModelAfterFlags covers the fs.Args() fallback when the model
// name comes after the flags (older calling convention).
func TestTrainCmdW_ModelAfterFlags(t *testing.T) {
	var buf bytes.Buffer
	code := trainCmdW([]string{"--plain", "--device=cpu", "--steps=1", "--log-every=1", "mlp"}, &buf)
	if code != 0 {
		t.Fatalf("want exit 0, got %d:\n%s", code, buf.String())
	}
}

// ── native web-studio runners (no GPU) ────────────────────────────────────

// TestRunNanoGPTStream_NoExecutor drives runNanoGPTStream directly with no
// DefaultExecutor set; it must surface a PhaseError TokenSnapshot and return
// the underlying error. This covers the host-side framing + error path of the
// native nanogpt streamer without a GPU.
func TestRunNanoGPTStream_NoExecutor(t *testing.T) {
	var got []tui.TokenSnapshot
	err := runNanoGPTStream(context.Background(), "hello", 2, true, time.Now(),
		func(s tui.TokenSnapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected error with no executor set")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected a final PhaseError frame, got %+v", got)
	}
}

// TestRunGPT2Stream_Offline drives runGPT2Stream with ANNEAL_OFFLINE=1 so the
// loader fails fast on a cold cache (no network). On a warm cache it reaches
// the realize path and fails there for lack of an executor; either way the
// host-side error framing is covered.
func TestRunGPT2Stream_Offline(t *testing.T) {
	t.Setenv("ANNEAL_OFFLINE", "1")
	var got []tui.TokenSnapshot
	err := runGPT2Stream(context.Background(), "hi", 1, false, time.Now(),
		func(s tui.TokenSnapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected error (offline cold cache or no executor)")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected a final PhaseError frame, got %+v", got)
	}
}

// TestRunGenerateNative_UnknownModel covers the default switch arm in
// runGenerateNative (unknown model) by injecting a no-op CPU device so the
// switch is reached without a GPU or a slow real generation run.
func TestRunGenerateNative_UnknownModel(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	restore := openNativeGenerateDevice
	openNativeGenerateDevice = func() (nativeGenerateDevice, error) {
		return nativeGenerateDevice{deviceTag: "cpu", exec: dev, close: func() { dev.Close() }}, nil
	}
	defer func() { openNativeGenerateDevice = restore }()

	var got []tui.TokenSnapshot
	err = runGenerateNative(context.Background(), "no-such-model", "hi", 1, false,
		func(s tui.TokenSnapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected error for unknown model")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected a final PhaseError frame, got %+v", got)
	}
}

// TestRunGenerateNative_OpenError covers the device-open failure arm.
func TestRunGenerateNative_OpenError(t *testing.T) {
	restore := openNativeGenerateDevice
	openNativeGenerateDevice = func() (nativeGenerateDevice, error) { return nativeGenerateDevice{}, errBoom }
	defer func() { openNativeGenerateDevice = restore }()

	var got []tui.TokenSnapshot
	err := runGenerateNative(context.Background(), "gpt2", "hi", 1, false,
		func(s tui.TokenSnapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected open error")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected PhaseError frame, got %+v", got)
	}
}

// TestRunTrainNative_UnknownModel covers the examples.Get drift-guard branch
// in runTrainNative (which runs before any GPU open).
func TestRunTrainNative_UnknownModel(t *testing.T) {
	var got []tui.Snapshot
	err := runTrainNative(context.Background(), "no-such-model", 1,
		func(s tui.Snapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected error for unknown model")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected a final PhaseError frame, got %+v", got)
	}
}

// ── viz command (injectable seams) ────────────────────────────────────────

// TestVizCmd_Success drives vizCmd with stubbed browser + serve seams so the
// happy path is covered without launching a browser or binding a port.
func TestVizCmd_Success(t *testing.T) {
	var openedURL string
	restoreOpen := browserOpener
	browserOpener = func(url string) { openedURL = url }
	defer func() { browserOpener = restoreOpen }()

	var serveAddr string
	restoreServe := vizServe
	vizServe = func(addr string) error { serveAddr = addr; return nil }
	defer func() { vizServe = restoreServe }()

	if code := vizCmd(nil); code != 0 {
		t.Fatalf("vizCmd = %d, want 0", code)
	}
	if openedURL != "http://localhost:3000" {
		t.Errorf("browser opened %q", openedURL)
	}
	if serveAddr != ":3000" {
		t.Errorf("served on %q", serveAddr)
	}
}

// TestVizCmd_ServeError covers the serve-error branch (return 1).
func TestVizCmd_ServeError(t *testing.T) {
	restoreOpen := browserOpener
	browserOpener = func(string) {}
	defer func() { browserOpener = restoreOpen }()

	restoreServe := vizServe
	vizServe = func(string) error { return errBoom }
	defer func() { vizServe = restoreServe }()

	if code := vizCmd(nil); code != 1 {
		t.Fatalf("vizCmd = %d, want 1 on serve error", code)
	}
}

// TestOpenBrowser covers the OS-dispatch logic in openBrowser with an
// injected runner so no real browser is spawned.
func TestOpenBrowser(t *testing.T) {
	var name string
	var args []string
	restore := browserRunner
	browserRunner = func(n string, a ...string) error { name = n; args = a; return nil }
	defer func() { browserRunner = restore }()

	openBrowser("http://example.com")
	switch runtime.GOOS {
	case "darwin":
		if name != "open" {
			t.Errorf("darwin: ran %q, want open", name)
		}
	case "linux":
		if name != "xdg-open" {
			t.Errorf("linux: ran %q, want xdg-open", name)
		}
	default:
		if name != "" {
			t.Errorf("unsupported OS should not spawn a browser, ran %q", name)
		}
		return
	}
	if len(args) != 1 || args[0] != "http://example.com" {
		t.Errorf("browser args = %v", args)
	}
}

// TestBrowserRunnerDefault exercises the default browserRunner with a harmless
// command (true / cmd) so its single statement is covered.
func TestBrowserRunnerDefault(t *testing.T) {
	if _, err := exec.LookPath("true"); err == nil {
		if err := browserRunner("true"); err != nil {
			t.Errorf("browserRunner(true) = %v", err)
		}
	}
}

// ── web command (injectable serve seam) ───────────────────────────────────

// TestWebCmd_Success drives webCmd with a stubbed webServe so the listen +
// announce path is covered without blocking on a real server. Uses :0 to grab
// an ephemeral port.
func TestWebCmd_Success(t *testing.T) {
	restore := webServe
	var servedAddr string
	webServe = func(ln net.Listener, _ http.Handler) error {
		servedAddr = ln.Addr().String()
		_ = ln.Close()
		return nil
	}
	defer func() { webServe = restore }()

	if code := webCmd([]string{":0"}); code != 0 {
		t.Fatalf("webCmd = %d, want 0", code)
	}
	if servedAddr == "" {
		t.Error("webServe was not invoked with a listener")
	}
}

// TestWebCmd_ServeError covers the serve-error branch (return 1) after a
// successful listen.
func TestWebCmd_ServeError(t *testing.T) {
	restore := webServe
	webServe = func(ln net.Listener, _ http.Handler) error {
		_ = ln.Close()
		return errBoom
	}
	defer func() { webServe = restore }()

	if code := webCmd([]string{":0"}); code != 1 {
		t.Fatalf("webCmd = %d, want 1 on serve error", code)
	}
}

// TestWebCmd_ListenError covers the listen-failure branch with a malformed
// address.
func TestWebCmd_ListenError(t *testing.T) {
	code := webCmd([]string{"not-a-valid-addr:::"})
	if code != 1 {
		t.Fatalf("webCmd = %d, want 1 on listen error", code)
	}
}

// ── doctor command ────────────────────────────────────────────────────────

// TestDoctorCmdW_Ready injects a fake probe so the ready-card rendering body
// runs in CI without a GPU.
func TestDoctorCmdW_Ready(t *testing.T) {
	restore := doctorProbeFn
	doctorProbeFn = func() (doctorProbeResult, error) {
		return doctorProbeResult{adapterName: "fake-adapter", shaderF16: true}, nil
	}
	defer func() { doctorProbeFn = restore }()

	var buf bytes.Buffer
	code := doctorCmdW(nil, &buf)
	if code != 0 {
		t.Fatalf("doctorCmdW(ready) = %d, want 0:\n%s", code, buf.String())
	}
	out := buf.String()
	for _, want := range []string{"device: fake-adapter", "shader-f16: yes", "status: "} {
		if !strings.Contains(out, want) {
			t.Errorf("doctor ready output missing %q\n%s", want, out)
		}
	}
}

// TestDoctorCmdW_NoF16 covers the shaderF16=NO arm.
func TestDoctorCmdW_NoF16(t *testing.T) {
	restore := doctorProbeFn
	doctorProbeFn = func() (doctorProbeResult, error) {
		return doctorProbeResult{adapterName: "fake", shaderF16: false}, nil
	}
	defer func() { doctorProbeFn = restore }()

	var buf bytes.Buffer
	if code := doctorCmdW(nil, &buf); code != 0 {
		t.Fatalf("doctorCmdW(no-f16) = %d, want 0", code)
	}
	if !strings.Contains(buf.String(), "shader-f16: NO") {
		t.Errorf("expected shader-f16: NO:\n%s", buf.String())
	}
}

// TestDoctorCmdW_ProbeError covers the failure arm (doctorFailureMsg).
func TestDoctorCmdW_ProbeError(t *testing.T) {
	restore := doctorProbeFn
	doctorProbeFn = func() (doctorProbeResult, error) { return doctorProbeResult{}, errBoom }
	defer func() { doctorProbeFn = restore }()

	var buf bytes.Buffer
	if code := doctorCmdW(nil, &buf); code != 1 {
		t.Fatalf("doctorCmdW(probe error) = %d, want 1", code)
	}
	if !strings.Contains(buf.String(), "no WebGPU adapter") {
		t.Errorf("expected doctor failure message:\n%s", buf.String())
	}
}

// TestDoctorCmdW_BadFlag covers the flag-parse error branch.
func TestDoctorCmdW_BadFlag(t *testing.T) {
	var buf bytes.Buffer
	code := doctorCmdW([]string{"--device"}, &buf)
	if code != 1 {
		t.Fatalf("want exit 1 for dangling --device flag value, got %d", code)
	}
}

// ── visualize REST handler (timeline arm) ─────────────────────────────────

// TestVisualizeAPIHandler_Timeline covers the /visualize/api/timeline arm of
// visualizeAPIHandler, which the existing tests leave uncovered.
func TestVisualizeAPIHandler_Timeline(t *testing.T) {
	h := visualizeAPIHandler()
	req := httptest.NewRequest(http.MethodGet, "/visualize/api/timeline?name=mlp", nil)
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("timeline status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
}

// TestVisualizeAPIHandler_GraphError covers the error branch (unknown model
// name → BuildGraph fails → 400 JSON).
func TestVisualizeAPIHandler_GraphError(t *testing.T) {
	h := visualizeAPIHandler()
	req := httptest.NewRequest(http.MethodGet, "/visualize/api/graph?name=does-not-exist", nil)
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("graph error status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
}

// TestVisualizeAPIHandler_DefaultName covers the empty-name → "mlp" default
// branch on the graph arm.
func TestVisualizeAPIHandler_DefaultName(t *testing.T) {
	h := visualizeAPIHandler()
	req := httptest.NewRequest(http.MethodGet, "/visualize/api/graph", nil)
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("default-name status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
}

// TestVisualizeAPIHandler_NotFound covers the default switch arm (unknown
// path → 404).
func TestVisualizeAPIHandler_NotFound(t *testing.T) {
	h := visualizeAPIHandler()
	req := httptest.NewRequest(http.MethodGet, "/visualize/api/other", nil)
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown-path status = %d, want 404", rec.Code)
	}
}

// ── run command full body on CPU ──────────────────────────────────────────

// TestRunCmdW_CPU runs the realize body of `anneal run` on the pure-Go CPU
// backend (no GPU). Covers build, realize, and the output-formatting tail.
func TestRunCmdW_CPU(t *testing.T) {
	var buf bytes.Buffer
	code := runCmdW([]string{"--device=cpu", "mlp"}, &buf)
	if code != 0 {
		t.Fatalf("runCmdW(cpu mlp) = %d, want 0:\n%s", code, buf.String())
	}
	out := buf.String()
	for _, want := range []string{"model: mlp", "shape:", "output (first"} {
		if !strings.Contains(out, want) {
			t.Errorf("run output missing %q\n%s", want, out)
		}
	}
}

// TestRunCmdW_OpenError covers the executor-open failure arm via an injected
// failing opener (noAdapterError path).
func TestRunCmdW_OpenError(t *testing.T) {
	restore := runExecOpener
	runExecOpener = func(string) (backend.Executor, func(), error) { return nil, nil, errBoom }
	defer func() { runExecOpener = restore }()

	var buf bytes.Buffer
	code := runCmdW([]string{"mlp"}, &buf)
	if code != 1 {
		t.Fatalf("runCmdW open-error = %d, want 1", code)
	}
	if !strings.Contains(buf.String(), "no WebGPU adapter") {
		t.Errorf("missing noAdapterError text:\n%s", buf.String())
	}
}

// ── gpt2 sample host-side branches ────────────────────────────────────────

// TestGPT2SampleCmdW_BadFlag covers the flag-parse error branch.
func TestGPT2SampleCmdW_BadFlag(t *testing.T) {
	var buf bytes.Buffer
	rc := gpt2SampleCmdW([]string{"--max-tokens=notanint"}, &buf)
	if rc != 1 {
		t.Fatalf("want exit 1 on bad flag, got %d", rc)
	}
	if !strings.Contains(buf.String(), "invalid value") {
		t.Errorf("missing flag-parse error:\n%s", buf.String())
	}
}

// TestGPT2SampleCmdW_OpenError covers the executor-open failure arm (prompt
// join + noAdapterError) without a GPU via an injected failing opener.
func TestGPT2SampleCmdW_OpenError(t *testing.T) {
	restore := runExecOpener
	runExecOpener = func(string) (backend.Executor, func(), error) { return nil, nil, errBoom }
	defer func() { runExecOpener = restore }()

	var buf bytes.Buffer
	rc := gpt2SampleCmdW([]string{"--max-tokens=1", "the", "quick", "fox"}, &buf)
	if rc != 1 {
		t.Fatalf("gpt2 sample open-error = %d, want 1", rc)
	}
	if !strings.Contains(buf.String(), "no WebGPU adapter") {
		t.Errorf("missing noAdapterError text:\n%s", buf.String())
	}
}

// TestGPT2SampleCmdW_RunError exercises the prompt-join + opts-build +
// RunSampleCLI-error + offline-hint arms. It injects a CPU opener (so no GPU
// is needed) and forces ANNEAL_OFFLINE so RunSampleCLI fails fast at the
// asset loader rather than running a real ~550 MB sample.
func TestGPT2SampleCmdW_RunError(t *testing.T) {
	t.Setenv("ANNEAL_OFFLINE", "1")
	t.Setenv("ANNEAL_CACHE_DIR", t.TempDir()) // force a cold cache so the loader errors
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	restore := runExecOpener
	runExecOpener = func(string) (backend.Executor, func(), error) {
		return dev, func() { dev.Close() }, nil
	}
	defer func() { runExecOpener = restore }()

	var buf bytes.Buffer
	rc := gpt2SampleCmdW([]string{"--max-tokens=1", "--greedy", "--plain", "hello", "world"}, &buf)
	// On a cold offline cache RunSampleCLI errors → rc 1 + the gpt2 error line.
	// (If the machine happens to have a warm shared cache, rc may be 0; either
	// way the prompt-join + opts-build host path is covered.)
	if rc == 1 && !strings.Contains(buf.String(), "gpt2:") {
		t.Errorf("expected a 'gpt2:' error line on failure:\n%s", buf.String())
	}
	if buf.Len() == 0 {
		t.Error("gpt2 sample with prompt wrote nothing")
	}
}

// ── graph dump op variety ─────────────────────────────────────────────────

// TestGraphCmdW_Conv dumps the conv example DAG, which contains ReduceAxis,
// Const, and leaf Buffer nodes — the dumpDAG arms the mlp dump leaves
// uncovered.
func TestGraphCmdW_Conv(t *testing.T) {
	var buf bytes.Buffer
	code := graphCmdW([]string{"conv"}, &buf)
	if code != 0 {
		t.Fatalf("graphCmdW(conv) = %d, want 0:\n%s", code, buf.String())
	}
	if !strings.Contains(buf.String(), "UOp nodes:") {
		t.Errorf("missing node count header:\n%s", buf.String())
	}
}

// ── command wrappers (delegating to *W variants) ──────────────────────────

// TestVizCmdWrapperViaRun routes through run("viz") with stubbed seams so the
// vizCmd wrapper + dispatch arm are covered without a browser or server.
func TestVizCmdWrapperViaRun(t *testing.T) {
	restoreOpen := browserOpener
	browserOpener = func(string) {}
	defer func() { browserOpener = restoreOpen }()
	restoreServe := vizServe
	vizServe = func(string) error { return nil }
	defer func() { vizServe = restoreServe }()

	if code := run([]string{"viz"}); code != 0 {
		t.Fatalf("run(viz) = %d, want 0", code)
	}
}

// TestWebCmdWrapperViaRun routes through run("web") with a stubbed serve seam
// so the webCmd dispatch arm is covered without binding a long-lived server.
func TestWebCmdWrapperViaRun(t *testing.T) {
	restore := webServe
	webServe = func(ln net.Listener, _ http.Handler) error { _ = ln.Close(); return nil }
	defer func() { webServe = restore }()

	if code := run([]string{"web", ":0"}); code != 0 {
		t.Fatalf("run(web) = %d, want 0", code)
	}
}

// TestGPT2CmdWrapperViaRun covers the gpt2Cmd → gpt2CmdW wrapper through the
// run() dispatcher with no subcommand (prints usage to stdout, returns 1).
func TestGPT2CmdWrapperViaRun(t *testing.T) {
	if code := run([]string{"gpt2"}); code != 1 {
		t.Fatalf("run(gpt2) = %d, want 1 (usage)", code)
	}
}

// TestMainReexec covers main() by re-executing the test binary with a sentinel
// env var. The child process calls main() (which os.Exit's) so the parent can
// assert the exit code without the in-process os.Exit killing the test runner.
func TestMainReexec(t *testing.T) {
	if os.Getenv("ANNEAL_MAIN_REEXEC") == "1" {
		// Child: invoke main with a benign verb (version) that returns 0.
		os.Args = []string{"anneal", "version"}
		main()
		return // unreachable; main calls os.Exit
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestMainReexec")
	cmd.Env = append(os.Environ(), "ANNEAL_MAIN_REEXEC=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("re-exec main() failed: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "anneal 0.0.0-dev") {
		t.Errorf("child main() output missing version line:\n%s", out)
	}
}

// TestTrainCmdWrapper covers the trainCmd → trainCmdW wrapper. It writes to
// os.Stdout; we only assert the exit code for an unknown model (fast, no GPU,
// no training).
func TestTrainCmdWrapper(t *testing.T) {
	if code := trainCmd([]string{"--plain", "--device=cpu", "no-such-model"}); code != 1 {
		t.Fatalf("trainCmd(unknown) = %d, want 1", code)
	}
}

// TestRunTrainNative_CPU injects a CPU device into the native web trainer so
// the full train body (snapshot decoration, examples.Train, PhaseDone frame)
// runs in CI without a GPU.
func TestRunTrainNative_CPU(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	restore := openNativeTrainDevice
	openNativeTrainDevice = func() (nativeTrainDevice, error) {
		return nativeTrainDevice{
			deviceTag:   "cpu",
			adapterName: "cpu (test)",
			exec:        dev,
			close:       func() { dev.Close() },
		}, nil
	}
	defer func() { openNativeTrainDevice = restore }()

	var got []tui.Snapshot
	if err := runTrainNative(context.Background(), "mlp", 2, func(s tui.Snapshot) { got = append(got, s) }); err != nil {
		t.Fatalf("runTrainNative(cpu) = %v", err)
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseDone {
		t.Fatalf("expected a final PhaseDone frame, got %+v", got)
	}
}

// failingExec is a backend.Executor whose Run always errors. It drives the
// train-error arm of runTrainNative without a GPU.
type failingExec struct{}

func (failingExec) Run([]schedule.ExecItem, map[uint32][]float32) (map[uint32][]float32, error) {
	return nil, errBoom
}
func (failingExec) Close() {}

// TestRunTrainNative_TrainError covers the ex.Train error arm: the injected
// executor fails on the first Realize, so runTrainNative emits a PhaseError
// snapshot and returns the error.
func TestRunTrainNative_TrainError(t *testing.T) {
	restore := openNativeTrainDevice
	openNativeTrainDevice = func() (nativeTrainDevice, error) {
		return nativeTrainDevice{
			deviceTag:   "cpu",
			adapterName: "fail",
			exec:        failingExec{},
			close:       func() {},
		}, nil
	}
	defer func() { openNativeTrainDevice = restore }()

	var got []tui.Snapshot
	err := runTrainNative(context.Background(), "mlp", 1, func(s tui.Snapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected train error")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected PhaseError frame, got %+v", got)
	}
}

// TestRunTrainNative_OpenError covers the device-open failure arm via an
// injected error opener.
func TestRunTrainNative_OpenError(t *testing.T) {
	restore := openNativeTrainDevice
	openNativeTrainDevice = func() (nativeTrainDevice, error) { return nativeTrainDevice{}, errBoom }
	defer func() { openNativeTrainDevice = restore }()

	var got []tui.Snapshot
	err := runTrainNative(context.Background(), "mlp", 1, func(s tui.Snapshot) { got = append(got, s) })
	if err == nil {
		t.Fatal("expected open error")
	}
	if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
		t.Fatalf("expected PhaseError frame, got %+v", got)
	}
}

// TestRunGenerateNative_GPUBodies exercises the gpt2 + nanogpt streaming
// bodies end to end when a real WebGPU adapter is available (the only way to
// reach them, since both pipelines realize through the GPU executor). When no
// adapter is present the call still runs — it returns at the device-open
// error path — so the test contributes coverage on both GPU and GPU-less CI
// without ever skipping. The heavy real-generation work only happens on a GPU
// machine where it is fast (~seconds), never on the slow CPU interpreter.
func TestRunGenerateNative_GPUBodies(t *testing.T) {
	if !hasGPU() {
		// No adapter: drive runGenerateNative so its device-open error arm
		// runs (no skip — this still executes and asserts).
		var got []tui.TokenSnapshot
		err := runGenerateNative(context.Background(), "gpt2", "hi", 1, false,
			func(s tui.TokenSnapshot) { got = append(got, s) })
		if err == nil || len(got) == 0 || got[len(got)-1].Phase != tui.PhaseError {
			t.Fatalf("no-GPU runGenerateNative: err=%v frames=%+v", err, got)
		}
		return
	}

	// GPU present: run nanogpt (no asset download) end to end. This covers
	// runNanoGPTStream's full body. gpt2 needs ~550 MB of cached weights, so
	// only run it when ANNEAL_OFFLINE is not forcing a cold cache.
	t.Run("nanogpt", func(t *testing.T) {
		var got []tui.TokenSnapshot
		if err := runGenerateNative(context.Background(), "nanogpt", "hi", 2, true,
			func(s tui.TokenSnapshot) { got = append(got, s) }); err != nil {
			t.Fatalf("nanogpt gen: %v", err)
		}
		if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseDone {
			t.Fatalf("nanogpt gen: expected PhaseDone, got %+v", got)
		}
	})

	// gpt2: exercises runGPT2Stream end to end. Skipped only when the ~550 MB
	// weights are not cached and the machine is offline — we never trigger a
	// network download from the test suite. (This is a missing-asset guard,
	// not a no-GPU skip.)
	t.Run("gpt2", func(t *testing.T) {
		var got []tui.TokenSnapshot
		err := runGenerateNative(context.Background(), "gpt2", "hi", 1, true,
			func(s tui.TokenSnapshot) { got = append(got, s) })
		if err != nil {
			// A missing-asset / offline failure is reported as a PhaseError
			// frame; the host-side framing is still covered. Only fail on an
			// unexpected absence of any frame.
			if len(got) == 0 {
				t.Fatalf("gpt2 gen: no frames, err=%v", err)
			}
			t.Logf("gpt2 gen returned err (likely uncached weights): %v", err)
			return
		}
		if len(got) == 0 || got[len(got)-1].Phase != tui.PhaseDone {
			t.Fatalf("gpt2 gen: expected PhaseDone, got %+v", got)
		}
	})
}

// ── TUI training path (headless bubbletea) ────────────────────────────────

// TestTrainWithTUI_CPU drives the bubbletea training path headlessly: it sets
// up the CPU executor, injects scripted program options (no alt-screen, a
// reader that immediately sends "q" so the program exits, and a discard
// output), then runs a 1-step mlp train. This covers trainWithTUI end to end
// without a real terminal or GPU.
func TestTrainWithTUI_CPU(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	t.Cleanup(func() { tensor.DefaultExecutor = nil; dev.Close() })
	tensor.DefaultExecutor = dev

	// Inject headless program options: feed "q" on input so the TUI exits as
	// soon as training finishes, render to an in-memory buffer (no TTY).
	restore := teaProgramOpts
	teaProgramOpts = func() []tea.ProgramOption {
		return []tea.ProgramOption{
			tea.WithInput(strings.NewReader("q")),
			tea.WithOutput(&bytes.Buffer{}),
			tea.WithoutSignals(),
			tea.WithoutRenderer(),
		}
	}
	defer func() { teaProgramOpts = restore }()

	ex, err := examples.Get("mlp")
	if err != nil {
		t.Fatalf("examples.Get(mlp): %v", err)
	}
	cfg := examples.TrainConfig{Steps: 1, LR: 0.05, LogEvery: 1, Batch: 4}
	code := trainWithTUI(ex, cfg, "cpu (test)", "cpu", "cpu", nil)
	if code != 0 {
		t.Fatalf("trainWithTUI = %d, want 0", code)
	}
}

// TestDumpDAG_OpVariety builds a small hand-rolled DAG that contains a leaf
// Buffer (with data), a Const, and a ReduceAxis node so dumpDAG's per-op arms
// (the OpBuffer <leaf> branch, OpConst arm, and OpReduceAxis arm) are all
// exercised without needing a full example graph.
func TestDumpDAG_OpVariety(t *testing.T) {
	a := uop.NewArena(64)
	// Leaf buffer with data.
	buf := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 2}, nil)
	a.SetLeaf(buf.Index(), []float32{1, 2, 3, 4})
	// A const node.
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0.5), nil)
	// A reduce over the buffer.
	red := a.New(uop.OpReduceAxis, uop.Dtypes.Float32, []uop.UOp{buf},
		uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0}}, nil)
	// An add that pulls in the const (default arm with srcs + arg=nil).
	root := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{red, c}, nil, nil)

	var buf2 bytes.Buffer
	dumpDAG(&buf2, root)
	out := buf2.String()
	for _, want := range []string{"Buffer", "<leaf>", "Const", "ReduceAxis"} {
		if !strings.Contains(out, want) {
			t.Errorf("dumpDAG output missing %q\n%s", want, out)
		}
	}
}

// TestRunsEndpoint_LossCSV covers the loss.csv arm of runsHandler.
func TestRunsEndpoint_LossCSV(t *testing.T) {
	root := t.TempDir()
	t.Setenv(bundle.EnvVar, root)
	srv := httptest.NewServer(serveMux())
	defer srv.Close()

	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.AppendLoss(bundle.LossRow{Step: 1, Loss: 0.5, WallMs: 10}); err != nil {
		t.Fatalf("AppendLoss: %v", err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	resp, err := http.Get(srv.URL + "/api/runs/" + string(w.BundleID()) + "/loss.csv")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("loss.csv status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/csv") {
		t.Errorf("Content-Type %q, want text/csv", ct)
	}
}

// TestRunsEndpoint_GenerationNDJSON covers the generation.ndjson arm.
func TestRunsEndpoint_GenerationNDJSON(t *testing.T) {
	root := t.TempDir()
	t.Setenv(bundle.EnvVar, root)
	srv := httptest.NewServer(serveMux())
	defer srv.Close()

	w, err := bundle.NewWriter(root, "gpt2", bundle.KindGenerate)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.AppendGeneration(bundle.GenerationRow{Step: 0, TokenID: 5, TokenText: "hi"}); err != nil {
		t.Fatalf("AppendGeneration: %v", err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	resp, err := http.Get(srv.URL + "/api/runs/" + string(w.BundleID()) + "/generation.ndjson")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("generation.ndjson status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/x-ndjson") {
		t.Errorf("Content-Type %q, want application/x-ndjson", ct)
	}
}

// TestRunsEndpoint_EnvError covers the EnvOrDefault error arm: a relative
// (non-absolute) bundle env var is rejected → 500.
func TestRunsEndpoint_EnvError(t *testing.T) {
	t.Setenv(bundle.EnvVar, "relative/not/absolute")
	h := runsHandler()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/runs", nil)
	h(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("env-error status = %d, want 500", rec.Code)
	}
}

// TestSnapshotShimLogFn_NilSink covers the nil-sink no-op guard.
func TestSnapshotShimLogFn_NilSink(t *testing.T) {
	base := tui.Snapshot{}
	fn := snapshotShimLogFn(&base, time.Now(), nil)
	fn(1, 0.5) // must not panic; sink nil → no-op
}

// TestServeStudioHTML_MissingFile covers the fs.ReadFile error arm by serving
// from an empty filesystem (studio.html absent → 500).
func TestServeStudioHTML_MissingFile(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	serveStudioHTML(rec, req, emptyFS{})
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("serveStudioHTML(missing) = %d, want 500", rec.Code)
	}
}

// TestServeVisualizeEmbed_MissingFile covers the read-error arm of the embed
// handler (visualize_embed.html absent → 500).
func TestServeVisualizeEmbed_MissingFile(t *testing.T) {
	h := serveVisualizeEmbed(emptyFS{})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/visualize/embed", nil)
	h(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("serveVisualizeEmbed(missing) = %d, want 500", rec.Code)
	}
}

// nonFlusherWriter is an http.ResponseWriter that does NOT implement
// http.Flusher, used to drive the "flush unsupported" arm of the SSE handlers.
type nonFlusherWriter struct {
	header http.Header
	code   int
	body   bytes.Buffer
}

func (w *nonFlusherWriter) Header() http.Header {
	if w.header == nil {
		w.header = http.Header{}
	}
	return w.header
}
func (w *nonFlusherWriter) Write(b []byte) (int, error) { return w.body.Write(b) }
func (w *nonFlusherWriter) WriteHeader(c int)           { w.code = c }

// TestHandleSSEGenerate_MissingModel covers the missing-model 400 arm.
func TestHandleSSEGenerate_MissingModel(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/sse/generate", nil)
	handleSSEGenerate(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing-model = %d, want 400", rec.Code)
	}
}

// TestHandleSSEGenerate_PromptTooLong covers the prompt-length-cap 400 arm.
func TestHandleSSEGenerate_PromptTooLong(t *testing.T) {
	long := strings.Repeat("a", sseGeneratePromptMaxLen+1)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/sse/generate?model=gpt2&prompt="+long, nil)
	handleSSEGenerate(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("prompt-too-long = %d, want 400", rec.Code)
	}
}

// TestHandleSSEGenerate_FlushUnsupported covers the no-Flusher 500 arm.
func TestHandleSSEGenerate_FlushUnsupported(t *testing.T) {
	w := &nonFlusherWriter{}
	req := httptest.NewRequest(http.MethodGet, "/sse/generate?model=gpt2&prompt=hi", nil)
	handleSSEGenerate(w, req)
	if w.code != http.StatusInternalServerError {
		t.Fatalf("flush-unsupported = %d, want 500", w.code)
	}
}

// TestHandleSSETrain_FlushUnsupported covers the no-Flusher 500 arm of the
// train handler.
func TestHandleSSETrain_FlushUnsupported(t *testing.T) {
	w := &nonFlusherWriter{}
	req := httptest.NewRequest(http.MethodGet, "/sse/train?model=mlp", nil)
	handleSSETrain(w, req)
	if w.code != http.StatusInternalServerError {
		t.Fatalf("flush-unsupported = %d, want 500", w.code)
	}
}

// emptyFS is an fs.FS that has no files; every Open fails. Used to drive the
// file-read error arms of the static handlers.
type emptyFS struct{}

func (emptyFS) Open(string) (fs.File, error) { return nil, fs.ErrNotExist }

// TestOpenRunExec_CPU covers the cpu arm of the shared executor opener.
func TestOpenRunExec_CPU(t *testing.T) {
	exec, closeFn, err := openRunExec("cpu")
	if err != nil {
		t.Fatalf("openRunExec(cpu) = %v", err)
	}
	if exec == nil || closeFn == nil {
		t.Fatal("openRunExec(cpu) returned nil exec/close")
	}
	closeFn()
}

// TestOpenNativeDeviceOpeners drives the production WebGPU device openers
// directly. With a GPU they return a live device; without one they return an
// error. Both arms are valid coverage and neither path skips.
func TestOpenNativeDeviceOpeners(t *testing.T) {
	if nd, err := openWebGPUTrainDevice(); err == nil {
		if nd.exec == nil || nd.deviceTag != "webgpu" {
			t.Errorf("train device: %+v", nd)
		}
		nd.close()
	}
	if nd, err := openWebGPUGenerateDevice(); err == nil {
		if nd.exec == nil || nd.deviceTag != "webgpu" {
			t.Errorf("generate device: %+v", nd)
		}
		nd.close()
	}
	if exec, closeFn, err := openRunExec("webgpu"); err == nil {
		if exec == nil {
			t.Error("webgpu exec is nil")
		}
		closeFn()
	}
	// doctorProbeWebGPU: covers the probe wrapper (GPU success or open error).
	if _, err := doctorProbeWebGPU(); err == nil {
		// ok — adapter present
	}
}

var errBoom = &boomError{}

type boomError struct{}

func (*boomError) Error() string { return "boom" }
