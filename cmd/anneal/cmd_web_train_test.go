//go:build !js

// Tests for W6 - the train view markup + /sse/train wire contract.
//
// Markup tests (TestWebTrain_*) pin the studio.html surface (controls,
// progressbar role, live region, sparkline, bundle checkbox, deep link).
//
// SSE wire tests (TestSSETrain_*) drive the handler via httptest with a
// stub trainRunner so the contract (Content-Type, frame shape, done event,
// cancel handling, invalid-model rejection) is pinned without a real GPU.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/georgebuilds/anneal/tui"
)

// withStubTrainRunner swaps trainRunnerFn for the duration of the test.
// The stub emits n snapshots (steps 1..n) with a fake loss curve, then
// returns. If ctx is cancelled mid-stream it stops cleanly.
func withStubTrainRunner(t *testing.T, runner trainRunner) {
	t.Helper()
	orig := trainRunnerFn
	trainRunnerFn = runner
	t.Cleanup(func() { trainRunnerFn = orig })
}

// stubRunner returns a trainRunner that pushes n snapshots and returns.
func stubRunner(n int) trainRunner {
	return func(ctx context.Context, model string, steps int, snap func(tui.Snapshot)) error {
		max := steps
		if max <= 0 {
			max = n
		}
		for i := 1; i <= n; i++ {
			select {
			case <-ctx.Done():
				return nil
			default:
			}
			snap(tui.Snapshot{
				Step:         i,
				MaxSteps:     max,
				Loss:         float32(1.0 / float64(i)),
				HasLoss:      true,
				Phase:        tui.PhaseTraining,
				UOpsCount:    100 * i,
				KernelsCount: 5 + i,
				FusedCount:   3 + i,
			})
		}
		snap(tui.Snapshot{Step: n, MaxSteps: max, Phase: tui.PhaseDone})
		return nil
	}
}

// ── markup tests ──────────────────────────────────────────────────────────

// TestWebTrain_StubReplaced pins the W6 markup is wired (not the W0/W1 stub).
func TestWebTrain_StubReplaced(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	// Find the train view section.
	start := strings.Index(body, `id="view-train"`)
	if start < 0 {
		t.Fatal("studio.html missing #view-train section")
	}
	end := strings.Index(body[start:], "</section>")
	section := body[start : start+end]
	if strings.Contains(section, "view: train coming soon") {
		t.Errorf("train view still contains W0 stub copy")
	}
	for _, want := range []string{
		`class="train-pane"`,
		`id="train-model"`,
		`id="train-steps"`,
		`id="train-start"`,
		`id="train-cancel"`,
		`id="train-bundle"`,
		`id="train-progress-bar"`,
		`id="train-stats"`,
		`id="loss-svg"`,
		`id="kernel-svg"`,
	} {
		if !strings.Contains(section, want) {
			t.Errorf("train view section missing %q", want)
		}
	}
}

// TestWebTrain_ControlsLabeled pins every form control has a label or
// aria-label.
func TestWebTrain_ControlsLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	// model select must have aria-label or a paired <label for=>.
	if !strings.Contains(body, `for="train-model"`) {
		t.Errorf("train view missing <label for=\"train-model\">")
	}
	if !strings.Contains(body, `for="train-steps"`) {
		t.Errorf("train view missing <label for=\"train-steps\">")
	}
	// Every <button id="train-*"> must have an aria-label.
	btnRE := regexp.MustCompile(`<button[^>]*id="(train-[a-z-]+)"[^>]*>`)
	for _, m := range btnRE.FindAllStringSubmatch(body, -1) {
		tag := m[0]
		id := m[1]
		if !strings.Contains(tag, "aria-label=") {
			t.Errorf("button #%s missing aria-label: %s", id, tag)
		}
	}
}

// TestWebTrain_ProgressbarRole pins the progressbar has the role attribute
// plus valuemin/max/now.
func TestWebTrain_ProgressbarRole(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<div[^>]*id="train-progress-bar"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("train view missing #train-progress-bar")
	}
	for _, want := range []string{
		`role="progressbar"`,
		`aria-valuemin="0"`,
		`aria-valuemax="100"`,
		`aria-valuenow="0"`,
		`aria-label="training progress"`,
	} {
		if !strings.Contains(tag, want) {
			t.Errorf("progressbar missing %s: %s", want, tag)
		}
	}
}

// TestWebTrain_LiveRegionStats pins #train-stats is a polite live region
// (so step + loss values are announced without stealing focus).
func TestWebTrain_LiveRegionStats(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<div[^>]*id="train-stats"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("train view missing #train-stats")
	}
	if !strings.Contains(tag, `aria-live="polite"`) {
		t.Errorf("#train-stats missing aria-live=\"polite\": %s", tag)
	}
	if !strings.Contains(tag, `aria-atomic="false"`) {
		t.Errorf("#train-stats missing aria-atomic=\"false\": %s", tag)
	}
}

// TestWebTrain_SparklineSVGLabeled pins the loss SVG has aria-label,
// tabindex=0, and a textual fallback via <desc>.
func TestWebTrain_SparklineSVGLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<svg[^>]*id="loss-svg"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("train view missing #loss-svg")
	}
	if !strings.Contains(tag, `aria-label="loss sparkline"`) {
		t.Errorf("loss-svg missing aria-label: %s", tag)
	}
	if !strings.Contains(tag, `tabindex="0"`) {
		t.Errorf("loss-svg missing tabindex=\"0\": %s", tag)
	}
	if !strings.Contains(body, `id="loss-svg-desc"`) {
		t.Errorf("studio.html missing #loss-svg-desc (textual fallback for the sparkline)")
	}
}

// TestWebTrain_BundleCheckboxLabeled pins the bundle checkbox is labelled
// AND defaults to checked (web tier default = ON, per spec §6).
func TestWebTrain_BundleCheckboxLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="train-bundle"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("train view missing #train-bundle")
	}
	if !strings.Contains(tag, `type="checkbox"`) {
		t.Errorf("#train-bundle is not a checkbox: %s", tag)
	}
	if !strings.Contains(tag, `aria-label=`) {
		t.Errorf("#train-bundle missing aria-label: %s", tag)
	}
	if !strings.Contains(tag, `checked`) {
		t.Errorf("#train-bundle must default to checked (web tier default = ON, spec §6): %s", tag)
	}
}

// TestWebTrain_DeepLinkURL pins /t/<model> resolves to the studio shell.
// The History API router takes it from there.
func TestWebTrain_DeepLinkURL(t *testing.T) {
	srv := newWebServer(t)
	for _, p := range []string{"/t/mlp", "/t/conv", "/t/nanogpt"} {
		resp, body := get(t, srv, p)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s: status %d, want 200", p, resp.StatusCode)
		}
		if !strings.Contains(body, `id="brand-mark"`) {
			t.Errorf("GET %s: not the studio shell", p)
		}
	}
}

// TestWebTrain_RendererWired pins studio.js routes /t/<model> to a real
// renderer (not the W0 no-op).
func TestWebTrain_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "renderTrainView") {
		t.Errorf("studio.js missing renderTrainView (W6 renderer)")
	}
	// The renderer must be wired into the RENDERERS dispatch table.
	if !regexp.MustCompile(`train:\s*function\s*renderTrain\b`).MatchString(body) {
		t.Errorf("studio.js train renderer not wired into RENDERERS")
	}
	// The dispatch body must call renderTrainView() (not be a no-op).
	if !regexp.MustCompile(`train:\s*function\s*renderTrain\(\)\s*\{\s*renderTrainView`).MatchString(body) {
		t.Errorf("studio.js train RENDERERS entry must call renderTrainView()")
	}
}

// ── SSE wire tests ─────────────────────────────────────────────────────────

// TestSSETrain_ContentTypeAndHeaders pins the SSE headers and 200 status.
func TestSSETrain_ContentTypeAndHeaders(t *testing.T) {
	withStubTrainRunner(t, stubRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/train?model=mlp&steps=2&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/train: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type %q, want text/event-stream", ct)
	}
	if cc := resp.Header.Get("Cache-Control"); !strings.Contains(cc, "no-cache") {
		t.Errorf("Cache-Control %q, want no-cache", cc)
	}
	// Drain the body so the goroutine cleans up.
	_, _ = io.Copy(io.Discard, resp.Body)
}

// TestSSETrain_SnapshotFrame pins the data: <json>\n\n frame format and
// that the JSON body decodes back to a Snapshot with the expected step.
func TestSSETrain_SnapshotFrame(t *testing.T) {
	withStubTrainRunner(t, stubRunner(3))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/train?model=mlp&steps=3&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/train: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read until done. The framer is the bufio.Scanner with a custom split
	// at blank lines; for simplicity we just slurp + parse.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	text := string(body)

	// Frame must contain "data: " prefix and a blank-line terminator.
	if !strings.Contains(text, "data: ") {
		t.Errorf("body missing 'data: ' prefix; body=%q", text)
	}
	// Pull every data: <line>\n\n.
	re := regexp.MustCompile(`(?m)^data: (.+)$`)
	matches := re.FindAllStringSubmatch(text, -1)
	if len(matches) < 1 {
		t.Fatalf("no data lines found; body=%q", text)
	}
	// First data frame must parse as a Snapshot with step=1.
	var snap tui.Snapshot
	if err := json.Unmarshal([]byte(matches[0][1]), &snap); err != nil {
		t.Fatalf("snapshot JSON: %v; raw=%q", err, matches[0][1])
	}
	if snap.Step != 1 {
		t.Errorf("first snapshot step=%d, want 1", snap.Step)
	}
	if !snap.HasLoss {
		t.Errorf("first snapshot HasLoss=false, want true")
	}
}

// TestSSETrain_DoneEvent pins the `event: done\ndata: {}\n\n` terminator.
func TestSSETrain_DoneEvent(t *testing.T) {
	withStubTrainRunner(t, stubRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/train?model=mlp&steps=2&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/train: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if !strings.Contains(string(body), "event: done") {
		t.Errorf("body missing 'event: done' terminator; body=%q", string(body))
	}
}

// TestSSETrain_ClientCancel pins that closing the response body (proxy
// for the browser tab closing) causes the server to stop cleanly.
//
// Strategy: connect, read one frame, close the body, verify the test
// completes within a short bound. The stub trainer emits a snapshot
// every iteration; the ctx-aware send means the stub returns once we
// close.
func TestSSETrain_ClientCancel(t *testing.T) {
	// Long-running stub: emits up to 1000 snapshots over time, polling ctx.
	withStubTrainRunner(t, func(ctx context.Context, _ string, _ int, snap func(tui.Snapshot)) error {
		for i := 1; i <= 1000; i++ {
			select {
			case <-ctx.Done():
				return nil
			default:
			}
			snap(tui.Snapshot{Step: i, MaxSteps: 1000, Loss: float32(1.0 / float64(i)), HasLoss: true})
			time.Sleep(10 * time.Millisecond)
		}
		return nil
	})
	srv := newWebServer(t)

	done := make(chan struct{})
	var bodyBytes int
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		resp, err := http.Get(srv.URL + "/sse/train?model=mlp&steps=1000&bundle=0")
		if err != nil {
			t.Errorf("GET: %v", err)
			close(done)
			return
		}
		// Read at least one frame.
		br := bufio.NewReader(resp.Body)
		for {
			line, err := br.ReadString('\n')
			if err != nil {
				break
			}
			bodyBytes += len(line)
			if strings.HasPrefix(line, "data: ") {
				break
			}
		}
		// Close mid-stream; the server's ctx should fire.
		_ = resp.Body.Close()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("client read did not complete within 5s after close")
	}
	wg.Wait()
	if bodyBytes == 0 {
		t.Errorf("read zero bytes from SSE body; expected at least one frame")
	}
}

// TestSSETrain_InvalidModel pins the 400 + JSON error body for an unknown
// model.
func TestSSETrain_InvalidModel(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/sse/train?model=not-a-real-model")
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("GET /sse/train?model=junk: status %d, want 400", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	var parsed map[string]string
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		t.Fatalf("body not JSON: %v (body=%q)", err, body)
	}
	if parsed["error"] != "unknown model" {
		t.Errorf("error field %q, want 'unknown model'", parsed["error"])
	}
}

// TestSSETrain_MissingModel pins the 400 for an absent model param.
func TestSSETrain_MissingModel(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/sse/train")
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("GET /sse/train: status %d, want 400", resp.StatusCode)
	}
	var parsed map[string]string
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		t.Fatalf("body not JSON: %v (body=%q)", err, body)
	}
	if parsed["error"] != "missing model" {
		t.Errorf("error field %q, want 'missing model'", parsed["error"])
	}
}

// TestSSETrain_InvalidSteps pins the 400 for an out-of-range steps value.
func TestSSETrain_InvalidSteps(t *testing.T) {
	srv := newWebServer(t)
	for _, q := range []string{"steps=0", "steps=-1", "steps=abc", "steps=99999999"} {
		resp, _ := get(t, srv, "/sse/train?model=mlp&"+q)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("steps query %q: status %d, want 400", q, resp.StatusCode)
		}
	}
}

// TestSSETrain_BundleDefaultIsOn pins that omitting ?bundle= still creates
// a bundle (web tier default = ON, per spec §6). Uses httptest with a
// custom bundle root so the test is isolated.
func TestSSETrain_BundleDefaultIsOn(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	withStubTrainRunner(t, stubRunner(2))
	srv := newWebServer(t)

	// Drive a run with no ?bundle= override.
	resp, err := http.Get(srv.URL + "/sse/train?model=mlp&steps=2")
	if err != nil {
		t.Fatalf("GET /sse/train: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	// Verify via /api/runs that a bundle was created.
	resp2, body2 := get(t, srv, "/api/runs")
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("/api/runs status %d", resp2.StatusCode)
	}
	if strings.TrimSpace(body2) == "[]" {
		t.Errorf("/api/runs is empty; web tier default should have written a bundle")
	}
}

// TestSSETrain_BundleZeroSkips pins ?bundle=0 disables the bundle tee.
func TestSSETrain_BundleZeroSkips(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	withStubTrainRunner(t, stubRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/train?model=mlp&steps=2&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/train: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	resp2, body2 := get(t, srv, "/api/runs")
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("/api/runs status %d", resp2.StatusCode)
	}
	if strings.TrimSpace(body2) != "[]" {
		t.Errorf("/api/runs should be empty with bundle=0; got %q", body2)
	}
}
