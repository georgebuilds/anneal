//go:build !js

// Tests for W7 — the generate view markup + /sse/generate wire contract.
//
// Markup tests (TestWebGenerate_*) pin the studio.html surface (controls,
// token stream live region, click-through, prompt input, compare toggle,
// warming hint, deep link).
//
// SSE wire tests (TestSSEGenerate_*) drive the handler via httptest with a
// stub generateRunner so the contract (Content-Type, frame shape, done
// terminator payload, cancel handling, invalid-model + missing-prompt +
// tokens-cap rejections, bundle default, compare toggle) is pinned in
// pure Go without a real gpt2.Sample invocation.

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

// withStubGenerateRunner swaps generateRunnerFn for the duration of the
// test. The stub emits n token snapshots then returns.
func withStubGenerateRunner(t *testing.T, runner generateRunner) {
	t.Helper()
	orig := generateRunnerFn
	generateRunnerFn = runner
	t.Cleanup(func() { generateRunnerFn = orig })
}

// stubGenerateRunner returns a generateRunner that pushes n token
// snapshots and returns. When compare is true each token carries a
// RefMatch=true so the studio's ref-match indicator can be tested.
func stubGenerateRunner(n int) generateRunner {
	return func(ctx context.Context, model, prompt string, maxTokens int, compare bool, emit func(tui.TokenSnapshot)) error {
		// Initial PhaseInit frame so the warming hint can hide.
		emit(tui.TokenSnapshot{Phase: tui.PhaseInit, MaxTokens: maxTokens, Step: 0})
		max := maxTokens
		if max <= 0 {
			max = n
		}
		for i := 0; i < n; i++ {
			select {
			case <-ctx.Done():
				return nil
			default:
			}
			var refMatch *bool
			if compare {
				yes := true
				refMatch = &yes
			}
			emit(tui.TokenSnapshot{
				Step:         i,
				MaxTokens:    max,
				TokenID:      100 + i,
				TokenText:    " tok" + string(rune('a'+i%26)),
				LogitArgmax:  100 + i,
				LogitSummary: "max=1.23 at idx 100",
				WallMs:       int64(i * 10),
				Phase:        tui.PhaseTraining,
				RefMatch:     refMatch,
				LastKernelID: "k_demo",
			})
		}
		emit(tui.TokenSnapshot{Step: n, MaxTokens: max, Phase: tui.PhaseDone, WallMs: int64(n * 10)})
		return nil
	}
}

// ── markup tests ──────────────────────────────────────────────────────────

// TestWebGenerate_StubReplaced pins the W7 markup is wired (not the W0/W1 stub).
func TestWebGenerate_StubReplaced(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	start := strings.Index(body, `id="view-generate"`)
	if start < 0 {
		t.Fatal("studio.html missing #view-generate section")
	}
	end := strings.Index(body[start:], "</section>")
	section := body[start : start+end]
	if strings.Contains(section, "view: generate coming soon") {
		t.Errorf("generate view still contains W0 stub copy")
	}
	for _, want := range []string{
		`class="generate-pane"`,
		`id="gen-model"`,
		`id="gen-prompt"`,
		`id="gen-tokens"`,
		`id="gen-start"`,
		`id="gen-cancel"`,
		`id="gen-compare"`,
		`id="gen-bundle"`,
		`id="gen-tokens-out"`,
		`id="gen-prompt-echo"`,
		`id="gen-last-text"`,
		`id="gen-last-id"`,
		`id="gen-last-logit"`,
		`id="gen-last-ref"`,
		`id="gen-click-through"`,
		`id="gen-warming"`,
		`id="gen-status"`,
	} {
		if !strings.Contains(section, want) {
			t.Errorf("generate view section missing %q", want)
		}
	}
}

// TestWebGenerate_ControlsLabeled pins every form control has a label or
// aria-label.
func TestWebGenerate_ControlsLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	for _, want := range []string{
		`for="gen-model"`,
		`for="gen-prompt"`,
		`for="gen-tokens"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("generate view missing %q", want)
		}
	}
	// Every <button id="gen-*"> must have an aria-label.
	btnRE := regexp.MustCompile(`<button[^>]*id="(gen-[a-z-]+)"[^>]*>`)
	for _, m := range btnRE.FindAllStringSubmatch(body, -1) {
		tag := m[0]
		id := m[1]
		if !strings.Contains(tag, "aria-label=") {
			t.Errorf("button #%s missing aria-label: %s", id, tag)
		}
	}
}

// TestWebGenerate_PromptInputLabeled pins the prompt input has maxlength
// and aria-describedby (the token-cap rule).
func TestWebGenerate_PromptInputLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="gen-prompt"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("generate view missing #gen-prompt")
	}
	if !strings.Contains(tag, `maxlength="2048"`) {
		t.Errorf("#gen-prompt missing maxlength=\"2048\": %s", tag)
	}
	if !strings.Contains(tag, `aria-describedby=`) {
		t.Errorf("#gen-prompt missing aria-describedby: %s", tag)
	}
	if !strings.Contains(tag, `aria-label=`) {
		t.Errorf("#gen-prompt missing aria-label: %s", tag)
	}
}

// TestWebGenerate_TokenStreamAriaLive pins the token stream is a polite
// live region (so tokens are announced incrementally).
func TestWebGenerate_TokenStreamAriaLive(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<div[^>]*class="token-stream"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("generate view missing .token-stream container")
	}
	if !strings.Contains(tag, `aria-live="polite"`) {
		t.Errorf("token-stream missing aria-live=\"polite\": %s", tag)
	}
	if !strings.Contains(tag, `aria-atomic="false"`) {
		t.Errorf("token-stream missing aria-atomic=\"false\": %s", tag)
	}
	if !strings.Contains(tag, `aria-label=`) {
		t.Errorf("token-stream missing aria-label: %s", tag)
	}
}

// TestWebGenerate_LogitPanelHeading pins the last-token aside has an h2
// heading and a region role for SR jump navigation.
func TestWebGenerate_LogitPanelHeading(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<aside[^>]*class="logit-panel"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("generate view missing .logit-panel aside")
	}
	if !strings.Contains(tag, `aria-label=`) {
		t.Errorf("logit-panel missing aria-label: %s", tag)
	}
	if !strings.Contains(body, `<h2>last token</h2>`) {
		t.Errorf("logit-panel missing <h2>last token</h2>")
	}
}

// TestWebGenerate_CompareCheckboxLabeled pins the compare-to-reference
// checkbox is labelled.
func TestWebGenerate_CompareCheckboxLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="gen-compare"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("generate view missing #gen-compare")
	}
	if !strings.Contains(tag, `type="checkbox"`) {
		t.Errorf("#gen-compare is not a checkbox: %s", tag)
	}
	if !strings.Contains(tag, `aria-label=`) {
		t.Errorf("#gen-compare missing aria-label: %s", tag)
	}
	if !strings.Contains(body, "compare to reference") {
		t.Errorf("generate view missing literal text 'compare to reference'")
	}
}

// TestWebGenerate_BundleDefaultOnInMarkup pins #gen-bundle defaults to
// checked (web tier default = ON, per spec §6).
func TestWebGenerate_BundleDefaultOnInMarkup(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="gen-bundle"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("generate view missing #gen-bundle")
	}
	if !strings.Contains(tag, `checked`) {
		t.Errorf("#gen-bundle must default to checked (web tier default = ON, spec §6): %s", tag)
	}
}

// TestWebGenerate_DeepLinkURL pins /g/<model> resolves to the studio shell.
func TestWebGenerate_DeepLinkURL(t *testing.T) {
	srv := newWebServer(t)
	for _, p := range []string{"/g/gpt2", "/g/nanogpt", "/g/gpt2?prompt=Hello"} {
		resp, body := get(t, srv, p)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s: status %d, want 200", p, resp.StatusCode)
		}
		if !strings.Contains(body, `id="brand-mark"`) {
			t.Errorf("GET %s: not the studio shell", p)
		}
	}
}

// TestWebGenerate_RendererWired pins studio.js routes /g/<model> to a real
// renderer.
func TestWebGenerate_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "renderGenerateView") {
		t.Errorf("studio.js missing renderGenerateView (W7 renderer)")
	}
	if !regexp.MustCompile(`generate:\s*function\s*renderGenerate\b`).MatchString(body) {
		t.Errorf("studio.js generate renderer not wired into RENDERERS")
	}
	if !regexp.MustCompile(`generate:\s*function\s*renderGenerate\(\)\s*\{\s*renderGenerateView`).MatchString(body) {
		t.Errorf("studio.js generate RENDERERS entry must call renderGenerateView()")
	}
}

// TestWebGenerate_ClickThroughWired pins each token gets a click handler
// that opens /k/<model>?kernel=... (the producing-kernel click-through
// per spec §5.6).
func TestWebGenerate_ClickThroughWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "openKernelForToken") {
		t.Errorf("studio.js missing openKernelForToken (W7 click-through)")
	}
	if !strings.Contains(body, "'/k/'") && !strings.Contains(body, `"/k/"`) {
		t.Errorf("studio.js openKernelForToken must navigate to /k/<model>")
	}
	if !strings.Contains(body, "?kernel=") {
		t.Errorf("studio.js openKernelForToken must include ?kernel= query")
	}
	// The token span must be created with tabindex="0" so it is keyboard-
	// focusable (a11y).
	if !regexp.MustCompile(`setAttribute\(['"]tabindex['"],\s*['"]0['"]\)`).MatchString(body) {
		t.Errorf("studio.js .tok spans must set tabindex=\"0\"")
	}
}

// TestWebGenerate_WarmingHintPresent pins the warming-up hint is in the
// markup hidden by default (the JS shows it on start and hides on first
// frame).
func TestWebGenerate_WarmingHintPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<span[^>]*id="gen-warming"[^>]*>`)
	tag := re.FindString(body)
	if tag == "" {
		t.Fatal("generate view missing #gen-warming hint")
	}
	if !strings.Contains(tag, `hidden`) {
		t.Errorf("#gen-warming must start hidden: %s", tag)
	}
	if !strings.Contains(body, "warming up") {
		t.Errorf("warming hint must contain literal text 'warming up'")
	}
}

// ── SSE wire tests ─────────────────────────────────────────────────────────

// TestSSEGenerate_ContentTypeAndHeaders pins the SSE headers and 200 status.
func TestSSEGenerate_ContentTypeAndHeaders(t *testing.T) {
	withStubGenerateRunner(t, stubGenerateRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=hi&tokens=2&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/generate: %v", err)
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
	_, _ = io.Copy(io.Discard, resp.Body)
}

// TestSSEGenerate_TokenFrame pins the data: <json>\n\n frame shape: the
// body JSON decodes into a TokenSnapshot with the expected step + token.
func TestSSEGenerate_TokenFrame(t *testing.T) {
	withStubGenerateRunner(t, stubGenerateRunner(3))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=hello&tokens=3&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/generate: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	text := string(body)
	if !strings.Contains(text, "data: ") {
		t.Errorf("body missing 'data: ' prefix; body=%q", text)
	}
	re := regexp.MustCompile(`(?m)^data: (.+)$`)
	matches := re.FindAllStringSubmatch(text, -1)
	if len(matches) < 2 {
		t.Fatalf("expected at least 2 data lines (init + token); got %d. body=%q", len(matches), text)
	}
	// First frame is the PhaseInit warming-complete signal.
	var init tui.TokenSnapshot
	if err := json.Unmarshal([]byte(matches[0][1]), &init); err != nil {
		t.Fatalf("init JSON: %v; raw=%q", err, matches[0][1])
	}
	if init.Phase != tui.PhaseInit {
		t.Errorf("first frame phase=%v, want init", init.Phase)
	}
	// Second frame should be the first generated token.
	var tok tui.TokenSnapshot
	if err := json.Unmarshal([]byte(matches[1][1]), &tok); err != nil {
		t.Fatalf("token JSON: %v; raw=%q", err, matches[1][1])
	}
	if tok.Step != 0 {
		t.Errorf("first token step=%d, want 0", tok.Step)
	}
	if tok.TokenText == "" {
		t.Errorf("first token TokenText empty; want a fragment")
	}
}

// TestSSEGenerate_DoneEvent pins the `event: done` terminator with a
// payload carrying total_tokens and wall_ms.
func TestSSEGenerate_DoneEvent(t *testing.T) {
	withStubGenerateRunner(t, stubGenerateRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=p&tokens=2&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/generate: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if !strings.Contains(string(body), "event: done") {
		t.Errorf("body missing 'event: done' terminator; body=%q", string(body))
	}
	// Find the done event payload.
	re := regexp.MustCompile(`event: done\ndata: (.+)`)
	m := re.FindStringSubmatch(string(body))
	if m == nil {
		t.Fatalf("could not parse done event payload from body=%q", string(body))
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(m[1]), &payload); err != nil {
		t.Fatalf("done payload not JSON: %v; raw=%q", err, m[1])
	}
	if _, ok := payload["total_tokens"]; !ok {
		t.Errorf("done payload missing total_tokens; got %v", payload)
	}
	if _, ok := payload["wall_ms"]; !ok {
		t.Errorf("done payload missing wall_ms; got %v", payload)
	}
}

// TestSSEGenerate_ClientCancel pins that closing the body stops the
// server cleanly.
func TestSSEGenerate_ClientCancel(t *testing.T) {
	withStubGenerateRunner(t, func(ctx context.Context, _ string, _ string, _ int, _ bool, emit func(tui.TokenSnapshot)) error {
		for i := 0; i < 1000; i++ {
			select {
			case <-ctx.Done():
				return nil
			default:
			}
			emit(tui.TokenSnapshot{
				Step: i, MaxTokens: 1000, TokenText: "x", Phase: tui.PhaseTraining,
			})
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
		resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=p&tokens=1000&bundle=0")
		if err != nil {
			t.Errorf("GET: %v", err)
			close(done)
			return
		}
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

// TestSSEGenerate_InvalidModel pins the 400 + JSON error body for an
// unknown model.
func TestSSEGenerate_InvalidModel(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/sse/generate?model=not-a-real-model&prompt=p")
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status %d, want 400", resp.StatusCode)
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

// TestSSEGenerate_MissingPrompt pins the 400 for an empty prompt param.
func TestSSEGenerate_MissingPrompt(t *testing.T) {
	srv := newWebServer(t)
	for _, p := range []string{"/sse/generate?model=gpt2", "/sse/generate?model=gpt2&prompt="} {
		resp, body := get(t, srv, p)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("GET %s: status %d, want 400", p, resp.StatusCode)
		}
		var parsed map[string]string
		if err := json.Unmarshal([]byte(body), &parsed); err != nil {
			t.Errorf("GET %s: body not JSON: %v", p, err)
			continue
		}
		if parsed["error"] != "missing prompt" {
			t.Errorf("GET %s: error field %q, want 'missing prompt'", p, parsed["error"])
		}
	}
}

// TestSSEGenerate_TokensCapEnforced pins that tokens > 256 returns 400.
func TestSSEGenerate_TokensCapEnforced(t *testing.T) {
	srv := newWebServer(t)
	for _, q := range []string{"tokens=0", "tokens=-1", "tokens=abc", "tokens=257", "tokens=99999"} {
		resp, _ := get(t, srv, "/sse/generate?model=gpt2&prompt=p&"+q)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("tokens query %q: status %d, want 400", q, resp.StatusCode)
		}
	}
}

// TestSSEGenerate_BundleDefaultIsOn pins that omitting ?bundle= still
// creates a bundle (web tier default = ON, per spec §6).
func TestSSEGenerate_BundleDefaultIsOn(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	withStubGenerateRunner(t, stubGenerateRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=hi&tokens=2")
	if err != nil {
		t.Fatalf("GET /sse/generate: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	resp2, body2 := get(t, srv, "/api/runs")
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("/api/runs status %d", resp2.StatusCode)
	}
	if strings.TrimSpace(body2) == "[]" {
		t.Errorf("/api/runs is empty; web tier default should have written a bundle")
	}
}

// TestSSEGenerate_BundleZeroSkips pins ?bundle=0 disables the bundle tee.
func TestSSEGenerate_BundleZeroSkips(t *testing.T) {
	t.Setenv("ANNEAL_RUN_DIR", t.TempDir())
	withStubGenerateRunner(t, stubGenerateRunner(2))
	srv := newWebServer(t)

	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=hi&tokens=2&bundle=0")
	if err != nil {
		t.Fatalf("GET /sse/generate: %v", err)
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

// TestSSEGenerate_CompareToggle pins that when ?compare=1, every token
// frame carries a ref_match field (true/false) and that without compare
// the field is omitted.
func TestSSEGenerate_CompareToggle(t *testing.T) {
	withStubGenerateRunner(t, stubGenerateRunner(2))
	srv := newWebServer(t)

	// With compare=1.
	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=p&tokens=2&compare=1&bundle=0")
	if err != nil {
		t.Fatalf("GET (compare=1): %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if !strings.Contains(string(body), `"ref_match":true`) && !strings.Contains(string(body), `"ref_match":false`) {
		t.Errorf("compare=1 body missing ref_match field; body=%q", string(body))
	}

	// Without compare.
	withStubGenerateRunner(t, stubGenerateRunner(2))
	resp2, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=p&tokens=2&bundle=0")
	if err != nil {
		t.Fatalf("GET (no compare): %v", err)
	}
	body2, _ := io.ReadAll(resp2.Body)
	_ = resp2.Body.Close()
	if strings.Contains(string(body2), `"ref_match"`) {
		t.Errorf("no-compare body should omit ref_match; body=%q", string(body2))
	}
}

// TestSSEGenerate_WarmingFrameOnFirstFrame pins the first SSE frame has
// phase="init" — the contract the studio reads to hide the warming hint.
// This documents the wire contract that the production runner emits a
// PhaseInit frame before any token; stubs in tests follow the same
// convention.
func TestSSEGenerate_WarmingFrameOnFirstFrame(t *testing.T) {
	withStubGenerateRunner(t, stubGenerateRunner(1))
	srv := newWebServer(t)
	resp, err := http.Get(srv.URL + "/sse/generate?model=gpt2&prompt=p&tokens=1&bundle=0")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	re := regexp.MustCompile(`(?m)^data: (.+)$`)
	matches := re.FindAllStringSubmatch(string(body), -1)
	if len(matches) < 1 {
		t.Fatalf("no data lines; body=%q", string(body))
	}
	var first tui.TokenSnapshot
	if err := json.Unmarshal([]byte(matches[0][1]), &first); err != nil {
		t.Fatalf("first frame JSON: %v", err)
	}
	if first.Phase != tui.PhaseInit {
		t.Errorf("first frame phase=%v, want init (warming-complete signal)", first.Phase)
	}
}
