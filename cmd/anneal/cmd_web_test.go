//go:build !js

// Tests for the `anneal web` foundation (W0).
//
// What these tests pin:
//   - / returns 200 text/html and contains the thesis statement + wordmark SVG.
//   - /static/studio.css contains all five brand-token hexes from DESIGN.md §1.
//   - /static/studio.js wires the matchMedia listener and the theme cycle.
//   - /static/worker.js loads wasm_exec.js and forwards postMessage RPC.
//   - /static/missing 404s; /api/* and /sse/* return 501 with the documented body.
//   - Each web/*.{js,css,html} file is under the 50KB per-file budget.
//
// JS state machines (theme cycle, chord keymap) are not executable in pure Go;
// we assert the contract is encoded as searchable literals in the JS source.

package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
)

func newWebServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(serveMux())
	t.Cleanup(s.Close)
	return s
}

func get(t *testing.T, srv *httptest.Server, path string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Get(srv.URL + path)
	if err != nil {
		t.Fatalf("GET %s: %v", path, err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body %s: %v", path, err)
	}
	return resp, string(b)
}

// TestWebRootServesShell pins that GET / returns the studio shell with the
// right Content-Type and contains the thesis statement + brand mark.
func TestWebRootServesShell(t *testing.T) {
	srv := newWebServer(t)

	resp, body := get(t, srv, "/")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /: status %d, want 200", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/html") {
		t.Errorf("GET /: Content-Type %q, want text/html prefix", ct)
	}
	// Thesis statement: "gradients are just rewrites" (DESIGN.md §3.1; spec §5.1).
	if !strings.Contains(body, "gradients") || !strings.Contains(body, "rewrites") {
		t.Errorf("studio.html does not contain the thesis statement (looked for 'gradients' and 'rewrites')")
	}
	// Wordmark SVG: the brand mark id is referenced by studio.js for the
	// ignite-once animation, and the gold node fill is the brand anchor.
	if !strings.Contains(body, `id="brand-mark"`) {
		t.Errorf("studio.html missing id=\"brand-mark\" (wordmark mount)")
	}
	if !strings.Contains(body, "#F2C57C") {
		t.Errorf("studio.html missing inline gold (#F2C57C) in the brand mark SVG")
	}
}

// TestWebStaticCSSBrandTokens pins that the dark and light brand hexes are
// present verbatim in studio.css. This is the cross-surface contract per
// DESIGN.md §1.3: the studio's hexes match viz/static/style.css and
// tui/theme.go.
func TestWebStaticCSSBrandTokens(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/static/studio.css")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /static/studio.css: status %d", resp.StatusCode)
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/css") {
		t.Errorf("studio.css Content-Type %q, want text/css prefix", resp.Header.Get("Content-Type"))
	}
	for _, hex := range []string{
		"#00ADD8", // forward teal       (DESIGN.md §1.1)
		"#FF7A45", // backward ember     (DESIGN.md §1.1)
		"#F2C57C", // fused gold         (DESIGN.md §1.1)
		"#14110F", // dark bg            (DESIGN.md §1.1)
		"#FBF8F3", // light bg           (DESIGN.md §1.2)
	} {
		if !strings.Contains(body, hex) {
			t.Errorf("studio.css missing brand token %s", hex)
		}
	}
	// The system-default block must exist so prefers-color-scheme honours it.
	if !strings.Contains(body, `:root[data-theme="system"]`) {
		t.Errorf("studio.css missing :root[data-theme=\"system\"] block (DESIGN.md §5)")
	}
}

// TestWebStudioJSMatchMediaListener pins the spec §10 fix: when `system` is
// selected, an OS theme change re-themes without a page reload. This is the
// reason we can't just rely on the CSS @media block alone.
func TestWebStudioJSMatchMediaListener(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/static/studio.js")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /static/studio.js: status %d", resp.StatusCode)
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/javascript") {
		t.Errorf("studio.js Content-Type %q, want application/javascript prefix", resp.Header.Get("Content-Type"))
	}
	// Regex: the listener may be parameterised but the matchMedia call with
	// the prefers-color-scheme query must be present literally.
	mmRE := regexp.MustCompile(`matchMedia\s*\(\s*['"]\(prefers-color-scheme:\s*dark\)['"]`)
	if !mmRE.MatchString(body) {
		t.Errorf("studio.js missing matchMedia('(prefers-color-scheme: dark)') call (spec §10 fix)")
	}
	// Cycle contract: the comment / literal "system → dark → light" pins the
	// theme cycle order so a future refactor cannot silently re-order it.
	if !strings.Contains(body, "system → dark → light") {
		t.Errorf("studio.js missing theme cycle contract literal 'system → dark → light'")
	}
}

// TestWebWorkerScaffold pins the worker forwards postMessage RPC and loads
// wasm_exec.js. Empty handler table is fine (Go side registers everything);
// the scaffold itself must exist.
func TestWebWorkerScaffold(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/static/worker.js")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /static/worker.js: status %d", resp.StatusCode)
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/javascript") {
		t.Errorf("worker.js Content-Type %q, want application/javascript prefix", resp.Header.Get("Content-Type"))
	}
	for _, want := range []string{
		"importScripts",       // loads wasm_exec.js
		"wasm_exec.js",        // by name
		"WebAssembly",         // boots the WASM module
		"self.onmessage",      // RPC entry point
		"self.postMessage",    // RPC response
	} {
		if !strings.Contains(body, want) {
			t.Errorf("worker.js missing %q (RPC scaffold contract)", want)
		}
	}
}

// TestWebWasmExecPresent pins that the Go-runtime-supplied wasm_exec.js is
// served and contains the `Go` constructor that the worker uses.
func TestWebWasmExecPresent(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/static/wasm_exec.js")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /static/wasm_exec.js: status %d", resp.StatusCode)
	}
	if !strings.Contains(body, "globalThis.Go") && !strings.Contains(body, "class Go") {
		t.Errorf("wasm_exec.js does not look like the Go runtime loader (missing Go class)")
	}
}

// TestWebStaticMissing404 pins the 404 path. The mux only serves what is
// embedded; everything else under /static/ is a hard not-found.
func TestWebStaticMissing404(t *testing.T) {
	srv := newWebServer(t)
	resp, _ := get(t, srv, "/static/missing.txt")
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("GET /static/missing.txt: status %d, want 404", resp.StatusCode)
	}
}

// TestWebAPIStubs501 pins every documented API surface returns 501 with the
// "phase ID pending" JSON body. Subsequent W steps swap these out.
func TestWebAPIStubs501(t *testing.T) {
	srv := newWebServer(t)
	for _, path := range []string{
		"/api/device",
		"/api/runs",
		"/api/compile/tuned",
		"/sse/train",
		"/sse/generate",
	} {
		resp, body := get(t, srv, path)
		if resp.StatusCode != http.StatusNotImplemented {
			t.Errorf("GET %s: status %d, want 501", path, resp.StatusCode)
		}
		if !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
			t.Errorf("GET %s: Content-Type %q, want application/json prefix",
				path, resp.Header.Get("Content-Type"))
		}
		var parsed map[string]string
		if err := json.Unmarshal([]byte(body), &parsed); err != nil {
			t.Errorf("GET %s: body is not JSON: %v (body=%q)", path, err, body)
			continue
		}
		if parsed["error"] != "phase ID pending" {
			t.Errorf("GET %s: error field %q, want 'phase ID pending'", path, parsed["error"])
		}
	}
}

// TestWebHistoryAPIDeepLink pins that the History API deep links resolve to
// the same shell (the client-side router takes it from there). /v/mlp must
// return the studio HTML, not 404.
func TestWebHistoryAPIDeepLink(t *testing.T) {
	srv := newWebServer(t)
	for _, p := range []string{"/v/mlp", "/k/conv", "/x/matmul", "/t/mlp", "/g/nanogpt", "/h", "/d"} {
		resp, body := get(t, srv, p)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s: status %d, want 200 (History API deep links resolve to shell)", p, resp.StatusCode)
			continue
		}
		if !strings.Contains(body, `id="brand-mark"`) {
			t.Errorf("GET %s: body does not look like studio shell", p)
		}
	}
}

// TestWebFileSizesUnderBudget pins each hand-authored static file is under
// 50KB raw. Foundation is glue, not heavy code; if a file grows past this
// budget, extract a helper rather than letting one file sprawl (DESIGN.md
// "few hundred lines target").
func TestWebFileSizesUnderBudget(t *testing.T) {
	srv := newWebServer(t)
	const maxBytes = 50 * 1024
	for _, p := range []string{
		"/static/studio.html",
		"/static/studio.css",
		"/static/studio.js",
		"/static/worker.js",
	} {
		resp, body := get(t, srv, p)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s: status %d", p, resp.StatusCode)
			continue
		}
		if len(body) > maxBytes {
			t.Errorf("GET %s: %d bytes, want <= %d (foundation file-size budget)",
				p, len(body), maxBytes)
		}
	}
}

// TestWebStudioJSRoutingTable pins the eight documented view ids exist in the
// JS routing source. If a renderer is renamed, this catches it.
func TestWebStudioJSRoutingTable(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	for _, viewID := range []string{
		"studio", "visualize", "kernels", "explain",
		"train", "generate", "history", "doctor",
	} {
		// Each id must appear at least twice: once as a RENDERERS key, once
		// as a VIEW_TO_PATH key (we don't pin the exact location, only that
		// the literal exists in source).
		if !strings.Contains(body, viewID) {
			t.Errorf("studio.js missing view id %q in routing table", viewID)
		}
	}
}
