//go:build !js

// Tests for the W8 ONNX dropzone. Source-text checks pin the studio.html
// surface (dropzone region, result section, action links, privacy claim) and
// the studio.js wiring (initStudioDropzone, annealImportONNX call). The
// privacy contract (spec §1.3 / §8) is asserted by the absence of any
// /api/onnx/* server endpoint - TestWebOnnx_NoServerEndpoint enumerates the
// canonical paths and proves the server has no handler for them.

package main

import (
	"regexp"
	"strings"
	"testing"
)

// TestWebOnnx_DropzonePresent pins the ONNX dropzone markup is in the studio
// view (not stubbed, not in another view).
func TestWebOnnx_DropzonePresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `id="onnx-dropzone"`) {
		t.Errorf(`studio.html missing id="onnx-dropzone"`)
	}
	if !strings.Contains(body, `id="onnx-result"`) {
		t.Errorf(`studio.html missing id="onnx-result" (dropzone result region)`)
	}
	// The dropzone must live inside the studio view, not somewhere else.
	studioStart := strings.Index(body, `id="view-studio"`)
	studioEnd := strings.Index(body[studioStart:], "</section>")
	if studioStart < 0 || studioEnd < 0 {
		t.Fatalf("view-studio section not found")
	}
	studioSection := body[studioStart : studioStart+studioEnd]
	if !strings.Contains(studioSection, `id="onnx-dropzone"`) {
		t.Errorf("onnx-dropzone is not inside view-studio")
	}
}

// TestWebOnnx_DropzoneRegionAria pins role=region + aria-label +
// aria-describedby on the dropzone (per A11Y.md §3f).
func TestWebOnnx_DropzoneRegionAria(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<div[^>]*id="onnx-dropzone"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal(`studio.html missing <div id="onnx-dropzone">`)
	}
	if !strings.Contains(m, `role="region"`) {
		t.Errorf(`onnx-dropzone missing role="region": %s`, m)
	}
	if !strings.Contains(m, `aria-label="ONNX model dropzone"`) {
		t.Errorf(`onnx-dropzone missing aria-label="ONNX model dropzone": %s`, m)
	}
	if !strings.Contains(m, `aria-describedby="dropzone-hint"`) {
		t.Errorf(`onnx-dropzone missing aria-describedby: %s`, m)
	}
}

// TestWebOnnx_DropzoneKeyboardAccessible pins tabindex="0" on the dropzone
// so keyboard users can reach it (drag/drop has a keyboard-equivalent path
// via the pick-file button).
func TestWebOnnx_DropzoneKeyboardAccessible(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<div[^>]*id="onnx-dropzone"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal(`studio.html missing <div id="onnx-dropzone">`)
	}
	if !strings.Contains(m, `tabindex="0"`) {
		t.Errorf(`onnx-dropzone missing tabindex="0": %s`, m)
	}
}

// TestWebOnnx_FilePickerLabeled pins the file <input> has an aria-label so
// screen-reader users hear a description.
func TestWebOnnx_FilePickerLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="onnx-picker"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal(`studio.html missing <input id="onnx-picker">`)
	}
	if !strings.Contains(m, `type="file"`) {
		t.Errorf(`onnx-picker not type="file": %s`, m)
	}
	if !strings.Contains(m, `accept=".onnx"`) {
		t.Errorf(`onnx-picker missing accept=".onnx": %s`, m)
	}
	if !regexp.MustCompile(`aria-label="[^"]+"`).MatchString(m) {
		t.Errorf(`onnx-picker missing aria-label: %s`, m)
	}

	// The pick button is the keyboard-equivalent drag/drop affordance.
	btnRe := regexp.MustCompile(`<button[^>]*id="onnx-picker-btn"[^>]*>`)
	btn := btnRe.FindString(body)
	if btn == "" {
		t.Fatal(`studio.html missing <button id="onnx-picker-btn"> (keyboard alt for drag/drop)`)
	}
	if !regexp.MustCompile(`aria-label="[^"]+"`).MatchString(btn) {
		t.Errorf(`onnx-picker-btn missing aria-label: %s`, btn)
	}
}

// TestWebOnnx_PrivacyClaimInHint pins the privacy claim - the dropzone hint
// must say "never leave the tab" so users see the WASM-tier contract before
// they drop a file.
func TestWebOnnx_PrivacyClaimInHint(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	hintIdx := strings.Index(body, `id="dropzone-hint"`)
	if hintIdx < 0 {
		t.Fatal(`studio.html missing id="dropzone-hint"`)
	}
	// Look for the phrase "never leave the tab" anywhere inside the hint.
	tail := body[hintIdx : hintIdx+200]
	if !strings.Contains(tail, "never leave the tab") {
		t.Errorf(`dropzone-hint does not contain privacy claim "never leave the tab": %q`, tail)
	}
}

// TestWebOnnx_ResultLiveRegion pins the result region is aria-live="polite"
// + aria-atomic="false" so screen readers announce the summary incrementally
// without stealing focus.
func TestWebOnnx_ResultLiveRegion(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<div[^>]*id="onnx-result"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal(`studio.html missing <div id="onnx-result">`)
	}
	if !strings.Contains(m, `aria-live="polite"`) {
		t.Errorf(`onnx-result missing aria-live="polite": %s`, m)
	}
	if !strings.Contains(m, `aria-atomic="false"`) {
		t.Errorf(`onnx-result missing aria-atomic="false": %s`, m)
	}
	if !strings.Contains(m, "hidden") {
		t.Errorf(`onnx-result should start hidden: %s`, m)
	}
}

// TestWebOnnx_VisualizeLinkPresent pins the visualize action link is in the
// dropzone result section so users can jump from the import summary to the
// visualize view.
func TestWebOnnx_VisualizeLinkPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `id="onnx-visualize-link"`) {
		t.Errorf(`studio.html missing id="onnx-visualize-link"`)
	}
}

// TestWebOnnx_KernelsLinkPresent pins the kernels action link.
func TestWebOnnx_KernelsLinkPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `id="onnx-kernels-link"`) {
		t.Errorf(`studio.html missing id="onnx-kernels-link"`)
	}
}

// TestWebOnnx_ExplainLinkPresent pins the explain action link.
func TestWebOnnx_ExplainLinkPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `id="onnx-explain-link"`) {
		t.Errorf(`studio.html missing id="onnx-explain-link"`)
	}
}

// TestWebOnnx_RendererWired pins studio.js wires the dropzone init. We scan
// the source for the renderer function name so a refactor cannot silently
// drop it.
func TestWebOnnx_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "initStudioDropzone") {
		t.Fatal(`studio.js missing initStudioDropzone (W8 init)`)
	}
	if !strings.Contains(body, "annealImportONNX") {
		t.Fatal(`studio.js does not dispatch annealImportONNX to WASM bridge`)
	}
	// The renderer must also stash imported summaries in sessionStorage
	// (the deep-link contract per spec §5.1).
	if !strings.Contains(body, "anneal-imported-") {
		t.Fatal(`studio.js does not key sessionStorage by anneal-imported-<gid>`)
	}
}

// TestWebOnnx_NoServerEndpoint asserts the privacy contract: there is NO
// /api/onnx/* server endpoint. The import is WASM-tier; model bytes never
// reach even the local server. This is spec §1.3 / §8 in test form - any
// future PR that adds a server-side ONNX path must justify breaking the
// dropzone's load-bearing privacy claim.
func TestWebOnnx_NoServerEndpoint(t *testing.T) {
	srv := newWebServer(t)
	for _, path := range []string{
		"/api/onnx",
		"/api/onnx/",
		"/api/onnx/import",
		"/api/onnx/visualize",
		"/sse/onnx",
	} {
		resp, body := get(t, srv, path)
		// The studio is an SPA: unknown / non-/static paths fall through to
		// the SPA shell. What we MUST avoid is a real handler that returns
		// 200 with anything resembling an import-summary JSON.
		ct := resp.Header.Get("Content-Type")
		if strings.HasPrefix(ct, "application/json") {
			t.Errorf("path %s returned application/json (looks like an ONNX endpoint): %s",
				path, ct)
		}
		// Spot-check the body does not contain canonical import-summary keys.
		for _, key := range []string{`"graph_id"`, `"node_count"`, `"initializer_count"`, `"unsupported_ops"`} {
			if strings.Contains(body, key) {
				t.Errorf("path %s body contains import-summary key %s (server should not handle ONNX)",
					path, key)
			}
		}
		_ = resp.StatusCode // status doesn't matter as long as no ONNX JSON leaks
	}
}

// TestWebOnnx_UnsupportedSectionPresent pins the unsupported-ops section
// markup so the studio's renderer has a real DOM target to unhide on miss.
func TestWebOnnx_UnsupportedSectionPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `id="onnx-unsupported-section"`) {
		t.Errorf(`studio.html missing id="onnx-unsupported-section"`)
	}
	if !strings.Contains(body, `id="onnx-unsupported-list"`) {
		t.Errorf(`studio.html missing id="onnx-unsupported-list"`)
	}
}

// TestWebOnnx_WASMExportName ensures the WASM bridge name is the one studio.js
// dispatches against. Pinned at the source-text level (the actual wasm
// artifact may or may not be present; this is a contract test).
func TestWebOnnx_WASMExportName(t *testing.T) {
	// Look for the JS reference; the Go export lives in viz/wasm/main.go and
	// is built separately (see web/embed.go for the //go:embed note).
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, `'annealImportONNX'`) && !strings.Contains(body, `"annealImportONNX"`) {
		t.Errorf("studio.js does not call wasm.call('annealImportONNX', ...)")
	}
}

// (Studio.html size budget is enforced globally by
// TestWebFileSizesUnderBudget; no duplicate gate here.)
