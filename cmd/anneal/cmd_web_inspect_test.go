//go:build !js

// Tests for the W9 tensor-inspect dropzone (HTML/CSS/JS contract). The
// dropzone is WASM-tier: bytes never touch the server, and there is no
// /api/inspect endpoint. These tests assert the contract is encoded as
// searchable literals in the studio assets. The pure-Go inspector tests
// live in viz/inspect_test.go.
//
// Spec: notes/anneal_web_spec.md §5.1.

package main

import (
	"net/http"
	"regexp"
	"strings"
	"testing"
)

// TestWebInspect_DropzonePresent pins that the studio home view (NOT a
// separate view) carries the tensor-inspect dropzone. Per spec §5.1 the
// dropzone lives on the studio home.
func TestWebInspect_DropzonePresent(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /: status %d", resp.StatusCode)
	}
	sStart := strings.Index(body, `id="view-studio"`)
	if sStart < 0 {
		t.Fatal("studio view section not found in shell")
	}
	sEnd := strings.Index(body[sStart:], "</section>")
	if sEnd < 0 {
		t.Fatal("studio view section unterminated")
	}
	section := body[sStart : sStart+sEnd]
	if !strings.Contains(section, `id="tensor-dropzone"`) {
		t.Errorf("studio view missing id=\"tensor-dropzone\"")
	}
	if !strings.Contains(section, `id="tensor-picker"`) {
		t.Errorf("studio view missing id=\"tensor-picker\" (the file input fallback)")
	}
	if !strings.Contains(section, `id="tensor-result"`) {
		t.Errorf("studio view missing id=\"tensor-result\" (the result container)")
	}
	if !strings.Contains(section, `id="tensor-rows"`) {
		t.Errorf("studio view missing id=\"tensor-rows\" (the result table body)")
	}
}

// TestWebInspect_DropzoneRegionAria pins the dropzone's a11y attributes:
// `role="region"`, an aria-label, an aria-describedby pointing at the hint.
// This matches the W8 ONNX dropzone pattern per A11Y.md.
func TestWebInspect_DropzoneRegionAria(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`(?s)<div[^>]*id="tensor-dropzone"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <div id=\"tensor-dropzone\">")
	}
	if !strings.Contains(m, `role="region"`) {
		t.Errorf("tensor-dropzone missing role=\"region\": %s", m)
	}
	if !regexp.MustCompile(`aria-label="[^"]+"`).MatchString(m) {
		t.Errorf("tensor-dropzone missing aria-label: %s", m)
	}
	if !strings.Contains(m, `aria-describedby="tensor-hint"`) {
		t.Errorf("tensor-dropzone missing aria-describedby=\"tensor-hint\": %s", m)
	}
	if !strings.Contains(m, `tabindex="0"`) {
		t.Errorf("tensor-dropzone missing tabindex=\"0\" (keyboard-focusable): %s", m)
	}
	if !strings.Contains(body, `id="tensor-hint"`) {
		t.Errorf("studio.html missing the hint paragraph the aria-describedby points at")
	}
}

// TestWebInspect_FormatAcceptList pins the file input's accept attribute:
// all three supported extensions must be present. The browser uses the
// accept list to filter the file picker for the user.
func TestWebInspect_FormatAcceptList(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="tensor-picker"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <input id=\"tensor-picker\">")
	}
	for _, ext := range []string{".npy", ".npz", ".safetensors"} {
		if !strings.Contains(m, ext) {
			t.Errorf("tensor-picker accept= missing %q: %s", ext, m)
		}
	}
	if !strings.Contains(m, `type="file"`) {
		t.Errorf("tensor-picker not a type=\"file\" input: %s", m)
	}
}

// TestWebInspect_ResultTableSemanticHeaders pins the result table uses a
// real <thead><th scope="col"> structure so screen readers expose column
// headers. Tables are the right element for tabular data per WCAG.
func TestWebInspect_ResultTableSemanticHeaders(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	resultStart := strings.Index(body, `id="tensor-result"`)
	if resultStart < 0 {
		t.Fatal("tensor-result section not found")
	}
	// Look ahead to the close of the surrounding section.
	tail := body[resultStart:]
	closeIdx := strings.Index(tail, "</section>")
	if closeIdx < 0 {
		closeIdx = len(tail)
	}
	section := tail[:closeIdx]
	if !strings.Contains(section, `<table`) {
		t.Errorf("result section missing a <table> element")
	}
	if !strings.Contains(section, `<thead>`) {
		t.Errorf("result section missing <thead>")
	}
	if !strings.Contains(section, `<tbody`) {
		t.Errorf("result section missing <tbody>")
	}
	if !regexp.MustCompile(`<th\s+scope="col">`).MatchString(section) {
		t.Errorf("result section missing <th scope=\"col\"> headers")
	}
	// The table itself must carry an aria-label so it announces with a
	// useful name even when the heading reference is far above.
	if !regexp.MustCompile(`<table[^>]*aria-label="[^"]+"`).MatchString(section) {
		t.Errorf("result table missing aria-label")
	}
}

// TestWebInspect_PrivacyClaimInHint pins that the dropzone's hint copy
// explicitly states the privacy property: bytes stay in the tab. This is
// the user-facing affirmation of the WASM-tier contract per spec §1.3.
func TestWebInspect_PrivacyClaimInHint(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	// The hint paragraph must mention that the payload stays in the tab
	// (or some equivalent claim). The exact wording is not load-bearing;
	// the privacy claim itself is.
	hintStart := strings.Index(body, `id="tensor-hint"`)
	if hintStart < 0 {
		t.Fatal("tensor-hint paragraph not found")
	}
	tail := body[hintStart:]
	closeIdx := strings.Index(tail, "</p>")
	if closeIdx < 0 {
		t.Fatal("tensor-hint paragraph unterminated")
	}
	hint := tail[:closeIdx]
	if !strings.Contains(hint, "tab") {
		t.Errorf("tensor-hint should mention the bytes staying in the tab; got %q", hint)
	}
}

// TestWebInspect_NoServerEndpoint pins that no /api/inspect* endpoint
// exists. Inspection is WASM-tier; the server must never receive tensor
// bytes (privacy property per spec §1.3 / §5.1).
func TestWebInspect_NoServerEndpoint(t *testing.T) {
	srv := newWebServer(t)
	for _, path := range []string{
		"/api/inspect",
		"/api/inspect/",
		"/api/inspect/tensor",
	} {
		resp, _ := get(t, srv, path)
		// 404 is the right answer (path not registered). Anything else
		// would suggest a server-side handler was added by mistake.
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("GET %s: status %d, want 404 (no server endpoint for tensor inspect)",
				path, resp.StatusCode)
		}
	}
}

// TestWebInspect_RendererWired pins that studio.js exports the inspector
// renderer hooks (so a manual console invocation or a future unit test
// driver can reach them) AND that the studio-home renderer calls into
// the inspector init. Without the call the dropzone would render but
// never wire its event listeners.
func TestWebInspect_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	for _, needle := range []string{
		"annealInspectTensor", // the WASM bridge call
		"detectInspectFormat", // format dispatch from filename
		"renderStudioView",    // the home-view renderer (calls dropzone init)
		"initInspectDropzone", // the dropzone wiring function
		"inspectFile",         // the per-file dispatch path
	} {
		if !strings.Contains(body, needle) {
			t.Errorf("studio.js missing %q (inspector wiring not landed)", needle)
		}
	}
}
