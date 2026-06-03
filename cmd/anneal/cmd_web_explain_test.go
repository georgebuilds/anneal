//go:build !js

// Tests for the W3 explain view markup and JS contract. JS state machines
// are not executable in pure Go; we assert the contract is encoded as
// searchable literals in the source. The Go-side parity gate (real
// symbolic.upat parsed → JSON shape, gradient rule lookup against
// tensor.Gradient) lives in viz/explain_test.go and in
// cmd_explain_test.go (CLI vs WASM coverage parity).

package main

import (
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestWebExplain_StubReplaced pins that the explain view markup is the W3
// pane, not the W0/W1 stub copy. If a future refactor accidentally re-stubs
// this view, the test catches it.
func TestWebExplain_StubReplaced(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /: status %d", resp.StatusCode)
	}
	if !strings.Contains(body, `class="explain-pane"`) {
		t.Errorf("studio.html missing class=\"explain-pane\" (W3 layout root)")
	}
	if !strings.Contains(body, `class="explain-list"`) {
		t.Errorf("studio.html missing class=\"explain-list\" (W3 left pane)")
	}
	if !strings.Contains(body, `class="explain-detail"`) {
		t.Errorf("studio.html missing class=\"explain-detail\" (W3 right pane)")
	}
	// Scope the stub-check to the explain view section so unrelated copy
	// elsewhere does not trip it.
	xStart := strings.Index(body, `id="view-explain"`)
	if xStart < 0 {
		t.Fatalf("explain view section not found in shell")
	}
	xEnd := strings.Index(body[xStart:], `id="view-train"`)
	if xEnd < 0 {
		xEnd = len(body) - xStart
	}
	section := body[xStart : xStart+xEnd]
	if strings.Contains(section, "view: explain coming soon") {
		t.Errorf("explain view still contains W0/W1 stub copy")
	}
}

// TestWebExplain_ListboxRole pins the op list is a real listbox and the
// detail region is a polite live region.
func TestWebExplain_ListboxRole(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	// The listbox role must appear inside the explain view.
	xStart := strings.Index(body, `id="view-explain"`)
	if xStart < 0 {
		t.Fatalf("explain view section not found")
	}
	xEnd := strings.Index(body[xStart:], `id="view-train"`)
	section := body[xStart : xStart+xEnd]
	if !strings.Contains(section, `role="listbox"`) {
		t.Errorf("explain view missing role=\"listbox\" on the op list (W3 a11y)")
	}
	if !strings.Contains(section, `id="op-list-items"`) {
		t.Errorf("explain view missing id=\"op-list-items\" (W3 mount point)")
	}
}

// TestWebExplain_AriaLiveDetailRegion pins the detail article exposes
// aria-live="polite" + aria-atomic="false" so selection changes announce
// without stealing focus.
func TestWebExplain_AriaLiveDetailRegion(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	xStart := strings.Index(body, `id="view-explain"`)
	xEnd := strings.Index(body[xStart:], `id="view-train"`)
	section := body[xStart : xStart+xEnd]
	if !strings.Contains(section, `aria-live="polite"`) {
		t.Errorf("explain-detail missing aria-live=\"polite\"")
	}
	if !strings.Contains(section, `aria-atomic="false"`) {
		t.Errorf("explain-detail missing aria-atomic=\"false\"")
	}
}

// TestWebExplain_SearchInputLabeled pins the search input has a real label.
// The label is visually-hidden but exposed to screen readers via the
// <label for="op-search"> pattern.
func TestWebExplain_SearchInputLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<input[^>]*id="op-search"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <input id=\"op-search\">")
	}
	if !strings.Contains(m, `aria-label="search ops"`) {
		t.Errorf("op-search input missing aria-label=\"search ops\": %s", m)
	}
	// And the matching visually-hidden label exists for redundancy.
	if !strings.Contains(body, `<label for="op-search"`) {
		t.Errorf("studio.html missing <label for=\"op-search\"> (W3 redundant labelling)")
	}
}

// TestWebExplain_GradientPreLabeled pins that the gradient rule <pre> region
// carries an aria-label so screen-reader users hear what they have landed
// in, and a tabindex so they can keyboard-scroll into it.
func TestWebExplain_GradientPreLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<pre[^>]*id="exp-grad"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <pre id=\"exp-grad\">")
	}
	if !strings.Contains(m, `aria-label="gradient rule pattern"`) {
		t.Errorf("exp-grad <pre> missing aria-label: %s", m)
	}
	if !strings.Contains(m, `tabindex="0"`) {
		t.Errorf("exp-grad <pre> missing tabindex=\"0\": %s", m)
	}
}

// TestWebExplain_PlayButtonAriaLabel pins the play-rewrite button has a
// descriptive aria-label (the visible text is "play rewrite"; a11y label
// makes the intent explicit for screen-reader users).
func TestWebExplain_PlayButtonAriaLabel(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<button[^>]*id="play-rewrite"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <button id=\"play-rewrite\">")
	}
	if !strings.Contains(m, `aria-label="play rewrite animation"`) {
		t.Errorf("play-rewrite missing aria-label: %s", m)
	}
}

// TestWebExplain_KeyboardNavWired pins that the explain-view section of
// studio.js has arrow-key handling. A regex pin so future refactors that
// split the switch into helper functions still match.
func TestWebExplain_KeyboardNavWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "initExplainKeyboard") {
		t.Errorf("studio.js missing initExplainKeyboard (W3 keyboard init)")
	}
	// All three direction keys must appear (already required by W2; keep
	// the W3 check here for self-contained gating).
	for _, want := range []string{"ArrowDown", "ArrowUp", "Home", "End"} {
		if !strings.Contains(body, "'"+want+"'") && !strings.Contains(body, `"`+want+`"`) {
			t.Errorf("studio.js missing %q key handler (W3 keyboard nav)", want)
		}
	}
}

// TestWebExplain_DeepLinkURL pins the URL contract: /x/<op>. The renderer
// must serialize its selection into /x/<op> so the deep link is real
// (DESIGN.md §7).
func TestWebExplain_DeepLinkURL(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "'/x/'") && !strings.Contains(body, `"/x/"`) {
		t.Errorf("studio.js does not push /x/ into the URL (DESIGN §7 deep link)")
	}
	// And the opFromExplainPath helper must exist so /x/<op> is parsed.
	if !strings.Contains(body, "opFromExplainPath") {
		t.Errorf("studio.js missing opFromExplainPath (W3 URL parser)")
	}
}

// TestWebExplain_HeadingHierarchy pins exactly one <h1> inside the explain
// view section AND that the sub-section headings are <h3>, not <h2> (the
// lede is a paragraph, not a heading; the view headings are h3 to keep the
// section's h1 the only one in the rendered a11y tree). Mirrors the
// foundation gate but scoped to the explain view.
func TestWebExplain_HeadingHierarchy(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	xStart := strings.Index(body, `id="view-explain"`)
	if xStart < 0 {
		t.Fatalf("explain view section not found")
	}
	xEnd := strings.Index(body[xStart:], `id="view-train"`)
	section := body[xStart : xStart+xEnd]
	h1Re := regexp.MustCompile(`<h1\b`)
	h2Re := regexp.MustCompile(`<h2\b`)
	h3Re := regexp.MustCompile(`<h3\b`)
	if n := len(h1Re.FindAllString(section, -1)); n != 1 {
		t.Errorf("explain view has %d <h1>; want exactly 1", n)
	}
	// One h2 is allowed (the op-name in the detail header). h3 is used for
	// rewrite-rules / gradient-rule / before-after section headings.
	if n := len(h2Re.FindAllString(section, -1)); n != 1 {
		t.Errorf("explain view has %d <h2>; want exactly 1 (the op-name header)", n)
	}
	if n := len(h3Re.FindAllString(section, -1)); n < 3 {
		t.Errorf("explain view has %d <h3>; want >= 3 (rewrite rules / gradient rule / before-after)", n)
	}
}

// TestWebExplain_RendererWired pins that the renderers table dispatches to
// a real implementation for "explain", not a W0/W1/W4 stub.
func TestWebExplain_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "renderExplainView") {
		t.Errorf("studio.js missing renderExplainView (W3 renderer)")
	}
	// And the WASM RPC call to annealExplainOp must be present.
	if !strings.Contains(body, "'annealExplainOp'") && !strings.Contains(body, `"annealExplainOp"`) {
		t.Errorf("studio.js missing annealExplainOp wasm.call (W3 bridge contract)")
	}
}

// TestWebExplain_UpatByteParity pins that the embedded copy of symbolic.upat
// in viz/ matches the upstream rewrite/rules/symbolic.upat byte-for-byte.
// Drift safety: editing one file without the other would silently desync
// the WASM rules list from what the rewrite engine actually fires.
func TestWebExplain_UpatByteParity(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller: cannot determine test file path")
	}
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	vizCopy, err := os.ReadFile(filepath.Join(repoRoot, "viz", "symbolic.upat"))
	if err != nil {
		t.Fatalf("read viz/symbolic.upat: %v", err)
	}
	upstream, err := os.ReadFile(filepath.Join(repoRoot, "rewrite", "rules", "symbolic.upat"))
	if err != nil {
		t.Fatalf("read rewrite/rules/symbolic.upat: %v", err)
	}
	if string(vizCopy) != string(upstream) {
		t.Errorf("viz/symbolic.upat is out of sync with rewrite/rules/symbolic.upat — re-copy with `cp rewrite/rules/symbolic.upat viz/symbolic.upat`")
	}
}
