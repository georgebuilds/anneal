//go:build !js

// Tests for the W4 visualize view (embed + node inspector drawer). JS state
// machines are not executable in pure Go; we assert the contract is encoded
// as searchable literals in the source. The Go-side node-detail tests live
// in viz/nodedetail_test.go.

package main

import (
	"net/http"
	"regexp"
	"strings"
	"testing"
)

// TestWebVisualize_StubReplaced pins that the visualize view markup is the
// W4 pane (iframe + drawer), not the W0–W3 stub copy.
func TestWebVisualize_StubReplaced(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /: status %d", resp.StatusCode)
	}
	vStart := strings.Index(body, `id="view-visualize"`)
	if vStart < 0 {
		t.Fatal("visualize view section not found in shell")
	}
	vEnd := strings.Index(body[vStart:], "</section>")
	if vEnd < 0 {
		vEnd = len(body) - vStart
	}
	section := body[vStart : vStart+vEnd]
	if strings.Contains(section, "view: visualize coming soon") {
		t.Errorf("visualize view still contains W0-W3 stub copy")
	}
	if !strings.Contains(section, `class="visualize-pane"`) {
		t.Errorf("visualize view missing class=\"visualize-pane\" (W4 layout root)")
	}
	if !strings.Contains(section, `id="viz-iframe"`) {
		t.Errorf("visualize view missing id=\"viz-iframe\" (W4 iframe)")
	}
	if !strings.Contains(section, `id="node-inspector"`) {
		t.Errorf("visualize view missing id=\"node-inspector\" (W4 drawer)")
	}
}

// TestWebVisualize_IframeSandbox pins the iframe's sandbox value: the
// minimum needed for the WASM viz to run (allow-scripts allow-same-origin).
// No allow-forms, no allow-top-navigation.
func TestWebVisualize_IframeSandbox(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<iframe[^>]*id="viz-iframe"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <iframe id=\"viz-iframe\">")
	}
	if !strings.Contains(m, `sandbox="allow-scripts allow-same-origin"`) {
		t.Errorf("viz-iframe sandbox attribute does not match contract: %s", m)
	}
	// And it MUST carry a title (WCAG 4.1.2 — iframes need accessible names).
	if !regexp.MustCompile(`title="[^"]+"`).MatchString(m) {
		t.Errorf("viz-iframe missing title attribute (WCAG 4.1.2): %s", m)
	}
	// Tabindex 0 so keyboard users can land on it.
	if !strings.Contains(m, `tabindex="0"`) {
		t.Errorf("viz-iframe missing tabindex=\"0\" (keyboard-focusable): %s", m)
	}
}

// TestWebVisualize_DrawerRoleRegion pins the drawer is `role="region"` with
// an aria-label. Per A11Y.md the view's h1 stays the section heading; the
// drawer carries an h2 inside.
func TestWebVisualize_DrawerRoleRegion(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<aside[^>]*id="node-inspector"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <aside id=\"node-inspector\">")
	}
	if !strings.Contains(m, `role="region"`) {
		t.Errorf("node-inspector missing role=\"region\": %s", m)
	}
	if !regexp.MustCompile(`aria-label="[^"]+"`).MatchString(m) {
		t.Errorf("node-inspector missing aria-label: %s", m)
	}
	if !strings.Contains(m, "hidden") {
		t.Errorf("node-inspector should start hidden: %s", m)
	}
}

// TestWebVisualize_CloseButtonAriaLabel pins the close button's accessible
// name. The visible text "close" pairs with the aria-label "close inspector".
func TestWebVisualize_CloseButtonAriaLabel(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<button[^>]*id="node-inspector-close"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <button id=\"node-inspector-close\">")
	}
	if !strings.Contains(m, `aria-label="close inspector"`) {
		t.Errorf("close button missing aria-label=\"close inspector\": %s", m)
	}
}

// TestWebVisualize_KeyboardEscapeWiring pins that the visualize view's JS
// block handles Escape. We scan the studio.js source for the Escape key
// handler in the visualize section so a refactor can not silently drop it.
func TestWebVisualize_KeyboardEscapeWiring(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	// The visualize section must reference 'Escape' on a keydown handler.
	// Pin a regex that targets the visualize area: the wireVisualizeListeners
	// function contains both "Escape" and "node-inspector".
	if !strings.Contains(body, "wireVisualizeListeners") {
		t.Fatal("studio.js missing wireVisualizeListeners (W4 init)")
	}
	if !strings.Contains(body, "'Escape'") && !strings.Contains(body, `"Escape"`) {
		t.Errorf("studio.js W4 block missing 'Escape' key handler")
	}
	if !strings.Contains(body, "closeNodeInspector") {
		t.Errorf("studio.js missing closeNodeInspector (W4 close path)")
	}
}

// TestWebVisualize_DeepLinkURL pins the /v/<model>?node=<id> contract.
// renderVisualizeView reads the URL via modelFromVizPath and nodeIdFromQuery.
func TestWebVisualize_DeepLinkURL(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "modelFromVizPath") {
		t.Errorf("studio.js missing modelFromVizPath (W4 URL parser)")
	}
	if !strings.Contains(body, "nodeIdFromQuery") {
		t.Errorf("studio.js missing nodeIdFromQuery (?node=<id> parser)")
	}
	// And the route table must already know about /v/.
	if !strings.Contains(body, "p.startsWith('/v/')") {
		t.Errorf("studio.js missing /v/ route (W4 deep-link spec §10)")
	}
	// Pushing the URL with ?node= on a click closes the loop.
	if !strings.Contains(body, "'?node='") && !strings.Contains(body, "?node=") {
		t.Errorf("studio.js does not push ?node= into the URL (DESIGN §7 deep link)")
	}

	// And the server must honour /v/mlp?node=n0 by serving the shell.
	resp, _ := get(t, srv, "/v/mlp?node=n0")
	if resp.StatusCode != http.StatusOK {
		t.Errorf("/v/mlp?node=n0 status %d, want 200", resp.StatusCode)
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
		t.Errorf("/v/mlp?node=n0 Content-Type %q, want text/html prefix", resp.Header.Get("Content-Type"))
	}
}

// TestWebVisualize_PostMessageProtocol pins the iframe/parent protocol:
// nodeClick + nodeSelected (per spec §5.2). Both message types must appear
// in the studio.js source.
func TestWebVisualize_PostMessageProtocol(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	for _, want := range []string{"nodeClick", "nodeSelected"} {
		if !strings.Contains(body, want) {
			t.Errorf("studio.js missing %q message type (W4 postMessage protocol)", want)
		}
	}
	// And the embed page (served at /visualize/embed) must speak both sides
	// of the protocol.
	resp, embed := get(t, srv, "/visualize/embed")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /visualize/embed: status %d", resp.StatusCode)
	}
	for _, want := range []string{"nodeClick", "nodeSelected"} {
		if !strings.Contains(embed, want) {
			t.Errorf("/visualize/embed missing %q message type", want)
		}
	}
}

// TestWebVisualize_EmbedEndpointServes pins that /visualize/embed returns
// 200 with the W4 embed wrapper HTML (it loads the viz artifact and adds
// the postMessage bridge).
func TestWebVisualize_EmbedEndpointServes(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/visualize/embed")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /visualize/embed: status %d", resp.StatusCode)
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
		t.Errorf("Content-Type %q, want text/html prefix", resp.Header.Get("Content-Type"))
	}
	// The page must load the viz artifact's app.js, not the studio.js.
	if !strings.Contains(body, "/visualize/static/app.js") {
		t.Errorf("/visualize/embed does not load /visualize/static/app.js (W4 verbatim viz)")
	}
	// And the standalone viz files must be accessible under /visualize/static/.
	resp2, _ := get(t, srv, "/visualize/static/app.js")
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("/visualize/static/app.js status %d, want 200", resp2.StatusCode)
	}
	// Also the standalone viz REST fallback.
	resp3, _ := get(t, srv, "/visualize/api/graph?name=mlp")
	if resp3.StatusCode != http.StatusOK {
		t.Errorf("/visualize/api/graph?name=mlp status %d, want 200", resp3.StatusCode)
	}
}

// TestWebVisualize_NodeAttrsDL pins the drawer's attribute layout: a <dl>
// with dt/dd pairs for dtype / shape / phase / arg / source.
func TestWebVisualize_NodeAttrsDL(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	vStart := strings.Index(body, `id="view-visualize"`)
	if vStart < 0 {
		t.Fatal("visualize view section not found")
	}
	vEnd := strings.Index(body[vStart:], "</section>")
	section := body[vStart : vStart+vEnd]
	if !regexp.MustCompile(`<dl[^>]*class="node-attrs"`).MatchString(section) {
		t.Errorf("visualize view missing <dl class=\"node-attrs\">")
	}
	for _, k := range []string{"dtype", "shape", "phase", "arg", "source"} {
		// Each key appears as a <dt> followed by a <dd>.
		re := regexp.MustCompile(`<dt[^>]*>` + k + `</dt><dd[^>]*></dd>`)
		if !re.MatchString(section) {
			t.Errorf("node-attrs missing <dt>%s</dt><dd></dd> pair", k)
		}
	}
}

// TestWebVisualize_FocusReturnsAfterClose pins that the JS captures the
// previously active element so Escape / close restores focus (WCAG 2.4.3 /
// 2.4.11; A11Y.md §3c).
func TestWebVisualize_FocusReturnsAfterClose(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	// Either the documented snake_case from the spec or our chosen camelCase.
	if !strings.Contains(body, "previousActiveElement.focus") &&
		!strings.Contains(body, "previousActiveElement") {
		t.Errorf("studio.js missing previousActiveElement restore (W4 focus return)")
	}
}

// TestWebVisualize_RendererWired pins the renderers table dispatches to a
// real implementation for "visualize", not a W0/W1 stub.
func TestWebVisualize_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "renderVisualizeView") {
		t.Errorf("studio.js missing renderVisualizeView (W4 renderer)")
	}
	if !strings.Contains(body, "'annealNodeDetail'") && !strings.Contains(body, `"annealNodeDetail"`) {
		t.Errorf("studio.js missing annealNodeDetail wasm.call (W4 bridge contract)")
	}
}

// TestWebVisualize_DrawerCSS pins the drawer + iframe styling: brand-token
// border colours, the slide-in animation, and the reduced-motion override.
func TestWebVisualize_DrawerCSS(t *testing.T) {
	srv := newWebServer(t)
	_, css := get(t, srv, "/static/studio.css")
	for _, want := range []string{
		".visualize-pane",
		".node-inspector",
		"ni-slide-in",
		"prefers-reduced-motion",
	} {
		if !strings.Contains(css, want) {
			t.Errorf("studio.css missing %q (W4 drawer styling)", want)
		}
	}
	// DD1: parent vs child glyphs (color is never alone).
	if !regexp.MustCompile(`section\[aria-label="parents"\][^}]*\{[^}]*content: "↑`).MatchString(css) &&
		!strings.Contains(css, `content: "↑ "`) {
		t.Errorf("studio.css missing parent ↑ glyph (DD1: pair colour with shape)")
	}
	if !strings.Contains(css, `content: "↓ "`) {
		t.Errorf("studio.css missing child ↓ glyph (DD1: pair colour with shape)")
	}
}
