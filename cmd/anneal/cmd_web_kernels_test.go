//go:build !js

// Tests for the W2 kernels view markup and JS contract. JS state machines
// are not executable in pure Go; we assert the contract is encoded as
// searchable literals in the source. The Go-side bytewise WGSL parity
// gate lives in cmd_kernels_test.go (TestKernelsWASMMatchesCLI_*).

package main

import (
	"net/http"
	"regexp"
	"strings"
	"testing"
)

// TestWebKernels_StubReplaced pins that the kernels view markup is the W2
// pane, not the W0/W1 stub copy. If a future refactor accidentally re-stubs
// this view, the test catches it.
func TestWebKernels_StubReplaced(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /: status %d", resp.StatusCode)
	}
	// The W2 markup carries kernel-list and kernel-detail; the old stub said
	// "view: kernels coming soon."
	if !strings.Contains(body, `class="kernels-pane"`) {
		t.Errorf("studio.html missing class=\"kernels-pane\" (W2 layout root)")
	}
	if !strings.Contains(body, "kernel-list") {
		t.Errorf("studio.html missing kernel-list (W2 left pane)")
	}
	if !strings.Contains(body, "kernel-detail") {
		t.Errorf("studio.html missing kernel-detail (W2 right pane)")
	}
	// And the old stub copy is gone from this view.
	// Search the kernels view section only (not the whole shell).
	kStart := strings.Index(body, `id="view-kernels"`)
	if kStart < 0 {
		t.Fatalf("kernels view section not found in shell")
	}
	kEnd := strings.Index(body[kStart:], "</section>")
	if kEnd < 0 {
		kEnd = len(body) - kStart
	}
	section := body[kStart : kStart+kEnd]
	if strings.Contains(section, "view: kernels coming soon") {
		t.Errorf("kernels view still contains W0/W1 stub copy")
	}
}

// TestWebKernels_ListboxRole pins the kernel list is a real listbox and the
// detail region is a polite live region.
func TestWebKernels_ListboxRole(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `role="listbox"`) {
		t.Errorf("studio.html missing role=\"listbox\" on the kernel list (W2 a11y)")
	}
	// The detail region announces selection changes via aria-live.
	// We check the live attribute appears within the kernels view section.
	kStart := strings.Index(body, `id="view-kernels"`)
	kEnd := strings.Index(body[kStart:], "</section>")
	section := body[kStart : kStart+kEnd]
	if !strings.Contains(section, `aria-live="polite"`) {
		t.Errorf("kernel-detail missing aria-live=\"polite\"")
	}
	if !strings.Contains(section, `aria-atomic="false"`) {
		t.Errorf("kernel-detail missing aria-atomic=\"false\"")
	}
}

// TestWebKernels_WGSLPreLabeled pins that the WGSL <pre> region carries an
// aria-label so screen-reader users hear what they have landed in, and a
// tabindex so they can keyboard-scroll into it.
func TestWebKernels_WGSLPreLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<pre[^>]*id="k-wgsl"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <pre id=\"k-wgsl\">")
	}
	if !strings.Contains(m, `aria-label="WGSL source"`) {
		t.Errorf("k-wgsl <pre> missing aria-label=\"WGSL source\": %s", m)
	}
	if !strings.Contains(m, `tabindex="0"`) {
		t.Errorf("k-wgsl <pre> missing tabindex=\"0\" (keyboard-scrollable region): %s", m)
	}
}

// TestWebKernels_DiffToggleAriaPressed pins the toggle uses the documented
// ARIA toggle pattern (aria-pressed=false initially).
func TestWebKernels_DiffToggleAriaPressed(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	re := regexp.MustCompile(`<button[^>]*id="diff-toggle"[^>]*>`)
	m := re.FindString(body)
	if m == "" {
		t.Fatal("studio.html missing <button id=\"diff-toggle\">")
	}
	if !strings.Contains(m, `aria-pressed="false"`) {
		t.Errorf("diff-toggle missing aria-pressed=\"false\" initial: %s", m)
	}
}

// TestWebKernels_KeyboardNav pins that the kernels-view section of studio.js
// has arrow-key handling. A regex pin so future refactors that split the
// switch into helper functions still match.
func TestWebKernels_KeyboardNav(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	// All three direction keys must appear.
	for _, want := range []string{"ArrowDown", "ArrowUp", "Home", "End"} {
		if !strings.Contains(body, "'"+want+"'") && !strings.Contains(body, `"`+want+`"`) {
			t.Errorf("studio.js missing %q key handler (kernels view keyboard nav)", want)
		}
	}
	// The kernels-keyboard initializer must be wired into the boot function.
	if !strings.Contains(body, "initKernelsKeyboard") {
		t.Errorf("studio.js missing initKernelsKeyboard (kernels view keyboard init)")
	}
}

// TestWebKernels_TokenizerKeywords pins that the WGSL tokenizer's keyword
// list includes a representative sample (fn, var, let, for, if, struct).
// Brittle by design: if the keyword list shrinks, the test catches it.
func TestWebKernels_TokenizerKeywords(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "tokenizeWGSL") {
		t.Fatal("studio.js missing tokenizeWGSL function")
	}
	// Look for the keyword set literal. The test does not pin the exact
	// syntax (new Set([...]) vs an object); it pins that the keywords are
	// present as quoted string literals nearby.
	for _, kw := range []string{
		"'fn'", "'var'", "'let'", "'for'", "'while'",
		"'if'", "'else'", "'return'", "'struct'",
	} {
		if !strings.Contains(body, kw) {
			t.Errorf("studio.js missing WGSL keyword %s in tokenizer set", kw)
		}
	}
	// Types: at least the four scalars + vec/mat parameterised ones.
	for _, tp := range []string{
		"'f32'", "'i32'", "'u32'", "'bool'",
		"'vec2'", "'vec3'", "'vec4'",
		"'mat2x2'", "'mat3x3'", "'mat4x4'",
		"'array'",
	} {
		if !strings.Contains(body, tp) {
			t.Errorf("studio.js missing WGSL type %s in tokenizer set", tp)
		}
	}
}

// TestWebKernels_TokenizerSpanClasses pins the CSS contract: the tokenizer
// emits .tk-keyword, .tk-type, .tk-number, .tk-string, .tk-comment, plus
// .tk-attribute (for @compute etc.) and .tk-builtin (for main, gid, ...).
func TestWebKernels_TokenizerSpanClasses(t *testing.T) {
	srv := newWebServer(t)
	_, css := get(t, srv, "/static/studio.css")
	for _, cls := range []string{
		".tk-keyword", ".tk-type", ".tk-number", ".tk-string",
		".tk-comment", ".tk-attribute", ".tk-builtin", ".tk-ident", ".tk-punct",
	} {
		if !strings.Contains(css, cls) {
			t.Errorf("studio.css missing %s rule (W2 tokenizer span styling)", cls)
		}
	}
	// DD1 (color is never alone): keywords should be bold or have another
	// non-color channel. Pin a font-weight or font-style on the keyword and
	// comment classes so a future "make it all teal" refactor breaks the
	// gate.
	if !regexp.MustCompile(`\.tk-keyword[^}]*font-weight`).MatchString(css) {
		t.Errorf("studio.css .tk-keyword missing font-weight (DD1: pair colour with weight)")
	}
	if !regexp.MustCompile(`\.tk-comment[^}]*font-style`).MatchString(css) {
		t.Errorf("studio.css .tk-comment missing font-style (DD1: pair colour with style)")
	}
}

// TestWebKernels_RendererWired pins that the renderers table dispatches to a
// real implementation for "kernels", not a W0/W1 stub.
func TestWebKernels_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	// The kernels renderer must call into a real function (renderKernelsView).
	if !strings.Contains(body, "renderKernelsView") {
		t.Errorf("studio.js missing renderKernelsView (W2 renderer)")
	}
	// And the WASM RPC call to annealGetKernels must be present.
	if !strings.Contains(body, "'annealGetKernels'") && !strings.Contains(body, `"annealGetKernels"`) {
		t.Errorf("studio.js missing annealGetKernels wasm.call (W2 bridge contract)")
	}
}

// TestWebKernels_FusionSpanGutterClasses pins the gutter CSS for the three
// span labels (fwd, bwd, fused). Each must have its own colour and the rule
// must reference the brand-token vars so the dark/light theme tokens flow
// through.
func TestWebKernels_FusionSpanGutterClasses(t *testing.T) {
	srv := newWebServer(t)
	_, css := get(t, srv, "/static/studio.css")
	for _, want := range []string{
		".wgsl .gutter.fwd",
		".wgsl .gutter.bwd",
		".wgsl .gutter.fused",
	} {
		if !strings.Contains(css, want) {
			t.Errorf("studio.css missing %s (W2 fusion-span gutter)", want)
		}
	}
	// And each rule must reference its brand token.
	if !regexp.MustCompile(`\.wgsl \.gutter\.fwd[^}]*--teal`).MatchString(css) {
		t.Errorf("studio.css .wgsl .gutter.fwd must reference var(--teal)")
	}
	if !regexp.MustCompile(`\.wgsl \.gutter\.bwd[^}]*--ember`).MatchString(css) {
		t.Errorf("studio.css .wgsl .gutter.bwd must reference var(--ember)")
	}
	if !regexp.MustCompile(`\.wgsl \.gutter\.fused[^}]*--gold`).MatchString(css) {
		t.Errorf("studio.css .wgsl .gutter.fused must reference var(--gold)")
	}
}

// TestWebKernels_DeepLinkURL pins the URL contract: /k/<model>?kernel=K3.
// The renderer must serialize its selection into ?kernel= so the deep link
// is real (DESIGN.md §7).
func TestWebKernels_DeepLinkURL(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "'kernel='") && !strings.Contains(body, "?kernel=") {
		t.Errorf("studio.js does not push ?kernel= into the URL (DESIGN §7 deep link)")
	}
	// And the modelFromPath helper must exist so /k/<model> is parsed.
	if !strings.Contains(body, "modelFromPath") {
		t.Errorf("studio.js missing modelFromPath (W2 URL parser)")
	}
}
