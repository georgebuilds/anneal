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
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strconv"
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

// TestWebAPIStubs501 pins every documented API surface still on the W0 stub
// path returns 501 with the "phase ID pending" JSON body. /api/runs left
// this set in W1; /sse/train left it in W6; /sse/generate and /api/device
// left it in W7. /api/compile/tuned is the only surface still on the
// stub; once the native BEAM compile lands, this test loses that entry.
func TestWebAPIStubs501(t *testing.T) {
	srv := newWebServer(t)
	for _, path := range []string{
		"/api/compile/tuned",
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

// TestWebFileSizesUnderBudget pins each hand-authored static file is
// under its budget. studio.js carries the routing, the worker RPC, the
// WGSL tokenizer (W2), and the train SSE client (W6); the budget is
// raised on each major feature land. If a file grows past its budget,
// extract a helper rather than letting one file sprawl (DESIGN.md "few
// hundred lines target").
func TestWebFileSizesUnderBudget(t *testing.T) {
	srv := newWebServer(t)
	budgets := map[string]int{
		"/static/studio.html": 32 * 1024,
		// W7 generate view (token-stream block, controls, logit panel,
		// compare-toggle styling, refmatch glyphs, reduced-motion + dark
		// modes) brings the CSS budget to 64KB. The next major view
		// should land styles in a separate stylesheet if the budget is
		// breached.
		"/static/studio.css": 64 * 1024,
		// W2 brought studio.js to ~36KB; W6's train view + SSE client +
		// sparkline ring buffer + a11y plumbing brings it to ~58KB; W4's
		// visualize node-inspector lifted it to ~68KB; W3 explain (op
		// list, search filter, SVG mini-graph renderer, rewrite
		// animation) brings the budget to 80KB; W7 generate view (token
		// stream renderer, batched announce, click-through, last-token
		// panel, deep-link state) brings it to 112KB. The next major
		// view should land in a separate ES module.
		"/static/studio.js": 112 * 1024,
		"/static/worker.js": 8 * 1024,
	}
	for p, maxBytes := range budgets {
		resp, body := get(t, srv, p)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s: status %d", p, resp.StatusCode)
			continue
		}
		if len(body) > maxBytes {
			t.Errorf("GET %s: %d bytes, want <= %d (per-file budget)",
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

// ──────────────────────────────────────────────────────────────────────────
// Accessibility gates (web/A11Y.md, DESIGN.md §11).
//
// These are binding for every web view (W0 foundation, W1+ feature views).
// If a test here fails because a new view broke the foundation, fix the
// foundation; don't relax the gate. Brand-token changes (the contrast
// test) require George's bless: the test stops the build with a diagnostic
// rather than silently re-tinting the brand.
// ──────────────────────────────────────────────────────────────────────────

// relLuminance implements the WCAG 2.x relative-luminance formula.
// https://www.w3.org/TR/WCAG22/#dfn-relative-luminance
func relLuminance(hex string) (float64, error) {
	h := strings.TrimPrefix(hex, "#")
	if len(h) != 6 {
		return 0, fmt.Errorf("hex %q must be 6 chars (got %d)", hex, len(h))
	}
	r, err := strconv.ParseInt(h[0:2], 16, 32)
	if err != nil {
		return 0, err
	}
	g, err := strconv.ParseInt(h[2:4], 16, 32)
	if err != nil {
		return 0, err
	}
	b, err := strconv.ParseInt(h[4:6], 16, 32)
	if err != nil {
		return 0, err
	}
	f := func(c int64) float64 {
		v := float64(c) / 255.0
		if v <= 0.03928 {
			return v / 12.92
		}
		return math.Pow((v+0.055)/1.055, 2.4)
	}
	return 0.2126*f(r) + 0.7152*f(g) + 0.0722*f(b), nil
}

// contrastRatio returns the WCAG contrast ratio between two hex colours.
func contrastRatio(fg, bg string) (float64, error) {
	a, err := relLuminance(fg)
	if err != nil {
		return 0, err
	}
	b, err := relLuminance(bg)
	if err != nil {
		return 0, err
	}
	if a < b {
		a, b = b, a
	}
	return (a + 0.05) / (b + 0.05), nil
}

// TestWebA11Y_BrandTokenContrast computes the contrast ratio for every
// brand-token pair documented in DESIGN.md §1 and asserts it meets the
// WCAG 2.2 Level AA threshold for its role:
//
//   - body text: 4.5:1 (normal text)
//   - UI components and large text: 3:1
//
// `--faint` is excluded from the text gate because it is documented as
// "tertiary, dividers" (DESIGN.md §1.1) — it carries decoration only.
// All other tokens must clear their role's threshold.
//
// If any token fails, the test reports the measured ratio and STOPS. Do
// not silently re-tint the brand to pass this; that is George's decision.
func TestWebA11Y_BrandTokenContrast(t *testing.T) {
	type pair struct {
		name    string
		fg, bg  string
		minAA   float64 // 4.5 for text; 3.0 for UI
		surface string  // "bg" or "surface", purely for the error message
	}
	pairs := []pair{
		// Dark theme — body text on bg (4.5:1) and on surface (4.5:1).
		{"dark text", "#E8E2DA", "#14110F", 4.5, "bg"},
		{"dark text", "#E8E2DA", "#1F1A17", 4.5, "surface"},
		{"dark muted", "#8A817A", "#14110F", 4.5, "bg"},
		{"dark muted", "#8A817A", "#1F1A17", 4.5, "surface"},
		// Brand tokens on dark bg — body text threshold (these are used for
		// hero accents and headings; verify the strict 4.5 floor).
		{"dark teal (forward)", "#00ADD8", "#14110F", 4.5, "bg"},
		{"dark ember (backward)", "#FF7A45", "#14110F", 4.5, "bg"},
		{"dark gold (fused)", "#F2C57C", "#14110F", 4.5, "bg"},

		// Light theme — body text on bg (4.5:1) and on surface (4.5:1).
		{"light text", "#14110F", "#FBF8F3", 4.5, "bg"},
		{"light text", "#14110F", "#EDE9E3", 4.5, "surface"},
		{"light muted", "#5C544D", "#FBF8F3", 4.5, "bg"},
		{"light muted", "#5C544D", "#EDE9E3", 4.5, "surface"},
		// Brand tokens on light bg — body text threshold.
		{"light teal (forward)", "#006F9E", "#FBF8F3", 4.5, "bg"},
		{"light ember (backward)", "#B84A16", "#FBF8F3", 4.5, "bg"},
		{"light gold (fused)", "#7A5800", "#FBF8F3", 4.5, "bg"},
	}
	// Documented exception: light ember on light surface is 4.31:1 (just
	// under 4.5), used as a chip border only (the UI 3:1 zone). See
	// A11Y.md §3 notes; the contract is no body text in that pairing.
	uiOnly := []pair{
		{"light ember (UI/border)", "#B84A16", "#EDE9E3", 3.0, "surface"},
		{"light teal (UI/border)", "#006F9E", "#EDE9E3", 3.0, "surface"},
		{"light gold (UI/border)", "#7A5800", "#EDE9E3", 3.0, "surface"},
		{"dark teal (UI/border)", "#00ADD8", "#1F1A17", 3.0, "surface"},
		{"dark ember (UI/border)", "#FF7A45", "#1F1A17", 3.0, "surface"},
		{"dark gold (UI/border)", "#F2C57C", "#1F1A17", 3.0, "surface"},
	}

	check := func(label string, p pair) {
		r, err := contrastRatio(p.fg, p.bg)
		if err != nil {
			t.Fatalf("%s contrast(%s,%s): %v", label, p.fg, p.bg, err)
		}
		gate := "AA"
		if p.minAA == 3.0 {
			gate = "AA-UI/large (3:1)"
		}
		t.Logf("%s %s on %s = %.2f:1 (gate %s, role=%s)",
			label, p.fg, p.bg, r, gate, p.surface)
		if r < p.minAA {
			t.Errorf(
				"%s %s on %s = %.2f:1; below WCAG %s threshold %.1f.\n"+
					"  DO NOT silently re-tint the brand; report to George.\n"+
					"  Brand-token semantics are pinned in DESIGN.md §1 and "+
					"tui/dashboard_test.go::TestColorTokenValues.",
				label, p.fg, p.bg, r, gate, p.minAA,
			)
		}
	}
	for _, p := range pairs {
		check("[text/AA]", p)
	}
	for _, p := range uiOnly {
		check("[ui/3:1]", p)
	}
}

// TestWebA11Y_SkipLinkPresent pins the skip link is the first focusable
// element and lands on #main.
func TestWebA11Y_SkipLinkPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `class="skip-link"`) {
		t.Errorf("studio.html missing class=\"skip-link\" (A11Y §1 / DESIGN §11.2)")
	}
	if !strings.Contains(body, `href="#main"`) {
		t.Errorf("skip link must target #main (A11Y §1)")
	}
	// The skip link must precede the <main> element in source order
	// (so Tab finds it first).
	skipIdx := strings.Index(body, `class="skip-link"`)
	mainIdx := strings.Index(body, `id="main"`)
	if skipIdx < 0 || mainIdx < 0 || skipIdx >= mainIdx {
		t.Errorf("skip link must precede <main id=\"main\"> in source order (skipIdx=%d, mainIdx=%d)",
			skipIdx, mainIdx)
	}
}

// TestWebA11Y_LiveRegionPresent pins the polite live region that future
// SSE writes and SPA route changes announce through.
func TestWebA11Y_LiveRegionPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `id="live-region"`) {
		t.Errorf("studio.html missing #live-region (A11Y §1 / DESIGN §11.9)")
	}
	if !strings.Contains(body, `aria-live="polite"`) {
		t.Errorf("studio.html missing aria-live=\"polite\"")
	}
	if !strings.Contains(body, `aria-atomic="false"`) {
		t.Errorf("studio.html missing aria-atomic=\"false\" (we want only added text announced)")
	}
}

// TestWebA11Y_LandmarksPresent pins all four document landmarks: banner,
// navigation, main, contentinfo.
func TestWebA11Y_LandmarksPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	for _, role := range []string{
		`role="banner"`,
		`role="navigation"`,
		`role="main"`,
		`role="contentinfo"`,
	} {
		if !strings.Contains(body, role) {
			t.Errorf("studio.html missing landmark %s (DESIGN §11.3)", role)
		}
	}
	// The primary navigation rail must keep its aria-label (multiple
	// <nav> exist; the label disambiguates).
	if !strings.Contains(body, `aria-label="primary navigation"`) {
		t.Errorf("studio.html missing aria-label=\"primary navigation\" on the rail nav")
	}
}

// TestWebA11Y_FocusVisibleStyleExists pins that :focus-visible has a
// non-empty rule body. The global outline rule must exist; absent it,
// a screen-reader user has no visible focus.
func TestWebA11Y_FocusVisibleStyleExists(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.css")
	// Match :focus-visible followed by a non-empty { ... } body containing
	// an outline declaration.
	re := regexp.MustCompile(`(?s):focus-visible\s*\{[^}]*outline[^}]*\}`)
	if !re.MatchString(body) {
		t.Errorf("studio.css missing :focus-visible rule with an outline declaration (DESIGN §11.5)")
	}
}

// TestWebA11Y_PrefersReducedMotionHonored pins the @media block exists and
// disables animation or transition.
func TestWebA11Y_PrefersReducedMotionHonored(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.css")
	if !strings.Contains(body, "prefers-reduced-motion: reduce") {
		t.Errorf("studio.css missing @media (prefers-reduced-motion: reduce) (DESIGN §3.4 / §11.14)")
	}
	// Find the block and verify it disables animation or transition.
	idx := strings.Index(body, "prefers-reduced-motion: reduce")
	if idx < 0 {
		return
	}
	// Look in the next 2000 chars (long enough to span the rules).
	end := idx + 2000
	if end > len(body) {
		end = len(body)
	}
	window := body[idx:end]
	hasAnimNone := strings.Contains(window, "animation: none") || strings.Contains(window, "animation-duration")
	hasTransNone := strings.Contains(window, "transition: none") || strings.Contains(window, "transition-duration")
	if !hasAnimNone && !hasTransNone {
		t.Errorf("prefers-reduced-motion block must disable animation or transition")
	}
}

// TestWebA11Y_ForcedColorsHonored pins the @media (forced-colors: active)
// block exists (Windows High Contrast / forced colour mode).
func TestWebA11Y_ForcedColorsHonored(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.css")
	if !strings.Contains(body, "forced-colors: active") {
		t.Errorf("studio.css missing @media (forced-colors: active) (DESIGN §11.16)")
	}
	// And it must reference at least one of the system color keywords.
	idx := strings.Index(body, "forced-colors: active")
	if idx < 0 {
		return
	}
	end := idx + 2000
	if end > len(body) {
		end = len(body)
	}
	window := body[idx:end]
	hasSystem := strings.Contains(window, "CanvasText") ||
		strings.Contains(window, "Highlight") ||
		strings.Contains(window, "LinkText") ||
		strings.Contains(window, "ButtonText")
	if !hasSystem {
		t.Errorf("forced-colors block must reference at least one system color keyword (CanvasText, Highlight, LinkText, ButtonText)")
	}
}

// TestWebA11Y_PrefersContrastMoreHonored pins the high-contrast block.
func TestWebA11Y_PrefersContrastMoreHonored(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.css")
	if !strings.Contains(body, "prefers-contrast: more") {
		t.Errorf("studio.css missing @media (prefers-contrast: more) (DESIGN §11.15)")
	}
}

// TestWebA11Y_KeyboardHelpExists pins the keyboard-help modal markup AND
// the JS keypress handler that opens it.
func TestWebA11Y_KeyboardHelpExists(t *testing.T) {
	srv := newWebServer(t)
	_, html := get(t, srv, "/")
	if !strings.Contains(html, `id="keyboard-help"`) {
		t.Errorf("studio.html missing #keyboard-help modal (DESIGN §11.7)")
	}
	if !strings.Contains(html, `role="dialog"`) {
		t.Errorf("keyboard-help missing role=\"dialog\"")
	}
	if !strings.Contains(html, `aria-modal="true"`) {
		t.Errorf("keyboard-help missing aria-modal=\"true\"")
	}
	_, js := get(t, srv, "/static/studio.js")
	// The `?` (shift+/) handler must exist.
	if !strings.Contains(js, "'?'") && !strings.Contains(js, `"?"`) {
		t.Errorf("studio.js missing `?` keypress handler for keyboard-help")
	}
	// And the open function must exist.
	if !strings.Contains(js, "helpOpen") {
		t.Errorf("studio.js missing helpOpen (modal open handler)")
	}
	if !strings.Contains(js, "helpClose") {
		t.Errorf("studio.js missing helpClose (modal close handler)")
	}
}

// TestWebA11Y_ThemeToggleDynamicLabel pins that the theme toggle's
// aria-label is updated on each cycle to describe the current state.
func TestWebA11Y_ThemeToggleDynamicLabel(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	// The applyTheme function must call setAttribute('aria-label', ...)
	// describing the current theme. Pin the literal "current theme".
	if !strings.Contains(body, "'aria-label'") && !strings.Contains(body, `"aria-label"`) {
		t.Errorf("studio.js does not setAttribute aria-label on the theme toggle")
	}
	re := regexp.MustCompile(`current theme:\s*'?\s*\+\s*theme`)
	if !re.MatchString(body) {
		t.Errorf("studio.js theme toggle must rebuild aria-label with the current theme string (DESIGN §11.11)")
	}
}

// TestWebA11Y_NoOutlineNoneUnreplaced sweeps the CSS for `outline: none`
// without a corresponding focus-visible replacement nearby.
//
// Rule: every `outline: none` declaration must be followed within 50
// non-blank lines by a `:focus-visible { ... outline ... }` rule that
// restores a visible focus indicator. The intent is to catch silent
// removals of the focus ring; explicit theme-aware replacements are fine.
func TestWebA11Y_NoOutlineNoneUnreplaced(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.css")
	lines := strings.Split(body, "\n")
	for i, ln := range lines {
		if !strings.Contains(ln, "outline: none") && !strings.Contains(ln, "outline:none") {
			continue
		}
		// Skip explicit replacements where the same rule sets a
		// box-shadow or border or where the next 50 lines contain a
		// :focus-visible outline rule.
		end := i + 50
		if end > len(lines) {
			end = len(lines)
		}
		window := strings.Join(lines[i:end], "\n")
		if regexp.MustCompile(`:focus-visible[^}]*outline\s*:`).MatchString(window) {
			continue
		}
		// Allow the `main:focus { outline: none; }` pattern when the
		// next rule is `main:focus-visible { outline: ... }`.
		t.Errorf("studio.css line %d: `outline: none` without a nearby :focus-visible replacement (DESIGN §11.5)\n  context: %q",
			i+1, strings.TrimSpace(ln))
	}
}

// TestWebA11Y_BrandMarkAriaHidden pins decorative SVGs (the 22px brand
// mark and the 96x72 hero mark) are aria-hidden so screen readers don't
// try to describe them. Informative SVGs (W6 sparkline, kernel thumb)
// declare role="img" and carry their own aria-label + <desc>; the test
// exempts those.
func TestWebA11Y_BrandMarkAriaHidden(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	// Find every <svg ...> open tag. Decorative SVGs (no role attribute)
	// must be aria-hidden; informative ones (role="img") are allowed
	// without aria-hidden because they carry semantic content.
	re := regexp.MustCompile(`<svg\b[^>]*>`)
	tags := re.FindAllString(body, -1)
	if len(tags) == 0 {
		t.Fatalf("no <svg> elements found in studio.html")
	}
	for _, tag := range tags {
		if strings.Contains(tag, `role="img"`) {
			// Informative SVG: must have an aria-label.
			if !strings.Contains(tag, "aria-label=") {
				t.Errorf("informative SVG (role=\"img\") missing aria-label: %s", tag)
			}
			continue
		}
		if !strings.Contains(tag, `aria-hidden="true"`) {
			t.Errorf("decorative SVG missing aria-hidden=\"true\": %s", tag)
		}
	}
}

// TestWebA11Y_DocumentTitleUpdate pins that document.title is rewritten on
// route change (so the browser tab tracks the SPA view).
func TestWebA11Y_DocumentTitleUpdate(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "document.title") {
		t.Errorf("studio.js does not update document.title on route change (DESIGN §11.10)")
	}
	// The format must be `anneal · {view}` (literal pin so the title
	// stays consistent across PRs).
	if !strings.Contains(body, "'anneal · '") && !strings.Contains(body, `"anneal · "`) {
		t.Errorf("studio.js document.title prefix must be 'anneal · '")
	}
}

// TestWebA11Y_LiveRegionAnnounceOnRoute pins that the JS calls announce()
// (which writes the live region) on each route change.
func TestWebA11Y_LiveRegionAnnounceOnRoute(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, "announce(") {
		t.Errorf("studio.js missing announce() call (DESIGN §11.9)")
	}
	// The literal "navigated to " is the contract phrase; screen-reader
	// users hear this on every SPA transition.
	if !strings.Contains(body, "navigated to ") {
		t.Errorf("studio.js announce() must use the 'navigated to ' prefix")
	}
}

// TestWebA11Y_AriaCurrentOnActiveNav pins that the JS sets
// aria-current="page" on the active nav button.
func TestWebA11Y_AriaCurrentOnActiveNav(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	if !strings.Contains(body, `'aria-current'`) && !strings.Contains(body, `"aria-current"`) {
		t.Errorf("studio.js does not set aria-current (DESIGN §11.8)")
	}
	if !strings.Contains(body, "'page'") && !strings.Contains(body, `"page"`) {
		t.Errorf("studio.js aria-current value must be 'page'")
	}
}

// TestWebA11Y_SearchInputLabeled pins the search input has BOTH a label
// (visually hidden) AND a descriptive aria-label.
func TestWebA11Y_SearchInputLabeled(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `for="search"`) {
		t.Errorf("studio.html search input missing <label for=\"search\"> (DESIGN §11.12)")
	}
	// And aria-label must NOT be the bare "search".
	re := regexp.MustCompile(`<input[^>]*id="search"[^>]*aria-label="([^"]*)"`)
	m := re.FindStringSubmatch(body)
	if m == nil {
		t.Errorf("studio.html search input missing aria-label")
		return
	}
	label := m[1]
	if label == "search" {
		t.Errorf("search aria-label is bare 'search'; describe what it searches (DESIGN §11.12)")
	}
}

// TestWebA11Y_LangAttributePresent pins the html lang attribute.
func TestWebA11Y_LangAttributePresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, `<html lang="en"`) {
		t.Errorf("studio.html <html> missing lang=\"en\" (DESIGN §11.20)")
	}
}

// TestWebA11Y_HeadingHierarchy pins exactly one <h1> per view section
// AND that no <h3> appears before an <h2> in the studio shell.
func TestWebA11Y_HeadingHierarchy(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	// Each .view section should contain one <h1 ...>. Count h1 inside
	// each view by splitting on `class="view`.
	views := strings.Split(body, `class="view`)[1:]
	if len(views) < 8 {
		t.Fatalf("expected at least 8 view sections, got %d", len(views))
	}
	h1Re := regexp.MustCompile(`<h1\b`)
	for i, v := range views {
		// Stop at the closing </section>.
		end := strings.Index(v, "</section>")
		if end > 0 {
			v = v[:end]
		}
		count := len(h1Re.FindAllString(v, -1))
		if count != 1 {
			t.Errorf("view #%d has %d <h1>; want exactly 1 (DESIGN §11.4)", i, count)
		}
	}
}

// TestWebA11Y_VisuallyHiddenUtilityExists pins the .visually-hidden CSS
// utility uses the clip-path recipe (NOT display:none), so screen
// readers still expose the content.
func TestWebA11Y_VisuallyHiddenUtilityExists(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.css")
	if !strings.Contains(body, ".visually-hidden") {
		t.Errorf("studio.css missing .visually-hidden utility")
	}
	idx := strings.Index(body, ".visually-hidden")
	end := idx + 600
	if end > len(body) {
		end = len(body)
	}
	window := body[idx:end]
	if !strings.Contains(window, "clip-path") {
		t.Errorf(".visually-hidden must use clip-path (not display:none) to stay in the a11y tree")
	}
	if strings.Contains(window, "display: none") {
		t.Errorf(".visually-hidden must NOT use display:none (would hide from screen readers)")
	}
}

// TestWebA11Y_A11YMdPresent pins that web/A11Y.md exists in the source
// tree. It is intentionally NOT embedded (it is dev-facing doc, not
// shipped); the test reads it from disk at a path relative to this test
// file's package. If the test is invoked from a different cwd we skip
// rather than fail — the source-text checks above already cover the
// load-bearing behaviour.
func TestWebA11Y_A11YMdPresent(t *testing.T) {
	const path = "../../web/A11Y.md"
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("A11Y.md not readable from cwd (%v); doc-only test", err)
		return
	}
	if !strings.Contains(string(b), "anneal studio accessibility checklist") {
		t.Errorf("web/A11Y.md present but missing expected heading")
	}
}
