//go:build !js

// Tests for the W9 doctor view: the /api/device endpoint and the studio's
// two-card markup (native + browser). The browser card's navigator.gpu
// probe runs in-page; this test pins the markup contract and the caveat
// copy, not the in-browser behaviour.
//
// Spec: notes/anneal_web_spec.md §5.8.

package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// withStubDeviceProbe swaps the GPU probe for a deterministic stub so the
// /api/device tests do not require a real GPU adapter (and run fast in CI).
// The cleanup restores the production probe.
func withStubDeviceProbe(t *testing.T, name string, f16 bool, maxBuf uint64, perr error) {
	t.Helper()
	prev := deviceProbeFn
	deviceProbeFn = func() (string, bool, uint64, error) { return name, f16, maxBuf, perr }
	t.Cleanup(func() { deviceProbeFn = prev })
}

// TestAPIDevice_ContentType pins that the device probe responds with
// Content-Type: application/json. The studio's fetch() consumes the body
// as JSON, so this is the wire contract.
func TestAPIDevice_ContentType(t *testing.T) {
	withStubDeviceProbe(t, "stub-adapter", true, 1<<30, nil)
	srv := newWebServer(t)
	resp, _ := get(t, srv, "/api/device")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/device: status %d, want 200", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		t.Errorf("/api/device Content-Type %q, want application/json prefix", ct)
	}
}

// TestAPIDevice_RequiredFields pins the JSON shape: adapter_name, backend,
// os, arch are required. The studio renders one row per field; missing
// fields would leave the doctor card with empty rows.
func TestAPIDevice_RequiredFields(t *testing.T) {
	withStubDeviceProbe(t, "stub-adapter", true, 1<<30, nil)
	srv := newWebServer(t)
	resp, body := get(t, srv, "/api/device")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("parse JSON: %v\nbody: %s", err, body)
	}
	for _, key := range []string{
		"adapter_name", "backend", "os", "arch", "anneal_version",
	} {
		if _, ok := got[key]; !ok {
			t.Errorf("/api/device missing required field %q (got keys: %v)", key, keys(got))
		}
	}
	// Stub adapter name surfaces verbatim.
	if got["adapter_name"] != "stub-adapter" {
		t.Errorf("adapter_name = %v, want stub-adapter", got["adapter_name"])
	}
	if got["anneal_version"] != version {
		t.Errorf("anneal_version = %v, want %q", got["anneal_version"], version)
	}
}

// TestAPIDevice_ProbeFailureSurfaces pins that a GPU probe error still
// returns 200 with a parseable JSON body and the error string in the
// `error` field. The studio renders the error inline; a 500 would leave
// the doctor card stuck on "loading…".
func TestAPIDevice_ProbeFailureSurfaces(t *testing.T) {
	withStubDeviceProbe(t, "", false, 0, &probeErr{"no adapter found"})
	srv := newWebServer(t)
	resp, body := get(t, srv, "/api/device")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200 even on probe failure", resp.StatusCode)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("parse JSON: %v", err)
	}
	if got["error"] != "no adapter found" {
		t.Errorf("error field = %v, want \"no adapter found\"", got["error"])
	}
	// The static fields are still populated so the studio renders SOMETHING.
	if got["backend"] == "" || got["os"] == "" {
		t.Errorf("backend / os should still populate on probe failure: %v", got)
	}
}

// probeErr is a tiny error type so the stub can return a known message.
type probeErr struct{ s string }

func (e *probeErr) Error() string { return e.s }

// TestWebDoctor_StubReplaced pins that the doctor view markup is the W9
// pane (two cards), not the "view: doctor coming soon" stub.
func TestWebDoctor_StubReplaced(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	dStart := strings.Index(body, `id="view-doctor"`)
	if dStart < 0 {
		t.Fatal("doctor view section not found")
	}
	dEnd := strings.Index(body[dStart:], "</section>")
	if dEnd < 0 {
		dEnd = len(body) - dStart
	}
	section := body[dStart : dStart+dEnd]
	if strings.Contains(section, "view: doctor coming soon") {
		t.Errorf("doctor view still contains the W0-W8 stub copy")
	}
	if !strings.Contains(section, `class="doctor-pane"`) {
		t.Errorf("doctor view missing class=\"doctor-pane\" (W9 layout root)")
	}
}

// TestWebDoctor_TwoCardsPresent pins both cards (native + browser) are
// present and that each carries an aria-label distinguishing it.
func TestWebDoctor_TwoCardsPresent(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	for _, label := range []string{
		`aria-label="native"`,
		`aria-label="browser"`,
		`id="native-info"`,
		`id="browser-info"`,
	} {
		if !strings.Contains(body, label) {
			t.Errorf("doctor view missing %q", label)
		}
	}
	if !strings.Contains(body, "navigator.gpu") {
		t.Errorf("doctor view should name navigator.gpu in the browser card heading")
	}
}

// TestWebDoctor_BrowserCardCaveat pins the binding caveat copy per spec
// §5.8: the two adapters are independent enumerations; anneal kernels do
// not run in the browser's WebGPU. This is the user-facing affirmation
// that the doctor view is a diagnostic only.
func TestWebDoctor_BrowserCardCaveat(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/")
	if !strings.Contains(body, "independent enumeration") {
		t.Errorf("doctor view missing the 'independent enumeration' caveat")
	}
	if !strings.Contains(body, "diagnostic only") {
		t.Errorf("doctor view missing the 'diagnostic only' framing")
	}
	if !strings.Contains(body, `class="caveat`) {
		t.Errorf("doctor view caveat should carry class=\"caveat\" (visual hook)")
	}
}

// TestWebDoctor_DeepLinkURL pins that the /d route resolves to the
// doctor view shell. The route table is in studio.js but the server-side
// catch-all serves the same shell for every deep link.
func TestWebDoctor_DeepLinkURL(t *testing.T) {
	srv := newWebServer(t)
	resp, body := get(t, srv, "/d")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /d: status %d, want 200 (catch-all)", resp.StatusCode)
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
		t.Errorf("GET /d: Content-Type %q, want text/html prefix", resp.Header.Get("Content-Type"))
	}
	// The shell must include the doctor section even on a deep link
	// (history-API SPA routing).
	if !strings.Contains(body, `id="view-doctor"`) {
		t.Errorf("GET /d: shell missing the doctor view section")
	}
}

// TestWebDoctor_RendererWired pins that studio.js exports the renderer
// hooks. The dispatch table maps the `doctor` view id to renderDoctorView;
// the two card fillers are exported under __studio for manual driving.
func TestWebDoctor_RendererWired(t *testing.T) {
	srv := newWebServer(t)
	_, body := get(t, srv, "/static/studio.js")
	for _, needle := range []string{
		"renderDoctorView", // the renderer
		"fillNativeCard",   // the native-card filler
		"fillBrowserCard",  // the browser-card filler
		"navigator.gpu",    // the browser-side probe
		"requestAdapter",   // navigator.gpu.requestAdapter()
		"/api/device",      // the native-side fetch
	} {
		if !strings.Contains(body, needle) {
			t.Errorf("studio.js missing %q (doctor wiring not landed)", needle)
		}
	}
}

// keys returns the (unordered) key slice of a string-keyed map for error
// reporting.
func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
