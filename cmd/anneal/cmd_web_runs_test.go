//go:build !js

// HTTP-surface tests for the W1 /api/runs[/...] endpoint family. The
// library round-trip is exercised in internal/bundle; these tests pin
// the wire shape: status codes, Content-Types, JSON error bodies, and
// path-traversal handling.

package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/internal/bundle"
)

func newRunsServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	root := t.TempDir()
	t.Setenv(bundle.EnvVar, root)
	s := httptest.NewServer(serveMux())
	t.Cleanup(s.Close)
	return s, root
}

func httpGet(t *testing.T, srv *httptest.Server, path string) (*http.Response, []byte) {
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
	return resp, b
}

func TestRunsEndpointEmpty(t *testing.T) {
	srv, _ := newRunsServer(t)
	resp, body := httpGet(t, srv, "/api/runs")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/runs: status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type %q, want application/json", ct)
	}
	// Empty root: must emit `[]`, not `null`.
	trimmed := strings.TrimSpace(string(body))
	if trimmed != "[]" {
		t.Errorf("empty body = %q, want []", trimmed)
	}
}

func TestRunsEndpointListsBundles(t *testing.T) {
	srv, root := newRunsServer(t)

	// Drop two bundles via the writer.
	w1, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w1.SetProvenance("0.0.0-dev", "abc123", "Apple M3", "Metal",
		"hash1", map[string]int64{"B": 16}); err != nil {
		t.Fatalf("SetProvenance: %v", err)
	}
	if err := w1.Finalize(100); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	w2, err := bundle.NewWriter(root, "conv", bundle.KindGenerate)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w2.Finalize(50); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	resp, body := httpGet(t, srv, "/api/runs")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var got []bundle.BundleSummary
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode list: %v (body=%q)", err, body)
	}
	if len(got) != 2 {
		t.Fatalf("list: got %d, want 2", len(got))
	}
	// Both summaries must round-trip the manifest.
	models := map[string]bool{}
	for _, s := range got {
		models[s.Manifest.Model] = true
	}
	if !models["mlp"] || !models["conv"] {
		t.Errorf("missing models in list: %v", models)
	}
}

func TestRunsEndpointManifestByID(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.SetProvenance("0.0.0-dev", "abc123", "Apple M3", "Metal",
		"deadbeef", map[string]int64{"B": 32}); err != nil {
		t.Fatalf("SetProvenance: %v", err)
	}
	if err := w.Finalize(777); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	resp, body := httpGet(t, srv, "/api/runs/"+string(w.BundleID()))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d (body=%s)", resp.StatusCode, body)
	}
	var mf bundle.Manifest
	if err := json.Unmarshal(body, &mf); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if mf.Model != "mlp" || mf.WGSLHash != "deadbeef" || mf.DurationMs != 777 {
		t.Errorf("manifest fields: %+v", mf)
	}
	if mf.SymBinds["B"] != 32 {
		t.Errorf("sym_binds[B] = %d, want 32", mf.SymBinds["B"])
	}
}

func TestRunsEndpointGraphJSON(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatal(err)
	}
	wantGraph := []byte(`{"nodes":[{"id":0}]}`)
	if err := w.WriteGraph(wantGraph); err != nil {
		t.Fatal(err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatal(err)
	}

	resp, body := httpGet(t, srv, "/api/runs/"+string(w.BundleID())+"/graph.json")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type %q", ct)
	}
	if string(body) != string(wantGraph) {
		t.Errorf("graph bytes: got %q, want %q", body, wantGraph)
	}
}

func TestRunsEndpointScheduleJSON(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte(`{"kernels":[]}`)
	if err := w.WriteSchedule(want); err != nil {
		t.Fatal(err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatal(err)
	}
	resp, body := httpGet(t, srv, "/api/runs/"+string(w.BundleID())+"/schedule.json")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if string(body) != string(want) {
		t.Errorf("schedule bytes: got %q, want %q", body, want)
	}
}

func TestRunsEndpointLossCSV(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AppendLoss(bundle.LossRow{Step: 0, Loss: 1.0, WallMs: 10}); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendLoss(bundle.LossRow{Step: 10, Loss: 0.5, WallMs: 200}); err != nil {
		t.Fatal(err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatal(err)
	}
	resp, body := httpGet(t, srv, "/api/runs/"+string(w.BundleID())+"/loss.csv")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/csv") {
		t.Errorf("Content-Type %q, want text/csv", ct)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) != 3 {
		t.Errorf("loss.csv lines: got %d, want 3", len(lines))
	}
	if lines[0] != bundle.LossCSVHeader {
		t.Errorf("header = %q, want %q", lines[0], bundle.LossCSVHeader)
	}
}

func TestRunsEndpointGenerationNDJSON(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "gpt2", bundle.KindGenerate)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AppendGeneration(bundle.GenerationRow{
		Step: 0, TokenID: 42, TokenText: "the",
	}); err != nil {
		t.Fatal(err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatal(err)
	}
	resp, body := httpGet(t, srv, "/api/runs/"+string(w.BundleID())+"/generation.ndjson")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/x-ndjson") {
		t.Errorf("Content-Type %q", ct)
	}
	if !strings.Contains(string(body), `"token_text":"the"`) {
		t.Errorf("body missing token_text: %q", body)
	}
}

func TestRunsEndpointKernel(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatal(err)
	}
	wgsl := "@compute fn k0() { /* hello */ }"
	if err := w.WriteKernel("K0", wgsl); err != nil {
		t.Fatal(err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatal(err)
	}
	resp, body := httpGet(t, srv, "/api/runs/"+string(w.BundleID())+"/kernels/K0.wgsl")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type %q, want text/plain", ct)
	}
	if string(body) != wgsl {
		t.Errorf("WGSL body: got %q, want %q", body, wgsl)
	}
}

func TestRunsEndpoint404ForMissingBundle(t *testing.T) {
	srv, _ := newRunsServer(t)
	// A syntactically-valid bundle id that simply does not exist on disk.
	resp, body := httpGet(t, srv, "/api/runs/20260101T000000Z-missing-abcdef")
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("missing bundle: status %d, want 404 (body=%s)", resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type %q", ct)
	}
	var parsed map[string]string
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Errorf("body not JSON: %v", err)
	}
	if parsed["error"] == "" {
		t.Errorf("missing 'error' field in response: %v", parsed)
	}
}

func TestRunsEndpointRejectsPathTraversal(t *testing.T) {
	srv, _ := newRunsServer(t)
	// Encoded path-traversal in the bundle id. Must be 404 with the JSON
	// error shape, never a 500 or 200 against an unintended path.
	// (A literal "/api/runs/.." is normalized by net/http before reaching
	// the handler - that path collapses to "/api/runs" via a 301 redirect,
	// which is the stdlib's job, not the bundle reader's.)
	for _, p := range []string{
		"/api/runs/..%2Fetc",
		"/api/runs/..%2F..%2Fetc%2Fpasswd",
		"/api/runs/not-a-valid-id",
	} {
		resp, body := httpGet(t, srv, p)
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("GET %s: status %d, want 404 (body=%s)", p, resp.StatusCode, body)
		}
		if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Errorf("GET %s: Content-Type %q", p, ct)
		}
	}
}

func TestRunsEndpointPostStub(t *testing.T) {
	srv, _ := newRunsServer(t)
	resp, err := http.Post(srv.URL+"/api/runs", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNotImplemented {
		t.Errorf("POST /api/runs: status %d, want 501", resp.StatusCode)
	}
}

func TestRunsEndpointUnknownSubpath(t *testing.T) {
	srv, root := newRunsServer(t)
	w, err := bundle.NewWriter(root, "mlp", bundle.KindTrain)
	if err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	resp, _ := httpGet(t, srv, "/api/runs/"+string(w.BundleID())+"/nonsense.txt")
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("unknown subpath: status %d, want 404", resp.StatusCode)
	}
}
