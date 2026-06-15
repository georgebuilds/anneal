//go:build !js

package viz

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	_ "github.com/georgebuilds/anneal/examples" // register mlp/conv
)

// TestStaticFS verifies the embedded static subtree resolves and exposes the
// SPA entrypoint.
func TestStaticFS(t *testing.T) {
	fsys, err := StaticFS()
	if err != nil {
		t.Fatalf("StaticFS: %v", err)
	}
	f, err := fsys.Open("index.html")
	if err != nil {
		t.Fatalf("open index.html from StaticFS: %v", err)
	}
	_ = f.Close()
}

// freePort grabs an ephemeral port from the OS, then releases it so Serve can
// bind it. The brief window between close and re-bind is acceptable for a test.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close()
	return port
}

// TestServeEndpoints starts Serve on an ephemeral port and exercises the
// static handler plus both API endpoints (success and error paths). Serve
// blocks forever, so it runs in a background goroutine the test does not join.
func TestServeEndpoints(t *testing.T) {
	port := freePort(t)
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	base := "http://" + addr

	go func() { _ = Serve(addr) }()

	// Wait for the listener to come up.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		c, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			_ = c.Close()
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	client := &http.Client{Timeout: 5 * time.Second}

	// /api/graph default name (mlp) -> 200 JSON with nodes.
	resp, err := client.Get(base + "/api/graph")
	if err != nil {
		t.Fatalf("GET /api/graph: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/api/graph status = %d, want 200", resp.StatusCode)
	}
	var g map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&g); err != nil {
		t.Fatalf("decode graph: %v", err)
	}
	_ = resp.Body.Close()
	if _, ok := g["nodes"]; !ok {
		t.Errorf("/api/graph response missing 'nodes': %v", g)
	}

	// /api/graph with an unknown model -> 400 with an error field.
	resp, err = client.Get(base + "/api/graph?name=definitely-not-a-model")
	if err != nil {
		t.Fatalf("GET /api/graph bad name: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("/api/graph bad-name status = %d, want 400", resp.StatusCode)
	}
	var ge map[string]string
	_ = json.NewDecoder(resp.Body).Decode(&ge)
	_ = resp.Body.Close()
	if ge["error"] == "" {
		t.Error("/api/graph bad name missing 'error' field")
	}

	// /api/timeline default name (mlp) -> 200.
	resp, err = client.Get(base + "/api/timeline")
	if err != nil {
		t.Fatalf("GET /api/timeline: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/api/timeline status = %d, want 200", resp.StatusCode)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	// /api/timeline with an unknown model -> 400.
	resp, err = client.Get(base + "/api/timeline?name=definitely-not-a-model")
	if err != nil {
		t.Fatalf("GET /api/timeline bad name: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("/api/timeline bad-name status = %d, want 400", resp.StatusCode)
	}
	_ = resp.Body.Close()

	// Static handler: GET / should serve index.html (200).
	resp, err = client.Get(base + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET / status = %d, want 200", resp.StatusCode)
	}
	_ = resp.Body.Close()
}
