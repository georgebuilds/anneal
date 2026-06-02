package assets

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// withTestAsset registers a temporary entry in Registry whose URL points at
// srv and whose SHA matches body. The cleanup hook restores Registry and
// the URL override map so tests cannot leak into each other.
func withTestAsset(t *testing.T, name string, body []byte, srv *httptest.Server) Asset {
	t.Helper()
	sum := sha256.Sum256(body)
	a := Asset{
		Name:   name,
		URL:    srv.URL + "/" + name,
		SHA256: hex.EncodeToString(sum[:]),
		Size:   int64(len(body)),
	}
	orig, hadOrig := Registry[name]
	Registry[name] = a
	downloadURLOverride[name] = a.URL
	t.Cleanup(func() {
		if hadOrig {
			Registry[name] = orig
		} else {
			delete(Registry, name)
		}
		delete(downloadURLOverride, name)
	})
	return a
}

// pointCacheAt redirects asset resolution into dir for the duration of the
// test. We exercise the ANNEAL_CACHE_DIR override path in a dedicated test
// rather than calling cacheRoot directly, so the helper is plain.
func pointCacheAt(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("ANNEAL_CACHE_DIR", dir)
	// Make sure no inherited ANNEAL_OFFLINE leaks in.
	t.Setenv("ANNEAL_OFFLINE", "")
}

func TestGet_HappyPath(t *testing.T) {
	body := []byte("hello shakespeare")
	mux := http.NewServeMux()
	mux.HandleFunc("/happy", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	withTestAsset(t, "happy", body, srv)

	got, err := Get("happy")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	want := filepath.Join(dir, "happy")
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	on, err := os.ReadFile(got)
	if err != nil {
		t.Fatalf("readback: %v", err)
	}
	if string(on) != string(body) {
		t.Fatalf("cached bytes = %q, want %q", on, body)
	}
}

func TestGet_SHAMismatch(t *testing.T) {
	expected := []byte("the real bytes")
	wrong := []byte("totally different bytes")
	mux := http.NewServeMux()
	mux.HandleFunc("/mismatch", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(wrong)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	withTestAsset(t, "mismatch", expected, srv) // registry SHA = SHA(expected); server returns wrong

	_, err := Get("mismatch")
	if err == nil {
		t.Fatalf("expected SHA mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Fatalf("error = %v, want SHA-256 mismatch", err)
	}
	// Neither the final path nor the .tmp may remain after a mismatch.
	final := filepath.Join(dir, "mismatch")
	if _, err := os.Stat(final); !os.IsNotExist(err) {
		t.Fatalf("final file unexpectedly exists: %v", err)
	}
	if _, err := os.Stat(final + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf(".tmp file unexpectedly exists: %v", err)
	}
}

func TestGet_CacheHit(t *testing.T) {
	body := []byte("cached content")
	var hits int32
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		http.Error(w, "should not be called", http.StatusInternalServerError)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	a := withTestAsset(t, "cached", body, srv)

	// Prepopulate the cache with the exact bytes the registry pins.
	if err := os.WriteFile(filepath.Join(dir, a.Name), body, 0o644); err != nil {
		t.Fatalf("seed cache: %v", err)
	}

	got, err := Get("cached")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != filepath.Join(dir, a.Name) {
		t.Fatalf("path = %q", got)
	}
	if n := atomic.LoadInt32(&hits); n != 0 {
		t.Fatalf("server hit %d times on cache hit", n)
	}
}

func TestGet_OfflineMode(t *testing.T) {
	body := []byte("never downloaded")
	var hits int32
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	t.Setenv("ANNEAL_OFFLINE", "1")
	a := withTestAsset(t, "offline", body, srv)

	_, err := Get("offline")
	if err == nil {
		t.Fatalf("expected offline error, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "ANNEAL_OFFLINE") {
		t.Fatalf("error missing ANNEAL_OFFLINE: %v", err)
	}
	if !strings.Contains(msg, a.URL) {
		t.Fatalf("error missing URL hint %q: %v", a.URL, err)
	}
	if n := atomic.LoadInt32(&hits); n != 0 {
		t.Fatalf("server hit %d times in offline mode", n)
	}
}

func TestGet_CacheDirOverride(t *testing.T) {
	body := []byte("override target")
	mux := http.NewServeMux()
	mux.HandleFunc("/override", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	withTestAsset(t, "override", body, srv)

	got, err := Get("override")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !strings.HasPrefix(got, dir+string(os.PathSeparator)) {
		t.Fatalf("path %q not under override dir %q", got, dir)
	}
}

func TestGet_AtomicOnTruncatedDownload(t *testing.T) {
	body := []byte("the full expected payload that will never finish arriving")
	mux := http.NewServeMux()
	mux.HandleFunc("/truncated", func(w http.ResponseWriter, r *http.Request) {
		// Advertise a Content-Length the body never reaches so io.Copy
		// surfaces an UnexpectedEOF, then hijack and slam the conn.
		w.Header().Set("Content-Length", "1000000")
		w.WriteHeader(http.StatusOK)
		// Write a small prefix then bail on the connection.
		_, _ = w.Write([]byte("partial"))
		hj, ok := w.(http.Hijacker)
		if !ok {
			return
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			return
		}
		_ = conn.Close()
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	withTestAsset(t, "truncated", body, srv)

	_, err := Get("truncated")
	if err == nil {
		t.Fatalf("expected truncated download to error, got nil")
	}
	final := filepath.Join(dir, "truncated")
	if _, err := os.Stat(final); !os.IsNotExist(err) {
		t.Fatalf("final file unexpectedly present after truncated download: %v", err)
	}
	if _, err := os.Stat(final + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf(".tmp unexpectedly present after truncated download: %v", err)
	}
}
