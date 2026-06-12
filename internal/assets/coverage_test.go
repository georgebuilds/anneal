package assets

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestGet_UnknownAsset exercises the registry-miss branch of Get.
func TestGet_UnknownAsset(t *testing.T) {
	_, err := Get("definitely-not-an-asset")
	if err == nil {
		t.Fatal("expected error for unknown asset, got nil")
	}
	if !strings.Contains(err.Error(), "unknown asset") {
		t.Fatalf("error = %v, want 'unknown asset'", err)
	}
}

// TestGet_BadStatus drives the non-200 branch of download.
func TestGet_BadStatus(t *testing.T) {
	body := []byte("payload")
	mux := http.NewServeMux()
	mux.HandleFunc("/bad", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	withTestAsset(t, "bad", body, srv)

	_, err := Get("bad")
	if err == nil {
		t.Fatal("expected error for 404 status, got nil")
	}
	if !strings.Contains(err.Error(), "status") {
		t.Fatalf("error = %v, want 'status'", err)
	}
}

// TestGet_StaleCacheRefetch seeds the cache with wrong bytes; verifyFile must
// drop the stale file (returning false) and Get must refetch the correct ones.
func TestGet_StaleCacheRefetch(t *testing.T) {
	body := []byte("the canonical bytes")
	mux := http.NewServeMux()
	mux.HandleFunc("/stale", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	a := withTestAsset(t, "stale", body, srv)

	// Seed with bytes that do not match the registry SHA.
	if err := os.WriteFile(filepath.Join(dir, a.Name), []byte("garbage"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}

	got, err := Get("stale")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	on, err := os.ReadFile(got)
	if err != nil {
		t.Fatalf("readback: %v", err)
	}
	if string(on) != string(body) {
		t.Fatalf("refetched bytes = %q, want %q", on, body)
	}
}

// TestCacheRoot_NonAbsoluteOverride rejects a relative ANNEAL_CACHE_DIR.
func TestCacheRoot_NonAbsoluteOverride(t *testing.T) {
	t.Setenv("ANNEAL_CACHE_DIR", "relative/path")
	_, err := cacheRoot()
	if err == nil {
		t.Fatal("expected error for non-absolute cache dir, got nil")
	}
	if !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("error = %v, want 'absolute'", err)
	}
}

// TestCacheRoot_DefaultUserCacheDir exercises the os.UserCacheDir fallback.
func TestCacheRoot_DefaultUserCacheDir(t *testing.T) {
	t.Setenv("ANNEAL_CACHE_DIR", "")
	root, err := cacheRoot()
	if err != nil {
		t.Fatalf("cacheRoot: %v", err)
	}
	if !strings.HasSuffix(filepath.ToSlash(root), cacheSubdir) {
		t.Fatalf("root %q does not end with %q", root, cacheSubdir)
	}
	if !filepath.IsAbs(root) {
		t.Fatalf("default cache root %q is not absolute", root)
	}
}

// TestVerifyFile_Missing returns (false, nil) for an absent file.
func TestVerifyFile_Missing(t *testing.T) {
	ok, err := verifyFile(filepath.Join(t.TempDir(), "nope"), "deadbeef")
	if err != nil {
		t.Fatalf("verifyFile: %v", err)
	}
	if ok {
		t.Fatal("verifyFile reported a missing file as present")
	}
}

// TestVerifyFile_Match returns (true, nil) for a hash hit and tolerates an
// uppercase want (verifyFile lower-cases internally).
func TestVerifyFile_Match(t *testing.T) {
	dir := t.TempDir()
	body := []byte("verify me")
	dst := filepath.Join(dir, "blob")
	if err := os.WriteFile(dst, body, 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	hexSum := hex.EncodeToString(sum[:])

	ok, err := verifyFile(dst, hexSum)
	if err != nil {
		t.Fatalf("verifyFile: %v", err)
	}
	if !ok {
		t.Fatal("verifyFile rejected a matching file")
	}
	// Uppercase want must also match (ToLower path).
	okUpper, err := verifyFile(dst, strings.ToUpper(hexSum))
	if err != nil {
		t.Fatalf("verifyFile upper: %v", err)
	}
	if !okUpper {
		t.Fatal("verifyFile rejected an uppercase-hash match")
	}
}

// TestMB covers the human-readable byte renderer branches.
func TestMB(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{0, "?"},
		{-1, "?"},
		{1024 * 1024, "1.0 MB"},
		{5 * 1024 * 1024, "5.0 MB"},
	}
	for _, c := range cases {
		if got := mb(c.in); got != c.want {
			t.Errorf("mb(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestIsTTY covers the nil and non-terminal branches. A regular file is not a
// character device, so isTTY must report false.
func TestIsTTY(t *testing.T) {
	if isTTY(nil) {
		t.Error("isTTY(nil) = true, want false")
	}
	f, err := os.CreateTemp(t.TempDir(), "tty")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if isTTY(f) {
		t.Error("isTTY(regular file) = true, want false")
	}
}

// TestCheckDiskSpace covers both the satisfied and insufficient branches.
func TestCheckDiskSpace(t *testing.T) {
	dir := t.TempDir()
	// A 1-byte need is trivially satisfiable on any real filesystem.
	if err := checkDiskSpace(dir, 1); err != nil {
		t.Fatalf("checkDiskSpace(1 byte): %v", err)
	}
	// An absurd need must trip the insufficient-space error.
	const absurd = int64(1) << 62
	if err := checkDiskSpace(dir, absurd); err == nil {
		t.Fatal("expected insufficient-space error for 2^62 bytes")
	} else if !strings.Contains(err.Error(), "insufficient disk space") {
		t.Fatalf("error = %v, want 'insufficient disk space'", err)
	}
}

// TestProgressReader drives the progressReader path used during TTY downloads.
// We force an old lastEmit so the throttled write branch runs.
func TestProgressReader(t *testing.T) {
	src := bytes.NewReader(make([]byte, 4096))
	pr := &progressReader{
		r:        src,
		total:    4096,
		start:    time.Now().Add(-time.Second),
		lastEmit: time.Now().Add(-time.Second),
	}
	buf := make([]byte, 1024)
	n, err := pr.Read(buf)
	if err != nil {
		t.Fatalf("progressReader.Read: %v", err)
	}
	if n != 1024 {
		t.Fatalf("read %d bytes, want 1024", n)
	}

	// total == 0 takes the alternate (MB count) formatting branch.
	src2 := bytes.NewReader(make([]byte, 1024))
	pr2 := &progressReader{
		r:        src2,
		total:    0,
		start:    time.Now().Add(-time.Second),
		lastEmit: time.Now().Add(-time.Second),
	}
	if _, err := pr2.Read(buf); err != nil {
		t.Fatalf("progressReader.Read (total=0): %v", err)
	}
}
