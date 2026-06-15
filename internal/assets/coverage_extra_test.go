package assets

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGet_MkdirCacheRootIsFile exercises the os.MkdirAll failure branch in Get
// by pointing the cache root at an existing regular file, which MkdirAll cannot
// turn into a directory.
func TestGet_MkdirCacheRootIsFile(t *testing.T) {
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	// Create a plain file and use it as the cache root.
	fileAsRoot := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(fileAsRoot, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	pointCacheAt(t, fileAsRoot)
	withTestAsset(t, "mkdirfail", body, srv)

	_, err := Get("mkdirfail")
	if err == nil {
		t.Fatal("expected mkdir error when cache root is a file, got nil")
	}
	if !strings.Contains(err.Error(), "mkdir cache") {
		t.Fatalf("error = %v, want 'mkdir cache'", err)
	}
}

// TestGet_VerifyFileErrorOnDirectory drives the verifyFile error branch in Get:
// when the cached path is a directory, os.Open succeeds but io.Copy fails with
// an EISDIR-class error, which propagates out of Get.
func TestGet_VerifyFileErrorOnDirectory(t *testing.T) {
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	a := withTestAsset(t, "dirasset", body, srv)

	// Make the destination path a directory so verifyFile's io.Copy hashing
	// fails (reading a directory as a stream returns an error on most OSes).
	if err := os.Mkdir(filepath.Join(dir, a.Name), 0o755); err != nil {
		t.Fatal(err)
	}

	_, err := Get("dirasset")
	if err == nil {
		t.Fatal("expected verifyFile error when cached path is a directory, got nil")
	}
	if !strings.Contains(err.Error(), "hash cached") {
		t.Fatalf("error = %v, want 'hash cached'", err)
	}
}

// TestGet_InsufficientDiskSpace drives the checkDiskSpace error branch reached
// from Get by registering an asset whose Size exceeds any real free space.
func TestGet_InsufficientDiskSpace(t *testing.T) {
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	pointCacheAt(t, dir)
	a := withTestAsset(t, "huge", body, srv)
	// Override the registered size to an absurd value (the helper pinned the
	// real small size); the +10% headroom math then trips checkDiskSpace.
	a.Size = int64(1) << 62
	Registry["huge"] = a

	_, err := Get("huge")
	if err == nil {
		t.Fatal("expected insufficient-disk-space error, got nil")
	}
	if !strings.Contains(err.Error(), "insufficient disk space") {
		t.Fatalf("error = %v, want 'insufficient disk space'", err)
	}
}

// TestCacheRoot_UserCacheDirError forces os.UserCacheDir to fail by clearing
// the env vars it consults (HOME and XDG_CACHE_HOME on unix). This exercises
// both cacheRoot's error path and Get's propagation of it.
func TestCacheRoot_UserCacheDirError(t *testing.T) {
	t.Setenv("ANNEAL_CACHE_DIR", "")
	t.Setenv("XDG_CACHE_HOME", "")
	t.Setenv("HOME", "")
	// macOS UserCacheDir also consults this; clear for completeness.
	t.Setenv("home", "")

	if _, err := cacheRoot(); err == nil {
		t.Skip("UserCacheDir still resolved on this platform; cannot force error")
	} else if !strings.Contains(err.Error(), "user cache dir") {
		t.Fatalf("error = %v, want 'user cache dir'", err)
	}

	// And Get must surface the same failure.
	if _, err := Get("shakespeare"); err == nil {
		t.Fatal("expected Get to fail when UserCacheDir errors")
	}
}

// TestVerifyFile_OpenErrorNonExist drives verifyFile's non-ErrNotExist open
// error: a path whose parent is a regular file yields ENOTDIR rather than
// ENOENT, which must propagate as an error.
func TestVerifyFile_OpenErrorNonExist(t *testing.T) {
	dir := t.TempDir()
	notDir := filepath.Join(dir, "afile")
	if err := os.WriteFile(notDir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	// notDir/child treats a file as a directory component -> ENOTDIR.
	ok, err := verifyFile(filepath.Join(notDir, "child"), "deadbeef")
	if err == nil {
		t.Skip("filesystem did not return a non-ENOENT open error for nested path")
	}
	if ok {
		t.Fatal("verifyFile returned ok=true on open error")
	}
	if !strings.Contains(err.Error(), "open cached") {
		t.Fatalf("error = %v, want 'open cached'", err)
	}
}

// TestGet_CreateTmpError drives download's os.Create(tmp) failure by making the
// cache directory read-only so the temp file cannot be created.
func TestGet_CreateTmpError(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: read-only directory perms are not enforced")
	}
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	// Pre-create the cache root, then strip write permission. Get's own
	// MkdirAll on an existing dir is a no-op regardless of perms, so the
	// failure surfaces at os.Create(tmp) inside download.
	if err := os.Chmod(dir, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o755) })
	pointCacheAt(t, dir)
	withTestAsset(t, "rotmp", body, srv)

	_, err := Get("rotmp")
	if err == nil {
		t.Fatal("expected create-tmp error on read-only cache dir, got nil")
	}
	if !strings.Contains(err.Error(), "create") {
		t.Fatalf("error = %v, want 'create'", err)
	}
}

// TestGet_NetworkError drives the httpClient.Get failure branch in download by
// pointing the override URL at a closed server.
func TestGet_NetworkError(t *testing.T) {
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	url := srv.URL
	srv.Close() // close immediately so the GET fails to connect

	dir := t.TempDir()
	pointCacheAt(t, dir)
	a := withTestAsset(t, "neterr", body, &httptest.Server{URL: url})
	downloadURLOverride[a.Name] = url + "/neterr"

	_, err := Get("neterr")
	if err == nil {
		t.Fatal("expected network error from closed server, got nil")
	}
	if !strings.Contains(err.Error(), "GET") {
		t.Fatalf("error = %v, want 'GET'", err)
	}
}
