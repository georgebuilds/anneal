package assets

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// cacheSubdir is the project-scoped subtree under the user cache dir. The
// version segment lets us bump cache layout without colliding with old data.
const cacheSubdir = "anneal/v1"

// httpClient is package-scoped so tests can swap it; default timeout is
// generous because gpt2-safetensors is ~550 MB.
var httpClient = &http.Client{Timeout: 30 * time.Minute}

// downloadURLOverride lets tests redirect fetches to httptest servers
// without mutating the Registry. Keyed by asset name.
var downloadURLOverride = map[string]string{}

// Get returns the local filesystem path to the named asset, fetching and
// SHA-verifying it on first use. Subsequent calls (cache hit + SHA match)
// return immediately without touching the network.
//
// When ANNEAL_OFFLINE=1 is set Get never makes a network request and
// returns an error if the asset is not already cached.
func Get(name string) (string, error) {
	a, ok := Registry[name]
	if !ok {
		return "", fmt.Errorf("assets: unknown asset %q", name)
	}
	root, err := cacheRoot()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", fmt.Errorf("assets: mkdir cache %s: %w", root, err)
	}
	dst := filepath.Join(root, a.Name)

	// Cache hit: verify SHA, return path on match. A mismatch deletes the
	// stale file so the next call refetches.
	if ok, err := verifyFile(dst, a.SHA256); err != nil {
		return "", err
	} else if ok {
		return dst, nil
	}

	if os.Getenv("ANNEAL_OFFLINE") == "1" {
		return "", fmt.Errorf("ANNEAL_OFFLINE=1: asset not in cache at %s; fetch manually from %s", dst, a.URL)
	}

	if a.Size > 0 {
		if err := checkDiskSpace(root, a.Size+a.Size/10); err != nil {
			return "", err
		}
	}

	url := a.URL
	if o, ok := downloadURLOverride[a.Name]; ok {
		url = o
	}
	if err := download(url, dst, a); err != nil {
		return "", err
	}
	return dst, nil
}

// cacheRoot resolves the directory that holds cached asset files. The
// ANNEAL_CACHE_DIR env var (which must be absolute) wins outright;
// otherwise we use os.UserCacheDir + "anneal/v1".
func cacheRoot() (string, error) {
	if v := os.Getenv("ANNEAL_CACHE_DIR"); v != "" {
		if !filepath.IsAbs(v) {
			return "", fmt.Errorf("assets: ANNEAL_CACHE_DIR must be absolute, got %q", v)
		}
		return v, nil
	}
	uc, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("assets: locate user cache dir: %w", err)
	}
	return filepath.Join(uc, cacheSubdir), nil
}

// verifyFile returns (true, nil) when dst exists and its SHA-256 matches
// want. On a mismatch it removes dst and returns (false, nil), so the
// caller can refetch. Missing file returns (false, nil).
func verifyFile(dst, want string) (bool, error) {
	f, err := os.Open(dst)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("assets: open cached %s: %w", dst, err)
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, fmt.Errorf("assets: hash cached %s: %w", dst, err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got == strings.ToLower(want) {
		return true, nil
	}
	// Stale cache entry: drop it so the next call redownloads.
	_ = os.Remove(dst)
	return false, nil
}

// download performs an atomic SHA-verified GET of a.URL into dst.
//
// Resumable downloads via Range: are intentionally deferred; they add
// complexity (server must advertise Accept-Ranges, partial hash state must
// be persisted) for a marginal win on a small asset set. Revisit when
// gpt2-safetensors-class assets become common.
func download(url, dst string, a Asset) error {
	tmp := dst + ".tmp"
	// Defense in depth: if a previous run died mid-write, drop the stub
	// before recreating it.
	_ = os.Remove(tmp)

	fmt.Fprintf(os.Stderr, "fetching %s (%s) -> %s\n", a.Name, mb(a.Size), dst)

	resp, err := httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("assets: GET %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("assets: GET %s: status %s", url, resp.Status)
	}

	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("assets: create %s: %w", tmp, err)
	}
	h := sha256.New()
	mw := io.MultiWriter(f, h)

	var src io.Reader = resp.Body
	if isTTY(os.Stderr) {
		src = &progressReader{r: resp.Body, total: a.Size, start: time.Now()}
	}

	n, copyErr := io.Copy(mw, src)
	closeErr := f.Close()
	if isTTY(os.Stderr) {
		fmt.Fprintln(os.Stderr)
	}
	if copyErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("assets: stream %s: %w", url, copyErr)
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("assets: close %s: %w", tmp, closeErr)
	}

	got := hex.EncodeToString(h.Sum(nil))
	want := strings.ToLower(a.SHA256)
	if got != want {
		_ = os.Remove(tmp)
		return fmt.Errorf("assets: SHA-256 mismatch for %s: got %s, want %s (downloaded %d bytes)", a.Name, got, want, n)
	}

	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("assets: rename %s -> %s: %w", tmp, dst, err)
	}
	return nil
}

// mb renders a byte count as a short human-readable string for log lines.
func mb(n int64) string {
	if n <= 0 {
		return "?"
	}
	return fmt.Sprintf("%.1f MB", float64(n)/(1024*1024))
}

// progressReader wraps an io.Reader and emits a \r-rewritten percent +
// MB/s line to stderr at most a few times per second. It is only installed
// when stderr is a TTY; non-TTY callers get a single "fetching" line.
type progressReader struct {
	r        io.Reader
	total    int64
	read     int64
	start    time.Time
	lastEmit time.Time
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.read += int64(n)
	if now := time.Now(); now.Sub(p.lastEmit) > 200*time.Millisecond {
		p.lastEmit = now
		elapsed := now.Sub(p.start).Seconds()
		mbps := 0.0
		if elapsed > 0 {
			mbps = float64(p.read) / (1024 * 1024) / elapsed
		}
		if p.total > 0 {
			pct := 100.0 * float64(p.read) / float64(p.total)
			fmt.Fprintf(os.Stderr, "\r  %5.1f%%  %6.2f MB/s", pct, mbps)
		} else {
			fmt.Fprintf(os.Stderr, "\r  %.1f MB  %6.2f MB/s", float64(p.read)/(1024*1024), mbps)
		}
	}
	return n, err
}

// isTTY reports whether f looks like a terminal. We avoid pulling in
// golang.org/x/term and instead consult the file mode; this is the same
// approach used in cmd/anneal/cmd_train.go.
func isTTY(f *os.File) bool {
	if f == nil {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}
