package bundle

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenBundleRejectsUnsupportedVersion(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "verbump", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	path := w.Path()
	_ = w.Close()
	// Rewrite manifest with bundle_version=99.
	bad := `{
  "bundle_version": 99,
  "kind": "train",
  "model": "verbump",
  "anneal_version": "",
  "git_rev": "",
  "device_name": "",
  "adapter": "",
  "wgsl_hash": "",
  "sym_binds": {},
  "created_at": "2026-01-01T00:00:00Z",
  "duration_ms": 0
}`
	if err := os.WriteFile(filepath.Join(path, "manifest.json"), []byte(bad), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err = OpenBundle(path)
	if err == nil {
		t.Fatalf("OpenBundle: want error for bundle_version=99")
	}
	msg := err.Error()
	if !strings.Contains(msg, "version") || !strings.Contains(msg, "99") {
		t.Errorf("error %q must mention 'version' and '99'", msg)
	}
}

func TestOpenBundleRejectsMalformedManifest(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "malformed", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	path := w.Path()
	_ = w.Close()
	if err := os.WriteFile(filepath.Join(path, "manifest.json"), []byte("not json"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := OpenBundle(path); err == nil {
		t.Errorf("OpenBundle malformed: want error")
	}
}

func TestOpenBundleRejectsInvalidName(t *testing.T) {
	root := t.TempDir()
	bad := filepath.Join(root, "not-a-bundle")
	if err := os.Mkdir(bad, 0o755); err != nil {
		t.Fatalf("Mkdir: %v", err)
	}
	if _, err := OpenBundle(bad); err == nil {
		t.Errorf("OpenBundle with invalid name: want error")
	}
}

func TestOpenBundleInRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	for _, bad := range []string{
		"../etc",
		"..",
		"../../etc/passwd",
		".",
		"",
		"/absolute",
		"valid-looking-but-not/../escape",
	} {
		if _, err := OpenBundleIn(root, bad); err == nil {
			t.Errorf("OpenBundleIn(%q): want error", bad)
		}
	}
}

func TestOpenBundleInResolves(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "ok", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundleIn(root, string(w.BundleID()))
	if err != nil {
		t.Fatalf("OpenBundleIn: %v", err)
	}
	if r.BundleID() != w.BundleID() {
		t.Errorf("BundleID round-trip mismatch")
	}
}

func TestOpenBundleNonexistent(t *testing.T) {
	root := t.TempDir()
	missing := filepath.Join(root, "20260101T000000Z-mlp-abcdef")
	if _, err := OpenBundle(missing); err == nil {
		t.Errorf("OpenBundle missing: want error")
	}
}

func TestKernelRejectsBadNames(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "kbad", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	_ = w.WriteKernel("K0", "wgsl")
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	for _, bad := range []string{"", "../escape", "foo/bar", "a..b"} {
		if _, err := r.Kernel(bad); err == nil {
			t.Errorf("Kernel(%q): want error", bad)
		}
	}
}

func TestReaderEmptyForMissingFiles(t *testing.T) {
	// A bundle with only manifest and graph should still return empty
	// slices for loss/generation/events/kernels - no errors.
	root := t.TempDir()
	w, err := NewWriter(root, "sparse", KindSaved)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.WriteGraph([]byte(`{}`)); err != nil {
		t.Fatalf("WriteGraph: %v", err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if got, _ := r.Loss(); len(got) != 0 {
		t.Errorf("Loss on sparse: want empty, got %d rows", len(got))
	}
	if got, _ := r.Generation(); len(got) != 0 {
		t.Errorf("Generation on sparse: want empty, got %d rows", len(got))
	}
	if got, _ := r.Events(); len(got) != 0 {
		t.Errorf("Events on sparse: want empty, got %d rows", len(got))
	}
	if got, _ := r.KernelNames(); len(got) != 0 {
		t.Errorf("KernelNames on sparse: want empty, got %v", got)
	}
}

func TestReaderLossMalformedHeader(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "badcsv", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	// Write a CSV with a wrong header directly.
	if err := os.WriteFile(filepath.Join(w.Path(), "loss.csv"),
		[]byte("wrong,header,here\n1,1.0,1\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Loss(); err == nil {
		t.Errorf("Loss with bad header: want error")
	}
}

func TestReaderGenerationMalformed(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "badgen", KindGenerate)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(w.Path(), "generation.ndjson"),
		[]byte("not json\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Generation(); err == nil {
		t.Errorf("Generation with bad json: want error")
	}
}

func TestIsValidBundleID(t *testing.T) {
	good := []string{
		"20260101T000000Z-mlp-abcdef",
		"20991231T235959Z-my-model-name-012345",
		"20260102T030405Z-m1-deadbe",
	}
	for _, s := range good {
		if !IsValidBundleID(s) {
			t.Errorf("IsValidBundleID(%q) = false, want true", s)
		}
	}
	bad := []string{
		"",
		"foo",
		"../etc/passwd",
		"20260101T000000Z-Mlp-abcdef",  // uppercase model
		"20260101T000000Z--abcdef",     // empty model
		"20260101T000000-mlp-abcdef",   // missing Z
		"20260101T000000Z-mlp-abcXYZ",  // non-hex shorthash
		"20260101T000000Z-mlp-abcd",    // shorthash too short
		"20260101T000000Z-mlp-abcdef0", // shorthash too long
	}
	for _, s := range bad {
		if IsValidBundleID(s) {
			t.Errorf("IsValidBundleID(%q) = true, want false", s)
		}
	}
}

func TestDefaultRootCreatesDir(t *testing.T) {
	// Override XDG_CACHE_HOME so we don't pollute the real user cache.
	tmp := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", tmp)
	t.Setenv(EnvVar, "")
	// On darwin os.UserCacheDir uses ~/Library/Caches; honor HOME there.
	t.Setenv("HOME", tmp)
	root, err := DefaultRoot()
	if err != nil {
		t.Fatalf("DefaultRoot: %v", err)
	}
	if st, err := os.Stat(root); err != nil || !st.IsDir() {
		t.Errorf("DefaultRoot did not create %q: %v", root, err)
	}
}

func TestEnvOrDefault(t *testing.T) {
	tmp := t.TempDir()
	custom := filepath.Join(tmp, "custom")
	t.Setenv(EnvVar, custom)
	root, err := EnvOrDefault()
	if err != nil {
		t.Fatalf("EnvOrDefault: %v", err)
	}
	if root != custom {
		t.Errorf("EnvOrDefault = %q, want %q", root, custom)
	}
	if st, err := os.Stat(custom); err != nil || !st.IsDir() {
		t.Errorf("EnvOrDefault did not create %q", custom)
	}
}

func TestEnvOrDefaultRejectsRelative(t *testing.T) {
	t.Setenv(EnvVar, "relative/path")
	if _, err := EnvOrDefault(); err == nil {
		t.Errorf("EnvOrDefault with relative path: want error")
	}
}

func TestKernelNamesFiltersNonWGSL(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "knfilter", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.WriteKernel("K0", "wgsl0"); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteKernel("K1", "wgsl1"); err != nil {
		t.Fatal(err)
	}
	// Drop in a stray non-wgsl file and a subdirectory inside kernels/.
	if err := os.WriteFile(filepath.Join(w.Path(), "kernels", "README.md"),
		[]byte("ignore"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(w.Path(), "kernels", "subdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	names, err := r.KernelNames()
	if err != nil {
		t.Fatalf("KernelNames: %v", err)
	}
	if len(names) != 2 {
		t.Errorf("KernelNames filtered: got %v, want [K0.wgsl K1.wgsl]", names)
	}
}

func TestBundleKindStringUnknown(t *testing.T) {
	k := BundleKind(99)
	if k.String() != "unknown" {
		t.Errorf("unknown kind.String() = %q, want 'unknown'", k.String())
	}
}

func TestReaderConfigMissing(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "nocfg", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Config(); err == nil {
		t.Errorf("Config when missing: want error")
	}
	if _, err := r.Graph(); err == nil {
		t.Errorf("Graph when missing: want error")
	}
	if _, err := r.Schedule(); err == nil {
		t.Errorf("Schedule when missing: want error")
	}
}
