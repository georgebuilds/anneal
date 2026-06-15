package bundle

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newFinalizedBundle creates a minimal finalized bundle and returns its Reader.
func newFinalizedBundle(t *testing.T, model string) (*Writer, *Reader) {
	t.Helper()
	root := t.TempDir()
	w, err := NewWriter(root, model, KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	return w, nil
}

// TestReaderConfigMalformed drives Config's json.Unmarshal error branch.
func TestReaderConfigMalformed(t *testing.T) {
	w, _ := newFinalizedBundle(t, "badconfig")
	if err := os.WriteFile(filepath.Join(w.Path(), "config.json"),
		[]byte("{ this is not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Config(); err == nil {
		t.Fatal("Config with malformed JSON: want error")
	} else if !strings.Contains(err.Error(), "parse config.json") {
		t.Fatalf("error = %v, want 'parse config.json'", err)
	}
}

// TestKernelNamesSkipsSubdirsAndNonWGSL exercises the IsDir() and non-".wgsl"
// continue branches of KernelNames.
func TestKernelNamesSkipsSubdirsAndNonWGSL(t *testing.T) {
	w, _ := newFinalizedBundle(t, "kfilter")
	kdir := filepath.Join(w.Path(), "kernels")
	// A real kernel, a non-wgsl file, and a nested subdirectory.
	if err := os.WriteFile(filepath.Join(kdir, "K0.wgsl"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(kdir, "README.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(kdir, "subdir"), 0o755); err != nil {
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
	if len(names) != 1 || names[0] != "K0.wgsl" {
		t.Fatalf("KernelNames = %v, want [K0.wgsl]", names)
	}
}

// TestKernelReadMissingFile drives Kernel's os.ReadFile error branch and the
// implicit ".wgsl" suffix append.
func TestKernelReadMissingFile(t *testing.T) {
	w, _ := newFinalizedBundle(t, "kmissing")
	_ = w.WriteKernel("K0", "wgsl text")
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	// Existing kernel resolved without explicit suffix.
	if got, err := r.Kernel("K0"); err != nil || got != "wgsl text" {
		t.Fatalf("Kernel(K0) = %q, %v", got, err)
	}
	// Missing kernel surfaces a read error.
	if _, err := r.Kernel("K99"); err == nil {
		t.Fatal("Kernel(K99): want read error")
	} else if !strings.Contains(err.Error(), "read kernel") {
		t.Fatalf("error = %v, want 'read kernel'", err)
	}
}

// TestLossSkipsBlankAndRejectsBadRow covers Loss's blank-line skip and the
// ParseLossRow error path on a malformed data row.
func TestLossSkipsBlankAndRejectsBadRow(t *testing.T) {
	w, _ := newFinalizedBundle(t, "lossrows")
	// Valid header, a blank line (skipped), then a malformed row.
	content := LossCSVHeader + "\n\nnot,a,valid,loss,row,with,extra\n"
	if err := os.WriteFile(filepath.Join(w.Path(), "loss.csv"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Loss(); err == nil {
		t.Fatal("Loss with malformed row: want error")
	} else if !strings.Contains(err.Error(), "loss.csv line") {
		t.Fatalf("error = %v, want 'loss.csv line'", err)
	}
}

// TestLossSkipsBlankValid verifies a blank trailing line does not corrupt a
// valid parse (the blank-line continue keeps the row count correct).
func TestLossSkipsBlankValid(t *testing.T) {
	w, _ := newFinalizedBundle(t, "lossblank")
	if err := w.AppendLoss(LossRow{Step: 1, Loss: 0.5, WallMs: 10}); err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	// Append a trailing blank line.
	f, err := os.OpenFile(filepath.Join(w.Path(), "loss.csv"), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString("\n")
	_ = f.Close()

	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	rows, err := r.Loss()
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("Loss rows = %d, want 1", len(rows))
	}
}

// TestGenerationBlankAndMalformed covers Generation's blank-line skip and the
// json.Unmarshal error path.
func TestGenerationBlankAndMalformed(t *testing.T) {
	w, _ := newFinalizedBundle(t, "genrows")
	content := "\n{not valid json}\n"
	if err := os.WriteFile(filepath.Join(w.Path(), "generation.ndjson"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Generation(); err == nil {
		t.Fatal("Generation with malformed line: want error")
	} else if !strings.Contains(err.Error(), "generation.ndjson line") {
		t.Fatalf("error = %v, want 'generation.ndjson line'", err)
	}
}

// TestEventsBlankAndMalformed covers Events's blank-line skip and the
// json.Unmarshal error path.
func TestEventsBlankAndMalformed(t *testing.T) {
	w, _ := newFinalizedBundle(t, "evtrows")
	content := "\n{not valid json}\n"
	if err := os.WriteFile(filepath.Join(w.Path(), "events.ndjson"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.Events(); err == nil {
		t.Fatal("Events with malformed line: want error")
	} else if !strings.Contains(err.Error(), "events.ndjson line") {
		t.Fatalf("error = %v, want 'events.ndjson line'", err)
	}
}

// TestLoadManifestNilSymBindsReset verifies loadManifest resets a nil SymBinds
// (a manifest written without the sym_binds field) to an empty map.
func TestLoadManifestNilSymBindsReset(t *testing.T) {
	root := t.TempDir()
	id := "20260101T000000Z-symnil-abcdef"
	dir := filepath.Join(root, id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Manifest JSON omitting sym_binds entirely.
	mjson := `{"bundle_version":1,"kind":"train","model":"symnil"}`
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(mjson), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := OpenBundle(dir)
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if r.Manifest().SymBinds == nil {
		t.Fatal("SymBinds is nil, want empty map")
	}
	if len(r.Manifest().SymBinds) != 0 {
		t.Fatalf("SymBinds = %v, want empty", r.Manifest().SymBinds)
	}
}

// TestListBundlesReadDirError drives the non-ENOENT ReadDir error branch by
// passing a regular file as rootDir.
func TestListBundlesReadDirError(t *testing.T) {
	f := filepath.Join(t.TempDir(), "afile")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ListBundles(f); err == nil {
		t.Fatal("ListBundles on a file: want error")
	} else if !strings.Contains(err.Error(), "read") {
		t.Fatalf("error = %v, want 'read'", err)
	}
}

// TestEnvOrDefaultAbsolute drives the absolute-env-var branch of EnvOrDefault.
func TestEnvOrDefaultAbsolute(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "runs-here")
	t.Setenv(EnvVar, dir)
	got, err := EnvOrDefault()
	if err != nil {
		t.Fatalf("EnvOrDefault: %v", err)
	}
	if got != dir {
		t.Fatalf("EnvOrDefault = %q, want %q", got, dir)
	}
	// The directory must have been created.
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		t.Fatalf("expected created dir at %q: %v", dir, err)
	}
}

// TestSetProvenanceNilSymBinds covers SetProvenance's nil-symBinds reset.
func TestSetProvenanceNilSymBinds(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "prov", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.SetProvenance("v1", "rev", "dev", "adapter", "hash", nil); err != nil {
		t.Fatalf("SetProvenance: %v", err)
	}
	_ = w.Finalize(0)
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	m := r.Manifest()
	if m.SymBinds == nil {
		t.Fatal("SymBinds nil after SetProvenance(nil)")
	}
	if m.AnnealVersion != "v1" || m.GitRev != "rev" {
		t.Fatalf("provenance not persisted: %+v", m)
	}
}

// TestSanitizeModelEdgeCases drives sanitizeModel's empty-input and
// leading/trailing-dash-strip branches directly.
func TestSanitizeModelEdgeCases(t *testing.T) {
	cases := []struct{ in, want string }{
		{"", "model"},
		{"---", "model"}, // fully-stripped -> "model"
		{"!!!", "model"}, // all-invalid -> all-dash -> stripped -> "model"
		{"-foo-", "foo"}, // leading/trailing dash stripped
		{"Foo Bar", "foo-bar"},
		{"a/b\\c", "a-b-c"},
		{"GPT2", "gpt2"},
	}
	for _, c := range cases {
		if got := sanitizeModel(c.in); got != c.want {
			t.Errorf("sanitizeModel(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestEnvOrDefaultRelativeRejected drives EnvOrDefault's non-absolute-path
// rejection branch.
func TestEnvOrDefaultRelativeRejected(t *testing.T) {
	t.Setenv(EnvVar, "relative/path")
	if _, err := EnvOrDefault(); err == nil {
		t.Fatal("EnvOrDefault with relative path: want error")
	} else if !strings.Contains(err.Error(), "must be absolute") {
		t.Fatalf("error = %v, want 'must be absolute'", err)
	}
}

// TestEnvOrDefaultMkdirError drives EnvOrDefault's MkdirAll failure by pointing
// the env var at a path whose parent is a regular file.
func TestEnvOrDefaultMkdirError(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "afile")
	if err := os.WriteFile(parent, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvVar, filepath.Join(parent, "child"))
	if _, err := EnvOrDefault(); err == nil {
		t.Fatal("EnvOrDefault with unmakeable dir: want error")
	} else if !strings.Contains(err.Error(), "create") {
		t.Fatalf("error = %v, want 'create'", err)
	}
}

// TestAppendOpenErrors drives the os.OpenFile failure branch in AppendLoss,
// AppendGeneration, and AppendEvent by pre-creating directories where the
// stream files would go, so OpenFile cannot open them as regular files.
func TestAppendOpenErrors(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "openerr", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	// Pre-create directories at the stream file paths.
	for _, name := range []string{"loss.csv", "generation.ndjson", "events.ndjson"} {
		if err := os.Mkdir(filepath.Join(w.Path(), name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.AppendLoss(LossRow{Step: 1, Loss: 1, WallMs: 1}); err == nil {
		t.Error("AppendLoss into a directory: want error")
	}
	if err := w.AppendGeneration(GenerationRow{}); err == nil {
		t.Error("AppendGeneration into a directory: want error")
	}
	if err := w.AppendEvent(Event{}); err == nil {
		t.Error("AppendEvent into a directory: want error")
	}
}

// TestKernelNamesReadDirError drives KernelNames's non-ENOENT ReadDir error by
// replacing the kernels directory with a regular file.
func TestKernelNamesReadDirError(t *testing.T) {
	root := t.TempDir()
	id := "20260101T000000Z-kdirfile-abcdef"
	dir := filepath.Join(root, id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	mjson := `{"bundle_version":1,"kind":"train","model":"kdirfile"}`
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(mjson), 0o644); err != nil {
		t.Fatal(err)
	}
	// kernels is a regular file, not a directory.
	if err := os.WriteFile(filepath.Join(dir, "kernels"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := OpenBundle(dir)
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if _, err := r.KernelNames(); err == nil {
		t.Fatal("KernelNames with kernels-as-file: want error")
	} else if !strings.Contains(err.Error(), "read kernels dir") {
		t.Fatalf("error = %v, want 'read kernels dir'", err)
	}
}

// TestOpenBundleNotADirectory drives OpenBundle's non-directory branch.
func TestOpenBundleNotADirectory(t *testing.T) {
	root := t.TempDir()
	// A regular file with a valid-looking bundle name.
	f := filepath.Join(root, "20260101T000000Z-notdir-abcdef")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenBundle(f); err == nil {
		t.Fatal("OpenBundle on a file: want error")
	} else if !strings.Contains(err.Error(), "not a directory") {
		t.Fatalf("error = %v, want 'not a directory'", err)
	}
}

// TestAtomicWriteRenameError drives atomicWriteFile via WriteGraph when the
// destination directory is read-only, so the temp write (or rename) fails.
func TestAtomicWriteReadOnlyDir(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: read-only directory perms are not enforced")
	}
	root := t.TempDir()
	w, err := NewWriter(root, "rodir", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := os.Chmod(w.Path(), 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(w.Path(), 0o755) })
	if err := w.WriteGraph([]byte(`{}`)); err == nil {
		t.Fatal("WriteGraph into read-only bundle dir: want error")
	}
}

// TestEnvOrDefaultFallsBackToDefaultRoot drives the no-env-var branch of
// EnvOrDefault, which delegates to DefaultRoot (under os.UserCacheDir).
func TestEnvOrDefaultFallsBackToDefaultRoot(t *testing.T) {
	t.Setenv(EnvVar, "")
	// Redirect the user cache dir to a temp location (XDG on linux, HOME on
	// macOS) so DefaultRoot does not touch the real cache.
	cache := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cache)
	t.Setenv("HOME", cache)

	got, err := EnvOrDefault()
	if err != nil {
		t.Fatalf("EnvOrDefault fallback: %v", err)
	}
	if !strings.Contains(got, "anneal") {
		t.Fatalf("DefaultRoot = %q, want a path containing 'anneal'", got)
	}
	if st, err := os.Stat(got); err != nil || !st.IsDir() {
		t.Fatalf("DefaultRoot dir not created at %q: %v", got, err)
	}
}

// TestNewWriterMkdirRootError drives NewWriter's os.MkdirAll(rootAbs) failure
// by passing a rootDir whose parent path component is a regular file.
func TestNewWriterMkdirRootError(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "afile")
	if err := os.WriteFile(parent, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	// rootDir is under a regular file, so MkdirAll cannot create it.
	if _, err := NewWriter(filepath.Join(parent, "runs"), "m", KindTrain); err == nil {
		t.Fatal("NewWriter with unmakeable root: want error")
	} else if !strings.Contains(err.Error(), "mkdir") {
		t.Fatalf("error = %v, want 'mkdir'", err)
	}
}

// TestFinalizeClosesOpenStreams verifies Finalize flushes and closes the lazily
// opened streaming files (the closeStreams non-nil branch for each handle).
func TestFinalizeClosesOpenStreams(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "streams", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.AppendLoss(LossRow{Step: 1, Loss: 1, WallMs: 1}); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendGeneration(GenerationRow{}); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendEvent(Event{}); err != nil {
		t.Fatal(err)
	}
	if err := w.Finalize(42); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	// All three streams must read back through a Reader.
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if rows, _ := r.Loss(); len(rows) != 1 {
		t.Fatalf("Loss rows = %d, want 1", len(rows))
	}
	if r.Manifest().DurationMs != 42 {
		t.Fatalf("DurationMs = %d, want 42", r.Manifest().DurationMs)
	}
}

// TestAtomicWriteRenameOverDir drives atomicWriteFile's os.Rename error branch:
// the temp file writes fine but the rename target is an existing directory,
// which cannot be replaced by a file.
func TestAtomicWriteRenameOverDir(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "renamefail", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	// Make graph.json a non-empty directory so rename(tmp, graph.json) fails.
	gdir := filepath.Join(w.Path(), "graph.json")
	if err := os.Mkdir(gdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(gdir, "occupant"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteGraph([]byte(`{}`)); err == nil {
		t.Fatal("WriteGraph onto a non-empty dir: want rename error")
	} else if !strings.Contains(err.Error(), "rename") {
		t.Fatalf("error = %v, want 'rename'", err)
	}
	// The temp file must be cleaned up on the rename failure.
	if _, err := os.Stat(gdir + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf(".tmp not cleaned up after rename failure: %v", err)
	}
}

// TestWriteConfigMarshalError drives WriteConfig's json.Marshal failure by
// stuffing a non-marshalable value (a channel) into Hyperparams.
func TestWriteConfigMarshalError(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "cfgmarshal", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	cfg := Config{
		Model:       "m",
		Hyperparams: map[string]any{"bad": make(chan int)},
	}
	if err := w.WriteConfig(cfg); err == nil {
		t.Fatal("WriteConfig with non-marshalable value: want error")
	} else if !strings.Contains(err.Error(), "marshal config") {
		t.Fatalf("error = %v, want 'marshal config'", err)
	}
}

// TestBundleKindUnmarshalNonString drives UnmarshalJSON's first error branch
// (the input is not a JSON string).
func TestBundleKindUnmarshalNonString(t *testing.T) {
	var k BundleKind
	if err := k.UnmarshalJSON([]byte("12345")); err == nil {
		t.Fatal("UnmarshalJSON(number): want error")
	} else if !strings.Contains(err.Error(), "kind") {
		t.Fatalf("error = %v, want 'kind'", err)
	}
}
