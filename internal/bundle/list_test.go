package bundle

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestListBundlesEmpty(t *testing.T) {
	root := t.TempDir()
	got, err := ListBundles(root)
	if err != nil {
		t.Fatalf("ListBundles: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("ListBundles on empty root: got %d, want 0", len(got))
	}
	if got == nil {
		t.Errorf("ListBundles returned nil slice; want empty (so JSON emits [])")
	}
}

func TestListBundlesNonexistentRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "does-not-exist")
	got, err := ListBundles(root)
	if err != nil {
		t.Fatalf("ListBundles: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("ListBundles on missing root: got %d, want 0", len(got))
	}
}

func TestListBundlesEmptyRootDirError(t *testing.T) {
	if _, err := ListBundles(""); err == nil {
		t.Errorf("ListBundles(\"\"): want error")
	}
}

func TestListBundlesSortsByCreatedAtDesc(t *testing.T) {
	root := t.TempDir()
	// Make three bundles with controlled CreatedAt times.
	makeAt := func(model string, ago time.Duration) string {
		w, err := NewWriter(root, model, KindTrain)
		if err != nil {
			t.Fatalf("NewWriter: %v", err)
		}
		// Override CreatedAt by writing the manifest directly.
		w.manifest.CreatedAt = time.Now().UTC().Add(-ago)
		_ = w.writeManifest()
		_ = w.Finalize(0)
		return string(w.BundleID())
	}
	oldID := makeAt("old", 24*time.Hour)
	midID := makeAt("mid", time.Hour)
	newID := makeAt("new", 0)

	got, err := ListBundles(root)
	if err != nil {
		t.Fatalf("ListBundles: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("ListBundles: got %d, want 3", len(got))
	}
	if string(got[0].ID) != newID || string(got[2].ID) != oldID {
		t.Errorf("ListBundles order: got %v, want [new, mid, old]",
			[]string{string(got[0].ID), string(got[1].ID), string(got[2].ID)})
	}
	_ = midID
}

func TestListBundlesSkipsInvalidEntries(t *testing.T) {
	root := t.TempDir()
	// One valid bundle.
	w, err := NewWriter(root, "valid", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	_ = w.Finalize(0)

	// Junk: a stray file, a dir with the wrong name, a dir with a valid
	// name but no manifest, a dir with a valid name and a malformed
	// manifest.
	if err := os.WriteFile(filepath.Join(root, "stray.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "not-a-bundle"), 0o755); err != nil {
		t.Fatal(err)
	}
	noMan := filepath.Join(root, "20260101T000000Z-foo-abcdef")
	if err := os.Mkdir(noMan, 0o755); err != nil {
		t.Fatal(err)
	}
	badMan := filepath.Join(root, "20260101T000000Z-bar-bcdef0")
	if err := os.Mkdir(badMan, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(badMan, "manifest.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := ListBundles(root)
	if err != nil {
		t.Fatalf("ListBundles: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("ListBundles: got %d valid bundles, want 1", len(got))
	}
	if string(got[0].ID) != string(w.BundleID()) {
		t.Errorf("ListBundles surfaced wrong bundle: %v", got[0].ID)
	}
}
