package web

import (
	"io/fs"
	"testing"
)

// TestFS_EmbedsStudioAssets verifies FS() exposes the embedded studio assets
// rooted at the package (so callers read "studio.html", not "web/studio.html").
func TestFS_EmbedsStudioAssets(t *testing.T) {
	f := FS()
	if f == nil {
		t.Fatal("FS() returned nil")
	}

	want := []string{
		"studio.html",
		"studio.css",
		"studio.js",
		"worker.js",
		"wasm_exec.js",
		"visualize_embed.html",
	}
	for _, name := range want {
		data, err := fs.ReadFile(f, name)
		if err != nil {
			t.Errorf("ReadFile(%q): %v", name, err)
			continue
		}
		if len(data) == 0 {
			t.Errorf("embedded asset %q is empty", name)
		}
	}

	// Assets are rooted at the package: the "web/" prefix must NOT resolve.
	if _, err := fs.ReadFile(f, "web/studio.html"); err == nil {
		t.Error("expected web/studio.html to be unreachable (assets are package-rooted)")
	}
}
