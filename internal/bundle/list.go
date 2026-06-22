package bundle

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

// ListBundles walks rootDir one level deep and returns one BundleSummary
// per valid bundle (a directory whose name matches bundleNameRe and that
// contains a parseable manifest.json). Invalid or unreadable entries are
// silently skipped - a half-written bundle does not break the listing.
//
// The result is sorted by Manifest.CreatedAt descending (newest first).
// An empty rootDir returns an empty slice (not nil) so the JSON encoder
// emits `[]`.
func ListBundles(rootDir string) ([]BundleSummary, error) {
	if rootDir == "" {
		return nil, fmt.Errorf("bundle: ListBundles: rootDir is empty")
	}
	rootAbs, err := filepath.Abs(filepath.Clean(rootDir))
	if err != nil {
		return nil, fmt.Errorf("bundle: ListBundles: abs: %w", err)
	}
	entries, err := os.ReadDir(rootAbs)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return []BundleSummary{}, nil
		}
		return nil, fmt.Errorf("bundle: ListBundles: read %q: %w", rootAbs, err)
	}
	out := make([]BundleSummary, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if !IsValidBundleID(name) {
			continue
		}
		r, err := OpenBundle(filepath.Join(rootAbs, name))
		if err != nil {
			// Skip half-written or schema-incompatible bundles. The list
			// endpoint is best-effort; per-bundle errors surface when the
			// user opens the bundle directly.
			continue
		}
		out = append(out, BundleSummary{
			ID:       r.BundleID(),
			Path:     r.Path(),
			Manifest: r.Manifest(),
		})
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Manifest.CreatedAt.After(out[j].Manifest.CreatedAt)
	})
	return out, nil
}
