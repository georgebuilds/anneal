package bundle

import (
	"fmt"
	"os"
	"path/filepath"
)

// rootSubdir is the sub-path under os.UserCacheDir that holds run bundles.
// Mirrors the layout convention used by internal/assets (which uses
// "anneal/v1").
const rootSubdir = "anneal/runs"

// EnvVar is the env var name that overrides the default root.
const EnvVar = "ANNEAL_RUN_DIR"

// DefaultRoot returns the default run-cache directory rooted at
// os.UserCacheDir / anneal/runs, creating it if missing. This is the
// location described in spec §6.
func DefaultRoot() (string, error) {
	uc, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("bundle: locate user cache dir: %w", err)
	}
	root := filepath.Join(uc, rootSubdir)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", fmt.Errorf("bundle: create root %q: %w", root, err)
	}
	return root, nil
}

// EnvOrDefault honors the ANNEAL_RUN_DIR env var if set (must be an
// absolute path); otherwise it falls back to DefaultRoot. The directory
// is created if missing in either branch.
func EnvOrDefault() (string, error) {
	if v := os.Getenv(EnvVar); v != "" {
		if !filepath.IsAbs(v) {
			return "", fmt.Errorf("bundle: %s must be absolute, got %q", EnvVar, v)
		}
		if err := os.MkdirAll(v, 0o755); err != nil {
			return "", fmt.Errorf("bundle: create %s root %q: %w", EnvVar, v, err)
		}
		return v, nil
	}
	return DefaultRoot()
}
