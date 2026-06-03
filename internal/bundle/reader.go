package bundle

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// bundleNameRe is the canonical bundle directory name shape:
// "<digits>T<digits>Z-<modelname>-<6hex>". The digits before 'T' are the
// date (YYYYMMDD), the digits after are the time (HHMMSS). The model name
// is lowercase alphanumeric/dash. The 6 hex chars are a random
// disambiguator. Anything else is refused to keep path-traversal off the
// table.
var bundleNameRe = regexp.MustCompile(`^[0-9]{8}T[0-9]{6}Z-[a-z0-9][a-z0-9-]*-[0-9a-f]{6}$`)

// IsValidBundleID reports whether s is a syntactically valid bundle
// directory name. The reader uses this before resolving paths so a
// caller-supplied id can never escape the configured root.
func IsValidBundleID(s string) bool {
	return bundleNameRe.MatchString(s)
}

// Reader is an open handle to a bundle on disk. The manifest is parsed
// eagerly so OpenBundle can validate the schema version before returning;
// the rest is parsed lazily on demand.
type Reader struct {
	path     string
	id       BundleID
	manifest Manifest
}

// OpenBundle opens the bundle at path. The path is validated for
// containment (it must resolve within its parent root after symlink
// resolution-free Clean), and the directory name is checked against
// bundleNameRe. The manifest is read and its bundle_version validated.
//
// Errors are returned for: nonexistent path, non-directory, invalid name,
// missing manifest.json, malformed manifest, and unsupported
// bundle_version.
func OpenBundle(path string) (*Reader, error) {
	clean := filepath.Clean(path)
	abs, err := filepath.Abs(clean)
	if err != nil {
		return nil, fmt.Errorf("bundle: OpenBundle: abs(%q): %w", path, err)
	}
	id := filepath.Base(abs)
	if !IsValidBundleID(id) {
		return nil, fmt.Errorf("bundle: OpenBundle: invalid bundle name %q (must match %s)", id, bundleNameRe.String())
	}
	st, err := os.Stat(abs)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("bundle: OpenBundle: %q does not exist", abs)
		}
		return nil, fmt.Errorf("bundle: OpenBundle: stat %q: %w", abs, err)
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("bundle: OpenBundle: %q is not a directory", abs)
	}
	r := &Reader{path: abs, id: BundleID(id)}
	if err := r.loadManifest(); err != nil {
		return nil, err
	}
	return r, nil
}

// OpenBundleIn opens a bundle id inside rootDir, enforcing that the
// resolved path is contained within rootDir. This is the safe API for
// HTTP handlers that take an id from a URL — it refuses path-traversal
// (e.g. "../etc/passwd") before any file I/O.
func OpenBundleIn(rootDir string, id string) (*Reader, error) {
	if !IsValidBundleID(id) {
		return nil, fmt.Errorf("bundle: OpenBundleIn: invalid bundle id %q", id)
	}
	rootAbs, err := filepath.Abs(filepath.Clean(rootDir))
	if err != nil {
		return nil, fmt.Errorf("bundle: OpenBundleIn: abs(%q): %w", rootDir, err)
	}
	target := filepath.Join(rootAbs, id)
	targetClean := filepath.Clean(target)
	// Containment check: targetClean must equal rootAbs + sep + id with no
	// traversal. filepath.Rel returning a path starting with ".." means
	// the target escaped.
	rel, err := filepath.Rel(rootAbs, targetClean)
	if err != nil {
		return nil, fmt.Errorf("bundle: OpenBundleIn: rel: %w", err)
	}
	if strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) || rel == "." {
		return nil, fmt.Errorf("bundle: OpenBundleIn: id %q escapes root %q", id, rootAbs)
	}
	return OpenBundle(targetClean)
}

// BundleID returns the bundle's directory name.
func (r *Reader) BundleID() BundleID { return r.id }

// Path returns the bundle's directory path.
func (r *Reader) Path() string { return r.path }

// Manifest returns a copy of the parsed manifest.
func (r *Reader) Manifest() Manifest { return r.manifest }

// Config parses and returns config.json.
func (r *Reader) Config() (Config, error) {
	b, err := os.ReadFile(filepath.Join(r.path, "config.json"))
	if err != nil {
		return Config{}, fmt.Errorf("bundle: read config.json: %w", err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return Config{}, fmt.Errorf("bundle: parse config.json: %w", err)
	}
	return c, nil
}

// Graph returns the raw bytes of graph.json (no re-marshal).
func (r *Reader) Graph() ([]byte, error) {
	b, err := os.ReadFile(filepath.Join(r.path, "graph.json"))
	if err != nil {
		return nil, fmt.Errorf("bundle: read graph.json: %w", err)
	}
	return b, nil
}

// Schedule returns the raw bytes of schedule.json (no re-marshal).
func (r *Reader) Schedule() ([]byte, error) {
	b, err := os.ReadFile(filepath.Join(r.path, "schedule.json"))
	if err != nil {
		return nil, fmt.Errorf("bundle: read schedule.json: %w", err)
	}
	return b, nil
}

// KernelNames returns the names of every kernels/*.wgsl file in
// alphabetical order. The ".wgsl" suffix is preserved so the HTTP layer
// can pass the value back through Kernel() verbatim.
func (r *Reader) KernelNames() ([]string, error) {
	dir := filepath.Join(r.path, "kernels")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return []string{}, nil
		}
		return nil, fmt.Errorf("bundle: read kernels dir: %w", err)
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".wgsl") {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out, nil
}

// Kernel returns the WGSL text of one kernel. The name is validated for
// containment (no separators, no "..") before resolving.
func (r *Reader) Kernel(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("bundle: Kernel: empty name")
	}
	if strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
		return "", fmt.Errorf("bundle: Kernel: invalid name %q", name)
	}
	if !strings.HasSuffix(name, ".wgsl") {
		name = name + ".wgsl"
	}
	b, err := os.ReadFile(filepath.Join(r.path, "kernels", name))
	if err != nil {
		return "", fmt.Errorf("bundle: read kernel %q: %w", name, err)
	}
	return string(b), nil
}

// Loss parses every row of loss.csv. The header row is skipped. Returns
// an empty slice (not nil) when the file is absent (a generate-only run).
func (r *Reader) Loss() ([]LossRow, error) {
	f, err := os.Open(filepath.Join(r.path, "loss.csv"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return []LossRow{}, nil
		}
		return nil, fmt.Errorf("bundle: open loss.csv: %w", err)
	}
	defer func() { _ = f.Close() }()
	out := []LossRow{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	first := true
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if first {
			first = false
			if line != LossCSVHeader {
				return nil, fmt.Errorf("bundle: loss.csv header: got %q, want %q", line, LossCSVHeader)
			}
			continue
		}
		if line == "" {
			continue
		}
		row, err := ParseLossRow(line)
		if err != nil {
			return nil, fmt.Errorf("bundle: loss.csv line %d: %w", lineNo, err)
		}
		out = append(out, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("bundle: scan loss.csv: %w", err)
	}
	return out, nil
}

// Generation parses every row of generation.ndjson. Returns an empty
// slice when the file is absent (a train-only run).
func (r *Reader) Generation() ([]GenerationRow, error) {
	f, err := os.Open(filepath.Join(r.path, "generation.ndjson"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return []GenerationRow{}, nil
		}
		return nil, fmt.Errorf("bundle: open generation.ndjson: %w", err)
	}
	defer func() { _ = f.Close() }()
	out := []GenerationRow{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if line == "" {
			continue
		}
		var row GenerationRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("bundle: generation.ndjson line %d: %w", lineNo, err)
		}
		out = append(out, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("bundle: scan generation.ndjson: %w", err)
	}
	return out, nil
}

// Events parses every row of events.ndjson. Returns an empty slice when
// the file is absent.
func (r *Reader) Events() ([]Event, error) {
	f, err := os.Open(filepath.Join(r.path, "events.ndjson"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return []Event{}, nil
		}
		return nil, fmt.Errorf("bundle: open events.ndjson: %w", err)
	}
	defer func() { _ = f.Close() }()
	out := []Event{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if line == "" {
			continue
		}
		var e Event
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			return nil, fmt.Errorf("bundle: events.ndjson line %d: %w", lineNo, err)
		}
		out = append(out, e)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("bundle: scan events.ndjson: %w", err)
	}
	return out, nil
}

// loadManifest reads + parses manifest.json and enforces the
// bundle_version contract.
func (r *Reader) loadManifest() error {
	b, err := os.ReadFile(filepath.Join(r.path, "manifest.json"))
	if err != nil {
		return fmt.Errorf("bundle: read manifest.json: %w", err)
	}
	// Two-pass parse so we can read bundle_version before committing to
	// the full struct shape (a future v2 might add unknown enum values).
	var probe struct {
		BundleVersion int `json:"bundle_version"`
	}
	if err := json.Unmarshal(b, &probe); err != nil {
		return fmt.Errorf("bundle: parse manifest.json: %w", err)
	}
	if probe.BundleVersion != BundleVersion {
		return fmt.Errorf("bundle: unsupported bundle_version %d (this anneal supports version %d)",
			probe.BundleVersion, BundleVersion)
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return fmt.Errorf("bundle: parse manifest.json: %w", err)
	}
	if m.SymBinds == nil {
		m.SymBinds = map[string]int64{}
	}
	r.manifest = m
	return nil
}
