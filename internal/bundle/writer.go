package bundle

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Writer is the open handle to a bundle being produced. Streaming sinks
// (loss.csv, generation.ndjson, events.ndjson) are append-mode files
// opened lazily on first write; one-shot artifacts (graph.json,
// schedule.json, config.json, kernels/*.wgsl) are written atomically via
// a temp-and-rename pattern.
//
// Concurrent calls are safe - a single mutex guards the streaming file
// handles and the manifest stub. The writer is single-bundle: open one
// Writer per run.
type Writer struct {
	mu sync.Mutex

	dir   string
	id    BundleID
	model string
	kind  BundleKind

	manifest Manifest

	lossFile *os.File
	lossHdr  bool // true once the CSV header has been written
	genFile  *os.File
	evtFile  *os.File

	finalized bool
}

// NewWriter creates a fresh bundle directory under rootDir named
// <ts>-<model>-<shorthash>, writes a stub manifest.json with
// bundle_version=1, created_at=now, and returns an open writer.
//
// The shorthash is 6 hex chars of crypto/rand so two runs of the same
// model in the same second do not collide. The model name is sanitized
// (lowercased, non-[a-z0-9-] replaced with '-') so it survives a path
// containment check.
func NewWriter(rootDir string, model string, kind BundleKind) (*Writer, error) {
	if rootDir == "" {
		return nil, fmt.Errorf("bundle: NewWriter: rootDir is empty")
	}
	if model == "" {
		return nil, fmt.Errorf("bundle: NewWriter: model name is empty")
	}
	rootAbs, err := filepath.Abs(rootDir)
	if err != nil {
		return nil, fmt.Errorf("bundle: NewWriter: abs(%q): %w", rootDir, err)
	}
	if err := os.MkdirAll(rootAbs, 0o755); err != nil {
		return nil, fmt.Errorf("bundle: NewWriter: mkdir %q: %w", rootAbs, err)
	}

	now := time.Now().UTC()
	ts := now.Format("20060102T150405Z")
	cleanModel := sanitizeModel(model)
	short, err := shortHash()
	if err != nil {
		return nil, err
	}
	id := fmt.Sprintf("%s-%s-%s", ts, cleanModel, short)

	dir := filepath.Join(rootAbs, id)
	if err := os.Mkdir(dir, 0o755); err != nil {
		return nil, fmt.Errorf("bundle: NewWriter: mkdir %q: %w", dir, err)
	}
	if err := os.Mkdir(filepath.Join(dir, "kernels"), 0o755); err != nil {
		return nil, fmt.Errorf("bundle: NewWriter: mkdir kernels: %w", err)
	}

	w := &Writer{
		dir:   dir,
		id:    BundleID(id),
		model: cleanModel,
		kind:  kind,
		manifest: Manifest{
			BundleVersion: BundleVersion,
			Kind:          kind,
			Model:         cleanModel,
			SymBinds:      map[string]int64{},
			CreatedAt:     now,
		},
	}
	if err := w.writeManifest(); err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	return w, nil
}

// BundleID returns the bundle's directory name (the BundleID, not a path).
func (w *Writer) BundleID() BundleID { return w.id }

// Path returns the bundle's directory path.
func (w *Writer) Path() string { return w.dir }

// SetProvenance updates the provenance fields on the manifest stub. Call
// this once after NewWriter and before Finalize; the new values are
// flushed to manifest.json immediately.
//
// SymBinds may be nil (treated as empty). Any zero-value field is left as
// the empty string in the manifest.
func (w *Writer) SetProvenance(annealVersion, gitRev, deviceName, adapter, wgslHash string, symBinds map[string]int64) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.manifest.AnnealVersion = annealVersion
	w.manifest.GitRev = gitRev
	w.manifest.DeviceName = deviceName
	w.manifest.Adapter = adapter
	w.manifest.WGSLHash = wgslHash
	if symBinds == nil {
		w.manifest.SymBinds = map[string]int64{}
	} else {
		w.manifest.SymBinds = symBinds
	}
	return w.writeManifest()
}

// AppendLoss writes one row to loss.csv. The header row is written on
// first call. Safe for concurrent callers.
func (w *Writer) AppendLoss(row LossRow) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.finalized {
		return fmt.Errorf("bundle: AppendLoss after Finalize")
	}
	if w.lossFile == nil {
		f, err := os.OpenFile(filepath.Join(w.dir, "loss.csv"),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("bundle: open loss.csv: %w", err)
		}
		w.lossFile = f
	}
	if !w.lossHdr {
		if _, err := w.lossFile.WriteString(LossCSVHeader + "\n"); err != nil {
			return fmt.Errorf("bundle: write loss header: %w", err)
		}
		w.lossHdr = true
	}
	if _, err := w.lossFile.WriteString(row.CSVRow() + "\n"); err != nil {
		return fmt.Errorf("bundle: write loss row: %w", err)
	}
	return nil
}

// AppendGeneration writes one row to generation.ndjson. Safe for
// concurrent callers.
func (w *Writer) AppendGeneration(row GenerationRow) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.finalized {
		return fmt.Errorf("bundle: AppendGeneration after Finalize")
	}
	if w.genFile == nil {
		f, err := os.OpenFile(filepath.Join(w.dir, "generation.ndjson"),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("bundle: open generation.ndjson: %w", err)
		}
		w.genFile = f
	}
	b, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("bundle: marshal generation row: %w", err)
	}
	b = append(b, '\n')
	if _, err := w.genFile.Write(b); err != nil {
		return fmt.Errorf("bundle: write generation row: %w", err)
	}
	return nil
}

// AppendEvent writes one event to events.ndjson. Safe for concurrent
// callers; each event lands as a single newline-framed JSON record.
func (w *Writer) AppendEvent(e Event) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.finalized {
		return fmt.Errorf("bundle: AppendEvent after Finalize")
	}
	if w.evtFile == nil {
		f, err := os.OpenFile(filepath.Join(w.dir, "events.ndjson"),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("bundle: open events.ndjson: %w", err)
		}
		w.evtFile = f
	}
	b, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("bundle: marshal event: %w", err)
	}
	b = append(b, '\n')
	if _, err := w.evtFile.Write(b); err != nil {
		return fmt.Errorf("bundle: write event: %w", err)
	}
	return nil
}

// WriteGraph atomically writes graphJSON to graph.json. The caller owns
// the marshalling - viz already produces this JSON and the bundle stores
// the exact bytes for byte-equal replay.
func (w *Writer) WriteGraph(graphJSON []byte) error {
	return w.atomicWrite("graph.json", graphJSON)
}

// WriteSchedule atomically writes scheduleJSON to schedule.json.
func (w *Writer) WriteSchedule(scheduleJSON []byte) error {
	return w.atomicWrite("schedule.json", scheduleJSON)
}

// WriteConfig atomically writes c to config.json.
func (w *Writer) WriteConfig(c Config) error {
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("bundle: marshal config: %w", err)
	}
	return w.atomicWrite("config.json", b)
}

// WriteKernel writes one WGSL shader to kernels/<name>.wgsl. If name does
// not already start with 'K' the leading "K" is prepended (so callers can
// pass either "K3" or "3").
func (w *Writer) WriteKernel(name string, wgsl string) error {
	if name == "" {
		return fmt.Errorf("bundle: WriteKernel: empty name")
	}
	// Reject path-separators in name to keep kernels/ scoped.
	if strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
		return fmt.Errorf("bundle: WriteKernel: invalid name %q", name)
	}
	fname := name
	if !strings.HasPrefix(fname, "K") {
		fname = "K" + fname
	}
	if !strings.HasSuffix(fname, ".wgsl") {
		fname = fname + ".wgsl"
	}
	return w.atomicWrite(filepath.Join("kernels", fname), []byte(wgsl))
}

// Finalize stamps the manifest with durationMs, flushes and closes the
// streaming sinks, and rewrites manifest.json atomically. After Finalize
// the writer rejects further Append/Write calls.
//
// Safe to call multiple times; only the first call has effect.
func (w *Writer) Finalize(durationMs int64) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.finalized {
		return nil
	}
	w.manifest.DurationMs = durationMs
	if err := w.writeManifest(); err != nil {
		return err
	}
	w.finalized = true
	return w.closeStreams()
}

// Close releases the streaming file handles without stamping a duration.
// A reader opening a Close'd-but-not-Finalize'd bundle will see
// duration_ms == 0 (the "writer was killed mid-run" case). Idempotent.
func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.finalized {
		// Streams already closed by Finalize.
		return nil
	}
	w.finalized = true
	return w.closeStreams()
}

// closeStreams closes whichever streaming files are open. Caller holds w.mu.
func (w *Writer) closeStreams() error {
	var firstErr error
	for _, fp := range []**os.File{&w.lossFile, &w.genFile, &w.evtFile} {
		if *fp != nil {
			if err := (*fp).Close(); err != nil && firstErr == nil {
				firstErr = err
			}
			*fp = nil
		}
	}
	return firstErr
}

// writeManifest writes w.manifest to manifest.json atomically. Caller
// holds w.mu.
func (w *Writer) writeManifest() error {
	b, err := json.MarshalIndent(w.manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("bundle: marshal manifest: %w", err)
	}
	return atomicWriteFile(filepath.Join(w.dir, "manifest.json"), b)
}

// atomicWrite is the locked wrapper around atomicWriteFile used by the
// per-artifact writers (graph, schedule, config, kernels). Caller must NOT
// hold w.mu - the writes themselves do not need the streaming-sink mutex,
// and serializing them would block streaming writes unnecessarily.
func (w *Writer) atomicWrite(relpath string, data []byte) error {
	w.mu.Lock()
	if w.finalized {
		w.mu.Unlock()
		return fmt.Errorf("bundle: atomicWrite %q after Finalize", relpath)
	}
	w.mu.Unlock()
	return atomicWriteFile(filepath.Join(w.dir, relpath), data)
}

// atomicWriteFile writes data to dst via "<dst>.tmp" + rename so a
// concurrent reader never sees a half-written file (POSIX rename is
// atomic within a filesystem).
func atomicWriteFile(dst string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("bundle: mkdir parent of %q: %w", dst, err)
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("bundle: write tmp %q: %w", tmp, err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("bundle: rename %q -> %q: %w", tmp, dst, err)
	}
	return nil
}

// sanitizeModel lowercases the model and replaces every non
// [a-z0-9-] character with '-' so the directory name passes the
// reader's bundle-name regex. Empty input yields "model".
func sanitizeModel(s string) string {
	if s == "" {
		return "model"
	}
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '-':
			out = append(out, c)
		case c >= 'A' && c <= 'Z':
			out = append(out, c+32)
		default:
			out = append(out, '-')
		}
	}
	// Strip leading and trailing '-' so a fully-sanitized name does not
	// produce paths like "-foo-" that look fishy in CLI output.
	start, end := 0, len(out)
	for start < end && out[start] == '-' {
		start++
	}
	for end > start && out[end-1] == '-' {
		end--
	}
	if start == end {
		return "model"
	}
	return string(out[start:end])
}

// shortHash returns 6 hex chars from crypto/rand. Used to disambiguate
// bundles produced in the same second by the same model.
func shortHash() (string, error) {
	var b [3]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("bundle: shortHash: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}
