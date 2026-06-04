// Package bundle implements the on-disk run bundle format for the anneal
// studio's history view (spec §6, §7).
//
// A bundle is a directory under ~/.cache/anneal/runs/ that captures one
// training or generation run as a citation rather than a screenshot: the
// frozen UOp graph, the schedule, generated WGSL, the loss curve, the SSE
// event stream, and a provenance manifest pinning the anneal version, git
// rev, device, adapter, and WGSL hash.
//
// # Layout
//
//	~/.cache/anneal/runs/<timestamp>-<model>-<shorthash>/
//	  manifest.json       provenance (immutable post-Finalize)
//	  config.json         model name, hyperparams, device
//	  schedule.json       realize-map + fused kernel set
//	  loss.csv            "step,loss,wall_ms" header + rows (train runs)
//	  generation.ndjson   per-token records (generate runs)
//	  kernels/K*.wgsl     compiled shader text (one file per kernel)
//	  graph.json          the frozen UOp graph (same JSON viz produces)
//	  events.ndjson       the SSE stream, replayable
//
// # Schema versioning
//
// Bundles carry a bundle_version integer in manifest.json. v1 is what this
// release ships. Future readers MUST either tolerate an older version
// (additive changes can be ignored by older code) or refuse with a clear
// message naming the version. The v1 reader refuses anything other than 1
// with: "bundle: unsupported bundle_version %d (this anneal supports
// version 1)". Any breaking change to manifest shape MUST bump
// bundle_version; additive fields that older readers can ignore do not.
//
// # Ordering
//
// Anything in manifest.json is keyed structurally (per spec §6). Map keys
// are encoded in lexicographic order so byte-equality holds across runs
// with the same payload — important for citations and content hashing.
//
// # Path safety
//
// The reader refuses to open bundles outside the configured root and
// validates bundle directory names match the canonical
// <digits>-<modelname>-<6hex> form, matching the ONNX importer's posture
// against path-traversal in user-supplied identifiers.
package bundle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"time"
)

// BundleVersion is the current on-disk schema version. Bump on any breaking
// change to manifest.json shape; additive fields do not require a bump.
const BundleVersion = 1

// BundleKind tags whether a bundle was produced by a training run, a
// generation run, or a manual save from the studio.
type BundleKind int

const (
	// KindTrain is a training run (loss.csv is the primary timeseries).
	KindTrain BundleKind = iota
	// KindGenerate is a generation run (generation.ndjson is the primary
	// timeseries).
	KindGenerate
	// KindSaved is a user-initiated snapshot from the studio (no live
	// timeseries; just graph + kernels + manifest).
	KindSaved
)

// String returns the canonical JSON-serializable name for kind.
func (k BundleKind) String() string {
	switch k {
	case KindTrain:
		return "train"
	case KindGenerate:
		return "generate"
	case KindSaved:
		return "saved"
	default:
		return "unknown"
	}
}

// MarshalJSON encodes a BundleKind as a JSON string.
func (k BundleKind) MarshalJSON() ([]byte, error) {
	return json.Marshal(k.String())
}

// UnmarshalJSON decodes a BundleKind from a JSON string. Unknown values
// return an error so older readers refuse newer kinds rather than silently
// downgrading.
func (k *BundleKind) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return fmt.Errorf("bundle: kind: %w", err)
	}
	switch s {
	case "train":
		*k = KindTrain
	case "generate":
		*k = KindGenerate
	case "saved":
		*k = KindSaved
	default:
		return fmt.Errorf("bundle: unknown kind %q (supported: train, generate, saved)", s)
	}
	return nil
}

// Manifest is the provenance block written to manifest.json. It is the
// source of truth for what produced the bundle: anneal version, git rev,
// device + adapter, WGSL hash, symbolic bindings, kind, created/duration
// timestamps, and the bundle_version integer.
//
// The shape is keyed structurally per spec §6 — extending it requires
// either an additive field (older readers will silently ignore) or a
// BundleVersion bump.
type Manifest struct {
	BundleVersion int              `json:"bundle_version"`
	Kind          BundleKind       `json:"kind"`
	Model         string           `json:"model"`
	AnnealVersion string           `json:"anneal_version"`
	GitRev        string           `json:"git_rev"`
	DeviceName    string           `json:"device_name"`
	Adapter       string           `json:"adapter"`
	WGSLHash      string           `json:"wgsl_hash"`
	SymBinds      map[string]int64 `json:"sym_binds"`
	CreatedAt     time.Time        `json:"created_at"`
	DurationMs    int64            `json:"duration_ms"`
}

// MarshalJSON encodes the manifest with deterministic key ordering for the
// SymBinds map (lexicographic) so two manifests with the same payload are
// byte-identical. The rest of the struct already has stable order from the
// field declarations.
func (m Manifest) MarshalJSON() ([]byte, error) {
	// Encode SymBinds with sorted keys.
	var sbBuf bytes.Buffer
	sbBuf.WriteByte('{')
	keys := make([]string, 0, len(m.SymBinds))
	for k := range m.SymBinds {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			sbBuf.WriteByte(',')
		}
		kb, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		sbBuf.Write(kb)
		sbBuf.WriteByte(':')
		sbBuf.WriteString(strconv.FormatInt(m.SymBinds[k], 10))
	}
	sbBuf.WriteByte('}')

	type aux struct {
		BundleVersion int             `json:"bundle_version"`
		Kind          BundleKind      `json:"kind"`
		Model         string          `json:"model"`
		AnnealVersion string          `json:"anneal_version"`
		GitRev        string          `json:"git_rev"`
		DeviceName    string          `json:"device_name"`
		Adapter       string          `json:"adapter"`
		WGSLHash      string          `json:"wgsl_hash"`
		SymBinds      json.RawMessage `json:"sym_binds"`
		CreatedAt     time.Time       `json:"created_at"`
		DurationMs    int64           `json:"duration_ms"`
	}
	return json.Marshal(aux{
		BundleVersion: m.BundleVersion,
		Kind:          m.Kind,
		Model:         m.Model,
		AnnealVersion: m.AnnealVersion,
		GitRev:        m.GitRev,
		DeviceName:    m.DeviceName,
		Adapter:       m.Adapter,
		WGSLHash:      m.WGSLHash,
		SymBinds:      sbBuf.Bytes(),
		CreatedAt:     m.CreatedAt,
		DurationMs:    m.DurationMs,
	})
}

// Config is the model configuration written to config.json. Hyperparams is
// intentionally typed as map[string]any so callers can add/remove keys
// freely without breaking older readers.
type Config struct {
	Model       string         `json:"model"`
	Device      string         `json:"device"`
	Hyperparams map[string]any `json:"hyperparams"`
}

// LossRow is one row of loss.csv. wall_ms is the elapsed wall-clock time
// since the start of training in milliseconds.
type LossRow struct {
	Step   int     `json:"step"`
	Loss   float32 `json:"loss"`
	WallMs int64   `json:"wall_ms"`
}

// CSVRow returns the row's three-column CSV encoding without a trailing
// newline. The float32 uses %.6g for compact lossless-enough decimal text;
// downstream consumers (the JS sparkline, the history compare view) parse
// it as a float.
func (r LossRow) CSVRow() string {
	return fmt.Sprintf("%d,%g,%d", r.Step, r.Loss, r.WallMs)
}

// ParseLossRow parses one CSV row written by CSVRow. Whitespace around
// fields is trimmed.
func ParseLossRow(s string) (LossRow, error) {
	parts := splitCSV(s)
	if len(parts) != 3 {
		return LossRow{}, fmt.Errorf("bundle: loss row: want 3 fields, got %d (%q)", len(parts), s)
	}
	step, err := strconv.Atoi(parts[0])
	if err != nil {
		return LossRow{}, fmt.Errorf("bundle: loss row step: %w", err)
	}
	loss, err := strconv.ParseFloat(parts[1], 32)
	if err != nil {
		return LossRow{}, fmt.Errorf("bundle: loss row loss: %w", err)
	}
	wallMs, err := strconv.ParseInt(parts[2], 10, 64)
	if err != nil {
		return LossRow{}, fmt.Errorf("bundle: loss row wall_ms: %w", err)
	}
	return LossRow{Step: step, Loss: float32(loss), WallMs: wallMs}, nil
}

// LossCSVHeader is the header row written at the top of loss.csv.
const LossCSVHeader = "step,loss,wall_ms"

// GenerationRow is one row of generation.ndjson. RefMatch is a pointer so
// the absence of a reference (no oracle configured) is distinguishable
// from a recorded "false" — JSON omits the field entirely when nil.
type GenerationRow struct {
	Step         int    `json:"step"`
	TokenID      int    `json:"token_id"`
	TokenText    string `json:"token_text"`
	LogitArgmax  int    `json:"logit_argmax"`
	LogitSummary string `json:"logit_summary"`
	RefMatch     *bool  `json:"ref_match,omitempty"`
}

// Event is one SSE event tee'd to events.ndjson. Payload is preserved as
// raw JSON so the reader can replay it byte-for-byte without re-encoding.
type Event struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// BundleID is the directory name of a bundle (the form
// <ts>-<model>-<shorthash>). It is a string, not a path, so the surface
// API can pass it around without path-traversal exposure.
type BundleID string

// BundleSummary is a bundle's manifest plus its on-disk path, used by the
// /api/runs listing endpoint.
type BundleSummary struct {
	ID       BundleID `json:"id"`
	Path     string   `json:"path"`
	Manifest Manifest `json:"manifest"`
}

// splitCSV is a minimal CSV row splitter — bundle CSV files have no
// quoted fields, so a naive split-on-comma is correct.
func splitCSV(s string) []string {
	out := []string{}
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			out = append(out, trimSpace(s[start:i]))
			start = i + 1
		}
	}
	out = append(out, trimSpace(s[start:]))
	return out
}

// trimSpace trims ASCII space/tab from both ends without allocating
// through strings.TrimSpace's unicode path.
func trimSpace(s string) string {
	i, j := 0, len(s)
	for i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') {
		i++
	}
	for j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n') {
		j--
	}
	return s[i:j]
}
