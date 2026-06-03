// Web W8: structure-only ONNX import summary for the studio's dropzone.
//
// The studio's onnx-dropzone reads a .onnx file in the browser, posts the
// bytes to the WASM worker, and renders the topology immediately. This file
// owns the summary the worker returns: a deterministic graph_id (sha256
// prefix of the model bytes), the imported graph JSON (forward topology
// only — no backward, no kernels), the input / output descriptors, the
// node + initializer counts, and the curated list of unsupported ops with
// a short human-readable reason for each.
//
// Privacy contract (spec §1.3 / §8): the import is WASM-tier; the bytes
// never reach even the local server. The studio's `annealImportONNX` call
// is the only consumer; the server has no /api/onnx/* endpoint.
//
// DD2: this file imports the real onnx package, runs the real Import with
// WithStructureOnly(), and reports the lowered Node count and (curated)
// unsupported ops. No mocks.

package viz

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/georgebuilds/anneal/onnx"
	"github.com/georgebuilds/anneal/uop"
)

// ImportSummary is the payload the studio renders after a successful drop.
// All fields stay JSON-serialisable; the studio decodes one round trip.
type ImportSummary struct {
	GraphID          string          `json:"graph_id"`
	Graph            json.RawMessage `json:"graph"`
	Kernels          json.RawMessage `json:"kernels,omitempty"`
	Inputs           []ImportIO      `json:"inputs"`
	Outputs          []ImportIO      `json:"outputs"`
	NodeCount        int             `json:"node_count"`
	InitializerCount int             `json:"initializer_count"`
	UnsupportedOps   []UnsupportedOp `json:"unsupported_ops,omitempty"`
	Opset            int64           `json:"opset"`
	Note             string          `json:"note,omitempty"`
}

// ImportIO is one graph input or output descriptor: name, dtype string, and
// the (concrete) shape vector. Symbolic dims are rendered as -1 for now;
// the studio surfaces them with the dim-param name when we extend this in
// W9+.
type ImportIO struct {
	Name  string  `json:"name"`
	DType string  `json:"dtype"`
	Shape []int64 `json:"shape,omitempty"`
}

// UnsupportedOp is one entry in the unsupported-ops list. Op types come from
// the lowered Node list; the reason comes from a curated map mirroring the
// Phase 4 conformance skip list (so the studio surfaces the same wording).
type UnsupportedOp struct {
	OpType string `json:"op_type"`
	Count  int    `json:"count"`
	Reason string `json:"reason"`
}

// ToJSON serializes the ImportSummary to canonical JSON bytes.
func (s *ImportSummary) ToJSON() ([]byte, error) { return json.Marshal(s) }

// BuildImportSummary parses the ONNX bytes via WithStructureOnly() and
// returns the summary payload. Errors surface as Go errors; the WASM
// wrapper converts them into a JSON {"error":...} body.
//
// The graph_id is the first 16 hex chars of sha256(modelBytes); the same
// bytes always produce the same id. Studio uses the id as the
// sessionStorage key for the per-tab imported-graph registry, so the
// visualize/kernels deep links can resolve it on a subsequent page load.
func BuildImportSummary(modelBytes []byte) (*ImportSummary, error) {
	if len(modelBytes) == 0 {
		return nil, fmt.Errorf("annealImportONNX: empty model bytes")
	}

	sum := sha256.Sum256(modelBytes)
	graphID := "imported-" + hex.EncodeToString(sum[:8])

	arena := uop.NewArena(4096)
	r, err := onnx.Import(modelBytes, arena, "wasm", onnx.WithStructureOnly())
	if err != nil {
		return nil, fmt.Errorf("annealImportONNX: %w", err)
	}

	// Walk the lowered node list and bucket op types that have no registered
	// handler against the curated reason map.
	unsupported := collectUnsupportedOps(r)

	out := &ImportSummary{
		GraphID:          graphID,
		Graph:            json.RawMessage(`{"name":"` + graphID + `","nodes":[],"edges":[],"stats":{"fwdNodes":0,"bwdNodes":0,"kernels":0,"allNodes":0}}`),
		Inputs:           importIOsFrom(r.Inputs()),
		Outputs:          importIOsFrom(r.Outputs()),
		NodeCount:        len(r.Nodes()),
		InitializerCount: len(r.Initializers()),
		UnsupportedOps:   unsupported,
		Opset:            r.Opset(),
		Note:             "structure only; payloads not materialised. visualize topology and dtypes; values are not executable.",
	}

	// Populate the imported graph JSON. We render a lightweight topology
	// document keyed off the lowered Node list — this is the visualize
	// payload the studio renders without ever touching the (absent) values.
	gjson, err := importedGraphJSON(graphID, r)
	if err == nil {
		out.Graph = gjson
	}

	return out, nil
}

// importIOsFrom converts onnx ValueInfo descriptors to the JSON-friendly
// ImportIO shape. Symbolic dims (shape.SymInt) render as -1 so the studio
// can surface "unknown" without needing to plumb the dim-param name through
// (W9 will lift the names in).
func importIOsFrom(vis []onnx.ValueInfo) []ImportIO {
	out := make([]ImportIO, 0, len(vis))
	for _, vi := range vis {
		ii := ImportIO{Name: vi.Name}
		if vi.DType != nil {
			ii.DType = dtypeStr(vi.DType)
		}
		if len(vi.Shape) > 0 {
			ii.Shape = make([]int64, len(vi.Shape))
			for i, s := range vi.Shape {
				if v, ok := s.ConstValue(); ok {
					ii.Shape[i] = v
				} else {
					ii.Shape[i] = -1
				}
			}
		}
		out = append(out, ii)
	}
	return out
}

// collectUnsupportedOps walks r.Nodes() and bins every op type whose name is
// not in the runner's registered handler set into UnsupportedOps. The reason
// string comes from importUnsupportedReason for known-deferred names; unknown
// names fall back to a generic "no handler registered" message so the studio
// always renders something.
func collectUnsupportedOps(r *onnx.Runner) []UnsupportedOp {
	if r == nil || len(r.Nodes()) == 0 {
		return nil
	}
	buckets := map[string]int{}
	for _, n := range r.Nodes() {
		if r.HasHandler(n.OpType) {
			continue
		}
		// Host-tier ops are also "supported" for structure-only viz; the
		// Runner is never asked to Run, so host-vs-device tiering is moot.
		if onnx.IsHostOp(n.OpType) {
			continue
		}
		buckets[n.OpType]++
	}
	if len(buckets) == 0 {
		return nil
	}
	out := make([]UnsupportedOp, 0, len(buckets))
	for op, count := range buckets {
		out = append(out, UnsupportedOp{
			OpType: op,
			Count:  count,
			Reason: importUnsupportedReason(op),
		})
	}
	// Sort for determinism (test stability).
	sort.Slice(out, func(i, j int) bool { return out[i].OpType < out[j].OpType })
	return out
}

// importUnsupportedReason returns a short, user-facing reason for an op type
// that anneal does not register a handler for. The list mirrors the major
// deferred items from the Phase 4 conformance skip list (notes/onnx_progress
// + onnx/conformance_skip.go), translated for the dropzone audience.
//
// Unknown op types get a generic fallback so the studio renders consistently.
func importUnsupportedReason(op string) string {
	if r, ok := importedOpReasons[op]; ok {
		return r
	}
	return "no handler registered yet (visualize only; cannot execute)"
}

var importedOpReasons = map[string]string{
	"Resize":            "Resize: deferred to v1.1 (one of the most spec-heavy ops)",
	"Upsample":          "Upsample: deferred to v1.1 (use Resize when available)",
	"Loop":              "Loop: control flow deferred to v1.1",
	"If":                "If: control flow deferred to v1.1",
	"Scan":              "Scan: control flow deferred to v1.1",
	"NonZero":           "NonZero: data-dependent shapes deferred to v1.1",
	"Unique":            "Unique: data-dependent shapes deferred to v1.1",
	"ScatterND":         "ScatterND: scatter family deferred to v1.1",
	"ScatterElements":   "ScatterElements: scatter family deferred to v1.1",
	"Scatter":           "Scatter: scatter family deferred to v1.1",
	"GatherND":          "GatherND: deferred to v1.1",
	"GatherElements":    "GatherElements: deferred to v1.1",
	"QLinearMatMul":     "Quantization out of v1 scope",
	"QLinearConv":       "Quantization out of v1 scope",
	"QuantizeLinear":    "Quantization out of v1 scope",
	"DequantizeLinear":  "Quantization out of v1 scope",
	"AveragePool":       "AveragePool: deferred (use GlobalAveragePool or punt to v1.1)",
	"LSTM":              "LSTM: recurrent ops deferred to v1.1",
	"GRU":               "GRU: recurrent ops deferred to v1.1",
	"RNN":               "RNN: recurrent ops deferred to v1.1",
	"Dropout":           "Dropout: inference-only is identity; train-time deferred",
	"TopK":              "TopK: deferred to v1.1",
	"OneHot":            "OneHot: deferred to v1.1",
	"NonMaxSuppression": "NonMaxSuppression: deferred to v1.1 (object detection)",
	"RoiAlign":          "RoiAlign: deferred to v1.1 (object detection)",
}

// importedGraphJSON renders a topology-only viz payload for the imported
// runner. Each lowered Node becomes a NodeData with op-type as label, the
// declared output dtype/shape (best-effort: we don't run shape inference
// here, so unknown shapes render empty), and edges drawn from each input-
// name's producer to the node.
//
// This is the v1 dropzone rendering — the studio's visualize view consumes
// the same GraphData shape it gets from BuildGraph(name), so the existing
// renderer drops in unchanged.
func importedGraphJSON(graphID string, r *onnx.Runner) (json.RawMessage, error) {
	type nd struct {
		ID    uint32  `json:"id"`
		Op    string  `json:"op"`
		DType string  `json:"dtype"`
		Shape []int64 `json:"shape,omitempty"`
		Class string  `json:"class"`
		Kind  string  `json:"kind"`
		Label string  `json:"label"`
	}
	type ed struct {
		From uint32 `json:"from"`
		To   uint32 `json:"to"`
	}
	type stats struct {
		FwdNodes int `json:"fwdNodes"`
		BwdNodes int `json:"bwdNodes"`
		Kernels  int `json:"kernels"`
		AllNodes int `json:"allNodes"`
	}
	type graph struct {
		Name  string `json:"name"`
		Nodes []nd   `json:"nodes"`
		Edges []ed   `json:"edges"`
		Stats stats  `json:"stats"`
	}

	nodes := r.Nodes()
	// Producer map: output name -> producing node index. We use 1-based ids
	// (the inputs reserve id 0) so a 0 id means "unresolved external".
	producer := make(map[string]uint32, len(nodes)*2)
	// First reserve ids for graph inputs (so edges from inputs anchor).
	idForInput := make(map[string]uint32, len(r.Inputs()))
	var nodeRecs []nd
	var edgeRecs []ed
	id := uint32(1)
	for _, in := range r.Inputs() {
		idForInput[in.Name] = id
		nodeRecs = append(nodeRecs, nd{
			ID:    id,
			Op:    "Input",
			DType: dtypeStrFor(in.DType),
			Class: ClassForward,
			Kind:  KindLeaf,
			Label: in.Name,
		})
		id++
	}

	for _, n := range nodes {
		thisID := id
		id++
		nodeRecs = append(nodeRecs, nd{
			ID:    thisID,
			Op:    n.OpType,
			Class: ClassForward,
			Kind:  KindDefault,
			Label: n.OpType,
		})
		for _, outName := range n.Outputs {
			if outName != "" {
				producer[outName] = thisID
			}
		}
		for _, inName := range n.Inputs {
			if inName == "" {
				continue
			}
			if from, ok := producer[inName]; ok {
				edgeRecs = append(edgeRecs, ed{From: from, To: thisID})
			} else if from, ok := idForInput[inName]; ok {
				edgeRecs = append(edgeRecs, ed{From: from, To: thisID})
			}
			// initializers and unresolved names go without an explicit edge
			// for v1 — the topology renders cleanly without dangling lines.
		}
	}

	g := graph{
		Name:  graphID,
		Nodes: nodeRecs,
		Edges: edgeRecs,
		Stats: stats{
			FwdNodes: len(nodeRecs),
			Kernels:  0,
			AllNodes: len(nodeRecs),
		},
	}
	return json.Marshal(g)
}

// dtypeStrFor is a nil-safe wrapper around viz.dtypeStr (graph.go).
func dtypeStrFor(dt *uop.DType) string {
	if dt == nil {
		return ""
	}
	return dtypeStr(dt)
}
