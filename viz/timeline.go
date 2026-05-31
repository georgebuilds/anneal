package viz

// timeline.go builds a multi-stage "scrub-timeline" snapshot of one compile.
//
// Each stage is a discrete compiler-pipeline checkpoint (forward construction,
// gradient pass, scheduling, fusion). The TimelineData payload contains the
// *union* of nodes/edges across all stages plus per-stage masks describing
// which nodes are visible and how each visible node should be classified at
// that stage.
//
// Design intent (DESIGN.md §3, §7):
//   - Node positions are shared across stages so the eye tracks the transition,
//     not the reshuffle (DESIGN.md §7: "the visual transitions are the teaching
//     value").
//   - Stage transitions surface the DD1 colour story: teal (forward) → ember
//     (backward appears) → gold (fused kernel boundaries materialise).
//
// The compiler is the source of truth. Nothing here patches or guesses; every
// stage's data comes from a real compile via tensor.BackwardWithTrace and
// schedule.CreateSchedule.

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// StageData captures one compiler-pipeline checkpoint as a per-node overlay
// over the union graph. NodeOverrides maps arena index → per-stage rendering
// hints. Nodes absent from NodeOverrides are *hidden* at this stage.
type StageData struct {
	ID          string                  `json:"id"`          // short slug, e.g. "forward"
	Label       string                  `json:"label"`       // sentence-case display label
	Description string                  `json:"description"` // one-line subtitle for the UI
	Stats       StageStats              `json:"stats"`
	Overrides   map[uint32]NodeOverride `json:"overrides"`
}

// StageStats are the compiler-stat counters surfaced for this stage. Zero is a
// valid value ("not yet meaningful at this stage") — the UI suppresses zero
// fields.
type StageStats struct {
	FwdNodes int `json:"fwdNodes"`
	BwdNodes int `json:"bwdNodes"`
	Kernels  int `json:"kernels"`
	Fused    int `json:"fused"` // kernels eliminated by fusion at this stage
}

// NodeOverride overlays the union-node's classification for one stage.
// Class/Kind reuse the same constants as NodeData ("forward", "backward",
// "default", "leaf", "reduce", "sink") plus the gold-only "fused" kind for
// nodes inside a fused kernel boundary. Empty fields fall back to the union
// node's defaults.
type NodeOverride struct {
	Class    string `json:"class,omitempty"`
	Kind     string `json:"kind,omitempty"`
	KernelID int    `json:"kernelId,omitempty"` // 1-based kernel membership, 0 = unassigned
}

// TimelineData is the serialisable payload for the scrub-timeline UI.
type TimelineData struct {
	Name   string     `json:"name"`
	Nodes  []NodeData `json:"nodes"` // union across all stages
	Edges  []EdgeData `json:"edges"` // union across all stages
	Stages []StageData `json:"stages"`
}

// ToJSON serialises t.
func (t *TimelineData) ToJSON() ([]byte, error) { return json.Marshal(t) }

// BuildTimeline runs the real compiler once for the named example and
// extracts per-stage snapshots from the same arena. The forward graph and
// the post-gradient graph share that arena; the scheduled-stage kernel
// counts are measured in fresh arenas so each fusion setting starts from a
// clean slate (see countKernels).
//
// Stages (canonical order):
//
//	0  forward    — loss expression, all teal
//	1  gradient   — + backward nodes (ember)
//	2  scheduled  — reduce nodes promoted to gold kernel boundaries
//
// The brief originally specified a separate pre-fusion / post-fusion split.
// We collapse that into one "scheduled" stage and surface both unfused and
// post-fusion kernel counts in its Stats (`Kernels`, `Fused`) — fusion now
// fires in this codebase, so Fused > 0 is the common case and a separate
// post-fusion stage would only differ in its stats line, not the visual.
// See SPEC §1.3 / the AGENTS rule: skip stages that are visually identical
// to their neighbours.
func BuildTimeline(name string) (*TimelineData, error) {
	ex, err := examples.Get(name)
	if err != nil {
		return nil, err
	}
	result, err := ex.Build("webgpu")
	if err != nil {
		return nil, fmt.Errorf("viz: timeline build %q: %w", name, err)
	}

	a := result.Arena
	out := result.Output

	// ── Stage 0: forward graph (loss only, pre-Backward) ──────────────────
	// MSE-style loss with a scalar post-reduce multiply. The final `.Mul(scale)`
	// is what unlocks epilogue fusion in the post-schedule stage: the Sum
	// produces a Removable BUFFERIZE whose sole consumer is the SINK source
	// BUFFERIZE wrapping the elementwise scale-multiply, which has no reduce
	// of its own. A bare `out.Sum(...)` instead has SINK consume Sum directly
	// (no intermediate elementwise BUFFERIZE), and fusion can't fire.
	n := int64(1)
	for _, d := range out.Shape() {
		n *= d
	}
	tgt := tensor.NewLeaf(a, out.Shape(), out.DType(), result.Device)
	tgt.SetData(make([]float32, n))
	diff := out.Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(n), out.DType(), result.Device)
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)
	forwardRoots := []uop.UOp{loss.Node()}
	forwardTopo := topoSortMultiRoot(forwardRoots)

	// ── Stage 1: gradient pass (BackwardWithTrace) ────────────────────────
	var grads map[*tensor.Tensor]*tensor.Tensor
	var trace *tensor.GradTrace
	if len(result.Leaves) > 0 {
		grads, trace = tensor.BackwardWithTrace(loss, result.Leaves)
	}

	gradRoots := make([]uop.UOp, 0, 1+len(grads))
	gradRoots = append(gradRoots, loss.Node())
	for _, g := range grads {
		gradRoots = append(gradRoots, g.Node())
	}
	gradTopo := topoSortMultiRoot(gradRoots)

	// ── Stages 2/3: scheduler kernel counts, pre/post fusion ──────────────
	// Each measurement uses a fresh arena. Calling CreateSchedule twice on
	// the same arena would let the first pass's added BUFFERIZE nodes leak
	// into the second pass's analysis (the second SINK still resolves to
	// the original output node, whose now-scheduled descendants pick up
	// extra consumers from the first run's residual nodes), which silently
	// suppresses fusion. Build a fresh model per measurement instead.
	unfusedKernels := countKernels(ex, false)
	fusedKernels := countKernels(ex, true)

	// ── Build the union node/edge tables from gradTopo ────────────────────
	// gradTopo is the superset (forward ∪ backward). Stage 0 hides backward
	// nodes via its Overrides map; Stages 2/3 reclassify reduce nodes as gold.
	nodes, edges := buildUnion(a, gradTopo)

	// ── Per-stage attribution maps ────────────────────────────────────────
	// Forward attribution: every node in forwardTopo is "forward" for stage 0.
	// Stage 0 hides anything not in forwardTopo by omitting it from Overrides.
	forwardSet := make(map[uint32]bool, len(forwardTopo))
	for _, u := range forwardTopo {
		forwardSet[u.Index()] = true
	}

	// Backward attribution from the trace: maps the forward node a rule fired
	// on → the seq number. Used only for the optional gradRule label in the UI.
	attribution := make(map[uint32]ruleAttribution)
	if trace != nil {
		for _, ev := range trace.Events {
			for _, idx := range ev.ProducedIdx {
				if idx != tensor.TraceSentinel {
					if _, exists := attribution[idx]; !exists {
						attribution[idx] = ruleAttribution{ev.ForwardOp.String(), ev.Seq}
					}
				}
			}
		}
	}

	stages := []StageData{
		buildForwardStage(forwardTopo, forwardSet, a),
		buildGradientStage(gradTopo, forwardSet, a, attribution),
		buildScheduledStage(gradTopo, forwardSet, a, attribution, unfusedKernels, fusedKernels),
	}

	return &TimelineData{
		Name:   name,
		Nodes:  nodes,
		Edges:  edges,
		Stages: stages,
	}, nil
}

// ruleAttribution is a per-node (gradRule, seq) pair from the GradTrace.
type ruleAttribution struct {
	rule string
	seq  int
}

// buildUnion produces the union node and edge tables from topo. Each node's
// Class / Kind / GradRule reflect its *final* state (post-gradient); per-stage
// overrides narrow this down.
func buildUnion(a *uop.Arena, topo []uop.UOp) ([]NodeData, []EdgeData) {
	nodeSet := make(map[uint32]bool, len(topo))
	for _, u := range topo {
		nodeSet[u.Index()] = true
	}

	nodes := make([]NodeData, 0, len(topo))
	edges := make([]EdgeData, 0, len(topo)*2)

	for _, u := range topo {
		class := ClassForward
		if a.Provenance(u.Index()) == uop.PhaseBackward {
			class = ClassBackward
		}
		nodes = append(nodes, NodeData{
			ID:    u.Index(),
			Op:    u.Op().String(),
			DType: dtypeStr(u.DType()),
			Shape: bufShape(u),
			Class: class,
			Kind:  kindOf(u, a),
			Label: nodeLabel(u),
			Arg:   argStr(u),
		})
		for i := 0; i < u.NSrc(); i++ {
			src := u.Src(i)
			if nodeSet[src.Index()] {
				edges = append(edges, EdgeData{From: src.Index(), To: u.Index()})
			}
		}
	}
	sortEdges(edges)
	return nodes, edges
}

// buildForwardStage shows only forward-construction nodes — everything teal,
// reduces are still teal diamonds (kernel-boundary semantics haven't been
// computed yet). Stats: forward count only.
func buildForwardStage(forwardTopo []uop.UOp, forwardSet map[uint32]bool, a *uop.Arena) StageData {
	overrides := make(map[uint32]NodeOverride, len(forwardTopo))
	for _, u := range forwardTopo {
		// At this stage every visible node is forward, even reduces — the
		// "kernel boundary" classification doesn't exist yet.
		ov := NodeOverride{Class: ClassForward, Kind: kindOf(u, a)}
		// Demote reduce-shape to default so the visual doesn't pre-claim
		// kernel boundaries (gold) before scheduling runs.
		if ov.Kind == KindReduce {
			ov.Kind = KindDefault
		}
		overrides[u.Index()] = ov
	}
	return StageData{
		ID:          "forward",
		Label:       "forward",
		Description: "forward construction — loss expression, no gradients yet",
		Stats:       StageStats{FwdNodes: len(forwardTopo)},
		Overrides:   overrides,
	}
}

// buildGradientStage shows forward + backward nodes; backward nodes inherit
// their durable PhaseBackward provenance and render ember. Reduces still
// render default-shaped (kernel boundaries not yet known).
func buildGradientStage(gradTopo []uop.UOp, forwardSet map[uint32]bool, a *uop.Arena, attribution map[uint32]ruleAttribution) StageData {
	overrides := make(map[uint32]NodeOverride, len(gradTopo))
	var fwd, bwd int
	for _, u := range gradTopo {
		class := ClassForward
		if !forwardSet[u.Index()] || a.Provenance(u.Index()) == uop.PhaseBackward {
			class = ClassBackward
		}
		if class == ClassForward {
			fwd++
		} else {
			bwd++
		}
		kind := kindOf(u, a)
		if kind == KindReduce {
			kind = KindDefault
		}
		overrides[u.Index()] = NodeOverride{Class: class, Kind: kind}
	}
	return StageData{
		ID:          "gradient",
		Label:       "gradient",
		Description: "after BackwardWithTrace — adjoint rules wove the backward subgraph",
		Stats:       StageStats{FwdNodes: fwd, BwdNodes: bwd},
		Overrides:   overrides,
	}
}

// buildScheduledStage promotes reduce-kind nodes to gold (kernel boundaries
// have been determined). Surfaces both unfused and post-fusion kernel counts
// in stats so the user sees how many epilogues fused; Fused = unfused -
// post-fusion.
func buildScheduledStage(gradTopo []uop.UOp, forwardSet map[uint32]bool, a *uop.Arena, attribution map[uint32]ruleAttribution, unfusedKernels, fusedKernels int) StageData {
	overrides := make(map[uint32]NodeOverride, len(gradTopo))
	var fwd, bwd int
	for _, u := range gradTopo {
		class := ClassForward
		if !forwardSet[u.Index()] || a.Provenance(u.Index()) == uop.PhaseBackward {
			class = ClassBackward
		}
		if class == ClassForward {
			fwd++
		} else {
			bwd++
		}
		// Reduce nodes go gold (kindOf already returns KindReduce). Leaf /
		// default kinds are unchanged.
		overrides[u.Index()] = NodeOverride{Class: class, Kind: kindOf(u, a)}
	}
	fused := unfusedKernels - fusedKernels
	if fused < 0 {
		fused = 0
	}
	return StageData{
		ID:          "scheduled",
		Label:       "scheduled",
		Description: "rangeify identified kernel boundaries — reduce nodes materialise to gold buffers",
		Stats:       StageStats{FwdNodes: fwd, BwdNodes: bwd, Kernels: fusedKernels, Fused: fused},
		Overrides:   overrides,
	}
}

// countKernels rebuilds the example in a fresh arena, threads the same
// MSE-style loss + Backward pass over the result, and returns the number of
// kernel-graph items produced under the given fusion setting. A fresh build
// is required because schedule.CreateSchedule adds BUFFERIZE/BUFFER nodes
// into the arena; reusing an already-scheduled arena would leak extra
// consumers into the next analysis and silently suppress fusion.
func countKernels(ex *examples.Example, fusion bool) int {
	result, err := ex.Build("webgpu")
	if err != nil {
		return 0
	}
	a := result.Arena
	out := result.Output

	n := int64(1)
	for _, d := range out.Shape() {
		n *= d
	}
	tgt := tensor.NewLeaf(a, out.Shape(), out.DType(), result.Device)
	tgt.SetData(make([]float32, n))
	diff := out.Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(n), out.DType(), result.Device)
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

	roots := []uop.UOp{loss.Node()}
	if len(result.Leaves) > 0 {
		grads := tensor.Backward(loss, result.Leaves)
		for _, l := range result.Leaves {
			if g, ok := grads[l]; ok {
				roots = append(roots, g.Node())
			}
		}
	}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, roots, nil, nil)

	prev := schedule.FusionEnabled
	schedule.FusionEnabled = fusion
	items := schedule.CreateSchedule(sink, "webgpu")
	schedule.FusionEnabled = prev
	return len(items)
}

// sortStageOverrides is a noop today but keeps a stable iteration order if a
// future stage emits per-node ordering hints.
var _ = sort.SliceStable
