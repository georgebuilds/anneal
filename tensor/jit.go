package tensor

import (
	"fmt"
	"sync/atomic"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// JIT captures an execution plan on the first Realize call and replays it on
// subsequent calls with updated leaf data.
//
// # Arena boundary contract — Design B (stable-identity remap)
//
// Each training step creates a fresh arena (a := uop.NewArena(...)), so the
// ExecItem.Bufs[*].UOpIdx values recorded from the capture step are indices in
// the capture arena, not in any later arena.  JIT resolves this as follows:
//
//  1. Capture: DFS-walk the tensor graph to collect all OpBuffer leaves with
//     data, in deterministic visit order (preorder DFS over sources, same for
//     every structurally-identical graph).  Record each leaf's capture-arena
//     UOpIdx and element count.
//
//  2. Replay: DFS-walk the fresh tensors in the same way, producing the same
//     number of leaves in the same ordinal order.  Map captured slot i →
//     fresh leaf i: replayInputs[capturedLeaves[i].uopIdx] = freshLeaves[i].data.
//     Then call Run(capturedItems, replayInputs) directly — no sink construction,
//     no scheduling, no leaf DFS for the executor.
//
// Positional mapping is bijective because DFS order is determined by graph
// topology, which is invariant across steps for a fixed model.  The caller
// ensures leaf data is current (param.Load sets fresh leaf data each step).
//
// # Match contract
//
// Replay fires iff (a) leaf count matches capture and (b) for static (non-
// symbolic) schedules, every leaf element count also matches.  Any mismatch
// triggers a fresh capture.  Stale plans are never replayed.
//
// # Scope
//
// In-process capture/replay only; no plan serialisation.  Dispatch always goes
// through the existing DefaultExecutor (preserving the GPU-owner-goroutine
// invariant for the Metal backend).
type JIT struct {
	captured bool
	symbolic bool // true when capture used RealizeWithBinding

	// Frozen plan — no arena references (Ast zeroed, UOpIdxs are plain uint32).
	items        []schedule.ExecItem
	capLeaves    []capturedLeaf   // leaves in DFS-visit order from capture arena
	capOuts      []capturedOutput // final-output UOpIdx → tensors-slice position
	device       string
	leafCount    int
	leafSizes    []int   // element counts in DFS order; checked only when !symbolic
	capSK        uint64  // structural key of captured output expression(s)

	nCaptures atomic.Int64
	nReplays  atomic.Int64
}

// capturedLeaf records the capture-arena UOpIdx for one DFS-ordered leaf slot.
type capturedLeaf struct {
	uopIdx uint32
}

// capturedOutput maps one final-output buffer (by capture-arena UOpIdx) to
// the position of its tensor in the tensors slice passed to Realize/RealizeWithBinding.
type capturedOutput struct {
	uopIdx    uint32
	tensorPos int
}

// NewJIT returns a ready-to-use JIT handle.
func NewJIT() *JIT { return &JIT{} }

// JITStats returns the cumulative (captures, replays) counts.
func (j *JIT) JITStats() (captures, replays int64) {
	return j.nCaptures.Load(), j.nReplays.Load()
}

// Realize is a drop-in for tensor.Realize: captures on the first call and
// replays on subsequent calls with matching graph structure.
func (j *JIT) Realize(tensors ...*Tensor) error {
	if len(tensors) == 0 {
		return nil
	}
	fl := jitDFSLeaves(tensors)
	if !j.captured || j.symbolic {
		return j.captureStatic(tensors, fl)
	}
	sk := subgraphSK(tensors)
	if !j.matchStatic(fl, sk) {
		return j.captureStatic(tensors, fl)
	}
	return j.replayStatic(fl, tensors)
}

// RealizeWithBinding is a drop-in for tensor.RealizeWithBinding: captures on
// the first call and replays on subsequent calls.  The binding may differ
// between calls (e.g. symbolic batch size may change); the match guard checks
// only leaf count, not leaf sizes.
func (j *JIT) RealizeWithBinding(binding map[string]int64, tensors ...*Tensor) error {
	if len(tensors) == 0 {
		return nil
	}
	fl := jitDFSLeaves(tensors)
	if !j.captured || !j.symbolic {
		return j.captureSym(binding, tensors, fl)
	}
	sk := subgraphSK(tensors)
	if !j.matchSym(fl, sk) {
		return j.captureSym(binding, tensors, fl)
	}
	return j.replaySym(fl, binding, tensors)
}

// ── Capture paths ──────────────────────────────────────────────────────────────

func (j *JIT) captureStatic(tensors []*Tensor, fl []jitLeaf) error {
	if err := Realize(tensors...); err != nil {
		return err
	}
	return j.storeCapture(tensors, fl, false)
}

func (j *JIT) captureSym(binding map[string]int64, tensors []*Tensor, fl []jitLeaf) error {
	if err := RealizeWithBinding(binding, tensors...); err != nil {
		return err
	}
	return j.storeCapture(tensors, fl, true)
}

// storeCapture records the execution plan after a successful Realize call.
// It re-builds the same sink (interned → cache hit within this arena), retrieves
// the items from the arena-local schedule cache, and stores them on j.
func (j *JIT) storeCapture(tensors []*Tensor, fl []jitLeaf, symbolic bool) error {
	a := tensors[0].arena()
	device := tensors[0].device

	// Rebuild the sink — because OpSink is interned, the same srcs → same index
	// within this arena → CreateSchedule below is guaranteed to be a cache hit.
	srcs := make([]uop.UOp, len(tensors))
	for i, t := range tensors {
		srcs[i] = t.node
	}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)

	// Cache hit: items already computed and stored by the Realize call above.
	// Items have Ast zeroed and WGSL pre-rendered by cacheStore, so they hold
	// no arena references and will survive arena GC between training steps.
	items := schedule.CreateSchedule(sink, device)

	cl := make([]capturedLeaf, len(fl))
	ls := make([]int, len(fl))
	for i, l := range fl {
		cl[i] = capturedLeaf{uopIdx: l.idx}
		ls[i] = len(l.data)
	}

	j.items = items
	j.capLeaves = cl
	j.capOuts = jitOutputMapping(tensors, items)
	j.device = device
	j.symbolic = symbolic
	j.leafCount = len(fl)
	j.leafSizes = ls
	j.capSK = subgraphSK(tensors)
	j.captured = true
	j.nCaptures.Add(1)
	return nil
}

// ── Replay paths ───────────────────────────────────────────────────────────────

func (j *JIT) replayStatic(fl []jitLeaf, tensors []*Tensor) error {
	outputs, err := DefaultExecutor.Run(j.items, j.remapInputs(fl))
	if err != nil {
		return fmt.Errorf("tensor: JIT replay: %w", err)
	}
	j.applyOutputs(tensors, outputs)
	j.nReplays.Add(1)
	return nil
}

func (j *JIT) replaySym(fl []jitLeaf, binding map[string]int64, tensors []*Tensor) error {
	exec, ok := DefaultExecutor.(backend.SymbolicExecutor)
	if !ok {
		return fmt.Errorf("tensor: JIT.RealizeWithBinding: executor does not implement SymbolicExecutor")
	}
	outputs, err := exec.RunSymbolic(j.items, j.remapInputs(fl), binding)
	if err != nil {
		return fmt.Errorf("tensor: JIT symbolic replay: %w", err)
	}
	j.applyOutputs(tensors, outputs)
	j.nReplays.Add(1)
	return nil
}

// remapInputs builds the executor inputs map for replay.
// Positional mapping: capturedLeaves[i].uopIdx → freshLeaves[i].data.
// The executor looks up inputs[buf.UOpIdx] for each leaf buffer in the
// captured schedule; using the capture-arena indices as keys is correct
// because those are exactly the UOpIdx values in capturedItems.
func (j *JIT) remapInputs(fl []jitLeaf) map[uint32][]float32 {
	m := make(map[uint32][]float32, len(j.capLeaves))
	for i, cl := range j.capLeaves {
		if i < len(fl) {
			m[cl.uopIdx] = fl[i].data
		}
	}
	return m
}

func (j *JIT) applyOutputs(tensors []*Tensor, outputs map[uint32][]float32) {
	for _, co := range j.capOuts {
		if d, ok := outputs[co.uopIdx]; ok {
			tensors[co.tensorPos].data = d
		}
	}
}

// ── Match guards ───────────────────────────────────────────────────────────────

// matchStatic checks that the fresh leaves match the capture snapshot.
// For static (non-symbolic) schedules, both count and element sizes must match,
// and the output structural key must be unchanged from capture.
func (j *JIT) matchStatic(fl []jitLeaf, sk uint64) bool {
	if len(fl) != j.leafCount || sk != j.capSK {
		return false
	}
	for i, l := range fl {
		if len(l.data) != j.leafSizes[i] {
			return false
		}
	}
	return true
}

// matchSym checks that the leaf count and output structural key match capture.
// Element sizes are allowed to differ (symbolic dims change with each binding).
func (j *JIT) matchSym(fl []jitLeaf, sk uint64) bool {
	return len(fl) == j.leafCount && sk == j.capSK
}

// ── Helpers ────────────────────────────────────────────────────────────────────

// jitLeaf holds a leaf buffer's arena index and its current data.
type jitLeaf struct {
	idx  uint32
	data []float32
}

// subgraphSK returns a combined structural key for the output expressions of
// tensors. It calls uop.StructuralKeys to compute position-independent hashes
// for every node in the arena, then mixes the keys of the output tensor nodes.
//
// Two calls return the same value iff every output tensor has a structurally
// identical UOp subgraph (same ops, dtypes, tree shape, and arg values in each
// position) — regardless of arena-local node indices or leaf data.
//
// Because structural keys are computed bottom-up over the arena's node slice,
// and tensor output nodes always precede any scheduler-added nodes (which have
// higher indices), this function is safe to call after storeCapture has run
// Realize (which adds scheduling nodes to the arena).
func subgraphSK(tensors []*Tensor) uint64 {
	if len(tensors) == 0 {
		return 0
	}
	keys := uop.StructuralKeys(tensors[0].arena())
	const prime uint64 = 1099511628211
	h := uint64(14695981039346656037)
	for i, t := range tensors {
		h = (h ^ uint64(i)) * prime
		idx := t.node.Index()
		if int(idx) < len(keys) {
			h = (h ^ keys[idx]) * prime
		}
	}
	return h
}

// jitDFSLeaves DFS-walks the UOp graph rooted at each tensor and collects
// OpBuffer leaves that have data registered, in deterministic preorder DFS visit
// order.  DFS order is determined solely by graph topology: for structurally
// identical graphs across different arenas (different node indices), the ordinal
// position of each leaf in the returned slice is invariant.  This invariance is
// the foundation of Design B's positional remap: capture captures positions,
// replay re-resolves positions from fresh tensors.
func jitDFSLeaves(tensors []*Tensor) []jitLeaf {
	var out []jitLeaf
	seen := make(map[uint32]bool)
	var walk func(u uop.UOp)
	walk = func(u uop.UOp) {
		idx := u.Index()
		if seen[idx] {
			return
		}
		seen[idx] = true
		if u.Op() == uop.OpBuffer {
			if data, ok := u.Arena().Leaf(idx); ok {
				out = append(out, jitLeaf{idx: idx, data: data})
				return
			}
		}
		for i := 0; i < u.NSrc(); i++ {
			walk(u.Src(i))
		}
	}
	for _, t := range tensors {
		walk(t.node)
	}
	return out
}

// jitOutputMapping replicates assignOutputs' logic to record, at capture time,
// which final-output buffer UOpIdx maps to which tensor position.
// On replay, applyOutputs uses this mapping to write output data into the fresh
// tensors' data fields without re-running assignOutputs.
func jitOutputMapping(tensors []*Tensor, items []schedule.ExecItem) []capturedOutput {
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}
	var finalOuts []uint32
	for _, item := range items {
		if idx := item.Bufs[0].UOpIdx; !readByAny[idx] {
			finalOuts = append(finalOuts, idx)
		}
	}
	var result []capturedOutput
	outSlot := 0
	for i, t := range tensors {
		if t.IsLeaf() {
			continue
		}
		if outSlot >= len(finalOuts) {
			break
		}
		result = append(result, capturedOutput{uopIdx: finalOuts[outSlot], tensorPos: i})
		outSlot++
	}
	return result
}
