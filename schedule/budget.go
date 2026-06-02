package schedule

import (
	"fmt"
	"os"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// budgetDebug enables verbose tracing of the auto-Contiguous pass. Set the
// environment variable ANNEAL_BUDGET_DEBUG to any non-empty string to enable.
var budgetDebug = os.Getenv("ANNEAL_BUDGET_DEBUG") != ""

// MaxBuffersPerKernel is the WebGPU-mandated cap on storage-buffer bindings
// per kernel (Metal enforces the same limit per stage). One slot is reserved
// for the kernel's output BUFFER, leaving up to MaxBuffersPerKernel-1 leaf
// reads. The auto-Contiguous pass below inserts OpContiguous nodes so every
// realize point's leaf-buffer reach stays within this budget.
const MaxBuffersPerKernel = 8

// enforceBufferBudget walks the SINK-rooted tensor-level UOp graph and, for
// every node that will become a kernel boundary (sink children plus any node
// in hardRealizeOps), counts the distinct upstream OpBuffer + realize-point
// nodes reachable without crossing another realize point. If the count plus
// one (for the kernel's own output) exceeds MaxBuffersPerKernel, the pass
// picks a "best cut" sub-expression and wraps it in OpContiguous, turning it
// into a fresh realize point that splits the over-budget kernel.
//
// The pass runs BEFORE runRangeify so that the indexing/range-threading
// machinery treats every inserted OpContiguous like any other manual
// Contiguous() call: hardRealizeOps[OpContiguous]==true, indexExprNode
// dissolves OpContiguous transparently at the index level (it acts as a
// realize-point marker only).
//
// Why not post-rangeify or post-removeBufferize? The existing budget check in
// removeBufferize (schedule.go ~line 233) already prevents fusion from
// creating over-budget kernels, its accounting of leaf-buffer reach is
// correct. The over-budget kernels we see today come from runRangeify
// emitting a single BUFFERIZE whose body already references > 7 distinct
// OpBuffer leaves, because the input graph has no internal realize point
// between them. Inserting OpContiguous pre-rangeify is the cleanest way to
// introduce the missing realize points: it operates on the un-indexed graph
// where shape/topology are easiest to reason about and reuses runRangeify's
// own BUFFERIZE construction (with Removable: false implied by the
// OpReduceAxis check in runRangeify being false for OpContiguous, so the
// inserted boundary is hard).
//
// Heuristic ("balanced cut"; see chooseCut for tier-by-tier policy):
//
//   - For each over-budget realize point R, walk R's subgraph topologically
//     (stopping at OpBuffer and at other realize points, the same
//     boundaries runRangeify uses) and collect candidates.
//   - For each candidate N, compute leaves(N) (its own leaf-buffer reach)
//     and shed(N) = (leaves uniquely reachable via N) - 1. The -1 accounts
//     for N becoming a new leaf in R's kernel after cutting.
//   - Tier 1 ("balanced"): pick N with leaves(N) <= MaxBuffersPerKernel-1
//     AND shed(N) >= 1. This guarantees the *upstream* new kernel (rooted
//     at N) is itself within budget in one cut. Ranked by smallest
//     materialised tensor first (cheap to store), then largest shed
//     (largest downstream relief), then smallest arena index for
//     determinism.
//   - Tier 2 ("progress-only"): no tier-1 candidate exists (the over-budget
//     kernel is wider than any single cut can fix). Pick the candidate with
//     the largest shed; subsequent iterations will drain the upstream side.
//   - Tier 3 (fallback): no candidate has shed >= 1. Pick the one with the
//     largest reach contribution to at least unblock progress.
//   - Insert OpContiguous around the chosen node and rebuild the graph.
//   - Re-check; iterate. Cap at maxIters to prevent pathological loops.
//
// Why constrain tier 1 by leaves(N) <= MaxBuffersPerKernel-1 (= 7)? Without
// that constraint, ranking by shed alone always picks the deepest
// dominating node, which has the largest shed but also the largest
// leaves(N), leaving the upstream side STILL over budget. The next
// iteration then sees the new OpContiguous as the over-budget kernel and
// picks the SAME candidate, oscillating. The leaves(N) cap ensures every
// iteration produces at least one in-budget kernel.
func enforceBufferBudget(sink uop.UOp) uop.UOp {
	a := sink.Arena()

	// Cap iterations to avoid infinite loops if the heuristic stalls.
	const maxIters = 64
	for iter := 0; iter < maxIters; iter++ {
		topo := topoSort(sink)
		realize := buildRealizeMapForBudget(sink, topo)

		// Find the first over-budget realize point in topo order (stable).
		var bad uop.UOp
		badFound := false
		for _, u := range topo {
			if !realize[u.Index()] {
				continue
			}
			if isLeafForBudget(u) {
				continue
			}
			reach := countLeafReach(u, realize)
			if budgetDebug {
				fmt.Fprintf(os.Stderr, "[budget iter=%d] node idx=%d op=%s reach=%d\n", iter, u.Index(), u.Op(), reach)
			}
			if reach+1 > MaxBuffersPerKernel {
				bad = u
				badFound = true
				break
			}
		}
		if !badFound {
			return sink
		}
		if budgetDebug {
			fmt.Fprintf(os.Stderr, "[budget iter=%d] OVER-BUDGET node idx=%d op=%s\n", iter, bad.Index(), bad.Op())
		}

		// Find the best cut inside `bad`'s kernel subgraph.
		cut := chooseCut(bad, realize)
		if !cut.Valid() {
			if budgetDebug {
				fmt.Fprintf(os.Stderr, "[budget iter=%d] no defensible cut for node idx=%d; stopping\n", iter, bad.Index())
			}
			// No defensible cut: stop. The remaining over-budget kernel will
			// surface at execute time as the same error it does today, which
			// is the correct behaviour for pathological inputs.
			return sink
		}
		if budgetDebug {
			fmt.Fprintf(os.Stderr, "[budget iter=%d] CUT at idx=%d op=%s\n", iter, cut.Index(), cut.Op())
		}

		// Wrap cut in OpContiguous and rebuild every downstream node.
		wrapped := a.New(uop.OpContiguous, cut.DType(), []uop.UOp{cut}, nil, nil)
		sink = rebuildWithReplacement(a, sink, cut.Index(), wrapped)
	}
	return sink
}

// buildRealizeMapForBudget mirrors buildRealizeMap but operates on a freshly
// computed topo. Sink children plus every hardRealizeOps node are realize
// points. Keeping this function private to budget.go avoids any risk of
// drifting from the canonical buildRealizeMap in rangeify.go.
func buildRealizeMapForBudget(sink uop.UOp, topo []uop.UOp) map[uint32]bool {
	realize := make(map[uint32]bool)
	for i := 0; i < sink.NSrc(); i++ {
		realize[sink.Src(i).Index()] = true
	}
	for _, u := range topo {
		if hardRealizeOps[u.Op()] {
			realize[u.Index()] = true
		}
	}
	return realize
}

// isLeafForBudget reports whether u is a node that does NOT consume a storage
// binding when read (constants are inlined as WGSL literals; defines/ranges
// have no data).
func isLeafForBudget(u uop.UOp) bool {
	switch u.Op() {
	case uop.OpConst, uop.OpRange, uop.OpLUnique, uop.OpDevice, uop.OpDefineVar:
		return true
	}
	return false
}

// kernelBoundary returns true when traversal should stop at u (treating u as
// a leaf for the enclosing kernel's binding-count accounting).
//
// The set mirrors runRangeify's behaviour: OpBuffer is the canonical leaf;
// any node in the realize map is materialised by its own kernel and seen as
// a single buffer by downstream consumers.
func kernelBoundary(u uop.UOp, realize map[uint32]bool) bool {
	if u.Op() == uop.OpBuffer {
		return true
	}
	return realize[u.Index()]
}

// countLeafReach returns the number of distinct OpBuffer + upstream
// realize-point nodes reachable from root without crossing another such
// boundary. The root itself is excluded (it IS the kernel output we're
// budgeting for).
func countLeafReach(root uop.UOp, realize map[uint32]bool) int {
	leaves := collectLeaves(root, realize)
	return len(leaves)
}

// collectLeaves returns the set of distinct OpBuffer + upstream realize-point
// indices reachable from root, treating root itself as transparent (so its
// own srcs are walked) but stopping at any subsequent boundary.
func collectLeaves(root uop.UOp, realize map[uint32]bool) map[uint32]bool {
	out := make(map[uint32]bool)
	seen := make(map[uint32]bool)
	var walk func(u uop.UOp, isRoot bool)
	walk = func(u uop.UOp, isRoot bool) {
		if seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		if !isRoot && kernelBoundary(u, realize) {
			out[u.Index()] = true
			return
		}
		if isLeafForBudget(u) {
			return
		}
		for i := 0; i < u.NSrc(); i++ {
			walk(u.Src(i), false)
		}
	}
	walk(root, true)
	return out
}

// chooseCut implements the "best cut" heuristic. Returns an invalid UOp if
// no defensible cut exists.
//
// Two-tier selection:
//
//	Tier 1, "balanced cut": pick a candidate C with
//	  leaves(C) <= MaxBuffersPerKernel-1 AND shed(C) >= 1.
//	The first half-kernel (rooted at C) is immediately within budget; the
//	downstream half-kernel's reach drops by shed(C). Among tier-1 candidates,
//	choose by (smallest tensor size, largest shed, smallest arena index).
//
//	Tier 2, "progress-only cut": no balanced candidate exists (the kernel
//	is unusually wide; every cut leaves one side over budget). Pick the cut
//	with the LARGEST shed so the over-budget side shrinks fastest in
//	subsequent iterations. Iterative re-application eventually drains.
//
// Why "leaves(C) <= 7" as the primary admissibility test? Without it, the
// natural arena order (deeper nodes have smaller indices, intern earlier)
// makes the heuristic pick a candidate that just bumps an OpContiguous
// boundary one level outward without shedding leaves on either side. The
// upper bound formalises "this cut alone fixes the upstream half."
func chooseCut(root uop.UOp, realize map[uint32]bool) uop.UOp {
	rootLeaves := collectLeaves(root, realize)

	// Walk root's subgraph (interior, not including boundaries or the root
	// itself) and collect candidate cut nodes plus their reach sets.
	candidates := make([]uop.UOp, 0)
	candLeaves := make(map[uint32]map[uint32]bool)
	{
		seen := make(map[uint32]bool)
		var walk func(u uop.UOp, isRoot bool)
		walk = func(u uop.UOp, isRoot bool) {
			if seen[u.Index()] {
				return
			}
			seen[u.Index()] = true
			if !isRoot && kernelBoundary(u, realize) {
				return
			}
			if isLeafForBudget(u) {
				return
			}
			if !isRoot && isCutCandidate(u) {
				leaves := collectLeaves(u, realize)
				if len(leaves) >= 2 {
					candidates = append(candidates, u)
					candLeaves[u.Index()] = leaves
				}
			}
			for i := 0; i < u.NSrc(); i++ {
				walk(u.Src(i), false)
			}
		}
		walk(root, true)
	}

	if len(candidates) == 0 {
		return uop.UOp{}
	}

	// For each candidate, compute shed (unique-via-candidate minus 1).
	scoreOf := make(map[uint32]int)
	for _, c := range candidates {
		ablated := collectLeavesWithExtraBoundary(root, realize, c.Index())
		shed := 0
		for ld := range candLeaves[c.Index()] {
			if !ablated[ld] && rootLeaves[ld] {
				shed++
			}
		}
		scoreOf[c.Index()] = shed - 1
	}

	// Tier 1: leaves(C) <= MaxBuffersPerKernel-1 AND shed >= 1.
	bestIdx := -1
	for i, c := range candidates {
		if len(candLeaves[c.Index()]) > MaxBuffersPerKernel-1 {
			continue
		}
		if scoreOf[c.Index()] < 1 {
			continue
		}
		if bestIdx == -1 || cutLessTier1(c, candidates[bestIdx], scoreOf) {
			bestIdx = i
		}
	}
	if bestIdx >= 0 {
		return candidates[bestIdx]
	}

	// Tier 2: maximum shed regardless of leaves(C). Guarantees progress on
	// pathological inputs (e.g., the over-budget kernel is itself a chain of
	// wide reductions where no single cut alone fits both halves under cap).
	for i, c := range candidates {
		if scoreOf[c.Index()] < 1 {
			continue
		}
		if bestIdx == -1 || cutLessTier2(c, candidates[bestIdx], scoreOf) {
			bestIdx = i
		}
	}
	if bestIdx >= 0 {
		return candidates[bestIdx]
	}

	// Tier 3 (fallback): no candidate has shed >= 1. Pick the one with the
	// LARGEST reach contribution; even with shed=0 (no net buffer savings)
	// this often unlocks subsequent cuts.
	for i, c := range candidates {
		if bestIdx == -1 || len(candLeaves[c.Index()]) > len(candLeaves[candidates[bestIdx].Index()]) {
			bestIdx = i
		}
	}
	if bestIdx == -1 {
		return uop.UOp{}
	}
	return candidates[bestIdx]
}

// cutLessTier1 ranks tier-1 candidates: smallest materialised tensor first,
// then largest shed, then largest leaves(C) (= deepest cut that still fits
// under cap), then smallest arena index as a stable tiebreaker.
func cutLessTier1(a, b uop.UOp, scoreOf map[uint32]int) bool {
	szA := candidateSize(a)
	szB := candidateSize(b)
	if szA != szB {
		return szA < szB
	}
	sA := scoreOf[a.Index()]
	sB := scoreOf[b.Index()]
	if sA != sB {
		return sA > sB
	}
	return a.Index() < b.Index()
}

// cutLessTier2 ranks tier-2 candidates: largest shed first (drains fastest),
// then smallest size, then smallest index for stability.
func cutLessTier2(a, b uop.UOp, scoreOf map[uint32]int) bool {
	sA := scoreOf[a.Index()]
	sB := scoreOf[b.Index()]
	if sA != sB {
		return sA > sB
	}
	szA := candidateSize(a)
	szB := candidateSize(b)
	if szA != szB {
		return szA < szB
	}
	return a.Index() < b.Index()
}

// candidateSize returns the product of u's shape dims. Symbolic dims count as
// a sentinel "large" weight so we prefer concrete-shape cuts over symbolic-
// shape cuts when both are viable. Pure-scalar nodes (size 1) are cheap and
// preferred.
func candidateSize(u uop.UOp) int64 {
	cache := make(map[uint32][]shape.Sint)
	// Build shape cache by topo-walking u's subgraph.
	topo := topoSort(u)
	for _, n := range topo {
		shapeOfNode(n, cache)
	}
	sh := cache[u.Index()]
	if sh == nil {
		return 1
	}
	var size int64 = 1
	const symWeight int64 = 1 << 40 // larger than any realistic concrete dim
	for _, d := range sh {
		if v, ok := d.ConstValue(); ok {
			if v <= 0 {
				v = 1
			}
			// Guard against int64 overflow.
			if size > (1<<60)/v {
				return 1 << 62
			}
			size *= v
		} else {
			size *= symWeight
			if size < 0 {
				return 1 << 62
			}
		}
	}
	return size
}

// collectLeavesWithExtraBoundary is like collectLeaves but additionally treats
// the node at extraBoundary as a hard boundary (counts it as a leaf). Used to
// compute "what root's leaf set would look like if candidate c were already
// realized", the set of leaves NOT in that result but present in
// collectLeaves(root) is the set of leaves uniquely reachable via c.
func collectLeavesWithExtraBoundary(root uop.UOp, realize map[uint32]bool, extraBoundary uint32) map[uint32]bool {
	out := make(map[uint32]bool)
	seen := make(map[uint32]bool)
	var walk func(u uop.UOp, isRoot bool)
	walk = func(u uop.UOp, isRoot bool) {
		if seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		if !isRoot && (kernelBoundary(u, realize) || u.Index() == extraBoundary) {
			out[u.Index()] = true
			return
		}
		if isLeafForBudget(u) {
			return
		}
		for i := 0; i < u.NSrc(); i++ {
			walk(u.Src(i), false)
		}
	}
	walk(root, true)
	return out
}

// isCutCandidate reports whether u is eligible to be wrapped in OpContiguous.
// Excludes: existing realize points (already a barrier), leaf nodes
// (OpBuffer / constants / ranges), and meta nodes (LUnique / Device / Sink).
// Permitted: ALU ops, movement ops, Cast/Bitcast, Where, Index.
func isCutCandidate(u uop.UOp) bool {
	if isLeafForBudget(u) {
		return false
	}
	switch u.Op() {
	case uop.OpBuffer, uop.OpBufferize,
		uop.OpContiguous, uop.OpContiguousBackward,
		uop.OpAssign, uop.OpBufferView, uop.OpEncDec,
		uop.OpReduceAxis,
		uop.OpSink, uop.OpCall, uop.OpEnd, uop.OpAfter, uop.OpStore, uop.OpParam,
		uop.OpLUnique, uop.OpDevice:
		return false
	}
	return true
}

// rebuildWithReplacement returns a copy of sink where every reference to
// oldIdx is replaced with replacement. Upstream nodes are unchanged; nodes on
// any path from oldIdx to sink are rebuilt with src updated. The arena's
// hash-consing folds duplicates automatically.
func rebuildWithReplacement(a *uop.Arena, sink uop.UOp, oldIdx uint32, replacement uop.UOp) uop.UOp {
	topo := topoSort(sink)
	rebuild := make(map[uint32]uint32, len(topo))
	rebuild[oldIdx] = replacement.Index()

	for _, u := range topo {
		if u.Index() == oldIdx {
			continue
		}
		srcs := make([]uop.UOp, u.NSrc())
		changed := false
		for i := 0; i < u.NSrc(); i++ {
			ch := u.Src(i)
			if newIdx, ok := rebuild[ch.Index()]; ok {
				srcs[i] = a.At(newIdx)
				if newIdx != ch.Index() {
					changed = true
				}
			} else {
				srcs[i] = ch
			}
		}
		if changed {
			rebuild[u.Index()] = a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag()).Index()
		} else {
			rebuild[u.Index()] = u.Index()
		}
	}
	if nIdx, ok := rebuild[sink.Index()]; ok {
		return a.At(nIdx)
	}
	return sink
}
