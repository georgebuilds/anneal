package tensor

import (
	"fmt"
	"sort"
	"sync"

	"github.com/georgebuilds/anneal/uop"
)

// ── Scatter-add host-side preprocessor registry ──────────────────────────────
//
// OpScatterAdd's backward kernel (Slice D) consumes two leaf buffers derived
// from the runtime idx tensor: sortedIdx and permutation. These are not
// constant graph inputs; their values depend on idx's data and must be
// recomputed every time idx changes.
//
// At Backward time the gradient rule for OpGather allocates the sortedIdx /
// permutation leaves and registers a preprocessor closure here. Just before
// Realize hands the schedule to the executor (and on every JIT replay), the
// runtime calls RunScatterPreprocessors which fires every closure registered
// against the arena, reading the current idx leaf data and writing fresh
// sortedIdx / permutation arrays via Arena.SetLeaf.
//
// The closure captures arena indices (not UOp handles or data slices) so a
// long-lived registration stays valid across multiple Realize calls. A nil
// idx-leaf data slice (idx never had SetData called) is reported via
// preprocFn returning false; Realize panics with a clear message rather than
// silently zeroing the scatter output.
//
// Lifetime: entries are keyed by (arena pointer, scatterAdd node index).
// Arenas are discarded at training-step boundaries (per JIT Design B); the
// registry naturally garbage-collects when an arena is dropped because the
// closure references the arena and the per-arena map gets removed at Reset
// time via ResetScatterPreprocessors (called from Arena.Reset's tensor-side
// shim, not added in Slice D because Reset is not part of the JIT cycle;
// stale entries from dead arenas are harmless because keys include the arena
// pointer).

// scatterPreprocFn runs the host sort for one OpScatterAdd node.
// Returns false if the source idx leaf has no data attached (caller should
// surface a helpful panic).
type scatterPreprocFn func() bool

var (
	scatterPreprocMu sync.Mutex
	scatterPreproc   = map[*uop.Arena]map[uint32]scatterPreprocFn{}
)

// registerScatterPreproc stores fn against (arena, scatterAddNodeIdx).
// Subsequent calls with the same key overwrite (last writer wins); this
// matches the gradient pass building a new closure every Backward call.
func registerScatterPreproc(a *uop.Arena, scatterIdx uint32, fn scatterPreprocFn) {
	scatterPreprocMu.Lock()
	defer scatterPreprocMu.Unlock()
	m, ok := scatterPreproc[a]
	if !ok {
		m = map[uint32]scatterPreprocFn{}
		scatterPreproc[a] = m
	}
	m[scatterIdx] = fn
}

// RunScatterPreprocessors invokes every registered scatter-add preprocessor
// for arena a in node-index order (deterministic). Called by Realize,
// RealizeWithBinding, and JIT.replay paths before they dispatch the executor.
//
// Ordering: closures are independent in practice (each touches its own
// sortedIdx / perm leaves), but we sort by node index so multi-scatter graphs
// produce reproducible execution and any future state-dependent preprocessor
// can rely on deterministic order.
//
// Panics if any closure reports a missing idx-data input; callers cannot
// recover from that without programmer intervention.
func RunScatterPreprocessors(a *uop.Arena) {
	scatterPreprocMu.Lock()
	m, ok := scatterPreproc[a]
	if !ok || len(m) == 0 {
		scatterPreprocMu.Unlock()
		return
	}
	keys := make([]uint32, 0, len(m))
	fns := make([]scatterPreprocFn, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	for _, k := range keys {
		fns = append(fns, m[k])
	}
	scatterPreprocMu.Unlock()

	for i, fn := range fns {
		if !fn() {
			panic(scatterPreprocMissingErr{nodeIdx: keys[i]})
		}
	}
}

// scatterPreprocMissingErr is panicked when a preprocessor closure cannot
// find data on the idx leaf it captured. The message points at the most
// common cause: forgetting tensor.SetData on the idx input.
type scatterPreprocMissingErr struct{ nodeIdx uint32 }

func (e scatterPreprocMissingErr) Error() string {
	return fmt.Sprintf(
		"tensor: scatter-add preprocessor: source idx leaf has no data attached (SetData not called?); scatter-add UOp idx=%s@%d",
		uop.OpScatterAdd, e.nodeIdx,
	)
}
