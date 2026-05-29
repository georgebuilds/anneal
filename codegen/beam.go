package codegen

import (
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// Action-space parameters. Kept compact to bound search time.
var (
	beamLocalArgs     = []int{8, 16, 32}
	beamTileArgs      = []int{8, 16, 32}
	beamUpcastArgs    = []int{2, 4}
	beamVectorizeArgs = []int{4}
	beamMaxAxis       = 4
)

// ActionSpace returns every Opt that non-trivially transforms sink.
// "Non-trivial" means the returned UOp has a different arena index than sink
// (i.e. ApplyOpt created new nodes rather than returning sink unchanged).
// Call on the live (possibly already-optimised) sink at each search depth.
func ActionSpace(sink uop.UOp) []Opt {
	var actions []Opt
	tryKind := func(kind OptKind, args []int) {
		for axis := 0; axis < beamMaxAxis; axis++ {
			for _, arg := range args {
				opt := Opt{Kind: kind, Axis: axis, Arg: arg}
				res := ApplyOpt(sink, opt)
				if res.Index() != sink.Index() {
					actions = append(actions, opt)
				}
			}
		}
	}
	tryKind(OptLocal, beamLocalArgs)
	tryKind(OptTile, beamTileArgs)
	tryKind(OptUpcast, beamUpcastArgs)
	tryKind(OptVectorize, beamVectorizeArgs)
	return actions
}

// KernelSK returns the structural key of the SINK-rooted kernel AST in item.
// The key is stable under arena append-only growth: StructuralKeys mixes
// children's SK values (not arena positions), so the original node's SK is
// invariant once built.
func KernelSK(item schedule.ExecItem) uint64 {
	if !item.Ast.Valid() {
		return 0
	}
	a := item.Ast.Arena()
	keys := uop.StructuralKeys(a)
	idx := item.Ast.Index()
	if int(idx) >= len(keys) {
		return 0
	}
	return keys[idx]
}

// beamEntry stores one completed beam-search result.
type beamEntry struct {
	opts []Opt // nil means identity was the winner
	set  bool  // true once a completed search has been persisted
}

var (
	beamMu       sync.Mutex
	beamCacheMap = map[uint64]beamEntry{}
)

// BeamCacheLookup returns the cached opt sequence for kernel SK.
// Returns (opts, true) on hit; opts may be nil (identity won).
func BeamCacheLookup(sk uint64) ([]Opt, bool) {
	beamMu.Lock()
	defer beamMu.Unlock()
	e, ok := beamCacheMap[sk]
	if !ok || !e.set {
		return nil, false
	}
	return e.opts, true
}

// BeamCacheStore records a winning opt sequence for kernel SK.
// Pass opts=nil to record that identity was the winner.
func BeamCacheStore(sk uint64, opts []Opt) {
	beamMu.Lock()
	defer beamMu.Unlock()
	beamCacheMap[sk] = beamEntry{opts: opts, set: true}
}

// BeamCacheReset clears all cache entries. Used by tests to isolate runs.
func BeamCacheReset() {
	beamMu.Lock()
	defer beamMu.Unlock()
	beamCacheMap = map[uint64]beamEntry{}
}

// BeamConfig parameterises the beam search.
type BeamConfig struct {
	Width    int // beam width: candidates kept per depth round
	MaxDepth int // maximum opt-sequence length to explore
	Warmup   int // per-candidate benchmark warmup iterations
	Iters    int // per-candidate benchmark measurement iterations
}

// DefaultBeamConfig returns sensible defaults.
// BEAM_WIDTH and MAX_DEPTH can be overridden via environment variables.
func DefaultBeamConfig() BeamConfig {
	w, d := 4, 4
	if s := os.Getenv("BEAM_WIDTH"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			w = v
		}
	}
	if s := os.Getenv("MAX_DEPTH"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			d = v
		}
	}
	return BeamConfig{Width: w, MaxDepth: d, Warmup: 2, Iters: 5}
}

// BeamResult holds the output of a beam search for one kernel.
type BeamResult struct {
	Opts       []Opt   // winning opt sequence; nil means identity
	MinMicros  float64 // winner's min-of-N timing
	BaseMicros float64 // identity baseline min-of-N timing
	Searched   int     // candidates successfully benchmarked
	WallNs     int64   // search wall-clock nanoseconds
	FromCache  bool    // true if result came from the beam cache
}

// beamCandidate is one node in the search frontier.
type beamCandidate struct {
	opts      []Opt
	sink      uop.UOp // AST root after applying opts; shares arena with baseItem.Ast
	minMicros float64
}

// BeamSearch runs a beam search over opt sequences for a single kernel item.
// exec runs kernels for value-identity checks; bench times each candidate.
//
// Correctness invariant: every returned opt sequence produces output
// bit-identical to the identity baseline (max-abs-diff == 0) on a small
// fixed test input. Any candidate that fails this check is silently dropped.
//
// Termination bound: the search stops when no depth-D+1 candidate improves
// over the current best, or when MaxDepth is reached. Total candidates
// evaluated ≤ MaxDepth × BeamWidth × |ActionSpace| (bounded and finite).
func BeamSearch(
	exec backend.Executor,
	bench backend.Benchmarker,
	item schedule.ExecItem,
	cfg BeamConfig,
) BeamResult {
	start := time.Now()
	sk := KernelSK(item)

	// ── Cache hit ─────────────────────────────────────────────────────────────
	if cached, ok := BeamCacheLookup(sk); ok {
		winItem := item
		winItem.WGSL = ""
		if len(cached) > 0 {
			winItem.Ast = ApplyOpts(item, cached).Ast
		}
		winRes, winErr := bench.Benchmark(winItem, cfg.Warmup, cfg.Iters)
		baseRes, _ := bench.Benchmark(item, cfg.Warmup, cfg.Iters)
		minMicros := baseRes.MinMicros
		if winErr == nil {
			minMicros = winRes.MinMicros
		}
		return BeamResult{
			Opts:       cached,
			MinMicros:  minMicros,
			BaseMicros: baseRes.MinMicros,
			FromCache:  true,
			WallNs:     time.Since(start).Nanoseconds(),
		}
	}

	// ── Baseline timing ───────────────────────────────────────────────────────
	baseRes, err := bench.Benchmark(item, cfg.Warmup, cfg.Iters)
	if err != nil {
		BeamCacheStore(sk, nil)
		return BeamResult{WallNs: time.Since(start).Nanoseconds()}
	}
	baseMicros := baseRes.MinMicros

	// ── Reference outputs for value-identity guard ────────────────────────────
	testIn := beamMakeTestInputs(item)
	refOut, ok := beamRunSingle(exec, item, testIn)
	if !ok {
		BeamCacheStore(sk, nil)
		return BeamResult{
			BaseMicros: baseMicros,
			MinMicros:  baseMicros,
			WallNs:     time.Since(start).Nanoseconds(),
		}
	}

	// ── Beam search ───────────────────────────────────────────────────────────
	best := beamCandidate{opts: nil, sink: item.Ast, minMicros: baseMicros}
	current := []beamCandidate{best}
	totalSearched := 0

	for depth := 0; depth < cfg.MaxDepth; depth++ {
		var next []beamCandidate

		for _, cand := range current {
			for _, action := range ActionSpace(cand.sink) {
				newSink := ApplyOpt(cand.sink, action)

				candItem := item
				candItem.Ast = newSink
				candItem.WGSL = ""

				// Value-identity guard: silently drop any semantically incorrect candidate.
				if !beamValueOK(exec, candItem, testIn, refOut) {
					continue
				}

				res, err := bench.Benchmark(candItem, cfg.Warmup, cfg.Iters)
				if err != nil {
					continue
				}
				totalSearched++

				newOpts := make([]Opt, len(cand.opts)+1)
				copy(newOpts, cand.opts)
				newOpts[len(cand.opts)] = action

				next = append(next, beamCandidate{
					opts:      newOpts,
					sink:      newSink,
					minMicros: res.MinMicros,
				})
			}
		}

		// Sort ascending by MinMicros. Tie-break deterministically:
		// prefer shorter sequence, then lexicographic on (Kind, Axis, Arg).
		sort.SliceStable(next, func(i, j int) bool {
			di, dj := next[i].minMicros, next[j].minMicros
			if math.Abs(di-dj) > 0.1 {
				return di < dj
			}
			oi, oj := next[i].opts, next[j].opts
			if len(oi) != len(oj) {
				return len(oi) < len(oj)
			}
			for k := 0; k < len(oi); k++ {
				if oi[k].Kind != oj[k].Kind {
					return oi[k].Kind < oj[k].Kind
				}
				if oi[k].Axis != oj[k].Axis {
					return oi[k].Axis < oj[k].Axis
				}
				if oi[k].Arg != oj[k].Arg {
					return oi[k].Arg < oj[k].Arg
				}
			}
			return false
		})
		if len(next) > cfg.Width {
			next = next[:cfg.Width]
		}

		// Stop when no candidate improves on the current best.
		if len(next) == 0 || next[0].minMicros >= best.minMicros {
			break
		}
		best = next[0]
		current = next
	}

	BeamCacheStore(sk, best.opts)

	return BeamResult{
		Opts:       best.opts,
		MinMicros:  best.minMicros,
		BaseMicros: baseMicros,
		Searched:   totalSearched,
		WallNs:     time.Since(start).Nanoseconds(),
	}
}

// beamMakeTestInputs creates deterministic float32 test data for each input buffer.
// The pattern varies by buffer index so that rank-1 and rank-2 inputs are distinct.
func beamMakeTestInputs(item schedule.ExecItem) map[uint32][]float32 {
	m := make(map[uint32][]float32, len(item.Bufs)-1)
	for i, buf := range item.Bufs[1:] {
		data := make([]float32, buf.Size)
		for j := range data {
			data[j] = float32((j+i*17)%13+1) * 0.01
		}
		m[buf.UOpIdx] = data
	}
	return m
}

// beamRunSingle executes item as a single-item schedule and returns the flat output.
func beamRunSingle(exec backend.Executor, item schedule.ExecItem, inputs map[uint32][]float32) ([]float32, bool) {
	item.WGSL = ""
	if len(item.Bufs) == 0 {
		return nil, false
	}
	outs, err := exec.Run([]schedule.ExecItem{item}, inputs)
	if err != nil {
		return nil, false
	}
	out, ok := outs[item.Bufs[0].UOpIdx]
	return out, ok
}

// beamValueOK checks that cand produces output bit-identical to ref.
// Returns false on run error, shape mismatch, or any non-zero element difference.
func beamValueOK(exec backend.Executor, cand schedule.ExecItem, inputs map[uint32][]float32, ref []float32) bool {
	got, ok := beamRunSingle(exec, cand, inputs)
	if !ok || len(got) != len(ref) {
		return false
	}
	for i, v := range got {
		if v != ref[i] {
			return false
		}
	}
	return true
}
