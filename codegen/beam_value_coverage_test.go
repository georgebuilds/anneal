package codegen_test

import (
	"testing"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// errExec always returns a run error, so beamRunSingle's error branch fires
// (refOut ok=false → BeamSearch returns baseline-only result).
type errExec struct{}

func (errExec) Run([]schedule.ExecItem, map[uint32][]float32) (map[uint32][]float32, error) {
	return nil, errBenchFail
}
func (errExec) Close() {}

// okBench returns a constant baseline so the search reaches the reference-output
// stage (where errExec fails).
type okBench struct{}

func (okBench) Benchmark(schedule.ExecItem, int, int) (backend.BenchmarkResult, error) {
	return backend.BenchmarkResult{MinMicros: 100.0}, nil
}

func reduceItemV(t *testing.T) schedule.ExecItem {
	t.Helper()
	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{32, 32}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{1}, false)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}
	return items[0]
}

// TestBeamSearchRefOutputFailure: exec errors on the reference run, so
// beamRunSingle returns ok=false and BeamSearch returns a baseline-only result
// with no opts. Exercises beamRunSingle's exec.Run error branch.
func TestBeamSearchRefOutputFailure(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	item := reduceItemV(t)
	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 1, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(errExec{}, okBench{}, item, cfg)

	if len(res.Opts) != 0 {
		t.Errorf("ref-output failure should yield no opts, got %v", res.Opts)
	}
	if res.BaseMicros != 100.0 || res.MinMicros != 100.0 {
		t.Errorf("expected baseline-only timings 100/100, got base=%v min=%v",
			res.BaseMicros, res.MinMicros)
	}
}

// mismatchExec returns the right shape for the FIRST (reference) run but a
// length-mismatched output for every subsequent (candidate) run, so
// beamValueOK rejects every candidate via the len(got) != len(ref) branch.
type mismatchExec struct{ calls int }

func (m *mismatchExec) Run(items []schedule.ExecItem, _ map[uint32][]float32) (map[uint32][]float32, error) {
	m.calls++
	out := make(map[uint32][]float32)
	for _, it := range items {
		if len(it.Bufs) == 0 {
			continue
		}
		size := it.Bufs[0].Size
		if m.calls > 1 {
			size++ // wrong length on candidate runs
		}
		out[it.Bufs[0].UOpIdx] = make([]float32, size)
	}
	return out, nil
}
func (m *mismatchExec) Close() {}

// TestBeamSearchAllCandidatesRejected: every candidate produces a wrong-length
// output, so beamValueOK rejects them all and the search keeps the identity
// (no opts) winner. Exercises beamValueOK's length-mismatch branch and the
// candidate-drop path in BeamSearch.
func TestBeamSearchAllCandidatesRejected(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	item := reduceItemV(t)
	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 2, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(&mismatchExec{}, okBench{}, item, cfg)

	if len(res.Opts) != 0 {
		t.Errorf("all-rejected search should keep identity (no opts), got %v", res.Opts)
	}
	// Identity baseline timing is retained.
	if res.MinMicros != 100.0 {
		t.Errorf("min should equal baseline when all candidates rejected, got %v", res.MinMicros)
	}
}

// valueMismatchExec returns correctly-sized but value-differing outputs on
// candidate runs, exercising beamValueOK's element-difference branch.
type valueMismatchExec struct{ calls int }

func (v *valueMismatchExec) Run(items []schedule.ExecItem, _ map[uint32][]float32) (map[uint32][]float32, error) {
	v.calls++
	out := make(map[uint32][]float32)
	for _, it := range items {
		if len(it.Bufs) == 0 {
			continue
		}
		buf := make([]float32, it.Bufs[0].Size)
		if v.calls > 1 {
			for i := range buf {
				buf[i] = 1 // differ from the all-zero reference
			}
		}
		out[it.Bufs[0].UOpIdx] = buf
	}
	return out, nil
}
func (v *valueMismatchExec) Close() {}

// improvingBench reports a high baseline on its first call and a fixed lower
// time afterward, so the first opt level strictly improves on the baseline and
// the search advances to a non-identity winner (then terminates on the tie at
// the next depth).
type improvingBench struct{ calls int }

func (b *improvingBench) Benchmark(schedule.ExecItem, int, int) (backend.BenchmarkResult, error) {
	b.calls++
	if b.calls == 1 {
		return backend.BenchmarkResult{MinMicros: 100.0}, nil
	}
	return backend.BenchmarkResult{MinMicros: 50.0}, nil
}

func TestBeamSearchFindsImprovingWinner(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	item := reduceItemV(t)
	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 2, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(&stubExec{}, &improvingBench{}, item, cfg)

	if len(res.Opts) == 0 {
		t.Error("improving bench should yield a non-identity winner")
	}
	if res.MinMicros >= res.BaseMicros {
		t.Errorf("winner min (%v) should beat baseline (%v)", res.MinMicros, res.BaseMicros)
	}
}

// TestBeamApplyToItemsSearchModeAppliesWinner runs the apply path in search
// mode with an improving bench so a non-identity winner is found, persisted,
// and applied (WGSL pre-filled). Covers the new-search persist + apply branch.
func TestBeamApplyToItemsSearchModeAppliesWinner(t *testing.T) {
	t.Setenv("ANNEAL_BEAM", "1")
	codegen.BeamCacheReset()
	codegen.BeamDiskCacheReset()
	defer func() {
		codegen.BeamCacheReset()
		codegen.BeamDiskCacheReset()
	}()

	item := reduceItemV(t)
	got := codegen.BeamApplyToItems([]schedule.ExecItem{item}, &stubExec{}, &improvingBench{})
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if got[0].WGSL == "" {
		t.Error("search-mode winner should pre-fill WGSL on the applied item")
	}
}

// TestBeamSearchCacheHitNonEmptyOpts seeds the in-memory cache with a real
// non-identity opt so the cache-hit branch applies it (winItem = ApplyOpts)
// before benchmarking - covering the len(cached) > 0 sub-branch.
func TestBeamSearchCacheHitNonEmptyOpts(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	item := reduceItemV(t)
	acts := codegen.ActionSpace(item.Ast)
	if len(acts) == 0 {
		t.Skip("no actions for kernel")
	}
	sk := codegen.KernelSK(item)
	codegen.BeamCacheStore(sk, []codegen.Opt{acts[0]})

	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 1, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(&stubExec{}, okBench{}, item, cfg)
	if !res.FromCache {
		t.Error("cache hit must set FromCache=true")
	}
	if len(res.Opts) != 1 {
		t.Errorf("cache-hit opts = %v, want the stored single opt", res.Opts)
	}
}

func TestBeamSearchValueMismatchRejected(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	item := reduceItemV(t)
	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 1, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(&valueMismatchExec{}, okBench{}, item, cfg)

	if len(res.Opts) != 0 {
		t.Errorf("value-mismatched candidates must be rejected, got %v", res.Opts)
	}
}
