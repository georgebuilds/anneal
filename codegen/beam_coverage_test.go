package codegen_test

import (
	"os"
	"sync/atomic"
	"testing"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── stubs for backend.Executor / backend.Benchmarker ─────────────────────────

// stubExec returns deterministic all-zero outputs of the correct size for each
// item, so beamValueOK sees bit-identical reference and candidate outputs.
type stubExec struct{ runCount atomic.Int64 }

func (s *stubExec) Run(items []schedule.ExecItem, _ map[uint32][]float32) (map[uint32][]float32, error) {
	s.runCount.Add(1)
	out := make(map[uint32][]float32)
	for _, it := range items {
		if len(it.Bufs) == 0 {
			continue
		}
		out[it.Bufs[0].UOpIdx] = make([]float32, it.Bufs[0].Size)
	}
	return out, nil
}
func (s *stubExec) Close() {}

// stubBench returns a small baseline and ever-decreasing candidate timings so
// the beam search prefers candidates but eventually terminates.
type stubBench struct{ n atomic.Int64 }

func (b *stubBench) Benchmark(_ schedule.ExecItem, _, _ int) (backend.BenchmarkResult, error) {
	i := b.n.Add(1)
	// Constant baseline = 100us; candidates always tied, so search will terminate.
	return backend.BenchmarkResult{MinMicros: 100.0 + float64(i%3)*0.01}, nil
}

// ── BeamCache (in-memory) ────────────────────────────────────────────────────

func TestBeamCacheStoreAndLookup(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	const sk uint64 = 0xabc123
	if _, ok := codegen.BeamCacheLookup(sk); ok {
		t.Error("lookup on empty cache: ok=true")
	}

	want := []codegen.Opt{{Kind: codegen.OptLocal, Axis: 0, Arg: 8}}
	codegen.BeamCacheStore(sk, want)
	got, ok := codegen.BeamCacheLookup(sk)
	if !ok {
		t.Fatal("lookup after store: ok=false")
	}
	if len(got) != 1 || got[0] != want[0] {
		t.Errorf("BeamCacheLookup = %v, want %v", got, want)
	}
}

func TestBeamCacheStoreNilMeansIdentity(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	const sk uint64 = 0xdef456
	codegen.BeamCacheStore(sk, nil)
	got, ok := codegen.BeamCacheLookup(sk)
	if !ok {
		t.Fatal("identity-winner lookup: ok=false")
	}
	if len(got) != 0 {
		t.Errorf("identity-winner opts non-empty: %v", got)
	}
}

func TestBeamCacheReset(t *testing.T) {
	codegen.BeamCacheStore(1, []codegen.Opt{{Kind: codegen.OptLocal}})
	codegen.BeamCacheReset()
	if _, ok := codegen.BeamCacheLookup(1); ok {
		t.Error("BeamCacheReset did not clear cache")
	}
}

// ── BeamDiskCache (synthetic injection) ──────────────────────────────────────

func TestBeamDiskCacheResetAndInject(t *testing.T) {
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	const sk uint64 = 0x123
	codegen.BeamDiskCacheInject(sk, []codegen.Opt{{Kind: codegen.OptTile, Axis: 1, Arg: 16}}, "deadbeefdeadbeef")
	// No direct read API - but BeamApplyToItems is the user. Verify via the apply path below.
}

// ── BeamWGSLHash ─────────────────────────────────────────────────────────────

func TestBeamWGSLHashStable(t *testing.T) {
	h1 := codegen.BeamWGSLHash("let t1: f32 = data1[r5];")
	h2 := codegen.BeamWGSLHash("let t1: f32 = data1[r5];")
	if h1 != h2 {
		t.Errorf("BeamWGSLHash not deterministic: %s vs %s", h1, h2)
	}
	if len(h1) != 16 {
		t.Errorf("BeamWGSLHash length = %d, want 16 hex chars", len(h1))
	}
}

func TestBeamWGSLHashIndexIndependent(t *testing.T) {
	h1 := codegen.BeamWGSLHash("let t1: f32 = data1[r5];")
	h2 := codegen.BeamWGSLHash("let t99: f32 = data1[r33];")
	if h1 != h2 {
		t.Errorf("BeamWGSLHash should normalize identifiers: %s vs %s", h1, h2)
	}
}

func TestBeamWGSLHashDistinguishesStructurally(t *testing.T) {
	h1 := codegen.BeamWGSLHash("let t1: f32 = data1[r1];")
	h2 := codegen.BeamWGSLHash("let t1: f32 = data2[r1];")
	if h1 == h2 {
		t.Error("BeamWGSLHash should distinguish data1 vs data2")
	}
}

// ── DefaultBeamConfig ────────────────────────────────────────────────────────

func TestDefaultBeamConfigDefaults(t *testing.T) {
	_ = os.Unsetenv("BEAM_WIDTH")
	_ = os.Unsetenv("MAX_DEPTH")
	cfg := codegen.DefaultBeamConfig()
	if cfg.Width != 4 || cfg.MaxDepth != 4 {
		t.Errorf("defaults = (W=%d D=%d), want (4,4)", cfg.Width, cfg.MaxDepth)
	}
	if cfg.Warmup <= 0 || cfg.Iters <= 0 {
		t.Errorf("Warmup/Iters must be > 0: got %d/%d", cfg.Warmup, cfg.Iters)
	}
}

func TestDefaultBeamConfigEnvOverride(t *testing.T) {
	t.Setenv("BEAM_WIDTH", "7")
	t.Setenv("MAX_DEPTH", "9")
	cfg := codegen.DefaultBeamConfig()
	if cfg.Width != 7 {
		t.Errorf("Width = %d, want 7 (BEAM_WIDTH=7)", cfg.Width)
	}
	if cfg.MaxDepth != 9 {
		t.Errorf("MaxDepth = %d, want 9 (MAX_DEPTH=9)", cfg.MaxDepth)
	}
}

func TestDefaultBeamConfigEnvBadIgnored(t *testing.T) {
	t.Setenv("BEAM_WIDTH", "notanumber")
	t.Setenv("MAX_DEPTH", "-1") // invalid (< 1)
	cfg := codegen.DefaultBeamConfig()
	if cfg.Width != 4 || cfg.MaxDepth != 4 {
		t.Errorf("malformed env should fall back to defaults; got W=%d D=%d", cfg.Width, cfg.MaxDepth)
	}
}

// ── KernelSK ─────────────────────────────────────────────────────────────────

func TestKernelSKInvalidReturnsZero(t *testing.T) {
	var item schedule.ExecItem
	if got := codegen.KernelSK(item); got != 0 {
		t.Errorf("KernelSK(zero) = %x, want 0", got)
	}
}

func TestKernelSKStableAcrossBuilds(t *testing.T) {
	build := func() uint64 {
		a := uop.NewArena(64)
		x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "webgpu")
		y := x.Exp2()
		sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
		items := schedule.CreateSchedule(sink, "webgpu")
		if len(items) == 0 {
			t.Fatal("no items")
		}
		return codegen.KernelSK(items[0])
	}
	sk1 := build()
	sk2 := build()
	if sk1 != sk2 {
		t.Errorf("KernelSK not stable across builds: %x vs %x", sk1, sk2)
	}
	if sk1 == 0 {
		t.Error("KernelSK returned 0 for valid kernel")
	}
}

// ── ActionSpace ──────────────────────────────────────────────────────────────

func TestActionSpaceNonEmptyForReduce(t *testing.T) {
	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{32, 32}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{1}, false)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}
	acts := codegen.ActionSpace(items[0].Ast)
	if len(acts) == 0 {
		t.Error("ActionSpace returned no actions for reduce kernel")
	}
	// All actions must be non-Identity and have axis < beamMaxAxis (4).
	for _, opt := range acts {
		if opt.Kind == codegen.OptIdentity {
			t.Error("ActionSpace returned OptIdentity")
		}
		if opt.Axis < 0 || opt.Axis >= 4 {
			t.Errorf("axis out of range: %d", opt.Axis)
		}
	}
}

// ── BeamApplyToItems (default mode + identity hit) ───────────────────────────

func TestBeamApplyToItemsDefaultModeIdentity(t *testing.T) {
	// Ensure search mode is off and disk cache is clean.
	_ = os.Unsetenv("ANNEAL_BEAM")
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}

	got := codegen.BeamApplyToItems(items, nil, nil)
	if len(got) != len(items) {
		t.Fatalf("BeamApplyToItems changed item count: got %d, want %d", len(got), len(items))
	}
	// Default mode + cache miss = identity, so each entry must be unchanged.
	for i := range items {
		if got[i].Ast.Index() != items[i].Ast.Index() {
			t.Errorf("item %d Ast index changed unexpectedly", i)
		}
	}
}

func TestBeamApplyToItemsIdentityHit(t *testing.T) {
	_ = os.Unsetenv("ANNEAL_BEAM")
	codegen.BeamDiskCacheReset()
	defer codegen.BeamDiskCacheReset()

	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}

	// Inject an identity hit (Opts=nil) for this kernel's SK.
	sk := codegen.KernelSK(items[0])
	codegen.BeamDiskCacheInject(sk, nil, "")

	got := codegen.BeamApplyToItems(items, nil, nil)
	if got[0].Ast.Index() != items[0].Ast.Index() {
		t.Error("identity hit should leave Ast unchanged")
	}
}

func TestBeamApplyToItemsInvalidAstSkipped(t *testing.T) {
	// Items with zero-value Ast should be passed through unchanged.
	var zero schedule.ExecItem
	got := codegen.BeamApplyToItems([]schedule.ExecItem{zero}, nil, nil)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if got[0].Ast.Valid() {
		t.Error("invalid ast became valid")
	}
}

// ── BeamSearch via stubs (exercises full inner loop) ─────────────────────────

func TestBeamSearchCacheHitPath(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{32, 32}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{1}, false)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}
	item := items[0]
	sk := codegen.KernelSK(item)

	// Pre-seed cache with an empty opt list so cache-hit path runs.
	codegen.BeamCacheStore(sk, nil)

	exec := &stubExec{}
	bench := &stubBench{}
	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 1, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(exec, bench, item, cfg)
	if !res.FromCache {
		t.Error("cache hit must set FromCache=true")
	}
}

func TestBeamSearchFullSearch(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{32, 32}, uop.Dtypes.Float32, "webgpu")
	y := x.Sum([]int{1}, false)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}
	item := items[0]

	exec := &stubExec{}
	bench := &stubBench{}
	cfg := codegen.BeamConfig{Width: 2, MaxDepth: 2, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(exec, bench, item, cfg)
	if res.FromCache {
		t.Error("fresh search must not be from cache")
	}
	if res.BaseMicros == 0 {
		t.Error("BaseMicros should be set after baseline benchmark")
	}
	// After the search, the cache must contain an entry for this SK.
	sk := codegen.KernelSK(item)
	if _, ok := codegen.BeamCacheLookup(sk); !ok {
		t.Error("BeamSearch did not store winning opts in cache")
	}
}

// failingBench always returns error so the baseline-fail branch fires.
type failingBench struct{}

func (failingBench) Benchmark(_ schedule.ExecItem, _, _ int) (backend.BenchmarkResult, error) {
	return backend.BenchmarkResult{}, errBenchFail
}

var errBenchFail = errBench("bench failed")

type errBench string

func (e errBench) Error() string { return string(e) }

func TestBeamSearchBaselineFailure(t *testing.T) {
	codegen.BeamCacheReset()
	defer codegen.BeamCacheReset()

	a := uop.NewArena(128)
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "webgpu")
	y := x.Exp2()
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{y.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")
	if len(items) == 0 {
		t.Fatal("no items")
	}

	exec := &stubExec{}
	cfg := codegen.BeamConfig{Width: 1, MaxDepth: 1, Warmup: 1, Iters: 1}
	res := codegen.BeamSearch(exec, failingBench{}, items[0], cfg)
	// Baseline failure path: result should be zero-valued aside from WallNs.
	if res.BaseMicros != 0 || res.MinMicros != 0 {
		t.Errorf("on baseline failure expected zero timings; got base=%v min=%v",
			res.BaseMicros, res.MinMicros)
	}
}
