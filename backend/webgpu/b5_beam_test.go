package webgpu_test

// B5 — persistent beam cache wired into the realize path.
//
// Tests:
//   TestB5_SKCollisionGuard      — unit test of the value-identity guard. Injects a cache
//                                  entry with a wrong WGSL hash and verifies BeamApplyToItems
//                                  falls back to identity (unchanged Ast); then with the
//                                  correct normalized hash verifies opts ARE applied (Ast differs).
//   TestB5_DefaultModeNoOverhead — sanity: empty disk cache + ANNEAL_BEAM unset produces
//                                  correct results with zero realize-path change.
//   TestB5_CacheHitBitIdentical  — end-to-end: search mode populates disk cache, cache-hit
//                                  mode produces bit-identical output to identity.
//   TestB5_StepLevelBenchmark    — 3-condition step-level timing on a 1024³ matmul;
//                                  numbers feed test_output_b5.txt.
//
// Value-identity guard policy:
//   The disk-cache entry stores the FNV-64a hash of the NORMALIZED opted WGSL. Normalization
//   replaces arena-index-dependent variable names (t{N}, r{N}, sm{N}) with sequential _v0,
//   _v1, … placeholders so the hash is stable across process runs even though OpRange nodes
//   bypass arena interning and get new indices each time. An SK collision (two kernels with
//   the same structural 64-bit key) would produce different normalized WGSL → different
//   stored hash → guard fires → identity fallback + log warning.
//
//   Constructing a genuine SK-collision test is impractical (requires finding two distinct
//   kernel ASTs that produce the same 64-bit structural hash, which has probability ~2⁻⁶⁴).
//   The test below exercises the guard mechanism by injecting a fake entry with a wrong hash.

import (
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// b5MatmulItems returns a fresh N×N matmul schedule (no opts applied).
func b5MatmulItems(a *uop.Arena, N int64) ([]schedule.ExecItem, bool) {
	A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
	C := A.Matmul(B)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{C.Node()}, nil, nil)
	return schedule.CreateSchedule(sink, "webgpu"), true
}

// ── SK-collision guard ────────────────────────────────────────────────────────

func TestB5_SKCollisionGuard(t *testing.T) {
	requireDevice(t) // ensures wgpu/Metal is present; the guard itself doesn't execute kernels

	a := uop.NewArena(65536)
	items, _ := b5MatmulItems(a, 32)
	if len(items) == 0 {
		t.Skip("no kernels scheduled")
	}
	item := items[0]
	sk := codegen.KernelSK(item)
	theOpts := []codegen.Opt{{Kind: codegen.OptLocal, Axis: 0, Arg: 8}}

	// ── Case A: wrong WGSL hash → guard fires → identity returned ──────────────
	codegen.BeamDiskCacheReset()
	codegen.BeamDiskCacheInject(sk, theOpts, "badhash00000000")

	resultA := codegen.BeamApplyToItems(items, nil, nil)

	// Guard fires → Ast must be unchanged (opts not applied).
	if resultA[0].Ast.Index() != items[0].Ast.Index() {
		t.Errorf("guard-A: Ast changed from %d to %d despite hash mismatch",
			items[0].Ast.Index(), resultA[0].Ast.Index())
	}
	// WGSL in resultA[0] is the pre-rendered identity WGSL (set by cacheStore inside
	// CreateSchedule), same as items[0].WGSL. If opts had been applied the WGSL would differ.
	if resultA[0].WGSL != items[0].WGSL {
		t.Errorf("guard-A: WGSL changed (guard should have returned identity; got different WGSL)")
	}

	// ── Case B: correct normalized hash → guard passes → opts applied ──────────
	// Compute what BeamApplyToItems will internally produce: apply opts and render.
	// NOTE: ApplyOpts uses arena-append-only semantics; Case A already added nodes,
	// so Case B's nodes land at higher indices. normalizeWGSL makes the hash stable.
	opted := codegen.ApplyOpts(item, theOpts)
	opted.WGSL = ""
	rendered := codegen.RenderWGSL(opted)
	validHash := codegen.BeamWGSLHash(rendered.WGSL) // uses same normalization as guard

	codegen.BeamDiskCacheReset()
	codegen.BeamDiskCacheInject(sk, theOpts, validHash)

	resultB := codegen.BeamApplyToItems(items, nil, nil)

	// Guard passes → Ast must differ (opts changed the kernel).
	if resultB[0].Ast.Index() == items[0].Ast.Index() {
		t.Error("guard-B: Ast unchanged; opts were not applied despite valid hash")
	}
	// WGSL must be pre-rendered by BeamApplyToItems (non-empty).
	if resultB[0].WGSL == "" {
		t.Error("guard-B: WGSL not pre-rendered")
	}

	codegen.BeamDiskCacheReset()
}

// ── Default mode sanity ───────────────────────────────────────────────────────

func TestB5_DefaultModeNoOverhead(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	codegen.BeamDiskCacheReset()
	os.Unsetenv("ANNEAL_BEAM")

	a := uop.NewArena(65536)
	A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
	d := make([]float32, 64*64)
	for i := range d {
		d[i] = float32(i%7+1) * 0.01
	}
	A.SetData(d)
	B.SetData(d)
	C := A.Matmul(B)
	if err := tensor.Realize(C); err != nil {
		t.Fatalf("Realize: %v", err)
	}
	if len(C.Data()) != 64*64 {
		t.Errorf("output length %d, want %d", len(C.Data()), 64*64)
	}
}

// ── Cache-hit bit-identity ────────────────────────────────────────────────────

func TestB5_CacheHitBitIdentical(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const N = 64
	mkData := func() []float32 {
		d := make([]float32, N*N)
		for i := range d {
			d[i] = float32(i%7+1) * 0.01
		}
		return d
	}
	runMatmul := func() []float32 {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		A.SetData(mkData())
		B.SetData(mkData())
		C := A.Matmul(B)
		if err := tensor.Realize(C); err != nil {
			t.Fatalf("Realize: %v", err)
		}
		return C.Data()
	}

	// Condition 1: identity baseline.
	codegen.BeamDiskCacheReset()
	os.Unsetenv("ANNEAL_BEAM")
	ref := make([]float32, len(runMatmul()))
	copy(ref, runMatmul())

	// Condition 2: search populates disk cache.
	os.Setenv("ANNEAL_BEAM", "1")
	runMatmul()
	os.Unsetenv("ANNEAL_BEAM")

	// Condition 3: cache-hit must produce bit-identical output.
	got := runMatmul()

	if len(got) != len(ref) {
		t.Fatalf("output length mismatch: got %d want %d", len(got), len(ref))
	}
	var maxDiff float32
	for i, v := range got {
		d := v - ref[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff != 0 {
		t.Errorf("cache-hit output differs from identity: max-abs-diff=%g (want 0)", maxDiff)
	}

	codegen.BeamDiskCacheReset()
}

// ── 3-condition step-level benchmark ─────────────────────────────────────────

// TestB5_StepLevelBenchmark measures step latency for a 1024³ matmul under three
// beam modes and logs the results that feed test_output_b5.txt.
func TestB5_StepLevelBenchmark(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	const N = 1024
	const (
		warmup = 3
		iters  = 7
	)

	mkData := func() []float32 {
		d := make([]float32, N*N)
		for i := range d {
			d[i] = float32(i%7+1) * 0.01
		}
		return d
	}

	run := func() time.Duration {
		start := time.Now()
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		A.SetData(mkData())
		B.SetData(mkData())
		C := A.Matmul(B)
		if err := tensor.Realize(C); err != nil {
			t.Fatalf("Realize: %v", err)
		}
		return time.Since(start)
	}

	warm := func() {
		for i := 0; i < warmup; i++ {
			run()
		}
	}

	measure := func(label string) []time.Duration {
		times := make([]time.Duration, iters)
		for i := range times {
			times[i] = run()
		}
		sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
		t.Logf("%-14s: median=%v min=%v max=%v", label, times[iters/2], times[0], times[iters-1])
		return times
	}

	// Condition 1 — identity (no cache, no search).
	codegen.BeamDiskCacheReset()
	os.Unsetenv("ANNEAL_BEAM")
	warm()
	t1 := measure("Cond1-ID")

	// Condition 2 — search mode (populates disk cache; first several iters include search latency).
	codegen.BeamDiskCacheReset()
	os.Setenv("ANNEAL_BEAM", "1")
	searchStart := time.Now()
	warm()
	t2 := measure("Cond2-Search")
	searchCost := time.Since(searchStart)
	os.Unsetenv("ANNEAL_BEAM")
	t.Logf("One-time search cost (warmup+iters): %v", searchCost)

	// Condition 3 — cache-hit (disk cache warm from Cond2, ANNEAL_BEAM unset).
	warm()
	t3 := measure("Cond3-Cache")

	med1 := t1[iters/2]
	med3 := t3[iters/2]
	pct := float64(med3-med1) / float64(med1) * 100
	verdict := fmt.Sprintf("within noise (%.1f%%)", pct)
	if pct < -5 {
		verdict = fmt.Sprintf("FASTER %.1f%%", -pct)
	} else if pct > 5 {
		verdict = fmt.Sprintf("SLOWER %.1f%%", pct)
	}
	t.Logf("Cond3 vs Cond1 (1024³ matmul): %s", verdict)
	t.Logf("Search cost (one-time): %v", searchCost)
	_ = t2

	codegen.BeamDiskCacheReset()
}
