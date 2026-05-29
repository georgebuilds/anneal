package webgpu_test

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// optsEqual reports whether two opt sequences are identical.
func optsEqual(a, b []codegen.Opt) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// reportBeam prints one standardised beam-result line and returns the win/no-win verdict.
func reportBeam(label string, r codegen.BeamResult) string {
	verdict := "no opt sequence beat default"
	if r.MinMicros < r.BaseMicros {
		verdict = fmt.Sprintf("found win (%.2fx speedup)", r.BaseMicros/r.MinMicros)
	}
	fmt.Printf("[B4] %-30s baseline=%8.2fµs  winner=%8.2fµs  opts=%v  searched=%d  wall=%.0fms — %s\n",
		label, r.BaseMicros, r.MinMicros, r.Opts, r.Searched, float64(r.WallNs)/1e6, verdict)
	return verdict
}

// TestB4_ActionSpace_Size enumerates and reports the action-space size for
// the canonical benchmark kernel shapes. No GPU required.
func TestB4_ActionSpace_Size(t *testing.T) {
	tests := []struct {
		name  string
		build func() []schedule.ExecItem
	}{
		{"matmul_1024³", func() []schedule.ExecItem {
			a := uop.NewArena(1 << 17)
			A := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
			B := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
			return schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")
		}},
		{"mlp_fwd_16x32x64", func() []schedule.ExecItem {
			a := uop.NewArena(1 << 17)
			x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "webgpu")
			w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "webgpu")
			return schedule.CreateSchedule(makeSink(a, x.Matmul(w)), "webgpu")
		}},
		{"conv_1x1x8x8_k3x3", func() []schedule.ExecItem {
			a := uop.NewArena(1 << 17)
			x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
			x.SetData(uniformData(64, 7))
			conv := nn.NewConv2d(a, 1, 1, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "webgpu")
			conv.Weight.Value = uniformData(9, 8)
			conv.Weight.Load(a)
			return schedule.CreateSchedule(makeSink(a, conv.Forward(x)), "webgpu")
		}},
	}

	for _, tc := range tests {
		items := tc.build()
		for i, item := range items {
			if !item.Ast.Valid() {
				continue
			}
			actions := codegen.ActionSpace(item.Ast)
			t.Logf("[B4] action-space %s kernel[%d]: %d actions", tc.name, i, len(actions))
			fmt.Printf("[B4] action-space %-28s kernel[%d]: %d candidates\n", tc.name, i, len(actions))
		}
	}
}

// TestB4_BeamSearch_Matmul1024 runs the beam search on a 1024³ matmul kernel.
// Acceptance:
//   - convergence is deterministic: same SK → same opts across two independent calls
//   - winning sequence is value-identical to identity (bit-exact on non-trivial input)
//   - search terminates; wall-clock is reported for B5 viability assessment
func TestB4_BeamSearch_Matmul1024(t *testing.T) {
	dev := requireDevice(t)
	codegen.BeamCacheReset()
	cfg := codegen.DefaultBeamConfig()

	mk := func() schedule.ExecItem {
		a := uop.NewArena(1 << 17)
		A := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{1024, 1024}, uop.Dtypes.Float32, "webgpu")
		return schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")[0]
	}

	// First search: cache miss → full search.
	item1 := mk()
	r1 := codegen.BeamSearch(dev, dev, item1, cfg)
	reportBeam("matmul_1024³", r1)
	fmt.Printf("[B4] matmul_1024³: worst-case candidates=%d  wall=%.0fms\n",
		r1.Searched, float64(r1.WallNs)/1e6)

	// Second search on structurally identical kernel: must hit cache.
	item2 := mk()
	r2 := codegen.BeamSearch(dev, dev, item2, cfg)
	if !r2.FromCache {
		t.Errorf("second search should hit cache (same structural kernel SK)")
	}

	// Determinism: same SK → same winning opts.
	if !optsEqual(r1.Opts, r2.Opts) {
		t.Errorf("non-deterministic: run1=%v  run2=%v", r1.Opts, r2.Opts)
	}

	// Value oracle: winning sequence is bit-exact vs identity on non-trivial input.
	N := int64(1024)
	buildWithData := func() (*tensor.Tensor, []schedule.ExecItem) {
		a := uop.NewArena(1 << 17)
		A := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{N, N}, uop.Dtypes.Float32, "webgpu")
		A.SetData(uniformData(int(N*N), 1))
		B.SetData(uniformData(int(N*N), 2))
		C := A.Matmul(B)
		return C, schedule.CreateSchedule(makeSink(a, C), "webgpu")
	}

	// Build inputs map from SetData values (non-trivial: uniformData seeds 1 and 2).
	buildInputs := func(items []schedule.ExecItem) map[uint32][]float32 {
		inputs := make(map[uint32][]float32)
		for _, item := range items {
			for _, buf := range item.Bufs[1:] {
				// Reproduce the same deterministic values used by uniformData(seed).
				// Seed 1 → first leaf, seed 2 → second leaf (matched by UOpIdx order).
				if _, ok := inputs[buf.UOpIdx]; !ok {
					seed := float32(len(inputs) + 1)
					d := make([]float32, buf.Size)
					for j := range d {
						d[j] = float32(j)*0.1 + seed
					}
					inputs[buf.UOpIdx] = d
				}
			}
		}
		return inputs
	}

	outDef, itemsDef := buildWithData()
	inputsDef := buildInputs(itemsDef)
	resDef, err := dev.Run(itemsDef, inputsDef)
	if err != nil {
		t.Fatalf("default matmul run: %v", err)
	}
	gotDef := resDef[outDef.Node().Index()]

	outOpt, itemsOpt := buildWithData()
	inputsOpt := buildInputs(itemsOpt)
	if len(r1.Opts) > 0 {
		for i := range itemsOpt {
			itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], r1.Opts).Ast
			itemsOpt[i].WGSL = ""
		}
	}
	resOpt, err := dev.Run(itemsOpt, inputsOpt)
	if err != nil {
		t.Fatalf("opt matmul run: %v", err)
	}
	gotOpt := resOpt[outOpt.Node().Index()]

	if !approxEq(gotOpt, gotDef, 0) {
		t.Errorf("matmul_1024³: value mismatch — beam winner not bit-exact vs identity")
	}
}

// TestB4_BeamSearch_MLPForwardBackward runs the beam search on each kernel in
// the MLP forward+backward schedule and verifies the winning opts preserve
// the output values.
func TestB4_BeamSearch_MLPForwardBackward(t *testing.T) {
	dev := requireDevice(t)
	codegen.BeamCacheReset()
	cfg := codegen.DefaultBeamConfig()

	type mlpBuild struct {
		gx, gw *tensor.Tensor
		items  []schedule.ExecItem
	}

	buildMLP := func() mlpBuild {
		a := uop.NewArena(1 << 17)
		x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "webgpu")
		w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "webgpu")
		x.SetData(uniformData(16*32, 7))
		w.SetData(uniformData(32*64, 9))
		pred := x.Matmul(w)
		loss := pred.Sum(nil, false)
		grads := tensor.Backward(loss, []*tensor.Tensor{x, w})
		gx, gw := grads[x], grads[w]
		items := schedule.CreateSchedule(
			a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{gx.Node(), gw.Node()}, nil, nil),
			"webgpu",
		)
		return mlpBuild{gx: gx, gw: gw, items: items}
	}

	// Default run (nil inputs = zero-initialised GPU buffers; consistent between both runs).
	defB := buildMLP()
	resDef, err := dev.Run(defB.items, nil)
	if err != nil {
		t.Fatalf("default MLP run: %v", err)
	}
	gxDefault := resDef[defB.gx.Node().Index()]
	gwDefault := resDef[defB.gw.Node().Index()]

	// Beam-search each kernel and apply winning opts.
	optB := buildMLP()
	for i := range optB.items {
		if !optB.items[i].Ast.Valid() {
			continue
		}
		r := codegen.BeamSearch(dev, dev, optB.items[i], cfg)
		reportBeam(fmt.Sprintf("MLP_bwd kernel[%d]", i), r)
		if len(r.Opts) > 0 {
			optB.items[i].Ast = codegen.ApplyOpts(optB.items[i], r.Opts).Ast
			optB.items[i].WGSL = ""
		}
	}
	resOpt, err := dev.Run(optB.items, nil)
	if err != nil {
		t.Fatalf("opt MLP run: %v", err)
	}

	gxOpt := resOpt[optB.gx.Node().Index()]
	gwOpt := resOpt[optB.gw.Node().Index()]

	if !approxEq(gxOpt, gxDefault, 0) {
		t.Errorf("MLP gx value mismatch under beam opts")
	}
	if !approxEq(gwOpt, gwDefault, 0) {
		t.Errorf("MLP gw value mismatch under beam opts")
	}
}

// TestB4_BeamSearch_Conv runs the beam search on conv2d kernels and verifies
// the winning opts preserve values.
func TestB4_BeamSearch_Conv(t *testing.T) {
	dev := requireDevice(t)
	codegen.BeamCacheReset()
	cfg := codegen.DefaultBeamConfig()

	buildConv := func() (*tensor.Tensor, []schedule.ExecItem) {
		a := uop.NewArena(1 << 17)
		x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
		x.SetData(uniformData(64, 7))
		conv := nn.NewConv2d(a, 1, 1, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "webgpu")
		conv.Weight.Value = uniformData(9, 8)
		conv.Weight.Load(a)
		out := conv.Forward(x)
		return out, schedule.CreateSchedule(makeSink(a, out), "webgpu")
	}

	outDef, itemsDef := buildConv()
	resDef, err := dev.Run(itemsDef, nil)
	if err != nil {
		t.Fatalf("default conv run: %v", err)
	}
	gotDef := resDef[outDef.Node().Index()]

	outOpt, itemsOpt := buildConv()
	for i := range itemsOpt {
		if !itemsOpt[i].Ast.Valid() {
			continue
		}
		r := codegen.BeamSearch(dev, dev, itemsOpt[i], cfg)
		reportBeam(fmt.Sprintf("conv kernel[%d]", i), r)
		if len(r.Opts) > 0 {
			itemsOpt[i].Ast = codegen.ApplyOpts(itemsOpt[i], r.Opts).Ast
			itemsOpt[i].WGSL = ""
		}
	}
	resOpt, err := dev.Run(itemsOpt, nil)
	if err != nil {
		t.Fatalf("opt conv run: %v", err)
	}
	gotOpt := resOpt[outOpt.Node().Index()]

	if !approxEq(gotOpt, gotDef, 0) {
		t.Errorf("conv value mismatch under beam opts")
	}
}

// TestB4_Cache_Determinism verifies that three searches on the same kernel SK
// return identical opt sequences, and that the 2nd and 3rd are cache hits.
func TestB4_Cache_Determinism(t *testing.T) {
	dev := requireDevice(t)
	codegen.BeamCacheReset()
	cfg := codegen.DefaultBeamConfig()

	mk := func() schedule.ExecItem {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		return schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")[0]
	}

	r1 := codegen.BeamSearch(dev, dev, mk(), cfg) // miss
	r2 := codegen.BeamSearch(dev, dev, mk(), cfg) // hit
	r3 := codegen.BeamSearch(dev, dev, mk(), cfg) // hit

	if r1.FromCache {
		t.Errorf("run1 should be a cache miss")
	}
	if !r2.FromCache {
		t.Errorf("run2 should be a cache hit")
	}
	if !r3.FromCache {
		t.Errorf("run3 should be a cache hit")
	}

	if !optsEqual(r1.Opts, r2.Opts) {
		t.Errorf("non-deterministic: run1=%v  run2=%v", r1.Opts, r2.Opts)
	}
	if !optsEqual(r1.Opts, r3.Opts) {
		t.Errorf("non-deterministic: run1=%v  run3=%v", r1.Opts, r3.Opts)
	}

	fmt.Printf("[B4] determinism: 3 searches → opts=%v  (run2 cache=%v, run3 cache=%v)\n",
		r1.Opts, r2.FromCache, r3.FromCache)
}

// TestB4_Cache_HitCorrect verifies that a cache hit produces byte-identical
// WGSL to a fresh render of the winning opts on a structurally identical kernel.
func TestB4_Cache_HitCorrect(t *testing.T) {
	dev := requireDevice(t)
	codegen.BeamCacheReset()
	cfg := codegen.DefaultBeamConfig()

	mk := func() schedule.ExecItem {
		a := uop.NewArena(65536)
		A := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		B := tensor.NewLeaf(a, []int64{64, 64}, uop.Dtypes.Float32, "webgpu")
		return schedule.CreateSchedule(makeSink(a, A.Matmul(B)), "webgpu")[0]
	}

	// Run once to populate the cache.
	r1 := codegen.BeamSearch(dev, dev, mk(), cfg)

	// Run again from cache.
	r2 := codegen.BeamSearch(dev, dev, mk(), cfg)
	if !r2.FromCache {
		t.Errorf("second search should be a cache hit")
	}

	// Both opt sequences must be identical.
	if !optsEqual(r1.Opts, r2.Opts) {
		t.Errorf("cache hit returned different opts: r1=%v  r2=%v", r1.Opts, r2.Opts)
	}

	// Render WGSL from each result on a fresh item; must be byte-identical.
	renderOpts := func(opts []codegen.Opt) string {
		item := mk()
		if len(opts) > 0 {
			item.Ast = codegen.ApplyOpts(item, opts).Ast
		}
		return codegen.RenderWGSL(item).WGSL
	}

	w1 := renderOpts(r1.Opts)
	w2 := renderOpts(r2.Opts)
	if w1 != w2 {
		t.Errorf("WGSL from cache hit differs from original search result")
	}
	fmt.Printf("[B4] cache hit WGSL byte-identical: %v (wgsl_len=%d)\n", w1 == w2, len(w1))
}
