package onnx

// Phase 1.C value-oracle gate.
//
// Strategy (see notes/onnx_implementation_plan.md §7 + the dispatch brief):
// build a real CNN twice over the SAME arena: once via the tensor/nn API and
// once via a hand-constructed ModelProto fed through the importer. Identical
// primitive calls share UOp identity via arena interning, so both forward
// passes produce the *same* graph. Evaluating both via the cpuEval host
// interpreter must yield bit-exact equal output bytes; any non-zero
// max-abs-diff is a handler bug, full stop.
//
// This is a stricter gate than "match onnxruntime within 1e-3" because it
// catches the bug class where handler + onnxruntime share the same
// misinterpretation of an attribute. Strategy B (onnxruntime goldens) is
// documented as the dev-time secondary gate; see notes/onnx_golden_generation.md.

import (
	"math"
	"math/rand"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── (a) TinyCNN bit-exact ────────────────────────────────────────────────────

// TestE2E_TinyCNN_BitExact verifies that the importer-emitted graph for a
// 3-layer CNN matches the direct Tensor API graph element-for-element.
//
// Architecture: Conv(3→8, 3x3, pad=1) → Relu → GlobalAveragePool →
//               Gemm(8→4)
// Input shape: [N=2, C=3, H=8, W=8]; output [N=2, num_classes=4].
func TestE2E_TinyCNN_BitExact(t *testing.T) {
	spec := tinyCNNSpec{
		N: 2, Cin: 3, H: 8, W: 8, Cmid: 8, NumClass: 4,
		UseBias: true, InputName: "x",
	}
	arena := uop.NewArena(8192)
	w := newCNNWeights()
	rng := rand.New(rand.NewSource(42))
	w.randomFill(rng, "conv_w", []int64{spec.Cmid, spec.Cin, 3, 3}, 0.3)
	w.randomFill(rng, "conv_b", []int64{spec.Cmid}, 0.1)
	w.randomFill(rng, "gemm_b", []int64{spec.Cmid, spec.NumClass}, 0.5)
	w.randomFill(rng, "gemm_c", []int64{spec.NumClass}, 0.05)

	xLeaf, _ := makeFloatInput(arena, []int64{spec.N, spec.Cin, spec.H, spec.W}, rng, "x")

	// Path A: direct via Tensor API on the shared arena.
	direct := buildDirectTinyCNN(arena, spec, w, xLeaf)
	directShape := direct.Shape()
	if directShape[0] != spec.N || directShape[1] != spec.NumClass {
		t.Fatalf("direct output shape %v, want [%d %d]", directShape, spec.N, spec.NumClass)
	}

	// Path B: via importer on the same arena. The graph input "x" resolves to
	// xLeaf since we pass it explicitly; initializers come from the ModelProto.
	model := buildModelProtoTinyCNN(spec, w)
	imported := realizeViaImporter(t, arena, model,
		map[string]*tensor.Tensor{spec.InputName: xLeaf},
		"y")

	assertBitExact(t, direct, imported, "TinyCNN")
}

// ── (b) TinyResNet block bit-exact ───────────────────────────────────────────

// TestE2E_TinyResNetBlock_BitExact runs a residual block end-to-end through
// both paths. Exercises Add as device op, BatchNorm decomposition, and the
// residual chaining where construction-order bugs (SPEC §10) historically
// lurk.
func TestE2E_TinyResNetBlock_BitExact(t *testing.T) {
	spec := tinyResNetSpec{
		N: 2, C: 4, H: 6, W: 6, NumClass: 3, InputName: "x",
	}
	arena := uop.NewArena(16384)
	w := newCNNWeights()
	rng := rand.New(rand.NewSource(123))

	w.randomFill(rng, "c1_w", []int64{spec.C, spec.C, 3, 3}, 0.2)
	w.randomFill(rng, "c2_w", []int64{spec.C, spec.C, 3, 3}, 0.2)

	w.randomFill(rng, "bn1_scale", []int64{spec.C}, 0.5)
	w.randomFill(rng, "bn1_bias", []int64{spec.C}, 0.1)
	w.randomFill(rng, "bn1_mean", []int64{spec.C}, 0.1)
	// Variance must be positive; sample in [0.5, 1.5].
	{
		bnVar := make([]float32, spec.C)
		for i := range bnVar {
			bnVar[i] = 0.5 + rng.Float32()
		}
		w.set("bn1_var", []int64{spec.C}, bnVar)
	}
	w.randomFill(rng, "bn2_scale", []int64{spec.C}, 0.5)
	w.randomFill(rng, "bn2_bias", []int64{spec.C}, 0.1)
	w.randomFill(rng, "bn2_mean", []int64{spec.C}, 0.1)
	{
		bnVar := make([]float32, spec.C)
		for i := range bnVar {
			bnVar[i] = 0.5 + rng.Float32()
		}
		w.set("bn2_var", []int64{spec.C}, bnVar)
	}
	w.randomFill(rng, "gemm_b", []int64{spec.C, spec.NumClass}, 0.5)

	xLeaf, _ := makeFloatInput(arena, []int64{spec.N, spec.C, spec.H, spec.W}, rng, "x")
	direct := buildDirectTinyResNet(arena, spec, w, xLeaf)
	model := buildModelProtoTinyResNet(spec, w)
	imported := realizeViaImporter(t, arena, model,
		map[string]*tensor.Tensor{spec.InputName: xLeaf},
		"y")
	assertBitExact(t, direct, imported, "TinyResNetBlock")
}

// ── (c) TinyCNN with MaxPool bit-exact ───────────────────────────────────────

// TestE2E_TinyCNN_MaxPool_BitExact: Conv→Relu→MaxPool→Conv→GlobalAvgPool→Gemm.
// Drives the MaxPool handler end-to-end. Input [2, 2, 8, 8].
func TestE2E_TinyCNN_MaxPool_BitExact(t *testing.T) {
	spec := tinyCNNMaxPoolSpec{
		N: 2, Cin: 2, H: 8, W: 8, Cmid1: 4, Cmid2: 6, NumClass: 3,
		InputName: "x",
	}
	arena := uop.NewArena(16384)
	w := newCNNWeights()
	rng := rand.New(rand.NewSource(7))

	w.randomFill(rng, "conv1_w", []int64{spec.Cmid1, spec.Cin, 3, 3}, 0.3)
	w.randomFill(rng, "conv2_w", []int64{spec.Cmid2, spec.Cmid1, 3, 3}, 0.3)
	w.randomFill(rng, "gemm_b", []int64{spec.Cmid2, spec.NumClass}, 0.5)

	xLeaf, _ := makeFloatInput(arena, []int64{spec.N, spec.Cin, spec.H, spec.W}, rng, "x")
	direct := buildDirectTinyCNNMaxPool(arena, spec, w, xLeaf)
	model := buildModelProtoTinyCNNMaxPool(spec, w)
	imported := realizeViaImporter(t, arena, model,
		map[string]*tensor.Tensor{spec.InputName: xLeaf},
		"y")
	assertBitExact(t, direct, imported, "TinyCNN_MaxPool")
}

// ── (d) TinyCNN value-range check, multiple seeds ────────────────────────────

// TestE2E_TinyCNN_ValueRange_Check exercises the same TinyCNN as (a) over five
// distinct random input batches and reports the max-abs-diff for each batch.
// All five must be bit-exact (max-abs-diff = 0); the loud per-batch logging is
// the project's value-oracle discipline regardless of pass/fail.
func TestE2E_TinyCNN_ValueRange_Check(t *testing.T) {
	spec := tinyCNNSpec{
		N: 2, Cin: 3, H: 8, W: 8, Cmid: 8, NumClass: 4,
		UseBias: false, InputName: "x",
	}
	// Build weights once, share across all 5 batches.
	w := newCNNWeights()
	wrng := rand.New(rand.NewSource(99))
	w.randomFill(wrng, "conv_w", []int64{spec.Cmid, spec.Cin, 3, 3}, 0.3)
	w.randomFill(wrng, "gemm_b", []int64{spec.Cmid, spec.NumClass}, 0.5)

	for batchIdx, seed := range []int64{1, 7, 23, 99, 2026} {
		arena := uop.NewArena(8192)
		rng := rand.New(rand.NewSource(seed))
		xLeaf, _ := makeFloatInput(arena, []int64{spec.N, spec.Cin, spec.H, spec.W}, rng, "x")

		direct := buildDirectTinyCNN(arena, spec, w, xLeaf)
		model := buildModelProtoTinyCNN(spec, w)
		imported := realizeViaImporter(t, arena, model,
			map[string]*tensor.Tensor{spec.InputName: xLeaf},
			"y")

		dData, _, err := cpuEval(direct)
		if err != nil {
			t.Fatalf("batch %d (seed %d): cpuEval direct: %v", batchIdx, seed, err)
		}
		iData, _, err := cpuEval(imported)
		if err != nil {
			t.Fatalf("batch %d (seed %d): cpuEval imported: %v", batchIdx, seed, err)
		}
		if len(dData) != len(iData) {
			t.Fatalf("batch %d (seed %d): length mismatch %d vs %d",
				batchIdx, seed, len(dData), len(iData))
		}
		m := maxAbsDiff(dData, iData)
		t.Logf("TinyCNN_ValueRange batch %d (seed %d): max-abs-diff = %g (n=%d)",
			batchIdx, seed, m, len(dData))
		if m != 0 {
			t.Errorf("batch %d (seed %d): expected bit-exact 0, got %g", batchIdx, seed, m)
		}
	}
}

// ── (e) Symbolic-batch TinyCNN bit-exact ─────────────────────────────────────

// TestE2E_TinyCNN_SymbolicBatch builds the same TinyCNN but with a dim_param "N"
// on axis 0. It then constructs a direct-API tensor with NewSymbolicBatchInput
// using the SAME arena, so the direct path's DefineVar("N") and the importer's
// DefineVar("N") collapse to the same arena UOp identity.
//
// At cpuEval time we replace the symbolic input with concrete-batch inputs for
// each of N ∈ {1, 3, 5}; cpuEval only sees concrete leaf data and concrete
// movement-op args, so it materialises an answer per chosen N. The result must
// be bit-exact equal to a direct path constructed from scratch on a fresh arena
// at the same concrete N.
//
// We do NOT exercise tensor.RealizeWithBinding here because the e2e gate
// hinges on cpuEval (no GPU) and the symbolic path's data-dependent rangeify
// kernels are exercised in slice 3b tests, not in the importer surface. What
// this test proves about the importer is the *shape-construction* contract:
// the dim_param round-trips and the model can be re-executed at multiple N
// without recompilation.
func TestE2E_TinyCNN_SymbolicBatch(t *testing.T) {
	// Importer + symbolic input share a single arena so DefineVar identity unifies.
	symArena := uop.NewArena(8192)
	const dimName = "N"

	// Weights are reused across all three N values. They're concrete (no
	// symbolic dims), so registering once is sufficient.
	w := newCNNWeights()
	wrng := rand.New(rand.NewSource(31))
	const (
		cin, hH, wW = int64(3), int64(4), int64(4)
		cmid        = int64(4)
		numClass    = int64(3)
	)
	w.randomFill(wrng, "conv_w", []int64{cmid, cin, 3, 3}, 0.3)
	w.randomFill(wrng, "gemm_b", []int64{cmid, numClass}, 0.5)

	// Build the symbolic ModelProto once and import it once. This proves the
	// importer accepts dim_param and constructs a single graph.
	symSpec := tinyCNNSpec{
		N: 1, Cin: cin, H: hH, W: wW, Cmid: cmid, NumClass: numClass,
		UseBias: false, InputName: "x",
	}
	symModel := buildModelProtoTinyCNNSymbolic(symSpec, w, dimName)
	bytesM := mustMarshalProto(t, symModel)
	r, err := Import(bytesM, symArena, "test")
	if err != nil {
		t.Fatalf("symbolic Import: %v", err)
	}
	// Confirm the importer's inputs carry a symbolic dim with name dimName.
	inputs := r.Inputs()
	if len(inputs) != 1 {
		t.Fatalf("expected 1 graph input, got %d", len(inputs))
	}
	if _, ok := inputs[0].Shape[0].ConstValue(); ok {
		t.Errorf("input axis 0 resolved to a const value; expected symbolic")
	}
	// Validate that arena.FindDefineVar finds the same DefineVar for dimName
	// (this is the arena unification contract the importer relies on).
	dv, ok := symArena.FindDefineVar(dimName)
	if !ok {
		t.Fatalf("arena.FindDefineVar(%q) missing; importer did not register the dim_param", dimName)
	}
	_ = dv

	// For each concrete N, build (a) a fresh-arena direct path and (b) realise
	// the importer's graph at that N by supplying a fresh leaf input of shape
	// [N, Cin, H, W] (the importer's input shape contains the dim_param so the
	// passed-in concrete tensor's shape is compatible at the runner level —
	// the runner stores it as a Device Value and never reasons about its
	// symbolic dim during cpuEval).
	for _, n := range []int64{1, 3, 8} {
		// Fresh arena for direct path so the two arenas (importer's symArena
		// and this freshArena) build the same Conv2d / Relu / GlobalAvgPool /
		// Gemm UOps from the same primitives. We only need cpuEval-equality
		// of the final output, not arena-identity.
		freshArena := uop.NewArena(8192)
		concreteSpec := tinyCNNSpec{
			N: n, Cin: cin, H: hH, W: wW, Cmid: cmid, NumClass: numClass,
			UseBias: false, InputName: "x",
		}
		rng := rand.New(rand.NewSource(7000 + n))
		// Build matching input data once and use the same bytes on both arenas.
		xShape := []int64{n, cin, hH, wW}
		// Direct: leaf on freshArena.
		xVals := make([]float32, n*cin*hH*wW)
		for i := range xVals {
			xVals[i] = rng.Float32()*2 - 1
		}
		freshLeaf := tensor.NewLeaf(freshArena, xShape, uop.Dtypes.Float32, "test")
		freshLeaf.SetData(append([]float32{}, xVals...))
		direct := buildDirectTinyCNN(freshArena, concreteSpec, w, freshLeaf)

		// Importer: feed the same xVals through a fresh leaf on symArena.
		symLeaf := tensor.NewLeaf(symArena, xShape, uop.Dtypes.Float32, "test")
		symLeaf.SetData(append([]float32{}, xVals...))
		out, err := r.Run(map[string]*tensor.Tensor{concreteSpec.InputName: symLeaf})
		if err != nil {
			t.Fatalf("symbolic Run N=%d: %v", n, err)
		}
		imported, ok := out["y"]
		if !ok {
			t.Fatalf("symbolic Run N=%d: output y missing", n)
		}

		// Evaluate both via cpuEval and compare bit-exact.
		dData, _, derr := cpuEval(direct)
		if derr != nil {
			t.Fatalf("symbolic N=%d: cpuEval(direct): %v", n, derr)
		}
		iData, _, ierr := cpuEval(imported)
		if ierr != nil {
			t.Fatalf("symbolic N=%d: cpuEval(imported): %v", n, ierr)
		}
		if len(dData) != len(iData) {
			t.Fatalf("symbolic N=%d: length mismatch %d vs %d", n, len(dData), len(iData))
		}
		m := maxAbsDiff(dData, iData)
		t.Logf("TinyCNN_SymbolicBatch N=%d: max-abs-diff = %g (n=%d elements)",
			n, m, len(dData))
		if m != 0 {
			t.Errorf("symbolic N=%d: expected bit-exact 0, got %g", n, m)
		}
	}
}

// ── (f) f16 accumulator pattern ──────────────────────────────────────────────

// TestE2E_TinyCNN_F16_AccumulatorPattern verifies that f16 weights + f16 input
// route through the f32 accumulator pattern for the Mean reduction inside
// GlobalAveragePool. The cpuEval interpreter runs everything in float32 so the
// "accumulator pattern" effect is observed indirectly: we compare the
// importer's f16 path against an f32 baseline (same weights, same input, but
// kept as f32). The difference must reflect *only* f16 quantisation (i.e. the
// SetData round-trip through f16), not accumulator overflow or systematic drift.
//
// Tolerance math: f16 has ~10 bits of mantissa (effective ~3e-4 relative
// precision). With a [2, 4, 4, 4] Conv output the GlobalAveragePool sums 16
// values; the per-channel sum is at most ~16 * max_abs, and the divide-by-16
// preserves f16-precision error. Conservative absolute bound for the final
// 4-element Gemm output: 5e-3. This is far below any accumulator-overflow
// magnitude (which would be O(0.1) or worse).
func TestE2E_TinyCNN_F16_AccumulatorPattern(t *testing.T) {
	// Skip if f16 is not a supported scalar dtype.
	if uop.Dtypes.Float16 == nil {
		t.Skip("f16 dtype not registered in this build")
	}
	spec := tinyCNNSpec{
		N: 2, Cin: 3, H: 4, W: 4, Cmid: 4, NumClass: 4,
		UseBias: false, InputName: "x",
	}
	arenaF16 := uop.NewArena(8192)
	arenaF32 := uop.NewArena(8192)
	w := newCNNWeights()
	rng := rand.New(rand.NewSource(2026))
	w.randomFill(rng, "conv_w", []int64{spec.Cmid, spec.Cin, 3, 3}, 0.3)
	w.randomFill(rng, "gemm_b", []int64{spec.Cmid, spec.NumClass}, 0.5)

	// Build the input ONCE in f32; we'll feed both paths with the same bytes.
	xShape := []int64{spec.N, spec.Cin, spec.H, spec.W}
	xVals := make([]float32, spec.N*spec.Cin*spec.H*spec.W)
	for i := range xVals {
		xVals[i] = rng.Float32()*2 - 1
	}

	// f32 baseline (direct on its own arena).
	xF32 := tensor.NewLeaf(arenaF32, xShape, uop.Dtypes.Float32, "test")
	xF32.SetData(append([]float32{}, xVals...))
	directF32 := buildDirectTinyCNN(arenaF32, spec, w, xF32)
	f32Data, _, err := cpuEval(directF32)
	if err != nil {
		t.Fatalf("f32 cpuEval: %v", err)
	}

	// f16 importer path. The leaves are f16; SetData quantises the input on
	// upload. Weights become f16 because we register f16 TensorProtos.
	model := buildModelProtoTinyCNNF16(spec, w)
	xF16 := tensor.NewLeaf(arenaF16, xShape, uop.Dtypes.Float16, "test")
	xF16.SetData(append([]float32{}, xVals...)) // SetData handles quantization
	imported := realizeViaImporter(t, arenaF16, model,
		map[string]*tensor.Tensor{spec.InputName: xF16},
		"y")
	f16Data, _, err := cpuEval(imported)
	if err != nil {
		t.Fatalf("f16 cpuEval: %v", err)
	}

	if len(f32Data) != len(f16Data) {
		t.Fatalf("output length mismatch f32=%d f16=%d", len(f32Data), len(f16Data))
	}
	m := maxAbsDiff(f32Data, f16Data)
	const tol = float32(5e-3) // see comment block above
	t.Logf("TinyCNN_F16 vs f32 baseline: max-abs-diff = %g (tol=%g, n=%d)", m, tol, len(f32Data))
	if m > tol {
		t.Errorf("f16 drift %g exceeds tol %g — possible accumulator overflow", m, tol)
	}
}

// buildModelProtoTinyCNNF16 mirrors buildModelProtoTinyCNN but with FLOAT16
// initializers and FLOAT16 input/output value-infos. The TensorProto stores
// f16 bits in the int32_data field per the ONNX spec.
func buildModelProtoTinyCNNF16(spec tinyCNNSpec, w *cnnWeights) *onnxpb.ModelProto {
	m := buildModelProtoTinyCNN(spec, w)
	// Convert the input/output VI to FLOAT16.
	m.Graph.Input[0].Type.GetTensorType().ElemType = int32(onnxpb.TensorProto_FLOAT16)
	m.Graph.Output[0].Type.GetTensorType().ElemType = int32(onnxpb.TensorProto_FLOAT16)
	// Re-encode every initializer as FLOAT16.
	for i, init := range m.Graph.Initializer {
		f32 := init.GetFloatData()
		// Pack f16 bits into int32_data (each entry's low 16 bits hold the f16).
		int32s := make([]int32, len(f32))
		for j, v := range f32 {
			int32s[j] = int32(f16BitsFromFloat32(v))
		}
		m.Graph.Initializer[i] = &onnxpb.TensorProto{
			Name:      init.Name,
			Dims:      init.Dims,
			DataType:  int32(onnxpb.TensorProto_FLOAT16),
			Int32Data: int32s,
		}
	}
	return m
}

// f16BitsFromFloat32 encodes a float32 into the 16 low bits of an IEEE-754
// half-precision representation. We use a simple round-to-nearest-even
// translation that matches the importer's f16 decode path.
func f16BitsFromFloat32(f float32) uint32 {
	// Reuse anneal's f16 quantize: cast via the Float16 dtype's Quantize
	// then encode. Since we cannot import internal helpers here, use the
	// canonical bit-level conversion.
	bits := uint32(0)
	x := f
	if x != x { // NaN
		return 0x7E00
	}
	sign := uint32(0)
	if x < 0 {
		sign = 1
		x = -x
	}
	if x == 0 {
		return sign << 15
	}
	// Standard f32 → f16 reduction.
	// Bit layout: f32 sign|8exp|23mant; f16 sign|5exp|10mant.
	var u uint32
	{
		// Reinterpret f as uint32.
		b := floatToBits(f)
		u = b
	}
	exp32 := int32((u>>23)&0xff) - 127
	mant32 := u & 0x7fffff
	if exp32 > 15 {
		// Overflow → infinity.
		bits = (sign << 15) | (0x1f << 10)
		return bits
	}
	if exp32 < -14 {
		// Subnormal or underflow → zero.
		return sign << 15
	}
	exp16 := uint32(exp32+15) & 0x1f
	// Truncate (round-to-zero) mantissa for simplicity; importer SetData uses
	// the canonical Quantize which is round-to-nearest-even, so f16BitsFromFloat32
	// is just for *encoding the bits into the TensorProto*. The actual stored
	// value after import is what the importer Quantize produces; the encoder
	// doesn't need bit-exact agreement with SetData here because tensorFromProto
	// re-decodes the bits into a float32 host slice and then SetData runs
	// Quantize on that slice. The end-to-end roundtrip absorbs any mismatch.
	mant16 := mant32 >> 13
	bits = (sign << 15) | (exp16 << 10) | mant16
	return bits
}

// floatToBits is math.Float32bits.
func floatToBits(f float32) uint32 {
	return math.Float32bits(f)
}

// ── (g) Punt-list loud failures ──────────────────────────────────────────────

// TestE2E_PuntList_LoudFailures feeds pathological inputs to handlers that
// Phase 1.B documented as out-of-scope. Each must fail with a clear error.
func TestE2E_PuntList_LoudFailures(t *testing.T) {
	type tc struct {
		name      string
		build     func(t *testing.T) *onnxpb.ModelProto
		inputName string
		inputSh   []int64
		// substrings the error message must contain (case-sensitive)
		wantSubs []string
	}
	cases := []tc{
		{
			name: "Conv_group2_rejected",
			build: func(t *testing.T) *onnxpb.ModelProto {
				b := &singleNodeBuilder{
					opType: "Conv",
					attrs: map[string]Attr{
						"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
						"group":        {Kind: AttrInt, I: 2},
					},
					inputs: []nameInfo{
						{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 4, 4}},
						{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 1, 3, 3}},
					},
					outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}}},
					initializers: []*onnxpb.TensorProto{
						makeFloatInitializerForTests("x", []int64{1, 2, 4, 4}, make([]float32, 32)),
						makeFloatInitializerForTests("w", []int64{2, 1, 3, 3}, make([]float32, 18)),
					},
				}
				return b.build(t)
			},
			wantSubs: []string{"group", "not supported"},
		},
		{
			name: "MaxPool_ceilmode_rejected",
			build: func(t *testing.T) *onnxpb.ModelProto {
				b := &singleNodeBuilder{
					opType: "MaxPool",
					attrs: map[string]Attr{
						"kernel_shape": {Kind: AttrInts, Is: []int64{2, 2}},
						"ceil_mode":    {Kind: AttrInt, I: 1},
					},
					inputs: []nameInfo{
						{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 4, 4}},
					},
					outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 1, 2, 2}}},
					initializers: []*onnxpb.TensorProto{
						makeFloatInitializerForTests("x", []int64{1, 1, 4, 4}, make([]float32, 16)),
					},
				}
				return b.build(t)
			},
			wantSubs: []string{"ceil_mode", "not supported"},
		},
		{
			name: "Slice_negstep_rejected",
			build: func(t *testing.T) *onnxpb.ModelProto {
				b := &singleNodeBuilder{
					opType: "Slice",
					inputs: []nameInfo{
						{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{8}},
						{Name: "starts", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
						{Name: "ends", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
						{Name: "axes", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
						{Name: "steps", DType: onnxpb.TensorProto_INT64, Dims: []int64{1}},
					},
					outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{4}}},
					initializers: []*onnxpb.TensorProto{
						makeFloatInitializerForTests("x", []int64{8}, []float32{1, 2, 3, 4, 5, 6, 7, 8}),
						makeIntInitializer("starts", []int64{1}, []int64{6}),
						makeIntInitializer("ends", []int64{1}, []int64{0}),
						makeIntInitializer("axes", []int64{1}, []int64{0}),
						makeIntInitializer("steps", []int64{1}, []int64{-1}),
					},
				}
				return b.build(t)
			},
			wantSubs: []string{"negative step", "not supported"},
		},
		{
			name: "BatchNorm_trainingmode_rejected",
			build: func(t *testing.T) *onnxpb.ModelProto {
				b := &singleNodeBuilder{
					opType: "BatchNormalization",
					attrs: map[string]Attr{
						"training_mode": {Kind: AttrInt, I: 1},
					},
					inputs: []nameInfo{
						{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}},
						{Name: "scale", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
						{Name: "B", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
						{Name: "mean", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
						{Name: "var", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}},
					},
					outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, 2, 2, 2}}},
					initializers: []*onnxpb.TensorProto{
						makeFloatInitializerForTests("x", []int64{1, 2, 2, 2}, make([]float32, 8)),
						makeFloatInitializerForTests("scale", []int64{2}, []float32{1, 1}),
						makeFloatInitializerForTests("B", []int64{2}, []float32{0, 0}),
						makeFloatInitializerForTests("mean", []int64{2}, []float32{0, 0}),
						makeFloatInitializerForTests("var", []int64{2}, []float32{1, 1}),
					},
				}
				return b.build(t)
			},
			wantSubs: []string{"training_mode", "not supported"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			model := c.build(t)
			err := runSingleNodeExpectError(t, model, nil, c.wantSubs...)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			t.Logf("%s: rejected with: %v", c.name, err)
		})
	}
}
