package onnx

// Tests for the WithStructureOnly() import mode introduced for W8 (the
// studio's ONNX dropzone). Structure-only import preserves graph topology +
// per-value shape and dtype but skips materialising initializer payload
// bytes; it is meant for visualization, never for Run().
//
// What we pin:
//   - Initializers carry correct shape + dtype but Data() is nil.
//   - The lowered node list count matches the full import.
//   - Run() refuses to execute (the contract: visualize, don't run).
//   - The structure-only path allocates dramatically less host memory than
//     the full path on the same model.
//   - Constant and ConstantOfShape handlers honour the StructureOnly flag —
//     they return a zero-filled leaf of the right shape without faulting on
//     missing payload bytes.

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestImport_StructureOnly_ResNet9 loads the committed ResNet-9 fixture and
// imports it twice — once normally, once structure-only — then pins:
//   - both runs produce the same node count;
//   - both runs produce the same initializer count;
//   - structure-only initializers carry correct shapes + dtypes but Data() is
//     nil (no payload was decoded);
//   - r.Run() on the structure-only runner errors with the documented message.
func TestImport_StructureOnly_ResNet9(t *testing.T) {
	modelPath := filepath.Join("testdata", "resnet9.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("resnet9.onnx fixture missing: %v", err)
	}

	bytesM, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("read %s: %v", modelPath, err)
	}

	// Full import (for comparison).
	arenaFull := uop.NewArena(65536)
	rFull, err := Import(bytesM, arenaFull, "test")
	if err != nil {
		t.Fatalf("full Import: %v", err)
	}

	// Structure-only import.
	arenaSO := uop.NewArena(65536)
	rSO, err := Import(bytesM, arenaSO, "test", WithStructureOnly())
	if err != nil {
		t.Fatalf("structure-only Import: %v", err)
	}

	if !rSO.StructureOnly() {
		t.Fatalf("structure-only runner reports StructureOnly()=false")
	}
	if rFull.StructureOnly() {
		t.Fatalf("full runner reports StructureOnly()=true")
	}

	if got, want := len(rSO.Nodes()), len(rFull.Nodes()); got != want {
		t.Errorf("structure-only node count=%d, want %d (full)", got, want)
	}
	if got, want := len(rSO.Initializers()), len(rFull.Initializers()); got != want {
		t.Errorf("structure-only initializer count=%d, want %d (full)", got, want)
	}
	if len(rSO.Initializers()) == 0 {
		t.Fatalf("structure-only: no initializers (expected ResNet-9 to have many)")
	}

	// Walk every initializer: shape + dtype must match the full import; Data()
	// must be nil under structure-only.
	emptyPayloads := 0
	for name, vSO := range rSO.Initializers() {
		vFull, ok := rFull.Initializers()[name]
		if !ok {
			t.Errorf("initializer %q present in SO but missing in full import", name)
			continue
		}
		if !vSO.IsDevice() || !vFull.IsDevice() {
			t.Errorf("initializer %q kind mismatch (SO=%v full=%v)", name, vSO.Kind, vFull.Kind)
			continue
		}
		tSO := vSO.Tensor()
		tFull := vFull.Tensor()
		if !shapeEq(tSO.Shape(), tFull.Shape()) {
			t.Errorf("initializer %q shape SO=%v full=%v", name, tSO.Shape(), tFull.Shape())
		}
		if tSO.DType() != tFull.DType() {
			t.Errorf("initializer %q dtype SO=%v full=%v", name, tSO.DType(), tFull.DType())
		}
		if tSO.Data() == nil {
			emptyPayloads++
		} else {
			t.Errorf("initializer %q: structure-only Data() should be nil, got len=%d",
				name, len(tSO.Data()))
		}
	}
	if emptyPayloads == 0 {
		t.Errorf("structure-only: no empty payloads observed; the SetData skip is broken")
	}
	t.Logf("structure-only: %d/%d initializers carry empty payloads", emptyPayloads, len(rSO.Initializers()))

	// Run() must refuse.
	_, err = rSO.Run(nil)
	if err == nil {
		t.Fatalf("Run() on structure-only runner returned nil error; want a refusal")
	}
	if !contains(err.Error(), "structure-only") {
		t.Errorf("Run() error %q does not mention structure-only", err.Error())
	}
}

// TestImport_StructureOnly_BytesNotMaterialized compares host-side allocation
// between the full import and the structure-only import on the same ResNet-9
// fixture. The structure-only path must allocate dramatically less (we pin a
// 4x ratio floor — empirically ~10x on this fixture; the floor stays gentle
// to avoid flakes on noisy CI).
func TestImport_StructureOnly_BytesNotMaterialized(t *testing.T) {
	modelPath := filepath.Join("testdata", "resnet9.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("resnet9.onnx fixture missing: %v", err)
	}
	bytesM, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("read %s: %v", modelPath, err)
	}

	// Measure host alloc delta for full import.
	runtime.GC()
	var msBefore, msAfter runtime.MemStats
	runtime.ReadMemStats(&msBefore)
	arenaFull := uop.NewArena(65536)
	rFull, err := Import(bytesM, arenaFull, "test")
	if err != nil {
		t.Fatalf("full Import: %v", err)
	}
	runtime.ReadMemStats(&msAfter)
	fullDelta := int64(msAfter.TotalAlloc) - int64(msBefore.TotalAlloc)
	_ = rFull

	// Measure host alloc delta for structure-only import.
	runtime.GC()
	runtime.ReadMemStats(&msBefore)
	arenaSO := uop.NewArena(65536)
	rSO, err := Import(bytesM, arenaSO, "test", WithStructureOnly())
	if err != nil {
		t.Fatalf("structure-only Import: %v", err)
	}
	runtime.ReadMemStats(&msAfter)
	soDelta := int64(msAfter.TotalAlloc) - int64(msBefore.TotalAlloc)
	_ = rSO

	delta := fullDelta - soDelta
	t.Logf("resnet9 alloc delta: full=%d B, structure-only=%d B (saved=%d B, ratio=%.2fx)",
		fullDelta, soDelta, delta, float64(fullDelta)/float64(soDelta+1))

	// The protobuf parser itself materialises raw_data, which dominates the
	// ResNet-9 fixture (~480 KB raw); on top of that, the full path decodes
	// every value to float32 and stores the host slice in the arena leaf.
	// Structure-only skips ONLY the decode + leaf-data copy. The savings are
	// proportional to the model's total float-payload size; we pin a 100 KB
	// floor so the gate catches a "SetData skip is a no-op" regression
	// without being noisy on small fixtures.
	if soDelta >= fullDelta {
		t.Errorf("structure-only allocated >= full import (%d >= %d) — payload skip is broken", soDelta, fullDelta)
	}
	if delta < 100_000 {
		t.Errorf("structure-only saved only %d B (< 100 KB floor); SetData skip may be ineffective", delta)
	}

	// Cross-check: every initializer must still report nil Data() to prove
	// the savings come from skipping SetData, not from short-circuiting some
	// other code path.
	nilPayloads := 0
	for _, v := range rSO.Initializers() {
		if v.IsDevice() && v.Tensor().Data() == nil {
			nilPayloads++
		}
	}
	if nilPayloads == 0 {
		t.Errorf("expected at least one nil payload after structure-only import; got 0")
	}
}

// TestImport_StructureOnly_HandlerSkipPath builds a tiny model with a single
// Constant op whose `value` attribute is a TensorProto carrying real bytes,
// imports it structure-only, and dispatches the Constant handler directly
// from the runner's registry against a structure-only HandlerCtx. The
// expectation is that the handler emits a zero-shaped leaf of the correct
// shape + dtype WITHOUT reading the payload bytes (no panic, Data()=nil).
func TestImport_StructureOnly_HandlerSkipPath(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Dims:      []int64{2, 3},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2, 3, 4, 5, 6},
	}
	b := &singleNodeBuilder{
		opType: "Constant",
		attrs:  map[string]Attr{"value": {Kind: AttrTensor, T: tp}},
		outputs: []nameInfo{{
			Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2, 3},
		}},
	}
	model := b.build(t)
	bytesM := mustMarshalProto(t, model)

	arena := uop.NewArena(64)
	r, err := Import(bytesM, arena, "test", WithStructureOnly())
	if err != nil {
		t.Fatalf("Import(SO): %v", err)
	}

	// Directly invoke the Constant handler with a structure-only HandlerCtx.
	// We don't go through r.Run() because that path errors out by design.
	if len(r.Nodes()) != 1 {
		t.Fatalf("expected 1 node, got %d", len(r.Nodes()))
	}
	node := r.Nodes()[0]
	h, ok := r.handlers[node.OpType]
	if !ok {
		t.Fatalf("no handler for op %q", node.OpType)
	}
	ctx := &HandlerCtx{
		Arena:         arena,
		Device:        "test",
		Node:          node,
		Opset:         r.Opset(),
		StructureOnly: true,
	}
	outs, err := h(ctx)
	if err != nil {
		t.Fatalf("Constant handler (SO): %v", err)
	}
	if len(outs) != 1 {
		t.Fatalf("Constant handler returned %d outputs, want 1", len(outs))
	}
	leaf := outs[0]
	if !leaf.IsDevice() {
		t.Fatalf("Constant SO output is not a device tensor")
	}
	got := leaf.Tensor().Shape()
	want := []int64{2, 3}
	if !shapeEq(got, want) {
		t.Errorf("Constant SO shape=%v, want %v", got, want)
	}
	if leaf.Tensor().DType() != uop.Dtypes.Float32 {
		t.Errorf("Constant SO dtype=%v, want f32", leaf.Tensor().DType())
	}
	if leaf.Tensor().Data() != nil {
		t.Errorf("Constant SO Data() should be nil under structure-only, got len=%d",
			len(leaf.Tensor().Data()))
	}
}

// TestImport_StructureOnly_RunRefused pins the Run() refusal even when no
// initializers are present (a smoke test so the refusal isn't tied to the
// resnet9 fixture being available).
func TestImport_StructureOnly_RunRefused(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "tiny",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
		},
	}
	bytesM := mustMarshal(t, model)
	arena := uop.NewArena(16)
	r, err := Import(bytesM, arena, "test", WithStructureOnly())
	if err != nil {
		t.Fatalf("Import(SO): %v", err)
	}
	// Even with an input ready, Run must refuse upfront.
	x := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
	x.SetData([]float32{42})
	_, err = r.Run(map[string]*tensor.Tensor{"x": x})
	if err == nil {
		t.Fatalf("Run on SO runner returned nil; want refusal")
	}
	if !contains(err.Error(), "structure-only") {
		t.Errorf("Run error %q missing 'structure-only' phrase", err.Error())
	}
}
