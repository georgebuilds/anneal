package onnx

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Accessors that the existing tests don't cover ────────────────────────────

func TestRunner_Accessors(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:  "acc",
			Input: []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{
				makeValueInfo("y", onnxpb.TensorProto_FLOAT, []int64{1}),
			},
			Node: []*onnxpb.NodeProto{
				{Name: "n1", OpType: "Identity", Input: []string{"x"}, Output: []string{"y"}},
			},
		},
	}
	bytes := mustMarshal(t, model)
	arena := uop.NewArena(8)
	r, err := Import(bytes, arena, "test-device")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	if r.Arena() != arena {
		t.Errorf("Arena() != input arena")
	}
	if r.Device() != "test-device" {
		t.Errorf("Device()=%q, want test-device", r.Device())
	}
	if len(r.Nodes()) != 1 || r.Nodes()[0].OpType != "Identity" {
		t.Errorf("Nodes() unexpected: %+v", r.Nodes())
	}
	if ins := r.Inputs(); len(ins) != 1 || ins[0].Name != "x" {
		t.Errorf("Inputs() unexpected: %+v", ins)
	}
	if outs := r.Outputs(); len(outs) != 1 || outs[0].Name != "y" {
		t.Errorf("Outputs() unexpected: %+v", outs)
	}
}

// ── ImportFile ────────────────────────────────────────────────────────────────

func TestImportFile_MatchesImport(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "if",
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("w", onnxpb.TensorProto_FLOAT, []int64{2})},
			Initializer: []*onnxpb.TensorProto{
				makeFloatTensor("w", []int64{2}, []float32{7, 8}),
			},
		},
	}
	bytes := mustMarshal(t, model)

	dir := t.TempDir()
	path := filepath.Join(dir, "tiny.onnx")
	if err := os.WriteFile(path, bytes, 0o644); err != nil {
		t.Fatalf("write tmp: %v", err)
	}

	arena1 := uop.NewArena(8)
	r1, err := ImportFile(path, arena1, "test")
	if err != nil {
		t.Fatalf("ImportFile: %v", err)
	}
	arena2 := uop.NewArena(8)
	r2, err := Import(bytes, arena2, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	if r1.Opset() != r2.Opset() {
		t.Errorf("opset mismatch: file=%d bytes=%d", r1.Opset(), r2.Opset())
	}
	if len(r1.Initializers()) != len(r2.Initializers()) {
		t.Errorf("initializer counts differ: file=%d bytes=%d",
			len(r1.Initializers()), len(r2.Initializers()))
	}
	// Spot-check the payload.
	w1 := r1.Initializers()["w"].Tensor().Data()
	w2 := r2.Initializers()["w"].Tensor().Data()
	if len(w1) != len(w2) || w1[0] != w2[0] || w1[1] != w2[1] {
		t.Errorf("payload differs: file=%v bytes=%v", w1, w2)
	}
}

func TestImportFile_MissingPath(t *testing.T) {
	arena := uop.NewArena(4)
	_, err := ImportFile("/nope/definitely/does/not/exist.onnx", arena, "test")
	if err == nil {
		t.Fatalf("expected error for missing file, got nil")
	}
	if !strings.Contains(err.Error(), "read") {
		t.Errorf("error %q missing 'read'", err.Error())
	}
}

// ── Malformed protobuf ────────────────────────────────────────────────────────

func TestImport_MalformedBytes(t *testing.T) {
	arena := uop.NewArena(4)
	_, err := Import([]byte{0xff, 0xff, 0xff, 0xff}, arena, "test")
	if err == nil {
		t.Fatalf("expected error for malformed bytes, got nil")
	}
	// Error mentions either "unmarshal" or "parse" per task contract.
	msg := strings.ToLower(err.Error())
	if !strings.Contains(msg, "unmarshal") && !strings.Contains(msg, "parse") {
		t.Errorf("error %q missing 'unmarshal'/'parse'", err.Error())
	}
}

func TestImport_NilArena(t *testing.T) {
	_, err := Import([]byte{}, nil, "test")
	if err == nil {
		t.Fatalf("expected nil-arena error, got nil")
	}
	if !strings.Contains(err.Error(), "nil arena") {
		t.Errorf("error %q missing 'nil arena'", err.Error())
	}
}

func TestImport_NoGraph(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		// Graph deliberately omitted.
	}
	arena := uop.NewArena(4)
	_, err := Import(mustMarshal(t, model), arena, "test")
	if err == nil {
		t.Fatalf("expected 'no graph' error, got nil")
	}
	if !strings.Contains(err.Error(), "no graph") {
		t.Errorf("error %q missing 'no graph'", err.Error())
	}
}

// ── Empty graph ───────────────────────────────────────────────────────────────

func TestImport_EmptyGraph(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{Name: "empty"},
	}
	arena := uop.NewArena(4)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	if len(r.Nodes()) != 0 {
		t.Errorf("Nodes()=%d, want 0", len(r.Nodes()))
	}
	if len(r.Initializers()) != 0 {
		t.Errorf("Initializers()=%d, want 0", len(r.Initializers()))
	}
	out, err := r.Run(nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("Run output=%v, want empty map", out)
	}
}

// ── Opset boundary cases ──────────────────────────────────────────────────────

func TestImport_OpsetAboveCeilingWarns(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 20},
		},
		Graph: &onnxpb.GraphProto{Name: "high"},
	}
	var captured []string
	restore := withWarnCapture(&captured)
	defer restore()

	arena := uop.NewArena(4)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	if r.Opset() != 20 {
		t.Errorf("Opset=%d, want 20", r.Opset())
	}
	if len(captured) != 1 {
		t.Fatalf("warn count=%d, want 1 (high-opset warning)", len(captured))
	}
	if !strings.Contains(captured[0], "20") {
		t.Errorf("warn %q missing version 20", captured[0])
	}
	if !strings.Contains(captured[0], "ceiling") {
		t.Errorf("warn %q missing 'ceiling'", captured[0])
	}
}

func TestImport_NoAiOnnxDomain(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "com.microsoft", Version: 13},
		},
		Graph: &onnxpb.GraphProto{Name: "no-default-domain"},
	}
	arena := uop.NewArena(4)
	_, err := Import(mustMarshal(t, model), arena, "test")
	if err == nil {
		t.Fatalf("expected error for missing ai.onnx import, got nil")
	}
	if !strings.Contains(err.Error(), "no ai.onnx opset_import") {
		t.Errorf("error %q missing 'no ai.onnx opset_import'", err.Error())
	}
}

func TestImport_AiOnnxDomainExplicit(t *testing.T) {
	// "ai.onnx" should be accepted (treated as the primary domain alias).
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "ai.onnx", Version: 13},
		},
		Graph: &onnxpb.GraphProto{Name: "ai.onnx-domain"},
	}
	arena := uop.NewArena(4)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	if r.Opset() != 13 {
		t.Errorf("Opset=%d, want 13", r.Opset())
	}
}

// ── Symbolic dim_param resolution ─────────────────────────────────────────────

// makeSymbolicValueInfo builds a ValueInfoProto with a mix of dim_value and
// dim_param dims. dim is per-axis: positive → DimValue, "" → empty
// (DimValue=0 — used as a dynamic-axis test), anything else → DimParam.
func makeSymbolicValueInfo(name string, dt onnxpb.TensorProto_DataType, dims []any) *onnxpb.ValueInfoProto {
	shape := &onnxpb.TensorShapeProto{}
	for _, d := range dims {
		switch v := d.(type) {
		case int64:
			shape.Dim = append(shape.Dim, &onnxpb.TensorShapeProto_Dimension{
				Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: v},
			})
		case string:
			shape.Dim = append(shape.Dim, &onnxpb.TensorShapeProto_Dimension{
				Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: v},
			})
		case nil:
			// Empty dim — neither DimValue nor DimParam set.
			shape.Dim = append(shape.Dim, &onnxpb.TensorShapeProto_Dimension{})
		}
	}
	return &onnxpb.ValueInfoProto{
		Name: name,
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: int32(dt),
					Shape:    shape,
				},
			},
		},
	}
}

func TestImport_DimParamUnification(t *testing.T) {
	// Two inputs that both use dim_param "seq" for axis 0. The lowered
	// ValueInfos must report SymInt-backed shapes whose Variable index is
	// identical (interning via FindDefineVar).
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name: "sym",
			Input: []*onnxpb.ValueInfoProto{
				makeSymbolicValueInfo("a", onnxpb.TensorProto_FLOAT, []any{"seq", int64(4)}),
				makeSymbolicValueInfo("b", onnxpb.TensorProto_FLOAT, []any{"seq", int64(8)}),
			},
			Output: []*onnxpb.ValueInfoProto{
				makeSymbolicValueInfo("c", onnxpb.TensorProto_FLOAT, []any{"seq", int64(4)}),
			},
		},
	}
	arena := uop.NewArena(16)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	ins := r.Inputs()
	if len(ins) != 2 {
		t.Fatalf("Inputs count=%d, want 2", len(ins))
	}
	if len(ins[0].Shape) != 2 || len(ins[1].Shape) != 2 {
		t.Fatalf("shapes: a=%v b=%v", ins[0].Shape, ins[1].Shape)
	}

	// Axis-0 dims must be symbolic and unified.
	aSym, ok := ins[0].Shape[0].(interface{ ConstValue() (int64, bool) })
	if !ok {
		t.Fatalf("axis-0 of a is not a Sint")
	}
	if _, concrete := aSym.ConstValue(); concrete {
		t.Fatalf("axis-0 of a is concrete; want symbolic")
	}

	// Extract the underlying UOp index for each axis-0 dim via the SymInt
	// concrete type.
	getSymNode := func(s any) uop.UOp {
		// We expect ins[i].Shape[0] to be a shape.SymInt.
		// Use reflect via interface assertion through shape.Sint's contract.
		// (Avoid importing reflect.)
		// shape.SymInt has Node field of type uop.UOp.
		type symHaver interface{ ConstValue() (int64, bool) }
		_ = s.(symHaver)
		// Use a type switch over the canonical Sint sum-type.
		return symInt(s).Node
	}
	a0 := getSymNode(ins[0].Shape[0])
	b0 := getSymNode(ins[1].Shape[0])
	if a0.Index() != b0.Index() {
		t.Errorf("dim_param 'seq' did not unify: a.idx=%d b.idx=%d",
			a0.Index(), b0.Index())
	}

	// Axis-1 dims are concrete.
	if cv, ok := ins[0].Shape[1].ConstValue(); !ok || cv != 4 {
		t.Errorf("a axis-1=%v, want concrete 4", ins[0].Shape[1])
	}
	if cv, ok := ins[1].Shape[1].ConstValue(); !ok || cv != 8 {
		t.Errorf("b axis-1=%v, want concrete 8", ins[1].Shape[1])
	}
}

// symInt is a tiny helper that extracts a shape.SymInt's Node, avoiding the
// need to import the shape package's full sum-type into tests directly. We
// import shape in this file as needed.
//
// Implemented in runner_sym_helper_test.go.

func TestImport_DimEmpty_FailsClosed(t *testing.T) {
	// Rank-only dims (neither DimValue nor DimParam) must fail closed per
	// plan §1: an anonymous Variable per axis would silently disconnect
	// axes the model author meant to relate. Caller must name the axis
	// or set a concrete dim.
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name: "empty-dim",
			Input: []*onnxpb.ValueInfoProto{
				makeSymbolicValueInfo("x", onnxpb.TensorProto_FLOAT, []any{nil, int64(3)}),
			},
			Output: []*onnxpb.ValueInfoProto{
				makeValueInfo("dummy", onnxpb.TensorProto_FLOAT, []int64{1}),
			},
		},
	}
	arena := uop.NewArena(8)
	_, err := Import(mustMarshal(t, model), arena, "test")
	if err == nil {
		t.Fatalf("expected fail-closed on rank-only dim, got nil")
	}
	for _, want := range []string{"x", "rank-only"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q missing %q", err.Error(), want)
		}
	}
}

func TestImport_UnsupportedInputDtype(t *testing.T) {
	// A graph input with COMPLEX64 dtype must error during lowerValueInfos.
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name: "bad-input-dtype",
			Input: []*onnxpb.ValueInfoProto{
				makeValueInfo("x", onnxpb.TensorProto_COMPLEX64, []int64{1}),
			},
		},
	}
	arena := uop.NewArena(4)
	_, err := Import(mustMarshal(t, model), arena, "test")
	if err == nil {
		t.Fatalf("expected unsupported elem_type error, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported elem_type") {
		t.Errorf("error %q missing 'unsupported elem_type'", err.Error())
	}
}

// ── Initializer / user-input collision ────────────────────────────────────────

// TestRun_UserInputBeatsInitializer documents the observed contract: when an
// initializer and a user-provided input share a name, Run() seeds the user
// input *after* the initializers, so the user input wins.
func TestRun_UserInputBeatsInitializer(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "collide",
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("w", onnxpb.TensorProto_FLOAT, []int64{2})},
			Initializer: []*onnxpb.TensorProto{
				makeFloatTensor("w", []int64{2}, []float32{1, 1}),
			},
		},
	}
	arena := uop.NewArena(8)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	user := tensor.NewLeaf(arena, []int64{2}, uop.Dtypes.Float32, "test")
	user.SetData([]float32{77, 88})

	out, err := r.Run(map[string]*tensor.Tensor{"w": user})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := out["w"]
	if got == nil {
		t.Fatalf("output 'w' missing")
	}
	if got != user {
		t.Errorf("output 'w' is not the user-supplied tensor (initializer won)")
	}
	d := got.Data()
	if len(d) != 2 || d[0] != 77 || d[1] != 88 {
		t.Errorf("output payload=%v, want [77 88]", d)
	}
}

// ── Run: missing required input ───────────────────────────────────────────────

func TestRun_MissingRequiredInput(t *testing.T) {
	// Graph declares input "x" AND a node that consumes it. Caller supplies
	// no inputs → resolution fails with a descriptive error.
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "miss",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("y", onnxpb.TensorProto_FLOAT, []int64{1})},
			Node: []*onnxpb.NodeProto{
				{Name: "n1", OpType: "Identity", Input: []string{"x"}, Output: []string{"y"}},
			},
		},
	}
	arena := uop.NewArena(8)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	_, err = r.Run(nil)
	if err == nil {
		t.Fatalf("expected unresolved-input error, got nil")
	}
	if !strings.Contains(err.Error(), `"x"`) {
		t.Errorf("error %q missing input name 'x'", err.Error())
	}
	if !strings.Contains(err.Error(), "unresolved input") {
		t.Errorf("error %q missing 'unresolved input'", err.Error())
	}
}

// ── Run: graph output never assigned ──────────────────────────────────────────

func TestRun_OutputNeverAssigned(t *testing.T) {
	// Graph declares output "z" but no node produces it.
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "noprod",
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("z", onnxpb.TensorProto_FLOAT, []int64{1})},
		},
	}
	arena := uop.NewArena(4)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	_, err = r.Run(nil)
	if err == nil {
		t.Fatalf("expected output-missing error, got nil")
	}
	if !strings.Contains(err.Error(), "never assigned") {
		t.Errorf("error %q missing 'never assigned'", err.Error())
	}
}

// ── RegisterHandler + dispatch ordering ───────────────────────────────────────

// TestRegisterHandler_CustomDispatch registers a custom handler and verifies
// the Runner invokes it with the right inputs and propagates the output.
func TestRegisterHandler_CustomDispatch(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "custom",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("y", onnxpb.TensorProto_FLOAT, []int64{1})},
			Node: []*onnxpb.NodeProto{
				{Name: "n1", OpType: "MyId", Input: []string{"x"}, Output: []string{"y"}},
			},
		},
	}
	arena := uop.NewArena(8)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	want := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
	want.SetData([]float32{99})

	called := false
	r.RegisterHandler("MyId", func(ctx *HandlerCtx) ([]Value, error) {
		called = true
		if ctx.Node.OpType != "MyId" {
			t.Errorf("handler saw OpType=%q, want MyId", ctx.Node.OpType)
		}
		if len(ctx.Inputs) != 1 {
			t.Errorf("handler saw %d inputs, want 1", len(ctx.Inputs))
		}
		return []Value{Device(want)}, nil
	})

	xt := newDummyTensor(arena)
	out, err := r.Run(map[string]*tensor.Tensor{"x": xt})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !called {
		t.Errorf("handler was not invoked")
	}
	if out["y"] != want {
		t.Errorf("output y is not the handler's tensor")
	}
}

func TestRegisterHandler_ErrorPropagates(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "errprop",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("y", onnxpb.TensorProto_FLOAT, []int64{1})},
			Node: []*onnxpb.NodeProto{
				{Name: "boom", OpType: "BoomOp", Input: []string{"x"}, Output: []string{"y"}},
			},
		},
	}
	arena := uop.NewArena(4)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	r.RegisterHandler("BoomOp", func(ctx *HandlerCtx) ([]Value, error) {
		return nil, errBoom
	})
	_, err = r.Run(map[string]*tensor.Tensor{"x": newDummyTensor(arena)})
	if err == nil {
		t.Fatalf("expected propagated handler error, got nil")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Errorf("error %q missing 'boom'", err.Error())
	}
	if !strings.Contains(err.Error(), "BoomOp") {
		t.Errorf("error %q missing op name 'BoomOp'", err.Error())
	}
}

func TestRegisterHandler_TooManyOutputs(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "extra",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("y", onnxpb.TensorProto_FLOAT, []int64{1})},
			Node: []*onnxpb.NodeProto{
				{Name: "n1", OpType: "ExtraOut", Input: []string{"x"}, Output: []string{"y"}},
			},
		},
	}
	arena := uop.NewArena(4)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	r.RegisterHandler("ExtraOut", func(ctx *HandlerCtx) ([]Value, error) {
		dummy := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
		dummy.SetData([]float32{0})
		return []Value{Device(dummy), Device(dummy)}, nil // 2 outs vs 1 declared
	})
	_, err = r.Run(map[string]*tensor.Tensor{"x": newDummyTensor(arena)})
	if err == nil {
		t.Fatalf("expected too-many-outputs error, got nil")
	}
	if !strings.Contains(err.Error(), "produced 2 outputs") {
		t.Errorf("error %q missing 'produced 2 outputs'", err.Error())
	}
}

// TestRegisterHandler_TopologicalOrder builds a 2-node graph where node 2
// consumes node 1's output, and asserts the dispatcher walks them in order.
func TestRegisterHandler_TopologicalOrder(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name:   "topo",
			Input:  []*onnxpb.ValueInfoProto{makeValueInfo("x", onnxpb.TensorProto_FLOAT, []int64{1})},
			Output: []*onnxpb.ValueInfoProto{makeValueInfo("z", onnxpb.TensorProto_FLOAT, []int64{1})},
			Node: []*onnxpb.NodeProto{
				{Name: "n1", OpType: "First", Input: []string{"x"}, Output: []string{"y"}},
				{Name: "n2", OpType: "Second", Input: []string{"y"}, Output: []string{"z"}},
			},
		},
	}
	arena := uop.NewArena(8)
	r, err := Import(mustMarshal(t, model), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}

	var seq []string
	mk := func(name string) Handler {
		return func(ctx *HandlerCtx) ([]Value, error) {
			seq = append(seq, name)
			out := tensor.NewLeaf(ctx.Arena, []int64{1}, uop.Dtypes.Float32, ctx.Device)
			out.SetData([]float32{float32(len(seq))})
			return []Value{Device(out)}, nil
		}
	}
	r.RegisterHandler("First", mk("First"))
	r.RegisterHandler("Second", mk("Second"))

	if _, err := r.Run(map[string]*tensor.Tensor{"x": newDummyTensor(arena)}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(seq) != 2 || seq[0] != "First" || seq[1] != "Second" {
		t.Errorf("dispatch order=%v, want [First Second]", seq)
	}
}

// ── Initializer error propagates out of Import ────────────────────────────────

func TestImport_BadInitializerPropagates(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph: &onnxpb.GraphProto{
			Name: "bad-init",
			Initializer: []*onnxpb.TensorProto{
				{
					Name:     "bad",
					Dims:     []int64{1},
					DataType: int32(onnxpb.TensorProto_STRING),
				},
			},
		},
	}
	arena := uop.NewArena(4)
	_, err := Import(mustMarshal(t, model), arena, "test")
	if err == nil {
		t.Fatalf("expected unsupported-dtype init error, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported dtype") {
		t.Errorf("error %q missing 'unsupported dtype'", err.Error())
	}
}

// ── A small sentinel error used by the handler propagation test ───────────────

type boomErr struct{}

func (boomErr) Error() string { return "boom" }

var errBoom = boomErr{}
