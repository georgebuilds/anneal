package viz

// Web W8: tests for BuildImportSummary (the ONNX dropzone backend). The
// ResNet-9 fixture path is shared with the onnx package's own conformance
// tests; we resolve it relative to this test file via go's testdata
// convention adjusted for the sibling package.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
)

// resnet9TestdataPath returns the absolute path to the committed ResNet-9
// fixture under onnx/testdata/. The viz tests live in viz/; the fixture lives
// in onnx/testdata/resnet9.onnx — one directory up + into onnx/testdata.
func resnet9TestdataPath(t *testing.T) string {
	t.Helper()
	// Tests run with cwd = viz/.
	abs, err := filepath.Abs(filepath.Join("..", "onnx", "testdata", "resnet9.onnx"))
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	return abs
}

// TestImportONNX_ResNet9_StructureOnly drives BuildImportSummary against the
// committed ResNet-9 fixture and pins the JSON shape: graph_id present,
// node_count > 0, initializer_count > 0, inputs/outputs populated, no
// unsupported ops (ResNet-9 is fully supported by the v1 importer), and
// the "structure only" note.
func TestImportONNX_ResNet9_StructureOnly(t *testing.T) {
	path := resnet9TestdataPath(t)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("resnet9.onnx fixture missing: %v", err)
	}
	bytesM, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	s, err := BuildImportSummary(bytesM)
	if err != nil {
		t.Fatalf("BuildImportSummary: %v", err)
	}

	if s.GraphID == "" {
		t.Error("GraphID is empty")
	}
	if !startsWith(s.GraphID, "imported-") {
		t.Errorf("GraphID %q does not start with 'imported-'", s.GraphID)
	}
	if s.NodeCount <= 0 {
		t.Errorf("NodeCount=%d, want > 0", s.NodeCount)
	}
	if s.InitializerCount <= 0 {
		t.Errorf("InitializerCount=%d, want > 0", s.InitializerCount)
	}
	if len(s.Inputs) == 0 {
		t.Error("Inputs empty; want at least 1")
	}
	if len(s.Outputs) == 0 {
		t.Error("Outputs empty; want at least 1")
	}
	if len(s.UnsupportedOps) != 0 {
		t.Errorf("UnsupportedOps non-empty for ResNet-9: %v", s.UnsupportedOps)
	}
	if s.Note == "" {
		t.Error("Note empty; expected 'structure only' note for dropzone use")
	}
	if !contains(s.Note, "structure only") {
		t.Errorf("Note %q missing 'structure only'", s.Note)
	}
	if s.Opset <= 0 {
		t.Errorf("Opset=%d, want > 0", s.Opset)
	}
	if len(s.Graph) == 0 {
		t.Error("Graph JSON is empty; expected at least the topology stub")
	}

	// Round-trip JSON shape: the summary must marshal cleanly and decode back
	// to a map with the contract keys.
	b, err := s.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	for _, k := range []string{"graph_id", "graph", "inputs", "outputs", "node_count", "initializer_count", "opset", "note"} {
		if _, ok := decoded[k]; !ok {
			t.Errorf("decoded summary missing key %q", k)
		}
	}
}

// TestImportONNX_GraphIDDeterminism pins that the same model bytes always
// produce the same graph_id. The studio relies on this to wire deep links
// from sessionStorage.
func TestImportONNX_GraphIDDeterminism(t *testing.T) {
	path := resnet9TestdataPath(t)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("resnet9.onnx fixture missing: %v", err)
	}
	bytesM, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	s1, err := BuildImportSummary(bytesM)
	if err != nil {
		t.Fatalf("BuildImportSummary #1: %v", err)
	}
	s2, err := BuildImportSummary(bytesM)
	if err != nil {
		t.Fatalf("BuildImportSummary #2: %v", err)
	}
	if s1.GraphID != s2.GraphID {
		t.Errorf("graph_id not deterministic: %q vs %q", s1.GraphID, s2.GraphID)
	}
	if s1.NodeCount != s2.NodeCount {
		t.Errorf("node_count not deterministic: %d vs %d", s1.NodeCount, s2.NodeCount)
	}
}

// TestImportONNX_WithUnsupportedOp constructs a minimal model whose single
// node is an unsupported op (Resize) and asserts the summary surfaces it in
// UnsupportedOps with the documented reason. The studio uses this list to
// render the dropzone's warning section.
func TestImportONNX_WithUnsupportedOp(t *testing.T) {
	model := &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: &onnxpb.GraphProto{
			Name:   "with_resize",
			Input:  []*onnxpb.ValueInfoProto{makeVI("x", onnxpb.TensorProto_FLOAT, []int64{1, 3, 4, 4})},
			Output: []*onnxpb.ValueInfoProto{makeVI("y", onnxpb.TensorProto_FLOAT, []int64{1, 3, 8, 8})},
			Node: []*onnxpb.NodeProto{
				{
					Name:   "resize0",
					OpType: "Resize",
					Input:  []string{"x"},
					Output: []string{"y"},
				},
			},
		},
	}
	modelBytes, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}

	s, err := BuildImportSummary(modelBytes)
	if err != nil {
		t.Fatalf("BuildImportSummary: %v", err)
	}
	if len(s.UnsupportedOps) != 1 {
		t.Fatalf("UnsupportedOps len=%d, want 1; got %+v", len(s.UnsupportedOps), s.UnsupportedOps)
	}
	got := s.UnsupportedOps[0]
	if got.OpType != "Resize" {
		t.Errorf("UnsupportedOps[0].OpType=%q, want %q", got.OpType, "Resize")
	}
	if got.Count != 1 {
		t.Errorf("UnsupportedOps[0].Count=%d, want 1", got.Count)
	}
	if !contains(got.Reason, "Resize") || !contains(got.Reason, "v1.1") {
		t.Errorf("UnsupportedOps[0].Reason=%q does not mention Resize / v1.1", got.Reason)
	}
}

// TestImportONNX_EmptyBytes pins the friendly error path.
func TestImportONNX_EmptyBytes(t *testing.T) {
	_, err := BuildImportSummary(nil)
	if err == nil {
		t.Fatalf("BuildImportSummary(nil) returned no error")
	}
}

// TestImportONNX_GraphPayloadShape pins the embedded graph JSON has the
// {nodes, edges, stats} contract the studio's visualize renderer consumes.
func TestImportONNX_GraphPayloadShape(t *testing.T) {
	path := resnet9TestdataPath(t)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("resnet9.onnx fixture missing: %v", err)
	}
	bytesM, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	s, err := BuildImportSummary(bytesM)
	if err != nil {
		t.Fatalf("BuildImportSummary: %v", err)
	}
	var g map[string]any
	if err := json.Unmarshal(s.Graph, &g); err != nil {
		t.Fatalf("graph JSON unmarshal: %v", err)
	}
	for _, key := range []string{"name", "nodes", "edges", "stats"} {
		if _, ok := g[key]; !ok {
			t.Errorf("graph JSON missing key %q", key)
		}
	}
	nodes, _ := g["nodes"].([]any)
	if len(nodes) == 0 {
		t.Error("graph JSON has 0 nodes; ResNet-9 should produce many")
	}
}

// makeVI constructs a TypeProto-wrapped ValueInfo with a concrete dim shape.
// Mirrors the helper in onnx/handlers_helper_test.go so this test file is
// self-contained.
func makeVI(name string, dt onnxpb.TensorProto_DataType, dims []int64) *onnxpb.ValueInfoProto {
	sh := &onnxpb.TensorShapeProto{}
	for _, d := range dims {
		sh.Dim = append(sh.Dim, &onnxpb.TensorShapeProto_Dimension{
			Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: d},
		})
	}
	return &onnxpb.ValueInfoProto{
		Name: name,
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: int32(dt),
					Shape:    sh,
				},
			},
		},
	}
}

func startsWith(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	return s[:len(prefix)] == prefix
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
