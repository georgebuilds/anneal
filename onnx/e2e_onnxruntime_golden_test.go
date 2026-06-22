package onnx

// Phase 2 Strategy B: cross-check the importer against an onnxruntime-
// produced golden output. The fixtures are committed in onnx/testdata/
// (resnet9.onnx, resnet9_input.npy, resnet9_output.npy). They are generated
// by notes/scripts/gen_resnet9_golden.py at dev time using a Python venv
// with numpy + onnx + onnxruntime; CI does not regenerate them.
//
// This is a *secondary* gate. The primary CI gate is Strategy A bit-exact
// (e2e_cnn_test.go, e2e_arch_test.go). Strategy B catches the class of bug
// where anneal and the importer agree but both diverge from the spec.

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/npy"
	"github.com/georgebuilds/anneal/uop"
)

// onnxruntimeGoldenDir returns the absolute path of onnx/testdata/ from this
// test file's location. Relative go-test cwd is the package directory.
func onnxruntimeGoldenDir(t *testing.T) string {
	t.Helper()
	// Tests run with cwd = onnx/. Goldens live in ./testdata/.
	abs, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	return abs
}

// TestE2E_ResNet9_OnnxRuntimeGolden loads the committed ResNet-9 ONNX model,
// feeds the golden input, runs through the importer, and asserts the result
// matches the golden output within 1e-3 max-abs-diff (the v1 conformance
// tolerance per the plan §8 Phase 2 gate).
//
// Skips if the fixtures are absent (e.g. a development checkout without
// notes/scripts/ pulled), so the CI gate stays self-contained.
func TestE2E_ResNet9_OnnxRuntimeGolden(t *testing.T) {
	dir := onnxruntimeGoldenDir(t)
	modelPath := filepath.Join(dir, "resnet9.onnx")
	inputPath := filepath.Join(dir, "resnet9_input.npy")
	outputPath := filepath.Join(dir, "resnet9_output.npy")
	for _, p := range []string{modelPath, inputPath, outputPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("Strategy B golden fixture missing: %s (run notes/scripts/gen_resnet9_golden.py to regenerate)", p)
		}
	}

	arena := uop.NewArena(65536)
	r, err := ImportFile(modelPath, arena, "test")
	if err != nil {
		t.Fatalf("ImportFile(%s): %v", modelPath, err)
	}

	xT, err := npy.Load(arena, inputPath, "test")
	if err != nil {
		t.Fatalf("npy.Load(%s): %v", inputPath, err)
	}
	xShape := xT.Shape()
	want := []int64{1, 3, 32, 32}
	if len(xShape) != len(want) {
		t.Fatalf("input shape %v, want %v", xShape, want)
	}
	for i := range want {
		if xShape[i] != want[i] {
			t.Fatalf("input shape %v, want %v", xShape, want)
		}
	}

	// Find the graph input name.
	if len(r.Inputs()) != 1 {
		t.Fatalf("expected 1 graph input, got %d", len(r.Inputs()))
	}
	inName := r.Inputs()[0].Name

	out, err := r.Run(map[string]*tensor.Tensor{inName: xT})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(r.Outputs()) != 1 {
		t.Fatalf("expected 1 graph output, got %d", len(r.Outputs()))
	}
	outName := r.Outputs()[0].Name
	yT, ok := out[outName]
	if !ok {
		t.Fatalf("output %q missing", outName)
	}

	// Evaluate via cpuEval.
	got, gotShape, err := cpuEval(yT)
	if err != nil {
		t.Fatalf("cpuEval: %v", err)
	}

	// Load golden output.
	yGoldenT, err := npy.Load(arena, outputPath, "test")
	if err != nil {
		t.Fatalf("npy.Load(%s): %v", outputPath, err)
	}
	wantData := yGoldenT.Data()
	wantShape := yGoldenT.Shape()

	if !shapeEq(gotShape, wantShape) {
		t.Fatalf("output shape mismatch: got %v, want %v", gotShape, wantShape)
	}
	if len(got) != len(wantData) {
		t.Fatalf("output length mismatch got=%d want=%d", len(got), len(wantData))
	}

	const tol = float32(1e-3)
	m := maxAbsDiff(got, wantData)
	t.Logf("ResNet9_OnnxRuntimeGolden: max-abs-diff = %g (tol=%g, n=%d, shape=%v)",
		m, tol, len(got), gotShape)
	if m > tol {
		// Log the per-element comparison for the first 8 elements when
		// the assertion fails - this is the value-oracle discipline.
		n := 8
		if len(got) < n {
			n = len(got)
		}
		for i := 0; i < n; i++ {
			t.Logf("  [%d]: got=%v want=%v diff=%v", i, got[i], wantData[i], got[i]-wantData[i])
		}
		t.Errorf("max-abs-diff %g exceeds tol %g", m, tol)
	}
}
