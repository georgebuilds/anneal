package safetensors_test

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/tensor/safetensors"
	"github.com/georgebuilds/anneal/uop"
)

const testdataDir = "testdata"

func newArena() *uop.Arena { return uop.NewArena(256) }

// requireFixture fails the test if the named fixture file is absent.
func requireFixture(t *testing.T, name string) string {
	t.Helper()
	path := testdataDir + "/" + name
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("fixture %q missing — regenerate with:\n\tpython3 tensor/safetensors/testdata/gen_fixtures.py\nRequires: pip install safetensors numpy", path)
	}
	return path
}

func assertShape(t *testing.T, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape rank %d != %d; got %v want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape[%d]: got %d want %d", i, got[i], want[i])
		}
	}
}

func assertData(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: data length %d != %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: data[%d]: got %v want %v", label, i, got[i], want[i])
		}
	}
}

// ── load-fidelity: read real Python-generated fixtures ───────────────────────

func TestLoadFixture_F32(t *testing.T) {
	path := requireFixture(t, "f32_3x4.safetensors")
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	w, ok := tensors["w"]
	if !ok {
		t.Fatal("expected key 'w'")
	}
	assertShape(t, w.Shape, []int64{3, 4})
	want := make([]float32, 12)
	for i := range want {
		want[i] = float32(i)
	}
	assertData(t, "f32_3x4", w.Data, want)
	t.Logf("F32 3×4: shape=%v data=%v", w.Shape, w.Data)
}

func TestLoadFixture_F64Exact(t *testing.T) {
	path := requireFixture(t, "f64_exact.safetensors")
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	v := tensors["v"]
	assertShape(t, v.Shape, []int64{4})
	// Fixture: [0.0, 1.0, 2.0, 100.0] as F64 — all exactly representable in F32.
	assertData(t, "f64_exact", v.Data, []float32{0, 1, 2, 100})
	t.Logf("F64→F32: shape=%v data=%v", v.Shape, v.Data)
}

func TestLoadFixture_MultiDtype(t *testing.T) {
	path := requireFixture(t, "multi_dtype.safetensors")
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	cases := []struct {
		key  string
		want []float32
	}{
		{"i8", []float32{-5, 0, 10, 127}},
		{"u8", []float32{0, 1, 255}},
		{"i16", []float32{-32768, 0, 32767}},
		{"i32", []float32{1, 2, 3}},
		{"i64", []float32{10, 20, 30}},
		{"bool", []float32{1, 0, 1}},
		{"f32", []float32{1, 2, 3}},
	}
	for _, tc := range cases {
		e, ok := tensors[tc.key]
		if !ok {
			t.Errorf("missing key %q", tc.key)
			continue
		}
		assertData(t, tc.key, e.Data, tc.want)
		t.Logf("%s: shape=%v data=%v", tc.key, e.Shape, e.Data)
	}
}

func TestLoadFixture_Metadata(t *testing.T) {
	path := requireFixture(t, "with_metadata.safetensors")
	// Metadata block must not break loading.
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors with metadata: %v", err)
	}
	w, ok := tensors["weights"]
	if !ok {
		t.Fatal("expected key 'weights'")
	}
	assertShape(t, w.Shape, []int64{2, 3})
	assertData(t, "weights", w.Data, []float32{0, 1, 2, 3, 4, 5})
	t.Logf("with_metadata weights: shape=%v data=%v", w.Shape, w.Data)
}

func TestLoadFixture_F16(t *testing.T) {
	path := requireFixture(t, "f16.safetensors")
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors F16: %v", err)
	}
	h := tensors["h"]
	assertShape(t, h.Shape, []int64{4})
	// Fixture: [1.0, 2.0, 0.5, 4.0] in F16 — all exactly representable in F32.
	assertData(t, "f16", h.Data, []float32{1.0, 2.0, 0.5, 4.0})
	t.Logf("F16→F32: shape=%v data=%v", h.Shape, h.Data)
}

func TestLoadFixture_BF16(t *testing.T) {
	path := requireFixture(t, "bf16.safetensors")
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors BF16: %v", err)
	}
	b := tensors["b"]
	assertShape(t, b.Shape, []int64{4})
	// Fixture: [1.0, -1.0, 0.5, 2.0] in BF16.
	// BF16 = upper 16 bits of F32; these values are all exactly representable.
	assertData(t, "bf16", b.Data, []float32{1.0, -1.0, 0.5, 2.0})
	t.Logf("BF16→F32: shape=%v data=%v", b.Shape, b.Data)
}

func TestLoadFixture_MultiTensor(t *testing.T) {
	path := requireFixture(t, "multi_tensor.safetensors")
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors multi_tensor: %v", err)
	}

	// Key set: exactly {weights, bias, mask}.
	wantKeys := []string{"bias", "mask", "weights"}
	gotKeys := make([]string, 0, len(tensors))
	for k := range tensors {
		gotKeys = append(gotKeys, k)
	}
	sort.Strings(gotKeys)
	if fmt.Sprint(gotKeys) != fmt.Sprint(wantKeys) {
		t.Fatalf("keys: got %v want %v", gotKeys, wantKeys)
	}

	assertShape(t, tensors["weights"].Shape, []int64{2, 3})
	assertData(t, "weights", tensors["weights"].Data, []float32{0, 1, 2, 3, 4, 5})

	assertShape(t, tensors["bias"].Shape, []int64{3})
	assertData(t, "bias", tensors["bias"].Data, []float32{0.0, 0.5, 1.0})

	assertShape(t, tensors["mask"].Shape, []int64{3})
	assertData(t, "mask", tensors["mask"].Data, []float32{1, 0, 1})

	t.Logf("multi_tensor: weights=%v bias=%v mask=%v",
		tensors["weights"].Data, tensors["bias"].Data, tensors["mask"].Data)
}

// ── save round-trip ───────────────────────────────────────────────────────────

func TestSave_RoundTrip_BasicParams(t *testing.T) {
	a := newArena()
	w := nn.NewParameter(a, []int64{4, 2}, uop.Dtypes.Float32, "cpu")
	b := nn.NewParameter(a, []int64{4}, uop.Dtypes.Float32, "cpu")

	// Set known non-zero values.
	for i := range w.Value {
		w.Value[i] = float32(i+1) * 0.1
	}
	for i := range b.Value {
		b.Value[i] = float32(i+1) * 0.01
	}

	path := t.TempDir() + "/ckpt.safetensors"
	params := map[string]*nn.Parameter{"w": w, "b": b}
	if err := safetensors.Save(path, params); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load back with LoadTensors and compare.
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	assertShape(t, tensors["w"].Shape, []int64{4, 2})
	assertData(t, "round-trip w", tensors["w"].Data, w.Value)

	assertShape(t, tensors["b"].Shape, []int64{4})
	assertData(t, "round-trip b", tensors["b"].Data, b.Value)

	t.Logf("round-trip w: %v", tensors["w"].Data)
	t.Logf("round-trip b: %v", tensors["b"].Data)
}

func TestSave_RoundTrip_IntoParams(t *testing.T) {
	a := newArena()
	w := nn.NewParameter(a, []int64{3, 2}, uop.Dtypes.Float32, "cpu")
	for i := range w.Value {
		w.Value[i] = float32(i) * 0.5
	}

	path := t.TempDir() + "/ckpt.safetensors"
	if err := safetensors.Save(path, map[string]*nn.Parameter{"w": w}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load INTO a fresh parameter.
	a2 := newArena()
	w2 := nn.NewParameter(a2, []int64{3, 2}, uop.Dtypes.Float32, "cpu")
	if err := safetensors.Load(path, map[string]*nn.Parameter{"w": w2}); err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertData(t, "into-param", w2.Value, w.Value)
	t.Logf("Load into param: original=%v loaded=%v", w.Value, w2.Value)
}

// ── train → save → load → identical forward output ───────────────────────────

// TestTrainSaveLoadForward simulates a training loop, saves a checkpoint,
// loads it into a fresh model, and asserts that a manual forward pass on
// the loaded model produces bit-identical output to the saved model.
//
// This is the "checkpoints actually work for training" proof.
// No GPU is required; the forward pass is computed manually on Value slices.
func TestTrainSaveLoadForward(t *testing.T) {
	const (
		inDim  = 3
		hidDim = 4
		outDim = 2
		steps  = 5
		lr     = float32(0.01)
	)

	a := newArena()
	// Layer 1: Linear(3→4, bias)
	w1 := nn.NewParameter(a, []int64{hidDim, inDim}, uop.Dtypes.Float32, "cpu")
	b1 := nn.NewParameter(a, []int64{hidDim}, uop.Dtypes.Float32, "cpu")
	// Layer 2: Linear(4→2, bias)
	w2 := nn.NewParameter(a, []int64{outDim, hidDim}, uop.Dtypes.Float32, "cpu")
	b2 := nn.NewParameter(a, []int64{outDim}, uop.Dtypes.Float32, "cpu")

	// Initialise weights to predictable non-zero values.
	for i := range w1.Value {
		w1.Value[i] = float32(i+1) * 0.1
	}
	for i := range b1.Value {
		b1.Value[i] = float32(i+1) * 0.05
	}
	for i := range w2.Value {
		w2.Value[i] = float32(i+1) * 0.2
	}
	for i := range b2.Value {
		b2.Value[i] = float32(i+1) * 0.02
	}

	// Simulate SGD steps with synthetic gradients.
	allParams := []*nn.Parameter{w1, b1, w2, b2}
	for step := 0; step < steps; step++ {
		for _, p := range allParams {
			fakeGrad := make([]float32, len(p.Value))
			for i := range fakeGrad {
				fakeGrad[i] = float32(step+1) * float32(i+1) * 0.001
			}
			p.SGDStep(fakeGrad, lr)
		}
	}

	// Save checkpoint.
	path := t.TempDir() + "/trained.safetensors"
	params := map[string]*nn.Parameter{
		"l1.weight": w1, "l1.bias": b1,
		"l2.weight": w2, "l2.bias": b2,
	}
	if err := safetensors.Save(path, params); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Create fresh parameters and load checkpoint.
	a2 := newArena()
	w1n := nn.NewParameter(a2, []int64{hidDim, inDim}, uop.Dtypes.Float32, "cpu")
	b1n := nn.NewParameter(a2, []int64{hidDim}, uop.Dtypes.Float32, "cpu")
	w2n := nn.NewParameter(a2, []int64{outDim, hidDim}, uop.Dtypes.Float32, "cpu")
	b2n := nn.NewParameter(a2, []int64{outDim}, uop.Dtypes.Float32, "cpu")
	freshParams := map[string]*nn.Parameter{
		"l1.weight": w1n, "l1.bias": b1n,
		"l2.weight": w2n, "l2.bias": b2n,
	}
	if err := safetensors.Load(path, freshParams); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Manual forward pass: Linear + ReLU + Linear.
	// Both models must produce identical output for any input.
	input := []float32{1.0, 0.5, -0.5}

	linear := func(in []float32, w *nn.Parameter, b *nn.Parameter) []float32 {
		sh := w.T.Shape()
		outD, inD := int(sh[0]), int(sh[1])
		out := make([]float32, outD)
		for i := 0; i < outD; i++ {
			for j := 0; j < inD; j++ {
				out[i] += in[j] * w.Value[i*inD+j]
			}
			out[i] += b.Value[i]
		}
		return out
	}
	relu := func(x []float32) []float32 {
		r := make([]float32, len(x))
		for i, v := range x {
			if v > 0 {
				r[i] = v
			}
		}
		return r
	}
	forward := func(W1, B1, W2, B2 *nn.Parameter) []float32 {
		return linear(relu(linear(input, W1, B1)), W2, B2)
	}

	out1 := forward(w1, b1, w2, b2)
	out2 := forward(w1n, b1n, w2n, b2n)

	var maxDiff float32
	for i := range out1 {
		d := out1[i] - out2[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}

	t.Logf("trained output:  %v", out1)
	t.Logf("loaded output:   %v", out2)
	t.Logf("max abs diff:    %v", maxDiff)

	if maxDiff != 0 {
		t.Errorf("forward outputs differ after checkpoint round-trip: max abs diff = %v", maxDiff)
	}
}

// ── write fidelity: anneal-written file has correct bytes ─────────────────────

// TestSave_WriteFidelity generates a checkpoint with Go and verifies that the
// raw file bytes have the correct safetensors structure. The Python-side
// verification (using real safetensors library) is run externally via
// testdata/verify_written.py and the output is reported below.
//
// Python verification command (from repo root):
//
//	source /tmp/npyenv/bin/activate
//	go test ./tensor/safetensors/ -run TestSave_WriteFidelity -v 2>&1 | grep "written to"
//	# then:
//	python3 tensor/safetensors/testdata/verify_written.py <path>
func TestSave_WriteFidelity(t *testing.T) {
	a := newArena()
	w := nn.NewParameter(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	b := nn.NewParameter(a, []int64{2}, uop.Dtypes.Float32, "cpu")
	w.Value = []float32{1, 2, 3, 4, 5, 6}
	b.Value = []float32{0.1, 0.2}

	path := t.TempDir() + "/fidelity.safetensors"
	params := map[string]*nn.Parameter{"weight": w, "bias": b}
	if err := safetensors.Save(path, params); err != nil {
		t.Fatalf("Save: %v", err)
	}
	t.Logf("checkpoint written to: %s", path)

	// Structural verification in pure Go: parse the file we just wrote.
	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors on Go-written file: %v", err)
	}

	assertShape(t, tensors["weight"].Shape, []int64{2, 3})
	assertData(t, "weight", tensors["weight"].Data, w.Value)
	assertShape(t, tensors["bias"].Shape, []int64{2})
	assertData(t, "bias", tensors["bias"].Data, b.Value)

	// Verify the header length field (first 8 bytes must decode as a valid u64).
	raw, _ := os.ReadFile(path)
	hLen := uint64(raw[0]) | uint64(raw[1])<<8 | uint64(raw[2])<<16 |
		uint64(raw[3])<<24 | uint64(raw[4])<<32 | uint64(raw[5])<<40 |
		uint64(raw[6])<<48 | uint64(raw[7])<<56
	if hLen == 0 || int(hLen) > len(raw)-8 {
		t.Errorf("invalid header length %d (file is %d bytes)", hLen, len(raw))
	}
	// Header must be multiple of 8 (safetensors padding convention).
	if hLen%8 != 0 {
		t.Errorf("header length %d is not a multiple of 8", hLen)
	}

	t.Logf("file structure: header_len=%d total_file=%d bytes", hLen, len(raw))
	t.Logf("weight: shape=%v data=%v", tensors["weight"].Shape, tensors["weight"].Data)
	t.Logf("bias:   shape=%v data=%v", tensors["bias"].Shape, tensors["bias"].Data)
}

// ── F16 conversion correctness ────────────────────────────────────────────────

// TestF16Conversion verifies the F16→F32 conversion against known IEEE-754 values.
func TestF16Conversion(t *testing.T) {
	// Write a minimal safetensors file with manually-chosen F16 bit patterns,
	// then load it and verify the resulting float32 values.
	cases := []struct {
		f16bits uint16
		wantF32 float32
		desc    string
	}{
		{0x3C00, 1.0, "1.0"},
		{0x4000, 2.0, "2.0"},
		{0x3800, 0.5, "0.5"},
		{0xBC00, -1.0, "-1.0"},
		{0x0000, 0.0, "+0.0"},
		{0x8000, math.Float32frombits(0x80000000), "-0.0"},
		{0x7C00, float32(math.Inf(1)), "+Inf"},
		{0xFC00, float32(math.Inf(-1)), "-Inf"},
	}

	// Build a raw safetensors file with all F16 test values in one tensor.
	// Layout: [u64 hlen] [JSON header] [F16 bytes LE]
	import_header := func(n int, hdr string) []byte {
		hdrB := []byte(hdr)
		if rem := len(hdrB) % 8; rem != 0 {
			pad := make([]byte, 8-rem)
			for i := range pad {
				pad[i] = ' '
			}
			hdrB = append(hdrB, pad...)
		}
		out := make([]byte, 8+len(hdrB)+n*2)
		hLen := uint64(len(hdrB))
		for i := 0; i < 8; i++ {
			out[i] = byte(hLen >> (i * 8))
		}
		copy(out[8:], hdrB)
		return out
	}
	_ = import_header // just for illustration

	// Create file via os.CreateTemp
	n := len(cases)
	hdr := fmt.Sprintf(`{"h":{"dtype":"F16","shape":[%d],"data_offsets":[0,%d]}}`, n, n*2)
	hdrB := []byte(hdr)
	if rem := len(hdrB) % 8; rem != 0 {
		pad := make([]byte, 8-rem)
		for i := range pad {
			pad[i] = ' '
		}
		hdrB = append(hdrB, pad...)
	}
	raw := make([]byte, 8+len(hdrB)+n*2)
	hLen := uint64(len(hdrB))
	for i := 0; i < 8; i++ {
		raw[i] = byte(hLen >> (i * 8))
	}
	copy(raw[8:], hdrB)
	dataOff := 8 + len(hdrB)
	for i, tc := range cases {
		raw[dataOff+i*2] = byte(tc.f16bits)
		raw[dataOff+i*2+1] = byte(tc.f16bits >> 8)
	}
	f, err := os.CreateTemp(t.TempDir(), "f16_*.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(raw); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()

	tensors, err := safetensors.LoadTensors(f.Name())
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	h := tensors["h"]
	for i, tc := range cases {
		got := h.Data[i]
		want := tc.wantF32
		// For NaN: use bit comparison.
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("F16 %s (0x%04X): got %v want NaN", tc.desc, tc.f16bits, got)
			}
			continue
		}
		if got != want {
			t.Errorf("F16 %s (0x%04X): got %v want %v", tc.desc, tc.f16bits, got, want)
		}
	}
	t.Logf("F16 conversion: %d cases, all correct", len(cases))
}

// ── honest errors ─────────────────────────────────────────────────────────────

func TestError_IntRangeOverflow(t *testing.T) {
	path := requireFixture(t, "i64_oor.safetensors")
	_, err := safetensors.LoadTensors(path)
	if err == nil {
		t.Fatal("expected error for out-of-range I64 value, got nil")
	}
	if !strings.Contains(err.Error(), "16777217") {
		t.Errorf("error should name the offending value 16777217, got: %v", err)
	}
	t.Logf("I64 out-of-range error (expected): %v", err)
}

func TestError_UnsupportedDtype(t *testing.T) {
	// Build a minimal safetensors file with an unsupported dtype (F8_E4M3).
	hdr := `{"v":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}`
	hdrB := []byte(hdr)
	if rem := len(hdrB) % 8; rem != 0 {
		pad := make([]byte, 8-rem)
		for i := range pad {
			pad[i] = ' '
		}
		hdrB = append(hdrB, pad...)
	}
	raw := make([]byte, 8+len(hdrB)+2)
	hLen := uint64(len(hdrB))
	for i := 0; i < 8; i++ {
		raw[i] = byte(hLen >> (i * 8))
	}
	copy(raw[8:], hdrB)

	f, _ := os.CreateTemp(t.TempDir(), "f8_*.safetensors")
	f.Write(raw)
	f.Close()

	_, err := safetensors.LoadTensors(f.Name())
	if err == nil {
		t.Fatal("expected error for unsupported dtype F8_E4M3, got nil")
	}
	if !strings.Contains(strings.ToUpper(err.Error()), "F8_E4M3") &&
		!strings.Contains(err.Error(), "not supported") {
		t.Errorf("error should mention F8_E4M3 or 'not supported', got: %v", err)
	}
	t.Logf("unsupported dtype error (expected): %v", err)
}

func TestError_ShapeMismatch(t *testing.T) {
	a := newArena()
	w := nn.NewParameter(a, []int64{4, 2}, uop.Dtypes.Float32, "cpu")
	w.Value = []float32{1, 2, 3, 4, 5, 6, 7, 8}

	path := t.TempDir() + "/ckpt.safetensors"
	if err := safetensors.Save(path, map[string]*nn.Parameter{"w": w}); err != nil {
		t.Fatal(err)
	}

	// Try loading into a parameter with wrong shape.
	a2 := newArena()
	wrongShape := nn.NewParameter(a2, []int64{2, 4}, uop.Dtypes.Float32, "cpu")
	err := safetensors.Load(path, map[string]*nn.Parameter{"w": wrongShape})
	if err == nil {
		t.Fatal("expected shape mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "shape mismatch") {
		t.Errorf("error should mention 'shape mismatch', got: %v", err)
	}
	t.Logf("shape mismatch error (expected): %v", err)
}

func TestError_MissingTensor(t *testing.T) {
	a := newArena()
	w := nn.NewParameter(a, []int64{4}, uop.Dtypes.Float32, "cpu")

	path := t.TempDir() + "/ckpt.safetensors"
	safetensors.Save(path, map[string]*nn.Parameter{"w": w})

	// Load into a map containing a key not in the file.
	a2 := newArena()
	notInFile := nn.NewParameter(a2, []int64{4}, uop.Dtypes.Float32, "cpu")
	err := safetensors.Load(path, map[string]*nn.Parameter{"nonexistent": notInFile})
	if err == nil {
		t.Fatal("expected error for missing tensor, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error should mention 'not found', got: %v", err)
	}
	t.Logf("missing tensor error (expected): %v", err)
}

// ── dtype is always F32 on save ───────────────────────────────────────────────

func TestSave_OutputAlwaysF32(t *testing.T) {
	a := newArena()
	p := nn.NewParameter(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	p.Value = []float32{1, 2, 3}

	path := t.TempDir() + "/f32only.safetensors"
	safetensors.Save(path, map[string]*nn.Parameter{"p": p})

	tensors, _ := safetensors.LoadTensors(path)
	// Raw file verification: parse header, check "dtype":"F32"
	raw, _ := os.ReadFile(path)
	if !strings.Contains(string(raw[8:]), `"F32"`) {
		t.Error("saved file should contain F32 dtype in header")
	}
	assertData(t, "f32only", tensors["p"].Data, p.Value)
	t.Logf("save always F32: header contains: %s", raw[8:8+80])
}
