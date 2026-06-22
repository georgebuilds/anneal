// Tests for viz.BuildInspect - the WASM-buildable path that drives the W9
// tensor-inspect dropzone. These tests run on native (no build tag); the
// same call applies in the WASM environment via the annealInspectTensor
// bridge. Fixtures live under tensor/npy/testdata + tensor/safetensors/testdata
// so the inspector and the existing arena-backed loaders share inputs.
//
// The inspect view never touches a real Arena, never opens a GPU device, and
// never reaches the server (bytes stay in the tab). The four tests below pin
// each format's shape / dtype / numel / preview contract and one malformed
// input path.

package viz

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

const (
	npyTestdata = "../tensor/npy/testdata"
	stTestdata  = "../tensor/safetensors/testdata"
)

// readFile is a tiny test helper that fails the test on read error.
func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %q: %v", path, err)
	}
	return b
}

// TestInspect_NPY_F32 reads the float32_3x4 fixture and pins shape, dtype,
// numel, and the leading preview values. The fixture is a 3×4 f32 array
// filled with the index sequence 0..11.
func TestInspect_NPY_F32(t *testing.T) {
	data := readFile(t, npyTestdata+"/float32_3x4.npy")
	r := BuildInspect("npy", data)
	if r.Error != "" {
		t.Fatalf("BuildInspect npy: unexpected error: %s", r.Error)
	}
	if r.Format != "npy" {
		t.Errorf("Format = %q, want npy", r.Format)
	}
	if len(r.Tensors) != 1 {
		t.Fatalf("len(Tensors) = %d, want 1", len(r.Tensors))
	}
	tn := r.Tensors[0]
	wantShape := []int64{3, 4}
	if len(tn.Shape) != 2 || tn.Shape[0] != wantShape[0] || tn.Shape[1] != wantShape[1] {
		t.Errorf("Shape = %v, want %v", tn.Shape, wantShape)
	}
	// Original numpy dtype is "<f4" (little-endian float32, 4 bytes).
	if tn.DType != "<f4" {
		t.Errorf("DType = %q, want <f4", tn.DType)
	}
	if tn.Numel != 12 {
		t.Errorf("Numel = %d, want 12", tn.Numel)
	}
	if tn.Bytes != 12*4 {
		t.Errorf("Bytes = %d, want 48", tn.Bytes)
	}
	if len(tn.Preview) != 12 {
		t.Fatalf("Preview len = %d, want 12 (tensor smaller than preview window)", len(tn.Preview))
	}
	for i := 0; i < 12; i++ {
		if tn.Preview[i] != float32(i) {
			t.Errorf("Preview[%d] = %v, want %v", i, tn.Preview[i], float32(i))
		}
	}
	// Round-trip through JSON so the WASM contract is exercised end-to-end.
	b, err := r.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	var back InspectResult
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("JSON roundtrip: %v", err)
	}
	if back.Tensors[0].Numel != 12 {
		t.Errorf("JSON roundtrip Numel = %d, want 12", back.Tensors[0].Numel)
	}
}

// TestInspect_NPZ_Multiple opens the multi.npz fixture (3 arrays of
// different dtypes) and verifies all of them surface with the right shape
// + dtype + numel.
func TestInspect_NPZ_Multiple(t *testing.T) {
	data := readFile(t, npyTestdata+"/multi.npz")
	r := BuildInspect("npz", data)
	if r.Error != "" {
		t.Fatalf("BuildInspect npz: unexpected error: %s", r.Error)
	}
	if r.Format != "npz" {
		t.Errorf("Format = %q, want npz", r.Format)
	}
	if len(r.Tensors) < 2 {
		t.Fatalf("len(Tensors) = %d, want >= 2 (multi.npz holds multiple arrays)", len(r.Tensors))
	}
	// Every tensor must populate the basic structural fields; the exact
	// content depends on the gen_fixtures.py script but the contract is
	// the same regardless of payload.
	for _, tn := range r.Tensors {
		if tn.Name == "" {
			t.Errorf("tensor with empty name: %+v", tn)
		}
		if tn.DType == "" {
			t.Errorf("tensor %q: empty DType", tn.Name)
		}
		if tn.Numel <= 0 {
			t.Errorf("tensor %q: Numel = %d, want > 0", tn.Name, tn.Numel)
		}
		if len(tn.Preview) > previewLen {
			t.Errorf("tensor %q: Preview length %d exceeds cap %d", tn.Name, len(tn.Preview), previewLen)
		}
	}
}

// TestInspect_Safetensors pins the contract over the multi_dtype fixture.
// Field-level dtype names are F32, F64, … (the safetensors capitalisation)
// not the npy descriptor strings.
func TestInspect_Safetensors(t *testing.T) {
	data := readFile(t, stTestdata+"/multi_dtype.safetensors")
	r := BuildInspect("safetensors", data)
	if r.Error != "" {
		t.Fatalf("BuildInspect safetensors: unexpected error: %s", r.Error)
	}
	if r.Format != "safetensors" {
		t.Errorf("Format = %q, want safetensors", r.Format)
	}
	if len(r.Tensors) == 0 {
		t.Fatalf("len(Tensors) = 0, want at least one (multi_dtype.safetensors)")
	}
	for _, tn := range r.Tensors {
		if tn.Name == "" {
			t.Errorf("tensor with empty name: %+v", tn)
		}
		// safetensors dtype labels are upper-case ASCII; verify the
		// inspector did not over-zealously rewrite them.
		if strings.ToUpper(tn.DType) != tn.DType {
			t.Errorf("tensor %q: DType %q not upper-case (safetensors convention)", tn.Name, tn.DType)
		}
		if tn.Numel <= 0 {
			t.Errorf("tensor %q: Numel = %d, want > 0", tn.Name, tn.Numel)
		}
	}
}

// TestInspect_MalformedNPY pins the error path: an invalid magic prefix
// must NOT panic; the InspectResult surfaces the parser error verbatim.
func TestInspect_MalformedNPY(t *testing.T) {
	r := BuildInspect("npy", []byte("not a numpy file"))
	if r.Error == "" {
		t.Fatal("BuildInspect malformed npy: expected non-empty Error")
	}
	if r.Format != "npy" {
		t.Errorf("Format = %q, want npy (format is echoed even on parse failure)", r.Format)
	}
	if len(r.Tensors) != 0 {
		t.Errorf("Tensors should be empty on parse failure, got %d", len(r.Tensors))
	}
	if !strings.Contains(strings.ToLower(r.Error), "numpy") && !strings.Contains(strings.ToLower(r.Error), "magic") {
		t.Errorf("Error %q does not look like the npy magic-byte failure message", r.Error)
	}
}

// TestInspect_UnknownFormat pins that unknown format strings get a clear
// blameless error and do not panic.
func TestInspect_UnknownFormat(t *testing.T) {
	r := BuildInspect("zip", []byte("anything"))
	if r.Error == "" {
		t.Fatal("BuildInspect unknown format: expected non-empty Error")
	}
	if !strings.Contains(r.Error, "unknown format") {
		t.Errorf("Error %q does not mention 'unknown format'", r.Error)
	}
}

// TestInspect_PreviewCap pins the 16-element preview window even when the
// payload is larger. The float32_2x3x4 fixture has 24 elements.
func TestInspect_PreviewCap(t *testing.T) {
	data := readFile(t, npyTestdata+"/float32_2x3x4.npy")
	r := BuildInspect("npy", data)
	if r.Error != "" {
		t.Fatalf("BuildInspect: %s", r.Error)
	}
	if len(r.Tensors) != 1 {
		t.Fatalf("len(Tensors) = %d, want 1", len(r.Tensors))
	}
	if got := len(r.Tensors[0].Preview); got != previewLen {
		t.Errorf("Preview len = %d, want %d (preview must cap at previewLen)", got, previewLen)
	}
	if r.Tensors[0].Numel != 24 {
		t.Errorf("Numel = %d, want 24", r.Tensors[0].Numel)
	}
}
