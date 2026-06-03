package safetensors_test

import (
	"os"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/safetensors"
)

// TestReadBytes_F32 pins the byte-based public API used by the WASM
// tensor-inspect view (W9). DType is the raw safetensors label ("F32")
// preserved verbatim so the inspector can show the original dtype.
func TestReadBytes_F32(t *testing.T) {
	data, err := os.ReadFile("testdata/f32_3x4.safetensors")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	entries, err := safetensors.ReadBytes(data)
	if err != nil {
		t.Fatalf("ReadBytes: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("ReadBytes returned no entries")
	}
	e := entries[0]
	if e.DType != "F32" {
		t.Errorf("DType = %q, want F32", e.DType)
	}
	if len(e.Shape) != 2 || e.Shape[0] != 3 || e.Shape[1] != 4 {
		t.Errorf("Shape = %v, want [3 4]", e.Shape)
	}
	if len(e.Data) != 12 {
		t.Errorf("len(Data) = %d, want 12", len(e.Data))
	}
}

// TestReadBytes_MultiDtype pins multi-tensor decoding. Different dtypes
// surface with their raw labels preserved.
func TestReadBytes_MultiDtype(t *testing.T) {
	data, err := os.ReadFile("testdata/multi_dtype.safetensors")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	entries, err := safetensors.ReadBytes(data)
	if err != nil {
		t.Fatalf("ReadBytes: %v", err)
	}
	if len(entries) < 2 {
		t.Fatalf("len(entries) = %d, want >= 2", len(entries))
	}
	for _, e := range entries {
		if e.Name == "" {
			t.Errorf("entry with empty name: %+v", e)
		}
		if strings.ToUpper(e.DType) != e.DType {
			t.Errorf("dtype %q not uppercase (safetensors convention)", e.DType)
		}
	}
}

// TestReadBytes_TooShort pins the error path for the inspector.
func TestReadBytes_TooShort(t *testing.T) {
	_, err := safetensors.ReadBytes([]byte("x"))
	if err == nil {
		t.Fatal("ReadBytes on tiny input: expected error")
	}
}
