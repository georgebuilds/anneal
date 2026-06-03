package npy_test

import (
	"os"
	"testing"

	"github.com/georgebuilds/anneal/tensor/npy"
)

// TestReadBytes_F32 pins the byte-based public API used by the WASM
// tensor-inspect view (W9). The dtype string is preserved verbatim ("<f4")
// so the inspector can show the original numpy descriptor.
func TestReadBytes_F32(t *testing.T) {
	data, err := os.ReadFile("testdata/float32_3x4.npy")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	e, err := npy.ReadBytes(data)
	if err != nil {
		t.Fatalf("ReadBytes: %v", err)
	}
	if e.DType != "<f4" {
		t.Errorf("DType = %q, want <f4", e.DType)
	}
	if len(e.Shape) != 2 || e.Shape[0] != 3 || e.Shape[1] != 4 {
		t.Errorf("Shape = %v, want [3 4]", e.Shape)
	}
	if len(e.Data) != 12 {
		t.Errorf("len(Data) = %d, want 12", len(e.Data))
	}
}

// TestReadZBytes_Multiple pins the byte-based NPZ reader. The fixture
// holds multiple arrays of different dtypes.
func TestReadZBytes_Multiple(t *testing.T) {
	data, err := os.ReadFile("testdata/multi.npz")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	names, entries, err := npy.ReadZBytes(data)
	if err != nil {
		t.Fatalf("ReadZBytes: %v", err)
	}
	if len(names) == 0 {
		t.Fatal("no entries returned from multi.npz")
	}
	if len(entries) != len(names) {
		t.Errorf("entries map size %d != names slice size %d", len(entries), len(names))
	}
	for _, n := range names {
		if _, ok := entries[n]; !ok {
			t.Errorf("entry %q in names but not in map", n)
		}
	}
}

// TestReadBytes_BadMagic pins the error path for the WASM inspector.
func TestReadBytes_BadMagic(t *testing.T) {
	_, err := npy.ReadBytes([]byte("garbage"))
	if err == nil {
		t.Fatal("ReadBytes on garbage: expected error")
	}
}
