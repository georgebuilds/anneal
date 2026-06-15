package safetensors

import (
	"encoding/binary"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Save error arm: uncreatable path ────────────────────────────────────────

func TestSaveCreateError(t *testing.T) {
	a := uop.NewArena(64)
	p := nn.NewParameter(a, []int64{2}, uop.Dtypes.Float32, "cpu")
	p.Value = []float32{1, 2}
	// A path whose parent directory does not exist cannot be created.
	bad := filepath.Join(t.TempDir(), "nonexistent-dir", "x.safetensors")
	if err := Save(bad, map[string]*nn.Parameter{"w": p}); err == nil {
		t.Error("Save to an uncreatable path should error")
	}
}

// ── Load: error propagation from LoadTensors ────────────────────────────────

func TestLoadPropagatesLoadTensorsError(t *testing.T) {
	a := uop.NewArena(64)
	p := nn.NewParameter(a, []int64{1}, uop.Dtypes.Float32, "cpu")
	// Missing file: LoadTensors fails and Load must surface it.
	miss := filepath.Join(t.TempDir(), "missing.safetensors")
	if err := Load(miss, map[string]*nn.Parameter{"w": p}); err == nil {
		t.Error("Load on missing file should error")
	}
}

// ── LoadTensors: file-level read error arms ─────────────────────────────────

func TestLoadTensorsHeaderLengthShort(t *testing.T) {
	// File shorter than the 8-byte header-length field.
	p := writeTemp(t, []byte{0x01, 0x02, 0x03})
	if _, err := LoadTensors(p); err == nil {
		t.Error("file too short for header length should error")
	}
}

func TestLoadTensorsTruncatedHeader(t *testing.T) {
	// header length says 100 bytes but no header bytes follow → ReadFull fails.
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, 100)
	p := writeTemp(t, b)
	if _, err := LoadTensors(p); err == nil {
		t.Error("truncated header should error")
	}
}

func TestLoadTensorsBadEntryJSON(t *testing.T) {
	// Valid top-level JSON, but an entry value is not a tensor object
	// (a bare number) → entry Unmarshal fails.
	hdr := `{"t": 12345}`
	p := writeTemp(t, buildST(hdr, nil))
	if _, err := LoadTensors(p); err == nil {
		t.Error("non-object entry value should error")
	}
}

func TestLoadTensorsMetadataSkipped(t *testing.T) {
	hdr := `{"__metadata__":{"format":"pt"},` +
		`"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`
	p := writeTemp(t, buildST(hdr, f32LE([]float32{5, 6})))
	out, err := LoadTensors(p)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	if len(out) != 1 || out["w"].Data[1] != 6 {
		t.Errorf("expected single tensor w with data[1]=6, got %v", out)
	}
}

// ── ReadBytes: entry-value parse error ──────────────────────────────────────

func TestReadBytesBadEntryJSON(t *testing.T) {
	hdr := `{"t": 99}` // entry value is a number, not a tensor object
	if _, err := ReadBytes(buildST(hdr, nil)); err == nil {
		t.Error("ReadBytes with non-object entry should error")
	}
}

// ── orderedTopKeys: edge cases ──────────────────────────────────────────────

func TestOrderedTopKeysEdges(t *testing.T) {
	// Unterminated string literal → nil (best-effort bail).
	if k := orderedTopKeys([]byte(`{"abc`)); k != nil {
		t.Errorf("unterminated string should yield nil, got %v", k)
	}
	// Whitespace between key and colon must still register the key.
	k := orderedTopKeys([]byte(`{"a"   : 1, "b"	:2}`))
	if strings.Join(k, ",") != "a,b" {
		t.Errorf("whitespace-before-colon keys = %v, want [a b]", k)
	}
	// A string at depth > 1 (inside a nested array) is not a top key.
	k2 := orderedTopKeys([]byte(`{"a":["x","y"],"b":1}`))
	if strings.Join(k2, ",") != "a,b" {
		t.Errorf("nested strings leaked into keys: %v", k2)
	}
	// A quoted value (not followed by ':') at depth 1 is not a key.
	k3 := orderedTopKeys([]byte(`{"a":"value"}`))
	if strings.Join(k3, ",") != "a" {
		t.Errorf("quoted value should not be a key: %v", k3)
	}
}

// ── decodeToFloat32: unsupported dtype direct ───────────────────────────────

func TestDecodeUnsupportedDtypeDirect(t *testing.T) {
	if _, err := decodeToFloat32([]byte{0}, "WEIRD", []int64{1}, "t"); err == nil {
		t.Error("unsupported dtype should error in decodeToFloat32")
	}
}

func TestDecodeBoolDirect(t *testing.T) {
	out, err := decodeToFloat32([]byte{0, 1, 7}, "BOOL", []int64{3}, "t")
	if err != nil {
		t.Fatalf("BOOL decode: %v", err)
	}
	want := []float32{0, 1, 1}
	for i := range want {
		if out[i] != want[i] {
			t.Errorf("BOOL[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}
