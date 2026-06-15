package npy

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// writeTemp writes data to a temp file and returns its path.
func writeTemp(t *testing.T, name string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatalf("write temp: %v", err)
	}
	return p
}

// ── Load: file-based happy path + error arms ────────────────────────────────

func TestLoadFileRoundTrip(t *testing.T) {
	a := uop.NewArena(64)
	data := buildNPY("<f4", false, "(2, 2)", f32Payload([]float32{1, 2, 3, 4}))
	p := writeTemp(t, "arr.npy", data)
	tn, err := Load(a, p, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := tn.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Errorf("shape = %v, want [2 2]", got)
	}
}

func TestLoadMissingFile(t *testing.T) {
	a := uop.NewArena(64)
	if _, err := Load(a, filepath.Join(t.TempDir(), "nope.npy"), "cpu"); err == nil {
		t.Error("missing file should error")
	}
}

func TestLoadBadData(t *testing.T) {
	a := uop.NewArena(64)
	p := writeTemp(t, "bad.npy", []byte("not a numpy file at all"))
	if _, err := Load(a, p, "cpu"); err == nil {
		t.Error("non-npy file should error")
	}
}

// ── parseNPY: branches distinct from parseNPYBytes ──────────────────────────

func TestParseNPYErrors(t *testing.T) {
	good := buildNPY("<f4", false, "(2,)", f32Payload([]float32{1, 2}))
	cases := []struct {
		name string
		data []byte
		frag string
	}{
		{"too-short", []byte{0x93, 'N'}, "too short"},
		{"bad-magic", append([]byte{0x00}, good[1:]...), "bad magic"},
		{"unsupported-version", func() []byte {
			d := append([]byte(nil), good...)
			d[6] = 9
			return d
		}(), "unsupported format version"},
		{"v2-short", []byte{0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0, 1, 0}, "too short for v2"},
		{"header-overflow", func() []byte {
			d := append([]byte(nil), good...)
			d[8] = 0xff
			d[9] = 0xff
			return d
		}(), "exceeds file length"},
		{"truncated-payload", buildNPY("<f4", false, "(8,)", f32Payload([]float32{1, 2})), "truncated"},
		{"bad-descr", buildNPY("zzz", false, "(1,)", []byte{0, 0, 0, 0}), ""},
		{"bad-header", func() []byte {
			// Replace the header dict with garbage that has no 'descr' key.
			return buildRawNPY("{'nope': 1}", []byte{0, 0, 0, 0})
		}(), "header parse"},
	}
	a := uop.NewArena(64)
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := parseNPY(a, c.data, "cpu")
			if err == nil {
				t.Fatalf("expected error (frag %q), got nil", c.frag)
			}
			if c.frag != "" && !strings.Contains(err.Error(), c.frag) {
				t.Fatalf("error = %v, want fragment %q", err, c.frag)
			}
		})
	}
}

func TestParseNPYScalar(t *testing.T) {
	a := uop.NewArena(64)
	data := buildNPY("<f4", false, "()", f32Payload([]float32{42}))
	tn, err := parseNPY(a, data, "cpu")
	if err != nil {
		t.Fatalf("parseNPY scalar: %v", err)
	}
	if len(tn.Shape()) != 0 {
		t.Errorf("scalar shape = %v, want []", tn.Shape())
	}
}

func TestParseNPYV2Header(t *testing.T) {
	a := uop.NewArena(64)
	data := buildNPYv2("<f4", "(3,)", f32Payload([]float32{1, 2, 3}))
	tn, err := parseNPY(a, data, "cpu")
	if err != nil {
		t.Fatalf("parseNPY v2: %v", err)
	}
	if tn.Shape()[0] != 3 {
		t.Errorf("v2 shape = %v, want [3]", tn.Shape())
	}
}

func TestParseNPYFortran(t *testing.T) {
	a := uop.NewArena(64)
	// Column-major 2x2: payload laid out [a00,a10,a01,a11].
	data := buildNPY("<f4", true, "(2, 2)", f32Payload([]float32{1, 3, 2, 4}))
	tn, err := parseNPY(a, data, "cpu")
	if err != nil {
		t.Fatalf("parseNPY fortran: %v", err)
	}
	// After fortranToC the row-major data should be [1,2,3,4].
	got := tn.Data()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("fortran data[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// ── LoadNPZ: file-based happy + error arms ──────────────────────────────────

func buildNPZ(t *testing.T, entries map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, data := range entries {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatalf("zip create: %v", err)
		}
		if _, err := w.Write(data); err != nil {
			t.Fatalf("zip write: %v", err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	return buf.Bytes()
}

func TestLoadNPZRoundTrip(t *testing.T) {
	a := uop.NewArena(128)
	npz := buildNPZ(t, map[string][]byte{
		"x.npy": buildNPY("<f4", false, "(2,)", f32Payload([]float32{1, 2})),
		"y.npy": buildNPY("<f4", false, "(3,)", f32Payload([]float32{3, 4, 5})),
	})
	p := writeTemp(t, "pair.npz", npz)
	m, err := LoadNPZ(a, p, "cpu")
	if err != nil {
		t.Fatalf("LoadNPZ: %v", err)
	}
	if len(m) != 2 || m["x"] == nil || m["y"] == nil {
		t.Errorf("npz map = %v, want x and y keys", m)
	}
	if m["y"].Shape()[0] != 3 {
		t.Errorf("y shape = %v, want [3]", m["y"].Shape())
	}
}

func TestLoadNPZMissingFile(t *testing.T) {
	a := uop.NewArena(64)
	if _, err := LoadNPZ(a, filepath.Join(t.TempDir(), "nope.npz"), "cpu"); err == nil {
		t.Error("missing npz should error")
	}
}

func TestLoadNPZNotZip(t *testing.T) {
	a := uop.NewArena(64)
	p := writeTemp(t, "fake.npz", []byte("definitely not a zip archive"))
	if _, err := LoadNPZ(a, p, "cpu"); err == nil {
		t.Error("non-zip npz should error")
	}
}

func TestLoadNPZCorruptEntry(t *testing.T) {
	a := uop.NewArena(64)
	npz := buildNPZ(t, map[string][]byte{
		"bad.npy": []byte("not numpy"),
	})
	p := writeTemp(t, "corrupt.npz", npz)
	if _, err := LoadNPZ(a, p, "cpu"); err == nil {
		t.Error("corrupt npz entry should error")
	}
}

// ── ReadZBytes error arm ────────────────────────────────────────────────────

func TestReadZBytesBadZip(t *testing.T) {
	if _, _, err := ReadZBytes([]byte("not a zip")); err == nil {
		t.Error("ReadZBytes on non-zip should error")
	}
}

func TestReadZBytesCorruptEntry(t *testing.T) {
	npz := buildNPZ(t, map[string][]byte{"bad.npy": []byte("xx")})
	if _, _, err := ReadZBytes(npz); err == nil {
		t.Error("ReadZBytes on corrupt entry should error")
	}
}

func TestReadBytesError(t *testing.T) {
	if _, err := ReadBytes([]byte{0x93, 'N'}); err == nil {
		t.Error("ReadBytes on truncated data should error")
	}
}

// buildRawNPY builds a v1 npy with an arbitrary header dict body (no padding
// constraints) and the given payload.
func buildRawNPY(hdrBody string, payload []byte) []byte {
	hdr := hdrBody
	// Pad to multiple of 64 ending in newline (matches numpy alignment).
	total := 10 + len(hdr) + 1
	if rem := total % 64; rem != 0 {
		hdr += strings.Repeat(" ", 64-rem)
	}
	hdr += "\n"
	out := []byte{0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0}
	out = append(out, byte(len(hdr)), byte(len(hdr)>>8))
	out = append(out, []byte(hdr)...)
	out = append(out, payload...)
	return out
}
