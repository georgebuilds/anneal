package npy_test

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/npy"
	"github.com/georgebuilds/anneal/uop"
)

const testdataDir = "testdata"

func newArena() *uop.Arena { return uop.NewArena(256) }

// requireFixture skips the test if the named fixture file is absent.
func requireFixture(t *testing.T, name string) string {
	t.Helper()
	path := testdataDir + "/" + name
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture %q not found; run testdata/gen_fixtures.py to generate it", name)
	}
	return path
}

// assertShape asserts that got equals want and reports shapes on mismatch.
func assertShape(t *testing.T, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape rank %d != %d; got %v want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape[%d]: got %d want %d (full shape: got %v want %v)", i, got[i], want[i], got, want)
		}
	}
}

// assertData asserts element-wise equality and prints a sample on mismatch.
func assertData(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("data length %d != %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("data[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

// ── float32 (bit-exact) ───────────────────────────────────────────────────────

func TestLoad_Float32_3x4(t *testing.T) {
	path := requireFixture(t, "float32_3x4.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{3, 4})

	want := make([]float32, 12)
	for i := range want {
		want[i] = float32(i)
	}
	assertData(t, ten.Data(), want)

	t.Logf("float32 3×4: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── float64 → float32 (precision loss, exact for these values) ───────────────

func TestLoad_Float64_Exact(t *testing.T) {
	path := requireFixture(t, "float64_exact.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{4})
	// Fixture: [0.0, 1.0, 2.0, 100.0] as float64 — all exactly representable in float32.
	want := []float32{0.0, 1.0, 2.0, 100.0}
	assertData(t, ten.Data(), want)

	t.Logf("float64→float32: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── integer types (by value) ──────────────────────────────────────────────────

func TestLoad_Int32_2x3(t *testing.T) {
	path := requireFixture(t, "int32_2x3.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{2, 3})
	// Fixture: [[1,2,3],[4,5,6]] as int32 → by value as float32.
	want := []float32{1, 2, 3, 4, 5, 6}
	assertData(t, ten.Data(), want)

	t.Logf("int32 2×3: shape=%v data=%v", ten.Shape(), ten.Data())
}

func TestLoad_Int8_Negative(t *testing.T) {
	path := requireFixture(t, "int8_1d.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{4})
	// Fixture: [-5, 0, 10, 127] as int8 → float32 by value.
	want := []float32{-5, 0, 10, 127}
	assertData(t, ten.Data(), want)

	t.Logf("int8 1D: shape=%v data=%v", ten.Shape(), ten.Data())
}

func TestLoad_Uint8(t *testing.T) {
	path := requireFixture(t, "uint8_1d.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{3})
	// Fixture: [0, 1, 255] as uint8 → float32 by value.
	want := []float32{0, 1, 255}
	assertData(t, ten.Data(), want)

	t.Logf("uint8 1D: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── bool → 0.0 / 1.0 ─────────────────────────────────────────────────────────

func TestLoad_Bool(t *testing.T) {
	path := requireFixture(t, "bool_1d.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{4})
	// Fixture: [True, False, True, True] → [1.0, 0.0, 1.0, 1.0].
	want := []float32{1.0, 0.0, 1.0, 1.0}
	assertData(t, ten.Data(), want)

	t.Logf("bool 1D: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── Fortran order → C order transposition ────────────────────────────────────

func TestLoad_FortranOrder(t *testing.T) {
	path := requireFixture(t, "fortran_2x3.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Fortran file stores [[0,1,2],[3,4,5]] in column-major order:
	// raw bytes = [0, 3, 1, 4, 2, 5].
	// After transposing to C order, logical data = [0, 1, 2, 3, 4, 5].
	assertShape(t, ten.Shape(), []int64{2, 3})
	want := []float32{0, 1, 2, 3, 4, 5}
	assertData(t, ten.Data(), want)

	t.Logf("fortran 2×3: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── scalar (0-D) ──────────────────────────────────────────────────────────────

func TestLoad_Scalar(t *testing.T) {
	path := requireFixture(t, "scalar_f32.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Shape () → empty int64 slice.
	assertShape(t, ten.Shape(), []int64{})
	assertData(t, ten.Data(), []float32{42.0})

	t.Logf("scalar: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── 3D array ──────────────────────────────────────────────────────────────────

func TestLoad_3D(t *testing.T) {
	path := requireFixture(t, "float32_2x3x4.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{2, 3, 4})
	want := make([]float32, 24)
	for i := range want {
		want[i] = float32(i)
	}
	assertData(t, ten.Data(), want)

	t.Logf("float32 2×3×4: shape=%v data[:4]=%v", ten.Shape(), ten.Data()[:4])
}

// ── big-endian byte order ─────────────────────────────────────────────────────

func TestLoad_BigEndian(t *testing.T) {
	path := requireFixture(t, "bigendian_f32.npy")
	a := newArena()

	ten, err := npy.Load(a, path, "cpu")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	assertShape(t, ten.Shape(), []int64{4})
	// Fixture: [1.0, 2.0, 3.0, 4.0] stored big-endian — logical values unchanged.
	want := []float32{1.0, 2.0, 3.0, 4.0}
	assertData(t, ten.Data(), want)

	t.Logf("big-endian float32: shape=%v data=%v", ten.Shape(), ten.Data())
}

// ── honest errors: unsupported dtype ─────────────────────────────────────────

func TestLoad_Error_Complex(t *testing.T) {
	path := requireFixture(t, "complex64.npy")
	a := newArena()

	_, err := npy.Load(a, path, "cpu")
	if err == nil {
		t.Fatal("expected error for complex dtype, got nil")
	}
	if !strings.Contains(err.Error(), "complex") {
		t.Errorf("error should mention 'complex', got: %v", err)
	}
	t.Logf("complex64 error (expected): %v", err)
}

// ── honest errors: out-of-range integer ──────────────────────────────────────

func TestLoad_Error_IntRange(t *testing.T) {
	path := requireFixture(t, "int64_oor.npy")
	a := newArena()

	_, err := npy.Load(a, path, "cpu")
	if err == nil {
		t.Fatal("expected error for out-of-range int64 value, got nil")
	}
	// Error should name the offending value (16777217 = 2^24+1).
	if !strings.Contains(err.Error(), "16777217") {
		t.Errorf("error should contain the out-of-range value 16777217, got: %v", err)
	}
	t.Logf("int64 out-of-range error (expected): %v", err)
}

// ── .npz round-trip ───────────────────────────────────────────────────────────

func TestLoadNPZ_MultiArray(t *testing.T) {
	path := requireFixture(t, "multi.npz")
	a := newArena()

	tensors, err := npy.LoadNPZ(a, path, "cpu")
	if err != nil {
		t.Fatalf("LoadNPZ: %v", err)
	}

	// Key set check: exactly {weights, bias, mask}.
	wantKeys := []string{"bias", "mask", "weights"}
	gotKeys := make([]string, 0, len(tensors))
	for k := range tensors {
		gotKeys = append(gotKeys, k)
	}
	sort.Strings(gotKeys)
	if fmt.Sprint(gotKeys) != fmt.Sprint(wantKeys) {
		t.Fatalf("npz keys: got %v want %v", gotKeys, wantKeys)
	}

	// weights: shape [2,3], values [0..5]
	w := tensors["weights"]
	assertShape(t, w.Shape(), []int64{2, 3})
	assertData(t, w.Data(), []float32{0, 1, 2, 3, 4, 5})
	t.Logf("npz weights: shape=%v data=%v", w.Shape(), w.Data())

	// bias: shape [3], values [0.0, 0.5, 1.0]
	b := tensors["bias"]
	assertShape(t, b.Shape(), []int64{3})
	assertData(t, b.Data(), []float32{0.0, 0.5, 1.0})
	t.Logf("npz bias: shape=%v data=%v", b.Shape(), b.Data())

	// mask: shape [3], bool [True,False,True] → [1.0, 0.0, 1.0]
	m := tensors["mask"]
	assertShape(t, m.Shape(), []int64{3})
	assertData(t, m.Data(), []float32{1.0, 0.0, 1.0})
	t.Logf("npz mask: shape=%v data=%v", m.Shape(), m.Data())
}

// ── dtype is always Float32 ───────────────────────────────────────────────────

func TestLoad_DTypeIsAlwaysFloat32(t *testing.T) {
	fixtures := []string{
		"float32_3x4.npy",
		"float64_exact.npy",
		"int32_2x3.npy",
		"int8_1d.npy",
		"uint8_1d.npy",
		"bool_1d.npy",
	}
	for _, fix := range fixtures {
		t.Run(fix, func(t *testing.T) {
			path := requireFixture(t, fix)
			a := newArena()
			ten, err := npy.Load(a, path, "cpu")
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if ten.DType() != uop.Dtypes.Float32 {
				t.Errorf("expected dtype Float32, got %s", ten.DType())
			}
		})
	}
}
