package codegen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// ── joinPlus ─────────────────────────────────────────────────────────────────

func TestJoinPlus(t *testing.T) {
	cases := []struct {
		in   []string
		want string
	}{
		{nil, "0"},
		{[]string{}, "0"},
		{[]string{"a"}, "a"},
		{[]string{"a", "b"}, "a + b"},
		{[]string{"x", "y", "z"}, "x + y + z"},
	}
	for _, c := range cases {
		if got := joinPlus(c.in); got != c.want {
			t.Errorf("joinPlus(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ── paramShape ───────────────────────────────────────────────────────────────

func TestParamShape(t *testing.T) {
	l := &lowerer{item: schedule.ExecItem{Bufs: []schedule.Buffer{
		{Shape: []int64{4, 8}},
		{Shape: []int64{16}},
	}}}
	if got := l.paramShape(0); len(got) != 2 || got[0] != 4 || got[1] != 8 {
		t.Errorf("paramShape(0) = %v, want [4 8]", got)
	}
	if got := l.paramShape(1); len(got) != 1 || got[0] != 16 {
		t.Errorf("paramShape(1) = %v, want [16]", got)
	}
	// Out-of-range param falls back to [1].
	if got := l.paramShape(9); len(got) != 1 || got[0] != 1 {
		t.Errorf("paramShape(9) = %v, want [1] fallback", got)
	}
}

// ── paramIsImage ─────────────────────────────────────────────────────────────

func TestParamIsImage(t *testing.T) {
	l := &lowerer{item: schedule.ExecItem{Bufs: []schedule.Buffer{
		{DType: uop.Dtypes.Float32},
	}}}
	if l.paramIsImage(0) {
		t.Error("plain f32 buffer must not be image")
	}
	// Out-of-range index is safely false (no panic).
	if l.paramIsImage(5) {
		t.Error("out-of-range param must report not-image")
	}
}

// ── paramDimFactor ───────────────────────────────────────────────────────────

func TestParamDimFactorConcrete(t *testing.T) {
	l := &lowerer{item: schedule.ExecItem{Bufs: []schedule.Buffer{
		{Shape: []int64{4, 8}},
	}}}
	// Concrete dim returns mulConst.
	acc := l.paramDimFactor(0, 1)
	if got := acc.renderU32(); got != "8u" {
		t.Errorf("concrete dim factor = %q, want '8u'", got)
	}
}

func TestParamDimFactorOutOfRange(t *testing.T) {
	l := &lowerer{item: schedule.ExecItem{Bufs: []schedule.Buffer{
		{Shape: []int64{4}},
	}}}
	// paramIdx beyond Bufs → identity.
	if got := l.paramDimFactor(9, 0).renderU32(); got != "1u" {
		t.Errorf("oob paramIdx factor = %q, want '1u'", got)
	}
	// dim beyond Shape → identity.
	if got := l.paramDimFactor(0, 5).renderU32(); got != "1u" {
		t.Errorf("oob dim factor = %q, want '1u'", got)
	}
	// Negative dim → identity.
	if got := l.paramDimFactor(0, -1).renderU32(); got != "1u" {
		t.Errorf("negative dim factor = %q, want '1u'", got)
	}
}

func TestParamDimFactorSymDimVar(t *testing.T) {
	// Symbolic dim (Shape[0]==0) resolved via SymDimVar/SymDimMul.
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0, 4}, SymDimVar: []string{"n"}, SymDimMul: []int64{1}},
		}},
		symSlotByName: map[string]int{"n": 2},
	}
	if got := l.paramDimFactor(0, 0).renderU32(); got != "params_n.n2" {
		t.Errorf("sym dim factor = %q, want 'params_n.n2'", got)
	}
}

func TestParamDimFactorSymDimVarWithMul(t *testing.T) {
	// SymDimMul > 1 multiplies the sym factor by a constant.
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0}, SymDimVar: []string{"n"}, SymDimMul: []int64{4}},
		}},
		symSlotByName: map[string]int{"n": 0},
	}
	got := l.paramDimFactor(0, 0).renderU32()
	if !strings.Contains(got, "params_n.n0") || !strings.Contains(got, "4u") {
		t.Errorf("sym*mul factor = %q, want params_n.n0 * 4u form", got)
	}
}

func TestParamDimFactorAffineSingleTerm(t *testing.T) {
	// SymDimAffine single term with Mul==1 → bare slot reference.
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0}, SymDimAffine: []schedule.SymDimAffineEntry{
				{Terms: []uop.AffineTerm{{Mul: 1, VarName: "n"}}},
			}},
		}},
		symSlotByName: map[string]int{"n": 1},
	}
	if got := l.paramDimFactor(0, 0).renderU32(); got != "params_n.n1" {
		t.Errorf("affine single-term = %q, want 'params_n.n1'", got)
	}
}

func TestParamDimFactorAffineMultiTermWithOffset(t *testing.T) {
	// SymDimAffine with two terms (one with Mul>1) plus an offset.
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0}, SymDimAffine: []schedule.SymDimAffineEntry{
				{Terms: []uop.AffineTerm{
					{Mul: 1, VarName: "a"},
					{Mul: 3, VarName: "b"},
				}, Offset: 5},
			}},
		}},
		symSlotByName: map[string]int{"a": 0, "b": 1},
	}
	got := l.paramDimFactor(0, 0).renderU32()
	for _, want := range []string{"params_n.n0", "params_n.n1 * 3u", "5u"} {
		if !strings.Contains(got, want) {
			t.Errorf("affine multi-term = %q, missing %q", got, want)
		}
	}
}

func TestParamDimFactorAffineEmptyTermsIdentity(t *testing.T) {
	// SymDimAffine entry with no terms and zero offset collapses to identity.
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0}, SymDimAffine: []schedule.SymDimAffineEntry{{}}},
		}},
		symSlotByName: map[string]int{},
	}
	if got := l.paramDimFactor(0, 0).renderU32(); got != "1u" {
		t.Errorf("empty-affine factor = %q, want '1u'", got)
	}
}

func TestParamDimFactorMissingSlotPanics(t *testing.T) {
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0}, SymDimVar: []string{"n"}},
		}},
		symSlotByName: map[string]int{}, // "n" absent
	}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when sym var not in slot map")
		}
	}()
	_ = l.paramDimFactor(0, 0)
}

// ── saveDiskMapLocked ────────────────────────────────────────────────────────

// saveDiskMapLocked persists diskMap to diskPath. Redirect diskPath to a temp
// file (snapshot/restore the package globals) so the real user cache is never
// touched, then verify the written JSON round-trips.
func TestSaveDiskMapLocked(t *testing.T) {
	diskMu.Lock()
	origPath := diskPath
	origMap := diskMap
	diskMu.Unlock()
	defer func() {
		diskMu.Lock()
		diskPath = origPath
		diskMap = origMap
		diskMu.Unlock()
	}()

	tmpDir := t.TempDir()
	diskMu.Lock()
	diskPath = filepath.Join(tmpDir, "nested", "beam_cache.json")
	diskMap = map[string]diskEntry{
		"00000000deadbeef": {Opts: []Opt{{Kind: OptLocal, Axis: 0, Arg: 8}}, WGSLHash: "abc"},
	}
	saveDiskMapLocked()
	diskMu.Unlock()

	data, err := os.ReadFile(diskPath)
	if err != nil {
		t.Fatalf("saveDiskMapLocked did not write file: %v", err)
	}
	s := string(data)
	if !strings.Contains(s, "deadbeef") || !strings.Contains(s, "abc") {
		t.Errorf("written cache missing expected content:\n%s", s)
	}
}
