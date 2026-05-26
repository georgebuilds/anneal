package shape

import (
	"testing"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func i2(lo, hi int64) [2]int64 { return [2]int64{lo, hi} }

func viewEq(t *testing.T, label string, got, want View) {
	t.Helper()
	if !shapesEqual(got.Shape, want.Shape) {
		t.Errorf("%s: shape got %v want %v", label, got.Shape, want.Shape)
	}
	if !stridesEqual(got.Strides, want.Strides) {
		t.Errorf("%s: strides got %v want %v", label, got.Strides, want.Strides)
	}
	if got.Offset != want.Offset {
		t.Errorf("%s: offset got %d want %d", label, got.Offset, want.Offset)
	}
	if !maskEq(got.Mask, want.Mask) {
		t.Errorf("%s: mask got %v want %v", label, got.Mask, want.Mask)
	}
	if got.Contiguous != want.Contiguous {
		t.Errorf("%s: contiguous got %v want %v", label, got.Contiguous, want.Contiguous)
	}
}

func maskEq(a, b [][2]int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ── stridesForShape ───────────────────────────────────────────────────────────

func TestStridesForShape(t *testing.T) {
	cases := []struct {
		shape   []int64
		strides []int64
	}{
		{[]int64{}, []int64{}},
		{[]int64{5}, []int64{1}},
		{[]int64{1}, []int64{0}}, // size-1 → stride 0
		{[]int64{2, 3}, []int64{3, 1}},
		{[]int64{2, 1, 4}, []int64{4, 0, 1}}, // middle size-1
		{[]int64{1, 1, 1}, []int64{0, 0, 0}},
		{[]int64{4, 3, 2}, []int64{6, 2, 1}},
	}
	for _, tc := range cases {
		got := stridesForShape(tc.shape)
		if !stridesEqual(got, tc.strides) {
			t.Errorf("stridesForShape(%v) = %v, want %v", tc.shape, got, tc.strides)
		}
	}
}

// ── contiguity ────────────────────────────────────────────────────────────────

func TestContiguous(t *testing.T) {
	cases := []struct {
		name       string
		v          View
		wantContig bool
	}{
		{"fresh 2x3", NewContiguousView([]int64{2, 3}), true},
		{"offset≠0", NewView([]int64{3}, []int64{1}, 5, nil), false},
		{"with mask", NewView([]int64{4}, []int64{1}, 0, [][2]int64{i2(1, 3)}), false},
		{"wrong strides", NewView([]int64{2, 3}, []int64{4, 1}, 0, nil), false},
		{"scalar", NewContiguousView([]int64{}), true},
		{"single elem", NewContiguousView([]int64{1}), true},
	}
	for _, tc := range cases {
		if tc.v.Contiguous != tc.wantContig {
			t.Errorf("%s: contiguous=%v want %v", tc.name, tc.v.Contiguous, tc.wantContig)
		}
	}
}

// ── Expand ────────────────────────────────────────────────────────────────────

func TestExpand(t *testing.T) {
	cases := []struct {
		name     string
		v        View
		newShape []int64
		want     View
	}{
		{
			name:     "broadcast dim-0",
			v:        NewContiguousView([]int64{1, 4}),
			newShape: []int64{3, 4},
			want:     NewView([]int64{3, 4}, []int64{0, 1}, 0, nil),
		},
		{
			name:     "broadcast dim-1",
			v:        NewContiguousView([]int64{3, 1}),
			newShape: []int64{3, 5},
			want:     NewView([]int64{3, 5}, []int64{1, 0}, 0, nil),
		},
		{
			name:     "no-op",
			v:        NewContiguousView([]int64{2, 3}),
			newShape: []int64{2, 3},
			want:     NewContiguousView([]int64{2, 3}),
		},
		{
			name:     "broadcast with mask (full)",
			v:        NewView([]int64{1, 3}, []int64{0, 1}, 0, [][2]int64{i2(0, 1), i2(1, 2)}),
			newShape: []int64{4, 3},
			// mask dim-0 was (0,1) → expands to (0,4)
			want: NewView([]int64{4, 3}, []int64{0, 1}, 0, [][2]int64{i2(0, 4), i2(1, 2)}),
		},
	}
	for _, tc := range cases {
		got := tc.v.Expand(tc.newShape)
		viewEq(t, tc.name, got, tc.want)
	}
}

// ── Permute ───────────────────────────────────────────────────────────────────

func TestPermute(t *testing.T) {
	cases := []struct {
		name  string
		v     View
		order []int
		want  View
	}{
		{
			name:  "transpose 2x3",
			v:     NewContiguousView([]int64{2, 3}),
			order: []int{1, 0},
			want:  NewView([]int64{3, 2}, []int64{1, 3}, 0, nil),
		},
		{
			name:  "3-dim rotation",
			v:     NewContiguousView([]int64{2, 3, 4}),
			order: []int{2, 0, 1},
			want:  NewView([]int64{4, 2, 3}, []int64{1, 12, 4}, 0, nil),
		},
		{
			name:  "permute with mask",
			v:     NewView([]int64{3, 4}, []int64{4, 1}, 0, [][2]int64{i2(1, 2), i2(0, 3)}),
			order: []int{1, 0},
			want:  NewView([]int64{4, 3}, []int64{1, 4}, 0, [][2]int64{i2(0, 3), i2(1, 2)}),
		},
		{
			name:  "identity permute",
			v:     NewContiguousView([]int64{2, 3}),
			order: []int{0, 1},
			want:  NewContiguousView([]int64{2, 3}),
		},
	}
	for _, tc := range cases {
		got := tc.v.Permute(tc.order)
		viewEq(t, tc.name, got, tc.want)
	}
}

// ── Pad ───────────────────────────────────────────────────────────────────────

func TestPad(t *testing.T) {
	cases := []struct {
		name string
		v    View
		arg  [][2]int64
		want View
	}{
		{
			name: "pad dim-1 by (1,1)",
			v:    NewContiguousView([]int64{2, 3}),
			arg:  [][2]int64{i2(0, 0), i2(1, 1)},
			// shape=(2,5), strides=(3,1), offset=-1, mask=[(0,2),(1,4)]
			want: NewView([]int64{2, 5}, []int64{3, 1}, -1, [][2]int64{i2(0, 2), i2(1, 4)}),
		},
		{
			name: "pad both dims",
			v:    NewContiguousView([]int64{3}),
			arg:  [][2]int64{i2(2, 1)},
			// shape=6, strides=1, offset=-2, mask=[(2,5)]
			want: NewView([]int64{6}, []int64{1}, -2, [][2]int64{i2(2, 5)}),
		},
		{
			name: "no-op pad",
			v:    NewContiguousView([]int64{3, 4}),
			arg:  [][2]int64{i2(0, 0), i2(0, 0)},
			want: NewContiguousView([]int64{3, 4}),
		},
	}
	for _, tc := range cases {
		got := tc.v.Pad(tc.arg)
		viewEq(t, tc.name, got, tc.want)
	}
}

// ── Shrink ────────────────────────────────────────────────────────────────────

func TestShrink(t *testing.T) {
	cases := []struct {
		name string
		v    View
		arg  [][2]int64
		want View
	}{
		{
			name: "shrink dim-0",
			v:    NewContiguousView([]int64{6, 4}),
			arg:  [][2]int64{i2(2, 5), i2(0, 4)},
			// offset=2*4=8, shape=(3,4)
			want: NewView([]int64{3, 4}, []int64{4, 1}, 8, nil),
		},
		{
			name: "shrink dim-1",
			v:    NewContiguousView([]int64{4, 6}),
			arg:  [][2]int64{i2(0, 4), i2(1, 4)},
			// offset=0+1*1=1, shape=(4,3)
			want: NewView([]int64{4, 3}, []int64{6, 1}, 1, nil),
		},
		{
			name: "shrink with mask intersection",
			// view with existing mask, then shrink
			v:   NewView([]int64{8}, []int64{1}, 0, [][2]int64{i2(2, 6)}),
			arg: [][2]int64{i2(3, 7)},
			// new offset=3, shape=4. Existing mask (2,6) shifted by -3 → (-1,3) clamped to (0,3)
			want: NewView([]int64{4}, []int64{1}, 3, [][2]int64{i2(0, 3)}),
		},
	}
	for _, tc := range cases {
		got := tc.v.Shrink(tc.arg)
		viewEq(t, tc.name, got, tc.want)
	}
}

// ── Flip ──────────────────────────────────────────────────────────────────────

func TestFlip(t *testing.T) {
	cases := []struct {
		name string
		v    View
		axes []bool
		want View
	}{
		{
			name: "flip 1D",
			v:    NewContiguousView([]int64{5}),
			axes: []bool{true},
			// offset=(5-1)*1=4, strides=(-1)
			want: NewView([]int64{5}, []int64{-1}, 4, nil),
		},
		{
			name: "flip dim-0 of 2D",
			v:    NewContiguousView([]int64{3, 4}),
			axes: []bool{true, false},
			// offset=(3-1)*4=8, strides=(-4,1)
			want: NewView([]int64{3, 4}, []int64{-4, 1}, 8, nil),
		},
		{
			name: "flip both dims",
			v:    NewContiguousView([]int64{2, 3}),
			axes: []bool{true, true},
			// offset=(2-1)*3+(3-1)*1=3+2=5, strides=(-3,-1)
			want: NewView([]int64{2, 3}, []int64{-3, -1}, 5, nil),
		},
		{
			name: "flip with mask",
			v:    NewView([]int64{6}, []int64{1}, 0, [][2]int64{i2(1, 4)}),
			axes: []bool{true},
			// offset=(6-1)*1=5, strides=(-1), mask=(6-4,6-1)=(2,5)
			want: NewView([]int64{6}, []int64{-1}, 5, [][2]int64{i2(2, 5)}),
		},
		{
			name: "flip no-op",
			v:    NewContiguousView([]int64{3}),
			axes: []bool{false},
			want: NewContiguousView([]int64{3}),
		},
	}
	for _, tc := range cases {
		got := tc.v.Flip(tc.axes)
		viewEq(t, tc.name, got, tc.want)
	}
}

// ── Reshape ───────────────────────────────────────────────────────────────────

func TestReshape(t *testing.T) {
	cases := []struct {
		name     string
		v        View
		newShape []int64
		wantOk   bool
		want     View
	}{
		{
			name:     "contiguous reshape",
			v:        NewContiguousView([]int64{6}),
			newShape: []int64{2, 3},
			wantOk:   true,
			want:     NewContiguousView([]int64{2, 3}),
		},
		{
			name:     "same shape no-op",
			v:        NewContiguousView([]int64{2, 3}),
			newShape: []int64{2, 3},
			wantOk:   true,
			want:     NewContiguousView([]int64{2, 3}),
		},
		{
			name:     "flatten 2D",
			v:        NewContiguousView([]int64{4, 5}),
			newShape: []int64{20},
			wantOk:   true,
			want:     NewContiguousView([]int64{20}),
		},
		{
			name:     "zero-size dim",
			v:        NewContiguousView([]int64{0, 5}),
			newShape: []int64{0},
			wantOk:   true,
			want:     NewContiguousView([]int64{0}),
		},
		{
			// Non-contiguous (stride-0 expanded) reshape that succeeds
			// shape=(2,3) strides=(0,1): broadcast dim-0.  Reshape to (2,1,3) keeps stride pattern.
			name:     "expanded reshape to higher rank",
			v:        NewView([]int64{2, 3}, []int64{0, 1}, 0, nil),
			newShape: []int64{2, 1, 3},
			wantOk:   true,
			want:     NewView([]int64{2, 1, 3}, []int64{0, 0, 1}, 0, nil),
		},
		{
			// Expanding (2,3) strides (0,1) to (6,) mixes broadcast+real → must fail.
			name:     "expanded reshape cannot merge broadcast with real",
			v:        NewView([]int64{2, 3}, []int64{0, 1}, 0, nil),
			newShape: []int64{6},
			wantOk:   false,
		},
		{
			// Mask (0,3) on shape (6,) → new shape (2,3): mask becomes (0,1),(0,3).
			name:     "reshape with mask aligned",
			v:        NewView([]int64{6}, []int64{1}, 0, [][2]int64{i2(0, 3)}),
			newShape: []int64{2, 3},
			wantOk:   true,
			want:     NewView([]int64{2, 3}, []int64{3, 1}, 0, [][2]int64{i2(0, 1), i2(0, 3)}),
		},
		{
			// Mask (3,6) on shape (6,) → new shape (2,3): mask becomes (1,2),(0,3).
			// extra_offset = 3*1 - 1*3 - 0*1 = 0; new offset = 0+0 = 0.
			name:     "reshape with mask offset",
			v:        NewView([]int64{6}, []int64{1}, 0, [][2]int64{i2(3, 6)}),
			newShape: []int64{2, 3},
			wantOk:   true,
			want:     NewView([]int64{2, 3}, []int64{3, 1}, 0, [][2]int64{i2(1, 2), i2(0, 3)}),
		},
		{
			// Mask (2,4) on shape (6,) crosses row boundary in (2,3) → must fail.
			name:     "reshape mask crosses boundary",
			v:        NewView([]int64{6}, []int64{1}, 0, [][2]int64{i2(2, 4)}),
			newShape: []int64{2, 3},
			wantOk:   false,
		},
		{
			// Reshape to scalar with full content (mask nil).
			name:     "reshape to scalar",
			v:        NewContiguousView([]int64{1}),
			newShape: []int64{},
			wantOk:   true,
			want:     NewContiguousView([]int64{}),
		},
	}
	for _, tc := range cases {
		got, ok := tc.v.Reshape(tc.newShape)
		if ok != tc.wantOk {
			t.Errorf("%s: ok=%v want %v", tc.name, ok, tc.wantOk)
			continue
		}
		if ok {
			viewEq(t, tc.name, got, tc.want)
		}
	}
}

// ── contiguity invalidation ───────────────────────────────────────────────────

func TestContiguityInvalidation(t *testing.T) {
	v := NewContiguousView([]int64{4, 4})
	if !v.Contiguous {
		t.Fatal("fresh view should be contiguous")
	}

	// Shrink makes non-contiguous.
	vs := v.Shrink([][2]int64{i2(1, 3), i2(0, 4)})
	if vs.Contiguous {
		t.Error("shrunk view should not be contiguous")
	}

	// Pad makes non-contiguous.
	vp := v.Pad([][2]int64{i2(1, 0), i2(0, 0)})
	if vp.Contiguous {
		t.Error("padded view should not be contiguous")
	}

	// Flip makes non-contiguous (negative stride).
	vf := v.Flip([]bool{true, false})
	if vf.Contiguous {
		t.Error("flipped view should not be contiguous")
	}

	// Reshape of contiguous stays contiguous.
	vr, ok := v.Reshape([]int64{16})
	if !ok || !vr.Contiguous {
		t.Error("reshape of contiguous should be contiguous")
	}

	// Expand makes non-contiguous (stride-0).
	ve := NewContiguousView([]int64{1, 4}).Expand([]int64{3, 4})
	if ve.Contiguous {
		t.Error("expanded view should not be contiguous")
	}
}

// ── pad + shrink roundtrip ────────────────────────────────────────────────────

func TestPadShrinkRoundtrip(t *testing.T) {
	v := NewContiguousView([]int64{3, 3})
	padded := v.Pad([][2]int64{i2(1, 1), i2(1, 1)})
	// shape=(5,5)
	if !shapesEqual(padded.Shape, []int64{5, 5}) {
		t.Fatalf("pad shape got %v want [5 5]", padded.Shape)
	}
	// Shrink back to original interior.
	restored := padded.Shrink([][2]int64{i2(1, 4), i2(1, 4)})
	// shape should be (3,3), offset=stride0*1+stride1*1=3+1=4 ... actually
	// let's just check FlatIndex for a known position.
	// In the original view, element (1,1) is at flat index 1*3+1=4.
	// After pad+shrink the element at (1,1) in the restored view is:
	// offset of restored + 1*strides[0] + 1*strides[1]
	origIdx := FlatIndex(v, []int64{1, 1})
	restoredIdx := FlatIndex(restored, []int64{1, 1})
	if origIdx != restoredIdx {
		t.Errorf("pad+shrink roundtrip: index mismatch orig=%d restored=%d", origIdx, restoredIdx)
	}
}

// ── mask intersection on repeated pad ────────────────────────────────────────

func TestMaskIntersectionOnRepeatedPad(t *testing.T) {
	// Pad once: shape=(1,) → (3,), valid=(1,2)
	v := NewContiguousView([]int64{1})
	v1 := v.Pad([][2]int64{i2(1, 1)})
	if !shapesEqual(v1.Shape, []int64{3}) {
		t.Fatalf("first pad shape %v", v1.Shape)
	}
	// Pad again: shape=(3,) → (5,), valid must intersect with (1+1, 2+1)=(2,3)
	// but also with existing (shifted) mask.
	v2 := v1.Pad([][2]int64{i2(1, 1)})
	if !shapesEqual(v2.Shape, []int64{5}) {
		t.Fatalf("second pad shape %v", v2.Shape)
	}
	// The only valid element is the original single element; it must be at index 2.
	for i := int64(0); i < 5; i++ {
		valid := IsValid(v2, []int64{i})
		wantValid := i == 2
		if valid != wantValid {
			t.Errorf("pos %d: valid=%v want %v", i, valid, wantValid)
		}
	}
}

// ── FlatIndex correctness ─────────────────────────────────────────────────────

func TestFlatIndex(t *testing.T) {
	cases := []struct {
		name    string
		v       View
		indices []int64
		want    int64
	}{
		{
			name:    "simple 2D",
			v:       NewContiguousView([]int64{4, 5}),
			indices: []int64{2, 3},
			want:    2*5 + 3,
		},
		{
			name:    "with offset",
			v:       NewView([]int64{3}, []int64{1}, 7, nil),
			indices: []int64{2},
			want:    9,
		},
		{
			name:    "negative stride (flip)",
			v:       NewContiguousView([]int64{4}).Flip([]bool{true}),
			indices: []int64{0},
			want:    3, // starts at last element
		},
		{
			name:    "stride-0 (broadcast)",
			v:       NewView([]int64{3, 4}, []int64{0, 1}, 0, nil),
			indices: []int64{2, 1},
			want:    1, // row doesn't matter, only col
		},
		{
			name:    "after shrink",
			v:       NewContiguousView([]int64{6}).Shrink([][2]int64{i2(2, 5)}),
			indices: []int64{0}, // element 0 of shrunk view = element 2 of original
			want:    2,
		},
		{
			name:    "3D",
			v:       NewContiguousView([]int64{2, 3, 4}),
			indices: []int64{1, 2, 3},
			want:    1*12 + 2*4 + 3,
		},
	}
	for _, tc := range cases {
		got := FlatIndex(tc.v, tc.indices)
		if got != tc.want {
			t.Errorf("%s: FlatIndex=%d want %d", tc.name, got, tc.want)
		}
	}
}

// ── IsValid ───────────────────────────────────────────────────────────────────

func TestIsValid(t *testing.T) {
	v := NewView([]int64{4, 4}, []int64{4, 1}, 0, [][2]int64{i2(1, 3), i2(1, 3)})
	cases := []struct {
		idx   []int64
		valid bool
	}{
		{[]int64{0, 0}, false},
		{[]int64{1, 1}, true},
		{[]int64{2, 2}, true},
		{[]int64{3, 1}, false},
		{[]int64{1, 3}, false},
	}
	for _, tc := range cases {
		got := IsValid(v, tc.idx)
		if got != tc.valid {
			t.Errorf("IsValid(%v)=%v want %v", tc.idx, got, tc.valid)
		}
	}
}

// ── composed op chains ────────────────────────────────────────────────────────

func TestComposedOpChain(t *testing.T) {
	// Transpose then shrink: take upper-left 2×2 of a 3×3 transposed.
	v := NewContiguousView([]int64{3, 3})
	vt := v.Permute([]int{1, 0}) // (3,3) strides (1,3)
	vs := vt.Shrink([][2]int64{i2(0, 2), i2(0, 2)})

	// Element (0,0): offset=0, idx=0*1+0*3=0
	// Element (1,0): offset=0, idx=1*1+0*3=1
	// Element (0,1): offset=0, idx=0*1+1*3=3
	// Element (1,1): offset=0, idx=1*1+1*3=4
	idxs := []struct {
		i, j int64
		want int64
	}{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 3},
		{1, 1, 4},
	}
	for _, x := range idxs {
		got := FlatIndex(vs, []int64{x.i, x.j})
		if got != x.want {
			t.Errorf("transpose+shrink (%d,%d)=%d want %d", x.i, x.j, got, x.want)
		}
	}
}

func TestFlipThenShrink(t *testing.T) {
	// Flip a 1D view then shrink to first 3 elements (which are the LAST 3 of original).
	v := NewContiguousView([]int64{6})
	vf := v.Flip([]bool{true}) // offset=5, stride=-1
	// Elements 0..2 of flipped = original 5,4,3
	vs := vf.Shrink([][2]int64{i2(0, 3)})
	for i := int64(0); i < 3; i++ {
		got := FlatIndex(vs, []int64{i})
		want := int64(5) - i
		if got != want {
			t.Errorf("flip+shrink idx %d=%d want %d", i, got, want)
		}
	}
}

// ── edge cases ────────────────────────────────────────────────────────────────

func TestZeroSizeDim(t *testing.T) {
	v := NewContiguousView([]int64{0, 5})
	if !v.Contiguous {
		t.Error("zero-size contiguous view should be contiguous")
	}
	// Reshape with zero-size: any new shape with same product is valid.
	v2, ok := v.Reshape([]int64{0})
	if !ok {
		t.Fatal("zero-size reshape should succeed")
	}
	if !shapesEqual(v2.Shape, []int64{0}) {
		t.Errorf("got shape %v", v2.Shape)
	}
}

func TestSingleElementView(t *testing.T) {
	v := NewContiguousView([]int64{1, 1, 1})
	if !v.Contiguous {
		t.Error("single-element view should be contiguous")
	}
	// strides should be all zero (size-1 dims).
	for i, st := range v.Strides {
		if st != 0 {
			t.Errorf("stride[%d]=%d want 0", i, st)
		}
	}
	got := FlatIndex(v, []int64{0, 0, 0})
	if got != 0 {
		t.Errorf("FlatIndex single-elem=%d want 0", got)
	}
}

func TestExpandBroadcastStride0(t *testing.T) {
	// Verify that expanded dims always have stride 0.
	v := NewContiguousView([]int64{1, 3, 1})
	ve := v.Expand([]int64{4, 3, 7})
	for i, st := range ve.Strides {
		wantZero := (v.Shape[i] == 1 && ve.Shape[i] != 1)
		if wantZero && st != 0 {
			t.Errorf("expanded dim %d: stride=%d want 0", i, st)
		}
	}
}

func TestReshapePreservesOffset(t *testing.T) {
	// Non-contiguous view with offset; reshape should carry the offset through.
	v := NewView([]int64{4}, []int64{1}, 10, nil)
	// Contiguous (since strides=stridesForShape and mask=nil) BUT offset≠0 → not contiguous.
	// Actually: contiguous requires offset==0. So this view is not contiguous.
	// Reshape to (2,2):
	got, ok := v.Reshape([]int64{2, 2})
	if !ok {
		t.Fatal("reshape should succeed on non-masked non-contiguous (only offset≠0)")
	}
	// The extra_offset from mask=nil is 0. So new offset = 10.
	if got.Offset != 10 {
		t.Errorf("offset=%d want 10", got.Offset)
	}
}

// ── reshapeMask unit tests ────────────────────────────────────────────────────

func TestReshapeMask(t *testing.T) {
	cases := []struct {
		name     string
		mask     [][2]int64
		shape    []int64
		newShape []int64
		wantOk   bool
		wantMask [][2]int64 // nil means no mask expected
	}{
		{
			name:     "nil mask always succeeds",
			mask:     nil,
			shape:    []int64{6},
			newShape: []int64{2, 3},
			wantOk:   true,
			wantMask: nil,
		},
		{
			name:     "full-range mask treated as nil",
			mask:     [][2]int64{i2(0, 6)},
			shape:    []int64{6},
			newShape: []int64{2, 3},
			wantOk:   true,
			wantMask: nil,
		},
		{
			name:     "aligned lower half",
			mask:     [][2]int64{i2(0, 3)},
			shape:    []int64{6},
			newShape: []int64{2, 3},
			wantOk:   true,
			wantMask: [][2]int64{i2(0, 1), i2(0, 3)},
		},
		{
			name:     "aligned upper half",
			mask:     [][2]int64{i2(3, 6)},
			shape:    []int64{6},
			newShape: []int64{2, 3},
			wantOk:   true,
			wantMask: [][2]int64{i2(1, 2), i2(0, 3)},
		},
		{
			name:     "cross-boundary mask fails",
			mask:     [][2]int64{i2(2, 4)},
			shape:    []int64{6},
			newShape: []int64{2, 3},
			wantOk:   false,
		},
		{
			name:     "merge 2D box to 1D",
			mask:     [][2]int64{i2(1, 3), i2(0, 3)},
			shape:    []int64{4, 3},
			newShape: []int64{12},
			wantOk:   true,
			wantMask: [][2]int64{i2(3, 9)},
		},
		{
			name:     "partial 2D box to 1D fails",
			mask:     [][2]int64{i2(1, 3), i2(1, 2)},
			shape:    []int64{4, 3},
			newShape: []int64{12},
			wantOk:   false,
		},
		{
			name:     "reshape mask across singleton dim",
			mask:     [][2]int64{i2(0, 1), i2(2, 5)},
			shape:    []int64{1, 6},
			newShape: []int64{6},
			wantOk:   true,
			wantMask: [][2]int64{i2(2, 5)},
		},
	}
	for _, tc := range cases {
		got, ok := reshapeMask(tc.mask, tc.shape, tc.newShape)
		if ok != tc.wantOk {
			t.Errorf("%s: ok=%v want %v", tc.name, ok, tc.wantOk)
			continue
		}
		if ok && !maskEq(got, tc.wantMask) {
			t.Errorf("%s: mask=%v want %v", tc.name, got, tc.wantMask)
		}
	}
}
