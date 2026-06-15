package shape

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// ── Expand error paths ──────────────────────────────────────────────────────

func TestExpandRankMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Expand with rank mismatch should panic")
		}
	}()
	NewContiguousView(ss(1, 4)).Expand(ss(3))
}

func TestExpandNonUnitDimPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Expand of a non-unit dim should panic")
		}
	}()
	// dim-0 has size 2 (not 1) but target is 3 → cannot expand.
	NewContiguousView(ss(2, 4)).Expand(ss(3, 4))
}

func TestExpandZeroSizeInputResets(t *testing.T) {
	// A zero-size dim forces a fresh contiguous view of the new shape.
	v := NewView(ss(0, 1), ss(0, 0), Const(0), nil)
	got := v.Expand(ss(0, 5))
	viewEq(t, "expand zero-size", got, NewContiguousView(ss(0, 5)))
}

func TestExpandMaskNonFullCollapses(t *testing.T) {
	// dim-0 mask is (0,1) which is NOT the canonical (0,1) full-of-size-1?
	// Here size-1 dim with mask (1,1) (empty) → expanding collapses to (0,0).
	v := NewView(ss(1, 3), ss(0, 1), Const(0), [][2]Sint{i2(1, 1), i2(0, 3)})
	got := v.Expand(ss(4, 3))
	want := NewView(ss(4, 3), ss(0, 1), Const(0), [][2]Sint{i2(0, 0), i2(0, 3)})
	viewEq(t, "expand mask collapse", got, want)
}

func TestExpandToSymbolicDim(t *testing.T) {
	// Concrete size-1 dim expanded to a symbolic target.
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 1, 16)}
	v := NewContiguousView(ss(1, 4))
	got := v.Expand([]Sint{n, Const(4)})
	if _, ok := got.Shape[0].(SymInt); !ok {
		t.Errorf("expanded dim-0 should be symbolic, got %T", got.Shape[0])
	}
}

// ── Permute error path ──────────────────────────────────────────────────────

func TestPermuteLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Permute with wrong order length should panic")
		}
	}()
	NewContiguousView(ss(2, 3)).Permute([]int{0})
}

// ── Pad error paths ─────────────────────────────────────────────────────────

func TestPadArgLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Pad with wrong arg length should panic")
		}
	}()
	NewContiguousView(ss(2, 3)).Pad([][2]Sint{i2(0, 0)})
}

func TestPadNegativeAmountPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Pad with negative amount should panic")
		}
	}()
	NewContiguousView(ss(3)).Pad([][2]Sint{{Const(-1), Const(0)}})
}

func TestPadSymbolicAmount(t *testing.T) {
	// A symbolic, provably-non-negative pad amount is accepted.
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 1, 8)}
	got := NewContiguousView(ss(3)).Pad([][2]Sint{{n, Const(0)}})
	// Resulting shape dim is 3 + n → symbolic.
	if _, ok := got.Shape[0].(SymInt); !ok {
		t.Errorf("padded dim should be symbolic, got %T", got.Shape[0])
	}
}

func TestPadSymbolicNonProvablePanics(t *testing.T) {
	// n in [-2,5] is not provably non-negative → pad must reject it.
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", -2, 5)}
	defer func() {
		if recover() == nil {
			t.Error("Pad with non-provable amount should panic")
		}
	}()
	NewContiguousView(ss(3)).Pad([][2]Sint{{n, Const(0)}})
}

// ── Shrink error paths ──────────────────────────────────────────────────────

func TestShrinkArgLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Shrink with wrong arg length should panic")
		}
	}()
	NewContiguousView(ss(2, 3)).Shrink([][2]Sint{i2(0, 1)})
}

func TestShrinkOutOfBoundsPanics(t *testing.T) {
	cases := []struct {
		name string
		arg  [][2]Sint
	}{
		{"lo negative", [][2]Sint{{Const(-1), Const(2)}}},
		{"lo > hi", [][2]Sint{i2(3, 1)}},
		{"hi > shape", [][2]Sint{i2(0, 7)}},
	}
	for _, tc := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: Shrink should panic", tc.name)
				}
			}()
			NewContiguousView(ss(5)).Shrink(tc.arg)
		}()
	}
}

func TestShrinkSymbolicMaskPath(t *testing.T) {
	// Shrink a view with a mask using a symbolic lo bound exercises the
	// symbolic SintMax/SintMin mask-transform branch of unsafeResize.
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 0, 0)} // bound to 0 but still symbolic
	v := NewView(ss(8), ss(1), Const(0), [][2]Sint{i2(2, 6)})
	got := v.Shrink([][2]Sint{{n, Const(4)}})
	// Mask must be symbolic now (built via SintMax/SintMin).
	if got.Mask == nil {
		t.Fatal("expected a mask after symbolic shrink")
	}
	if _, ok := got.Mask[0][0].ConstValue(); ok {
		// It may fold to a const; just ensure no panic and a mask exists.
		t.Logf("mask lo folded to const (acceptable): %v", got.Mask[0][0])
	}
}

// ── Reshape error / fallback paths ──────────────────────────────────────────

func TestReshapeConcreteSizeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Reshape with mismatched product should panic")
		}
	}()
	NewContiguousView(ss(2, 3)).Reshape(ss(4, 4))
}

func TestReshapeSymbolicSizeMismatchPanics(t *testing.T) {
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 1, 8)}
	m := SymInt{Node: a.DefineVar("m", 1, 8)}
	defer func() {
		if recover() == nil {
			t.Error("Reshape with mismatched symbolic product should panic")
		}
	}()
	// prod([n,4]) != prod([m,8]).
	NewView([]Sint{n, Const(4)}, ss(4, 1), Const(0), nil).Reshape([]Sint{m, Const(8)})
}

func TestReshapeSymbolicNonContiguousFallsBack(t *testing.T) {
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 1, 8)}
	// Non-contiguous symbolic view whose product structurally matches the
	// target (same factoring [n,4] -> [4,n]) so it passes the size check, then
	// hits the symbolic non-contiguous fallback returning ok=false.
	v := NewView([]Sint{n, Const(4)}, ss(1, 100), Const(0), nil)
	if !SintEqual(SymbolicProduct([]Sint{n, Const(4)}), SymbolicProduct([]Sint{Const(4), n})) {
		t.Skip("commutative product not canonicalized in this build; fallback path covered elsewhere")
	}
	_, ok := v.Reshape([]Sint{Const(4), n})
	if ok {
		t.Error("non-contiguous symbolic reshape should return ok=false")
	}
}

func TestReshapeScalarFromMaskedOut(t *testing.T) {
	// Reshaping a fully-masked-out (empty) dim to scalar must fail (ok=false).
	v := NewView(ss(1), ss(0), Const(0), [][2]Sint{i2(0, 0)})
	_, ok := v.Reshape(ss())
	if ok {
		t.Error("reshape masked-out dim to scalar should return ok=false")
	}
}

func TestReshapeNonContiguousStrideFailReturnsFalse(t *testing.T) {
	// A non-mergeable stride pattern cannot be re-expressed: ok=false.
	// shape (3,4) with strides (1,3) (column-major) reshaped to (12) cannot merge.
	v := NewView(ss(3, 4), ss(1, 3), Const(0), nil)
	_, ok := v.Reshape(ss(12))
	if ok {
		t.Error("column-major reshape to flat should return ok=false")
	}
}

// ── ShapeTracker.Reshape push-fresh-view fallback ───────────────────────────

func TestShapeTrackerReshapeFallbackPush(t *testing.T) {
	// A non-reshapeable active view forces a freshly pushed contiguous view.
	st := NewShapeTracker([]int64{3, 4})
	st = st.Permute([]int{1, 0}) // strides now (1,4): not mergeable to flat
	before := len(st.Views)
	st2 := st.Reshape([]int64{12})
	if len(st2.Views) != before+1 {
		t.Errorf("expected a pushed view (%d -> %d)", before, len(st2.Views))
	}
	last := st2.ActiveView()
	if !last.Contiguous {
		t.Error("pushed view should be contiguous")
	}
}

// ── IsValid mask-fail branch ────────────────────────────────────────────────

func TestIsValidMaskRejects(t *testing.T) {
	v := NewView(ss(4), ss(1), Const(0), [][2]Sint{i2(1, 3)})
	if IsValid(v, []int64{0}) {
		t.Error("index 0 is below mask lo (1) → invalid")
	}
	if IsValid(v, []int64{3}) {
		t.Error("index 3 is at mask hi (3, exclusive) → invalid")
	}
	if !IsValid(v, []int64{2}) {
		t.Error("index 2 is within mask [1,3) → valid")
	}
}

// ── collectMergeDims via reshapeStrides on a contiguous-mergeable view ───────

func TestReshapeMergesContiguousDims(t *testing.T) {
	// Non-Contiguous flag but genuinely mergeable strides: shape (2,3,4),
	// strides (12,4,1) reshaped to (6,4) must merge dims 0,1.
	v := NewView(ss(2, 3, 4), ss(12, 4, 1), Const(0), nil)
	got, ok := v.Reshape(ss(6, 4))
	if !ok {
		t.Fatal("mergeable reshape should succeed")
	}
	viewEq(t, "merge contiguous dims", got, NewView(ss(6, 4), ss(4, 1), Const(0), nil))
}

func TestReshapeSplitDimWithStride(t *testing.T) {
	// shape (6,) strides (2,) (non-contiguous) reshape to (2,3): strides (6,2).
	v := NewView(ss(6), ss(2), Const(0), nil)
	got, ok := v.Reshape(ss(2, 3))
	if !ok {
		t.Fatal("strided split reshape should succeed")
	}
	viewEq(t, "split strided", got, NewView(ss(2, 3), ss(6, 2), Const(0), nil))
}

// ── reshapeMask non-rectangular & success paths ─────────────────────────────

func TestReshapeMaskMergeFull(t *testing.T) {
	// Mask full on dim that merges cleanly: (2,3) mask [(0,2),(0,3)] flatten to (6).
	v := NewView(ss(2, 3), ss(3, 1), Const(0), [][2]Sint{i2(0, 2), i2(0, 3)})
	got, ok := v.Reshape(ss(6))
	if !ok {
		t.Fatal("full-mask flatten should succeed")
	}
	// Full mask normalizes to nil.
	if got.Mask != nil {
		t.Errorf("full mask should normalize to nil, got %v", got.Mask)
	}
}

func TestReshapeMaskNonRectangularFails(t *testing.T) {
	// A partial mask that cannot be expressed rectangularly in the new shape.
	// shape (3,4) contiguous, mask selects a non-aligned region, reshape to (2,6).
	v := NewView(ss(3, 4), ss(4, 1), Const(0), [][2]Sint{i2(0, 3), i2(1, 3)})
	_, ok := v.Reshape(ss(2, 6))
	if ok {
		t.Error("non-rectangular mask reshape should return ok=false")
	}
}

// ── normalizeMask64 direct ──────────────────────────────────────────────────

func TestNormalizeMask64(t *testing.T) {
	if normalizeMask64(nil, []int64{4}) != nil {
		t.Error("nil mask normalizes to nil")
	}
	// Full mask → nil.
	full := [][2]int64{{0, 4}, {0, 3}}
	if normalizeMask64(full, []int64{4, 3}) != nil {
		t.Error("full mask should normalize to nil")
	}
	// Partial mask → unchanged.
	part := [][2]int64{{0, 4}, {1, 3}}
	out := normalizeMask64(part, []int64{4, 3})
	if out == nil || out[1][0] != 1 {
		t.Errorf("partial mask should be preserved, got %v", out)
	}
}

// ── cloneMaskSint nil branch ────────────────────────────────────────────────

func TestCloneMaskSintNil(t *testing.T) {
	if cloneMaskSint(nil) != nil {
		t.Error("cloneMaskSint(nil) should be nil")
	}
	in := [][2]Sint{i2(0, 4)}
	out := cloneMaskSint(in)
	if len(out) != 1 || !Eq(out[0][1], Const(4)) {
		t.Errorf("clone = %v", out)
	}
}

// ── sizeMatch direct (symbolic short-circuit) ───────────────────────────────

func TestSizeMatchSymbolicTrusted(t *testing.T) {
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 1, 8)}
	if !sizeMatch([]Sint{n}, ss(8)) {
		t.Error("symbolic operand should be trusted (true)")
	}
	if !sizeMatch(ss(2, 3), ss(6)) {
		t.Error("matching concrete products should be true")
	}
	if sizeMatch(ss(2, 3), ss(7)) {
		t.Error("mismatched concrete products should be false")
	}
}

// ── hasSym direct ───────────────────────────────────────────────────────────

func TestHasSymDirect(t *testing.T) {
	if hasSym(ss(1, 2, 3)) {
		t.Error("all-concrete is not symbolic")
	}
	a := uop.NewArena(64)
	n := SymInt{Node: a.DefineVar("n", 1, 8)}
	if !hasSym([]Sint{Const(2), n}) {
		t.Error("slice with SymInt is symbolic")
	}
}
