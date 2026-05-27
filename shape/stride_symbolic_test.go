package shape_test

import (
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// TestStridesForShapeSymbolic proves that stridesForShape produces concrete strides
// for the symbolic-batch shapes used in Option A (batch always outermost).
//
// Invariants tested:
//   - [n, 4]   → strides [Const(4), Const(1)]   (all concrete; no symbolic stride)
//   - [n, 4, 8] → strides [Const(32), Const(8), Const(1)]
//   - [4, n]   → strides [SymInt(n), Const(1)]  (outermost stride is symbolic when n is not outermost)
func TestStridesForShapeSymbolic(t *testing.T) {
	a := uop.NewArena(64)
	defVar := a.DefineVar("n", 1, 1024)
	sym := shape.SymInt{Node: defVar}

	t.Run("n_4", func(t *testing.T) {
		sh := []shape.Sint{sym, shape.Const(4)}
		strides := shape.StridesForShape(sh)
		if len(strides) != 2 {
			t.Fatalf("len(strides) = %d, want 2", len(strides))
		}
		// stride[0] = 4 (concrete: product of trailing concrete dims)
		if v, ok := strides[0].ConstValue(); !ok || v != 4 {
			t.Errorf("strides[0] = %v (isConc=%v), want Const(4)", strides[0], ok)
		}
		// stride[1] = 1 (last dim is always 1)
		if v, ok := strides[1].ConstValue(); !ok || v != 1 {
			t.Errorf("strides[1] = %v (isConc=%v), want Const(1)", strides[1], ok)
		}
		t.Logf("[n, 4] strides: [Const(%v), Const(%v)] — all concrete ✓", mustConst(strides[0]), mustConst(strides[1]))
	})

	t.Run("n_4_8", func(t *testing.T) {
		sh := []shape.Sint{sym, shape.Const(4), shape.Const(8)}
		strides := shape.StridesForShape(sh)
		if len(strides) != 3 {
			t.Fatalf("len(strides) = %d, want 3", len(strides))
		}
		if v, ok := strides[0].ConstValue(); !ok || v != 32 {
			t.Errorf("strides[0] = %v, want Const(32)", strides[0])
		}
		if v, ok := strides[1].ConstValue(); !ok || v != 8 {
			t.Errorf("strides[1] = %v, want Const(8)", strides[1])
		}
		if v, ok := strides[2].ConstValue(); !ok || v != 1 {
			t.Errorf("strides[2] = %v, want Const(1)", strides[2])
		}
		t.Logf("[n, 4, 8] strides: [Const(32), Const(8), Const(1)] — all concrete ✓")
	})

	t.Run("4_n", func(t *testing.T) {
		// [4, n]: stride[0] = n (symbolic), stride[1] = 1 (concrete).
		sh := []shape.Sint{shape.Const(4), sym}
		strides := shape.StridesForShape(sh)
		if len(strides) != 2 {
			t.Fatalf("len(strides) = %d, want 2", len(strides))
		}
		// stride[0] should be symbolic (= n)
		if _, ok := strides[0].ConstValue(); ok {
			t.Errorf("strides[0] = Const(%v), want SymInt(n)", strides[0])
		}
		if _, ok := strides[0].(shape.SymInt); !ok {
			t.Errorf("strides[0] type = %T, want shape.SymInt", strides[0])
		}
		// stride[1] = 1
		if v, ok := strides[1].ConstValue(); !ok || v != 1 {
			t.Errorf("strides[1] = %v, want Const(1)", strides[1])
		}
		t.Logf("[4, n] strides: [SymInt(n), Const(1)] — inner stride is symbolic ✓")
	})

	t.Run("contiguous_view_n_4", func(t *testing.T) {
		// NewContiguousView([n, 4]) must be Contiguous=true (strides match expected).
		sh := []shape.Sint{sym, shape.Const(4)}
		v := shape.NewContiguousView(sh)
		if !v.Contiguous {
			t.Errorf("NewContiguousView([n, 4]).Contiguous = false, want true")
		}
		t.Logf("NewContiguousView([n, 4]).Contiguous = true ✓")
	})
}

func mustConst(s shape.Sint) int64 {
	v, ok := s.ConstValue()
	if !ok {
		panic("mustConst: not concrete")
	}
	return v
}
