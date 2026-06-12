package cpu

// Unit tests driving evalFloat/evalInt directly over hand-built UOp nodes.
// Covers the Slice 2 comparison restructure (int-operand dispatch + the
// float fallback arms, including OpCmpNe which no tensor-level op emits),
// the OpAnd conjunction in both evaluators, the OpGatherIdx delegation,
// and the fail-loud unimplemented-op contract.

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

func newEvalState() *state {
	return &state{
		rangeVal:    make(map[int]int64),
		paramShapes: make(map[int][]int64),
	}
}

func TestEvalFloatComparisons(t *testing.T) {
	a := uop.NewArena(256)
	cf := func(v float64) uop.UOp { return a.New(uop.OpConst, uop.Dtypes.Float32, nil, v, nil) }
	ci := func(v int64) uop.UOp { return a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil) }
	st := newEvalState()

	evalF := func(t *testing.T, u uop.UOp) float32 {
		t.Helper()
		v, err := st.evalFloat(u)
		if err != nil {
			t.Fatalf("evalFloat: %v", err)
		}
		return v
	}

	// Float fallback arms (operand dtype = f32).
	floatCases := []struct {
		op   uop.Op
		a, b float64
		want float32
	}{
		{uop.OpCmpLt, 1, 2, 1},
		{uop.OpCmpLt, 2, 1, 0},
		{uop.OpCmpNe, 1, 2, 1},
		{uop.OpCmpNe, 2, 2, 0},
		{uop.OpCmpEq, 2, 2, 1},
		{uop.OpCmpEq, 1, 2, 0},
	}
	for _, tc := range floatCases {
		n := a.New(tc.op, uop.Dtypes.Bool, []uop.UOp{cf(tc.a), cf(tc.b)}, nil, nil)
		if got := evalF(t, n); got != tc.want {
			t.Errorf("float %v(%v,%v) = %v, want %v", tc.op, tc.a, tc.b, got, tc.want)
		}
	}

	// Int-operand dispatch (operand dtype = Index → evalInt path; float32
	// would lose exactness above 2^24, so verify with a value pair that
	// float32 cannot distinguish).
	big, bigPlus1 := int64(1<<24+1), int64(1<<24+2) // adjacent, both round to same f32
	n := a.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{ci(big), ci(bigPlus1)}, nil, nil)
	if got := evalF(t, n); got != 1 {
		t.Errorf("int CmpNe(2^24+1, 2^24+2) = %v, want 1 (float32 would collapse the pair)", got)
	}
	n = a.New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{ci(big), ci(big)}, nil, nil)
	if got := evalF(t, n); got != 1 {
		t.Errorf("int CmpEq same = %v, want 1", got)
	}
	n = a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{ci(big), ci(bigPlus1)}, nil, nil)
	if got := evalF(t, n); got != 1 {
		t.Errorf("int CmpLt = %v, want 1", got)
	}

	// OpAnd over 0/1 bools, float side.
	b0 := a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{cf(2), cf(1)}, nil, nil) // 0
	b1 := a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{cf(1), cf(2)}, nil, nil) // 1
	and := func(x, y uop.UOp) uop.UOp {
		return a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{x, y}, nil, nil)
	}
	if got := evalF(t, and(b1, b1)); got != 1 {
		t.Errorf("And(1,1) = %v", got)
	}
	if got := evalF(t, and(b1, b0)); got != 0 {
		t.Errorf("And(1,0) = %v", got)
	}
}

func TestEvalIntAndGatherIdx(t *testing.T) {
	a := uop.NewArena(256)
	ci := func(v int64) uop.UOp { return a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil) }
	st := newEvalState()

	evalI := func(t *testing.T, u uop.UOp) int64 {
		t.Helper()
		v, err := st.evalInt(u)
		if err != nil {
			t.Fatalf("evalInt: %v", err)
		}
		return v
	}

	// And over int-side 0/1.
	one := a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{ci(1), ci(2)}, nil, nil)
	zero := a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{ci(2), ci(1)}, nil, nil)
	and := a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{one, zero}, nil, nil)
	if got := evalI(t, and); got != 0 {
		t.Errorf("int And(1,0) = %d", got)
	}
	and2 := a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{one, one}, nil, nil)
	if got := evalI(t, and2); got != 1 {
		t.Errorf("int And(1,1) = %d", got)
	}

	// CmpNe / CmpEq int arms.
	ne := a.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{ci(3), ci(4)}, nil, nil)
	if got := evalI(t, ne); got != 1 {
		t.Errorf("int CmpNe(3,4) = %d", got)
	}
	eq := a.New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{ci(3), ci(3)}, nil, nil)
	if got := evalI(t, eq); got != 1 {
		t.Errorf("int CmpEq(3,3) = %d", got)
	}

	// OpGatherIdx delegates to Src(0) (contract: the inner node is the real
	// index load; positional carriers in Src(1:) are schedule-level only).
	gi := a.New(uop.OpGatherIdx, uop.Dtypes.Index, []uop.UOp{ci(7)}, nil, nil)
	if got := evalI(t, gi); got != 7 {
		t.Errorf("GatherIdx delegate = %d, want 7", got)
	}
}

// TestEvalErrorPropagation drives the error arms of the new comparison /
// conjunction cases: an unimplemented operand (Sin / Xor) must surface
// through OpAnd and the int-dispatch comparisons, in either operand slot.
func TestEvalErrorPropagation(t *testing.T) {
	a := uop.NewArena(256)
	cf := a.New(uop.OpConst, uop.Dtypes.Float32, nil, 1.0, nil)
	ci := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	sin := a.New(uop.OpSin, uop.Dtypes.Float32, []uop.UOp{cf}, nil, nil)
	xor := a.New(uop.OpXor, uop.Dtypes.Index, []uop.UOp{ci, ci}, nil, nil)
	st := newEvalState()

	mustErrF := func(u uop.UOp, label string) {
		t.Helper()
		if _, err := st.evalFloat(u); err == nil {
			t.Errorf("%s: expected error", label)
		}
	}
	mustErrI := func(u uop.UOp, label string) {
		t.Helper()
		if _, err := st.evalInt(u); err == nil {
			t.Errorf("%s: expected error", label)
		}
	}

	mustErrF(a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{sin, cf}, nil, nil), "float And lhs")
	mustErrF(a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{cf, sin}, nil, nil), "float And rhs")
	mustErrF(a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{sin, cf}, nil, nil), "float cmp lhs")
	mustErrF(a.New(uop.OpCmpLt, uop.Dtypes.Bool, []uop.UOp{cf, sin}, nil, nil), "float cmp rhs")
	mustErrF(a.New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{xor, ci}, nil, nil), "int-dispatch cmp lhs")
	mustErrF(a.New(uop.OpCmpEq, uop.Dtypes.Bool, []uop.UOp{ci, xor}, nil, nil), "int-dispatch cmp rhs")
	mustErrI(a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{xor, ci}, nil, nil), "int And lhs")
	mustErrI(a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{ci, xor}, nil, nil), "int And rhs")
	mustErrI(a.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{xor, ci}, nil, nil), "int cmp lhs")
	mustErrI(a.New(uop.OpCmpNe, uop.Dtypes.Bool, []uop.UOp{ci, xor}, nil, nil), "int cmp rhs")
}

func TestEvalUnimplementedOpFailsLoud(t *testing.T) {
	a := uop.NewArena(256)
	cf := a.New(uop.OpConst, uop.Dtypes.Float32, nil, 1.0, nil)
	st := newEvalState()

	sin := a.New(uop.OpSin, uop.Dtypes.Float32, []uop.UOp{cf}, nil, nil)
	if _, err := st.evalFloat(sin); err == nil {
		t.Error("evalFloat(Sin) must fail loud (not yet implemented)")
	}

	ci := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	xor := a.New(uop.OpXor, uop.Dtypes.Index, []uop.UOp{ci, ci}, nil, nil)
	if _, err := st.evalInt(xor); err == nil {
		t.Error("evalInt(Xor) must fail loud (not yet implemented)")
	}
}
