package rules

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// White-box tests for the symbolic rule handlers' guard branches. The capture
// map is normally populated by the matcher before a handler runs, but the
// handlers defensively return ok=false when the expected capture is absent.
// These tests pin that contract (the false path is otherwise unreachable
// through the public ruleset).

func TestReturnHandlersMissingCapture(t *testing.T) {
	empty := map[string]uop.UOp{}
	handlers := []struct {
		name string
		fn   func(map[string]uop.UOp, any) (uop.UOp, bool)
	}{
		{"hReturnX", hReturnX},
		{"hReturnV", hReturnV},
		{"hReturnA", hReturnA},
		{"hReturnB", hReturnB},
		{"hReturnBase", hReturnBase},
	}
	for _, h := range handlers {
		if _, ok := h.fn(empty, nil); ok {
			t.Errorf("%s with empty captures returned ok=true, want false", h.name)
		}
	}
}

func TestReturnHandlersPresentCapture(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	cases := []struct {
		key string
		fn  func(map[string]uop.UOp, any) (uop.UOp, bool)
	}{
		{"x", hReturnX},
		{"v", hReturnV},
		{"a", hReturnA},
		{"b", hReturnB},
		{"base", hReturnBase},
	}
	for _, tc := range cases {
		got, ok := tc.fn(map[string]uop.UOp{tc.key: x}, nil)
		if !ok || got != x {
			t.Errorf("handler[%s]: got (%v,%v), want (x,true)", tc.key, got, ok)
		}
	}
}

// hMulZero / hAndZeroInt only fold for integer/bool dtypes; the float case must
// decline (returns ok=false) so float x*0 is not constant-folded to int 0.

func TestMulZeroDtypeGating(t *testing.T) {
	a := uop.NewArena(8)
	// Integer: folds to 0.
	xi := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(3), nil)
	zi := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(0), nil)
	muli := a.New(uop.OpMul, uop.Dtypes.Int32, []uop.UOp{xi, zi}, nil, nil)
	if r, ok := hMulZero(map[string]uop.UOp{"node": muli}, nil); !ok || r.Arg().(int64) != 0 {
		t.Errorf("hMulZero int: got (%v,%v), want Const 0", r, ok)
	}
	// Float: declines.
	xf := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(3), nil)
	zf := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(0), nil)
	mulf := a.New(uop.OpMul, uop.Dtypes.Float32, []uop.UOp{xf, zf}, nil, nil)
	if _, ok := hMulZero(map[string]uop.UOp{"node": mulf}, nil); ok {
		t.Error("hMulZero float: returned ok=true, want false (float 0 not foldable to int)")
	}
}

func TestAndZeroIntDtypeGating(t *testing.T) {
	a := uop.NewArena(8)
	xi := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(5), nil)
	andi := a.New(uop.OpAnd, uop.Dtypes.Int32, []uop.UOp{xi, xi}, nil, nil)
	if r, ok := hAndZeroInt(map[string]uop.UOp{"node": andi}, nil); !ok || r.Arg().(int64) != 0 {
		t.Errorf("hAndZeroInt int: got (%v,%v), want Const 0", r, ok)
	}
	// Bool dtype is not Int → declines.
	xb := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	andb := a.New(uop.OpAnd, uop.Dtypes.Bool, []uop.UOp{xb, xb}, nil, nil)
	if _, ok := hAndZeroInt(map[string]uop.UOp{"node": andb}, nil); ok {
		t.Error("hAndZeroInt bool: returned ok=true, want false")
	}
}

// hIdentityCast returns x only when root and x share a dtype; otherwise declines.
func TestIdentityCastDtypeGating(t *testing.T) {
	a := uop.NewArena(8)
	x := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(1), nil)
	// Same dtype → identity.
	same := a.New(uop.OpCast, uop.Dtypes.Int32, []uop.UOp{x}, nil, nil)
	if r, ok := hIdentityCast(map[string]uop.UOp{"root": same, "x": x}, nil); !ok || r != x {
		t.Errorf("hIdentityCast same dtype: got (%v,%v), want (x,true)", r, ok)
	}
	// Different dtype → declines.
	diff := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{x}, nil, nil)
	if _, ok := hIdentityCast(map[string]uop.UOp{"root": diff, "x": x}, nil); ok {
		t.Error("hIdentityCast diff dtype: returned ok=true, want false")
	}
}
