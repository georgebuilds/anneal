package cpu

// Edge-case white-box tests for the CPU interpreter: error paths in interpret,
// evalInt, evalIntIndex, evalIndexLoadFloat, evalReduce, the allocator, and
// allocateBuffers. All drive the internals directly with hand-built AST.

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// ── allocator error / edge paths ────────────────────────────────────────────

func TestNewBufferNegativeElems(t *testing.T) {
	if _, err := newBuffer(-1, uop.Dtypes.Float32); err == nil {
		t.Error("negative elems should error")
	}
}

func TestNewBufferZeroElemsFloors(t *testing.T) {
	// Zero elems is floored to 1 (symbolic-leaf contract).
	b, err := newBuffer(0, uop.Dtypes.Float32)
	if err != nil {
		t.Fatalf("zero elems: %v", err)
	}
	if got := len(b.asF32()); got != 1 {
		t.Errorf("zero-elems buffer len = %d, want 1", got)
	}
}

func TestNewBufferUnsupportedDtype(t *testing.T) {
	if _, err := newBuffer(4, uop.Dtypes.Bool); err == nil {
		t.Error("Bool dtype should be unsupported on CPU backend")
	}
}

func TestAllocSlotReusesBuffer(t *testing.T) {
	a := newAllocator()
	defer a.Reset()
	b1, err := a.AllocSlot(7, 16, uop.Dtypes.Float32, "")
	if err != nil {
		t.Fatalf("first AllocSlot: %v", err)
	}
	// Second call with different size must return the SAME buffer.
	b2, err := a.AllocSlot(7, 999, uop.Dtypes.Float32, "")
	if err != nil {
		t.Fatalf("second AllocSlot: %v", err)
	}
	if b1 != b2 {
		t.Error("AllocSlot should return the cached buffer for a known slot")
	}
}

func TestAllocSlotNegativeSizeErrors(t *testing.T) {
	a := newAllocator()
	defer a.Reset()
	if _, err := a.AllocSlot(1, -5, uop.Dtypes.Float32, ""); err == nil {
		t.Error("AllocSlot with negative size should error")
	}
}

func TestResetIdempotent(t *testing.T) {
	a := newAllocator()
	if _, err := a.AllocSlot(1, 4, uop.Dtypes.Float32, ""); err != nil {
		t.Fatal(err)
	}
	a.Reset()
	a.Reset() // second reset must not panic.
	if len(a.all) != 0 || len(a.slots) != 0 {
		t.Error("Reset should clear all allocations")
	}
}

// ── interpret: range-collection error arms ──────────────────────────────────

// mkSink builds a SINK(END(STORE(Index(Param0,...), body), ranges...)).
func mkStoreKernel(a *uop.Arena, body uop.UOp, outIdx uop.UOp, ranges ...uop.UOp) uop.UOp {
	store := a.New(uop.OpStore, uop.Dtypes.Float32, []uop.UOp{outIdx, body}, nil, nil)
	endSrcs := append([]uop.UOp{store}, ranges...)
	end := a.New(uop.OpEnd, uop.Dtypes.Float32, endSrcs, nil, nil)
	return a.New(uop.OpSink, uop.Dtypes.Float32, []uop.UOp{end}, nil, nil)
}

func TestInterpretBadRangeOp(t *testing.T) {
	a := uop.NewArena(256)
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	// A non-Const, non-Range src in End.Src[1:] is rejected.
	bogus := a.New(uop.OpSqrt, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx, bogus)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Float32, Slot: -1}}}
	if err := interpret(item, map[uint32]*Buffer{}); err == nil {
		t.Error("non-Range/Const End src should error")
	}
}

func TestInterpretUnknownAxisType(t *testing.T) {
	a := uop.NewArena(256)
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	// A Range with an out-of-enum axis type.
	rng := a.New(uop.OpRange, uop.Dtypes.Index,
		[]uop.UOp{ci(a, 0), ci(a, 4)},
		uop.RangeArg{ID: 99, Type: uop.AxisType(100)}, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx, rng)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 0, Size: 4, DType: uop.Dtypes.Float32, Slot: -1}}}
	if err := interpret(item, map[uint32]*Buffer{0: mustBuf(t, 4)}); err == nil {
		t.Error("unknown axis type should error")
	}
}

func TestInterpretOutputIndexNotIndex(t *testing.T) {
	a := uop.NewArena(256)
	// store.Src[0] is not an OpIndex.
	notIndex := cf(a, 1)
	sink := mkStoreKernel(a, cf(a, 1), notIndex)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Float32, Slot: -1}}}
	if err := interpret(item, map[uint32]*Buffer{}); err == nil {
		t.Error("non-Index store target should error")
	}
}

func TestInterpretOutputBaseNotParam(t *testing.T) {
	a := uop.NewArena(256)
	// Index base is not OpParam.
	notParam := cf(a, 1)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{notParam}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Float32, Slot: -1}}}
	if err := interpret(item, map[uint32]*Buffer{}); err == nil {
		t.Error("non-Param index base should error")
	}
}

func TestInterpretOutputParamNonZero(t *testing.T) {
	a := uop.NewArena(256)
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil) // idx 1, not 0
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Float32, Slot: -1}}}
	if err := interpret(item, map[uint32]*Buffer{}); err == nil {
		t.Error("output param idx != 0 should error")
	}
}

func TestInterpretNoBuffers(t *testing.T) {
	a := uop.NewArena(256)
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx)
	item := schedule.ExecItem{Ast: sink, Bufs: nil}
	if err := interpret(item, map[uint32]*Buffer{}); err == nil {
		t.Error("kernel with no buffers should error")
	}
}

func TestInterpretMissingOutputBuffer(t *testing.T) {
	a := uop.NewArena(256)
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 5, Size: 1, DType: uop.Dtypes.Float32, Slot: -1}}}
	// bufs map has no entry for UOpIdx 5.
	if err := interpret(item, map[uint32]*Buffer{}); err == nil {
		t.Error("missing output buffer should error")
	}
}

func TestInterpretOutputNotF32(t *testing.T) {
	a := uop.NewArena(256)
	param := a.New(uop.OpParam, uop.Dtypes.Int32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Int32, []uop.UOp{param}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), idx)
	i32buf, _ := newBuffer(1, uop.Dtypes.Int32)
	item := schedule.ExecItem{Ast: sink, Bufs: []schedule.Buffer{{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Int32, Slot: -1}}}
	if err := interpret(item, map[uint32]*Buffer{0: i32buf}); err == nil {
		t.Error("i32 output buffer (no f32 storage) should error")
	}
}

// ── evalInt: extra ops & error arms ─────────────────────────────────────────

func TestEvalIntIDivModByZero(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	div := a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{ci(a, 6), ci(a, 0)}, nil, nil)
	if _, err := st.evalInt(div); err == nil {
		t.Error("IDiv by zero should error")
	}
	mod := a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{ci(a, 6), ci(a, 0)}, nil, nil)
	if _, err := st.evalInt(mod); err == nil {
		t.Error("Mod by zero should error")
	}
}

func TestEvalIntIDivModFloor(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	div := a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{ci(a, 7), ci(a, 2)}, nil, nil)
	if v, err := st.evalInt(div); err != nil || v != 3 {
		t.Errorf("7/2 = %d (err %v), want 3", v, err)
	}
	mod := a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{ci(a, 7), ci(a, 3)}, nil, nil)
	if v, err := st.evalInt(mod); err != nil || v != 1 {
		t.Errorf("7%%3 = %d (err %v), want 1", v, err)
	}
}

func TestEvalIntWhere(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	tru := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{ci(a, 1), ci(a, 10), ci(a, 20)}, nil, nil)
	if v, err := st.evalInt(tru); err != nil || v != 10 {
		t.Errorf("Where(1,10,20) = %d (err %v), want 10", v, err)
	}
	fls := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{ci(a, 0), ci(a, 10), ci(a, 20)}, nil, nil)
	if v, err := st.evalInt(fls); err != nil || v != 20 {
		t.Errorf("Where(0,10,20) = %d (err %v), want 20", v, err)
	}
}

func TestEvalIntMaxMinNeg(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	mx := a.New(uop.OpMax, uop.Dtypes.Index, []uop.UOp{ci(a, 3), ci(a, 8)}, nil, nil)
	if v, _ := st.evalInt(mx); v != 8 {
		t.Errorf("Max = %d, want 8", v)
	}
	mn := a.New(uop.OpMin, uop.Dtypes.Index, []uop.UOp{ci(a, 3), ci(a, 8)}, nil, nil)
	if v, _ := st.evalInt(mn); v != 3 {
		t.Errorf("Min = %d, want 3", v)
	}
	ng := a.New(uop.OpNeg, uop.Dtypes.Index, []uop.UOp{ci(a, 5)}, nil, nil)
	if v, _ := st.evalInt(ng); v != -5 {
		t.Errorf("Neg = %d, want -5", v)
	}
}

func TestEvalIntConstFloatBool(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	cflt := a.New(uop.OpConst, uop.Dtypes.Index, nil, 3.9, nil)
	if v, _ := st.evalInt(cflt); v != 3 {
		t.Errorf("int const(3.9) = %d, want 3", v)
	}
	cb := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	if v, _ := st.evalInt(cb); v != 1 {
		t.Errorf("int const(true) = %d, want 1", v)
	}
	cbad := a.New(uop.OpConst, uop.Dtypes.Index, nil, "oops", nil)
	if _, err := st.evalInt(cbad); err == nil {
		t.Error("int const with string arg should error")
	}
}

func TestEvalIntRangeUnbound(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{ci(a, 4)}, uop.RangeArg{ID: 42, Type: uop.AxisLoop}, nil)
	if _, err := st.evalInt(rng); err == nil {
		t.Error("unbound range should error in evalInt")
	}
}

// ── evalIntIndex error arms ─────────────────────────────────────────────────

func TestEvalIntIndexNonIndex(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	if _, err := st.evalIntIndex(cf(a, 1)); err == nil {
		t.Error("evalIntIndex on non-Index should error")
	}
}

func TestEvalIntIndexBaseNotParam(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), ci(a, 0)}, nil, nil)
	if _, err := st.evalIntIndex(idx); err == nil {
		t.Error("Index base not Param should error")
	}
}

func TestEvalIntIndexShapeTooShort(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 0, Shape: []int64{4}}} // 1 dim
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	// 2-dim index over a 1-dim shape.
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 0), ci(a, 1)}, nil, nil)
	if _, err := st.evalIntIndex(idx); err == nil {
		t.Error("nDims > shape len should error")
	}
}

func TestEvalIntIndexSymbolicDim(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	// Shape has a 0 (symbolic) read as a stride multiplier → static-only error.
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 0, Shape: []int64{4, 0, 4}}}
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 0), ci(a, 1), ci(a, 2)}, nil, nil)
	if _, err := st.evalIntIndex(idx); err == nil {
		t.Error("symbolic dim in shape should error")
	}
}

func TestEvalIntIndexZeroAndOneDim(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	// 0-dim index → flat 0.
	idx0 := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	if v, err := st.evalIntIndex(idx0); err != nil || v != 0 {
		t.Errorf("0-dim index = %d (err %v)", v, err)
	}
	// 1-dim index → evalInt of the single subscript.
	idx1 := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 5)}, nil, nil)
	if v, err := st.evalIntIndex(idx1); err != nil || v != 5 {
		t.Errorf("1-dim index = %d (err %v)", v, err)
	}
}

// ── evalIndexLoadFloat error arms ───────────────────────────────────────────

func TestEvalIndexLoadParamOutOfRange(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 0}}
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(9), nil) // idx 9, only 1 buf
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 0)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idx); err == nil {
		t.Error("param idx out of range should error")
	}
}

func TestEvalIndexLoadMissingBuffer(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 3}}
	st.bufs = map[uint32]*Buffer{} // no buffer for UOpIdx 3
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 0)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idx); err == nil {
		t.Error("missing buffer should error")
	}
}

func TestEvalIndexLoadF32OutOfRange(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 0}}
	b, _ := newBuffer(2, uop.Dtypes.Float32)
	st.bufs = map[uint32]*Buffer{0: b}
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 99)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idx); err == nil {
		t.Error("f32 load out of range should error")
	}
}

func TestEvalIndexLoadI32Path(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 0}}
	b, _ := newBuffer(2, uop.Dtypes.Int32)
	b.asI32()[1] = 42
	st.bufs = map[uint32]*Buffer{0: b}
	param := a.New(uop.OpParam, uop.Dtypes.Int32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Int32, []uop.UOp{param, ci(a, 1)}, nil, nil)
	if v, err := st.evalIndexLoadFloat(idx); err != nil || v != 42 {
		t.Errorf("i32 load = %v (err %v), want 42", v, err)
	}
	// Out-of-range i32 load.
	idxOOR := a.New(uop.OpIndex, uop.Dtypes.Int32, []uop.UOp{param, ci(a, 99)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idxOOR); err == nil {
		t.Error("i32 load out of range should error")
	}
}

// ── evalReduce error arms ───────────────────────────────────────────────────

func TestEvalReduceBadArgType(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	// Reduce arg is not a uop.Op.
	red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, int64(5), nil)
	if _, err := st.evalReduce(red); err == nil {
		t.Error("reduce with non-Op arg should error")
	}
}

func TestEvalReduceUnimplementedAccOp(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, uop.OpXor, nil)
	if _, err := st.evalReduce(red); err == nil {
		t.Error("reduce with unsupported accumulator op should error")
	}
}

func TestEvalReduceNonRangeSrc(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	// Src[1] is neither Const nor Range.
	bad := a.New(uop.OpSqrt, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, nil, nil)
	red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), bad}, uop.OpAdd, nil)
	if _, err := st.evalReduce(red); err == nil {
		t.Error("reduce with non-Range reduce src should error")
	}
}

func TestEvalReduceProductAndConstRange(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	// Product reduce over a single Const "range" (size 1) of body=3 → 3.
	c := ci(a, 1)
	red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 3), c}, uop.OpMul, nil)
	if v, err := st.evalReduce(red); err != nil || v != 3 {
		t.Errorf("product reduce const-range = %v (err %v), want 3", v, err)
	}
}

// ── evalFloat: extra arms ───────────────────────────────────────────────────

func TestEvalFloatConstArms(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	cb := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	if v, _ := st.evalFloat(cb); v != 1 {
		t.Errorf("const(true) = %v, want 1", v)
	}
	cbF := a.New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil)
	if v, _ := st.evalFloat(cbF); v != 0 {
		t.Errorf("const(false) = %v, want 0", v)
	}
	cInt := a.New(uop.OpConst, uop.Dtypes.Float32, nil, int64(7), nil)
	if v, _ := st.evalFloat(cInt); v != 7 {
		t.Errorf("const(int64 7) = %v, want 7", v)
	}
	cBad := a.New(uop.OpConst, uop.Dtypes.Float32, nil, "x", nil)
	if _, err := st.evalFloat(cBad); err == nil {
		t.Error("const with string arg should error")
	}
}

func TestEvalFloatRange(t *testing.T) {
	a := uop.NewArena(256)
	st := stWithRange(3, 9)
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{ci(a, 16)}, uop.RangeArg{ID: 3, Type: uop.AxisLoop}, nil)
	if v, err := st.evalFloat(rng); err != nil || v != 9 {
		t.Errorf("range float = %v (err %v), want 9", v, err)
	}
	// Unbound range.
	st2 := newEvalState()
	if _, err := st2.evalFloat(rng); err == nil {
		t.Error("unbound range should error in evalFloat")
	}
}

func TestEvalFloatCastIntInner(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	// Cast wrapping an int-typed inner expression routes through evalInt.
	inner := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{ci(a, 3), ci(a, 4)}, nil, nil)
	cast := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{inner}, nil, nil)
	if v, err := st.evalFloat(cast); err != nil || v != 7 {
		t.Errorf("cast(int 3+4) = %v (err %v), want 7", v, err)
	}
}

func TestEvalFloatWhereAndMulAcc(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	w := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cf(a, 0), cf(a, 1), cf(a, 2)}, nil, nil)
	if v, _ := st.evalFloat(w); v != 2 {
		t.Errorf("Where(0,..) = %v, want 2", v)
	}
	wt := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), cf(a, 1), cf(a, 2)}, nil, nil)
	if v, _ := st.evalFloat(wt); v != 1 {
		t.Errorf("Where(1,..) = %v, want 1", v)
	}
	ma := a.New(uop.OpMulAcc, uop.Dtypes.Float32, []uop.UOp{cf(a, 2), cf(a, 3), cf(a, 4)}, nil, nil)
	if v, _ := st.evalFloat(ma); v != 10 {
		t.Errorf("MulAcc(2,3,4) = %v, want 10 (2*3+4)", v)
	}
}

func TestEvalFloatRecipMaxMin(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	r := a.New(uop.OpReciprocal, uop.Dtypes.Float32, []uop.UOp{cf(a, 4)}, nil, nil)
	if v, _ := st.evalFloat(r); v != 0.25 {
		t.Errorf("Reciprocal(4) = %v, want 0.25", v)
	}
	mx := a.New(uop.OpMax, uop.Dtypes.Float32, []uop.UOp{cf(a, 2), cf(a, 5)}, nil, nil)
	if v, _ := st.evalFloat(mx); v != 5 {
		t.Errorf("Max = %v, want 5", v)
	}
	mn := a.New(uop.OpMin, uop.Dtypes.Float32, []uop.UOp{cf(a, 2), cf(a, 5)}, nil, nil)
	if v, _ := st.evalFloat(mn); v != 2 {
		t.Errorf("Min = %v, want 2", v)
	}
}

// ── error propagation across every binary/ternary arm ───────────────────────

// TestEvalFloatBinaryErrorPropagation feeds an unimplemented op (Sin) as each
// operand slot of every float ALU arm and asserts the error surfaces. This
// pins the err != nil propagation branches that the happy-path tests skip.
func TestEvalFloatBinaryErrorPropagation(t *testing.T) {
	a := uop.NewArena(1024)
	st := newEvalState()
	bad := a.New(uop.OpSin, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, nil, nil)
	ok := cf(a, 2)

	binOps := []uop.Op{uop.OpAdd, uop.OpSub, uop.OpMul, uop.OpFDiv, uop.OpMax, uop.OpMin}
	for _, op := range binOps {
		lhs := a.New(op, uop.Dtypes.Float32, []uop.UOp{bad, ok}, nil, nil)
		if _, err := st.evalFloat(lhs); err == nil {
			t.Errorf("float %v: lhs error not propagated", op)
		}
		rhs := a.New(op, uop.Dtypes.Float32, []uop.UOp{ok, bad}, nil, nil)
		if _, err := st.evalFloat(rhs); err == nil {
			t.Errorf("float %v: rhs error not propagated", op)
		}
	}

	// Unary arms.
	unOps := []uop.Op{uop.OpNeg, uop.OpReciprocal, uop.OpSqrt, uop.OpExp2, uop.OpLog2}
	for _, op := range unOps {
		u := a.New(op, uop.Dtypes.Float32, []uop.UOp{bad}, nil, nil)
		if _, err := st.evalFloat(u); err == nil {
			t.Errorf("float unary %v: error not propagated", op)
		}
	}

	// Where: cond err, then-branch err (cond true), else-branch err (cond false).
	wCond := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{bad, ok, ok}, nil, nil)
	if _, err := st.evalFloat(wCond); err == nil {
		t.Error("float Where cond: error not propagated")
	}
	wThen := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), bad, ok}, nil, nil)
	if _, err := st.evalFloat(wThen); err == nil {
		t.Error("float Where then: error not propagated")
	}
	wElse := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cf(a, 0), ok, bad}, nil, nil)
	if _, err := st.evalFloat(wElse); err == nil {
		t.Error("float Where else: error not propagated")
	}

	// MulAcc: each of three slots.
	for slot := 0; slot < 3; slot++ {
		srcs := []uop.UOp{cf(a, 1), cf(a, 1), cf(a, 1)}
		srcs[slot] = bad
		m := a.New(uop.OpMulAcc, uop.Dtypes.Float32, srcs, nil, nil)
		if _, err := st.evalFloat(m); err == nil {
			t.Errorf("float MulAcc slot %d: error not propagated", slot)
		}
	}

	// Cast wrapping a failing float inner.
	c := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{bad}, nil, nil)
	if _, err := st.evalFloat(c); err == nil {
		t.Error("float Cast: inner error not propagated")
	}
}

// TestEvalIntBinaryErrorPropagation does the same for the integer evaluator.
func TestEvalIntBinaryErrorPropagation(t *testing.T) {
	a := uop.NewArena(1024)
	st := newEvalState()
	bad := a.New(uop.OpXor, uop.Dtypes.Index, []uop.UOp{ci(a, 1), ci(a, 1)}, nil, nil)
	ok := ci(a, 2)

	binOps := []uop.Op{uop.OpAdd, uop.OpSub, uop.OpMul, uop.OpIDiv, uop.OpMod, uop.OpMax, uop.OpMin}
	for _, op := range binOps {
		lhs := a.New(op, uop.Dtypes.Index, []uop.UOp{bad, ok}, nil, nil)
		if _, err := st.evalInt(lhs); err == nil {
			t.Errorf("int %v: lhs error not propagated", op)
		}
		rhs := a.New(op, uop.Dtypes.Index, []uop.UOp{ci(a, 6), bad}, nil, nil)
		if _, err := st.evalInt(rhs); err == nil {
			t.Errorf("int %v: rhs error not propagated", op)
		}
	}

	// Neg, Cast unary.
	ng := a.New(uop.OpNeg, uop.Dtypes.Index, []uop.UOp{bad}, nil, nil)
	if _, err := st.evalInt(ng); err == nil {
		t.Error("int Neg: error not propagated")
	}
	c := a.New(uop.OpCast, uop.Dtypes.Index, []uop.UOp{bad}, nil, nil)
	if _, err := st.evalInt(c); err == nil {
		t.Error("int Cast: error not propagated")
	}

	// Where: cond err, then err (cond true), else err (cond false).
	wCond := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{bad, ok, ok}, nil, nil)
	if _, err := st.evalInt(wCond); err == nil {
		t.Error("int Where cond: error not propagated")
	}
	wThen := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{ci(a, 1), bad, ok}, nil, nil)
	if _, err := st.evalInt(wThen); err == nil {
		t.Error("int Where then: error not propagated")
	}
	wElse := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{ci(a, 0), ok, bad}, nil, nil)
	if _, err := st.evalInt(wElse); err == nil {
		t.Error("int Where else: error not propagated")
	}
}

func TestEvalIndexLoadBufferNoStorage(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	st.item.Bufs = []schedule.Buffer{{UOpIdx: 0}}
	st.bufs = map[uint32]*Buffer{0: {}} // Buffer with neither f32 nor i32 storage
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 0)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idx); err == nil {
		t.Error("storage-less buffer should error on load")
	}
}

func TestEvalReduceBodyError(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	bad := a.New(uop.OpSin, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, nil, nil)
	red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{bad, ci(a, 1)}, uop.OpAdd, nil)
	if _, err := st.evalReduce(red); err == nil {
		t.Error("reduce body error should propagate")
	}
}

func TestEvalReduceSymbolicRange(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	// A reduce range whose bound is not a Const → symbolic, rejected.
	v := a.DefineVar("k", 1, 8)
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{v}, uop.RangeArg{ID: 0, Type: uop.AxisReduce}, nil)
	red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), rng}, uop.OpAdd, nil)
	if _, err := st.evalReduce(red); err == nil {
		t.Error("symbolic reduce range should error")
	}
}

func TestEvalIntConstFalse(t *testing.T) {
	a := uop.NewArena(256)
	st := newEvalState()
	cb := a.New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil)
	if v, _ := st.evalInt(cb); v != 0 {
		t.Errorf("int const(false) = %d, want 0", v)
	}
}

// ── Run: full E2E with f32 leaf upload and output collection ─────────────────

func TestRunE2EAddConst(t *testing.T) {
	a := uop.NewArena(512)
	// out[i] = in[i] + 1  over a length-4 loop.
	out := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	in := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{ci(a, 4)}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	loadIdx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{in, rng}, nil, nil)
	body := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{loadIdx, cf(a, 1)}, nil, nil)
	outIdx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{out, rng}, nil, nil)
	store := a.New(uop.OpStore, uop.Dtypes.Float32, []uop.UOp{outIdx, body}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Float32, []uop.UOp{store, rng}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Float32, []uop.UOp{end}, nil, nil)

	item := schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			{UOpIdx: 0, Size: 4, DType: uop.Dtypes.Float32, Slot: -1, Shape: []int64{4}},
			{UOpIdx: 1, Size: 4, DType: uop.Dtypes.Float32, Slot: -1, Shape: []int64{4}},
		},
	}
	d, _ := Open()
	out2, err := d.Run([]schedule.ExecItem{item}, map[uint32][]float32{1: {10, 20, 30, 40}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := out2[0]
	want := []float32{11, 21, 31, 41}
	if len(got) != 4 {
		t.Fatalf("output len = %d, want 4", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRunUnsupportedInputDtype(t *testing.T) {
	a := uop.NewArena(256)
	out := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	outIdx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{out}, nil, nil)
	sink := mkStoreKernel(a, cf(a, 1), outIdx)
	item := schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Float32, Slot: -1},
			{UOpIdx: 1, Size: 1, DType: uop.Dtypes.Bool, Slot: -1},
		},
	}
	d, _ := Open()
	// Leaf 1 has an unsupported (Bool) input dtype; allocateBuffers will reject
	// it before upload (Bool has no CPU storage).
	if _, err := d.Run([]schedule.ExecItem{item}, map[uint32][]float32{1: {1}}); err == nil {
		t.Error("unsupported leaf dtype should make Run fail")
	}
}

// ── helpers ─────────────────────────────────────────────────────────────────

func mustBuf(t *testing.T, n int64) *Buffer {
	t.Helper()
	b, err := newBuffer(n, uop.Dtypes.Float32)
	if err != nil {
		t.Fatalf("newBuffer: %v", err)
	}
	return b
}
