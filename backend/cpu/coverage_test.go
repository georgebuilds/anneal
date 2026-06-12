package cpu

// White-box coverage tests driving the interp.go evaluators directly over
// hand-built UOp nodes. These reach the elementwise ALU arms, reduce
// accumulators, integer index arithmetic, and error paths that the
// tensor-pipeline E2E tests do not exercise individually.

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// stWithRange returns a fresh evaluator state with range id bound to v.
func stWithRange(id int, v int64) *state {
	s := &state{
		rangeVal:    map[int]int64{id: v},
		paramShapes: make(map[int][]int64),
	}
	return s
}

func cf(a *uop.Arena, v float64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Float32, nil, v, nil)
}
func ci(a *uop.Arena, v int64) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil)
}

// ── evalFloat: unary + binary ALU arms ───────────────────────────────────────

func TestEvalFloat_ALU(t *testing.T) {
	a := uop.NewArena(512)
	st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}

	bin := func(op uop.Op, x, y float64) float32 {
		n := a.New(op, uop.Dtypes.Float32, []uop.UOp{cf(a, x), cf(a, y)}, nil, nil)
		v, err := st.evalFloat(n)
		if err != nil {
			t.Fatalf("%v: %v", op, err)
		}
		return v
	}
	un := func(op uop.Op, x float64) float32 {
		n := a.New(op, uop.Dtypes.Float32, []uop.UOp{cf(a, x)}, nil, nil)
		v, err := st.evalFloat(n)
		if err != nil {
			t.Fatalf("%v: %v", op, err)
		}
		return v
	}

	checks := []struct {
		got, want float32
		label     string
	}{
		{bin(uop.OpAdd, 2, 3), 5, "add"},
		{bin(uop.OpSub, 5, 2), 3, "sub"},
		{bin(uop.OpMul, 4, 3), 12, "mul"},
		{bin(uop.OpFDiv, 9, 3), 3, "fdiv"},
		{bin(uop.OpMax, 2, 7), 7, "max"},
		{bin(uop.OpMin, 2, 7), 2, "min"},
		{un(uop.OpNeg, 5), -5, "neg"},
		{un(uop.OpReciprocal, 4), 0.25, "recip"},
		{un(uop.OpSqrt, 9), 3, "sqrt"},
		{un(uop.OpExp2, 3), 8, "exp2"},
		{un(uop.OpLog2, 8), 3, "log2"},
	}
	for _, c := range checks {
		if math.Abs(float64(c.got-c.want)) > 1e-5 {
			t.Errorf("%s: got %v want %v", c.label, c.got, c.want)
		}
	}

	// MulAcc: a*b + c.
	mac := a.New(uop.OpMulAcc, uop.Dtypes.Float32, []uop.UOp{cf(a, 2), cf(a, 3), cf(a, 1)}, nil, nil)
	if v, _ := st.evalFloat(mac); v != 7 {
		t.Errorf("mulacc: got %v want 7", v)
	}

	// Where: cond != 0 → src1 else src2.
	wT := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), cf(a, 10), cf(a, 20)}, nil, nil)
	if v, _ := st.evalFloat(wT); v != 10 {
		t.Errorf("where true: got %v", v)
	}
	wF := a.New(uop.OpWhere, uop.Dtypes.Float32, []uop.UOp{cf(a, 0), cf(a, 10), cf(a, 20)}, nil, nil)
	if v, _ := st.evalFloat(wF); v != 20 {
		t.Errorf("where false: got %v", v)
	}
}

// ── evalFloat: const variants + range + cast ─────────────────────────────────

func TestEvalFloat_ConstRangeCast(t *testing.T) {
	a := uop.NewArena(512)
	st := stWithRange(3, 9)

	// int64 const.
	if v, _ := st.evalFloat(ci(a, 42)); v != 42 {
		t.Errorf("int64 const: %v", v)
	}
	// bool consts.
	bt := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	bf := a.New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil)
	if v, _ := st.evalFloat(bt); v != 1 {
		t.Errorf("bool true: %v", v)
	}
	if v, _ := st.evalFloat(bf); v != 0 {
		t.Errorf("bool false: %v", v)
	}
	// Bad const type.
	bad := a.New(uop.OpConst, uop.Dtypes.Float32, nil, "oops", nil)
	if _, err := st.evalFloat(bad); err == nil {
		t.Error("string const should error")
	}

	// Range bound → value.
	rng := a.New(uop.OpRange, uop.Dtypes.Index,
		[]uop.UOp{ci(a, 16)}, uop.RangeArg{ID: 3, Type: uop.AxisLoop}, nil)
	if v, _ := st.evalFloat(rng); v != 9 {
		t.Errorf("range: %v", v)
	}
	// Unbound range.
	rng2 := a.New(uop.OpRange, uop.Dtypes.Index,
		[]uop.UOp{ci(a, 16)}, uop.RangeArg{ID: 99, Type: uop.AxisLoop}, nil)
	if _, err := st.evalFloat(rng2); err == nil {
		t.Error("unbound range should error")
	}

	// Cast with int inner → integer eval then float.
	castI := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{ci(a, 7)}, nil, nil)
	if v, _ := st.evalFloat(castI); v != 7 {
		t.Errorf("cast int inner: %v", v)
	}
	// Cast with float inner → recurse.
	castF := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{cf(a, 2.5)}, nil, nil)
	if v, _ := st.evalFloat(castF); v != 2.5 {
		t.Errorf("cast float inner: %v", v)
	}
}

// ── evalInt: arithmetic + control arms + errors ──────────────────────────────

func TestEvalInt_Arith(t *testing.T) {
	a := uop.NewArena(512)
	st := stWithRange(1, 5)

	bin := func(op uop.Op, x, y int64) (int64, error) {
		n := a.New(op, uop.Dtypes.Index, []uop.UOp{ci(a, x), ci(a, y)}, nil, nil)
		return st.evalInt(n)
	}
	mustEq := func(op uop.Op, x, y, want int64) {
		v, err := bin(op, x, y)
		if err != nil || v != want {
			t.Errorf("%v(%d,%d) = %d,%v want %d", op, x, y, v, err, want)
		}
	}
	mustEq(uop.OpAdd, 2, 3, 5)
	mustEq(uop.OpSub, 7, 2, 5)
	mustEq(uop.OpMul, 4, 3, 12)
	mustEq(uop.OpIDiv, 7, 2, 3)
	mustEq(uop.OpMod, 7, 3, 1)
	mustEq(uop.OpMax, 4, 9, 9)
	mustEq(uop.OpMin, 4, 9, 4)

	// Neg unary.
	neg := a.New(uop.OpNeg, uop.Dtypes.Index, []uop.UOp{ci(a, 6)}, nil, nil)
	if v, _ := st.evalInt(neg); v != -6 {
		t.Errorf("neg: %v", v)
	}

	// IDiv / Mod by zero.
	if _, err := bin(uop.OpIDiv, 1, 0); err == nil {
		t.Error("idiv by zero should error")
	}
	if _, err := bin(uop.OpMod, 1, 0); err == nil {
		t.Error("mod by zero should error")
	}

	// Const variants.
	cFloat := a.New(uop.OpConst, uop.Dtypes.Index, nil, 3.0, nil)
	if v, _ := st.evalInt(cFloat); v != 3 {
		t.Errorf("float const → int: %v", v)
	}
	cBoolT := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	if v, _ := st.evalInt(cBoolT); v != 1 {
		t.Errorf("bool const → int: %v", v)
	}
	cBad := a.New(uop.OpConst, uop.Dtypes.Index, nil, "x", nil)
	if _, err := st.evalInt(cBad); err == nil {
		t.Error("bad int const should error")
	}

	// Range bound + unbound.
	rng := a.New(uop.OpRange, uop.Dtypes.Index,
		[]uop.UOp{ci(a, 8)}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)
	if v, _ := st.evalInt(rng); v != 5 {
		t.Errorf("int range: %v", v)
	}
	rng2 := a.New(uop.OpRange, uop.Dtypes.Index,
		[]uop.UOp{ci(a, 8)}, uop.RangeArg{ID: 77, Type: uop.AxisLoop}, nil)
	if _, err := st.evalInt(rng2); err == nil {
		t.Error("unbound int range should error")
	}

	// Cast no-op.
	cast := a.New(uop.OpCast, uop.Dtypes.Int64, []uop.UOp{ci(a, 11)}, nil, nil)
	if v, _ := st.evalInt(cast); v != 11 {
		t.Errorf("int cast: %v", v)
	}

	// Where int.
	wT := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{ci(a, 1), ci(a, 10), ci(a, 20)}, nil, nil)
	if v, _ := st.evalInt(wT); v != 10 {
		t.Errorf("int where true: %v", v)
	}
	wF := a.New(uop.OpWhere, uop.Dtypes.Index, []uop.UOp{ci(a, 0), ci(a, 10), ci(a, 20)}, nil, nil)
	if v, _ := st.evalInt(wF); v != 20 {
		t.Errorf("int where false: %v", v)
	}
}

// ── evalReduce: all accumulators + error arms ────────────────────────────────

func TestEvalReduce_Accumulators(t *testing.T) {
	a := uop.NewArena(512)

	// Build a reduce over a single range [0,4) whose body is the range value.
	// Sum=0+1+2+3=6, Mul=0 (includes 0), Max=3, Min=0.
	build := func(accOp uop.Op) (uop.UOp, *state) {
		st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}
		rng := a.New(uop.OpRange, uop.Dtypes.Index,
			[]uop.UOp{ci(a, 4)}, uop.RangeArg{ID: 5, Type: uop.AxisReduce}, nil)
		// body: cast(range) to float.
		body := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{rng}, nil, nil)
		red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{body, rng}, accOp, nil)
		return red, st
	}

	cases := []struct {
		op   uop.Op
		want float32
	}{
		{uop.OpAdd, 6},
		{uop.OpMul, 0},
		{uop.OpMax, 3},
		{uop.OpMin, 0},
	}
	for _, c := range cases {
		red, st := build(c.op)
		v, err := st.evalReduce(red)
		if err != nil {
			t.Fatalf("reduce %v: %v", c.op, err)
		}
		if v != c.want {
			t.Errorf("reduce %v: got %v want %v", c.op, v, c.want)
		}
	}

	// Mul reduce over [1,4) excluding zero would need different body; verify a
	// non-trivial Mul: range value + 1 over [0,3) → 1*2*3 = 6.
	{
		st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}
		rng := a.New(uop.OpRange, uop.Dtypes.Index,
			[]uop.UOp{ci(a, 3)}, uop.RangeArg{ID: 6, Type: uop.AxisReduce}, nil)
		castR := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{rng}, nil, nil)
		body := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{castR, cf(a, 1)}, nil, nil)
		red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{body, rng}, uop.OpMul, nil)
		v, err := st.evalReduce(red)
		if err != nil || v != 6 {
			t.Errorf("mul reduce: got %v,%v want 6", v, err)
		}
	}

	// Unsupported reduce op.
	{
		st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}
		rng := a.New(uop.OpRange, uop.Dtypes.Index,
			[]uop.UOp{ci(a, 2)}, uop.RangeArg{ID: 7, Type: uop.AxisReduce}, nil)
		castR := a.New(uop.OpCast, uop.Dtypes.Float32, []uop.UOp{rng}, nil, nil)
		red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{castR, rng}, uop.OpSub, nil)
		if _, err := st.evalReduce(red); err == nil {
			t.Error("unsupported reduce op should error")
		}
	}

	// Bad reduce arg type (not a uop.Op).
	{
		st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}
		red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, int64(0), nil)
		if _, err := st.evalReduce(red); err == nil {
			t.Error("non-Op reduce arg should error")
		}
	}

	// Reduce src that is neither Const nor Range.
	{
		st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}
		bogus := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{ci(a, 1), ci(a, 2)}, nil, nil)
		red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 1), bogus}, uop.OpAdd, nil)
		if _, err := st.evalReduce(red); err == nil {
			t.Error("non-range reduce src should error")
		}
	}

	// Const reduce range (size-1) accumulates exactly one body eval.
	{
		st := &state{rangeVal: map[int]int64{}, paramShapes: map[int][]int64{}}
		constR := ci(a, 1)
		red := a.New(uop.OpReduce, uop.Dtypes.Float32, []uop.UOp{cf(a, 5), constR}, uop.OpAdd, nil)
		v, err := st.evalReduce(red)
		if err != nil || v != 5 {
			t.Errorf("const-range reduce: got %v,%v want 5", v, err)
		}
	}
}

// ── evalIntIndex: rank handling + errors ─────────────────────────────────────

func TestEvalIntIndex(t *testing.T) {
	a := uop.NewArena(512)
	st := &state{
		rangeVal:    map[int]int64{},
		paramShapes: map[int][]int64{},
		item: schedule.ExecItem{
			Bufs: []schedule.Buffer{
				{UOpIdx: 0, Shape: []int64{2, 3}},
			},
		},
	}
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)

	// nDims == 0: returns 0.
	idx0 := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	if v, err := st.evalIntIndex(idx0); err != nil || v != 0 {
		t.Errorf("nDims=0: %v,%v", v, err)
	}

	// nDims == 1: passthrough.
	idx1 := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 4)}, nil, nil)
	if v, err := st.evalIntIndex(idx1); err != nil || v != 4 {
		t.Errorf("nDims=1: %v,%v", v, err)
	}

	// nDims == 2: row*stride + col over shape [2,3].
	idx2 := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param, ci(a, 1), ci(a, 2)}, nil, nil)
	if v, err := st.evalIntIndex(idx2); err != nil || v != 1*3+2 {
		t.Errorf("nDims=2: got %v,%v want 5", v, err)
	}

	// Non-Index op.
	if _, err := st.evalIntIndex(param); err == nil {
		t.Error("non-Index should error")
	}

	// shape shorter than nDims.
	stShort := &state{
		rangeVal:    map[int]int64{},
		paramShapes: map[int][]int64{},
		item: schedule.ExecItem{
			Bufs: []schedule.Buffer{{UOpIdx: 0, Shape: []int64{2}}},
		},
	}
	if _, err := stShort.evalIntIndex(idx2); err == nil {
		t.Error("short shape should error")
	}

	// symbolic (zero) dim mid-shape.
	stSym := &state{
		rangeVal:    map[int]int64{},
		paramShapes: map[int][]int64{},
		item: schedule.ExecItem{
			Bufs: []schedule.Buffer{{UOpIdx: 0, Shape: []int64{2, 0}}},
		},
	}
	if _, err := stSym.evalIntIndex(idx2); err == nil {
		t.Error("symbolic dim should error")
	}
}

// ── evalIndexLoadFloat: f32/i32 loads + error arms ───────────────────────────

func TestEvalIndexLoadFloat(t *testing.T) {
	a := uop.NewArena(512)
	bf, _ := newBuffer(4, uop.Dtypes.Float32)
	copy(bf.asF32(), []float32{10, 11, 12, 13})
	bi, _ := newBuffer(4, uop.Dtypes.Int32)
	copy(bi.asI32(), []int32{100, 101, 102, 103})

	st := &state{
		rangeVal:    map[int]int64{},
		paramShapes: map[int][]int64{},
		bufs:        map[uint32]*Buffer{0: bf, 1: bi},
		item: schedule.ExecItem{
			Bufs: []schedule.Buffer{
				{UOpIdx: 0, Shape: []int64{4}},
				{UOpIdx: 1, Shape: []int64{4}},
			},
		},
	}

	p0 := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	p1 := a.New(uop.OpParam, uop.Dtypes.Int32, nil, int64(1), nil)

	// f32 load at index 2.
	idxF := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{p0, ci(a, 2)}, nil, nil)
	if v, err := st.evalIndexLoadFloat(idxF); err != nil || v != 12 {
		t.Errorf("f32 load: %v,%v", v, err)
	}
	// i32 load at index 1 → float32(101).
	idxI := a.New(uop.OpIndex, uop.Dtypes.Int32, []uop.UOp{p1, ci(a, 1)}, nil, nil)
	if v, err := st.evalIndexLoadFloat(idxI); err != nil || v != 101 {
		t.Errorf("i32 load: %v,%v", v, err)
	}
	// Out-of-range f32.
	idxOOR := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{p0, ci(a, 99)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idxOOR); err == nil {
		t.Error("oor f32 load should error")
	}
	// Out-of-range i32.
	idxOORi := a.New(uop.OpIndex, uop.Dtypes.Int32, []uop.UOp{p1, ci(a, 99)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idxOORi); err == nil {
		t.Error("oor i32 load should error")
	}
	// Non-Param base.
	notParam := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{ci(a, 0), ci(a, 0)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(notParam); err == nil {
		t.Error("non-Param base should error")
	}
	// Param idx out of bufs range.
	pBad := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(9), nil)
	idxBad := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{pBad, ci(a, 0)}, nil, nil)
	if _, err := st.evalIndexLoadFloat(idxBad); err == nil {
		t.Error("param idx out of range should error")
	}
}

// ── Device.Run / Close edge paths ────────────────────────────────────────────

func TestRun_EmptyAndClose(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	out, err := d.Run(nil, nil)
	if err != nil || out != nil {
		t.Errorf("empty Run: out=%v err=%v", out, err)
	}
	d.Close() // no-op, covers the method.
}

func TestRun_SymbolicKernelRejected(t *testing.T) {
	a := uop.NewArena(256)
	// Minimal SINK→END→STORE AST so allocateBuffers/interpret are reachable,
	// but mark the item symbolic so Run rejects it before interpret.
	param := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{param}, nil, nil)
	body := cf(a, 1)
	store := a.New(uop.OpStore, uop.Dtypes.Float32, []uop.UOp{idx, body}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Float32, []uop.UOp{store}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Float32, []uop.UOp{end}, nil, nil)

	item := schedule.ExecItem{
		Ast:     sink,
		Bufs:    []schedule.Buffer{{UOpIdx: 0, Size: 1, DType: uop.Dtypes.Float32, Slot: -1}},
		SymVars: []string{"N"},
	}
	d, _ := Open()
	if _, err := d.Run([]schedule.ExecItem{item}, nil); err == nil {
		t.Fatal("symbolic kernel should be rejected")
	}
}

// ── interpret: AST structural error paths ────────────────────────────────────

func TestInterpret_StructuralErrors(t *testing.T) {
	a := uop.NewArena(256)
	bufs := map[uint32]*Buffer{}

	// Non-SINK root.
	notSink := cf(a, 1)
	if err := interpret(schedule.ExecItem{Ast: notSink}, bufs); err == nil {
		t.Error("non-SINK AST should error")
	}

	// SINK whose Src[0] is not END.
	badEnd := a.New(uop.OpSink, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, nil, nil)
	if err := interpret(schedule.ExecItem{Ast: badEnd}, bufs); err == nil {
		t.Error("SINK without END should error")
	}

	// END whose Src[0] is not STORE.
	end := a.New(uop.OpEnd, uop.Dtypes.Float32, []uop.UOp{cf(a, 1)}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Float32, []uop.UOp{end}, nil, nil)
	if err := interpret(schedule.ExecItem{Ast: sink}, bufs); err == nil {
		t.Error("END without STORE should error")
	}
}

// ── incCounters mixed-radix wraparound ───────────────────────────────────────

func TestIncCounters(t *testing.T) {
	counters := []int64{0, 0}
	sizes := []int64{2, 2}
	seen := 0
	for {
		seen++
		if !incCounters(counters, sizes) {
			break
		}
		if seen > 10 {
			t.Fatal("incCounters did not terminate")
		}
	}
	if seen != 4 {
		t.Errorf("incCounters visited %d combos, want 4", seen)
	}
}
