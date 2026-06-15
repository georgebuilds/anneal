package codegen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// ── saveDiskMapLocked: MkdirAll failure path ─────────────────────────────────

// When diskPath's parent cannot be created (a regular file occupies a path
// component), saveDiskMapLocked returns early without writing or panicking.
func TestSaveDiskMapLockedMkdirFails(t *testing.T) {
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
	// Create a regular file, then make diskPath descend through it as if it
	// were a directory — MkdirAll must fail.
	blocker := filepath.Join(tmpDir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	badPath := filepath.Join(blocker, "sub", "beam_cache.json")

	diskMu.Lock()
	diskPath = badPath
	diskMap = map[string]diskEntry{"abc": {WGSLHash: "h"}}
	saveDiskMapLocked() // must not panic; returns on MkdirAll error
	diskMu.Unlock()

	// The cache file must not exist as a readable JSON cache (MkdirAll failed).
	if _, err := os.ReadFile(badPath); err == nil {
		t.Errorf("MkdirAll-failure path should not have produced a readable cache at %s", badPath)
	}
}

// ── renderSymBoundExpr: every supported op + each panic guard ────────────────

// newSymLowerer builds a lowerer whose symSlot map resolves the given
// DefineVar nodes to sequential slot indices, mirroring how lowerSink wires
// them up. Returns the lowerer plus the arena for node construction.
func newSymLowerer(vars ...uop.UOp) *lowerer {
	l := &lowerer{
		symSlot:       map[uint32]int{},
		symSlotByName: map[string]int{},
	}
	for i, v := range vars {
		l.symSlot[v.Index()] = i
		if va, ok := v.Arg().(uop.VarArg); ok {
			l.symSlotByName[va.Name] = i
		}
	}
	return l
}

func TestRenderSymBoundExprDefineVar(t *testing.T) {
	a := uop.NewArena(16)
	n := a.DefineVar("n", 1, 16)
	l := newSymLowerer(n)
	if got := l.renderSymBoundExpr(n); got != "params_n.n0" {
		t.Errorf("DefineVar bound = %q, want 'params_n.n0'", got)
	}
}

func TestRenderSymBoundExprConst(t *testing.T) {
	a := uop.NewArena(16)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(7), nil)
	l := newSymLowerer()
	if got := l.renderSymBoundExpr(c); got != "7u" {
		t.Errorf("Const bound = %q, want '7u'", got)
	}
}

func TestRenderSymBoundExprBinaryOps(t *testing.T) {
	a := uop.NewArena(32)
	n := a.DefineVar("n", 1, 16)
	c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	l := newSymLowerer(n)

	cases := []struct {
		op   uop.Op
		want string
	}{
		{uop.OpAdd, "(params_n.n0 + 4u)"},
		{uop.OpSub, "(params_n.n0 - 4u)"},
		{uop.OpMul, "(params_n.n0 * 4u)"},
		{uop.OpIDiv, "(params_n.n0 / 4u)"},
		{uop.OpMod, "(params_n.n0 % 4u)"},
	}
	for _, c := range cases {
		node := a.New(c.op, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
		if got := l.renderSymBoundExpr(node); got != c.want {
			t.Errorf("%s bound = %q, want %q", c.op, got, c.want)
		}
	}
}

func TestRenderSymBoundExprNested(t *testing.T) {
	// (n * 4) + 1 nests recursive rendering on both sides.
	a := uop.NewArena(32)
	n := a.DefineVar("n", 1, 16)
	c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	c1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
	add := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, c1}, nil, nil)
	l := newSymLowerer(n)
	if got := l.renderSymBoundExpr(add); got != "((params_n.n0 * 4u) + 1u)" {
		t.Errorf("nested bound = %q", got)
	}
}

func TestRenderSymBoundExprDefineVarMissingSlotPanics(t *testing.T) {
	a := uop.NewArena(16)
	n := a.DefineVar("n", 1, 16)
	l := newSymLowerer() // n absent from symSlot
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for DefineVar not in symSlot map")
		}
	}()
	_ = l.renderSymBoundExpr(n)
}

func TestRenderSymBoundExprNegativeConstPanics(t *testing.T) {
	a := uop.NewArena(16)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(-3), nil)
	l := newSymLowerer()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative constant bound")
		}
	}()
	_ = l.renderSymBoundExpr(c)
}

func TestRenderSymBoundExprConstWrongArgTypePanics(t *testing.T) {
	a := uop.NewArena(16)
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2), nil)
	l := newSymLowerer()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-int64 const arg")
		}
	}()
	_ = l.renderSymBoundExpr(c)
}

func TestRenderSymBoundExprUnsupportedOpPanics(t *testing.T) {
	a := uop.NewArena(16)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	bad := a.New(uop.OpMax, uop.Dtypes.Index, []uop.UOp{c, c}, nil, nil)
	l := newSymLowerer()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unsupported bound op")
		}
	}()
	_ = l.renderSymBoundExpr(bad)
}

// ── symBoundEmission: ALU-bound branch ───────────────────────────────────────

func TestSymBoundEmissionBareDefineVar(t *testing.T) {
	a := uop.NewArena(32)
	n := a.DefineVar("n", 1, 16)
	// OpRange whose bound (Src(0)) is a bare DefineVar → returns (slot, "").
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{n}, int64(0), nil)
	l := newSymLowerer(n)
	slot, expr := l.symBoundEmission(r)
	if slot != 0 || expr != "" {
		t.Errorf("bare DefineVar emission = (%d, %q), want (0, \"\")", slot, expr)
	}
}

func TestSymBoundEmissionALUBound(t *testing.T) {
	a := uop.NewArena(32)
	n := a.DefineVar("n", 1, 16)
	c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c4}, nil, nil)
	// OpRange whose bound is an ALU expr → returns (-1, "(... )").
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{mul}, int64(0), nil)
	l := newSymLowerer(n)
	slot, expr := l.symBoundEmission(r)
	if slot != -1 || expr != "(params_n.n0 * 4u)" {
		t.Errorf("ALU bound emission = (%d, %q)", slot, expr)
	}
}

// ── ApplyOptBufs: OptIdentity passthrough + unknown-kind default ─────────────

func TestApplyOptBufsIdentity(t *testing.T) {
	a := uop.NewArena(8)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, nil, nil, nil)
	got := ApplyOptBufs(sink, Opt{Kind: OptIdentity}, nil)
	if got.Index() != sink.Index() {
		t.Error("OptIdentity must return the sink unchanged")
	}
}

func TestApplyOptBufsUnknownKind(t *testing.T) {
	a := uop.NewArena(8)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, nil, nil, nil)
	// An out-of-range OptKind hits the default branch → passthrough.
	got := ApplyOptBufs(sink, Opt{Kind: OptKind(99)}, nil)
	if got.Index() != sink.Index() {
		t.Error("unknown OptKind must passthrough unchanged")
	}
}

// ── outputStoreScalar: empty Bufs / nil DType / narrow dtype ─────────────────

func TestOutputStoreScalarEmptyBufs(t *testing.T) {
	if got := outputStoreScalar(schedule.ExecItem{}); got != nil {
		t.Errorf("empty Bufs scalar = %v, want nil", got)
	}
}

func TestOutputStoreScalarNilDType(t *testing.T) {
	item := schedule.ExecItem{Bufs: []schedule.Buffer{{DType: nil}}}
	if got := outputStoreScalar(item); got != nil {
		t.Errorf("nil DType scalar = %v, want nil", got)
	}
}

func TestOutputStoreScalarNarrow(t *testing.T) {
	item := schedule.ExecItem{Bufs: []schedule.Buffer{{DType: uop.Dtypes.BFloat16}}}
	got := outputStoreScalar(item)
	if got != uop.Dtypes.BFloat16 {
		t.Errorf("bf16 store scalar = %v, want BFloat16", got)
	}
}

// ── emitExpr: panic + simple-node branches via minimal lowerer ───────────────

func newEmitLowerer(vars ...uop.UOp) *lowerer {
	l := &lowerer{
		exprOf:        make(map[uint32]string),
		symSlot:       map[uint32]int{},
		symSlotByName: map[string]int{},
	}
	for i, v := range vars {
		l.symSlot[v.Index()] = i
		if va, ok := v.Arg().(uop.VarArg); ok {
			l.symSlotByName[va.Name] = i
		}
	}
	return l
}

func TestEmitExprConstCaches(t *testing.T) {
	a := uop.NewArena(8)
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(2.5), nil)
	l := newEmitLowerer()
	first := l.emitExpr(c)
	if !strings.Contains(first, "2.5") {
		t.Errorf("const expr = %q", first)
	}
	// Second call must return the cached value (covers the exprOf hit path).
	if second := l.emitExpr(c); second != first {
		t.Errorf("cached const expr = %q, want %q", second, first)
	}
}

func TestEmitExprParam(t *testing.T) {
	a := uop.NewArena(8)
	p := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(3), nil)
	l := newEmitLowerer()
	if got := l.emitExpr(p); got != "data3" {
		t.Errorf("param expr = %q, want 'data3'", got)
	}
}

func TestEmitExprDefineVar(t *testing.T) {
	a := uop.NewArena(8)
	n := a.DefineVar("n", 1, 16)
	l := newEmitLowerer(n)
	if got := l.emitExpr(n); got != "i32(params_n.n0)" {
		t.Errorf("DefineVar expr = %q, want 'i32(params_n.n0)'", got)
	}
}

func TestEmitExprDefineVarMissingSlotPanics(t *testing.T) {
	a := uop.NewArena(8)
	n := a.DefineVar("n", 1, 16)
	l := newEmitLowerer() // n absent
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for DefineVar not in symSlot map")
		}
	}()
	_ = l.emitExpr(n)
}

func TestEmitExprRangePanics(t *testing.T) {
	a := uop.NewArena(8)
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c}, int64(0), nil)
	l := newEmitLowerer()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unregistered Range")
		}
	}()
	_ = l.emitExpr(r)
}

// ── Lower: non-SINK AST panics loudly ────────────────────────────────────────

func TestLowerNonSinkPanics(t *testing.T) {
	a := uop.NewArena(8)
	notSink := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	item := schedule.ExecItem{}
	item.SetAst(notSink)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when Lower receives a non-SINK AST")
		}
	}()
	_, _, _, _ = Lower(item)
}

// ── computeDType: narrow promotions + f16 widen toggle ───────────────────────

func TestComputeDTypeNil(t *testing.T) {
	l := &lowerer{}
	a := uop.NewArena(8)
	// A Sink node with no dtype → computeDType returns nil unchanged.
	n := a.New(uop.OpSink, nil, nil, nil, nil)
	if got := l.computeDType(n); got != nil {
		t.Errorf("nil dtype node = %v, want nil", got)
	}
}

func TestComputeDTypeNarrowToF32(t *testing.T) {
	l := &lowerer{}
	a := uop.NewArena(8)
	for _, dt := range []*uop.DType{uop.Dtypes.BFloat16, uop.Dtypes.FP8E4M3, uop.Dtypes.FP8E5M2} {
		n := a.New(uop.OpConst, dt, nil, float64(0), nil)
		if got := l.computeDType(n); got != uop.Dtypes.Float32 {
			t.Errorf("%v should promote to Float32, got %v", dt, got)
		}
	}
}

func TestComputeDTypeF16WidenToggle(t *testing.T) {
	a := uop.NewArena(8)
	n := a.New(uop.OpConst, uop.Dtypes.Float16, nil, float64(0), nil)

	// widenF16 off: f16 stays f16.
	l := &lowerer{widenF16: false}
	if got := l.computeDType(n); got != uop.Dtypes.Float16 {
		t.Errorf("widenF16=false should keep f16, got %v", got)
	}
	// widenF16 on: f16 widens to f32.
	l.widenF16 = true
	if got := l.computeDType(n); got != uop.Dtypes.Float32 {
		t.Errorf("widenF16=true should widen f16 to f32, got %v", got)
	}
}

func TestComputeDTypePassthrough(t *testing.T) {
	l := &lowerer{}
	a := uop.NewArena(8)
	n := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(0), nil)
	if got := l.computeDType(n); got != uop.Dtypes.Int32 {
		t.Errorf("i32 passthrough = %v, want Int32", got)
	}
}

// ── symParamIdxFor: ALU-bound (non-DefineVar) + missing-slot panic ───────────

func TestSymParamIdxForNonDefineVar(t *testing.T) {
	a := uop.NewArena(16)
	c4 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	// OpRange whose bound is a Const (not a DefineVar) → (-1, false).
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c4}, int64(0), nil)
	l := newSymLowerer()
	if slot, ok := l.symParamIdxFor(r); ok || slot != -1 {
		t.Errorf("non-DefineVar bound = (%d, %v), want (-1, false)", slot, ok)
	}
}

func TestSymParamIdxForBareDefineVar(t *testing.T) {
	a := uop.NewArena(16)
	n := a.DefineVar("n", 1, 16)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{n}, int64(0), nil)
	l := newSymLowerer(n)
	if slot, ok := l.symParamIdxFor(r); !ok || slot != 0 {
		t.Errorf("bare DefineVar = (%d, %v), want (0, true)", slot, ok)
	}
}

func TestSymParamIdxForMissingSlotPanics(t *testing.T) {
	a := uop.NewArena(16)
	n := a.DefineVar("n", 1, 16)
	rng := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{n}, int64(0), nil)
	l := newSymLowerer() // n absent
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when DefineVar missing from symSlot")
		}
	}()
	_, _ = l.symParamIdxFor(rng)
}

// ── rangeBoundFactor: const placeholder / concrete / symbolic ────────────────

func TestRangeBoundFactorConstPlaceholder(t *testing.T) {
	a := uop.NewArena(8)
	// A bare OpConst (size-1 dim placeholder) → identity factor.
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	l := newSymLowerer()
	if got := l.rangeBoundFactor(c).renderU32(); got != "1u" {
		t.Errorf("const placeholder factor = %q, want '1u'", got)
	}
}

func TestRangeBoundFactorConcrete(t *testing.T) {
	a := uop.NewArena(8)
	c8 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(8), nil)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c8}, int64(0), nil)
	l := newSymLowerer()
	if got := l.rangeBoundFactor(r).renderU32(); got != "8u" {
		t.Errorf("concrete range factor = %q, want '8u'", got)
	}
}

func TestRangeBoundFactorSymbolic(t *testing.T) {
	a := uop.NewArena(16)
	n := a.DefineVar("n", 1, 16)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{n}, int64(0), nil)
	l := newSymLowerer(n)
	if got := l.rangeBoundFactor(r).renderU32(); got != "params_n.n0" {
		t.Errorf("symbolic range factor = %q, want 'params_n.n0'", got)
	}
}

// ── localSplitDivides: every guard branch ────────────────────────────────────

func TestLocalSplitDividesGuards(t *testing.T) {
	a := uop.NewArena(8)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, nil, nil, nil)

	// localSize <= 0 → false.
	if localSplitDivides(sink, 0, 0) {
		t.Error("localSize=0 must return false")
	}
	// Not a Sink → false.
	notSink := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	if localSplitDivides(notSink, 0, 8) {
		t.Error("non-Sink must return false")
	}
}

func TestLocalSplitDividesNotEnd(t *testing.T) {
	a := uop.NewArena(8)
	// Sink whose Src(0) is not an OpEnd → false.
	inner := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{inner}, nil, nil)
	if localSplitDivides(sink, 0, 8) {
		t.Error("Sink->non-End must return false")
	}
}

func TestLocalSplitDividesDivisibleAndNot(t *testing.T) {
	build := func(size int64) uop.UOp {
		a := uop.NewArena(32)
		c := a.New(uop.OpConst, uop.Dtypes.Index, nil, size, nil)
		r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
		// END src[0] is a placeholder body; loop ranges live at src>=1.
		body := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
		end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{body, r}, nil, nil)
		return a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, nil, nil)
	}
	if !localSplitDivides(build(16), 0, 8) {
		t.Error("16 % 8 == 0 should be divisible")
	}
	if localSplitDivides(build(15), 0, 8) {
		t.Error("15 % 8 != 0 should be non-divisible")
	}
	// Axis index beyond available loop ranges → false.
	if localSplitDivides(build(16), 5, 8) {
		t.Error("axis beyond range count must return false")
	}
}

func TestLocalSplitDividesSymbolicRange(t *testing.T) {
	a := uop.NewArena(32)
	n := a.DefineVar("n", 1, 16)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{n}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	body := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{body, r}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, nil, nil)
	if localSplitDivides(sink, 0, 8) {
		t.Error("symbolic range must report non-divisible (false)")
	}
}

// ── paramDimFactor: dim>0 with a preceding symbolic dim (symIdx counter) ─────

func TestParamDimFactorPrecedingSymDim(t *testing.T) {
	// Shape [0, 0]: dim 1 is symbolic and is preceded by another symbolic dim,
	// so the symIdx-counting loop must advance to index 1.
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0, 0}, SymDimVar: []string{"a", "b"}, SymDimMul: []int64{1, 1}},
		}},
		symSlotByName: map[string]int{"a": 0, "b": 1},
	}
	if got := l.paramDimFactor(0, 1).renderU32(); got != "params_n.n1" {
		t.Errorf("dim1 preceding-sym factor = %q, want 'params_n.n1'", got)
	}
}

func TestParamDimFactorSymVarsFallback(t *testing.T) {
	// Buffer carries no SymDimVar; the name is resolved from item.SymVars.
	l := &lowerer{
		item: schedule.ExecItem{
			Bufs:    []schedule.Buffer{{Shape: []int64{0}}},
			SymVars: []string{"n"},
		},
		symSlotByName: map[string]int{"n": 3},
	}
	if got := l.paramDimFactor(0, 0).renderU32(); got != "params_n.n3" {
		t.Errorf("SymVars-fallback factor = %q, want 'params_n.n3'", got)
	}
}

func TestParamDimFactorNoNamePanics(t *testing.T) {
	// Symbolic dim with no SymDimVar and no SymVars → fail-loud panic.
	l := &lowerer{
		item:          schedule.ExecItem{Bufs: []schedule.Buffer{{Shape: []int64{0}}}},
		symSlotByName: map[string]int{},
	}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when no SymDimVar/SymVars resolves the sym dim")
		}
	}()
	_ = l.paramDimFactor(0, 0)
}

func TestParamDimFactorAffineVarMissingSlotPanics(t *testing.T) {
	l := &lowerer{
		item: schedule.ExecItem{Bufs: []schedule.Buffer{
			{Shape: []int64{0}, SymDimAffine: []schedule.SymDimAffineEntry{
				{Terms: []uop.AffineTerm{{Mul: 1, VarName: "z"}}},
			}},
		}},
		symSlotByName: map[string]int{}, // "z" absent
	}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when affine term var missing from symSlot")
		}
	}()
	_ = l.paramDimFactor(0, 0)
}

// ── ApplyOpts: invalid-Ast early return ──────────────────────────────────────

func TestApplyOptsInvalidAst(t *testing.T) {
	var item schedule.ExecItem // zero Ast
	got := ApplyOpts(item, []Opt{{Kind: OptLocal, Axis: 0, Arg: 8}})
	if got.Ast.Valid() {
		t.Error("invalid Ast should be returned unchanged (still invalid)")
	}
}

// ── emitIndex: scalar / 1-dim / multi-dim concrete / narrow-storage loads ────

// emitIndexLowerer builds a lowerer with the given buffer table.
func emitIndexLowerer(bufs ...schedule.Buffer) *lowerer {
	return &lowerer{
		item:          schedule.ExecItem{Bufs: bufs},
		exprOf:        make(map[uint32]string),
		symSlot:       map[uint32]int{},
		symSlotByName: map[string]int{},
	}
}

func TestEmitIndexScalar(t *testing.T) {
	a := uop.NewArena(16)
	p := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	// Index with only the param src → nDims == 0 → flat "0u".
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{p}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{Shape: []int64{1}, DType: uop.Dtypes.Float32})
	name := l.emitIndex(idx)
	if !strings.HasPrefix(name, "t") {
		t.Errorf("emitIndex name = %q, want t-prefixed", name)
	}
	if len(l.instrs) != 1 || l.instrs[0].Expr != "data0[0u]" {
		t.Errorf("scalar index expr = %q, want 'data0[0u]'", l.instrs[0].Expr)
	}
}

func TestEmitIndex1Dim(t *testing.T) {
	a := uop.NewArena(16)
	p := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)
	ix := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(5), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{p, ix}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{}, schedule.Buffer{}, schedule.Buffer{Shape: []int64{8}, DType: uop.Dtypes.Float32})
	l.emitIndex(idx)
	if l.instrs[0].Expr != "data2[5]" {
		t.Errorf("1-dim index expr = %q, want 'data2[5]'", l.instrs[0].Expr)
	}
}

func TestEmitIndexMultiDimConcrete(t *testing.T) {
	a := uop.NewArena(32)
	p := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	i0 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	i1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{p, i0, i1}, nil, nil)
	// Shape [4, 8]: stride of dim0 is 8 → "(1 * 8) + 2".
	l := emitIndexLowerer(schedule.Buffer{Shape: []int64{4, 8}, DType: uop.Dtypes.Float32})
	l.emitIndex(idx)
	if l.instrs[0].Expr != "data0[(1 * 8) + 2]" {
		t.Errorf("2-dim index expr = %q, want 'data0[(1 * 8) + 2]'", l.instrs[0].Expr)
	}
}

func TestEmitIndexBF16StorageBitcast(t *testing.T) {
	a := uop.NewArena(16)
	p := a.New(uop.OpParam, uop.Dtypes.BFloat16, nil, int64(0), nil)
	ix := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.BFloat16, []uop.UOp{p, ix}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{Shape: []int64{4}, DType: uop.Dtypes.BFloat16})
	l.emitIndex(idx)
	if l.instrs[0].Expr != "bitcast<f32>(data0[0])" {
		t.Errorf("bf16 load expr = %q, want bitcast<f32> wrap", l.instrs[0].Expr)
	}
	if l.instrs[0].DType != uop.Dtypes.Float32 {
		t.Errorf("bf16 load emitted dtype = %v, want Float32", l.instrs[0].DType)
	}
}

func TestEmitIndexF16WidenLoad(t *testing.T) {
	a := uop.NewArena(16)
	p := a.New(uop.OpParam, uop.Dtypes.Float16, nil, int64(0), nil)
	ix := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float16, []uop.UOp{p, ix}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{Shape: []int64{4}, DType: uop.Dtypes.Float16})
	l.widenF16 = true
	l.emitIndex(idx)
	if l.instrs[0].Expr != "f32(data0[0])" {
		t.Errorf("f16 widen load expr = %q, want f32() wrap", l.instrs[0].Expr)
	}
}

func TestEmitIndexMultiDimSymbolicFactor(t *testing.T) {
	// Buffer [4, n]: dim0's stride is the symbolic inner extent n, so the
	// stride accumulator carries a WGSL u32 symbolic factor (params_n.n0).
	a := uop.NewArena(32)
	p := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	i0 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
	i1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{p, i0, i1}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{
		Shape:     []int64{4, 0}, // dim1 symbolic
		SymDimVar: []string{"n"},
		SymDimMul: []int64{1},
		DType:     uop.Dtypes.Float32,
	})
	l.symSlotByName = map[string]int{"n": 0}
	l.emitIndex(idx)
	expr := l.instrs[0].Expr
	// dim0 stride = n (symbolic), dim1 stride = 1.
	if !strings.Contains(expr, "i32(params_n.n0)") {
		t.Errorf("symbolic-factor index expr = %q, want symbolic stride", expr)
	}
}

func TestEmitIndexImageStorage(t *testing.T) {
	a := uop.NewArena(16)
	p := a.New(uop.OpParam, uop.Dtypes.ImageFloat32, nil, int64(0), nil)
	ix := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(3), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.ImageFloat32, []uop.UOp{p, ix}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{Shape: []int64{8}, DType: uop.Dtypes.ImageFloat32})
	l.emitIndex(idx)
	expr := l.instrs[0].Expr
	// Image loads use a 4-way swizzle select-chain over data0[idx/4u].{x,y,z,w}.
	if !strings.Contains(expr, "data0[u32(3) / 4u].w") || !strings.Contains(expr, "select(") {
		t.Errorf("image load expr = %q, want swizzle select-chain", expr)
	}
}

func TestEmitIndexFP8StorageBitcast(t *testing.T) {
	a := uop.NewArena(16)
	p := a.New(uop.OpParam, uop.Dtypes.FP8E4M3, nil, int64(0), nil)
	ix := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	idx := a.New(uop.OpIndex, uop.Dtypes.FP8E4M3, []uop.UOp{p, ix}, nil, nil)
	l := emitIndexLowerer(schedule.Buffer{Shape: []int64{4}, DType: uop.Dtypes.FP8E4M3})
	l.emitIndex(idx)
	if l.instrs[0].Expr != "bitcast<f32>(data0[0])" {
		t.Errorf("fp8 load expr = %q, want bitcast<f32> wrap", l.instrs[0].Expr)
	}
	if l.instrs[0].DType != uop.Dtypes.Float32 {
		t.Errorf("fp8 load emitted dtype = %v, want Float32", l.instrs[0].DType)
	}
}

func TestEmitExprDefineLocalAndBarrier(t *testing.T) {
	a := uop.NewArena(8)
	dl := a.New(uop.OpDefineLocal, uop.Dtypes.Float32, nil, int64(16), nil)
	l := newEmitLowerer()
	name := l.emitExpr(dl)
	if !strings.HasPrefix(name, "sm") {
		t.Errorf("DefineLocal name = %q, want sm-prefixed", name)
	}
	if len(l.instrs) != 1 || l.instrs[0].Kind != InstrDefineLocal {
		t.Fatalf("DefineLocal did not emit InstrDefineLocal: %+v", l.instrs)
	}
	if l.instrs[0].LocalSize != 16 {
		t.Errorf("DefineLocal size = %d, want 16", l.instrs[0].LocalSize)
	}

	bar := a.New(uop.OpBarrier, uop.Dtypes.Void, nil, nil, nil)
	if got := l.emitExpr(bar); got != "" {
		t.Errorf("Barrier expr = %q, want empty", got)
	}
	if l.instrs[len(l.instrs)-1].Kind != InstrBarrier {
		t.Error("Barrier did not emit InstrBarrier")
	}
}
