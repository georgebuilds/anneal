package shape

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// symVar builds a fresh arena-backed symbolic variable with inclusive bounds
// [min, max] and wraps it as a SymInt. Used to exercise the symbolic paths of
// the Sint algebra and the bounds walker without an SMT solver.
func symVar(min, max int64) (SymInt, *uop.Arena) {
	a := uop.NewArena(64)
	n := a.DefineVar("n", min, max)
	return SymInt{Node: n}, a
}

// ── constant Sint arithmetic ────────────────────────────────────────────────

func TestConstArithmetic(t *testing.T) {
	cases := []struct {
		name string
		got  int64
		want int64
	}{
		{"add", cv(Add(Const(3), Const(4))), 7},
		{"add-zero-l", cv(Add(Const(0), Const(9))), 9},
		{"add-zero-r", cv(Add(Const(9), Const(0))), 9},
		{"sub", cv(Sub(Const(10), Const(4))), 6},
		{"sub-zero", cv(Sub(Const(5), Const(0))), 5},
		{"mul", cv(Mul(Const(6), Const(7))), 42},
		{"mul-zero", cv(Mul(Const(0), Const(7))), 0},
		{"mul-one", cv(Mul(Const(1), Const(7))), 7},
		{"neg", cv(Neg(Const(8))), -8},
		{"idiv", cv(IDiv(Const(17), Const(5))), 3},
		{"idiv-exact", cv(IDiv(Const(20), Const(4))), 5},
		{"mod", cv(Mod(Const(17), Const(5))), 2},
		{"mod-zero", cv(Mod(Const(20), Const(4))), 0},
		{"max-a", cv(SintMax(Const(9), Const(2))), 9},
		{"max-b", cv(SintMax(Const(2), Const(9))), 9},
		{"max-eq", cv(SintMax(Const(5), Const(5))), 5},
		{"min-a", cv(SintMin(Const(2), Const(9))), 2},
		{"min-b", cv(SintMin(Const(9), Const(2))), 2},
		{"min-eq", cv(SintMin(Const(5), Const(5))), 5},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %d want %d", c.name, c.got, c.want)
		}
	}
}

func TestConstComparisons(t *testing.T) {
	if !Eq(Const(4), Const(4)) || Eq(Const(4), Const(5)) {
		t.Error("Eq")
	}
	if !Lt(Const(3), Const(4)) || Lt(Const(4), Const(4)) || Lt(Const(5), Const(4)) {
		t.Error("Lt")
	}
	if !Le(Const(4), Const(4)) || !Le(Const(3), Const(4)) || Le(Const(5), Const(4)) {
		t.Error("Le")
	}
	if !EqI(Const(7), 7) || EqI(Const(7), 8) {
		t.Error("EqI")
	}
}

func TestCVExported(t *testing.T) {
	if CV(Const(42)) != 42 {
		t.Errorf("CV(Const(42))=%d", CV(Const(42)))
	}
}

// ── symbolic Sint construction (builds UOp nodes) ───────────────────────────

func TestSymbolicArithmeticBuildsNodes(t *testing.T) {
	n, _ := symVar(1, 1024)

	// Each op with a symbolic operand should produce a SymInt (no fold).
	checks := []struct {
		name string
		s    Sint
	}{
		{"add", Add(n, Const(3))},
		{"add-rev", Add(Const(3), n)},
		{"sub", Sub(n, Const(2))},
		{"sub-rev", Sub(Const(2), n)},
		{"mul", Mul(n, Const(4))},
		{"mul-rev", Mul(Const(4), n)},
		{"neg", Neg(n)},
		{"idiv", IDiv(n, Const(2))},
		{"idiv-rev", IDiv(Const(64), n)},
		{"mod", Mod(n, Const(3))},
		{"mod-rev", Mod(Const(64), n)},
		{"max", SintMax(n, Const(5))},
		{"min", SintMin(n, Const(5))},
	}
	for _, c := range checks {
		if _, ok := c.s.(SymInt); !ok {
			t.Errorf("%s: expected SymInt, got %T", c.name, c.s)
		}
		if _, ok := c.s.ConstValue(); ok {
			t.Errorf("%s: symbolic Sint should not have a const value", c.name)
		}
	}
}

func TestSymbolicIdentityFolds(t *testing.T) {
	n, _ := symVar(1, 1024)

	// Folds that collapse back to const or to the symbolic operand identity.
	if v, ok := Add(n, Const(0)).(SymInt); !ok || v.Node != n.Node {
		t.Error("Add(n,0) should fold to n")
	}
	if v, ok := Add(Const(0), n).(SymInt); !ok || v.Node != n.Node {
		t.Error("Add(0,n) should fold to n")
	}
	if v, ok := Sub(n, Const(0)).(SymInt); !ok || v.Node != n.Node {
		t.Error("Sub(n,0) should fold to n")
	}
	if v, ok := Mul(n, Const(1)).(SymInt); !ok || v.Node != n.Node {
		t.Error("Mul(n,1) should fold to n")
	}
	if v, ok := Mul(Const(1), n).(SymInt); !ok || v.Node != n.Node {
		t.Error("Mul(1,n) should fold to n")
	}
	if got, ok := Mul(n, Const(0)).ConstValue(); !ok || got != 0 {
		t.Error("Mul(n,0) should fold to 0")
	}
	if got, ok := Mul(Const(0), n).ConstValue(); !ok || got != 0 {
		t.Error("Mul(0,n) should fold to 0")
	}
}

func TestSymArenaPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("symArena with no SymInt operand should panic")
		}
	}()
	symArena(Const(1), Const(2))
}

func TestSymArena1Panics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("symArena1 with no SymInt operand should panic")
		}
	}()
	symArena1(Const(1))
}

func TestCVPanicsOnSymbolic(t *testing.T) {
	n, _ := symVar(1, 8)
	defer func() {
		if recover() == nil {
			t.Error("cv on symbolic Sint should panic")
		}
	}()
	cv(n)
}

// ── bounds: ResolveNonNeg / ResolveLE / boundsOfUOp ─────────────────────────

func TestResolveNonNegConcrete(t *testing.T) {
	if !ResolveNonNeg(Const(0)) || !ResolveNonNeg(Const(7)) {
		t.Error("non-negative consts should resolve true")
	}
	if ResolveNonNeg(Const(-1)) {
		t.Error("negative const should resolve false")
	}
}

func TestResolveNonNegSymbolic(t *testing.T) {
	// n in [1,1024] is provably >= 0.
	n, _ := symVar(1, 1024)
	if !ResolveNonNeg(n) {
		t.Error("n in [1,1024] should resolve non-neg")
	}
	// n in [1,8] minus 3 spans [-2,5] → not provably non-negative.
	if ResolveNonNeg(Sub(n, Const(3))) {
		t.Error("n-3 with n in [1,8] should not resolve non-neg")
	}
	// n*4 in [4,4096] is provably non-negative.
	if !ResolveNonNeg(Mul(n, Const(4))) {
		t.Error("n*4 should resolve non-neg")
	}
}

func TestResolveLE(t *testing.T) {
	// Concrete.
	if !ResolveLE(Const(3), Const(5)) || ResolveLE(Const(5), Const(3)) {
		t.Error("concrete ResolveLE")
	}
	// Identical symbolic node: a <= a.
	n, _ := symVar(1, 16)
	if !ResolveLE(n, n) {
		t.Error("a <= a should resolve true via identity short-circuit")
	}
	// n in [1,16] vs Const(0): 0 <= n provable (n - 0 has min 1 >= 0).
	if !ResolveLE(Const(0), n) {
		t.Error("0 <= n should resolve true")
	}
	// n <= Const(20) provable (20 - n has min 4 >= 0 with n in [1,16]).
	if !ResolveLE(n, Const(20)) {
		t.Error("n <= 20 should resolve true for n in [1,16]")
	}
	// n <= Const(0) is not provable (n is at least 1).
	if ResolveLE(n, Const(0)) {
		t.Error("n <= 0 should resolve false")
	}
	// Two non-SymInt non-equal operands with no arena: unprovable → false.
	// (ConstValue path already covered; this guards the no-SymInt branch.)
	// Two distinct symbolic with no relation: not provable.
	a := uop.NewArena(64)
	x := SymInt{Node: a.DefineVar("x", 1, 4)}
	y := SymInt{Node: a.DefineVar("y", 100, 200)}
	// x in [1,4], y in [100,200]: y - x in [96,199] >= 0 → provable x <= y.
	if !ResolveLE(x, y) {
		t.Error("x<=y with disjoint ranges should resolve true")
	}
	if ResolveLE(y, x) {
		t.Error("y<=x should resolve false")
	}
}

func TestBoundsOfUOpOps(t *testing.T) {
	a := uop.NewArena(128)
	n := a.DefineVar("n", 2, 10) // inclusive [2,10]

	// DefineVar bounds.
	if b := boundsOfUOp(n); !b.valid || b.min != 2 || b.max != 10 {
		t.Errorf("DefineVar bounds = %+v want [2,10]", b)
	}

	mkConst := func(v int64) uop.UOp {
		return a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil)
	}
	c3 := mkConst(3)

	// Add: [2,10]+[3,3] = [5,13].
	add := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{n, c3}, nil, nil)
	if b := boundsOfUOp(add); !b.valid || b.min != 5 || b.max != 13 {
		t.Errorf("add bounds = %+v want [5,13]", b)
	}
	// Sub: [2,10]-[3,3] = [-1,7].
	sub := a.New(uop.OpSub, uop.Dtypes.Index, []uop.UOp{n, c3}, nil, nil)
	if b := boundsOfUOp(sub); !b.valid || b.min != -1 || b.max != 7 {
		t.Errorf("sub bounds = %+v want [-1,7]", b)
	}
	// Mul: [2,10]*[3,3] = [6,30].
	mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{n, c3}, nil, nil)
	if b := boundsOfUOp(mul); !b.valid || b.min != 6 || b.max != 30 {
		t.Errorf("mul bounds = %+v want [6,30]", b)
	}
	// Neg: -[2,10] = [-10,-2].
	neg := a.New(uop.OpNeg, uop.Dtypes.Index, []uop.UOp{n}, nil, nil)
	if b := boundsOfUOp(neg); !b.valid || b.min != -10 || b.max != -2 {
		t.Errorf("neg bounds = %+v want [-10,-2]", b)
	}
	// IDiv with sign-definite positive divisor: [2,10]/[3,3] = [0,3].
	div := a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{n, c3}, nil, nil)
	if b := boundsOfUOp(div); !b.valid || b.min != 0 || b.max != 3 {
		t.Errorf("idiv bounds = %+v want [0,3]", b)
	}
	// Mod by constant 3: [0,2].
	mod := a.New(uop.OpMod, uop.Dtypes.Index, []uop.UOp{n, c3}, nil, nil)
	if b := boundsOfUOp(mod); !b.valid || b.min != 0 || b.max != 2 {
		t.Errorf("mod bounds = %+v want [0,2]", b)
	}
}

func TestBoundsOfUOpInvalid(t *testing.T) {
	// Invalid (zero) UOp returns invalid bounds.
	if b := boundsOfUOp(uop.UOp{}); b.valid {
		t.Error("zero UOp should be invalid")
	}
	a := uop.NewArena(64)
	n := a.DefineVar("n", -2, 5) // straddles zero
	c := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(2), nil)
	// IDiv with divisor const(2) is sign-definite positive → valid.
	div := a.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{c, n}, nil, nil)
	// Here divisor is n which straddles zero → b.min*b.max <= 0 → invalid.
	if b := boundsOfUOp(div); b.valid {
		t.Error("idiv by sign-straddling divisor should be invalid")
	}
}

// ── slice helpers: Product / SymbolicProduct / HasSymbolic ──────────────────

func TestProduct(t *testing.T) {
	got := cv(Product([]Sint{Const(2), Const(3), Const(4)}))
	if got != 24 {
		t.Errorf("Product = %d want 24", got)
	}
	if cv(Product(nil)) != 1 {
		t.Error("empty Product should be 1")
	}
}

func TestSymbolicProduct(t *testing.T) {
	// All concrete behaves like Product.
	if got := cv(SymbolicProduct([]Sint{Const(2), Const(3), Const(4)})); got != 24 {
		t.Errorf("SymbolicProduct concrete = %d want 24", got)
	}
	// Size-1 dims are dropped.
	if got := cv(SymbolicProduct([]Sint{Const(4), Const(1), Const(3)})); got != 12 {
		t.Errorf("SymbolicProduct drop-1 = %d want 12", got)
	}
	// All ones → 1.
	if got := cv(SymbolicProduct([]Sint{Const(1), Const(1)})); got != 1 {
		t.Errorf("SymbolicProduct all-ones = %d want 1", got)
	}
	if got := cv(SymbolicProduct(nil)); got != 1 {
		t.Error("empty SymbolicProduct should be 1")
	}
	// Mixed symbolic: [n, 4] should build a symbolic node.
	n, _ := symVar(1, 8)
	p := SymbolicProduct([]Sint{n, Const(4)})
	if _, ok := p.(SymInt); !ok {
		t.Errorf("SymbolicProduct with symbolic dim should be SymInt, got %T", p)
	}
	// Single symbolic dim returns it directly.
	if v, ok := SymbolicProduct([]Sint{n}).(SymInt); !ok || v.Node != n.Node {
		t.Error("SymbolicProduct of single sym should return that sym")
	}
	// prod([n,4]) == prod([n,4,1]) under identity drop.
	p2 := SymbolicProduct([]Sint{n, Const(4), Const(1)})
	if !SintEqual(p, p2) {
		t.Error("size-1 should not change symbolic product")
	}
}

func TestHasSymbolic(t *testing.T) {
	if HasSymbolic([]Sint{Const(1), Const(2)}) {
		t.Error("all-concrete should not be symbolic")
	}
	if HasSymbolic(nil) {
		t.Error("nil should not be symbolic")
	}
	n, _ := symVar(1, 8)
	if !HasSymbolic([]Sint{Const(1), n}) {
		t.Error("slice with SymInt should be symbolic")
	}
}

func TestSintEqual(t *testing.T) {
	if !SintEqual(Const(5), Const(5)) {
		t.Error("equal consts")
	}
	if SintEqual(Const(5), Const(6)) {
		t.Error("unequal consts")
	}
	n, _ := symVar(1, 8)
	if !SintEqual(n, n) {
		t.Error("same sym node")
	}
	if SintEqual(n, Const(5)) {
		t.Error("sym vs const")
	}
}

func TestSintFromShapeDim(t *testing.T) {
	a := uop.NewArena(64)
	// Concrete dim.
	d := uop.ShapeDim{Sym: false, V: 12}
	s := SintFromShapeDim(a, d)
	if v, ok := s.ConstValue(); !ok || v != 12 {
		t.Errorf("concrete ShapeDim → %v", s)
	}
	// Symbolic dim — VarName must be registered in the arena before rebuild.
	a.DefineVar("b", 1, 32)
	dsym := uop.ShapeDim{Sym: true, VarName: "b", Mul: 1}
	ssym := SintFromShapeDim(a, dsym)
	if _, ok := ssym.(SymInt); !ok {
		t.Errorf("symbolic ShapeDim → %T, want SymInt", ssym)
	}
}

func TestSintFromShapeDimMul(t *testing.T) {
	a := uop.NewArena(64)
	a.DefineVar("b", 1, 16)
	// Symbolic dim with a multiplier, e.g. 4*b.
	d := uop.ShapeDim{Sym: true, VarName: "b", Mul: 4}
	s := SintFromShapeDim(a, d)
	if _, ok := s.(SymInt); !ok {
		t.Errorf("4*b ShapeDim → %T, want SymInt", s)
	}
}

// ── slice conversion helpers ────────────────────────────────────────────────

func TestAsSintsAsInts(t *testing.T) {
	if AsSints(nil) != nil {
		t.Error("AsSints(nil) should be nil")
	}
	if AsInts(nil) != nil {
		t.Error("AsInts(nil) should be nil")
	}
	in := []int64{1, 2, 3}
	out := AsInts(AsSints(in))
	if len(out) != 3 || out[0] != 1 || out[2] != 3 {
		t.Errorf("round-trip = %v", out)
	}
}

func TestAsMaskAsIntMask(t *testing.T) {
	if AsMaskSint(nil) != nil {
		t.Error("AsMaskSint(nil) should be nil")
	}
	if AsIntMask(nil) != nil {
		t.Error("AsIntMask(nil) should be nil")
	}
	m := [][2]int64{{0, 4}, {1, 3}}
	out := AsIntMask(AsMaskSint(m))
	if out[0][1] != 4 || out[1][0] != 1 {
		t.Errorf("mask round-trip = %v", out)
	}
}

func TestSintShapesEqual(t *testing.T) {
	if !SintShapesEqual(AsSints([]int64{2, 3}), AsSints([]int64{2, 3})) {
		t.Error("equal shapes")
	}
	if SintShapesEqual(AsSints([]int64{2, 3}), AsSints([]int64{2, 4})) {
		t.Error("unequal value")
	}
	if SintShapesEqual(AsSints([]int64{2}), AsSints([]int64{2, 3})) {
		t.Error("unequal length")
	}
	// Const vs Sym mismatch.
	n, _ := symVar(1, 8)
	if SintShapesEqual([]Sint{n}, []Sint{Const(2)}) {
		t.Error("sym vs const should differ")
	}
	if SintShapesEqual([]Sint{Const(2)}, []Sint{n}) {
		t.Error("const vs sym should differ")
	}
	if !SintShapesEqual([]Sint{n}, []Sint{n}) {
		t.Error("same sym node should be equal")
	}
}
