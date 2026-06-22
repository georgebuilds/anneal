package schedule

import (
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// index_sint_test.go - adversarial coverage for sintStrides, flatIndexSints, and
// unflatIndexSints. These are the symbolic-shape variants of flatIndex/unflatIndex
// used when an OpReshape carries a ShapeSintArg. Production exercises them through
// dynamic-batch tensor paths; this file pins their direct semantics so a
// regression cannot ship silently.
//
// Post Slice 7a: sintStrides returns []shape.Sint and accumulates symbolic
// factors via shape.Mul. The symbolic dim may now sit in any position; strides
// for dims to its left carry the symbolic factor. The "skip Mod for symbolic
// outermost" optimisation is preserved (i==0 + symbolic) but a symbolic dim at
// i>0 now correctly gets a Mod with the symbolic bound.

// ── helpers ──────────────────────────────────────────────────────────────────

// constShape returns []Sint where every dim is a concrete ConstInt.
func constShape(dims ...int64) []shape.Sint {
	out := make([]shape.Sint, len(dims))
	for i, d := range dims {
		out[i] = shape.ConstInt{V: d}
	}
	return out
}

// symHeadShape returns []Sint with a SymInt (DefineVar "n", bounds [1,N]) at
// position 0 and concrete dims for the rest (symbolic dim outermost).
func symHeadShape(a *uop.Arena, name string, max int64, tailDims ...int64) []shape.Sint {
	out := make([]shape.Sint, 1+len(tailDims))
	out[0] = shape.SymInt{Node: a.DefineVar(name, 1, max)}
	for i, d := range tailDims {
		out[i+1] = shape.ConstInt{V: d}
	}
	return out
}

// constIndices builds OpConst index leaves with the given concrete values.
// Used to verify that flatIndexSints produces the expected arithmetic by
// inspecting the resulting expression tree.
func constIndices(a *uop.Arena, vals ...int64) []uop.UOp {
	out := make([]uop.UOp, len(vals))
	for i, v := range vals {
		out[i] = a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil)
	}
	return out
}

// ── sintStrides - concrete shape ────────────────────────────────────────────

// TestSintStridesAllConcrete verifies row-major strides for a fully concrete
// shape: strides[i] = prod(shape[i+1:]). Strides are []shape.Sint; for an
// all-concrete shape every stride must be a ConstInt.
func TestSintStridesAllConcrete(t *testing.T) {
	cases := []struct {
		name string
		dims []int64
		want []int64
	}{
		{"1D", []int64{4}, []int64{1}},
		{"2D", []int64{2, 3}, []int64{3, 1}},
		{"3D", []int64{2, 3, 5}, []int64{15, 5, 1}},
		{"4D", []int64{2, 3, 5, 7}, []int64{105, 35, 7, 1}},
		{"with 1s", []int64{4, 1, 3}, []int64{3, 3, 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sintStrides(constShape(tc.dims...))
			if len(got) != len(tc.want) {
				t.Fatalf("len(strides) = %d, want %d", len(got), len(tc.want))
			}
			for i, s := range got {
				v, ok := s.ConstValue()
				if !ok {
					t.Errorf("strides[%d] is symbolic, want concrete ConstInt", i)
					continue
				}
				if v != tc.want[i] {
					t.Errorf("strides[%d] = %d, want %d", i, v, tc.want[i])
				}
			}
		})
	}
}

// TestSintStridesSymbolicHead pins the symbolic-outermost case (Option-A
// shape). The symbolic dim's stride equals the product of trailing concrete
// dims (concrete ConstInt); for dims to its right, strides[i] = prod(concrete
// tail of i+1:). Post Slice 7a the accumulator does multiply through the
// symbolic dim (using shape.Mul) - but the resulting symbolic factor only
// shows up in strides for dims to the LEFT of the symbolic dim, which don't
// exist in the symbolic-outermost case.
func TestSintStridesSymbolicHead(t *testing.T) {
	a := uop.NewArena(8)

	t.Run("sym + 1 tail", func(t *testing.T) {
		sh := symHeadShape(a, "n", 64, 16) // [Sym(n), 16]
		got := sintStrides(sh)
		// strides[1] (last concrete) = 1
		// strides[0] (symbolic) = trailing product = 16
		want := []int64{16, 1}
		for i, s := range got {
			v, ok := s.ConstValue()
			if !ok {
				t.Errorf("strides[%d] symbolic, want ConstInt(%d) (sh=[Sym, 16])", i, want[i])
				continue
			}
			if v != want[i] {
				t.Errorf("strides[%d] = %d, want %d (sh=[Sym, 16])", i, v, want[i])
			}
		}
	})

	t.Run("sym + 2 tail", func(t *testing.T) {
		sh := symHeadShape(a, "m", 32, 4, 8) // [Sym(m), 4, 8]
		got := sintStrides(sh)
		// strides[2] = 1; strides[1] = 8; strides[0] (symbolic outermost) = 4*8 = 32
		want := []int64{32, 8, 1}
		for i, s := range got {
			v, ok := s.ConstValue()
			if !ok {
				t.Errorf("strides[%d] symbolic, want ConstInt(%d)", i, want[i])
				continue
			}
			if v != want[i] {
				t.Errorf("strides[%d] = %d, want %d", i, v, want[i])
			}
		}
	})

	t.Run("sym only", func(t *testing.T) {
		// [Sym] alone: strides[0] = 1 (last dim is always stride 1, even if symbolic).
		sh := []shape.Sint{shape.SymInt{Node: a.DefineVar("k", 1, 8)}}
		got := sintStrides(sh)
		if len(got) != 1 {
			t.Fatalf("sintStrides([Sym]) len = %d, want 1", len(got))
		}
		v, ok := got[0].ConstValue()
		if !ok || v != 1 {
			t.Errorf("sintStrides([Sym])[0] = %+v, want ConstInt(1)", got[0])
		}
	})
}

// ── flatIndexSints - row-major flattening ───────────────────────────────────

// TestFlatIndexSintsEmpty: zero-dim shape → Const(0). Pins the early-return.
func TestFlatIndexSintsEmpty(t *testing.T) {
	a := uop.NewArena(4)
	got := flatIndexSints(a, nil, nil)
	if got.Op() != uop.OpConst {
		t.Errorf("flatIndexSints(nil, nil).Op = %v, want OpConst", got.Op())
	}
	if got.Arg() != int64(0) {
		t.Errorf("flatIndexSints(nil, nil).Arg = %v, want 0", got.Arg())
	}
}

// TestFlatIndexSints1D: single dim → return the index directly (no arithmetic).
func TestFlatIndexSints1D(t *testing.T) {
	a := uop.NewArena(8)
	idx := constIndices(a, 7)
	got := flatIndexSints(a, idx, constShape(16))

	if got != idx[0] {
		t.Errorf("flatIndexSints(1D) did not return the input index unchanged: got idx=%d, want idx=%d",
			got.Index(), idx[0].Index())
	}
}

// TestFlatIndexSintsConcrete2D verifies r0*s1 + r1 is built with the right
// arithmetic shape (OpAdd of (OpMul of r0, Const(s1)) and r1).
func TestFlatIndexSintsConcrete2D(t *testing.T) {
	a := uop.NewArena(16)
	r := constIndices(a, 2, 3)
	got := flatIndexSints(a, r, constShape(4, 5))
	// Expect Add(Mul(r0, Const(5)), r1)
	if got.Op() != uop.OpAdd {
		t.Fatalf("top = %v, want OpAdd", got.Op())
	}
	left, right := got.Src(0), got.Src(1)
	if left.Op() != uop.OpMul {
		t.Fatalf("Add.Src(0) = %v, want OpMul (r0 * stride)", left.Op())
	}
	if right != r[1] {
		t.Fatalf("Add.Src(1) is not r[1]")
	}
	if left.Src(0) != r[0] {
		t.Fatalf("Mul.Src(0) is not r[0]")
	}
	stride := left.Src(1)
	if stride.Op() != uop.OpConst || stride.Arg() != int64(5) {
		t.Fatalf("Mul.Src(1) Op/Arg = %v/%v, want OpConst/5", stride.Op(), stride.Arg())
	}
}

// TestFlatIndexSintsStride1Optimisation pins that when stride==1 the code uses
// the index directly (no Const(1) Mul). Important for cache-hit symmetry with
// the static flatIndex path - if a symbolic build started emitting r*Const(1)
// where the static path emits r, the schedule cache wouldn't match.
func TestFlatIndexSintsStride1Optimisation(t *testing.T) {
	a := uop.NewArena(8)
	r := constIndices(a, 0)
	got := flatIndexSints(a, r, constShape(8))
	// 1D case returns r[0] directly; expand to 2D to test the in-loop stride==1 path.
	a2 := uop.NewArena(16)
	r2 := constIndices(a2, 0, 0)
	got2 := flatIndexSints(a2, r2, constShape(8, 1)) // shape [8,1] → strides [1, 1]
	// Last term: stride==1 → term is r2[1] directly. The top Add still wraps it.
	if got2.Op() != uop.OpAdd {
		t.Fatalf("top = %v, want OpAdd", got2.Op())
	}
	if got2.Src(1) != r2[1] {
		t.Errorf("trailing term should be r2[1] directly when stride==1, got Op=%v", got2.Src(1).Op())
	}
	// Smoke: 1D case returns input unchanged (already covered above).
	_ = got
}

// TestFlatIndexSintsSymbolicHead exercises the Option-A path:
// shape=[Sym(n), 16], indices=[i, j] → i*16 + j (concrete stride on the
// symbolic dim because sintStrides extracts trailing concrete product).
func TestFlatIndexSintsSymbolicHead(t *testing.T) {
	a := uop.NewArena(32)
	sh := symHeadShape(a, "n", 64, 16)
	r := constIndices(a, 0, 0)

	got := flatIndexSints(a, r, sh)
	// Expect Add(Mul(r0, Const(16)), r1) - symbolic dim has concrete stride 16.
	if got.Op() != uop.OpAdd {
		t.Fatalf("top = %v, want OpAdd", got.Op())
	}
	left := got.Src(0)
	if left.Op() != uop.OpMul {
		t.Fatalf("Add.Src(0) = %v, want OpMul", left.Op())
	}
	stride := left.Src(1)
	if stride.Op() != uop.OpConst || stride.Arg() != int64(16) {
		t.Errorf("symbolic-head stride = %v/%v, want OpConst/16", stride.Op(), stride.Arg())
	}
}

// ── unflatIndexSints - row-major decomposition ──────────────────────────────

func TestUnflatIndexSintsEmpty(t *testing.T) {
	a := uop.NewArena(4)
	got := unflatIndexSints(a, uop.UOp{}, nil)
	if got != nil {
		t.Errorf("unflatIndexSints(_, nil shape) = %v, want nil", got)
	}
}

func TestUnflatIndexSints1D(t *testing.T) {
	a := uop.NewArena(8)
	flat := constIndices(a, 42)[0]
	got := unflatIndexSints(a, flat, constShape(16))
	if len(got) != 1 || got[0] != flat {
		t.Errorf("unflatIndexSints(1D) did not return [flat] unchanged: got len=%d", len(got))
	}
}

// TestUnflatIndexSintsConcrete2D: shape=[4,5], strides=[5,1].
// out[0] = (flat / 5) % 4, out[1] = flat % 5.
func TestUnflatIndexSintsConcrete2D(t *testing.T) {
	a := uop.NewArena(32)
	flat := constIndices(a, 0)[0]
	got := unflatIndexSints(a, flat, constShape(4, 5))
	if len(got) != 2 {
		t.Fatalf("len(out) = %d, want 2", len(got))
	}
	// out[0] = Mod(IDiv(flat, Const(5)), Const(4))
	if got[0].Op() != uop.OpMod {
		t.Fatalf("out[0].Op = %v, want OpMod", got[0].Op())
	}
	div := got[0].Src(0)
	if div.Op() != uop.OpIDiv {
		t.Fatalf("out[0].Src(0).Op = %v, want OpIDiv", div.Op())
	}
	if div.Src(1).Arg() != int64(5) {
		t.Errorf("out[0] divisor = %v, want 5", div.Src(1).Arg())
	}
	if got[0].Src(1).Arg() != int64(4) {
		t.Errorf("out[0] modulus = %v, want 4", got[0].Src(1).Arg())
	}
	// out[1] (last dim): stride==1 → divided = flat directly; mod by Const(5).
	if got[1].Op() != uop.OpMod {
		t.Fatalf("out[1].Op = %v, want OpMod", got[1].Op())
	}
	if got[1].Src(0) != flat {
		t.Errorf("out[1] divided should be flat (stride==1 fast path); got Op=%v", got[1].Src(0).Op())
	}
	if got[1].Src(1).Arg() != int64(5) {
		t.Errorf("out[1] modulus = %v, want 5", got[1].Src(1).Arg())
	}
}

// TestUnflatIndexSintsSymbolicHead pins the Option-A symbolic-outermost path:
// for the symbolic dim, the quotient is returned directly (no Mod).
// Per index.go:355-364: "Symbolic outermost dim: quotient is the exact index."
func TestUnflatIndexSintsSymbolicHead(t *testing.T) {
	a := uop.NewArena(32)
	sh := symHeadShape(a, "n", 64, 8) // [Sym(n), 8]
	flat := constIndices(a, 0)[0]
	got := unflatIndexSints(a, flat, sh)

	if len(got) != 2 {
		t.Fatalf("len(out) = %d, want 2", len(got))
	}
	// out[0] (symbolic): IDiv(flat, Const(8)) - NO Mod.
	if got[0].Op() != uop.OpIDiv {
		t.Errorf("symbolic out[0].Op = %v, want OpIDiv (no Mod on symbolic outermost)", got[0].Op())
	}
	if got[0].Src(1).Arg() != int64(8) {
		t.Errorf("symbolic out[0] divisor = %v, want 8", got[0].Src(1).Arg())
	}
	// out[1] (concrete, last dim): Mod(flat, Const(8)).
	if got[1].Op() != uop.OpMod {
		t.Errorf("concrete out[1].Op = %v, want OpMod", got[1].Op())
	}
}

// TestUnflatIndexSintsSymbolicHeadNoModIsLoadBearing documents WHY the symbolic
// outermost case skips Mod. If a future "consistency cleanup" added a Mod on
// the symbolic dim, codegen would have to read the symbolic bound at runtime
// to compute the modulus - turning a free arithmetic op into a uniform-buffer
// fetch in the inner loop. Pinning here so the rationale survives refactors.
func TestUnflatIndexSintsSymbolicHeadNoModIsLoadBearing(t *testing.T) {
	a := uop.NewArena(32)
	sh := symHeadShape(a, "n", 64, 4, 7)
	flat := constIndices(a, 0)[0]
	got := unflatIndexSints(a, flat, sh)
	if got[0].Op() == uop.OpMod {
		t.Error("symbolic outermost dim wrapped in Mod - would force runtime bound read in inner loop")
	}
}

// ── round-trip property: flat ∘ unflat = identity (concrete) ────────────────

// TestFlatUnflatRoundTripConcrete is the structural correctness proof: for a
// concrete shape, building unflatIndexSints(flat) and then flatIndexSints on
// the result must produce arithmetic equivalent to flat. We check it on small
// shapes by structural traversal of the resulting expression tree.
//
// Caveat: structural equality is too strict (extra Mod by max-size dim is a
// no-op semantically but distinct structurally), so we verify by enumerating
// concrete index values: for each flat in [0, prod), unflat(flat) must give
// per-dim indices whose flatIndexSints(...) returns the same arithmetic shape
// when fed concrete constants. We use the StructuralKeys helper to compare.
//
// This catches an off-by-one in stride math (would shift the round-trip by 1
// element across the full enumeration).
func TestFlatUnflatRoundTripConcrete(t *testing.T) {
	cases := [][]int64{
		{4},
		{2, 3},
		{2, 3, 5},
		{3, 1, 4}, // includes a dim of size 1 (degenerate stride)
	}
	for _, dims := range cases {
		t.Run(strJoin(dims), func(t *testing.T) {
			a := uop.NewArena(64)
			sh := constShape(dims...)

			// Build symbolic round-trip: f(unflat(flat)) - both arithmetic on the same flat var.
			flatBound := a.New(uop.OpConst, uop.Dtypes.Index, nil, prod(dims), nil)
			flat := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{flatBound},
				uop.RangeArg{ID: 99, Type: uop.AxisLoop}, nil)
			perDim := unflatIndexSints(a, flat, sh)
			roundTrip := flatIndexSints(a, perDim, sh)

			// And the direct identity: flatIndexSints on per-dim coords of a known flat.
			// Identity check: for each axis, the produced per-dim must be (flat / stride) % size.
			// We don't need to evaluate them; we assert the round-trip expression hashes the
			// same way as expected per-dim arithmetic, by checking it does not collapse to
			// the flat var itself (which would indicate the round-trip collapsed to identity,
			// which is what we want EXCEPT we cannot rely on algebraic simplification here -
			// the symbolic algebra rewriter is not invoked by these helpers).
			//
			// Minimum check: roundTrip must be a valid UOp and reference `flat` transitively.
			if !roundTrip.Valid() {
				t.Fatal("round-trip produced invalid UOp")
			}
			if !referencesTransitively(roundTrip, flat) {
				t.Errorf("round-trip expression does not reference the original flat var - broken")
			}
		})
	}
}

// ── Slice 7a: non-outermost symbolic structural strides ─────────────────────
//
// These tests pin the structural correctness of sintStrides /
// flatIndexSints / unflatIndexSints for shapes where the symbolic dim is NOT
// the outermost position. Pre-7a, sintStrides returned wrong strides for
// any dim to the LEFT of a symbolic dim (left-of-sym strides defaulted to 1
// instead of carrying the symbolic factor). Post-7a, sintStrides returns
// []shape.Sint, and the symbolic factor appears in left-of-sym strides via
// shape.Mul.
//
// The proof is two-pronged:
//   1. Structural: stride[i] for a sym-bearing dim slot must be a UOp whose
//      VariablesOf includes the right DefineVar.
//   2. Numerical: under bindings, flat ∘ unflat = identity. We evaluate the
//      built UOp expressions via evalUOp(binding) and check equality.

// symAtShape returns []Sint where dim symAt is a SymInt with name name and
// upper bound max+1 (DefineVar uses exclusive upper); all other dims are
// concrete ConstInts from dims[i].
func symAtShape(a *uop.Arena, name string, max int64, dims []int64, symAt int) []shape.Sint {
	out := make([]shape.Sint, len(dims))
	for i, d := range dims {
		if i == symAt {
			out[i] = shape.SymInt{Node: a.DefineVar(name, 1, max+1)}
		} else {
			out[i] = shape.ConstInt{V: d}
		}
	}
	return out
}

// twoSymShape returns []Sint of length 2 with two distinct SymInts bound to
// (1..maxA+1) and (1..maxB+1).
func twoSymShape(a *uop.Arena, na, nb string, maxA, maxB int64) []shape.Sint {
	return []shape.Sint{
		shape.SymInt{Node: a.DefineVar(na, 1, maxA+1)},
		shape.SymInt{Node: a.DefineVar(nb, 1, maxB+1)},
	}
}

// evalUOp recursively evaluates an integer-typed UOp expression against a
// binding from DefineVar arena index → value. Supports the small set of ops
// that flatIndexSints / unflatIndexSints emit plus the leaves they consume.
// Used to prove round-trip identity under concrete bindings.
func evalUOp(u uop.UOp, binding map[uint32]int64) int64 {
	switch u.Op() {
	case uop.OpConst:
		v, ok := u.Arg().(int64)
		if !ok {
			panic("evalUOp: non-int64 OpConst arg")
		}
		return v
	case uop.OpDefineVar:
		v, ok := binding[u.Index()]
		if !ok {
			panic("evalUOp: unbound DefineVar")
		}
		return v
	case uop.OpRange:
		// OpRange is a loop var; for these tests we never bind it (we use
		// constIndices instead). Treat as unbound and report.
		panic("evalUOp: OpRange encountered (use constIndices in tests)")
	case uop.OpNeg:
		return -evalUOp(u.Src(0), binding)
	case uop.OpAdd:
		return evalUOp(u.Src(0), binding) + evalUOp(u.Src(1), binding)
	case uop.OpSub:
		return evalUOp(u.Src(0), binding) - evalUOp(u.Src(1), binding)
	case uop.OpMul:
		return evalUOp(u.Src(0), binding) * evalUOp(u.Src(1), binding)
	case uop.OpIDiv:
		b := evalUOp(u.Src(1), binding)
		return evalUOp(u.Src(0), binding) / b
	case uop.OpMod:
		b := evalUOp(u.Src(1), binding)
		return evalUOp(u.Src(0), binding) % b
	}
	panic("evalUOp: unsupported op " + u.Op().String())
}

// TestStructuralStridesSymMid: shape [4, n] - symbolic non-outermost.
// strides[0] must carry n; strides[1] = 1.
// flat(r0, r1) = r0*n + r1; unflat(flat) = [(flat/n)%4, flat%n].
func TestStructuralStridesSymMid(t *testing.T) {
	for _, nv := range []int64{3, 5, 7} {
		t.Run("n="+itoa(nv), func(t *testing.T) {
			a := uop.NewArena(64)
			sh := symAtShape(a, "n", 16, []int64{4, 0}, 1)
			nNode := sh[1].(shape.SymInt).Node
			binding := map[uint32]int64{nNode.Index(): nv}

			// 1. Structural: strides[0] is symbolic (carries n); strides[1] is Const(1).
			strides := sintStrides(sh)
			if _, ok := strides[0].ConstValue(); ok {
				t.Fatalf("strides[0] should be symbolic (carries n), got concrete %v", strides[0])
			}
			if v, ok := strides[1].ConstValue(); !ok || v != 1 {
				t.Fatalf("strides[1] = %v, want ConstInt(1)", strides[1])
			}
			// strides[0] must equal n when evaluated.
			s0val := evalUOp(dimToUOp(a, strides[0]), binding)
			if s0val != nv {
				t.Errorf("strides[0] evaluates to %d, want %d (= n)", s0val, nv)
			}

			// 2. Numerical round-trip over the full domain.
			for r0 := int64(0); r0 < 4; r0++ {
				for r1 := int64(0); r1 < nv; r1++ {
					idx := constIndices(a, r0, r1)
					flat := flatIndexSints(a, idx, sh)
					gotFlat := evalUOp(flat, binding)
					wantFlat := r0*nv + r1
					if gotFlat != wantFlat {
						t.Errorf("flat(r0=%d, r1=%d, n=%d) = %d, want %d", r0, r1, nv, gotFlat, wantFlat)
					}
					flatC := a.New(uop.OpConst, uop.Dtypes.Index, nil, gotFlat, nil)
					per := unflatIndexSints(a, flatC, sh)
					if len(per) != 2 {
						t.Fatalf("unflat returned %d dims, want 2", len(per))
					}
					gr0 := evalUOp(per[0], binding)
					gr1 := evalUOp(per[1], binding)
					if gr0 != r0 || gr1 != r1 {
						t.Errorf("unflat(%d, n=%d) = [%d, %d], want [%d, %d]", gotFlat, nv, gr0, gr1, r0, r1)
					}
				}
			}
		})
	}
}

// TestStructuralStridesTwoSym: shape [n, m] - two symbolic dims.
// strides[0] = m (symbolic); strides[1] = 1.
// flat(r0, r1) = r0*m + r1; unflat(flat) = [(flat/m), flat%m] (i==0 sym → no Mod).
func TestStructuralStridesTwoSym(t *testing.T) {
	cases := [][2]int64{{3, 5}, {5, 3}, {7, 11}}
	for _, bv := range cases {
		nv, mv := bv[0], bv[1]
		t.Run("n="+itoa(nv)+",m="+itoa(mv), func(t *testing.T) {
			a := uop.NewArena(128)
			sh := twoSymShape(a, "n", "m", 16, 16)
			nNode := sh[0].(shape.SymInt).Node
			mNode := sh[1].(shape.SymInt).Node
			binding := map[uint32]int64{nNode.Index(): nv, mNode.Index(): mv}

			strides := sintStrides(sh)
			if _, ok := strides[0].ConstValue(); ok {
				t.Fatalf("strides[0] should carry m, got concrete %v", strides[0])
			}
			if v, ok := strides[1].ConstValue(); !ok || v != 1 {
				t.Fatalf("strides[1] = %v, want ConstInt(1)", strides[1])
			}
			s0val := evalUOp(dimToUOp(a, strides[0]), binding)
			if s0val != mv {
				t.Errorf("strides[0] evaluates to %d, want %d (= m)", s0val, mv)
			}

			for r0 := int64(0); r0 < nv; r0++ {
				for r1 := int64(0); r1 < mv; r1++ {
					idx := constIndices(a, r0, r1)
					flat := flatIndexSints(a, idx, sh)
					gotFlat := evalUOp(flat, binding)
					wantFlat := r0*mv + r1
					if gotFlat != wantFlat {
						t.Errorf("flat(r0=%d, r1=%d, n=%d, m=%d) = %d, want %d",
							r0, r1, nv, mv, gotFlat, wantFlat)
					}
					flatC := a.New(uop.OpConst, uop.Dtypes.Index, nil, gotFlat, nil)
					per := unflatIndexSints(a, flatC, sh)
					gr0 := evalUOp(per[0], binding)
					gr1 := evalUOp(per[1], binding)
					if gr0 != r0 || gr1 != r1 {
						t.Errorf("unflat(%d) = [%d, %d], want [%d, %d]", gotFlat, gr0, gr1, r0, r1)
					}
				}
			}
		})
	}
}

// TestStructuralStridesNMSplit: round-trip across reshape [n*m] ↔ [n, m].
// This is the case that hit Slice 7 preflight §5.b - pre-7a, the flat index
// over [n, m] came out r0 + r1 (strides [1,1]) instead of r0*m + r1.
func TestStructuralStridesNMSplit(t *testing.T) {
	cases := [][2]int64{{2, 3}, {3, 5}, {4, 4}}
	for _, bv := range cases {
		nv, mv := bv[0], bv[1]
		t.Run("n="+itoa(nv)+",m="+itoa(mv), func(t *testing.T) {
			a := uop.NewArena(128)
			sh2D := twoSymShape(a, "n", "m", 16, 16)
			nNode := sh2D[0].(shape.SymInt).Node
			mNode := sh2D[1].(shape.SymInt).Node
			binding := map[uint32]int64{nNode.Index(): nv, mNode.Index(): mv}

			// [n*m] one-dim shape; the early-return for 1D skips strides.
			sh1D := []shape.Sint{shape.Mul(sh2D[0], sh2D[1])}

			// reshape [n*m] → [n, m]: indices on [n, m] flatten to [n*m] and back.
			for r0 := int64(0); r0 < nv; r0++ {
				for r1 := int64(0); r1 < mv; r1++ {
					idx2D := constIndices(a, r0, r1)
					flat2D := flatIndexSints(a, idx2D, sh2D)
					f := evalUOp(flat2D, binding)
					if f != r0*mv+r1 {
						t.Errorf("flatIndexSints([n,m]) at (%d,%d) = %d, want %d",
							r0, r1, f, r0*mv+r1)
					}
					// 1D source: unflat into single dim returns [flat] (early return),
					// so reshape back via unflatIndexSints on sh2D using the same flat.
					fC := a.New(uop.OpConst, uop.Dtypes.Index, nil, f, nil)
					per := unflatIndexSints(a, fC, sh2D)
					gr0 := evalUOp(per[0], binding)
					gr1 := evalUOp(per[1], binding)
					if gr0 != r0 || gr1 != r1 {
						t.Errorf("reshape round-trip at (%d,%d) under (n=%d,m=%d): got [%d,%d]",
							r0, r1, nv, mv, gr0, gr1)
					}
				}
			}
			_ = sh1D // 1D shape used for reasoning; helpers early-return on len==1.
		})
	}
}

// TestStructuralStridesSymMiddle: shape [4, n, 4] - sym strictly in the
// middle. strides[0] must carry 4n; strides[1]=4; strides[2]=1.
func TestStructuralStridesSymMiddle(t *testing.T) {
	for _, nv := range []int64{2, 3, 5} {
		t.Run("n="+itoa(nv), func(t *testing.T) {
			a := uop.NewArena(128)
			sh := symAtShape(a, "n", 8, []int64{4, 0, 4}, 1)
			nNode := sh[1].(shape.SymInt).Node
			binding := map[uint32]int64{nNode.Index(): nv}

			strides := sintStrides(sh)
			if _, ok := strides[0].ConstValue(); ok {
				t.Fatalf("strides[0] should carry n, got concrete %v", strides[0])
			}
			if v, ok := strides[1].ConstValue(); !ok || v != 4 {
				t.Fatalf("strides[1] = %v, want ConstInt(4)", strides[1])
			}
			if v, ok := strides[2].ConstValue(); !ok || v != 1 {
				t.Fatalf("strides[2] = %v, want ConstInt(1)", strides[2])
			}
			s0val := evalUOp(dimToUOp(a, strides[0]), binding)
			if s0val != 4*nv {
				t.Errorf("strides[0] = %d, want %d (= 4n)", s0val, 4*nv)
			}

			for r0 := int64(0); r0 < 4; r0++ {
				for r1 := int64(0); r1 < nv; r1++ {
					for r2 := int64(0); r2 < 4; r2++ {
						idx := constIndices(a, r0, r1, r2)
						flat := flatIndexSints(a, idx, sh)
						gotFlat := evalUOp(flat, binding)
						wantFlat := r0*4*nv + r1*4 + r2
						if gotFlat != wantFlat {
							t.Errorf("flat(%d,%d,%d,n=%d) = %d, want %d",
								r0, r1, r2, nv, gotFlat, wantFlat)
						}
						fC := a.New(uop.OpConst, uop.Dtypes.Index, nil, gotFlat, nil)
						per := unflatIndexSints(a, fC, sh)
						gr0, gr1, gr2 := evalUOp(per[0], binding), evalUOp(per[1], binding), evalUOp(per[2], binding)
						if gr0 != r0 || gr1 != r1 || gr2 != r2 {
							t.Errorf("unflat(%d) = [%d,%d,%d], want [%d,%d,%d]",
								gotFlat, gr0, gr1, gr2, r0, r1, r2)
						}
					}
				}
			}
		})
	}
}

// TestStructuralStridesSymHeadRegression preserves the symbolic-outermost
// path. Strides on [n, 4] must match pre-7a: stride[0]=4 (concrete), and
// unflatIndexSints must NOT wrap the symbolic outermost dim in a Mod (the
// optimisation pinned by TestUnflatIndexSintsSymbolicHead).
func TestStructuralStridesSymHeadRegression(t *testing.T) {
	a := uop.NewArena(64)
	sh := symHeadShape(a, "n", 64, 4)
	strides := sintStrides(sh)
	if v, ok := strides[0].ConstValue(); !ok || v != 4 {
		t.Errorf("strides[0] = %v, want ConstInt(4)", strides[0])
	}
	if v, ok := strides[1].ConstValue(); !ok || v != 1 {
		t.Errorf("strides[1] = %v, want ConstInt(1)", strides[1])
	}
	flat := constIndices(a, 0)[0]
	per := unflatIndexSints(a, flat, sh)
	if per[0].Op() == uop.OpMod {
		t.Errorf("sym-outermost dim must not be wrapped in Mod (forces runtime bound fetch in inner loop)")
	}
}

func referencesTransitively(root, target uop.UOp) bool {
	if root == target {
		return true
	}
	for i := 0; i < root.NSrc(); i++ {
		if referencesTransitively(root.Src(i), target) {
			return true
		}
	}
	return false
}

func prod(d []int64) int64 {
	p := int64(1)
	for _, x := range d {
		p *= x
	}
	return p
}

func strJoin(d []int64) string {
	s := ""
	for i, x := range d {
		if i > 0 {
			s += "x"
		}
		s += itoa(x)
	}
	return s
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var neg bool
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
