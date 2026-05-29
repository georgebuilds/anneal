package uop_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// hash_arg_test.go — adversarial coverage for hashArg/equalArg, the structural-key
// primitives that back the schedule cache and the BEAM disk cache.
//
// Methodology: hashArg and equalArg are private, but the interning path exercises
// both together. If two args are equalArg-equal, intern collapses them to one node;
// if either equalArg returns false OR hashes differ, two News produce two nodes.
// Therefore:
//
//   - To verify EQUAL args produce identical hashes AND compare equal: build twice,
//     expect a.Len() == 1 (intern hit).
//   - To verify DIFFERENT args produce distinct nodes (regardless of hash collision):
//     build twice, expect a.Len() == 2 AND u1 != u2.
//
// Tests use non-bypass ops so that interning is actually exercised. OpRange, OpBuffer,
// OpUnique, OpLUnique, OpDefineLocal are in bypassInternSet and would skip intern.
// We use OpConst (not in bypass) as the carrier and attach the arg types we want to
// stress; the arg machinery is op-agnostic.

// ── helpers ──────────────────────────────────────────────────────────────────

// internsEqual builds two nodes with the given args on a fresh arena and reports
// whether they interned to the same UOp (i.e. equalArg returned true AND hashes
// matched). dtype must be the same for both — we vary only the arg.
func internsEqual(t *testing.T, arg1, arg2 any) (same bool, a *uop.Arena) {
	t.Helper()
	a = uop.NewArena(8)
	u1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, arg1, nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Index, nil, arg2, nil)
	return u1 == u2, a
}

// ── BufferizeArg — load-bearing for BEAM cache (Removable flag) ─────────────

// TestEqualArgBufferizeRemovableDistinct is the headline case for B5: a hard
// (Removable=false) and a soft (Removable=true) Bufferize must NEVER alias.
// If they did, the schedule cache would silently swap kernel-output materialization
// rules across realize calls and the BEAM cache would key both under one hash.
func TestEqualArgBufferizeRemovableDistinct(t *testing.T) {
	same, a := internsEqual(t,
		uop.BufferizeArg{Removable: false},
		uop.BufferizeArg{Removable: true},
	)
	if same {
		t.Fatal("BufferizeArg{Removable:false} and {Removable:true} interned to the same node — "+
			"this would collapse hard and soft bufferize boundaries in the schedule/BEAM cache")
	}
	if a.Len() != 2 {
		t.Errorf("arena Len = %d, want 2", a.Len())
	}
}

func TestEqualArgBufferizeSameInterns(t *testing.T) {
	for _, removable := range []bool{false, true} {
		same, a := internsEqual(t,
			uop.BufferizeArg{Removable: removable},
			uop.BufferizeArg{Removable: removable},
		)
		if !same {
			t.Errorf("BufferizeArg{Removable:%v} duplicated: arena Len=%d", removable, a.Len())
		}
	}
}

// ── RangeArg — every field participates in the key ───────────────────────────

// TestEqualArgRangeAllFields walks every RangeArg field; for each, two values that
// differ in only that field must produce distinct interned nodes. RangeArg backs
// loop-variable identity in the codegen; if any field were dropped from the key,
// the WGSL hash on the BEAM disk cache could collide kernels with different loop
// semantics.
func TestEqualArgRangeAllFields(t *testing.T) {
	base := uop.RangeArg{ID: 1, Size: 8, Type: uop.AxisLoop, Symbolic: false, SymParamIdx: 0, VarName: "n"}

	cases := []struct {
		name    string
		mutated uop.RangeArg
	}{
		{"ID", func() uop.RangeArg { r := base; r.ID = 2; return r }()},
		{"Size", func() uop.RangeArg { r := base; r.Size = 9; return r }()},
		{"Type", func() uop.RangeArg { r := base; r.Type = uop.AxisReduce; return r }()},
		{"Symbolic", func() uop.RangeArg { r := base; r.Symbolic = true; return r }()},
		{"SymParamIdx", func() uop.RangeArg { r := base; r.SymParamIdx = 1; return r }()},
		{"VarName", func() uop.RangeArg { r := base; r.VarName = "m"; return r }()},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			same, a := internsEqual(t, base, tc.mutated)
			if same {
				t.Errorf("RangeArg differing only in field %s interned to the same node — "+
					"%s is dropped from the intern key", tc.name, tc.name)
			}
			if a.Len() != 2 {
				t.Errorf("arena Len = %d, want 2", a.Len())
			}
		})
	}
}

func TestEqualArgRangeIdenticalInterns(t *testing.T) {
	r := uop.RangeArg{ID: 7, Size: 16, Type: uop.AxisUpcast, Symbolic: true, SymParamIdx: 3, VarName: "batch"}
	same, _ := internsEqual(t, r, r)
	if !same {
		t.Error("identical RangeArg failed to intern")
	}
}

// ── ReduceArg — Op + sorted Axes ────────────────────────────────────────────

func TestEqualArgReduceAllFields(t *testing.T) {
	base := uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0, 1}}

	t.Run("different op", func(t *testing.T) {
		same, _ := internsEqual(t, base, uop.ReduceArg{Op: uop.OpMax, Axes: []int{0, 1}})
		if same {
			t.Error("ReduceArg with different Op interned together")
		}
	})
	t.Run("different axes content", func(t *testing.T) {
		same, _ := internsEqual(t, base, uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0, 2}})
		if same {
			t.Error("ReduceArg with different Axes interned together")
		}
	})
	t.Run("different axes length", func(t *testing.T) {
		same, _ := internsEqual(t, base, uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0}})
		if same {
			t.Error("ReduceArg with different Axes length interned together")
		}
	})
	t.Run("axes order matters", func(t *testing.T) {
		// Axes are documented as sorted; the equalArg code compares index-by-index.
		// If the scheduler ever produces unsorted axes, two semantically-equivalent
		// reduces would not alias — this test pins the index-by-index contract.
		same, _ := internsEqual(t, base, uop.ReduceArg{Op: uop.OpAdd, Axes: []int{1, 0}})
		if same {
			t.Error("ReduceArg with reordered Axes interned together — equalArg does NOT canonicalise order")
		}
	})
	t.Run("nil vs empty axes", func(t *testing.T) {
		same, _ := internsEqual(t,
			uop.ReduceArg{Op: uop.OpAdd, Axes: nil},
			uop.ReduceArg{Op: uop.OpAdd, Axes: []int{}},
		)
		if !same {
			t.Error("ReduceArg with nil vs empty axes did NOT intern — both length-0 axes should alias")
		}
	})
}

// ── ShapeSintArg — concrete + symbolic dims ─────────────────────────────────

func TestEqualArgShapeSintAllFields(t *testing.T) {
	base := uop.ShapeSintArg{
		{V: 0, Sym: true, VarIdx: 5},
		{V: 32, Sym: false},
	}

	t.Run("identical interns", func(t *testing.T) {
		same, _ := internsEqual(t, base, append(uop.ShapeSintArg(nil), base...))
		if !same {
			t.Error("identical ShapeSintArg failed to intern")
		}
	})
	t.Run("different length", func(t *testing.T) {
		short := uop.ShapeSintArg{base[0]}
		same, _ := internsEqual(t, base, short)
		if same {
			t.Error("ShapeSintArg of different length interned together")
		}
	})
	t.Run("Sym flag flipped", func(t *testing.T) {
		mut := append(uop.ShapeSintArg(nil), base...)
		mut[0] = uop.ShapeDim{V: 0, Sym: false, VarIdx: 5} // same VarIdx, but concrete
		same, _ := internsEqual(t, base, mut)
		if same {
			t.Error("ShapeSintArg with Sym flag flipped (Sym vs concrete) interned together")
		}
	})
	t.Run("VarIdx differs (symbolic dim)", func(t *testing.T) {
		mut := append(uop.ShapeSintArg(nil), base...)
		mut[0] = uop.ShapeDim{V: 0, Sym: true, VarIdx: 6}
		same, _ := internsEqual(t, base, mut)
		if same {
			t.Error("ShapeSintArg with different VarIdx on symbolic dim interned together — "+
				"two different DefineVars would alias in the reshape arg")
		}
	})
	t.Run("V differs (concrete dim)", func(t *testing.T) {
		mut := append(uop.ShapeSintArg(nil), base...)
		mut[1] = uop.ShapeDim{V: 64, Sym: false}
		same, _ := internsEqual(t, base, mut)
		if same {
			t.Error("ShapeSintArg with different V on concrete dim interned together")
		}
	})
	t.Run("V distinguishes on symbolic dim (hash/equal asymmetry)", func(t *testing.T) {
		// FINDING: hashArg and equalArg disagree on V when Sym=true.
		//   hashArg (uop.go:443-450): mixes only VarIdx when Sym=true; V is skipped.
		//   equalArg (uop.go:546-550): compares the whole ShapeDim struct via !=,
		//     so V is part of the equality check regardless of Sym.
		//
		// Net effect: two ShapeSintArgs with same VarIdx but different V on a Sym
		// dim hash to the SAME bucket but compare UNEQUAL, producing two arena
		// nodes in one bucket. This is SAFE for intern correctness (no false alias)
		// but inefficient on hash collisions.
		//
		// In production, toShapeSintArg (tensor/movement.go:208) always sets V=0
		// when Sym=true, so the asymmetry never bites. This test pins the current
		// asymmetric behaviour — if either side is changed (hashArg starts mixing
		// V on Sym=true OR equalArg starts ignoring it), this test will flip.
		mut := append(uop.ShapeSintArg(nil), base...)
		mut[0] = uop.ShapeDim{V: 999, Sym: true, VarIdx: 5}
		same, a := internsEqual(t, base, mut)
		if same {
			t.Error("ShapeSintArg with different V on Sym=true dim aliased — "+
				"equalArg used to distinguish them; if you intended to ignore V on Sym, "+
				"update both hashArg and equalArg together")
		}
		if a.Len() != 2 {
			t.Errorf("arena Len = %d, want 2", a.Len())
		}
	})
}

// ── primitive types ─────────────────────────────────────────────────────────

// TestEqualArgInt64Boundaries: signed boundary values must produce distinct nodes.
// Critical because uint64(int64(-1)) == math.MaxUint64; if hashArg ever used
// math.Float64bits on an int64 or wrong cast, sign would collapse.
func TestEqualArgInt64Boundaries(t *testing.T) {
	cases := []int64{0, 1, -1, math.MaxInt64, math.MinInt64, 1 << 32, -(1 << 32)}
	seen := make(map[uint32]int64)
	a := uop.NewArena(len(cases))
	for _, v := range cases {
		u := a.New(uop.OpConst, uop.Dtypes.Index, nil, v, nil)
		if prev, ok := seen[u.Index()]; ok {
			t.Errorf("int64 collision: %d and %d both intern at idx %d", prev, v, u.Index())
		}
		seen[u.Index()] = v
	}
}

// TestEqualArgFloat64NaNBits: two NaNs with identical bit patterns are equal
// (documented); NaNs with different bit patterns must be distinct.
func TestEqualArgFloat64NaNBits(t *testing.T) {
	nan1 := math.Float64frombits(0x7ff8000000000001)
	nan2 := math.Float64frombits(0x7ff8000000000001)
	nan3 := math.Float64frombits(0x7ff8000000000002)

	t.Run("same NaN bits intern", func(t *testing.T) {
		same, _ := internsEqual(t, nan1, nan2)
		if !same {
			t.Error("identical NaN bit pattern failed to intern (Float64bits equality contract broken)")
		}
	})
	t.Run("different NaN bits distinct", func(t *testing.T) {
		same, _ := internsEqual(t, nan1, nan3)
		if same {
			t.Error("NaNs with different bit patterns aliased — Float64bits not consulted")
		}
	})
	t.Run("+0.0 vs -0.0 distinct", func(t *testing.T) {
		// +0.0 == -0.0 as floats but their bit patterns differ; the intern key uses bits.
		same, _ := internsEqual(t, float64(0), math.Copysign(0, -1))
		if same {
			t.Error("+0.0 and -0.0 aliased — intern uses Float64bits, these have different bits")
		}
	})
}

// ── slice arg types ─────────────────────────────────────────────────────────

func TestEqualArgInt64Slice(t *testing.T) {
	t.Run("identical interns", func(t *testing.T) {
		same, _ := internsEqual(t, []int64{1, 2, 3}, []int64{1, 2, 3})
		if !same {
			t.Error("identical []int64 failed to intern")
		}
	})
	t.Run("element differs", func(t *testing.T) {
		same, _ := internsEqual(t, []int64{1, 2, 3}, []int64{1, 2, 4})
		if same {
			t.Error("[]int64 differing in one element aliased")
		}
	})
	t.Run("length differs (prefix)", func(t *testing.T) {
		// Length is mixed into the hash, so a prefix should NOT alias with the full.
		same, _ := internsEqual(t, []int64{1, 2, 3}, []int64{1, 2})
		if same {
			t.Error("[]int64 prefix aliased with full slice — length not mixed into key")
		}
	})
	t.Run("nil vs empty", func(t *testing.T) {
		// Go switch case []int64 catches both nil and empty; both have length 0,
		// hashed identically, equal under index-by-index loop (no iterations).
		same, _ := internsEqual(t, []int64(nil), []int64{})
		if !same {
			t.Error("[]int64(nil) and []int64{} did NOT intern — both are length-0 of same type")
		}
	})
}

func TestEqualArgInt64PairSlice(t *testing.T) {
	t.Run("identical interns", func(t *testing.T) {
		same, _ := internsEqual(t, [][2]int64{{0, 1}, {2, 3}}, [][2]int64{{0, 1}, {2, 3}})
		if !same {
			t.Error("identical [][2]int64 failed to intern")
		}
	})
	t.Run("inner pair differs", func(t *testing.T) {
		same, _ := internsEqual(t, [][2]int64{{0, 1}}, [][2]int64{{0, 2}})
		if same {
			t.Error("[][2]int64 differing in second of pair aliased")
		}
	})
	t.Run("first of pair differs", func(t *testing.T) {
		same, _ := internsEqual(t, [][2]int64{{0, 1}}, [][2]int64{{1, 1}})
		if same {
			t.Error("[][2]int64 differing in first of pair aliased")
		}
	})
}

// ── KernelInfo ───────────────────────────────────────────────────────────────

func TestEqualArgKernelInfo(t *testing.T) {
	t.Run("identical interns", func(t *testing.T) {
		same, _ := internsEqual(t, uop.KernelInfo{NumParams: 3}, uop.KernelInfo{NumParams: 3})
		if !same {
			t.Error("identical KernelInfo failed to intern")
		}
	})
	t.Run("NumParams differs", func(t *testing.T) {
		same, _ := internsEqual(t, uop.KernelInfo{NumParams: 3}, uop.KernelInfo{NumParams: 4})
		if same {
			t.Error("KernelInfo with different NumParams aliased")
		}
	})
}

// ── Op-as-arg ────────────────────────────────────────────────────────────────

// TestEqualArgOpAsArg: kernel-level REDUCE carries its accumulation op as the
// arg. Two REDUCEs with different acc ops must produce different intern keys.
func TestEqualArgOpAsArg(t *testing.T) {
	t.Run("identical Op interns", func(t *testing.T) {
		same, _ := internsEqual(t, uop.OpAdd, uop.OpAdd)
		if !same {
			t.Error("identical Op arg failed to intern")
		}
	})
	t.Run("different Op distinct", func(t *testing.T) {
		same, _ := internsEqual(t, uop.OpAdd, uop.OpMax)
		if same {
			t.Error("Op arg with different value aliased")
		}
	})
}

// ── VarArg ───────────────────────────────────────────────────────────────────

func TestEqualArgVarArg(t *testing.T) {
	t.Run("identical names intern", func(t *testing.T) {
		same, _ := internsEqual(t, uop.VarArg{Name: "batch"}, uop.VarArg{Name: "batch"})
		if !same {
			t.Error("identical VarArg failed to intern")
		}
	})
	t.Run("different names distinct", func(t *testing.T) {
		same, _ := internsEqual(t, uop.VarArg{Name: "batch"}, uop.VarArg{Name: "seq"})
		if same {
			t.Error("VarArg with different names aliased")
		}
	})
	t.Run("empty name", func(t *testing.T) {
		same, _ := internsEqual(t, uop.VarArg{Name: ""}, uop.VarArg{Name: ""})
		if !same {
			t.Error("identical empty-name VarArg failed to intern")
		}
	})
	t.Run("prefix collision", func(t *testing.T) {
		// "n" should not alias with "nn" — strings are length-prefixed.
		same, _ := internsEqual(t, uop.VarArg{Name: "n"}, uop.VarArg{Name: "nn"})
		if same {
			t.Error("VarArg name prefix aliased — string length leaked")
		}
	})
}

// ── string arg ──────────────────────────────────────────────────────────────

func TestEqualArgString(t *testing.T) {
	t.Run("identical interns", func(t *testing.T) {
		same, _ := internsEqual(t, "hello", "hello")
		if !same {
			t.Error("identical string failed to intern")
		}
	})
	t.Run("empty interns", func(t *testing.T) {
		same, _ := internsEqual(t, "", "")
		if !same {
			t.Error("empty strings failed to intern")
		}
	})
	t.Run("prefix collision", func(t *testing.T) {
		same, _ := internsEqual(t, "ab", "abc")
		if same {
			t.Error("string prefix aliased")
		}
	})
}

// ── cross-type discriminator: every type tagged uniquely ────────────────────

// TestEqualArgCrossTypeDiscriminators is the deepest test: every supported arg
// type carries a distinct discriminator. Two args of DIFFERENT types whose payload
// could otherwise mix to the same hash bytes must NOT alias.
//
// We seed each arg type with a "zero-ish" value (the most ambiguous content for
// that type) and verify the type tag prevents collision. If any pair below ever
// aliases, hashArg has dropped or duplicated a discriminator and the schedule
// cache will silently confuse kernels with semantically distinct args.
func TestEqualArgCrossTypeDiscriminators(t *testing.T) {
	zeroArgs := []struct {
		name string
		arg  any
	}{
		{"nil", nil},
		{"int64(0)", int64(0)},
		{"float64(0)", float64(0)},
		{"bool(false)", false},
		{"string('')", ""},
		{"[]int64(nil)", []int64(nil)},
		{"[][2]int64(nil)", [][2]int64(nil)},
		{"ReduceArg{}", uop.ReduceArg{}},
		{"RangeArg{}", uop.RangeArg{}},
		{"BufferizeArg{}", uop.BufferizeArg{}},
		{"Op(0)", uop.Op(0)},
		{"KernelInfo{}", uop.KernelInfo{}},
		{"VarArg{}", uop.VarArg{}},
		{"ShapeSintArg(nil)", uop.ShapeSintArg(nil)},
	}

	a := uop.NewArena(len(zeroArgs))
	indexOf := make(map[uint32]string)
	for _, tc := range zeroArgs {
		u := a.New(uop.OpConst, uop.Dtypes.Index, nil, tc.arg, nil)
		if prev, ok := indexOf[u.Index()]; ok {
			t.Errorf("cross-type alias: %s and %s both intern at arena idx %d — "+
				"discriminator collision in hashArg/equalArg", prev, tc.name, u.Index())
		}
		indexOf[u.Index()] = tc.name
	}
	if a.Len() != len(zeroArgs) {
		t.Errorf("arena Len = %d, want %d (one node per arg type)", a.Len(), len(zeroArgs))
	}
}

// ── tag participates the same way as arg ─────────────────────────────────────

// TestEqualArgTagAndArgUseSameMachinery verifies that tag goes through hashArg
// and equalArg the same as arg. Two nodes with arg=nil tag=int64(1) and arg=nil
// tag=int64(2) must be distinct; nothing in equalArg branches on field role.
func TestEqualArgTagAndArgUseSameMachinery(t *testing.T) {
	a := uop.NewArena(8)
	u1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, nil, int64(1))
	u2 := a.New(uop.OpConst, uop.Dtypes.Index, nil, nil, int64(2))
	u3 := a.New(uop.OpConst, uop.Dtypes.Index, nil, nil, int64(1))

	if u1 == u2 {
		t.Error("different tag values aliased")
	}
	if u1 != u3 {
		t.Error("identical tag values failed to intern")
	}
}

// ── nil-tag-vs-zero-int tag matches the arg pattern ─────────────────────────

// TestEqualArgNilTagVsTypedTag pins that a tag of nil and a tag of a typed-zero
// follow the same discriminator-table rules as args (regression for any future
// refactor that special-cases tag handling).
func TestEqualArgNilTagVsTypedTag(t *testing.T) {
	a := uop.NewArena(8)
	u1 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	u2 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), int64(0))
	if u1 == u2 {
		t.Error("nil tag and int64(0) tag aliased — tag-side nil discriminator broken")
	}
}

// ── equalArg panic path on unsupported type at hash-collision ───────────────

// TestEqualArgUnsupportedTypePanics exercises the equalArg default branch.
// hashArg's panic is covered by TestUnsupportedArgPanics in uop_test.go; here we
// ensure that even if hashArg were extended (or a hash collision arose), the
// equalArg default panic fires. We can only trigger this through a public API
// that hits a hash collision on an unsupported type — not reachable directly.
// Instead, we document the contract: any new arg type MUST be added to BOTH
// hashArg and equalArg, and the unsupported-type test in uop_test.go pins the
// hashArg side. This test pins that the same enforcement exists at New time.
func TestEqualArgUnsupportedTypePanics(t *testing.T) {
	a := uop.NewArena(4)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unsupported arg type via tag, got none")
		}
	}()
	// Same as TestUnsupportedArgPanics but via tag — both fields go through hashArg.
	a.New(uop.OpConst, uop.Dtypes.Void, nil, nil, []int{1, 2, 3})
}
