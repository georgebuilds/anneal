package codegen

// PROBE for Slice 7d preflight (transient — not part of the production suite).
// Goal: confirm whether applyLocal can in principle be relaxed for multi-dim
// symbolic kernels by inspecting what the rendered WGSL becomes if we bypass
// the sym-bail guard locally.
//
// We can't actually run the GPU here (no device in the codegen package), but
// we can render the WGSL and trace the level-0 stride decomposition.

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// buildSymC1Kernel returns a probe-side [n, m] elementwise sym kernel as a
// SINK-rooted ExecItem. Mirrors backend/webgpu/slice7b_ccase_test.go's
// buildC1Kernel without depending on the webgpu package's helpers.
func buildSymC1Kernel(a *uop.Arena, varN, varM string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	defM := a.DefineVar(varM, 1, 100)
	rN := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	rM := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defM}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	mul := a.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{rN, defM}, nil, nil)
	flatIdx := a.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, rM}, nil, nil)

	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, flatIdx}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, flatIdx}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, rN, rM}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	return schedule.ExecItem{
		Ast:     sink,
		Bufs:    []schedule.Buffer{{UOpIdx: paramOut.Index(), DType: uop.Dtypes.Float32, Slot: -1}, {UOpIdx: paramA.Index(), DType: uop.Dtypes.Float32, Slot: -1}, {UOpIdx: paramB.Index(), DType: uop.Dtypes.Float32, Slot: -1}},
		SymVars: []string{varN, varM},
	}
}

// applyLocalUnchecked is a probe version of applyLocal that skips the
// RangeIsSymbolic bail. Used only by this preflight test to inspect the
// hypothetical relaxed output. Do NOT use in production code.
func applyLocalUnchecked(sink uop.UOp, axisIdx int, localSize int) uop.UOp {
	if sink.Op() != uop.OpSink {
		return sink
	}
	arena := sink.Arena()
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return sink
	}

	var targetRange uop.UOp
	var targetIdx int
	currIdx := 0
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			ra := r.Arg().(uop.RangeArg)
			if ra.Type == uop.AxisLoop {
				if currIdx == axisIdx {
					targetRange = r
					targetIdx = i
					break
				}
				currIdx++
			}
		}
	}
	if !targetRange.Valid() {
		return sink
	}

	ra := targetRange.Arg().(uop.RangeArg)
	L := int64(localSize)

	maxID := -1
	for i := 0; i < arena.Len(); i++ {
		u := arena.At(uint32(i))
		if u.Op() == uop.OpRange {
			rid := u.Arg().(uop.RangeArg).ID
			if rid > maxID {
				maxID = rid
			}
		}
	}

	var wBoundUOp uop.UOp
	if uop.RangeIsSymbolic(targetRange) {
		boundUOp := targetRange.Src(0)
		lMinusOne := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L-1, nil)
		sumNode := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{boundUOp, lMinusOne}, nil, nil)
		lConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L, nil)
		wBoundUOp = arena.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{sumNode, lConst}, nil, nil)
	} else {
		S := uop.RangeSize(targetRange)
		W := (S + L - 1) / L
		wBoundUOp = arena.New(uop.OpConst, uop.Dtypes.Index, nil, W, nil)
	}

	rwg := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{wBoundUOp}, uop.RangeArg{ID: ra.ID, Type: uop.AxisWorkgroup}, nil)
	lBoundConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L, nil)
	rloc := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{lBoundConst}, uop.RangeArg{ID: maxID + 1, Type: uop.AxisLocal}, nil)

	lConst2 := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L, nil)
	mul := arena.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{rwg, lConst2}, nil, nil)
	add := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, rloc}, nil, nil)

	store := end.Src(0)
	cache := make(map[uint32]uop.UOp)
	newBody := rewriteBody(store.Src(1), targetRange, add, cache)
	newStore := arena.New(uop.OpStore, store.DType(), []uop.UOp{store.Src(0), newBody}, store.Arg(), store.Tag())

	newEndSrcs := make([]uop.UOp, end.NSrc()+1)
	newEndSrcs[0] = newStore
	dest := 1
	for i := 1; i < end.NSrc(); i++ {
		if i == targetIdx {
			newEndSrcs[dest] = rwg
			newEndSrcs[dest+1] = rloc
			dest += 2
		} else {
			newEndSrcs[dest] = end.Src(i)
			dest++
		}
	}
	newEnd := arena.New(uop.OpEnd, end.DType(), newEndSrcs, end.Arg(), end.Tag())
	return arena.New(uop.OpSink, sink.DType(), []uop.UOp{newEnd}, sink.Arg(), sink.Tag())
}

// build1DSymKernel returns a 1D [n] elementwise sym kernel: c[i] = a[i] + b[i].
func build1DSymKernel(a *uop.Arena, varN string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	rN := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, rN}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, rN}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, rN}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	return schedule.ExecItem{
		Ast:     sink,
		Bufs:    []schedule.Buffer{{UOpIdx: paramOut.Index(), DType: uop.Dtypes.Float32, Slot: -1}, {UOpIdx: paramA.Index(), DType: uop.Dtypes.Float32, Slot: -1}, {UOpIdx: paramB.Index(), DType: uop.Dtypes.Float32, Slot: -1}},
		SymVars: []string{varN},
	}
}

// TestSlice7dProbe_LocalOn1DSym renders WGSL for a [n] sym kernel before/after
// applying (hypothetical relaxed) OptLocal. The 1D case should compose cleanly:
// rwg=wid.x, rloc=lid.x, body uses (rwg*L + rloc) = gid_x.
func TestSlice7dProbe_LocalOn1DSym(t *testing.T) {
	a := uop.NewArena(256)
	item := build1DSymKernel(a, "cn")

	preWGSL := RenderWGSL(item).WGSL
	t.Logf("PRE-LOCAL (1D sym) WGSL:\n%s", preWGSL)

	relaxedItem := item
	relaxedItem.Ast = applyLocalUnchecked(item.Ast, 0, 8)
	postWGSL := RenderWGSL(relaxedItem).WGSL
	t.Logf("POST-LOCAL(axis=0, L=8) (1D sym) WGSL:\n%s", postWGSL)
}

// TestSlice7dProbe_LocalOnMultiDimSym renders WGSL for a [n, m] sym kernel
// before and after applying a (hypothetical relaxed) OptLocal, and reports
// the indexing expressions for r0 (n) and r1 (m). If the post-Local r0/r1
// expressions diverge from the pre-Local ones in a way that changes the
// flat-index decomposition, the relaxation is unsafe under the current
// 1D-flattened sym dispatch model.
func TestSlice7dProbe_LocalOnMultiDimSym(t *testing.T) {
	a := uop.NewArena(256)
	item := buildSymC1Kernel(a, "cn", "cm")

	preWGSL := RenderWGSL(item).WGSL
	t.Logf("PRE-LOCAL WGSL:\n%s", preWGSL)

	// Apply LOCAL (unchecked) on axisIdx=1 (rM, the inner sym dim), L=8.
	relaxedItem := item
	relaxedItem.Ast = applyLocalUnchecked(item.Ast, 1, 8)
	postWGSL := RenderWGSL(relaxedItem).WGSL
	t.Logf("POST-LOCAL(axis=1, L=8) WGSL:\n%s", postWGSL)

	// Report whether r0 and r1 lines changed.
	preLines := strings.Split(preWGSL, "\n")
	postLines := strings.Split(postWGSL, "\n")
	t.Logf("--- r0/r1 lines pre ---")
	for _, ln := range preLines {
		if strings.Contains(ln, "let r0:") || strings.Contains(ln, "let r1:") || strings.Contains(ln, "let r2:") {
			t.Logf("  %s", strings.TrimSpace(ln))
		}
	}
	t.Logf("--- r0/r1 lines post ---")
	for _, ln := range postLines {
		if strings.Contains(ln, "let r0:") || strings.Contains(ln, "let r1:") || strings.Contains(ln, "let r2:") {
			t.Logf("  %s", strings.TrimSpace(ln))
		}
	}
}
