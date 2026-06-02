// opt.go — Opt seam: the four composable kernel transforms and their apply entry point.
//
// An Opt{Kind, Axis, Arg} describes one kernel transformation. ApplyOpt rewrites a
// kernel SINK-rooted AST before Lower/RenderWGSL. This is the only extension point for
// kernel-level performance work; the lowerer and renderer are never modified for tuning.
//
// Four composable transforms:
//   - OptLocal(axis, localSize): multi-dim @workgroup_size + 2D/3D dispatch. Also fixes
//     the WebGPU 65535 1D-dispatch limit by spreading workgroups across dimensions.
//   - OptTile(axis, TS): shared-memory tiling on a reduce axis — var<workgroup> tile
//     buffers, workgroupBarrier(), coalesced + branchless tile loads.
//   - OptUpcast(axis, factor): per-thread register-blocked micro-tile; rejects reduce
//     and Symbolic axes.
//   - OptVectorize(axis, W): vec4 inner-K, WGSL source-level vec4 packaging; W=4 only.
//     On Apple/Metal this lowers to 4 scalar loads, not a hardware SIMD instruction.
//
// Opts compose: call ApplyOpts with a sequence. BEAM search (beam.go) finds the best
// sequence automatically.

package codegen

import (
	"fmt"
	"strings"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// KernelHasTiledReduce reports whether sink contains at least one OpReduce
// tagged with "tile:" (set by OptTile). This is the prerequisite for the
// emitTiledReduce lowerer path that actually consumes AxisUpcast positions
// per (mr, nr) iteration. Without it, AxisUpcast lowering assigns the
// placeholder expression "0" and each thread silently writes only lane 0 of
// its factor-wide stripe.
//
// Used by applyUpcast (fail-loud assertion at opt-application time), by
// ActionSpace (pre-filter so BEAM's probe never triggers the assertion),
// and by spray-mode callers that need to apply OptUpcast across a kernel
// list containing non-matmul items.
func KernelHasTiledReduce(sink uop.UOp) bool {
	for _, u := range uop.TopoSort(sink) {
		if u.Op() != uop.OpReduce {
			continue
		}
		tag, ok := u.Tag().(string)
		if ok && strings.HasPrefix(tag, "tile:") {
			return true
		}
	}
	return false
}

// OptKind identifies the type of a kernel optimization.
type OptKind int

const (
	// OptIdentity returns the kernel AST unchanged.
	OptIdentity OptKind = iota
	// OptLocal splits an axis into workgroup and local dimensions.
	OptLocal
	// OptTile blocks a reduction axis and uses shared memory tiling.
	OptTile
	// OptUpcast splits a parallel (output) axis into outer + inner-unrolled
	// micro-tile stripes. Each thread covers `factor` sequential outputs in
	// that dim, enabling register blocking when composed with OptTile.
	OptUpcast
	// OptVectorize splits a parallel axis into outer + AxisVectorize(width) inner.
	// The inner dimension is emitted as SIMD vec4 operations in the lowerer.
	// Rejects AxisReduce and Symbolic axes. Only width=4 has a working lowerer.
	// Compose after OptTile+OptUpcast for the B3.7 register-blocking+vec4 pipeline.
	OptVectorize
)

// Opt captures one kernel optimization: an op kind, an axis index, and an int arg.
type Opt struct {
	Kind OptKind
	Axis int
	Arg  int
}

// ApplyOpt applies an optimization to a kernel SINK-rooted AST.
func ApplyOpt(sink uop.UOp, opt Opt) uop.UOp {
	switch opt.Kind {
	case OptIdentity:
		return sink
	case OptLocal:
		return applyLocal(sink, opt.Axis, opt.Arg)
	case OptTile:
		return applyTile(sink, opt.Axis, opt.Arg)
	case OptUpcast:
		return applyUpcast(sink, opt.Axis, opt.Arg)
	case OptVectorize:
		return applyVectorize(sink, opt.Axis, opt.Arg)
	default:
		return sink
	}
}

func applyLocal(sink uop.UOp, axisIdx int, localSize int) uop.UOp {
	if sink.Op() != uop.OpSink {
		return sink
	}
	arena := sink.Arena()
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return sink
	}

	// Find the target AxisLoop range
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

	// Find max existing Range ID to pick unique ones for the new ranges
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

	// Create new ranges. We reuse the original ID for the workgroup part.
	// Static path: workgroup bound is the concrete int64 ceil(S/L).
	// Symbolic path: workgroup bound is the UOp expression (boundUOp + (L-1)) IDiv L,
	// which boundExprFromUOp turns into a dispatch-time-evaluable BoundExpr.
	var wBoundUOp uop.UOp
	if uop.RangeIsSymbolic(targetRange) {
		boundUOp := targetRange.Src(0)
		lMinusOne := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L-1, nil)
		sumNode := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{boundUOp, lMinusOne}, nil, nil)
		lConstForDiv := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L, nil)
		wBoundUOp = arena.New(uop.OpIDiv, uop.Dtypes.Index, []uop.UOp{sumNode, lConstForDiv}, nil, nil)
	} else {
		S := uop.RangeSize(targetRange)
		W := (S + L - 1) / L
		wBoundUOp = arena.New(uop.OpConst, uop.Dtypes.Index, nil, W, nil)
	}
	rwg := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{wBoundUOp}, uop.RangeArg{
		ID:   ra.ID,
		Type: uop.AxisWorkgroup,
	}, nil)

	lBoundConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L, nil)
	rloc := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{lBoundConst}, uop.RangeArg{
		ID:   maxID + 1,
		Type: uop.AxisLocal,
	}, nil)

	// Build the replacement expression: (rwg * L) + rloc
	lConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, L, nil)
	mul := arena.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{rwg, lConst}, nil, nil)
	add := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, rloc}, nil, nil)

	// Rewrite the body (store.src[1])
	store := end.Src(0)
	cache := make(map[uint32]uop.UOp)
	newBody := rewriteBody(store.Src(1), targetRange, add, cache)

	// Rebuild STORE
	newStore := arena.New(uop.OpStore, store.DType(), []uop.UOp{store.Src(0), newBody}, store.Arg(), store.Tag())

	// Rebuild END with new range list
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

	// Rebuild SINK
	return arena.New(uop.OpSink, sink.DType(), []uop.UOp{newEnd}, sink.Arg(), sink.Tag())
}

func rewriteBody(u, old, new uop.UOp, cache map[uint32]uop.UOp) uop.UOp {
	if u == old {
		return new
	}
	if u.NSrc() == 0 {
		return u
	}
	if r, ok := cache[u.Index()]; ok {
		return r
	}

	changed := false
	srcs := make([]uop.UOp, u.NSrc())
	for i := 0; i < u.NSrc(); i++ {
		srcs[i] = rewriteBody(u.Src(i), old, new, cache)
		if srcs[i] != u.Src(i) {
			changed = true
		}
	}

	if !changed {
		cache[u.Index()] = u
		return u
	}

	res := u.Arena().New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag())
	cache[u.Index()] = res
	return res
}

// containsGatherIdx reports whether the UOp subgraph rooted at u contains an
// OpGatherIdx node. Used by applyTile to skip tiling on reduce bodies whose
// index operands transitively read through an indirect index; the tiled
// lowerer assumes affine OpIndex chains and cannot hoist a gather load into
// a coalesced workgroup-tile copy.
//
// Uses a per-call seen map (kept local; applyTile is called per (axis, tile)
// candidate, not in a hot loop). For each (axis, tile) we walk both Mul
// operand subgraphs, which are small relative to the full kernel.
func containsGatherIdx(u uop.UOp) bool {
	seen := make(map[uint32]bool)
	var walk func(n uop.UOp) bool
	walk = func(n uop.UOp) bool {
		if !n.Valid() {
			return false
		}
		idx := n.Index()
		if seen[idx] {
			return false
		}
		seen[idx] = true
		if n.Op() == uop.OpGatherIdx {
			return true
		}
		for i := 0; i < n.NSrc(); i++ {
			if walk(n.Src(i)) {
				return true
			}
		}
		return false
	}
	return walk(u)
}

func applyTile(sink uop.UOp, axisIdx int, tileSize int) uop.UOp {
	if sink.Op() != uop.OpSink {
		return sink
	}
	arena := sink.Arena()
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return sink
	}
	store := end.Src(0)
	if store.Op() != uop.OpStore {
		return sink
	}
	reduce := store.Src(1)
	if reduce.Op() != uop.OpReduce {
		return sink
	}

	if axisIdx < 0 || axisIdx >= reduce.NSrc()-1 {
		return sink
	}

	// The tiled-reduce lowerer only supports Mul(Index, Index) element nodes.
	// Return sink unchanged for any other reduce shape so the no-op filter in
	// ActionSpace correctly excludes them from the beam search.
	body := reduce.Src(0)
	if body.Op() != uop.OpMul || body.NSrc() < 2 ||
		body.Src(0).Op() != uop.OpIndex || body.Src(1).Op() != uop.OpIndex {
		return sink
	}

	// Indirect-index containment guard (Slice C). The tiled-reduce lowerer
	// assumes both OpIndex operands flatten to affine arithmetic over loop
	// counters. An OpGatherIdx anywhere in either operand's subtree breaks
	// that assumption (the gather load cannot be hoisted into a coalesced
	// tile-load), so the tiling rewrite must skip. Returning sink unchanged
	// keeps the kernel correct under the elementwise lowering path.
	if containsGatherIdx(body.Src(0)) || containsGatherIdx(body.Src(1)) {
		return sink
	}

	targetRange := reduce.Src(axisIdx + 1)
	ra := targetRange.Arg().(uop.RangeArg)
	if uop.RangeIsSymbolic(targetRange) {
		return sink
	}

	TS := int64(tileSize)
	S := uop.RangeSize(targetRange)
	W := (S + TS - 1) / TS

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

	// Split K into k_outer (AxisLoop/Reduce) and k_inner (AxisLoop/Reduce).
	// We use AxisReduce for both to signal sequential accumulation.
	wBound := arena.New(uop.OpConst, uop.Dtypes.Index, nil, W, nil)
	rk_outer := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{wBound}, uop.RangeArg{
		ID:   ra.ID,
		Type: uop.AxisReduce,
	}, nil)

	tsBound := arena.New(uop.OpConst, uop.Dtypes.Index, nil, TS, nil)
	rk_inner := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{tsBound}, uop.RangeArg{
		ID:   maxID + 1,
		Type: uop.AxisReduce,
	}, nil)

	tsConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, TS, nil)
	mul := arena.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{rk_outer, tsConst}, nil, nil)
	add := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, rk_inner}, nil, nil)

	cache := make(map[uint32]uop.UOp)
	newBody := rewriteBody(reduce.Src(0), targetRange, add, cache)

	newReduceSrcs := make([]uop.UOp, reduce.NSrc()+1)
	newReduceSrcs[0] = newBody
	dest := 1
	for i := 1; i < reduce.NSrc(); i++ {
		if i == axisIdx+1 {
			newReduceSrcs[dest] = rk_outer
			newReduceSrcs[dest+1] = rk_inner
			dest += 2
		} else {
			newReduceSrcs[dest] = reduce.Src(i)
			dest++
		}
	}

	newReduce := arena.New(uop.OpReduce, reduce.DType(), newReduceSrcs, reduce.Arg(), fmt.Sprintf("tile:%d", tileSize))

	newStore := arena.New(uop.OpStore, store.DType(), []uop.UOp{store.Src(0), newReduce}, store.Arg(), store.Tag())

	// We also need to add the new range to the END node so Lower can find it.
	newEndSrcs := make([]uop.UOp, end.NSrc()+1)
	newEndSrcs[0] = newStore
	for i := 1; i < end.NSrc(); i++ {
		newEndSrcs[i] = end.Src(i)
	}
	newEndSrcs[end.NSrc()] = rk_inner
	newEnd := arena.New(uop.OpEnd, end.DType(), newEndSrcs, end.Arg(), end.Tag())

	return arena.New(uop.OpSink, sink.DType(), []uop.UOp{newEnd}, sink.Arg(), sink.Tag())
}

// applyUpcast splits the axisIdx-th eligible END-level parallel range (AxisWorkgroup,
// AxisLoop, or AxisLocal — whichever appears first, in END order, that does NOT
// already have an immediately-following AxisUpcast partner) into:
//   - outer range of size N/factor, keeping the original AxisType and ID
//   - inner range of size factor, with AxisType=AxisUpcast and a fresh ID
//
// The inner range is unrolled at lower time; each thread owns a factor-wide
// stripe in that dim. Reject AxisReduce targets (B3 invariant: reduce axes
// never become per-thread unrolled output stripes) and Symbolic ranges.
func applyUpcast(sink uop.UOp, axisIdx int, factor int) uop.UOp {
	if sink.Op() != uop.OpSink {
		return sink
	}
	if factor <= 1 {
		return sink
	}
	arena := sink.Arena()
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return sink
	}

	// Walk END.src[1:] looking for the axisIdx-th eligible parallel range.
	// Eligible = AxisLoop|AxisWorkgroup|AxisLocal AND the next sibling is NOT
	// AxisUpcast (so we skip already-upcasted axes).
	var targetRange uop.UOp
	var targetIdx int
	currIdx := 0
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() != uop.OpRange {
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		if ra.Type == uop.AxisReduce || ra.Type == uop.AxisUpcast {
			continue
		}
		if ra.Type != uop.AxisLoop && ra.Type != uop.AxisWorkgroup && ra.Type != uop.AxisLocal {
			continue
		}
		// Skip if already has an AxisUpcast partner immediately following.
		if i+1 < end.NSrc() {
			nxt := end.Src(i + 1)
			if nxt.Op() == uop.OpRange {
				nra := nxt.Arg().(uop.RangeArg)
				if nra.Type == uop.AxisUpcast {
					continue
				}
			}
		}
		if currIdx == axisIdx {
			targetRange = r
			targetIdx = i
			break
		}
		currIdx++
	}

	if !targetRange.Valid() {
		return sink
	}

	ra := targetRange.Arg().(uop.RangeArg)
	if uop.RangeIsSymbolic(targetRange) {
		return sink
	}

	// Fail-loud guard: AxisUpcast lowering outside emitTiledReduce assigns the
	// upcast range the placeholder expression "0", so each thread silently
	// writes only lane 0 of its factor-wide stripe. emitTiledReduce only runs
	// when the kernel's OpReduce carries a "tile:" tag (set by OptTile). If
	// we're about to mutate a kernel that lacks that prerequisite, refuse
	// rather than producing silently-wrong outputs at runtime.
	//
	// ActionSpace pre-filters this case so BEAM never reaches the panic;
	// callers via direct ApplyOpt/ApplyOpts compose OptTile before OptUpcast.
	if !KernelHasTiledReduce(sink) {
		panic(fmt.Sprintf(
			"codegen: OptUpcast(axis=%d, factor=%d): kernel lacks an OptTile-tagged OpReduce; "+
				"AxisUpcast lowering outside emitTiledReduce silently drops lanes. "+
				"Compose OptTile before OptUpcast, or skip OptUpcast on this kernel.",
			axisIdx, factor))
	}

	F := int64(factor)
	S := uop.RangeSize(targetRange)
	W := (S + F - 1) / F

	// Find max existing Range ID to pick a unique one for the inner range.
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

	// Outer keeps original ID and Type; inner is fresh AxisUpcast.
	wBound := arena.New(uop.OpConst, uop.Dtypes.Index, nil, W, nil)
	rOuter := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{wBound}, uop.RangeArg{
		ID:   ra.ID,
		Type: ra.Type,
	}, nil)
	fBound := arena.New(uop.OpConst, uop.Dtypes.Index, nil, F, nil)
	rInner := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{fBound}, uop.RangeArg{
		ID:   maxID + 1,
		Type: uop.AxisUpcast,
	}, nil)

	// Substitute body's references: oldRange → (rOuter * factor) + rInner.
	fConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, F, nil)
	mul := arena.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{rOuter, fConst}, nil, nil)
	add := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, rInner}, nil, nil)

	store := end.Src(0)
	cache := make(map[uint32]uop.UOp)
	newBody := rewriteBody(store.Src(1), targetRange, add, cache)
	newStore := arena.New(uop.OpStore, store.DType(), []uop.UOp{store.Src(0), newBody}, store.Arg(), store.Tag())

	// Insert rInner immediately after rOuter in END's range list.
	newEndSrcs := make([]uop.UOp, end.NSrc()+1)
	newEndSrcs[0] = newStore
	dest := 1
	for i := 1; i < end.NSrc(); i++ {
		if i == targetIdx {
			newEndSrcs[dest] = rOuter
			newEndSrcs[dest+1] = rInner
			dest += 2
		} else {
			newEndSrcs[dest] = end.Src(i)
			dest++
		}
	}
	newEnd := arena.New(uop.OpEnd, end.DType(), newEndSrcs, end.Arg(), end.Tag())
	return arena.New(uop.OpSink, sink.DType(), []uop.UOp{newEnd}, sink.Arg(), sink.Tag())
}

// applyVectorize splits the axisIdx-th eligible END-level parallel range into:
//   - outer range of size ceil(N/width), keeping the original AxisType and ID
//   - inner range of size width, with AxisType=AxisVectorize and a fresh ID
//
// Eligibility rules (same guard as applyUpcast):
//   - AxisLoop | AxisWorkgroup | AxisLocal only; AxisReduce and AxisUpcast rejected.
//   - The range must NOT already have an AxisUpcast or AxisVectorize partner
//     immediately following it.
//   - Symbolic ranges are rejected.
//
// The body is rewritten: oldRange → (outer * width) + inner.
// The inner (AxisVectorize) range is added to END.src[] immediately after the outer.
//
// Boundary: if N is not divisible by width, outer size = ceil(N/width). The lowerer
// emits per-component scalar guard checks for out-of-range elements. The outer range's
// effective bound extends to cover the padded tile; guards prevent OOB memory access.
func applyVectorize(sink uop.UOp, axisIdx int, width int) uop.UOp {
	if sink.Op() != uop.OpSink {
		return sink
	}
	if width <= 1 {
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
		if r.Op() != uop.OpRange {
			continue
		}
		ra := r.Arg().(uop.RangeArg)
		// Reject reduce and upcast axes.
		if ra.Type == uop.AxisReduce || ra.Type == uop.AxisUpcast || ra.Type == uop.AxisVectorize {
			continue
		}
		if ra.Type != uop.AxisLoop && ra.Type != uop.AxisWorkgroup && ra.Type != uop.AxisLocal {
			continue
		}
		// Skip if already has an AxisUpcast or AxisVectorize partner immediately after.
		if i+1 < end.NSrc() {
			nxt := end.Src(i + 1)
			if nxt.Op() == uop.OpRange {
				nra := nxt.Arg().(uop.RangeArg)
				if nra.Type == uop.AxisUpcast || nra.Type == uop.AxisVectorize {
					continue
				}
			}
		}
		if currIdx == axisIdx {
			targetRange = r
			targetIdx = i
			break
		}
		currIdx++
	}

	if !targetRange.Valid() {
		return sink
	}
	ra := targetRange.Arg().(uop.RangeArg)
	if uop.RangeIsSymbolic(targetRange) {
		return sink
	}

	// Fail-loud guard: AxisVectorize lowering relies on the vecTileActive
	// machinery set inside emitTiledReduce. Without an OptTile-tagged
	// OpReduce, the WGSL store path for an AxisVectorize range silently
	// loses 3 of the 4 lanes (vec4 packaging falls through the placeholder
	// "0" expression). Mirrors the OptUpcast guard in applyUpcast; same
	// predicate, same composition requirement.
	if !KernelHasTiledReduce(sink) {
		panic(fmt.Sprintf(
			"codegen: OptVectorize(axis=%d, width=%d): kernel lacks an OptTile-tagged OpReduce; "+
				"AxisVectorize lowering outside emitTiledReduce silently drops lanes. "+
				"Compose OptTile (and typically OptUpcast) before OptVectorize, or skip OptVectorize on this kernel.",
			axisIdx, width))
	}

	W := int64(width)
	S := uop.RangeSize(targetRange)
	outer := (S + W - 1) / W

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

	outerBound := arena.New(uop.OpConst, uop.Dtypes.Index, nil, outer, nil)
	rOuter := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{outerBound}, uop.RangeArg{
		ID:   ra.ID,
		Type: ra.Type,
	}, nil)
	wBound := arena.New(uop.OpConst, uop.Dtypes.Index, nil, W, nil)
	rInner := arena.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{wBound}, uop.RangeArg{
		ID:   maxID + 1,
		Type: uop.AxisVectorize,
	}, nil)

	wConst := arena.New(uop.OpConst, uop.Dtypes.Index, nil, W, nil)
	mul := arena.New(uop.OpMul, uop.Dtypes.Index, []uop.UOp{rOuter, wConst}, nil, nil)
	add := arena.New(uop.OpAdd, uop.Dtypes.Index, []uop.UOp{mul, rInner}, nil, nil)

	store := end.Src(0)
	cache := make(map[uint32]uop.UOp)
	newBody := rewriteBody(store.Src(1), targetRange, add, cache)
	newStore := arena.New(uop.OpStore, store.DType(), []uop.UOp{store.Src(0), newBody}, store.Arg(), store.Tag())

	newEndSrcs := make([]uop.UOp, end.NSrc()+1)
	newEndSrcs[0] = newStore
	dest := 1
	for i := 1; i < end.NSrc(); i++ {
		if i == targetIdx {
			newEndSrcs[dest] = rOuter
			newEndSrcs[dest+1] = rInner
			dest += 2
		} else {
			newEndSrcs[dest] = end.Src(i)
			dest++
		}
	}
	newEnd := arena.New(uop.OpEnd, end.DType(), newEndSrcs, end.Arg(), end.Tag())
	return arena.New(uop.OpSink, sink.DType(), []uop.UOp{newEnd}, sink.Arg(), sink.Tag())
}

// ApplyOpts applies a sequence of optimizations to a kernel's AST.
func ApplyOpts(item schedule.ExecItem, opts []Opt) schedule.ExecItem {
	if !item.Ast.Valid() {
		return item
	}
	for _, opt := range opts {
		item.Ast = ApplyOpt(item.Ast, opt)
	}
	return item
}
