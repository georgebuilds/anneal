package codegen

import (
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// OptKind identifies the type of a kernel optimization.
type OptKind int

const (
	// OptIdentity returns the kernel AST unchanged.
	OptIdentity OptKind = iota
	// OptLocal splits an axis into workgroup and local dimensions.
	OptLocal
	// Future kinds: OptUpcast, OptUnroll, OptPadTo, OptSwap
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
	if ra.Symbolic {
		// Multi-dim symbolic dispatch is out of scope for B1
		return sink
	}

	// Split S into wg_size = ceil(S/L) and local_size = L
	L := int64(localSize)
	S := ra.Size
	W := (S + L - 1) / L

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
	rwg := arena.New(uop.OpRange, uop.Dtypes.Index, nil, uop.RangeArg{
		ID:   ra.ID,
		Size: W,
		Type: uop.AxisWorkgroup,
	}, nil)

	rloc := arena.New(uop.OpRange, uop.Dtypes.Index, nil, uop.RangeArg{
		ID:   maxID + 1,
		Size: L,
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
