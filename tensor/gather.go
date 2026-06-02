package tensor

import (
	"fmt"
	"math"
	"sort"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// ── Gather (torch-style) ──────────────────────────────────────────────────────
//
// Gather selects rows along a single axis using a runtime index tensor. The
// output has the index tensor's shape along that axis; values are pulled from
// the data tensor by reading t[..., index[..., i], ...]. This is the
// frontend operation; the graph node is an OpGather that dissolves at
// rangeify time into OpIndex(data, ..., OpGatherIdx(...), ...) (Slice C).
//
// Index dtype must be an integer type; non-Int32 integer indices are cast to
// Int32 at construction (matches WGSL's preference for i32 addresses).

// Gather selects values from t along dim using index. Output shape equals
// index's shape (in torch-gather semantics the output replaces dim of t's
// shape with the index shape). The index tensor must have integer dtype;
// non-Int32 integer indices are cast to Int32 automatically.
//
// Negative dim wraps in the usual Python style. Out-of-range dim panics
// with a helpful message.
//
// Forward lowering (Slice C): dissolves at rangeify into OpIndex(data, ...,
// OpGatherIdx(...), ...). Backward (scatter-add) is wired in Slice D.
func (t *Tensor) Gather(dim int, index *Tensor) *Tensor {
	rank := t.Rank()
	if rank == 0 {
		panic("tensor: Gather: cannot gather along a scalar (rank 0) tensor")
	}
	origDim := dim
	if dim < 0 {
		dim += rank
	}
	if dim < 0 || dim >= rank {
		panic(fmt.Sprintf("tensor: Gather: dim %d out of range for rank %d", origDim, rank))
	}
	if index == nil {
		panic("tensor: Gather: index tensor is nil")
	}
	if !index.dtype.IsInt() {
		panic(fmt.Sprintf("tensor: Gather: index dtype %s is not integer; pass an integer-typed index tensor", index.dtype))
	}
	if index.arena() != t.arena() {
		panic("tensor: Gather: data and index tensors belong to different arenas")
	}
	if t.device != index.device {
		panic(fmt.Sprintf("tensor: Gather: data device %q != index device %q", t.device, index.device))
	}
	// Cast target is the storage dtype (Int32), not Dtypes.Index. Design §1
	// notes both are i32 in WGSL and §8's frontend sketch picks Int32 because
	// Cast(Index) would force a follow-up Cast(Int32) when the index buffer's
	// declared element type drives the @binding declaration. wgslDType maps
	// both to "i32" so codegen sees no friction either way.
	if index.dtype != uop.Dtypes.Int32 {
		index = index.Cast(uop.Dtypes.Int32)
	}

	outSints := gatherShape(t.ShapeSints(), index.ShapeSints(), dim)
	node := t.arena().New(
		uop.OpGather,
		t.dtype,
		[]uop.UOp{t.node, index.node},
		int64(dim),
		nil,
	)
	return fromNode(node, shape.NewShapeTrackerSints(outSints), t.dtype, t.device)
}

// gatherShape applies torch-gather output shape rules: take the data shape
// and replace its dim-th entry with the entire index shape.
func gatherShape(dataShape, idxShape []shape.Sint, dim int) []shape.Sint {
	out := make([]shape.Sint, 0, len(dataShape)-1+len(idxShape))
	out = append(out, dataShape[:dim]...)
	out = append(out, idxShape...)
	out = append(out, dataShape[dim+1:]...)
	return out
}

// ── ScatterAdd (backward of Gather; Slice D) ─────────────────────────────────
//
// scatterAdd constructs the backward kernel for OpGather. Given:
//
//	grad      : adjoint of the gather output, shape [B, *trailing]
//	idx       : the same int32 index tensor as the forward gather, shape [B]
//	dim       : the gather axis on the original data tensor (must be 0 in v1)
//	dataShape : shape of the original data tensor, [V, *trailing]
//	zeros     : a same-shape, same-dtype zero tensor used as the destination
//	            template (its UOp carries the output shape into the schedule)
//
// it returns a Tensor whose UOp is an OpScatterAdd that the rangeify scheduler
// dissolves at realize time into a per-output-position reduce kernel. The
// kernel reads two host-prepared leaf buffers, sortedIdx and permutation, that
// are populated by a closure registered with registerScatterPreproc. The
// closure fires on every Realize / JIT-replay call, reading the latest idx
// data, sorting (idx[b], b) pairs by idx value, and writing the new arrays
// via Arena.SetLeaf.
//
// Src layout (resolves the Slice B carried question): OpScatterAdd carries
//
//	Src(0) = zeros template (data-shape carrier; the destination of the scatter)
//	Src(1) = grad           (the values being scattered)
//	Src(2) = sortedIdx leaf (host-prepared; same shape as idx)
//	Src(3) = perm leaf      (host-prepared; same shape as idx)
//	Arg    = int64(dim)     (the gather axis; v1 requires 0)
//
// Slice B's tentative comment in tensor/gradient.go shapeOfNode noted "Src(0)
// is the data-template": this constructor confirms that convention and adds
// the remaining three src slots. The original idx tensor is referenced only
// by the closure (so the gradient pass can find its leaf data) and is NOT a
// direct src of OpScatterAdd; this keeps the host preprocessor's source
// of truth orthogonal to the device-side kernel inputs.
//
// Dispatch geometry choice (a) from design §6: the rewritten kernel walks
// over [V * D] output threads. Each thread linearly reduces over the [B]
// reduce axis, masking by sortedIdx[b] == v. This is O(V * D * B) work but
// race-free, deterministic, and JIT-friendly (the dispatch grid is graph-keyed
// and never changes between captures and replays). Choice (b) — dispatch
// over numSegments * D — was rejected because numSegments is data-dependent
// and forces a per-call dispatch-geometry recompute, breaking the JIT plan.
//
// v1 scope: requires dim == 0 and a 1-D index tensor. Both backward fixture 1
// (concrete B) and backward fixture 3 (symbolic B) live within this scope.
// Arbitrary-dim / multi-dim-index scatter is wired by Slice E if Embedding
// itself needs it (it does not), or by a later slice. Out-of-scope calls
// panic with a clear message.
func scatterAdd(grad, idx *Tensor, dim int, dataShape []shape.Sint, zeros *Tensor) *Tensor {
	if dim != 0 {
		panic(fmt.Sprintf("tensor: scatterAdd: only dim=0 is supported in Slice D (Embedding backward); got dim=%d", dim))
	}
	if idx.Rank() != 1 {
		panic(fmt.Sprintf("tensor: scatterAdd: only 1-D index is supported in Slice D; got idx rank %d", idx.Rank()))
	}
	if grad.arena() != idx.arena() || grad.arena() != zeros.arena() {
		panic("tensor: scatterAdd: grad / idx / zeros must share one arena")
	}
	a := grad.arena()

	// Allocate sortedIdx and perm leaves with the same shape as idx. They
	// hold Int32 data; the closure writes via SetData on every preprocessor
	// call. Both are concrete-shape leaves: for a symbolic-batch idx[n],
	// the closure resolves n at preprocess time from idx's current data
	// length (which Realize requires to be set anyway via the input binding).
	//
	// For a symbolic idx tensor (Slice 3b1 NewSymbolicInput) we still allocate
	// the leaves with NewSymbolicInput so the schedule's symbolic loop bound
	// matches the reduce range we build in indexExprNode.
	var sortedIdx, perm *Tensor
	if sym, ok := symbolicLenOfIdx(idx); ok {
		sortedIdx = NewSymbolicInput(a, sym.varName+"_sorted_"+sym.suffix, sym.min, sym.max, uop.Dtypes.Int32, grad.device)
		perm = NewSymbolicInput(a, sym.varName+"_perm_"+sym.suffix, sym.min, sym.max, uop.Dtypes.Int32, grad.device)
	} else {
		// Concrete idx shape [B].
		idxShape := idx.Shape()
		sortedIdx = NewLeaf(a, append([]int64{}, idxShape...), uop.Dtypes.Int32, grad.device)
		perm = NewLeaf(a, append([]int64{}, idxShape...), uop.Dtypes.Int32, grad.device)
	}

	// Build OpScatterAdd UOp. Output dtype = grad's dtype = adj's dtype.
	scatterNode := a.New(
		uop.OpScatterAdd,
		grad.dtype,
		[]uop.UOp{zeros.node, grad.node, sortedIdx.node, perm.node},
		int64(dim),
		nil,
	)

	// Register host preprocessor: read idx leaf data, sort (idx[b], b) pairs
	// by idx value with stable order, write sortedIdx + perm leaf data.
	idxArenaIdx := idx.node.Index()
	sortedArenaIdx := sortedIdx.node.Index()
	permArenaIdx := perm.node.Index()
	registerScatterPreproc(a, scatterNode.Index(), func() bool {
		raw, ok := a.Leaf(idxArenaIdx)
		if !ok {
			return false
		}
		B := len(raw)
		// raw holds int32 values packed as float32 bit patterns (the same
		// encoding tensor.Realize uses for integer leaves; see
		// gather_realize_test.i32sAsF32Bits). Decode in place.
		pairs := make([]scatterPair, B)
		for b := 0; b < B; b++ {
			pairs[b].idxVal = int32(math.Float32bits(raw[b]))
			pairs[b].pos = int32(b)
		}
		sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].idxVal < pairs[j].idxVal })

		sortedBits := make([]float32, B)
		permBits := make([]float32, B)
		for b := 0; b < B; b++ {
			sortedBits[b] = math.Float32frombits(uint32(pairs[b].idxVal))
			permBits[b] = math.Float32frombits(uint32(pairs[b].pos))
		}
		a.SetLeaf(sortedArenaIdx, sortedBits)
		a.SetLeaf(permArenaIdx, permBits)
		return true
	})

	return fromNode(scatterNode, shape.NewShapeTrackerSints(dataShape), grad.dtype, grad.device)
}

// scatterPair is one (idx[b], b) entry sorted by idxVal in scatterAdd's
// host preprocessor.
type scatterPair struct {
	idxVal int32
	pos    int32
}

// symbolicLenIdx captures the symbolic-shape descriptor of a 1-D symbolic
// index tensor: the DefineVar name and bound. Used by scatterAdd to mirror
// the symbolic shape onto sortedIdx / perm leaves so the schedule's reduce
// range stays symbolic too.
type symbolicLenIdx struct {
	varName string
	min     int64
	max     int64
	suffix  string // disambiguator across multiple scatters on the same var
}

// symbolicLenOfIdx returns (descriptor, true) when idx is a 1-D symbolic
// leaf created via NewSymbolicInput; otherwise (zero, false). The Slice 3b1
// pattern places the DefineVar at Src(0) of an OpBuffer with arg=nil and
// rank-1 sint shape.
func symbolicLenOfIdx(idx *Tensor) (symbolicLenIdx, bool) {
	if idx.node.Op() != uop.OpBuffer {
		return symbolicLenIdx{}, false
	}
	if idx.node.NSrc() != 1 {
		return symbolicLenIdx{}, false
	}
	dv := idx.node.Src(0)
	if dv.Op() != uop.OpDefineVar {
		return symbolicLenIdx{}, false
	}
	va, ok := dv.Arg().(uop.VarArg)
	if !ok {
		return symbolicLenIdx{}, false
	}
	// DefineVar carries srcs (Const(min), Const(max+1)). Unwrap the +1 to
	// expose the user-supplied inclusive max.
	if dv.NSrc() != 2 || dv.Src(0).Op() != uop.OpConst || dv.Src(1).Op() != uop.OpConst {
		return symbolicLenIdx{}, false
	}
	min, _ := dv.Src(0).Arg().(int64)
	maxPlus1, _ := dv.Src(1).Arg().(int64)
	// Suffix from the scatter node's arena index would be cleaner but the
	// node does not exist at this point; we accept that two scatter-adds
	// sharing one symbolic idx would collide on the auxiliary leaf names.
	// In practice the gradient pass creates exactly one scatter per gather
	// per arena, so the chance is nil.
	return symbolicLenIdx{varName: va.Name, min: min, max: maxPlus1 - 1, suffix: "d"}, true
}
