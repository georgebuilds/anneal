package tensor

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// ── Movement ops ──────────────────────────────────────────────────────────────
//
// Every movement op creates a new UOp node in the graph AND updates the
// ShapeTracker. The UOp arg encodes the operation parameters so the scheduler
// (Phase 7) can reconstruct the full transformation.

// Reshape returns a tensor with the same data arranged in newShape.
func (t *Tensor) Reshape(newShape []int64) *Tensor {
	if shapesEqual(t.Shape(), newShape) {
		return t
	}
	newST := t.st.Reshape(newShape)
	arg := cloneShape(newShape)
	node := t.arena().New(uop.OpReshape, t.dtype, []uop.UOp{t.node}, []int64(arg), nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// Expand broadcasts t to newShape. Each dim of newShape must equal t's dim or
// t's dim must be 1 (broadcast). Rank must match.
func (t *Tensor) Expand(newShape []int64) *Tensor {
	if shapesEqual(t.Shape(), newShape) {
		return t
	}
	newST := t.st.Expand(newShape)
	arg := cloneShape(newShape)
	node := t.arena().New(uop.OpExpand, t.dtype, []uop.UOp{t.node}, []int64(arg), nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// Permute reorders dimensions according to order, a permutation of [0, rank).
func (t *Tensor) Permute(order []int) *Tensor {
	n := t.Rank()
	if len(order) != n {
		panic(fmt.Sprintf("tensor: permute: order length %d != rank %d", len(order), n))
	}
	newST := t.st.Permute(order)
	arg := make([]int64, len(order))
	for i, o := range order {
		arg[i] = int64(o)
	}
	node := t.arena().New(uop.OpPermute, t.dtype, []uop.UOp{t.node}, arg, nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// Pad adds zero padding. arg[i] = {lo, hi} pads dim i with lo elements before
// and hi elements after.
func (t *Tensor) Pad(arg [][2]int64) *Tensor {
	allZero := true
	for _, p := range arg {
		if p[0] != 0 || p[1] != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return t
	}
	newST := t.st.Pad(arg)
	node := t.arena().New(uop.OpPad, t.dtype, []uop.UOp{t.node}, [][2]int64(arg), nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// Shrink selects a sub-region. arg[i] = {lo, hi} selects [lo, hi) of dim i.
func (t *Tensor) Shrink(arg [][2]int64) *Tensor {
	sh := t.Shape()
	identity := true
	for i, p := range arg {
		if p[0] != 0 || p[1] != sh[i] {
			identity = false
			break
		}
	}
	if identity {
		return t
	}
	newST := t.st.Shrink(arg)
	node := t.arena().New(uop.OpShrink, t.dtype, []uop.UOp{t.node}, [][2]int64(arg), nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// Flip reverses elements along the specified axes.
func (t *Tensor) Flip(axes []bool) *Tensor {
	newST := t.st.Flip(axes)
	arg := make([]int64, len(axes))
	for i, f := range axes {
		if f {
			arg[i] = 1
		}
	}
	node := t.arena().New(uop.OpFlip, t.dtype, []uop.UOp{t.node}, arg, nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// ── Convenience wrappers ──────────────────────────────────────────────────────

// Transpose swaps the last two dimensions (or the only two dims for 2-D tensors).
func (t *Tensor) Transpose() *Tensor {
	n := t.Rank()
	if n < 2 {
		panic("tensor: Transpose requires at least 2 dimensions")
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	order[n-2], order[n-1] = n-1, n-2
	return t.Permute(order)
}

// T is an alias for Transpose for 2-D tensors.
func (t *Tensor) T() *Tensor { return t.Transpose() }

// Flatten collapses all dimensions into a single dimension.
func (t *Tensor) Flatten() *Tensor {
	n := int64(1)
	for _, s := range t.Shape() {
		n *= s
	}
	return t.Reshape([]int64{n})
}

// FlattenFrom collapses dimensions from startDim onward into a single dimension,
// keeping dimensions before startDim intact.
func (t *Tensor) FlattenFrom(startDim int) *Tensor {
	sh := t.Shape()
	if startDim < 0 {
		startDim += len(sh)
	}
	if startDim >= len(sh) {
		return t
	}
	newShape := make([]int64, startDim+1)
	copy(newShape, sh[:startDim])
	tail := int64(1)
	for _, s := range sh[startDim:] {
		tail *= s
	}
	newShape[startDim] = tail
	return t.Reshape(newShape)
}

// Unsqueeze inserts a size-1 dimension at position dim.
func (t *Tensor) Unsqueeze(dim int) *Tensor {
	sh := t.Shape()
	if dim < 0 {
		dim += len(sh) + 1
	}
	newShape := make([]int64, len(sh)+1)
	copy(newShape[:dim], sh[:dim])
	newShape[dim] = 1
	copy(newShape[dim+1:], sh[dim:])
	return t.Reshape(newShape)
}

// newShapeTracker is a package-level alias to avoid importing shape in caller files.
func newShapeTracker(sh []int64) shape.ShapeTracker {
	return shape.NewShapeTracker(sh)
}

// ReshapeSints reshapes t to newShape which may contain symbolic dimensions.
// For fully-concrete newShape, falls back to the regular []int64 Reshape path.
// For shapes with symbolic dims, creates an OpReshape with ShapeSintArg.
func (t *Tensor) ReshapeSints(newShape []shape.Sint) *Tensor {
	// Fast path: all concrete → use regular Reshape (byte-identical static path)
	if !shape.HasSymbolic(newShape) {
		return t.Reshape(shape.AsInts(newShape))
	}
	if shape.SintShapesEqual(t.st.ShapeSints(), newShape) {
		return t
	}
	newST := t.st.ReshapeSints(newShape)
	arg := toShapeSintArg(newShape)
	node := t.arena().New(uop.OpReshape, t.dtype, []uop.UOp{t.node}, arg, nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// ExpandSints broadcasts t to newShape which may contain symbolic dimensions.
// For fully-concrete newShape, falls back to the regular []int64 Expand path.
// For shapes with symbolic dims, creates an OpExpand with ShapeSintArg.
func (t *Tensor) ExpandSints(newShape []shape.Sint) *Tensor {
	if !shape.HasSymbolic(newShape) {
		return t.Expand(shape.AsInts(newShape))
	}
	if shape.SintShapesEqual(t.st.ShapeSints(), newShape) {
		return t
	}
	newST := t.st.ExpandSints(newShape)
	arg := toShapeSintArg(newShape)
	node := t.arena().New(uop.OpExpand, t.dtype, []uop.UOp{t.node}, arg, nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// toShapeSintArg converts a []shape.Sint to a uop.ShapeSintArg for use as a
// UOp arg. SymInt dims encode their DefineVar by name in VarName (Mul=1 — the
// tensor-side construction path produces only bare-DefineVar bounds; derived
// bounds are introduced later by the scheduler).
//
// Enforces the SPEC §10 ShapeSintArg V-on-symbolic-dim invariant: when Sym=true,
// V must be 0. The VarName encoding makes the arg structurally portable across
// arenas (Option B Slice 4).
func toShapeSintArg(sh []shape.Sint) uop.ShapeSintArg {
	result := make(uop.ShapeSintArg, len(sh))
	for i, s := range sh {
		result[i] = toShapeDim(s, i)
	}
	return result
}

// toShapeDim converts a single shape.Sint to a uop.ShapeDim, enforcing the
// SPEC §10 V-on-symbolic-dim invariant. Shared by toShapeSintArg and the
// Pad/Shrink Sint constructors so encoding stays consistent across all three.
func toShapeDim(s shape.Sint, dimIdx int) uop.ShapeDim {
	if sym, ok := s.(shape.SymInt); ok {
		name, mul := uop.SymBoundFactor(sym.Node)
		d := uop.ShapeDim{Sym: true, VarName: name, Mul: mul}
		if d.V != 0 {
			panic(fmt.Sprintf("uop: ShapeDim.V must be 0 when Sym=true (SPEC §10); got V=%d VarName=%q at dim %d", d.V, d.VarName, dimIdx))
		}
		return d
	}
	v, _ := s.ConstValue()
	return uop.ShapeDim{V: v}
}

// PadSints adds padding using a [][2]shape.Sint mask; pad amounts may be
// symbolic. For all-concrete arg, falls back to the regular [][2]int64
// Pad path (byte-identical static behaviour).
func (t *Tensor) PadSints(arg [][2]shape.Sint) *Tensor {
	if !maskHasSymbolic(arg) {
		conc := make([][2]int64, len(arg))
		for i, p := range arg {
			v0, _ := p[0].ConstValue()
			v1, _ := p[1].ConstValue()
			conc[i] = [2]int64{v0, v1}
		}
		return t.Pad(conc)
	}
	allZero := true
	for _, p := range arg {
		v0, ok0 := p[0].ConstValue()
		v1, ok1 := p[1].ConstValue()
		if !ok0 || !ok1 || v0 != 0 || v1 != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return t
	}
	newST := t.st.PadSints(arg)
	uopArg := make(uop.PadSintArg, len(arg))
	for i, p := range arg {
		uopArg[i] = [2]uop.ShapeDim{toShapeDim(p[0], i), toShapeDim(p[1], i)}
	}
	node := t.arena().New(uop.OpPad, t.dtype, []uop.UOp{t.node}, uopArg, nil)
	return fromNode(node, newST, t.dtype, t.device)
}

// ShrinkSints selects a sub-region using a [][2]shape.Sint mask; lo/hi may be
// symbolic. For all-concrete arg, falls back to the regular [][2]int64 path.
func (t *Tensor) ShrinkSints(arg [][2]shape.Sint) *Tensor {
	if !maskHasSymbolic(arg) {
		conc := make([][2]int64, len(arg))
		for i, p := range arg {
			v0, _ := p[0].ConstValue()
			v1, _ := p[1].ConstValue()
			conc[i] = [2]int64{v0, v1}
		}
		return t.Shrink(conc)
	}
	newST := t.st.ShrinkSints(arg)
	uopArg := make(uop.ShrinkSintArg, len(arg))
	for i, p := range arg {
		uopArg[i] = [2]uop.ShapeDim{toShapeDim(p[0], i), toShapeDim(p[1], i)}
	}
	node := t.arena().New(uop.OpShrink, t.dtype, []uop.UOp{t.node}, uopArg, nil)
	return fromNode(node, newST, t.dtype, t.device)
}

func maskHasSymbolic(arg [][2]shape.Sint) bool {
	for _, p := range arg {
		if _, ok := p[0].ConstValue(); !ok {
			return true
		}
		if _, ok := p[1].ConstValue(); !ok {
			return true
		}
	}
	return false
}
