package tensor

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// Tensor is a lazy handle over a UOp graph node. It owns no data and computes
// nothing; every op constructs graph. Realization happens via Realize().
type Tensor struct {
	node   uop.UOp
	st     shape.ShapeTracker
	dtype  *uop.DType
	device string
	data   []float32 // nil until realized; set by Realize() or SetData()
}

// Node returns the underlying UOp graph node.
func (t *Tensor) Node() uop.UOp { return t.node }

// Shape returns the current logical shape.
func (t *Tensor) Shape() []int64 { return t.st.Shape() }

// Rank returns the number of dimensions.
// Uses ShapeSints so it works for symbolic shapes without panicking.
func (t *Tensor) Rank() int { return len(t.st.ShapeSints()) }

// DType returns the element data type.
func (t *Tensor) DType() *uop.DType { return t.dtype }

// Device returns the device placeholder string.
func (t *Tensor) Device() string { return t.device }

// ST returns the ShapeTracker (for inspection/testing).
func (t *Tensor) ST() shape.ShapeTracker { return t.st }

// Arena returns the arena this tensor's node belongs to.
func (t *Tensor) Arena() *uop.Arena { return t.node.Arena() }

func (t *Tensor) arena() *uop.Arena { return t.node.Arena() }

// IsLeaf reports whether this tensor is a parameter/buffer leaf.
// Phase 6 uses this to identify differentiation targets.
func (t *Tensor) IsLeaf() bool { return t.node.Op() == uop.OpBuffer }

// Data returns the realized float32 data, or nil if not yet realized.
func (t *Tensor) Data() []float32 { return t.data }

// SetData attaches concrete float32 data to a leaf tensor before realization.
// This is the mechanism for providing input values to the GPU pipeline.
// The slice is owned by the caller; Realize() reads but does not mutate it.
//
// For f16 and bf16 tensors, SetData creates a quantized copy of the input
// data so that host-side inspection (via Data()) matches device-side precision.
func (t *Tensor) SetData(d []float32) {
	t.data = d
	if s := t.dtype.Scalar(); s == uop.Dtypes.Float16 || s == uop.Dtypes.BFloat16 {
		quantized := make([]float32, len(d))
		for i, v := range d {
			quantized[i] = t.dtype.Quantize(v)
		}
		t.data = quantized
	}
	t.node.Arena().SetLeaf(t.node.Index(), t.data)
}

// IsRealized reports whether this tensor has concrete data.
func (t *Tensor) IsRealized() bool { return t.data != nil }

// ── Construction ──────────────────────────────────────────────────────────────

// NewLeaf creates a leaf tensor backed by a BUFFER node — a trainable parameter
// or external input. Phase 6 differentiates w.r.t. leaves; Phase 9 updates them.
// The shape is stored in the arg so the gradient pass can recover it without
// a ShapeTracker (all movement-op args already encode shape; Buffer was the gap).
func NewLeaf(a *uop.Arena, sh []int64, dtype *uop.DType, device string) *Tensor {
	node := a.New(uop.OpBuffer, dtype, nil, cloneShape(sh), nil)
	return &Tensor{node: node, st: shape.NewShapeTracker(sh), dtype: dtype, device: device}
}

// NewSymbolicInput creates a leaf tensor backed by a BUFFER node whose shape
// contains one symbolic dimension [n] where n ∈ [min, max]. The dimension is
// represented as a DefineVar UOp stored as src[0] of the BUFFER node (arg=nil).
// The scheduler recognises this pattern and builds a symbolic OpRange for n.
// Use RealizeWithBinding to provide a concrete value at dispatch time.
func NewSymbolicInput(a *uop.Arena, name string, min, max int64, dtype *uop.DType, device string) *Tensor {
	defVar := a.DefineVar(name, min, max)
	node := a.New(uop.OpBuffer, dtype, []uop.UOp{defVar}, nil, nil)
	sh := []shape.Sint{shape.SymInt{Node: defVar}}
	return &Tensor{node: node, st: shape.NewShapeTrackerSints(sh), dtype: dtype, device: device}
}

// NewSymbolicBatchInput creates a leaf tensor with shape [n, d0, d1, ...] where
// n ∈ [min, max] is the symbolic outer (batch) dim and d0... are concrete inner dims.
// The arg is a ShapeSintArg encoding the full shape; the scheduler builds a
// symbolic OpRange for n and concrete OpRanges for d0....
// Use RealizeWithBinding to provide the concrete batch size at dispatch time.
func NewSymbolicBatchInput(a *uop.Arena, name string, min, max int64, innerShape []int64, dtype *uop.DType, device string) *Tensor {
	defVar := a.DefineVar(name, min, max)
	arg := make(uop.ShapeSintArg, 1+len(innerShape))
	arg[0] = uop.ShapeDim{Sym: true, VarIdx: defVar.Index()}
	// SPEC §10 ShapeSintArg V-on-symbolic-dim invariant: Sym=true requires V=0.
	if arg[0].V != 0 {
		panic(fmt.Sprintf("uop: ShapeSintArg.V must be 0 when Sym=true (SPEC §10); got V=%d VarIdx=%d at dim 0", arg[0].V, arg[0].VarIdx))
	}
	for i, s := range innerShape {
		arg[i+1] = uop.ShapeDim{V: s}
	}
	node := a.New(uop.OpBuffer, dtype, []uop.UOp{defVar}, uop.ShapeSintArg(arg), nil)
	sh := make([]shape.Sint, 1+len(innerShape))
	sh[0] = shape.SymInt{Node: defVar}
	for i, s := range innerShape {
		sh[i+1] = shape.Const(s)
	}
	return &Tensor{node: node, st: shape.NewShapeTrackerSints(sh), dtype: dtype, device: device}
}

// ShapeSints returns the Sint slice of the current logical shape.
// Use this to inspect symbolic dimensions without forcing concretisation.
func (t *Tensor) ShapeSints() []shape.Sint { return t.st.ShapeSints() }

// ConstScalar creates a scalar constant graph node.
func ConstScalar(a *uop.Arena, val float64, dtype *uop.DType, device string) *Tensor {
	var arg any
	if dtype.IsFloat() {
		arg = val
	} else {
		arg = int64(val)
	}
	node := a.New(uop.OpConst, dtype, nil, arg, nil)
	return &Tensor{node: node, st: shape.NewShapeTracker([]int64{}), dtype: dtype, device: device}
}

// Full creates a constant tensor of shape sh filled with val.
// Builds: CONST → RESHAPE([1,...,1]) → EXPAND(sh).
func Full(a *uop.Arena, sh []int64, val float64, dtype *uop.DType, device string) *Tensor {
	t := ConstScalar(a, val, dtype, device)
	if len(sh) == 0 {
		return t
	}
	ones := make([]int64, len(sh))
	for i := range ones {
		ones[i] = 1
	}
	t = t.Reshape(ones)
	t = t.Expand(sh)
	return t
}

// FullSints creates a constant tensor of Sint shape sh filled with val.
// For fully-concrete shapes, delegates to Full. For symbolic shapes, builds a
// scalar constant and expands it via ReshapeSints + ExpandSints.
func FullSints(a *uop.Arena, sh []shape.Sint, val float64, dtype *uop.DType, device string) *Tensor {
	if !shape.HasSymbolic(sh) {
		return Full(a, shape.AsInts(sh), val, dtype, device)
	}
	t := ConstScalar(a, val, dtype, device)
	if len(sh) == 0 {
		return t
	}
	// Reshape scalar to all-ones shape, then expand to target.
	ones := make([]int64, len(sh))
	for i := range ones {
		ones[i] = 1
	}
	t = t.Reshape(ones)
	t = t.ExpandSints(sh)
	return t
}

// Zeros returns a zero-filled constant tensor.
func Zeros(a *uop.Arena, sh []int64, dtype *uop.DType, device string) *Tensor {
	return Full(a, sh, 0.0, dtype, device)
}

// Ones returns a ones-filled constant tensor.
func Ones(a *uop.Arena, sh []int64, dtype *uop.DType, device string) *Tensor {
	return Full(a, sh, 1.0, dtype, device)
}

// Arange creates a 1-D graph node representing [0, 1, ..., n-1].
// Represented as a BUFFER leaf to be filled at realize time (index range machinery
// is Phase 7; this seam is correct for the scheduler to complete).
func Arange(a *uop.Arena, n int64, dtype *uop.DType, device string) *Tensor {
	node := a.New(uop.OpBuffer, dtype, nil, n, nil)
	return &Tensor{node: node, st: shape.NewShapeTracker([]int64{n}), dtype: dtype, device: device}
}

// RandnGraph creates a graph node representing a sample from N(0,1) with the given shape.
// At realize time the scheduler materialises this as a random-fill kernel.
func RandnGraph(a *uop.Arena, sh []int64, dtype *uop.DType, device string) *Tensor {
	node := a.New(uop.OpBuffer, dtype, nil, "randn", nil)
	return &Tensor{node: node, st: shape.NewShapeTracker(sh), dtype: dtype, device: device}
}

// ── Internal helpers ──────────────────────────────────────────────────────────

func fromNode(node uop.UOp, st shape.ShapeTracker, dtype *uop.DType, device string) *Tensor {
	return &Tensor{node: node, st: st, dtype: dtype, device: device}
}

func cloneShape(s []int64) []int64 {
	c := make([]int64, len(s))
	copy(c, s)
	return c
}

func shapesEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// BroadcastTo returns t expanded to targetShape, prepending rank-1 dims as needed.
// Exported so nn sub-package can use it.
func BroadcastTo(t *Tensor, targetShape []int64) *Tensor {
	cur := t.Shape()
	if len(cur) > len(targetShape) {
		panic(fmt.Sprintf("tensor: broadcastTo: cannot broadcast shape %v to %v", cur, targetShape))
	}
	if len(cur) < len(targetShape) {
		newShape := make([]int64, len(targetShape))
		off := len(targetShape) - len(cur)
		for i := 0; i < off; i++ {
			newShape[i] = 1
		}
		copy(newShape[off:], cur)
		t = t.Reshape(newShape)
	}
	if !shapesEqual(t.Shape(), targetShape) {
		t = t.Expand(targetShape)
	}
	return t
}

// BroadcastToSints returns t expanded to targetShape (Sint slice), prepending
// rank-1 dims as needed. Handles symbolic dims in targetShape via ReshapeSints
// and ExpandSints.
func BroadcastToSints(t *Tensor, targetShape []shape.Sint) *Tensor {
	cur := t.ShapeSints()
	if len(cur) > len(targetShape) {
		panic(fmt.Sprintf("tensor: broadcastToSints: rank too high %d > %d", len(cur), len(targetShape)))
	}
	if len(cur) < len(targetShape) {
		newShape := make([]shape.Sint, len(targetShape))
		off := len(targetShape) - len(cur)
		for i := 0; i < off; i++ {
			newShape[i] = shape.Const(1)
		}
		copy(newShape[off:], cur)
		t = t.ReshapeSints(newShape)
	}
	if !shape.SintShapesEqual(t.ShapeSints(), targetShape) {
		t = t.ExpandSints(targetShape)
	}
	return t
}

// broadcastShapesSints computes the Sint broadcast output shape for two shapes.
// Concrete dims follow normal broadcast rules; symbolic dims must be structurally
// equal, or one side must be concrete-1 (which the symbolic dim wins).
func broadcastShapesSints(a, b []shape.Sint) []shape.Sint {
	na, nb := len(a), len(b)
	n := na
	if nb > n {
		n = nb
	}
	out := make([]shape.Sint, n)
	for i := 0; i < n; i++ {
		ai := i - (n - na)
		bi := i - (n - nb)
		var av, bv shape.Sint
		if ai >= 0 {
			av = a[ai]
		} else {
			av = shape.Const(1)
		}
		if bi >= 0 {
			bv = b[bi]
		} else {
			bv = shape.Const(1)
		}
		if shape.SintShapesEqual([]shape.Sint{av}, []shape.Sint{bv}) {
			out[i] = av
			continue
		}
		avv, aok := av.ConstValue()
		bvv, bok := bv.ConstValue()
		switch {
		case aok && avv == 1:
			out[i] = bv
		case bok && bvv == 1:
			out[i] = av
		case aok && bok && avv == bvv:
			out[i] = av
		default:
			panic(fmt.Sprintf("tensor: broadcastShapesSints: incompatible dims at %d", i))
		}
	}
	return out
}

// broadcastShapes returns aligned shapes and broadcast output shape.
func broadcastShapes(a, b []int64) (aOut, bOut []int64, out []int64) {
	na, nb := len(a), len(b)
	n := na
	if nb > n {
		n = nb
	}
	a2 := make([]int64, n)
	b2 := make([]int64, n)
	for i := 0; i < n; i++ {
		if i < n-na {
			a2[i] = 1
		} else {
			a2[i] = a[i-(n-na)]
		}
		if i < n-nb {
			b2[i] = 1
		} else {
			b2[i] = b[i-(n-nb)]
		}
	}
	out = make([]int64, n)
	for i := 0; i < n; i++ {
		switch {
		case a2[i] == b2[i]:
			out[i] = a2[i]
		case a2[i] == 1:
			out[i] = b2[i]
		case b2[i] == 1:
			out[i] = a2[i]
		default:
			panic(fmt.Sprintf("tensor: incompatible broadcast shapes %v and %v", a, b))
		}
	}
	return a2, b2, out
}

// broadcast returns a and b expanded to their common broadcast shape.
// For purely concrete shapes it uses broadcastShapes. For symbolic shapes
// (Option A: batch is the outermost dim), it uses Sint-aware broadcast.
func broadcast(a, b *Tensor) (*Tensor, *Tensor) {
	aSints := a.st.ShapeSints()
	bSints := b.st.ShapeSints()
	if shape.SintShapesEqual(aSints, bSints) {
		return a, b
	}
	if shape.HasSymbolic(aSints) || shape.HasSymbolic(bSints) {
		out := broadcastShapesSints(aSints, bSints)
		return BroadcastToSints(a, out), BroadcastToSints(b, out)
	}
	_, _, out := broadcastShapes(a.Shape(), b.Shape())
	return BroadcastTo(a, out), BroadcastTo(b, out)
}

// ln2 is log(2) used for exp/log derived primitives.
const ln2 = math.Ln2
