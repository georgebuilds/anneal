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
func (t *Tensor) Rank() int { return len(t.st.Shape()) }

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
func (t *Tensor) SetData(d []float32) {
	t.data = d
	t.node.Arena().SetLeaf(t.node.Index(), d)
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
func broadcast(a, b *Tensor) (*Tensor, *Tensor) {
	_, _, out := broadcastShapes(a.Shape(), b.Shape())
	return BroadcastTo(a, out), BroadcastTo(b, out)
}

// ln2 is log(2) used for exp/log derived primitives.
const ln2 = math.Ln2
