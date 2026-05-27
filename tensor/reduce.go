package tensor

import (
	"fmt"
	"sort"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// ── Reduce ops ────────────────────────────────────────────────────────────────

// reduce is the shared implementation of all reduction ops.
// axes may contain negative indices; duplicates are silently de-duped.
func (t *Tensor) reduce(op uop.Op, axes []int, keepdim bool) *Tensor {
	sints := t.ShapeSints()
	rank := len(sints)

	// Normalize axes.
	seen := make(map[int]bool, len(axes))
	normAxes := make([]int, 0, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			panic(fmt.Sprintf("tensor: reduce: axis %d out of range for rank %d", ax, rank))
		}
		if !seen[ax] {
			seen[ax] = true
			normAxes = append(normAxes, ax)
		}
	}
	sort.Ints(normAxes)

	// Compute output Sint shape (preserves symbolic dims on non-reduced axes).
	outSints := make([]shape.Sint, 0, rank)
	for i, s := range sints {
		isReduced := seen[i]
		if !isReduced {
			outSints = append(outSints, s)
		} else if keepdim {
			outSints = append(outSints, shape.Const(1))
		}
	}

	arg := uop.ReduceArg{Op: op, Axes: normAxes}
	node := t.arena().New(uop.OpReduceAxis, t.dtype, []uop.UOp{t.node}, arg, nil)
	return fromNode(node, shape.NewShapeTrackerSints(outSints), t.dtype, t.device)
}

// Sum reduces along axes, optionally keeping dimensions.
// If axes is nil or empty, reduces over all dimensions.
func (t *Tensor) Sum(axes []int, keepdim bool) *Tensor {
	if len(axes) == 0 {
		axes = allAxes(t.Rank())
	}
	return t.reduce(uop.OpAdd, axes, keepdim)
}

// Max reduces by taking the maximum along axes.
func (t *Tensor) Max(axes []int, keepdim bool) *Tensor {
	if len(axes) == 0 {
		axes = allAxes(t.Rank())
	}
	return t.reduce(uop.OpMax, axes, keepdim)
}

// Mean computes sum / count along axes. The divisor is injected as a graph
// constant so the scheduler sees a pure divide op (not a Python-side scalar).
func (t *Tensor) Mean(axes []int, keepdim bool) *Tensor {
	if len(axes) == 0 {
		axes = allAxes(t.Rank())
	}
	sints := t.ShapeSints()
	n := int64(1)
	for _, ax := range axes {
		a := ax
		if a < 0 {
			a += len(sints)
		}
		sv, ok := sints[a].ConstValue()
		if !ok {
			panic("tensor: Mean: reducing over symbolic axis not supported")
		}
		n *= sv
	}
	summed := t.Sum(axes, keepdim)
	dtype := uop.LeastUpperDType(t.dtype, uop.Dtypes.Float32)
	divisor := FullSints(t.arena(), summed.ShapeSints(), float64(n), dtype, t.device)
	return summed.Cast(dtype).Div(divisor)
}

// ── Matmul ────────────────────────────────────────────────────────────────────

// Matmul computes matrix multiplication via broadcast-multiply-reduce.
//
// For A[..., M, K] and B[..., K, N]:
//  1. Unsqueeze A to [..., M, K, 1], expand to [..., M, K, N].
//  2. Unsqueeze B to [..., 1, K, N], expand to [..., M, K, N].
//  3. Multiply element-wise → [..., M, K, N].
//  4. Sum over K (axis=-2) → [..., M, N].
//
// Supports symbolic outer (batch) dims in A (Option A): K and N must be concrete.
func (t *Tensor) Matmul(other *Tensor) *Tensor {
	aSints := t.ShapeSints()
	bSints := other.ShapeSints()

	if len(aSints) == 0 || len(bSints) == 0 {
		panic("tensor: Matmul: operands must be at least 1D")
	}

	// K = innermost dim of A (must be concrete for Option A)
	Ks := aSints[len(aSints)-1]
	Kv, ok := Ks.ConstValue()
	if !ok {
		panic("tensor: Matmul: inner dim K must be concrete")
	}

	// Matrix-vector: A[..., M, K] @ B[K] → A[..., M]
	if len(bSints) == 1 {
		bv, ok := bSints[0].ConstValue()
		if !ok || bv != Kv {
			panic(fmt.Sprintf("tensor: Matmul: vector dim mismatch"))
		}
		b := BroadcastToSints(other, aSints)
		prod := t.Mul(b)
		return prod.Sum([]int{len(aSints) - 1}, false)
	}

	// N = last dim of B; K-check
	Ns := bSints[len(bSints)-1]
	Nv, ok := Ns.ConstValue()
	if !ok {
		panic("tensor: Matmul: output dim N must be concrete")
	}
	kDimS := bSints[len(bSints)-2]
	kDimV, ok := kDimS.ConstValue()
	if !ok || kDimV != Kv {
		panic(fmt.Sprintf("tensor: Matmul: inner dim mismatch"))
	}

	// Unsqueeze A: [..., M, K] → [..., M, K, 1]
	aNew := make([]shape.Sint, len(aSints)+1)
	copy(aNew, aSints)
	aNew[len(aSints)] = shape.Const(1)
	a := t.ReshapeSints(aNew)

	// Unsqueeze B: [..., K, N] → [..., 1, K, N]  (B is always concrete)
	bNew := make([]int64, len(bSints)+1)
	for i := 0; i < len(bSints)-2; i++ {
		sv, _ := bSints[i].ConstValue()
		bNew[i] = sv
	}
	bNew[len(bSints)-2] = 1
	bNew[len(bSints)-1] = Kv
	bNew[len(bSints)] = Nv
	b := other.Reshape(bNew)

	// Broadcast (Sint-aware for symbolic batch).
	ab, bb := broadcast(a, b)

	prod := ab.Mul(bb)
	kAxis := prod.Rank() - 2
	return prod.Sum([]int{kAxis}, false)
}

// allAxes returns [0, 1, ..., rank-1].
func allAxes(rank int) []int {
	axes := make([]int, rank)
	for i := range axes {
		axes[i] = i
	}
	return axes
}
