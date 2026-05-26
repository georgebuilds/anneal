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
	sh := t.Shape()
	rank := len(sh)

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

	// Compute output shape.
	outShape := make([]int64, 0, rank)
	for i, s := range sh {
		isReduced := seen[i]
		if !isReduced {
			outShape = append(outShape, s)
		} else if keepdim {
			outShape = append(outShape, 1)
		}
	}

	arg := uop.ReduceArg{Op: op, Axes: normAxes}
	node := t.arena().New(uop.OpReduceAxis, t.dtype, []uop.UOp{t.node}, arg, nil)
	return fromNode(node, shape.NewShapeTracker(outShape), t.dtype, t.device)
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
	sh := t.Shape()
	n := int64(1)
	for _, ax := range axes {
		a := ax
		if a < 0 {
			a += len(sh)
		}
		n *= sh[a]
	}
	summed := t.Sum(axes, keepdim)
	dtype := uop.LeastUpperDType(t.dtype, uop.Dtypes.Float32)
	divisor := Full(t.arena(), summed.Shape(), float64(n), dtype, t.device)
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
// This exposes the full graph to autodiff and the scheduler without a matmul
// primitive
func (t *Tensor) Matmul(other *Tensor) *Tensor {
	aShape := t.Shape()
	bShape := other.Shape()

	if len(aShape) == 0 || len(bShape) == 0 {
		panic("tensor: Matmul: operands must be at least 1D")
	}

	K := aShape[len(aShape)-1]

	// Matrix-vector: A[..., M, K] @ B[K] → A[..., M]
	if len(bShape) == 1 {
		if bShape[0] != K {
			panic(fmt.Sprintf("tensor: Matmul: inner dim mismatch %d != %d", K, bShape[0]))
		}
		b := BroadcastTo(other, aShape)
		prod := t.Mul(b)
		return prod.Sum([]int{len(aShape) - 1}, false)
	}

	kDim := bShape[len(bShape)-2]
	if K != kDim {
		panic(fmt.Sprintf("tensor: Matmul: inner dim mismatch %d != %d", K, kDim))
	}
	N := bShape[len(bShape)-1]

	// Unsqueeze A: [..., M, K] → [..., M, K, 1]
	aNew := make([]int64, len(aShape)+1)
	copy(aNew, aShape)
	aNew[len(aShape)] = 1
	a := t.Reshape(aNew)

	// Unsqueeze B: [..., K, N] → [..., 1, K, N]
	bNew := make([]int64, len(bShape)+1)
	copy(bNew[:len(bShape)-2], bShape[:len(bShape)-2])
	bNew[len(bShape)-2] = 1
	bNew[len(bShape)-1] = K
	bNew[len(bShape)] = N
	b := other.Reshape(bNew)

	// Broadcast batch dims together.
	ab, bb := broadcast(a, b)

	// Element-wise multiply, then sum over K (second-to-last axis).
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
