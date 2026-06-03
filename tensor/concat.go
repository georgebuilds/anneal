package tensor

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
)

// Concat joins tensors along axis. All inputs must share rank, device, and
// dtype-after-promotion, and must match on every non-concat dim. Negative axis
// wraps in the usual Python style.
//
// Implementation note: OpCat is reserved but not implemented in the rewrite
// engine; we compose Concat from Pad+Add. For A[..., d_a, ...] and
// B[..., d_b, ...] along axis k, pad A on the right of axis k by d_b and pad B
// on the left of axis k by d_a (zeros elsewhere), then elementwise-add. Integer
// dtypes use the same identity (0 is additive identity). For symbolic dims
// (e.g. dynamic batch), pad amounts are built via shape.Sint arithmetic.
func Concat(tensors []*Tensor, axis int) *Tensor {
	if len(tensors) == 0 {
		panic("tensor: Concat: at least one tensor is required")
	}
	if len(tensors) == 1 {
		return tensors[0]
	}

	rank := tensors[0].Rank()
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		panic(fmt.Sprintf("tensor: Concat: axis %d out of range for rank %d", axis, rank))
	}

	// Validate rank / device / non-concat-dim equality.
	dev := tensors[0].Device()
	refShape := tensors[0].ShapeSints()
	for i, t := range tensors {
		if t.Rank() != rank {
			panic(fmt.Sprintf("tensor: Concat: input %d has rank %d, want %d", i, t.Rank(), rank))
		}
		if t.Device() != dev {
			panic(fmt.Sprintf("tensor: Concat: input %d device %q != %q", i, t.Device(), dev))
		}
		ts := t.ShapeSints()
		for d := 0; d < rank; d++ {
			if d == axis {
				continue
			}
			if !shape.SintEqual(refShape[d], ts[d]) {
				panic(fmt.Sprintf("tensor: Concat: input %d dim %d mismatch", i, d))
			}
		}
	}

	// Collect concat-axis sizes for each input.
	axisDims := make([]shape.Sint, len(tensors))
	for i, t := range tensors {
		axisDims[i] = t.ShapeSints()[axis]
	}

	// Symbolic concat along a symbolic axis is a v1 limitation: Pad's internal
	// shape formula yields structurally distinct Sint expressions for the two
	// sides which do not re-intern equal, breaking the subsequent broadcast-Add.
	// Detect and reject this case loudly rather than emit silently-wrong graphs.
	// Concat along a *concrete* axis (the common case in CNN imports — Concat
	// of feature-channel slices, classifier-tail shape glue) is fully supported.
	if _, axisConcrete := axisDims[0].ConstValue(); !axisConcrete {
		panic(fmt.Sprintf("tensor: Concat: concat along symbolic axis %d not supported in v1 (Pad+Add shape canonicalisation breaks on Sub(Add, Neg) vs Add(_, _) asymmetry)", axis))
	}
	for i := 1; i < len(axisDims); i++ {
		if _, ok := axisDims[i].ConstValue(); !ok {
			panic(fmt.Sprintf("tensor: Concat: concat along symbolic axis %d not supported in v1", axis))
		}
	}

	// Build padded contributions and sum them.
	// pre[i] = sum of axisDims[0..i-1]; post[i] = sum of axisDims[i+1..n-1].
	var result *Tensor
	pre := shape.Const(0)
	for i, t := range tensors {
		// post = sum of axisDims[i+1..]
		post := shape.Const(0)
		for j := i + 1; j < len(tensors); j++ {
			post = shape.Add(post, axisDims[j])
		}
		// Build pad mask: 0s except for (pre, post) at axis.
		padMask := make([][2]shape.Sint, rank)
		for d := 0; d < rank; d++ {
			if d == axis {
				padMask[d] = [2]shape.Sint{pre, post}
			} else {
				padMask[d] = [2]shape.Sint{shape.Const(0), shape.Const(0)}
			}
		}
		padded := t.PadSints(padMask)
		if result == nil {
			result = padded
		} else {
			result = result.Add(padded)
		}
		pre = shape.Add(pre, axisDims[i])
	}
	return result
}
