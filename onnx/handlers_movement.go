package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func handleReshape(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 2 {
		return nil, fmt.Errorf("Reshape: expected 2 inputs, got %d", len(ctx.Inputs))
	}
	if !ctx.Inputs[0].IsDevice() {
		return nil, fmt.Errorf("Reshape: data input is not a device tensor")
	}
	x := ctx.Inputs[0].Tensor()
	target, err := resolveShapeInput(ctx.Inputs[1])
	if err != nil {
		return nil, fmt.Errorf("Reshape: %w", err)
	}

	allowzero := ctx.Node.Attrs["allowzero"].Int(0) != 0
	xSh := x.ShapeSints()

	// Normalise -1 / 0 in target.
	// First pass: replace 0 with copy of input dim (when allowzero=0).
	resolved := make([]shape.Sint, len(target))
	negOneAt := -1
	for i, s := range target {
		v, ok := s.ConstValue()
		switch {
		case ok && v == 0 && !allowzero:
			if i >= len(xSh) {
				return nil, fmt.Errorf("Reshape: 0 at axis %d but input rank is %d", i, len(xSh))
			}
			resolved[i] = xSh[i]
		case ok && v == -1:
			if negOneAt != -1 {
				return nil, fmt.Errorf("Reshape: multiple -1 entries")
			}
			negOneAt = i
			resolved[i] = shape.Const(1) // placeholder; replaced below
		default:
			resolved[i] = s
		}
	}

	// If -1 present, infer it from the total volume.
	if negOneAt >= 0 {
		// Total input volume.
		inVol := shape.SymbolicProduct(xSh)
		// Product of the resolved target (excluding -1 placeholder which is 1).
		var others []shape.Sint
		for i, s := range resolved {
			if i == negOneAt {
				continue
			}
			others = append(others, s)
		}
		otherVol := shape.SymbolicProduct(others)
		ivol, iok := inVol.ConstValue()
		ovol, ook := otherVol.ConstValue()
		if !iok || !ook {
			return nil, fmt.Errorf("Reshape: cannot infer -1 from symbolic volume")
		}
		if ovol == 0 {
			return nil, fmt.Errorf("Reshape: cannot infer -1 with zero-volume target")
		}
		if ivol%ovol != 0 {
			return nil, fmt.Errorf("Reshape: input volume %d not divisible by product of other dims %d", ivol, ovol)
		}
		resolved[negOneAt] = shape.Const(ivol / ovol)
	}

	return []Value{Device(x.ReshapeSints(resolved))}, nil
}

func handleFlatten(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Flatten")
	if err != nil {
		return nil, err
	}
	axis := int(ctx.Node.Attrs["axis"].Int(1))
	sh := x.ShapeSints()
	if axis < 0 {
		axis += len(sh)
	}
	if axis < 0 || axis > len(sh) {
		return nil, fmt.Errorf("Flatten: axis %d out of range for rank %d", axis, len(sh))
	}
	// Compute outer product (dims 0..axis-1) and inner (axis..end).
	outer := shape.Const(1)
	for i := 0; i < axis; i++ {
		outer = shape.Mul(outer, sh[i])
	}
	inner := shape.Const(1)
	for i := axis; i < len(sh); i++ {
		inner = shape.Mul(inner, sh[i])
	}
	target := []shape.Sint{outer, inner}
	return []Value{Device(x.ReshapeSints(target))}, nil
}

func handleSqueeze(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Squeeze")
	if err != nil {
		return nil, err
	}
	// axes opset <=12 attr, >=13 input.
	var axes []int
	if len(ctx.Inputs) >= 2 && ctx.Inputs[1].Kind != KindUnset {
		vs, verr := asHostIntVec(ctx.Inputs[1])
		if verr != nil {
			return nil, fmt.Errorf("Squeeze: axes: %w", verr)
		}
		for _, a := range vs {
			axes = append(axes, int(a))
		}
	} else if a, ok := ctx.Node.Attrs["axes"]; ok && a.Kind == AttrInts {
		for _, v := range a.Is {
			axes = append(axes, int(v))
		}
	}

	sh := x.ShapeSints()
	rank := len(sh)
	normAxes := make(map[int]bool, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, fmt.Errorf("Squeeze: axis %d out of range for rank %d", ax, rank)
		}
		normAxes[ax] = true
	}
	// Squeeze either listed axes (if axes specified) or all size-1 dims.
	var newSh []shape.Sint
	for i, s := range sh {
		if len(axes) > 0 {
			if normAxes[i] {
				// must be 1 — assert.
				v, ok := s.ConstValue()
				if !ok || v != 1 {
					return nil, fmt.Errorf("Squeeze: axis %d is not 1", i)
				}
				continue
			}
			newSh = append(newSh, s)
		} else {
			if v, ok := s.ConstValue(); ok && v == 1 {
				continue
			}
			newSh = append(newSh, s)
		}
	}
	return []Value{Device(x.ReshapeSints(newSh))}, nil
}

func handleUnsqueeze(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Unsqueeze")
	if err != nil {
		return nil, err
	}
	// axes opset <=12 attr, >=13 input.
	var axes []int
	if len(ctx.Inputs) >= 2 && ctx.Inputs[1].Kind != KindUnset {
		vs, verr := asHostIntVec(ctx.Inputs[1])
		if verr != nil {
			return nil, fmt.Errorf("Unsqueeze: axes: %w", verr)
		}
		for _, a := range vs {
			axes = append(axes, int(a))
		}
	} else if a, ok := ctx.Node.Attrs["axes"]; ok && a.Kind == AttrInts {
		for _, v := range a.Is {
			axes = append(axes, int(v))
		}
	}
	if len(axes) == 0 {
		return []Value{Device(x)}, nil
	}
	sh := x.ShapeSints()
	outRank := len(sh) + len(axes)
	posSet := make(map[int]bool, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += outRank
		}
		if ax < 0 || ax >= outRank {
			return nil, fmt.Errorf("Unsqueeze: axis %d out of range for output rank %d", ax, outRank)
		}
		posSet[ax] = true
	}
	newSh := make([]shape.Sint, 0, outRank)
	srcI := 0
	for i := 0; i < outRank; i++ {
		if posSet[i] {
			newSh = append(newSh, shape.Const(1))
		} else {
			newSh = append(newSh, sh[srcI])
			srcI++
		}
	}
	return []Value{Device(x.ReshapeSints(newSh))}, nil
}

func handleTranspose(ctx *HandlerCtx) ([]Value, error) {
	x, err := oneTensorInput(ctx, "Transpose")
	if err != nil {
		return nil, err
	}
	rank := x.Rank()
	perm := ctx.Node.Attrs["perm"].Ints(nil)
	var order []int
	if perm == nil {
		// Default: reverse dims.
		order = make([]int, rank)
		for i := range order {
			order[i] = rank - 1 - i
		}
	} else {
		if len(perm) != rank {
			return nil, fmt.Errorf("Transpose: perm length %d != rank %d", len(perm), rank)
		}
		order = make([]int, rank)
		for i, p := range perm {
			order[i] = int(p)
		}
	}
	return []Value{Device(x.Permute(order))}, nil
}

func handleConcat(ctx *HandlerCtx) ([]Value, error) {
	axis := int(ctx.Node.Attrs["axis"].Int(0))
	ts := make([]*tensor.Tensor, len(ctx.Inputs))
	for i, v := range ctx.Inputs {
		if !v.IsDevice() {
			return nil, fmt.Errorf("Concat: input %d is not a device tensor", i)
		}
		ts[i] = v.Tensor()
	}
	return []Value{Device(tensor.Concat(ts, axis))}, nil
}

func handleGather(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 2 {
		return nil, fmt.Errorf("Gather: expected 2 inputs")
	}
	if !ctx.Inputs[0].IsDevice() || !ctx.Inputs[1].IsDevice() {
		return nil, fmt.Errorf("Gather: data and indices must be device tensors at this entry point")
	}
	axis := int(ctx.Node.Attrs["axis"].Int(0))
	return []Value{Device(ctx.Inputs[0].Tensor().Gather(axis, ctx.Inputs[1].Tensor()))}, nil
}

// handleSlice supports opset-10+ where starts/ends/axes/steps are inputs.
// Negative step is rejected (deferred per plan §6).
func handleSlice(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 3 {
		return nil, fmt.Errorf("Slice: expected ≥ 3 inputs (data, starts, ends), got %d", len(ctx.Inputs))
	}
	if !ctx.Inputs[0].IsDevice() {
		return nil, fmt.Errorf("Slice: data input is not a device tensor")
	}
	x := ctx.Inputs[0].Tensor()
	starts, err := asHostIntVec(ctx.Inputs[1])
	if err != nil {
		return nil, fmt.Errorf("Slice: starts: %w", err)
	}
	ends, err := asHostIntVec(ctx.Inputs[2])
	if err != nil {
		return nil, fmt.Errorf("Slice: ends: %w", err)
	}
	if len(starts) != len(ends) {
		return nil, fmt.Errorf("Slice: starts (%d) and ends (%d) length mismatch", len(starts), len(ends))
	}
	var axes []int64
	if len(ctx.Inputs) >= 4 && ctx.Inputs[3].Kind != KindUnset {
		axes, err = asHostIntVec(ctx.Inputs[3])
		if err != nil {
			return nil, fmt.Errorf("Slice: axes: %w", err)
		}
	} else {
		axes = make([]int64, len(starts))
		for i := range axes {
			axes[i] = int64(i)
		}
	}
	var steps []int64
	if len(ctx.Inputs) >= 5 && ctx.Inputs[4].Kind != KindUnset {
		steps, err = asHostIntVec(ctx.Inputs[4])
		if err != nil {
			return nil, fmt.Errorf("Slice: steps: %w", err)
		}
	}

	sh := x.Shape()
	rank := len(sh)
	loHi := make([][2]int64, rank)
	for i := 0; i < rank; i++ {
		loHi[i] = [2]int64{0, sh[i]}
	}
	for i, s := range starts {
		ax := int(axes[i])
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, fmt.Errorf("Slice: axis %d out of range for rank %d", axes[i], rank)
		}
		e := ends[i]
		st := int64(1)
		if i < len(steps) {
			st = steps[i]
		}
		if st < 0 {
			return nil, fmt.Errorf("Slice: negative step not supported in v1 (axis=%d step=%d)", ax, st)
		}
		if st != 1 {
			return nil, fmt.Errorf("Slice: step != 1 not supported in v1 (axis=%d step=%d)", ax, st)
		}
		// Clamp.
		lo := s
		hi := e
		if lo < 0 {
			lo += sh[ax]
		}
		if hi < 0 {
			hi += sh[ax]
		}
		if lo < 0 {
			lo = 0
		}
		if lo > sh[ax] {
			lo = sh[ax]
		}
		if hi < 0 {
			hi = 0
		}
		if hi > sh[ax] {
			hi = sh[ax]
		}
		if hi < lo {
			hi = lo
		}
		loHi[ax] = [2]int64{lo, hi}
	}
	return []Value{Device(x.Shrink(loHi))}, nil
}

func handleExpand(ctx *HandlerCtx) ([]Value, error) {
	if len(ctx.Inputs) < 2 {
		return nil, fmt.Errorf("Expand: expected 2 inputs, got %d", len(ctx.Inputs))
	}
	if !ctx.Inputs[0].IsDevice() {
		return nil, fmt.Errorf("Expand: data input is not a device tensor")
	}
	x := ctx.Inputs[0].Tensor()
	target, err := resolveShapeInput(ctx.Inputs[1])
	if err != nil {
		return nil, fmt.Errorf("Expand: %w", err)
	}
	return []Value{Device(tensor.BroadcastToSints(x, target))}, nil
}

// asHostIntVec coerces an integer Value into []int64. Accepts host-tier
// scalars/vectors and device-tier integer leaf tensors (initializers come in
// as device-side leaves).
func asHostIntVec(v Value) ([]int64, error) {
	switch v.Kind {
	case KindHostInts:
		return append([]int64{}, v.Is...), nil
	case KindHostInt64:
		return []int64{v.I}, nil
	case KindDevice:
		t := v.Tensor()
		if !t.DType().IsInt() {
			return nil, fmt.Errorf("device shape vector has non-integer dtype %v", t.DType())
		}
		d := t.Data()
		if d == nil {
			return nil, fmt.Errorf("device shape vector has no host-side data")
		}
		out := make([]int64, len(d))
		for i, f := range d {
			out[i] = int64(f)
		}
		return out, nil
	}
	return nil, fmt.Errorf("expected host int vector, got kind %d", v.Kind)
}

var _ = uop.Dtypes
