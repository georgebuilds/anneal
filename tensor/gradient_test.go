package tensor_test

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Tiny float32 evaluator ────────────────────────────────────────────────────
//
// Interprets the UOp DAG rooted at `root` into concrete float32 data.
// Only the ops produced by the tensor frontend (and by gradient.go) are handled.
// Shapes and data are computed together; the evaluator never materializes
// intermediate state except as flat float32 slices in row-major (C) order.

type evalT struct {
	data  []float32
	shape []int64
}

func numel(sh []int64) int {
	n := 1
	for _, s := range sh {
		n *= int(s)
	}
	return n
}

// multiIndex converts a flat index to a multi-dimensional index in row-major order.
func multiIndex(flat int, shape []int64) []int {
	idx := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		idx[i] = flat % int(shape[i])
		flat /= int(shape[i])
	}
	return idx
}

// flatIndex converts a multi-dimensional index to flat row-major index.
func flatIndex(idx []int, shape []int64) int {
	flat := 0
	for i := range shape {
		flat = flat*int(shape[i]) + idx[i]
	}
	return flat
}

// evalGraph evaluates the UOp graph numerically.
// bufData maps buffer UOp index → flat float32 data.
// bufShapes maps buffer UOp index → shape (required for OpBuffer nodes).
func evalGraph(root uop.UOp, bufData map[uint32][]float32, bufShapes map[uint32][]int64) evalT {
	// Topological sort (reuse logic from production code).
	seen := make(map[uint32]bool)
	var topo []uop.UOp
	type frame struct {
		u   uop.UOp
		idx int
	}
	stack := []frame{{root, 0}}
	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u
		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}
		pushed := false
		for f.idx < u.NSrc() {
			child := u.Src(f.idx)
			f.idx++
			if !seen[child.Index()] {
				stack = append(stack, frame{child, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			topo = append(topo, u)
			stack = stack[:len(stack)-1]
		}
	}

	shapeCache := make(map[uint32][]int64)
	dataCache := make(map[uint32][]float32)

	for _, u := range topo {
		evalNode(u, shapeCache, dataCache, bufData, bufShapes)
	}

	return evalT{data: dataCache[root.Index()], shape: shapeCache[root.Index()]}
}

func evalNode(u uop.UOp, shapes map[uint32][]int64, data map[uint32][]float32, bufData map[uint32][]float32, bufShapes map[uint32][]int64) {
	idx := u.Index()
	getSh := func(i int) []int64 { return shapes[u.Src(i).Index()] }
	getDat := func(i int) []float32 { return data[u.Src(i).Index()] }

	switch u.Op() {
	case uop.OpConst:
		shapes[idx] = []int64{} // scalar
		switch v := u.Arg().(type) {
		case float64:
			data[idx] = []float32{float32(v)}
		case int64:
			data[idx] = []float32{float32(v)}
		case bool:
			if v {
				data[idx] = []float32{1}
			} else {
				data[idx] = []float32{0}
			}
		}

	case uop.OpBuffer:
		shapes[idx] = bufShapes[idx]
		data[idx] = bufData[idx]

	case uop.OpCast, uop.OpBitcast:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		copy(out, src)
		data[idx] = out

	// Unary elementwise.
	case uop.OpNeg:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = -v
		}
		data[idx] = out

	case uop.OpExp2:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = float32(math.Exp2(float64(v)))
		}
		data[idx] = out

	case uop.OpLog2:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = float32(math.Log2(float64(v)))
		}
		data[idx] = out

	case uop.OpSin:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = float32(math.Sin(float64(v)))
		}
		data[idx] = out

	case uop.OpSqrt:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = float32(math.Sqrt(float64(v)))
		}
		data[idx] = out

	case uop.OpReciprocal:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = 1.0 / v
		}
		data[idx] = out

	case uop.OpTrunc:
		shapes[idx] = getSh(0)
		src := getDat(0)
		out := make([]float32, len(src))
		for i, v := range src {
			out[i] = float32(math.Trunc(float64(v)))
		}
		data[idx] = out

	// Binary elementwise — both srcs must have same shape after broadcasting.
	case uop.OpAdd, uop.OpSub, uop.OpMul, uop.OpMax, uop.OpCmpEq, uop.OpCmpLt, uop.OpFDiv:
		shapes[idx] = getSh(0)
		s0, s1 := getDat(0), getDat(1)
		out := make([]float32, len(s0))
		switch u.Op() {
		case uop.OpAdd:
			for i := range s0 {
				out[i] = s0[i] + s1[i]
			}
		case uop.OpSub:
			for i := range s0 {
				out[i] = s0[i] - s1[i]
			}
		case uop.OpMul:
			for i := range s0 {
				out[i] = s0[i] * s1[i]
			}
		case uop.OpMax:
			for i := range s0 {
				if s0[i] > s1[i] {
					out[i] = s0[i]
				} else {
					out[i] = s1[i]
				}
			}
		case uop.OpFDiv:
			for i := range s0 {
				out[i] = s0[i] / s1[i]
			}
		case uop.OpCmpEq:
			for i := range s0 {
				if s0[i] == s1[i] {
					out[i] = 1
				} else {
					out[i] = 0
				}
			}
		case uop.OpCmpLt:
			for i := range s0 {
				if s0[i] < s1[i] {
					out[i] = 1
				} else {
					out[i] = 0
				}
			}
		}
		data[idx] = out

	case uop.OpWhere:
		shapes[idx] = getSh(1)
		cond, xd, yd := getDat(0), getDat(1), getDat(2)
		out := make([]float32, len(xd))
		for i := range xd {
			if cond[i] != 0 {
				out[i] = xd[i]
			} else {
				out[i] = yd[i]
			}
		}
		data[idx] = out

	case uop.OpMulAcc:
		shapes[idx] = getSh(0)
		s0, s1, s2 := getDat(0), getDat(1), getDat(2)
		out := make([]float32, len(s0))
		for i := range s0 {
			out[i] = s0[i]*s1[i] + s2[i]
		}
		data[idx] = out

	// Movement ops.
	case uop.OpReshape:
		newSh := u.Arg().([]int64)
		shapes[idx] = newSh
		src := getDat(0)
		out := make([]float32, len(src))
		copy(out, src)
		data[idx] = out

	case uop.OpExpand:
		dstSh := u.Arg().([]int64)
		srcSh := getSh(0)
		srcData := getDat(0)
		shapes[idx] = dstSh
		total := numel(dstSh)
		out := make([]float32, total)
		rank := len(dstSh)
		for i := 0; i < total; i++ {
			mi := multiIndex(i, dstSh)
			// Map each broadcast dimension (where srcSh[k]==1) to index 0.
			srcMi := make([]int, rank)
			for k := 0; k < rank; k++ {
				if srcSh[k] == 1 {
					srcMi[k] = 0
				} else {
					srcMi[k] = mi[k]
				}
			}
			out[i] = srcData[flatIndex(srcMi, srcSh)]
		}
		data[idx] = out

	case uop.OpPermute:
		permArg := u.Arg().([]int64)
		srcSh := getSh(0)
		srcData := getDat(0)
		rank := len(permArg)
		dstSh := make([]int64, rank)
		for k, p := range permArg {
			dstSh[k] = srcSh[p]
		}
		shapes[idx] = dstSh
		total := numel(dstSh)
		out := make([]float32, total)
		for i := 0; i < total; i++ {
			dstMi := multiIndex(i, dstSh)
			srcMi := make([]int, rank)
			for k, p := range permArg {
				srcMi[p] = dstMi[k]
			}
			out[i] = srcData[flatIndex(srcMi, srcSh)]
		}
		data[idx] = out

	case uop.OpPad:
		padding := u.Arg().([][2]int64)
		srcSh := getSh(0)
		srcData := getDat(0)
		rank := len(srcSh)
		dstSh := make([]int64, rank)
		for k, s := range srcSh {
			dstSh[k] = s + padding[k][0] + padding[k][1]
		}
		shapes[idx] = dstSh
		total := numel(dstSh)
		out := make([]float32, total) // zero-initialized
		for i := 0; i < numel(srcSh); i++ {
			srcMi := multiIndex(i, srcSh)
			dstMi := make([]int, rank)
			for k := range srcMi {
				dstMi[k] = srcMi[k] + int(padding[k][0])
			}
			out[flatIndex(dstMi, dstSh)] = srcData[i]
		}
		data[idx] = out

	case uop.OpShrink:
		shrinkArg := u.Arg().([][2]int64)
		srcSh := getSh(0)
		srcData := getDat(0)
		rank := len(shrinkArg)
		dstSh := make([]int64, rank)
		for k, p := range shrinkArg {
			dstSh[k] = p[1] - p[0]
		}
		shapes[idx] = dstSh
		total := numel(dstSh)
		out := make([]float32, total)
		for i := 0; i < total; i++ {
			dstMi := multiIndex(i, dstSh)
			srcMi := make([]int, rank)
			for k := range dstMi {
				srcMi[k] = dstMi[k] + int(shrinkArg[k][0])
			}
			out[i] = srcData[flatIndex(srcMi, srcSh)]
		}
		data[idx] = out

	case uop.OpFlip:
		flipArg := u.Arg().([]int64)
		srcSh := getSh(0)
		srcData := getDat(0)
		shapes[idx] = srcSh
		total := numel(srcSh)
		rank := len(srcSh)
		out := make([]float32, total)
		for i := 0; i < total; i++ {
			mi := multiIndex(i, srcSh)
			srcMi := make([]int, rank)
			for k := range mi {
				if flipArg[k] != 0 {
					srcMi[k] = int(srcSh[k]) - 1 - mi[k]
				} else {
					srcMi[k] = mi[k]
				}
			}
			out[i] = srcData[flatIndex(srcMi, srcSh)]
		}
		data[idx] = out

	case uop.OpReduceAxis:
		ra := u.Arg().(uop.ReduceArg)
		srcSh := getSh(0)
		srcData := getDat(0)
		axSet := make(map[int]bool, len(ra.Axes))
		for _, ax := range ra.Axes {
			axSet[ax] = true
		}
		var dstSh []int64
		for k, s := range srcSh {
			if !axSet[k] {
				dstSh = append(dstSh, s)
			}
		}
		if dstSh == nil {
			dstSh = []int64{}
		}
		shapes[idx] = dstSh
		dstN := numel(dstSh)
		if dstN == 0 {
			dstN = 1
		}
		rank := len(srcSh)

		// Build dstSh with all dims (including reduced as 1) for index mapping.
		fullDst := make([]int64, rank)
		for k, s := range srcSh {
			if axSet[k] {
				fullDst[k] = 1
			} else {
				fullDst[k] = s
			}
		}

		// Initialize output (for sum: 0; for max: -inf).
		out := make([]float32, dstN)
		switch ra.Op {
		case uop.OpAdd:
			// already zero
		case uop.OpMax:
			for i := range out {
				out[i] = float32(math.Inf(-1))
			}
		}

		// Enumerate all src elements, accumulate into correct dst position.
		for i := 0; i < numel(srcSh); i++ {
			srcMi := multiIndex(i, srcSh)
			// Compute dst flat index (project out reduced dims).
			dstMi := make([]int, len(dstSh))
			di := 0
			for k := range srcSh {
				if !axSet[k] {
					dstMi[di] = srcMi[k]
					di++
				}
			}
			dFlat := 0
			if len(dstSh) > 0 {
				dFlat = flatIndex(dstMi, dstSh)
			}
			switch ra.Op {
			case uop.OpAdd:
				out[dFlat] += srcData[i]
			case uop.OpMax:
				if srcData[i] > out[dFlat] {
					out[dFlat] = srcData[i]
				}
			}
		}
		data[idx] = out
	}
}

// ── Finite-difference helper ──────────────────────────────────────────────────

// finiteDiffGrad computes the numerical gradient of the scalar loss
// (= sum of all elements of the graph rooted at lossNode) with respect to
// the parameter stored in paramNode, using central differences.
func finiteDiffGrad(
	t *testing.T,
	lossNode uop.UOp,
	paramIdx uint32,
	paramShape []int64,
	allBufData map[uint32][]float32,
	allBufShapes map[uint32][]int64,
	eps float32,
) []float32 {
	t.Helper()
	n := numel(paramShape)
	grad := make([]float32, n)
	paramData := allBufData[paramIdx]

	for i := 0; i < n; i++ {
		orig := paramData[i]

		paramData[i] = orig + eps
		resPlus := evalGraph(lossNode, allBufData, allBufShapes)
		plus := float32(0)
		for _, v := range resPlus.data {
			plus += v
		}

		paramData[i] = orig - eps
		resMinus := evalGraph(lossNode, allBufData, allBufShapes)
		minus := float32(0)
		for _, v := range resMinus.data {
			minus += v
		}

		paramData[i] = orig
		grad[i] = (plus - minus) / (2 * eps)
	}
	return grad
}

// checkGrads computes analytic vs numerical gradients and asserts they match within tol.
func checkGrads(t *testing.T, loss *tensor.Tensor, param *tensor.Tensor, paramData []float32, paramShape []int64, allBufData map[uint32][]float32, allBufShapes map[uint32][]int64, eps, tol float32) {
	t.Helper()

	// Analytic gradient via Backward.
	grads := tensor.Backward(loss, []*tensor.Tensor{param})
	gTensor, ok := grads[param]
	if !ok {
		t.Fatalf("Backward did not produce gradient for param")
	}

	// Evaluate analytic gradient.
	analytic := evalGraph(gTensor.Node(), allBufData, allBufShapes)

	// Numerical gradient.
	numerical := finiteDiffGrad(t, loss.Node(), param.Node().Index(), paramShape, allBufData, allBufShapes, eps)

	if len(analytic.data) != len(numerical) {
		t.Fatalf("gradient size mismatch: analytic=%d numerical=%d", len(analytic.data), len(numerical))
	}

	maxErr := float32(0)
	for i := range numerical {
		diff := analytic.data[i] - numerical[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxErr {
			maxErr = diff
		}
	}
	if maxErr > tol {
		t.Fatalf("gradient mismatch (max err=%.6f > tol=%.6f)\n  analytic: %v\n  numerical: %v",
			maxErr, tol, analytic.data, numerical)
	}
}

// ── Helper: allocate uniform random-ish data avoiding singularities ───────────

// testData returns deterministic float32 data for a given shape,
// avoiding zeros and values near singularities of log/sqrt/recip.
func testData(shape []int64, offset float32) []float32 {
	n := numel(shape)
	data := make([]float32, n)
	for i := range data {
		// Values in [0.5, 2.5] to keep ops well-defined.
		data[i] = 0.5 + float32(i)*0.3 + offset
		if data[i] > 2.5 {
			data[i] = 0.5 + float32(i%4)*0.3
		}
	}
	return data
}

// ── Primitive unary gradient tests ───────────────────────────────────────────

func TestGradNeg(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Neg().Sum(nil, false)

	xData := testData([]int64{3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-4)
}

func TestGradExp2(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Exp2().Sum(nil, false)

	xData := testData([]int64{3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

func TestGradLog2(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Log2().Sum(nil, false)

	xData := testData([]int64{3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

func TestGradSin(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	loss := x.Sin().Sum(nil, false)

	xData := testData([]int64{4}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {4}}

	checkGrads(t, loss, x, xData, []int64{4}, bufs, bSh, 1e-3, 1e-3)
}

func TestGradSqrt(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Sqrt().Sum(nil, false)

	xData := testData([]int64{3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

func TestGradRecip(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Recip().Sum(nil, false)

	xData := testData([]int64{3}, 1.0) // keep away from 0
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

// ── Exp/Log composition (no direct rule — chain rule through Exp2/Log2/Mul) ──

// TestGradExpComposition verifies that Exp (= exp2(x/ln2)) differentiates
// correctly via chain rule through the primitive rules, with no custom rule.
func TestGradExpComposition(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Exp().Sum(nil, false) // Exp2(x/ln2)

	xData := testData([]int64{3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

// TestGradLogComposition verifies that Log (= Log2(x)*ln2) differentiates
// correctly via chain rule, with no custom rule.
func TestGradLogComposition(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Log().Sum(nil, false) // Log2(x)*ln2

	xData := testData([]int64{3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

// ── Binary ops ────────────────────────────────────────────────────────────────

func TestGradAdd(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Add(y).Sum(nil, false)

	xData := testData([]int64{3}, 0)
	yData := testData([]int64{3}, 1)
	bufs := map[uint32][]float32{x.Node().Index(): xData, y.Node().Index(): yData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}, y.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-4)
	checkGrads(t, loss, y, yData, []int64{3}, bufs, bSh, 1e-3, 1e-4)
}

func TestGradMul(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	loss := x.Mul(y).Sum(nil, false)

	xData := testData([]int64{3}, 0)
	yData := testData([]int64{3}, 1)
	bufs := map[uint32][]float32{x.Node().Index(): xData, y.Node().Index(): yData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}, y.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
	checkGrads(t, loss, y, yData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

// ── Diamond: node used by two consumers (tests adjoint summation) ─────────────

func TestGradDiamond(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	// loss = x * x + x  (x is shared by two consumers — adjoint must be summed)
	loss := x.Mul(x).Add(x).Sum(nil, false)

	xData := testData([]int64{3}, 0.5)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-3)
}

// ── Deep diamond: single param via short path AND 50-deep chain ──────────────

func TestGradDeepDiamond(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")

	// Build a 50-deep chain of Sin ops rooted at x.
	chainT := x
	const depth = 50
	for i := 0; i < depth; i++ {
		chainT = chainT.Sin()
	}

	// loss = sum(x + chain50)
	// x contributes via depth-1 (direct Add input) AND depth-52 (through chain).
	loss := x.Add(chainT).Sum(nil, false)

	xData := []float32{0.1, 0.2, 0.3}
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {3}}

	checkGrads(t, loss, x, xData, []int64{3}, bufs, bSh, 1e-3, 1e-2)
}

// ── Broadcasting gradient (Expand backward → sum reduction) ──────────────────

func TestGradBroadcast(t *testing.T) {
	a := newArena()
	// w has shape [1,3], x has shape [2,3]; Add broadcasts w to [2,3].
	// dL/dw must sum the adjoint over the broadcast axis 0.
	w := tensor.NewLeaf(a, []int64{1, 3}, uop.Dtypes.Float32, "cpu")
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	loss := w.Add(x).Sum(nil, false)

	wData := testData([]int64{1, 3}, 0)
	xData := testData([]int64{2, 3}, 1)
	bufs := map[uint32][]float32{w.Node().Index(): wData, x.Node().Index(): xData}
	bSh := map[uint32][]int64{w.Node().Index(): {1, 3}, x.Node().Index(): {2, 3}}

	checkGrads(t, loss, w, wData, []int64{1, 3}, bufs, bSh, 1e-3, 1e-3)
}

// ── Matmul gradient (no custom rule; falls out of primitives) ─────────────────

func TestGradMatmul(t *testing.T) {
	a := newArena()
	// [2,3] @ [3,4] → [2,4]
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	w := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	loss := x.Matmul(w).Sum(nil, false)

	xData := testData([]int64{2, 3}, 0)
	wData := testData([]int64{3, 4}, 0.2)
	bufs := map[uint32][]float32{x.Node().Index(): xData, w.Node().Index(): wData}
	bSh := map[uint32][]int64{x.Node().Index(): {2, 3}, w.Node().Index(): {3, 4}}

	checkGrads(t, loss, x, xData, []int64{2, 3}, bufs, bSh, 1e-2, 5e-3)
	checkGrads(t, loss, w, wData, []int64{3, 4}, bufs, bSh, 1e-2, 5e-3)
}

// ── Linear layer gradient ─────────────────────────────────────────────────────

func TestGradLinear(t *testing.T) {
	a := newArena()
	// y = x @ W.T + b, loss = sum(y)
	x := tensor.NewLeaf(a, []int64{2, 4}, uop.Dtypes.Float32, "cpu")
	W := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu") // [out, in]
	b := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")

	y := x.Matmul(W.Permute([]int{1, 0}))
	bExp := tensor.BroadcastTo(b, y.Shape())
	loss := y.Add(bExp).Sum(nil, false)

	xData := testData([]int64{2, 4}, 0)
	wData := testData([]int64{3, 4}, 0.1)
	bData := testData([]int64{3}, 0.5)

	bufs := map[uint32][]float32{
		x.Node().Index(): xData,
		W.Node().Index(): wData,
		b.Node().Index(): bData,
	}
	bSh := map[uint32][]int64{
		x.Node().Index(): {2, 4},
		W.Node().Index(): {3, 4},
		b.Node().Index(): {3},
	}

	checkGrads(t, loss, W, wData, []int64{3, 4}, bufs, bSh, 1e-2, 5e-3)
	checkGrads(t, loss, b, bData, []int64{3}, bufs, bSh, 1e-3, 3e-3)
}

// ── 2-layer MLP gradient ──────────────────────────────────────────────────────

func TestGradMLP(t *testing.T) {
	a := newArena()

	// Tiny MLP: [2,4] → [4,3] → ReLU → [3,2] → sum
	x := tensor.NewLeaf(a, []int64{2, 4}, uop.Dtypes.Float32, "cpu")
	W1 := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	W2 := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")

	h1 := x.Matmul(W1.Permute([]int{1, 0}))
	zeros := tensor.Zeros(a, h1.Shape(), uop.Dtypes.Float32, "cpu")
	h1relu := h1.Maximum(zeros) // ReLU
	out := h1relu.Matmul(W2.Permute([]int{1, 0}))
	loss := out.Sum(nil, false)

	xData := testData([]int64{2, 4}, 0)
	w1Data := testData([]int64{3, 4}, 0.1)
	w2Data := testData([]int64{2, 3}, 0.2)

	bufs := map[uint32][]float32{
		x.Node().Index():  xData,
		W1.Node().Index(): w1Data,
		W2.Node().Index(): w2Data,
	}
	bSh := map[uint32][]int64{
		x.Node().Index():  {2, 4},
		W1.Node().Index(): {3, 4},
		W2.Node().Index(): {2, 3},
	}

	checkGrads(t, loss, W1, w1Data, []int64{3, 4}, bufs, bSh, 1e-2, 5e-3)
	checkGrads(t, loss, W2, w2Data, []int64{2, 3}, bufs, bSh, 1e-2, 5e-3)
}

// ── Conv2d gradient (no custom rule; falls out of shrink/matmul primitives) ───

func TestGradConv2d(t *testing.T) {
	a := newArena()

	// Very small: N=1, Cin=1, H=4, W=4, Cout=2, kernel=2x2, no pad, stride=1.
	N, Cin, H, W := int64(1), int64(1), int64(4), int64(4)
	Cout, kH, kW := int64(2), int64(2), int64(2)
	Ho := H - kH + 1 // 3
	Wo := W - kW + 1 // 3

	x := tensor.NewLeaf(a, []int64{N, Cin, H, W}, uop.Dtypes.Float32, "cpu")
	weight := tensor.NewLeaf(a, []int64{Cout, Cin, kH, kW}, uop.Dtypes.Float32, "cpu")

	// Manually replicate Conv2d forward for a small kernel.
	var out *tensor.Tensor
	for ki := int64(0); ki < kH; ki++ {
		for kj := int64(0); kj < kW; kj++ {
			patch := x.Shrink([][2]int64{{0, N}, {0, Cin}, {ki, ki + Ho}, {kj, kj + Wo}})
			pFlat := patch.Reshape([]int64{N, Cin, Ho * Wo})
			pMat := pFlat.Permute([]int{0, 2, 1})
			wSlice := weight.Shrink([][2]int64{
				{0, Cout}, {0, Cin}, {ki, ki + 1}, {kj, kj + 1},
			}).Reshape([]int64{Cout, Cin}).Permute([]int{1, 0})
			contrib := pMat.Matmul(wSlice).Permute([]int{0, 2, 1}).Reshape([]int64{N, Cout, Ho, Wo})
			if out == nil {
				out = contrib
			} else {
				out = out.Add(contrib)
			}
		}
	}
	loss := out.Sum(nil, false)

	xData := testData([]int64{N, Cin, H, W}, 0)
	wData := testData([]int64{Cout, Cin, kH, kW}, 0.1)

	bufs := map[uint32][]float32{
		x.Node().Index():      xData,
		weight.Node().Index(): wData,
	}
	bSh := map[uint32][]int64{
		x.Node().Index():      {N, Cin, H, W},
		weight.Node().Index(): {Cout, Cin, kH, kW},
	}

	checkGrads(t, loss, weight, wData, []int64{Cout, Cin, kH, kW}, bufs, bSh, 1e-2, 5e-3)
}

// ── ReduceAxis gradient ───────────────────────────────────────────────────────

func TestGradReduceSum(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	// Sum over axis 1: [2,3] → [2], then sum to scalar.
	loss := x.Sum([]int{1}, false).Sum(nil, false)

	xData := testData([]int64{2, 3}, 0)
	bufs := map[uint32][]float32{x.Node().Index(): xData}
	bSh := map[uint32][]int64{x.Node().Index(): {2, 3}}

	checkGrads(t, loss, x, xData, []int64{2, 3}, bufs, bSh, 1e-3, 5e-4)
}

// ── Where gradient ────────────────────────────────────────────────────────────

func TestGradWhere(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	// Use a constant condition so the graph is differentiable w.r.t. x and y.
	// cond selects x for indices 0,2 and y for 1,3.
	condData := []float32{1, 0, 1, 0}
	cond := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	loss := tensor.Where(cond, x, y).Sum(nil, false)

	xData := testData([]int64{4}, 0)
	yData := testData([]int64{4}, 1)
	bufs := map[uint32][]float32{
		x.Node().Index():    xData,
		y.Node().Index():    yData,
		cond.Node().Index(): condData,
	}
	bSh := map[uint32][]int64{
		x.Node().Index():    {4},
		y.Node().Index():    {4},
		cond.Node().Index(): {4},
	}

	checkGrads(t, loss, x, xData, []int64{4}, bufs, bSh, 1e-3, 5e-4)
	checkGrads(t, loss, y, yData, []int64{4}, bufs, bSh, 1e-3, 5e-4)
}
