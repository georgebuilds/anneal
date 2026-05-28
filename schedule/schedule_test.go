package schedule_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── Kernel-level interpreter ──────────────────────────────────────────────────
//
// evalKernelGraph interprets the post-schedule kernel IR (AFTER nodes with
// RANGE/INDEX/REDUCE/WHERE bodies) to produce concrete float32 results.
// This is the value oracle for reduce-over-padded tests.

// kernelEval holds per-evaluation state: current RANGE variable bindings and
// the flat float32 data for every BUFFER in scope.
type kernelEval struct {
	rangeVal map[uint32]int64   // range node index → current iteration value
	bufData  map[uint32][]float32
	bufShape map[uint32][]int64
}

func newKernelEval(bufData map[uint32][]float32, bufShape map[uint32][]int64) *kernelEval {
	return &kernelEval{
		rangeVal: make(map[uint32]int64),
		bufData:  bufData,
		bufShape: bufShape,
	}
}

func (ke *kernelEval) eval(u uop.UOp) float32 {
	switch u.Op() {
	case uop.OpConst:
		switch v := u.Arg().(type) {
		case float64:
			return float32(v)
		case int64:
			return float32(v)
		case bool:
			if v {
				return 1
			}
			return 0
		}
		panic(fmt.Sprintf("kernelEval: unsupported Const arg %T", u.Arg()))

	case uop.OpRange:
		return float32(ke.rangeVal[u.Index()])

	case uop.OpIndex:
		// src[0] = buffer; src[1:] = per-dim index expressions.
		buf := u.Src(0)
		sh := ke.bufShape[buf.Index()]
		dat := ke.bufData[buf.Index()]
		flat := 0
		for d := 0; d < len(sh); d++ {
			i := int(ke.eval(u.Src(d + 1)))
			flat = flat*int(sh[d]) + i
		}
		return dat[flat]

	case uop.OpWhere:
		if ke.eval(u.Src(0)) != 0 {
			return ke.eval(u.Src(1))
		}
		return ke.eval(u.Src(2))

	case uop.OpReduce:
		reduceOp := u.Arg().(uop.Op)
		elem := u.Src(0)
		var acc float32
		switch reduceOp {
		case uop.OpAdd:
			acc = 0
		case uop.OpMax:
			acc = float32(math.Inf(-1))
		case uop.OpMul:
			acc = 1
		default:
			panic(fmt.Sprintf("kernelEval: unhandled reduce op %s", reduceOp))
		}
		// Collect AxisReduce ranges (src[1:]).
		redRanges := make([]uop.UOp, u.NSrc()-1)
		for i := 1; i < u.NSrc(); i++ {
			redRanges[i-1] = u.Src(i)
		}
		ke.enumRanges(redRanges, 0, func() {
			v := ke.eval(elem)
			switch reduceOp {
			case uop.OpAdd:
				acc += v
			case uop.OpMax:
				if v > acc {
					acc = v
				}
			case uop.OpMul:
				acc *= v
			}
		})
		return acc

	// Boolean / comparison ops used by validity guards.
	case uop.OpCmpLt:
		if ke.eval(u.Src(0)) < ke.eval(u.Src(1)) {
			return 1
		}
		return 0
	case uop.OpAnd:
		if ke.eval(u.Src(0)) != 0 && ke.eval(u.Src(1)) != 0 {
			return 1
		}
		return 0

	// Arithmetic ops that appear in index expressions.
	case uop.OpAdd:
		return ke.eval(u.Src(0)) + ke.eval(u.Src(1))
	case uop.OpSub:
		return ke.eval(u.Src(0)) - ke.eval(u.Src(1))
	case uop.OpMul:
		return ke.eval(u.Src(0)) * ke.eval(u.Src(1))
	case uop.OpMax:
		a, b := ke.eval(u.Src(0)), ke.eval(u.Src(1))
		if a > b {
			return a
		}
		return b
	case uop.OpNeg:
		return -ke.eval(u.Src(0))
	case uop.OpExp2:
		return float32(math.Exp2(float64(ke.eval(u.Src(0)))))
	}
	panic(fmt.Sprintf("kernelEval: unhandled op %s", u.Op()))
}

// enumRanges enumerates all combinations of (RANGE node index, size) pairs in
// forward order and calls fn for each combination.
func (ke *kernelEval) enumRanges(ranges []uop.UOp, i int, fn func()) {
	if i == len(ranges) {
		fn()
		return
	}
	r := ranges[i]
	ra := r.Arg().(uop.RangeArg)
	for v := int64(0); v < ra.Size; v++ {
		ke.rangeVal[r.Index()] = v
		ke.enumRanges(ranges, i+1, fn)
	}
}

// evalKernel executes a single AFTER node and returns (data, shape).
// bufData/bufShape must contain entries for every input BUFFER the kernel reads.
func evalKernel(after uop.UOp, bufData map[uint32][]float32, bufShape map[uint32][]int64) ([]float32, []int64) {
	end := after.Src(1)
	store := end.Src(0)
	body := store.Src(1)

	// Collect AxisLoop ranges from END.src[1:] for the output iteration space.
	var outRanges []uop.UOp
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			if ra, ok := r.Arg().(uop.RangeArg); ok && ra.Type == uop.AxisLoop {
				outRanges = append(outRanges, r)
			}
		}
	}
	outShape := make([]int64, len(outRanges))
	for i, r := range outRanges {
		outShape[i] = r.Arg().(uop.RangeArg).Size
	}
	outN := 1
	for _, s := range outShape {
		outN *= int(s)
	}

	ke := newKernelEval(bufData, bufShape)
	out := make([]float32, outN)

	var enumOut func(dim int, flatOut int)
	enumOut = func(dim int, flatOut int) {
		if dim == len(outRanges) {
			out[flatOut] = ke.eval(body)
			return
		}
		r := outRanges[dim]
		ra := r.Arg().(uop.RangeArg)
		for v := int64(0); v < ra.Size; v++ {
			ke.rangeVal[r.Index()] = v
			enumOut(dim+1, flatOut*int(ra.Size)+int(v))
		}
	}
	enumOut(0, 0)
	return out, outShape
}

// evalFirstKernel finds the first AFTER node (by arena index) whose output
// buffer has an LUnique arg matching the AFTER node — i.e. the single kernel
// produced by a simple single-output graph — and evaluates it.
func evalFirstKernel(root uop.UOp, bufData map[uint32][]float32, bufShape map[uint32][]int64) ([]float32, []int64) {
	a := root.Arena()
	for i := 0; i < a.Len(); i++ {
		u := a.At(uint32(i))
		if u.Op() == uop.OpAfter && u.NSrc() == 2 && u.Src(1).Op() == uop.OpEnd {
			return evalKernel(u, bufData, bufShape)
		}
	}
	panic("evalFirstKernel: no AFTER node found")
}

func newArena() *uop.Arena { return uop.NewArena(1024) }

// kernelCount walks the graph rooted at root and counts scheduler-created BUFFER
// nodes (identified by having OpLUnique as src[0]).
func kernelCount(root uop.UOp) int {
	seen := make(map[uint32]bool)
	count := 0
	var walk func(u uop.UOp)
	walk = func(u uop.UOp) {
		if !u.Valid() || seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		if u.Op() == uop.OpBuffer && u.NSrc() >= 1 && u.Src(0).Op() == uop.OpLUnique {
			count++
		}
		for i := 0; i < u.NSrc(); i++ {
			walk(u.Src(i))
		}
	}
	walk(root)
	return count
}

// makeSink builds a SINK wrapping the given tensor nodes.
func makeSink(a *uop.Arena, ts ...*tensor.Tensor) uop.UOp {
	srcs := make([]uop.UOp, len(ts))
	for i, t := range ts {
		srcs[i] = t.Node()
	}
	return a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
}

// ── Loop-nest verifier ────────────────────────────────────────────────────────

// verifyKernelGraph finds every AFTER node in the arena and verifies the
// loop-nest invariants for each kernel.
func verifyKernelGraph(t *testing.T, root uop.UOp) {
	t.Helper()
	a := root.Arena()
	for i := 0; i < a.Len(); i++ {
		u := a.At(uint32(i))
		if u.Op() == uop.OpAfter {
			verifyKernel(t, u)
		}
	}
}

// verifyKernel checks that a single kernel (AFTER node) satisfies:
//  1. No movement ops survive in the kernel body.
//  2. Every RANGE variable referenced in the body is bound by END's loop list.
//  3. Every REDUCE node's AxisReduce ranges are in END's loop list.
//  4. Every BUFFER access in the body goes through an INDEX node.
func verifyKernel(t *testing.T, after uop.UOp) {
	t.Helper()
	if after.Op() != uop.OpAfter {
		t.Fatalf("verifyKernel: got %s, expected After", after.Op())
	}
	if after.NSrc() != 2 {
		t.Fatalf("After has %d srcs, expected 2", after.NSrc())
	}

	end := after.Src(1)
	if end.Op() != uop.OpEnd {
		t.Errorf("After.src[1] is %s, expected End", end.Op())
		return
	}

	// Collect loop-bound RANGEs from END.src[1:].
	boundRanges := make(map[uint32]bool)
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			boundRanges[r.Index()] = true
		}
	}

	if end.NSrc() < 1 {
		t.Errorf("End has no srcs")
		return
	}
	store := end.Src(0)
	if store.Op() != uop.OpStore {
		t.Errorf("End.src[0] is %s, expected Store", store.Op())
		return
	}
	if store.NSrc() < 2 {
		t.Errorf("Store has %d srcs, expected ≥2", store.NSrc())
		return
	}
	body := store.Src(1)

	seen := make(map[uint32]bool)

	var walkBody func(u uop.UOp)
	walkBody = func(u uop.UOp) {
		if !u.Valid() || seen[u.Index()] {
			return
		}
		seen[u.Index()] = true

		switch u.Op() {
		case uop.OpReshape, uop.OpPermute, uop.OpExpand, uop.OpPad, uop.OpShrink, uop.OpFlip:
			t.Errorf("movement op %s survived in kernel body (index propagation missing)", u.Op())

		case uop.OpReduceAxis:
			t.Errorf("tensor-level ReduceAxis survived in kernel body (should be converted to Reduce)")

		case uop.OpBufferize:
			t.Errorf("Bufferize survived in kernel body after addBuffers")

		case uop.OpRange:
			ra, ok := u.Arg().(uop.RangeArg)
			if !ok {
				t.Errorf("Range.arg is %T, not RangeArg", u.Arg())
				return
			}
			if !boundRanges[u.Index()] {
				t.Errorf("Range(id=%d,size=%d,type=%d) used in body but not in END loop list",
					ra.ID, ra.Size, int(ra.Type))
			}
			return // leaf — no srcs

		case uop.OpReduce:
			// src[0] = element expr; src[1:] = AxisReduce RANGE nodes.
			for i := 1; i < u.NSrc(); i++ {
				r := u.Src(i)
				if r.Op() != uop.OpRange {
					t.Errorf("Reduce.src[%d] is %s, expected Range", i, r.Op())
					continue
				}
				ra, ok := r.Arg().(uop.RangeArg)
				if !ok {
					t.Errorf("Reduce Range.arg is %T", r.Arg())
					continue
				}
				if ra.Type != uop.AxisReduce {
					t.Errorf("Reduce.src[%d] Range(id=%d) has type %d, expected AxisReduce",
						i, ra.ID, int(ra.Type))
				}
				if !boundRanges[r.Index()] {
					t.Errorf("Reduce AxisReduce Range(id=%d) not in END loop list", ra.ID)
				}
			}

		case uop.OpBuffer:
			return // leaf boundary — don't recurse into LUNIQUE/DEVICE

		case uop.OpConst, uop.OpLUnique, uop.OpDevice, uop.OpDefineVar:
			return // true leaf nodes

		default:
			// For any op that is NOT Index: no src should be a bare Buffer.
			if u.Op() != uop.OpIndex {
				for i := 0; i < u.NSrc(); i++ {
					if u.Src(i).Op() == uop.OpBuffer {
						t.Errorf("%s.src[%d] is a bare Buffer not accessed through Index", u.Op(), i)
					}
				}
			}
		}

		for i := 0; i < u.NSrc(); i++ {
			walkBody(u.Src(i))
		}
	}
	walkBody(body)
}

// ── Elementwise fusion ────────────────────────────────────────────────────────

func TestGetKernelGraph_SingleElemwise(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := x.Exp2()

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

func TestGetKernelGraph_ChainFuses(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "cpu")
	y := x.Exp2()
	z := y.Log2()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel (fused chain), got %d", n)
	}
	verifyKernelGraph(t, result)
}

func TestGetKernelGraph_LongChainFuses(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sin()
	z := y.Exp2()
	w := z.Neg()

	sink := makeSink(a, w)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// ── Reduce boundary ───────────────────────────────────────────────────────────

func TestGetKernelGraph_ReduceAlone(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0, 1}, false)

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel for a single reduce, got %d", n)
	}
	verifyKernelGraph(t, result)
}

func TestGetKernelGraph_ReduceThenElemwise(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0}, false) // [8]
	z := y.Exp2()               // [8]

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	// Pre-fusion: 2 kernels (reduce + elemwise)
	// Post-fusion: 1 kernel (fused reduce + exp2)
	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel (fused reduce + elemwise), got %d", n)
	}
	verifyKernelGraph(t, result)
}

func TestGetKernelGraph_FusionRespectsBufferLimit(t *testing.T) {
	a := newArena()
	// Create a reduce with 4 inputs. (Total 5 buffers: 4 in, 1 out)
	in1 := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
	in2 := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
	in3 := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
	in4 := tensor.NewLeaf(a, []int64{16, 16}, uop.Dtypes.Float32, "webgpu")
	red := in1.Add(in2).Add(in3).Add(in4).Sum([]int{0}, false) // [16]

	// Create an elementwise consumer with 4 OTHER inputs.
	// Total buffers if fused:
	//   Producers of red: in1, in2, in3, in4 (4)
	//   Other inputs of consumer: in5, in6, in7, in8 (4)
	//   Output of consumer: res (1)
	//   Total = 4 + 4 + 1 = 9 buffers. (Exceeds 8)
	in5 := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	in6 := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	in7 := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	in8 := tensor.NewLeaf(a, []int64{16}, uop.Dtypes.Float32, "webgpu")
	res := red.Add(in5).Add(in6).Add(in7).Add(in8)

	sink := makeSink(a, res)
	// Use "webgpu" device to ensure the 8-buffer limit is relevant.
	// Note: removeBufferize currently has hardcoded 8 limit anyway.
	result := schedule.GetKernelGraph(sink, "webgpu")

	// Should NOT fuse because 9 buffers > 8.
	// Expected 2 kernels: one for red (5 bufs), one for res (5 bufs: red, in5, in6, in7, in8).
	if n := kernelCount(result); n != 2 {
		t.Errorf("expected 2 kernels due to buffer limit, got %d", n)
	}
	verifyKernelGraph(t, result)
}

func TestGetKernelGraph_TwoIndependentOutputs(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	oy := x.Exp2()
	oz := y.Sin()

	sink := makeSink(a, oy, oz)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 2 {
		t.Errorf("expected 2 kernels (two independent chains), got %d", n)
	}
	verifyKernelGraph(t, result)
}

// ── Movement ops in a chain ───────────────────────────────────────────────────

func TestGetKernelGraph_ReshapeFuses(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Reshape([]int64{32})
	z := y.Exp2()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel (reshape+elemwise fused), got %d", n)
	}
	verifyKernelGraph(t, result)
}

// ── Scalar output ─────────────────────────────────────────────────────────────

func TestGetKernelGraph_ScalarOutput(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{}, uop.Dtypes.Float32, "cpu")
	y := x.Exp2()

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel for scalar computation, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// ── Graph structure sanity ────────────────────────────────────────────────────

func TestGetKernelGraph_SinkPreserved(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := x.Sin()

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	if !result.Valid() {
		t.Fatal("GetKernelGraph returned invalid UOp")
	}
	if result.Op() != uop.OpSink {
		t.Errorf("expected SINK root, got %s", result.Op())
	}
	verifyKernelGraph(t, result)
}

func TestGetKernelGraph_SinkSourcesAreAfter(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := x.Exp2()

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	if result.NSrc() != 1 {
		t.Fatalf("expected SINK with 1 src, got %d", result.NSrc())
	}
	src := result.Src(0)
	if src.Op() != uop.OpAfter {
		t.Errorf("SINK src should be OpAfter, got %s", src.Op())
	}
	verifyKernelGraph(t, result)
}

// ── Backward+forward graph ────────────────────────────────────────────────────

func TestGetKernelGraph_ForwardBackward(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	loss := x.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{x})
	gx, ok := grads[x]
	if !ok {
		t.Fatal("Backward returned no gradient for x")
	}

	sink := makeSink(a, loss, gx)
	result := schedule.GetKernelGraph(sink, "cpu")

	if !result.Valid() {
		t.Fatal("GetKernelGraph on fwd+bwd graph returned invalid UOp")
	}
	n := kernelCount(result)
	if n < 1 {
		t.Errorf("expected ≥1 kernels for fwd+bwd graph, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// ── Movement-op kernel tests (primary loop-nest verification) ─────────────────

// TestKernel_Reshape: flat→2D reshape fused with unary op; verifies that flat
// index decomposition produces a complete, lowerable loop nest.
func TestKernel_Reshape(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{32}, uop.Dtypes.Float32, "cpu")
	y := x.Reshape([]int64{4, 8})
	z := y.Neg()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// TestKernel_Permute: transpose [3,4] → [4,3]; verifies that output range i is
// mapped to source dim perm[i] (inverse permutation).
func TestKernel_Permute(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	y := x.Permute([]int{1, 0}) // [4, 3]
	z := y.Sin()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// TestKernel_Expand: broadcast size-1 dim [1,4]→[3,4]; verifies that broadcast
// dims produce Const(0) in the source index (no extra RANGE created).
func TestKernel_Expand(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Float32, "cpu")
	y := x.Expand([]int64{3, 4})
	z := y.Sqrt()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// TestKernel_Shrink: slice [0,1)×[2,6) from a [4,8] tensor; verifies offset
// arithmetic (r + lo) in the source index.
func TestKernel_Shrink(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Shrink([][2]int64{{0, 2}, {2, 6}}) // [2, 4]
	z := y.Recip()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// TestKernel_Pad: zero-pad [4]→[8] with 2 elements on each side; verifies that
// the body contains a WHERE validity guard and no movement ops.
func TestKernel_Pad(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y := x.Pad([][2]int64{{2, 2}}) // [8]
	z := y.Exp2()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// TestKernel_Flip: mirror a [4,4] tensor along axis 0; verifies that the source
// index is (size-1)-r for the flipped axis.
func TestKernel_Flip(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 4}, uop.Dtypes.Float32, "cpu")
	y := x.Flip([]bool{true, false})
	z := y.Log2()

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)
}

// TestKernel_ReduceHasReduceRanges: scalar sum over [4,8] must produce a kernel
// with AxisReduce RANGE nodes in the END loop list and a REDUCE node in the body.
func TestKernel_ReduceHasReduceRanges(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0, 1}, false) // scalar

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel, got %d", n)
	}
	verifyKernelGraph(t, result)

	// Additional: confirm AxisReduce ranges appear in END.
	arena := result.Arena()
	foundReduce := false
	for i := 0; i < arena.Len(); i++ {
		u := arena.At(uint32(i))
		if u.Op() != uop.OpAfter {
			continue
		}
		end := u.Src(1)
		for j := 1; j < end.NSrc(); j++ {
			r := end.Src(j)
			if r.Op() == uop.OpRange {
				if ra, ok := r.Arg().(uop.RangeArg); ok && ra.Type == uop.AxisReduce {
					foundReduce = true
				}
			}
		}
	}
	if !foundReduce {
		t.Error("no AxisReduce Range found in END loop list for a reduce kernel")
	}
}

// TestKernel_ReduceOfPermuted: sum over axis 0 of a transposed [3,4] tensor.
// Exercises both permute swizzle (pass 3, Part 1) and reduce range creation
// (pass 3, Part 2) in the same kernel.
func TestKernel_ReduceOfPermuted(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	y := x.Permute([]int{1, 0}) // [4, 3]
	z := y.Sum([]int{0}, false)  // [3]

	sink := makeSink(a, z)
	result := schedule.GetKernelGraph(sink, "cpu")

	if n := kernelCount(result); n != 1 {
		t.Errorf("expected 1 kernel (permute+reduce fused), got %d", n)
	}
	verifyKernelGraph(t, result)

	// Confirm both AxisLoop and AxisReduce ranges in the kernel.
	arena := result.Arena()
	var loopCount, reduceCount int
	for i := 0; i < arena.Len(); i++ {
		u := arena.At(uint32(i))
		if u.Op() != uop.OpAfter {
			continue
		}
		end := u.Src(1)
		for j := 1; j < end.NSrc(); j++ {
			r := end.Src(j)
			if r.Op() != uop.OpRange {
				continue
			}
			if ra, ok := r.Arg().(uop.RangeArg); ok {
				if ra.Type == uop.AxisLoop {
					loopCount++
				} else if ra.Type == uop.AxisReduce {
					reduceCount++
				}
			}
		}
	}
	if loopCount != 1 {
		t.Errorf("expected 1 AxisLoop range (output dim), got %d", loopCount)
	}
	if reduceCount != 1 {
		t.Errorf("expected 1 AxisReduce range (reduce dim), got %d", reduceCount)
	}
}

// ── Reduce-over-padded value tests ────────────────────────────────────────────
//
// These tests verify that the pad fill value inside a reduce body is the
// identity element of the enclosing reduce op, not always 0.
// verifyKernelGraph confirms the structural loop-nest invariants; the value
// oracle (evalFirstKernel) confirms correctness at runtime.

// TestKernelValue_MaxOverPaddedNegative is the canonical bug repro.
// x = [-3,-1,-4,-2] padded [2 zeros | data | 2 zeros] → Max over all 8 elements.
// With the old fill=0 the answer was 0 (wrong); with fill=−∞ it is −1 (correct).
func TestKernelValue_MaxOverPaddedNegative(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	padded := x.Pad([][2]int64{{2, 2}}) // shape [8]
	y := padded.Max([]int{0}, false)    // scalar

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	verifyKernelGraph(t, result)

	xIdx := x.Node().Index()
	bufData := map[uint32][]float32{xIdx: {-3, -1, -4, -2}}
	bufShape := map[uint32][]int64{xIdx: {4}}

	got, gotShape := evalFirstKernel(result, bufData, bufShape)

	if len(gotShape) != 0 {
		t.Fatalf("expected scalar output (rank 0), got shape %v", gotShape)
	}
	const want = float32(-1)
	if got[0] != want {
		t.Errorf("Max over padded [-3,-1,-4,-2]: got %v, want %v (bug: fill was 0, should be -inf)", got[0], want)
	}
}

// TestKernelValue_SumOverPadded is the regression guard: Sum over a padded
// tensor must still produce the correct answer (pad fill = 0 for Sum).
func TestKernelValue_SumOverPadded(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	padded := x.Pad([][2]int64{{1, 1}}) // shape [6]
	y := padded.Sum([]int{0}, false)    // scalar

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	verifyKernelGraph(t, result)

	xIdx := x.Node().Index()
	bufData := map[uint32][]float32{xIdx: {1, 2, 3, 4}}
	bufShape := map[uint32][]int64{xIdx: {4}}

	got, _ := evalFirstKernel(result, bufData, bufShape)

	const want = float32(10) // 0 + 1 + 2 + 3 + 4 + 0
	if got[0] != want {
		t.Errorf("Sum over padded [1,2,3,4]: got %v, want %v", got[0], want)
	}
}

// TestKernelValue_MaxOverPaddedPositive confirms that padding with −∞ also
// works when the real values are positive (pad zeros must NOT win over positives).
func TestKernelValue_MaxOverPaddedPositive(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3}, uop.Dtypes.Float32, "cpu")
	padded := x.Pad([][2]int64{{3, 3}}) // shape [9], data at [3..5]
	y := padded.Max([]int{0}, false)    // scalar

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	verifyKernelGraph(t, result)

	xIdx := x.Node().Index()
	bufData := map[uint32][]float32{xIdx: {5, 2, 8}}
	bufShape := map[uint32][]int64{xIdx: {3}}

	got, _ := evalFirstKernel(result, bufData, bufShape)

	const want = float32(8)
	if got[0] != want {
		t.Errorf("Max over padded [5,2,8]: got %v, want %v", got[0], want)
	}
}

// TestKernelValue_MaxOverPadded2D verifies the identity fix for a 2-D reduce
// (sum over axis 0 of a matrix padded along that axis).  The padded rows must
// not contribute to the sum.
func TestKernelValue_SumOverPadded2D(t *testing.T) {
	a := newArena()
	// x shape [2,3]: [[1,2,3],[4,5,6]]
	x := tensor.NewLeaf(a, []int64{2, 3}, uop.Dtypes.Float32, "cpu")
	// Pad axis 0 by 1 row on each side → shape [4,3]
	padded := x.Pad([][2]int64{{1, 1}, {0, 0}})
	y := padded.Sum([]int{0}, false) // shape [3]: col sums, ignoring pad rows

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "cpu")

	verifyKernelGraph(t, result)

	xIdx := x.Node().Index()
	bufData := map[uint32][]float32{xIdx: {1, 2, 3, 4, 5, 6}}
	bufShape := map[uint32][]int64{xIdx: {2, 3}}

	got, gotShape := evalFirstKernel(result, bufData, bufShape)

	want := []float32{5, 7, 9} // col sums of [[1,2,3],[4,5,6]]
	if len(gotShape) != 1 || gotShape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", gotShape)
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("col %d: got %v, want %v", i, got[i], w)
		}
	}
}

// ── Passes 7-10: CreateSchedule tests ────────────────────────────────────────

// verifyKernelSinkAST checks that a kernel SINK (from split_kernels) has a valid
// loop nest with PARAM nodes instead of BUFFER nodes.
func verifyKernelSinkAST(t *testing.T, kernelSink uop.UOp) {
	t.Helper()
	// Must be SINK with KernelInfo arg
	if kernelSink.Op() != uop.OpSink {
		t.Fatalf("verifyKernelSinkAST: got %s, want Sink", kernelSink.Op())
	}
	ki, ok := kernelSink.Arg().(uop.KernelInfo)
	if !ok {
		t.Fatalf("kernel SINK arg is %T, want KernelInfo", kernelSink.Arg())
	}
	if ki.NumParams < 1 {
		t.Errorf("KernelInfo.NumParams=%d, want ≥1", ki.NumParams)
	}
	if kernelSink.NSrc() < 1 {
		t.Fatalf("kernel SINK has no srcs")
	}
	end := kernelSink.Src(0)
	if end.Op() != uop.OpEnd {
		t.Fatalf("kernel SINK src[0] is %s, want End", end.Op())
	}
	// Collect bound ranges from END.src[1:]
	boundRanges := make(map[uint32]bool)
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpRange {
			boundRanges[r.Index()] = true
		}
	}
	store := end.Src(0)
	if store.Op() != uop.OpStore {
		t.Fatalf("END.src[0] is %s, want Store", store.Op())
	}
	body := store.Src(1)

	seen := make(map[uint32]bool)
	var walkBody func(u uop.UOp)
	walkBody = func(u uop.UOp) {
		if !u.Valid() || seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		switch u.Op() {
		case uop.OpReshape, uop.OpPermute, uop.OpExpand, uop.OpPad, uop.OpShrink, uop.OpFlip:
			t.Errorf("movement op %s survived in kernel SINK body", u.Op())
		case uop.OpReduceAxis:
			t.Errorf("ReduceAxis survived in kernel SINK body")
		case uop.OpBuffer:
			t.Errorf("bare BUFFER survived in kernel SINK body (should be PARAM)")
		case uop.OpBufferize:
			t.Errorf("BUFFERIZE survived in kernel SINK body")
		case uop.OpRange:
			ra, ok := u.Arg().(uop.RangeArg)
			if !ok {
				t.Errorf("Range.arg is %T", u.Arg())
				return
			}
			if !boundRanges[u.Index()] {
				t.Errorf("Range(id=%d,size=%d) used in body but not in END loop list", ra.ID, ra.Size)
			}
			return
		case uop.OpReduce:
			for i := 1; i < u.NSrc(); i++ {
				r := u.Src(i)
				if r.Op() != uop.OpRange {
					t.Errorf("Reduce.src[%d] is %s, want Range", i, r.Op())
					continue
				}
				if !boundRanges[r.Index()] {
					t.Errorf("Reduce AxisReduce Range not in END loop list")
				}
			}
		case uop.OpParam:
			return // valid leaf
		case uop.OpConst, uop.OpLUnique, uop.OpDevice, uop.OpDefineVar:
			return
		}
		for i := 0; i < u.NSrc(); i++ {
			walkBody(u.Src(i))
		}
	}
	walkBody(body)

	// Also walk STORE's write destination (src[0] = INDEX(PARAM(0), *ranges))
	dst := store.Src(0)
	if dst.Op() == uop.OpIndex {
		if dst.NSrc() > 0 {
			writeDst := dst.Src(0)
			if writeDst.Op() != uop.OpParam {
				t.Errorf("STORE write dest INDEX.src[0] is %s, want Param", writeDst.Op())
			}
			if writeDst.Arg() != int64(0) {
				t.Errorf("STORE write dest PARAM arg=%v, want 0", writeDst.Arg())
			}
		}
	}
}

func TestCreateSchedule_PARAMNumbering(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0}, false) // [8] — forces 2 kernels: reduce + scalar
	z := y.Exp2()               // [8] — fuses with reduce output

	sink := makeSink(a, z)
	items := schedule.CreateSchedule(sink, "cpu")

	if len(items) == 0 {
		t.Fatal("CreateSchedule returned 0 items")
	}
	for i, item := range items {
		if !item.Ast.Valid() {
			t.Errorf("item[%d].Ast is invalid", i)
			continue
		}
		if item.Ast.Op() != uop.OpSink {
			t.Errorf("item[%d].Ast is %s, want Sink", i, item.Ast.Op())
		}
		verifyKernelSinkAST(t, item.Ast)

		// PARAM numbering: Bufs[N] is the buffer for PARAM(arg=N)
		ki, ok := item.Ast.Arg().(uop.KernelInfo)
		if !ok {
			t.Errorf("item[%d].Ast.Arg() is %T, want KernelInfo", i, item.Ast.Arg())
			continue
		}
		if ki.NumParams != len(item.Bufs) {
			t.Errorf("item[%d]: KernelInfo.NumParams=%d but len(Bufs)=%d", i, ki.NumParams, len(item.Bufs))
		}

		// No two Bufs entries may share UOpIdx
		seen := make(map[uint32]bool)
		for n, buf := range item.Bufs {
			if seen[buf.UOpIdx] {
				t.Errorf("item[%d]: duplicate UOpIdx %d at Bufs[%d]", i, buf.UOpIdx, n)
			}
			seen[buf.UOpIdx] = true
			if buf.Size <= 0 {
				t.Errorf("item[%d].Bufs[%d].Size=%d, want >0", i, n, buf.Size)
			}
			if buf.DType == nil {
				t.Errorf("item[%d].Bufs[%d].DType is nil", i, n)
			}
		}
	}
}

func TestCreateSchedule_TopoOrder(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	y := x.Sum([]int{0}, false) // forces 2 kernels
	z := y.Exp2()

	sink := makeSink(a, z)
	items := schedule.CreateSchedule(sink, "cpu")

	if len(items) < 2 {
		t.Skip("expected ≥2 kernels for topo order test")
	}

	// For each kernel, collect the UOpIdx of its output buffer (Bufs[0]).
	writtenAt := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenAt[item.Bufs[0].UOpIdx] = i
	}

	// Verify: every input buffer (Bufs[1..]) is either a leaf (not in writtenAt)
	// or written by an EARLIER kernel.
	for i, item := range items {
		for n, buf := range item.Bufs[1:] {
			if writerIdx, ok := writtenAt[buf.UOpIdx]; ok {
				if writerIdx >= i {
					t.Errorf("item[%d] input Bufs[%d] (UOpIdx=%d) is written by item[%d] — topo violation",
						i, n+1, buf.UOpIdx, writerIdx)
				}
			}
		}
	}
}

func TestCreateSchedule_Determinism(t *testing.T) {
	buildGraph := func() (uop.UOp, *uop.Arena) {
		a := uop.NewArena(1024)
		x := tensor.NewLeaf(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
		y := x.Sum([]int{0}, false)
		z := y.Exp2()
		srcs := []uop.UOp{z.Node()}
		sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
		return sink, a
	}

	sink1, _ := buildGraph()
	sink2, _ := buildGraph()

	items1 := schedule.CreateSchedule(sink1, "cpu")
	items2 := schedule.CreateSchedule(sink2, "cpu")

	if len(items1) != len(items2) {
		t.Fatalf("different schedule lengths: %d vs %d", len(items1), len(items2))
	}
	for i := range items1 {
		if len(items1[i].Bufs) != len(items2[i].Bufs) {
			t.Errorf("item[%d]: different Bufs lengths %d vs %d", i, len(items1[i].Bufs), len(items2[i].Bufs))
		}
		ki1, ok1 := items1[i].Ast.Arg().(uop.KernelInfo)
		ki2, ok2 := items2[i].Ast.Arg().(uop.KernelInfo)
		if !ok1 || !ok2 {
			t.Errorf("item[%d]: KernelInfo assertion failed", i)
			continue
		}
		if ki1.NumParams != ki2.NumParams {
			t.Errorf("item[%d]: NumParams %d vs %d", i, ki1.NumParams, ki2.NumParams)
		}
	}
}

func TestCreateSchedule_MemoryPlanner_NoOverlap(t *testing.T) {
	// A linear chain A→B→C: A writes buf1, B reads buf1 writes buf2, C reads buf2 writes buf3.
	// buf1 and buf3 have non-overlapping lifetimes → they can share a slot.
	// buf2 overlaps with both (buf2 live while B is running, which is between A and C).
	a := newArena()
	x := tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "cpu")
	// Chain of 3 reduces to force 3 kernels.
	y := x.Sum([]int{0}, false)                            // scalar — kernel 0
	y2 := y.Reshape([]int64{1}).Expand([]int64{8}).Exp2() // [8] — fuses, 1 kernel
	z := y2.Sum([]int{0}, false)                          // scalar — kernel 1

	sink := makeSink(a, z)
	items := schedule.CreateSchedule(sink, "cpu")

	// Verify memory planner correctness: no two bufs with overlapping lifetimes share a slot.
	// Build lifetime map: bufUOpIdx → [firstWrite, lastRead]
	type lt struct{ first, last int }
	lifetimes := make(map[uint32]lt)
	for i, item := range items {
		buf := item.Bufs[0]
		lifetimes[buf.UOpIdx] = lt{i, i}
	}
	for i, item := range items {
		for _, buf := range item.Bufs[1:] {
			if lf, ok := lifetimes[buf.UOpIdx]; ok {
				if i > lf.last {
					lifetimes[buf.UOpIdx] = lt{lf.first, i}
				}
			}
		}
	}

	// For each pair of bufs sharing a slot, verify non-overlapping lifetimes.
	slotBufs := make(map[int][]uint32)
	for _, item := range items {
		for _, buf := range item.Bufs {
			if buf.Slot >= 0 {
				slotBufs[buf.Slot] = append(slotBufs[buf.Slot], buf.UOpIdx)
			}
		}
	}
	for slot, bufs := range slotBufs {
		for i := 0; i < len(bufs); i++ {
			for j := i + 1; j < len(bufs); j++ {
				a, b := lifetimes[bufs[i]], lifetimes[bufs[j]]
				// Overlap if a.first <= b.last && b.first <= a.last
				if a.first <= b.last && b.first <= a.last {
					t.Errorf("slot %d: bufs %d and %d have overlapping lifetimes [%d,%d] and [%d,%d]",
						slot, bufs[i], bufs[j], a.first, a.last, b.first, b.last)
				}
			}
		}
	}

	// Inputs must never be aliased (Slot=-1).
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			if _, ok := lifetimes[buf.UOpIdx]; !ok {
				// This is a leaf buffer (not written by any kernel).
				if buf.Slot != -1 {
					t.Errorf("leaf input buffer (UOpIdx=%d) has Slot=%d, want -1", buf.UOpIdx, buf.Slot)
				}
			}
		}
	}
}

func TestCreateSchedule_ForwardBackward_EndToEnd(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	loss := x.Sum(nil, false)

	grads := tensor.Backward(loss, []*tensor.Tensor{x})
	gx, ok := grads[x]
	if !ok {
		t.Fatal("Backward returned no gradient for x")
	}

	sink := makeSink(a, loss, gx)
	items := schedule.CreateSchedule(sink, "cpu")

	if len(items) == 0 {
		t.Fatal("CreateSchedule returned 0 items for fwd+bwd graph")
	}

	// Every kernel must be a valid SINK AST.
	for i, item := range items {
		verifyKernelSinkAST(t, item.Ast)
		if len(item.Bufs) == 0 {
			t.Errorf("item[%d] has 0 Bufs", i)
		}
	}

	// Verify topo order.
	writtenAt := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenAt[item.Bufs[0].UOpIdx] = i
	}
	for i, item := range items {
		for n, buf := range item.Bufs[1:] {
			if writerIdx, ok := writtenAt[buf.UOpIdx]; ok {
				if writerIdx >= i {
					t.Errorf("item[%d] input Bufs[%d] written by item[%d] — topo violation", i, n+1, writerIdx)
				}
			}
		}
	}
}

// ── Determinism: different construction order ─────────────────────────────────

// TestCreateSchedule_DeterminismBuildOrder proves that scheduling is order-invariant:
// the same logical graph built with leaf buffers in different arena construction
// orders produces identical kernel count, PARAM numbering, and input-buffer sizes.
//
// The critical case is a kernel with TWO input buffers of different sizes.  With
// the old sort-by-Index() code, building leaf_x first vs leaf_y first swaps their
// arena indices, which swaps their PARAM numbers.  After the fix (keep DFS encounter
// order, which is structural), PARAM(1) is always the leaf encountered first in the
// body DFS regardless of construction order.
func TestCreateSchedule_DeterminismBuildOrder(t *testing.T) {
	// x has shape [1] (size 1), y has shape [8] (size 8).
	// Computation: loss = sum(x.Expand([8]) + y).
	// The single kernel reads both x and y; the DFS of its body encounters x before y
	// (x.Expand is src[0] of Add), so after the fix PARAM(1) always corresponds to
	// the buffer of size 1 (x) and PARAM(2) to the buffer of size 8 (y).
	type kernelSummary struct {
		numKernels int
		numParams  []int
		inputSizes []int64 // input buffer sizes for all kernels, in schedule order
	}

	extract := func(items []schedule.ExecItem) kernelSummary {
		ks := kernelSummary{numKernels: len(items)}
		for _, item := range items {
			ki, ok := item.Ast.Arg().(uop.KernelInfo)
			if !ok {
				t.Fatalf("Ast.Arg() is %T, want KernelInfo", item.Ast.Arg())
			}
			ks.numParams = append(ks.numParams, ki.NumParams)
			for _, buf := range item.Bufs[1:] {
				ks.inputSizes = append(ks.inputSizes, buf.Size)
			}
		}
		return ks
	}

	build := func(xFirst bool) kernelSummary {
		a := uop.NewArena(1024)
		var x, y *tensor.Tensor
		if xFirst {
			x = tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Float32, "cpu") // x at lower idx
			y = tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "cpu")
		} else {
			y = tensor.NewLeaf(a, []int64{8}, uop.Dtypes.Float32, "cpu") // y at lower idx
			x = tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Float32, "cpu")
		}
		// x.Expand([8]) + y  — x is src[0] of Add in both arenas.
		xExp := x.Expand([]int64{8})
		sum := xExp.Add(y)
		loss := sum.Sum(nil, false) // scalar reduce — forces kernel boundary
		sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{loss.Node()}, nil, nil)
		return extract(schedule.CreateSchedule(sink, "cpu"))
	}

	ks1 := build(true)  // x built first: x.Index() < y.Index()
	ks2 := build(false) // y built first: y.Index() < x.Index()

	t.Logf("xFirst  → numKernels=%d, numParams=%v, inputSizes=%v", ks1.numKernels, ks1.numParams, ks1.inputSizes)
	t.Logf("yFirst  → numKernels=%d, numParams=%v, inputSizes=%v", ks2.numKernels, ks2.numParams, ks2.inputSizes)

	if ks1.numKernels != ks2.numKernels {
		t.Fatalf("numKernels: %d (xFirst) vs %d (yFirst)", ks1.numKernels, ks2.numKernels)
	}
	for i := range ks1.numParams {
		if ks1.numParams[i] != ks2.numParams[i] {
			t.Errorf("kernel[%d] NumParams: %d vs %d", i, ks1.numParams[i], ks2.numParams[i])
		}
	}
	if len(ks1.inputSizes) != len(ks2.inputSizes) {
		t.Fatalf("total input buf count: %d vs %d", len(ks1.inputSizes), len(ks2.inputSizes))
	}
	for i, sz1 := range ks1.inputSizes {
		sz2 := ks2.inputSizes[i]
		if sz1 != sz2 {
			t.Errorf("inputBuf[%d] Size: %d (xFirst) vs %d (yFirst) — "+
				"PARAM numbering is construction-order dependent (BUG)", i, sz1, sz2)
		}
	}
	t.Logf("PASS: PARAM numbering is identical regardless of construction order")
}
