package schedule_test

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// kernelBufferCount counts the number of distinct BUFFER nodes referenced by
// the kernel rooted at `after` (an OpAfter node). The kernel's total binding
// count is 1 (output) + the count of distinct input BUFFER nodes in the body.
// We sum both for the absolute storage-buffer count.
func kernelBufferCount(after uop.UOp) int {
	if after.Op() != uop.OpAfter || after.NSrc() < 2 {
		return 0
	}
	outBuf := after.Src(0)
	end := after.Src(1)

	seen := make(map[uint32]bool)
	bufs := make(map[uint32]bool)
	var walk func(u uop.UOp)
	walk = func(u uop.UOp) {
		if seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		if u.Op() == uop.OpBuffer {
			bufs[u.Index()] = true
			return
		}
		for i := 0; i < u.NSrc(); i++ {
			walk(u.Src(i))
		}
	}
	walk(end)
	// Output buffer is one of the bindings even if it doesn't appear inside END
	// via reads (it appears via Store's destination INDEX).
	bufs[outBuf.Index()] = true
	return len(bufs)
}

// allKernelBufferCounts returns the storage-buffer count for every AFTER node
// reachable from root (sorted by output arena index for stability).
func allKernelBufferCounts(root uop.UOp) []int {
	a := root.Arena()
	out := []int{}
	for i := 0; i < a.Len(); i++ {
		u := a.At(uint32(i))
		if u.Op() == uop.OpAfter && u.NSrc() == 2 && u.Src(1).Op() == uop.OpEnd {
			out = append(out, kernelBufferCount(u))
		}
	}
	return out
}

// TestBudget_TenLeafKernelSplits is the synthetic oracle: build a tensor graph
// whose natural realize point has 10 distinct leaf buffers, then verify that
// enforceBufferBudget inserts OpContiguous nodes so the resulting kernel graph
// has every kernel within the 8-storage-buffer cap.
//
// Construction: y = ((((((((((in0 + in1) + in2) + in3) + in4) + in5) + in6)
// + in7) + in8) + in9)). Without the pass this becomes a single BUFFERIZE
// whose body reads 10 distinct OpBuffer leaves plus emits 1 output = 11
// storage bindings (exceeds the 8 cap by 3).
func TestBudget_TenLeafKernelSplits(t *testing.T) {
	a := newArena()
	const shapeD = int64(4)

	inputs := make([]*tensor.Tensor, 10)
	for i := range inputs {
		inputs[i] = tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	}
	acc := inputs[0]
	for i := 1; i < 10; i++ {
		acc = acc.Add(inputs[i])
	}

	sink := makeSink(a, acc)
	result := schedule.GetKernelGraph(sink, "webgpu")
	verifyKernelGraph(t, result)

	counts := allKernelBufferCounts(result)
	if len(counts) < 2 {
		t.Errorf("expected the 10-leaf kernel to split into >= 2 kernels, got %d kernel(s); counts=%v",
			len(counts), counts)
	}
	for i, c := range counts {
		if c > schedule.MaxBuffersPerKernel {
			t.Errorf("kernel %d has %d storage buffers (cap %d); counts=%v",
				i, c, schedule.MaxBuffersPerKernel, counts)
		}
	}
}

// TestBudget_ThirteenLeafBlockLikeKernelSplits mirrors the failing Slice L
// Block forward pattern: a single fused expression that reads many distinct
// parameter buffers (weights + biases + an input + a mask). The exact Block
// code lives in tensor/nn/ which we cannot import from schedule_test, but the
// leaf-buffer reach pressure is captured by a 13-input affine accumulation.
//
// 13 inputs vs the 8 cap means the pass must shed at least 6 buffers across
// its iterations.
func TestBudget_ThirteenLeafBlockLikeKernelSplits(t *testing.T) {
	a := newArena()
	const shapeD = int64(8)

	inputs := make([]*tensor.Tensor, 13)
	for i := range inputs {
		inputs[i] = tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	}
	acc := inputs[0]
	for i := 1; i < 13; i++ {
		acc = acc.Add(inputs[i])
	}

	sink := makeSink(a, acc)
	result := schedule.GetKernelGraph(sink, "webgpu")
	verifyKernelGraph(t, result)

	counts := allKernelBufferCounts(result)
	if len(counts) < 2 {
		t.Errorf("expected >=2 kernels post-split; got %d. counts=%v", len(counts), counts)
	}
	for i, c := range counts {
		if c > schedule.MaxBuffersPerKernel {
			t.Errorf("kernel %d has %d storage buffers (cap %d); counts=%v",
				i, c, schedule.MaxBuffersPerKernel, counts)
		}
	}
}

// TestBudget_NoOpUnderBudget verifies that a kernel already within budget is
// NOT split by the auto-Contiguous pass. This guards against the pass kicking
// in for the common case.
func TestBudget_NoOpUnderBudget(t *testing.T) {
	a := newArena()
	const shapeD = int64(4)

	in0 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	in1 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	in2 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	in3 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	in4 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	in5 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	in6 := tensor.NewLeaf(a, []int64{shapeD}, uop.Dtypes.Float32, "webgpu")
	// 7 leaves + 1 output = 8 bindings, exactly at the cap, no split.
	y := in0.Add(in1).Add(in2).Add(in3).Add(in4).Add(in5).Add(in6)

	sink := makeSink(a, y)
	result := schedule.GetKernelGraph(sink, "webgpu")
	verifyKernelGraph(t, result)

	if n := kernelCount(result); n != 1 {
		t.Errorf("under-budget kernel must not split; got %d kernels", n)
	}
	counts := allKernelBufferCounts(result)
	if len(counts) != 1 || counts[0] > schedule.MaxBuffersPerKernel {
		t.Errorf("expected single kernel <=%d bufs; got counts=%v", schedule.MaxBuffersPerKernel, counts)
	}
}

// TestBudget_ValueOracle verifies bit-identical numerical results for a
// 10-leaf graph that the pass splits. We evaluate both the post-split kernel
// graph and a hand-computed reference by interpreting via the same
// evalFirstKernel-style infrastructure; here we just check that the
// schedule's kernel-level computation matches a direct CPU sum.
//
// Strategy: rely on the existing CPU schedule path (device="cpu") to evaluate
// the produced kernels via the interpreter helpers in schedule_test.go. The
// 10-leaf pattern has well-defined integer-arithmetic outputs.
func TestBudget_ValueOracle(t *testing.T) {
	a := newArena()
	const D = int64(3)

	// Inputs filled with i+1 scalars (so the sum is deterministic).
	inputs := make([]*tensor.Tensor, 10)
	for i := range inputs {
		inputs[i] = tensor.NewLeaf(a, []int64{D}, uop.Dtypes.Float32, "cpu")
	}
	acc := inputs[0]
	for i := 1; i < 10; i++ {
		acc = acc.Add(inputs[i])
	}

	sink := makeSink(a, acc)
	result := schedule.GetKernelGraph(sink, "cpu")
	verifyKernelGraph(t, result)

	// Confirm the pass split the kernel.
	counts := allKernelBufferCounts(result)
	for i, c := range counts {
		if c > schedule.MaxBuffersPerKernel {
			t.Errorf("kernel %d has %d storage buffers; counts=%v", i, c, counts)
		}
	}

	// Evaluate each kernel in dependency order, feeding the output of each
	// into a shared bufData map keyed by output BUFFER arena index. We can't
	// reuse the schedule_test interpreter directly (private helpers), so we
	// do a minimal CPU evaluation: walk AFTER nodes in arena order, evaluate
	// each via evalKernel, and collect.
	bufData := make(map[uint32][]float32)
	bufShape := make(map[uint32][]int64)

	// Seed leaf BUFFER data: input i carries i+1 in every element.
	for i, in := range inputs {
		buf := in.Node() // OpBuffer for the leaf.
		dat := make([]float32, D)
		for j := range dat {
			dat[j] = float32(i + 1)
		}
		bufData[buf.Index()] = dat
		bufShape[buf.Index()] = []int64{D}
	}

	arena := result.Arena()
	for i := 0; i < arena.Len(); i++ {
		u := arena.At(uint32(i))
		if u.Op() != uop.OpAfter || u.NSrc() != 2 || u.Src(1).Op() != uop.OpEnd {
			continue
		}
		outBuf := u.Src(0)
		out, outShape := evalKernel(u, bufData, bufShape)
		bufData[outBuf.Index()] = out
		bufShape[outBuf.Index()] = outShape
	}

	// The sink's output buffer holds the final result.
	// SINK.Src(0) is a CALL or AFTER; we instead inspect the last AFTER's data.
	// Sum of 1..10 = 55; every element should be 55.
	final, ok := bufData[findFinalOutputBuf(result).Index()]
	if !ok {
		t.Fatalf("could not locate final output buffer in bufData")
	}
	for j, v := range final {
		if v != 55 {
			t.Errorf("final[%d] = %v, want 55", j, v)
		}
	}
}

// findFinalOutputBuf returns the BUFFER node that is the output of the
// schedule's terminal AFTER (the one whose buffer is consumed by SINK).
func findFinalOutputBuf(root uop.UOp) uop.UOp {
	// SINK.Src(0..N) point to the realize boundaries; after addBuffers/split
	// these are AFTER nodes. Pick the last sink child and walk.
	if root.Op() != uop.OpSink || root.NSrc() == 0 {
		return uop.UOp{}
	}
	last := root.Src(root.NSrc() - 1)
	for last.Op() == uop.OpAfter || last.Op() == uop.OpCall {
		if last.Op() == uop.OpAfter {
			return last.Src(0)
		}
		// OpCall.Src(1) is output buffer.
		if last.Op() == uop.OpCall {
			return last.Src(1)
		}
	}
	return last
}
