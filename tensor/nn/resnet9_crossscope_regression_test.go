package nn_test

// Regression for the sibling-reduce-shared codegen fix (ResNet-9 backward
// manifestation of the cross-scope-shared identifier bug).
//
// The L0'-fix slice originally added hoistCrossScopeShared in codegen/lower.go
// to hoist ALU UOps shared between a reduce body and the post-reduce (outer)
// scope, fixing the LayerNorm-backward-shaped Block kernel: see
// block_crossscope_regression_test.go.
//
// This test exercises a second manifestation of the same class of bug: the
// kernel has multiple sibling top-level reduces (one per output channel of a
// conv-backward sum), and a loop-invariant index ALU UOp is reachable from
// BOTH reduce-body subtrees but never from the outer scope. The pre-Slice
// hoist only considered (innerReachable ∩ outerReachable), so sibling-only
// shares were missed; the first emitReduce emitted the let INSIDE its loop
// and cached the identifier, and the second emitReduce hit the cache and
// referenced the out-of-scope `t<N>`. Naga rejects with:
//
//   function main body: const 't170221' initializer: unresolved identifier: t170200
//
// The fix generalises the hoist to color each top-level reduce separately
// and the outer scope as color 0; any UOp reached by 2+ colors is hoisted.
//
// A small residual block (Conv → BN → ReLU → Conv → BN + residual) backward
// produces a kernel with many sibling reduces (one per output of the conv-
// transpose-style gradient computation) sharing input-index ALU. We compile
// it and assert every gradient Realize succeeds - independently of any
// gradient-value oracle.

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestResNet9Backward_WGSLCompiles(t *testing.T) {
	requireGPU(t)
	if testing.Short() {
		t.Skip("ResNet-9 backward is heavy; skipped under -short")
	}

	const B = int64(1)
	a := uop.NewArena(1 << 25) // 32 MiB

	// Use the smallest scale that still exercises every Conv/BN/Pool/residual
	// edge - TestResNet9Forward establishes that this scale is the floor.
	// The backward fusion produces the sibling-reduce-shared kernels that
	// the L0' hoist must handle.
	m := nn.NewResNet9Scaled(a, [4]int64{4, 8, 16, 32}, 10, uop.Dtypes.Float32, "webgpu")
	m.Load(a)
	m.Train()

	rng := rand.New(rand.NewSource(11))
	xData := make([]float32, int(B*3*32*32))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.3
	}
	x := tensor.NewLeaf(a, []int64{B, 3, 32, 32}, uop.Dtypes.Float32, "webgpu")
	x.SetData(xData)

	logits := m.Forward(x)
	loss := logits.Mul(logits).Sum(nil, false)
	leaves := make([]*tensor.Tensor, 0, len(m.Params()))
	for _, p := range m.Params() {
		leaves = append(leaves, p.T)
	}
	grads := tensor.Backward(loss, leaves)

	// Realize every parameter gradient. Before the sibling-reduce hoist fix,
	// one of the conv-backward kernels emitted WGSL like:
	//
	//   for (var r178 ...) { let t170200 = ...; ... }
	//   ...
	//   for (var r178 ...) { ...; let t170221 = (t170200 + ...); ... }
	//
	// where t170200 is loop-invariant but defined inside reduce #1 and
	// referenced from reduce #2 - Naga rejects with "unresolved identifier".
	// The pre-Slice hoist only considered (innerReachable ∩ outerReachable),
	// so sibling-only shares were missed; the generalised color-count hoist
	// in codegen/lower.go now lifts these to kernel-top.
	// Realize every parameter gradient in ONE pass. A separate Realize per grad
	// re-schedules and re-compiles the shared backward graph each time (a
	// multi-minute step); batching compiles the whole backward once. A single
	// Realize still compiles every grad's WGSL, so the sibling-reduce regression
	// this test guards is still exercised across all conv-backward kernels.
	allGrads := make([]*tensor.Tensor, 0, len(leaves))
	for i, leaf := range leaves {
		g, ok := grads[leaf]
		if !ok {
			t.Fatalf("Param[%d]: no gradient", i)
		}
		allGrads = append(allGrads, g)
	}
	if err := tensor.Realize(allGrads...); err != nil {
		t.Fatalf("Realize grads failed (sibling-reduce WGSL regression): %v", err)
	}
}
