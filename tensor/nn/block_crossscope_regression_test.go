package nn_test

// Regression for the L0'-fix slice (auto-Contiguous WGSL naming bug on
// backward graphs).
//
// L0' (schedule/budget.go) inserts an OpContiguous on any over-budget kernel
// boundary so the leaf-buffer reach stays under MaxBuffersPerKernel = 8. The
// inserted Contiguous splits an over-budget kernel into two smaller kernels;
// on a backward Block graph this splits a LayerNorm-backward-shaped kernel
// whose body has BOTH a reduce loop AND a post-reduce normalization step
// that, after arena hash-consing, share loop-invariant index ALU UOps.
//
// Before the lower.go hoistCrossScopeShared pre-pass, the WGSL emitter
// emitted those shared ALU lets as `let t<N>: i32 = ...` INSIDE the reduce
// loop on first visit, cached the name in exprOf, and then re-used the
// cached identifier from a post-reduce expression, producing WGSL like:
//
//   for (var r51 ...) {
//     let t1288: i32 = (t1287 + t1286);   // declared in loop
//     ...
//   }
//   let t1291: i32 = (t1288 / 4);          // referenced OUT of scope
//
// which Naga rejects with "unresolved identifier: t1288".
//
// This test runs the full Block forward+backward on the same tiny config
// from TestBlockFDGradCheck and asserts tensor.Realize succeeds for every
// gradient, independent of the gradient-tolerance budget that block_test.go
// applies on top. The two concerns are factored separately so a future
// tolerance regression in TestBlockFDGradCheck does not mask a relapse of
// the underlying WGSL emitter bug, and vice versa.

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestBlockBackward_WGSLCompiles(t *testing.T) {
	requireGPU(t)

	const (
		B         = int64(1)
		T         = int64(4)
		nEmbd     = 8
		nHead     = 2
		blockSize = 4
	)

	a0 := uop.NewArena(4096)
	b := nn.NewBlock(a0, nEmbd, nHead, blockSize)
	rng := rand.New(rand.NewSource(7))
	blockInitSmall(b, 0.1, rng)

	xData := make([]float32, B*T*int64(nEmbd))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}

	a := uop.NewArena(131072)
	for _, p := range b.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, T, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := b.Forward(x).Sum(nil, false)

	leaves := make([]*tensor.Tensor, 0, len(b.Params())+1)
	for _, p := range b.Params() {
		leaves = append(leaves, p.T)
	}
	leaves = append(leaves, x)
	grads := tensor.Backward(loss, leaves)

	// Realize every gradient. Before the fix, kernel 13's WGSL contained an
	// out-of-scope `t<N>` identifier (loop-invariant index let emitted in
	// the reduce body, referenced from post-reduce code), so this Realize
	// call panicked with a Naga "unresolved identifier" CreateShaderModule
	// error. With the hoist pre-pass in codegen/lower.go, every shared ALU
	// UOp is pre-emitted at kernel-top scope; Realize succeeds.
	for _, leaf := range leaves {
		g, ok := grads[leaf]
		if !ok {
			t.Fatalf("no gradient for leaf")
		}
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("Realize grad failed (cross-scope-shared WGSL regression): %v", err)
		}
	}
}
