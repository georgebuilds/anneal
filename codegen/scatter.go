package codegen

import "github.com/georgebuilds/anneal/uop"

// ── Scatter-add WGSL design notes (Slice D) ──────────────────────────────────
//
// Slice D's backward kernel for OpGather is NOT a separate WGSL template.
// The rangeify scheduler dissolves OpScatterAdd in schedule/index.go into a
// per-output-position reduce body using only existing UOps:
//
//   OpReduce(OpWhere(OpCmpEq(sortedIdx[r_B], v),
//                    OpIndex(grad, OpGatherIdx(perm[r_B], ...), *t),
//                    OpConst(0)),
//            r_B,
//            OpAdd)
//
// Codegen then renders this through the same path as any other reduce kernel
// (lower.go emitReduce + emitGatherIdx). The WGSL skeleton is therefore the
// same shape as any element-wise-mapped reduce:
//
//   @compute @workgroup_size(8, 8, 1)
//   fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//     // r0 = v in [0, V), r1..rD = trailing
//     let r0: i32 = i32(gid.x % Vu);
//     ...
//     var acc_0: f32 = 0.0;
//     for (var r_B: i32 = 0; r_B < B; r_B++) {
//       let s: i32 = data2[r_B];           // sortedIdx[b]
//       let p: i32 = data3[r_B];           // perm[b]
//       let g: f32 = data1[p * D + d];     // grad[perm[b], d]
//       let m: bool = s == r0;             // sortedIdx[b] == v
//       let w: f32 = select(0.0, g, m);    // where(m, g, 0)
//       acc_0 = acc_0 + w;
//     }
//     data0[r0 * D + d] = acc_0;
//
// Geometry choice (a) from design §6: dispatch over V * D threads, one
// register accumulator per thread, linear reduce over B. Race-free without
// atomics because distinct (v, *t) write to disjoint dW addresses, and the
// reduce uses thread-private state. Determinism follows from the host sort
// (stable) plus an ordered scalar accumulation.
//
// Geometry choice (b), dispatch over numSegments * D, was rejected:
//   * numSegments is data-dependent (changes with every idx) so the dispatch
//     grid would need to recompute every Realize, breaking the JIT design's
//     graph-keyed dispatch invariant.
//   * For Embedding workloads at nanoGPT scale (V=65) the per-thread work
//     in (a) is already tiny; the saving from (b) only matters at GPT-2
//     vocab 50257, and GPT-2 is forward-only per the demo plan.
//
// Pre-zeroing is implicit: the schedule wires a zeros tensor as Src(0) of
// OpScatterAdd which produces an OpConst(0)-broadcast kernel before the
// scatter reduce. The existing schedule pipeline materialises both kernels
// in topological order; no new dispatch infrastructure required.

// containsScatterAdd reports whether the UOp subgraph rooted at u contains an
// OpScatterAdd node. Reserved for any future optimizer that wants to skip
// fusion/tiling decisions on subgraphs that produce scatter outputs. Not
// wired in Slice D because the scatter UOp is already a hard realize point
// (it bottoms out at a BUFFER leaf the rangeifier sees, exactly like
// OpReshape on a buffer); the OpGatherIdx containment scan already protects
// tile-rewrites that would consume scatter outputs in a downstream matmul.
//
// Kept exported via the unexported name (Slice D internal use) so a future
// slice can lift it without an API churn.
func containsScatterAdd(u uop.UOp) bool {
	seen := make(map[uint32]bool)
	var walk func(n uop.UOp) bool
	walk = func(n uop.UOp) bool {
		if !n.Valid() {
			return false
		}
		idx := n.Index()
		if seen[idx] {
			return false
		}
		seen[idx] = true
		if n.Op() == uop.OpScatterAdd {
			return true
		}
		for i := 0; i < n.NSrc(); i++ {
			if walk(n.Src(i)) {
				return true
			}
		}
		return false
	}
	return walk(u)
}
