// Package cpu is a pure-Go CPU backend for anneal. It implements
// backend.Executor by walking the kernel SINK-rooted UOp AST on the host
// rather than rendering WGSL and dispatching on a GPU. There is no codegen
// step, no Renderer/Compiler/Program - Device.Run interprets the IR
// directly against host-side []float32 / []int32 buffers.
//
// The backend covers the static (non-symbolic) op set the MLP demo needs:
// PARAM/Index load+store, the standard ALU set (Add, Sub, Mul, Neg, Div,
// Where, CmpLt, Max, Min, ReLU via Maximum, Reciprocal, Sqrt, Exp2, Log2),
// scalar Const, integer arithmetic for index expressions, OpReduce (Add /
// Max) over inner ranges, and Cast on f32. Ops outside this set produce a
// clear "not yet implemented" error.
//
// Symbolic kernels (binding-driven dispatch) are not yet implemented;
// SymbolicExecutor is intentionally not satisfied. The interpreter assumes
// every OpRange in the kernel has a concrete OpConst bound.
package cpu
