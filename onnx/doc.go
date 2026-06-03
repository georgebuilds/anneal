// Package onnx is anneal's ONNX import frontend: it parses .onnx protobuf
// models, walks the op graph in topological order, and emits anneal Tensor ops
// that lower through the existing UOp pipeline. No new backend, no new IR,
// ideally no new kernels. Zero-CGO end to end (the generated protobuf bindings
// in ./onnxpb are pure Go; CGO_ENABLED=0 builds and tests must remain green).
//
// v1 scope: inference-only with static shapes (symbolic dims supported via
// anneal's existing Variable / shape.Sint seam — dim_param survives import as
// a symbolic dimension). Operator coverage targets the CNN core (ResNet-50,
// ResNet-9, MobileNetV2) and the transformer core (GPT-2-small, distilgpt2,
// BERT-base). Quantized ops, training graphs, control flow (If/Loop/Scan),
// Scatter*/GatherElements/GatherND, Resize/Upsample, ConvTranspose, auto_pad,
// external-data, data-dependent shapes, and Sequence/Map/Optional container
// types are explicit non-goals: they error loudly rather than silently
// producing wrong output.
//
// Architecture: the Runner holds a single uop.Arena. Import() parses the
// ModelProto into a small internal Node form (see node.go), interns the
// initializers as DeviceTensor leaves on the arena (by structural content
// hash, not by load position), and resolves the primary ai.onnx opset. Run()
// walks the lowered nodes once, threading values through a per-Run state
// map. Every value is either a host-tier integer / shape expression
// (HostValue, see value.go) or a device tensor (DeviceTensor). Shape-arithmetic
// nodes resolve to HostValue; tensor-math nodes resolve to DeviceTensor via
// per-op handlers registered in handlers.go.
package onnx
