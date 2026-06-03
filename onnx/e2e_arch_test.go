package onnx

// Phase 2 architectural bit-exact E2E tests.
//
// Per the Phase 2 plan: validate that the importer's Stage-1 CNN op surface
// holds end-to-end for the architectural shapes of the v1 model targets
// (ResNet-9, ResNet-50 bottleneck, MobileNetV2 depthwise punt). Same strategy
// as e2e_cnn_test.go: build the model twice on the same arena (direct +
// importer), evaluate both via cpuEval, assert bit-exact equality of the
// realized []float32 output.
//
// Scope:
//
//   - TestE2E_ResNet9_BitExact: ResNet-9-shaped architecture (stem +
//     residual blocks + downsample transitions + classifier head).
//   - TestE2E_ResNet50_Block_BitExact: ResNet-50 bottleneck-block-stack
//     architecture (1x1 -> 3x3 -> 1x1 with projection shortcut, repeated).
//   - TestE2E_MobileNetV2_DepthwiseBlock_PuntsLoudly: a depthwise Conv
//     (group=in_channels) must be rejected with a clear error since
//     tensor/nn.Conv2d does not support groups>1; MobileNetV2 is deferred
//     to v1.1.
//   - TestE2E_ClassifierTail_BitExact: older-PyTorch Shape/Gather/Unsqueeze
//     /Concat/Reshape tail vs the clean GlobalAveragePool/Flatten/Gemm tail,
//     bit-exact across both via importer.
//   - TestE2E_ResNet9_MultiBatch: ResNet-9 bit-exact at N in {1, 4, 8}.
//   - TestE2E_ResNet9_SymbolicBatch: ResNet-9 with dim_param "N", realized
//     at the same three N values, bit-exact each time.

import (
	"math/rand"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── ResNet-9 shaped architecture ─────────────────────────────────────────────

// resnet9Spec is a compact ResNet-9-shape: stem conv, three residual stages,
// classifier head. Channels grow 16 -> 32 -> 64; spatial dims halve at each
// transition via MaxPool. With H=W=32 this mirrors the CIFAR ResNet-9 layout
// (just at half the channel widths to keep arena/eval cheap).
type resnet9Spec struct {
	N           int64
	Cin         int64 // 3
	H, W        int64 // 32
	C1, C2, C3  int64 // 16, 32, 64
	NumClass    int64
	InputName   string
}

func defaultResNet9Spec(N int64) resnet9Spec {
	return resnet9Spec{
		N:        N,
		Cin:      3,
		H:        32,
		W:        32,
		C1:       16,
		C2:       32,
		C3:       64,
		NumClass: 10,
		InputName: "x",
	}
}

// registerResNet9Weights fills w with all named tensors for the ResNet-9
// builder.  Variance initialisers are sampled in [0.5, 1.5] so the BN denom
// (sqrt(var + eps)) stays well-conditioned. Other weights use a small
// centred uniform.
func registerResNet9Weights(w *cnnWeights, rng *rand.Rand, spec resnet9Spec) {
	// Stem.
	w.randomFill(rng, "stem_w", []int64{spec.C1, spec.Cin, 3, 3}, 0.3)
	registerBN(w, rng, "stem_bn", spec.C1)
	// Block1: 16 -> 16, residual no projection.
	w.randomFill(rng, "b1_c1_w", []int64{spec.C1, spec.C1, 3, 3}, 0.2)
	registerBN(w, rng, "b1_bn1", spec.C1)
	w.randomFill(rng, "b1_c2_w", []int64{spec.C1, spec.C1, 3, 3}, 0.2)
	registerBN(w, rng, "b1_bn2", spec.C1)
	// Transition 1: MaxPool then 16->32 conv.
	w.randomFill(rng, "tr1_w", []int64{spec.C2, spec.C1, 3, 3}, 0.2)
	registerBN(w, rng, "tr1_bn", spec.C2)
	// Block2: 32 -> 32, residual no projection.
	w.randomFill(rng, "b2_c1_w", []int64{spec.C2, spec.C2, 3, 3}, 0.2)
	registerBN(w, rng, "b2_bn1", spec.C2)
	w.randomFill(rng, "b2_c2_w", []int64{spec.C2, spec.C2, 3, 3}, 0.2)
	registerBN(w, rng, "b2_bn2", spec.C2)
	// Transition 2: MaxPool then 32->64 conv.
	w.randomFill(rng, "tr2_w", []int64{spec.C3, spec.C2, 3, 3}, 0.2)
	registerBN(w, rng, "tr2_bn", spec.C3)
	// Block3: 64 -> 64.
	w.randomFill(rng, "b3_c1_w", []int64{spec.C3, spec.C3, 3, 3}, 0.2)
	registerBN(w, rng, "b3_bn1", spec.C3)
	w.randomFill(rng, "b3_c2_w", []int64{spec.C3, spec.C3, 3, 3}, 0.2)
	registerBN(w, rng, "b3_bn2", spec.C3)
	// Classifier head.
	w.randomFill(rng, "fc_w", []int64{spec.C3, spec.NumClass}, 0.5)
	w.randomFill(rng, "fc_b", []int64{spec.NumClass}, 0.05)
}

// registerBN fills BN scale/bias/mean/var with sensible random values. var
// is in [0.5, 1.5] to keep sqrt(var + eps) well-conditioned.
func registerBN(w *cnnWeights, rng *rand.Rand, prefix string, c int64) {
	w.randomFill(rng, prefix+"_scale", []int64{c}, 0.5)
	w.randomFill(rng, prefix+"_bias", []int64{c}, 0.1)
	w.randomFill(rng, prefix+"_mean", []int64{c}, 0.1)
	v := make([]float32, c)
	for i := range v {
		v[i] = 0.5 + rng.Float32()
	}
	w.set(prefix+"_var", []int64{c}, v)
}

// buildDirectResNet9 builds the ResNet-9 forward graph via the tensor/nn
// API. Returns the [N, NumClass] logits.
func buildDirectResNet9(arena *uop.Arena, spec resnet9Spec, w *cnnWeights, x *tensor.Tensor) *tensor.Tensor {
	// Stem.
	h := convForward(arena, w, "stem_w", x, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, "stem_bn", spec.C1)
	h = nn.ReLU(h)
	// Block 1.
	h = residualBlock(arena, h, w, "b1", spec.C1)
	// Transition 1: maxpool 2x2 then conv 3x3 + BN + ReLU.
	h = nn.MaxPool2D(h, 2, 2, 2, 2)
	h = convForward(arena, w, "tr1_w", h, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, "tr1_bn", spec.C2)
	h = nn.ReLU(h)
	// Block 2.
	h = residualBlock(arena, h, w, "b2", spec.C2)
	// Transition 2.
	h = nn.MaxPool2D(h, 2, 2, 2, 2)
	h = convForward(arena, w, "tr2_w", h, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, "tr2_bn", spec.C3)
	h = nn.ReLU(h)
	// Block 3.
	h = residualBlock(arena, h, w, "b3", spec.C3)
	// Classifier head: GAP, Flatten, Linear+bias.
	rank := h.Rank()
	axes := make([]int, 0, rank-2)
	for i := 2; i < rank; i++ {
		axes = append(axes, i)
	}
	h = h.Mean(axes, true)
	h = h.Reshape([]int64{spec.N, spec.C3})
	wMat := leafParam(arena, w, "fc_w").T
	logits := h.Matmul(wMat)
	biasT := leafParam(arena, w, "fc_b").T
	biasB := tensor.BroadcastToSints(biasT, logits.ShapeSints())
	logits = logits.Add(biasB)
	return logits
}

// residualBlock builds a no-projection residual block: Conv-BN-Relu-Conv-BN-+x-Relu.
// Both convs operate on `c` channels with 3x3 kernels, stride 1, pad 1.
func residualBlock(arena *uop.Arena, x *tensor.Tensor, w *cnnWeights, prefix string, c int64) *tensor.Tensor {
	h := convForward(arena, w, prefix+"_c1_w", x, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, prefix+"_bn1", c)
	h = nn.ReLU(h)
	h = convForward(arena, w, prefix+"_c2_w", h, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, prefix+"_bn2", c)
	h = h.Add(x)
	return nn.ReLU(h)
}

// convForward wires up an nn.Conv2d with the registered weight tensor.
func convForward(arena *uop.Arena, w *cnnWeights, name string, x *tensor.Tensor, stride, pad [2]int) *tensor.Tensor {
	conv := &nn.Conv2d{
		Weight: leafParam(arena, w, name),
		Stride: stride,
		Pad:    pad,
	}
	return conv.Forward(x)
}

// buildModelProtoResNet9 constructs the ModelProto for the same ResNet-9
// using ONNX Conv / BatchNormalization / Relu / Add / MaxPool /
// GlobalAveragePool / Flatten / Gemm.
func buildModelProtoResNet9(spec resnet9Spec, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "resnet9"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{
		makeVI(spec.InputName, onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.Cin, spec.H, spec.W}),
	}
	g.Output = []*onnxpb.ValueInfoProto{
		makeVI("y", onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.NumClass}),
	}

	prev := spec.InputName
	prev = addConvNode(g, "stem", prev, "stem_w", "")
	prev = addBNNode(g, "stem_bn", prev, "stem_bn")
	prev = addReluNode(g, "stem_relu", prev)
	prev = addResidualBlockNodes(g, "b1", prev)
	prev = addMaxPoolNode(g, "mp1", prev)
	prev = addConvNode(g, "tr1", prev, "tr1_w", "")
	prev = addBNNode(g, "tr1_bn", prev, "tr1_bn")
	prev = addReluNode(g, "tr1_relu", prev)
	prev = addResidualBlockNodes(g, "b2", prev)
	prev = addMaxPoolNode(g, "mp2", prev)
	prev = addConvNode(g, "tr2", prev, "tr2_w", "")
	prev = addBNNode(g, "tr2_bn", prev, "tr2_bn")
	prev = addReluNode(g, "tr2_relu", prev)
	prev = addResidualBlockNodes(g, "b3", prev)
	gapOut := "gap"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: gapOut, OpType: "GlobalAveragePool",
		Input: []string{prev}, Output: []string{gapOut},
	})
	flat := "flat"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: flat, OpType: "Flatten",
		Input:     []string{gapOut},
		Output:    []string{flat},
		Attribute: []*onnxpb.AttributeProto{intAttr("axis", 1)},
	})
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: "fc", OpType: "Gemm",
		Input:  []string{flat, "fc_w", "fc_b"},
		Output: []string{"y"},
	})

	return &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph:       g,
	}
}

// addConvNode appends a Conv with stride=1, pad=1 3x3 explicit pads. weight is
// the initializer name; bias may be "" to skip.
func addConvNode(g *onnxpb.GraphProto, name, x, weight, bias string) string {
	out := name + "_out"
	inputs := []string{x, weight}
	if bias != "" {
		inputs = append(inputs, bias)
	}
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: name, OpType: "Conv",
		Input:  inputs,
		Output: []string{out},
		Attribute: []*onnxpb.AttributeProto{
			intsAttr("kernel_shape", []int64{3, 3}),
			intsAttr("strides", []int64{1, 1}),
			intsAttr("pads", []int64{1, 1, 1, 1}),
		},
	})
	return out
}

// addBNNode appends a BatchNormalization node. prefix names the four init
// arrays scale/bias/mean/var via "_scale", "_bias", "_mean", "_var".
func addBNNode(g *onnxpb.GraphProto, name, x, prefix string) string {
	out := name + "_out"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: name, OpType: "BatchNormalization",
		Input:     []string{x, prefix + "_scale", prefix + "_bias", prefix + "_mean", prefix + "_var"},
		Output:    []string{out},
		Attribute: []*onnxpb.AttributeProto{floatAttr("epsilon", 1e-5)},
	})
	return out
}

func addReluNode(g *onnxpb.GraphProto, name, x string) string {
	out := name + "_out"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: name, OpType: "Relu",
		Input: []string{x}, Output: []string{out},
	})
	return out
}

// addMaxPoolNode appends a 2x2 stride-2 MaxPool node with explicit zero pads.
func addMaxPoolNode(g *onnxpb.GraphProto, name, x string) string {
	out := name + "_out"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: name, OpType: "MaxPool",
		Input: []string{x}, Output: []string{out},
		Attribute: []*onnxpb.AttributeProto{
			intsAttr("kernel_shape", []int64{2, 2}),
			intsAttr("strides", []int64{2, 2}),
			intsAttr("pads", []int64{0, 0, 0, 0}),
		},
	})
	return out
}

// addResidualBlockNodes appends the Conv-BN-Relu-Conv-BN-Add-Relu chain. The
// initialisers used are prefix+"_c1_w", prefix+"_bn1_{scale,bias,mean,var}",
// prefix+"_c2_w", prefix+"_bn2_{...}".
func addResidualBlockNodes(g *onnxpb.GraphProto, prefix, x string) string {
	c1 := addConvNode(g, prefix+"_c1", x, prefix+"_c1_w", "")
	bn1 := addBNNode(g, prefix+"_bn1", c1, prefix+"_bn1")
	r1 := addReluNode(g, prefix+"_relu1", bn1)
	c2 := addConvNode(g, prefix+"_c2", r1, prefix+"_c2_w", "")
	bn2 := addBNNode(g, prefix+"_bn2", c2, prefix+"_bn2")
	addOut := prefix + "_add_out"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: prefix + "_add", OpType: "Add",
		Input: []string{bn2, x}, Output: []string{addOut},
	})
	return addReluNode(g, prefix+"_relu2", addOut)
}

// TestE2E_ResNet9_BitExact is the headline Phase 2 ResNet-9 gate.
func TestE2E_ResNet9_BitExact(t *testing.T) {
	spec := defaultResNet9Spec(1)
	arena := uop.NewArena(65536)
	w := newCNNWeights()
	rng := rand.New(rand.NewSource(2026))
	registerResNet9Weights(w, rng, spec)

	xLeaf, _ := makeFloatInput(arena, []int64{spec.N, spec.Cin, spec.H, spec.W}, rng, "x")

	direct := buildDirectResNet9(arena, spec, w, xLeaf)
	sh := direct.Shape()
	if sh[0] != spec.N || sh[1] != spec.NumClass {
		t.Fatalf("direct output shape %v, want [%d %d]", sh, spec.N, spec.NumClass)
	}
	model := buildModelProtoResNet9(spec, w)
	imported := realizeViaImporter(t, arena, model,
		map[string]*tensor.Tensor{spec.InputName: xLeaf}, "y")
	assertBitExact(t, direct, imported, "ResNet9")
}

// ── ResNet-50 bottleneck-block stack ─────────────────────────────────────────

// resnet50StackSpec is a small but architecturally-faithful ResNet-50 stack:
// stem 7x7 (replaced with 3x3 for compactness) -> 4 bottleneck stages.
// Each stage has one bottleneck block (vs ResNet-50's [3,4,6,3]); the
// projection shortcut and 1x1-3x3-1x1 expansion are bit-equivalent
// architecturally to the real model. Channels: 16 / [16,16,32] /
// [16,16,64] / [16,16,128] / [16,16,256]. Spatial 16->16->8->4.
// (Smaller widths to keep arena/test cheap.)
type resnet50StackSpec struct {
	N         int64
	Cin       int64 // 3
	H, W      int64 // 16
	Stem      int64 // 16
	Stage1Mid int64 // 8 (compressed)
	Stage1Out int64 // 32
	Stage2Mid int64 // 8
	Stage2Out int64 // 64
	NumClass  int64
	InputName string
}

func defaultResNet50StackSpec() resnet50StackSpec {
	return resnet50StackSpec{
		N: 1, Cin: 3, H: 16, W: 16,
		Stem:      16,
		Stage1Mid: 8, Stage1Out: 32,
		Stage2Mid: 8, Stage2Out: 64,
		NumClass:  5,
		InputName: "x",
	}
}

// registerResNet50StackWeights fills w with stem + two bottleneck blocks +
// classifier head weights. Each bottleneck has 1x1, 3x3, 1x1 + projection
// shortcut convs (for resolution + channel change).
func registerResNet50StackWeights(w *cnnWeights, rng *rand.Rand, spec resnet50StackSpec) {
	// Stem 3x3 conv, BN, ReLU. Stem maps 3 -> Stem.
	w.randomFill(rng, "stem_w", []int64{spec.Stem, spec.Cin, 3, 3}, 0.3)
	registerBN(w, rng, "stem_bn", spec.Stem)
	// Stage 1 bottleneck: Stem -> Stage1Out via Stage1Mid.
	registerBottleneckWeights(w, rng, "s1", spec.Stem, spec.Stage1Mid, spec.Stage1Out)
	// Stage 2 bottleneck: Stage1Out -> Stage2Out via Stage2Mid.
	registerBottleneckWeights(w, rng, "s2", spec.Stage1Out, spec.Stage2Mid, spec.Stage2Out)
	// Classifier.
	w.randomFill(rng, "fc_w", []int64{spec.Stage2Out, spec.NumClass}, 0.4)
	w.randomFill(rng, "fc_b", []int64{spec.NumClass}, 0.05)
}

func registerBottleneckWeights(w *cnnWeights, rng *rand.Rand, prefix string, cin, cmid, cout int64) {
	// 1x1 reduce.
	w.randomFill(rng, prefix+"_c1_w", []int64{cmid, cin, 1, 1}, 0.2)
	registerBN(w, rng, prefix+"_bn1", cmid)
	// 3x3.
	w.randomFill(rng, prefix+"_c2_w", []int64{cmid, cmid, 3, 3}, 0.2)
	registerBN(w, rng, prefix+"_bn2", cmid)
	// 1x1 expand.
	w.randomFill(rng, prefix+"_c3_w", []int64{cout, cmid, 1, 1}, 0.2)
	registerBN(w, rng, prefix+"_bn3", cout)
	// Projection shortcut: 1x1 conv mapping cin -> cout.
	w.randomFill(rng, prefix+"_proj_w", []int64{cout, cin, 1, 1}, 0.2)
	registerBN(w, rng, prefix+"_proj_bn", cout)
}

// buildDirectResNet50Stack builds the bottleneck-stack forward graph
// directly. Returns logits [N, NumClass].
func buildDirectResNet50Stack(arena *uop.Arena, spec resnet50StackSpec, w *cnnWeights, x *tensor.Tensor) *tensor.Tensor {
	h := convForward(arena, w, "stem_w", x, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, "stem_bn", spec.Stem)
	h = nn.ReLU(h)
	h = bottleneckBlock(arena, h, w, "s1", spec.Stage1Mid, spec.Stage1Out)
	h = bottleneckBlock(arena, h, w, "s2", spec.Stage2Mid, spec.Stage2Out)
	// GAP + classifier.
	rank := h.Rank()
	axes := make([]int, 0, rank-2)
	for i := 2; i < rank; i++ {
		axes = append(axes, i)
	}
	h = h.Mean(axes, true)
	h = h.Reshape([]int64{spec.N, spec.Stage2Out})
	wMat := leafParam(arena, w, "fc_w").T
	logits := h.Matmul(wMat)
	biasT := leafParam(arena, w, "fc_b").T
	biasB := tensor.BroadcastToSints(biasT, logits.ShapeSints())
	return logits.Add(biasB)
}

// bottleneckBlock builds a ResNet-50 bottleneck: 1x1 -> 3x3 -> 1x1 with a
// 1x1 projection shortcut. Each conv is followed by BN; the 3rd conv's BN
// output is added to the projected shortcut, then ReLU. cmid is the inner
// (1x1-reduce/3x3) channel width; cout is the output / projection-target
// width.
func bottleneckBlock(arena *uop.Arena, x *tensor.Tensor, w *cnnWeights, prefix string, cmid, cout int64) *tensor.Tensor {
	// Main path.
	h := convForward(arena, w, prefix+"_c1_w", x, [2]int{1, 1}, [2]int{0, 0})
	h = applyBatchNorm(arena, h, w, prefix+"_bn1", cmid)
	h = nn.ReLU(h)
	h = convForward(arena, w, prefix+"_c2_w", h, [2]int{1, 1}, [2]int{1, 1})
	h = applyBatchNorm(arena, h, w, prefix+"_bn2", cmid)
	h = nn.ReLU(h)
	h = convForward(arena, w, prefix+"_c3_w", h, [2]int{1, 1}, [2]int{0, 0})
	h = applyBatchNorm(arena, h, w, prefix+"_bn3", cout)
	// Shortcut.
	skip := convForward(arena, w, prefix+"_proj_w", x, [2]int{1, 1}, [2]int{0, 0})
	skip = applyBatchNorm(arena, skip, w, prefix+"_proj_bn", cout)
	h = h.Add(skip)
	return nn.ReLU(h)
}

// buildModelProtoResNet50Stack constructs the matching ModelProto for the
// bottleneck-stack architecture.
func buildModelProtoResNet50Stack(spec resnet50StackSpec, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "resnet50_stack"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{
		makeVI(spec.InputName, onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.Cin, spec.H, spec.W}),
	}
	g.Output = []*onnxpb.ValueInfoProto{
		makeVI("y", onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.NumClass}),
	}
	prev := spec.InputName
	prev = addConvNode(g, "stem", prev, "stem_w", "")
	prev = addBNNode(g, "stem_bn", prev, "stem_bn")
	prev = addReluNode(g, "stem_relu", prev)
	prev = addBottleneckNodes(g, "s1", prev)
	prev = addBottleneckNodes(g, "s2", prev)
	g.Node = append(g.Node,
		&onnxpb.NodeProto{
			Name: "gap", OpType: "GlobalAveragePool",
			Input: []string{prev}, Output: []string{"gap"},
		},
		&onnxpb.NodeProto{
			Name: "flat", OpType: "Flatten",
			Input: []string{"gap"}, Output: []string{"flat"},
			Attribute: []*onnxpb.AttributeProto{intAttr("axis", 1)},
		},
		&onnxpb.NodeProto{
			Name: "fc", OpType: "Gemm",
			Input:  []string{"flat", "fc_w", "fc_b"},
			Output: []string{"y"},
		},
	)
	return &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph:       g,
	}
}

// addBottleneckNodes appends a bottleneck block's conv/BN/ReLU/proj/add/relu
// nodes. The 1x1 convs use pads=[0,0,0,0]; the 3x3 uses pads=[1,1,1,1].
func addBottleneckNodes(g *onnxpb.GraphProto, prefix, x string) string {
	c1 := addConv1x1(g, prefix+"_c1", x, prefix+"_c1_w")
	bn1 := addBNNode(g, prefix+"_bn1", c1, prefix+"_bn1")
	r1 := addReluNode(g, prefix+"_relu1", bn1)
	c2 := addConvNode(g, prefix+"_c2", r1, prefix+"_c2_w", "") // 3x3
	bn2 := addBNNode(g, prefix+"_bn2", c2, prefix+"_bn2")
	r2 := addReluNode(g, prefix+"_relu2", bn2)
	c3 := addConv1x1(g, prefix+"_c3", r2, prefix+"_c3_w")
	bn3 := addBNNode(g, prefix+"_bn3", c3, prefix+"_bn3")
	// Projection shortcut on the original input.
	proj := addConv1x1(g, prefix+"_proj", x, prefix+"_proj_w")
	projBN := addBNNode(g, prefix+"_proj_bn", proj, prefix+"_proj_bn")
	addOut := prefix + "_add_out"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: prefix + "_add", OpType: "Add",
		Input: []string{bn3, projBN}, Output: []string{addOut},
	})
	return addReluNode(g, prefix+"_relu3", addOut)
}

func addConv1x1(g *onnxpb.GraphProto, name, x, weight string) string {
	out := name + "_out"
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: name, OpType: "Conv",
		Input:  []string{x, weight},
		Output: []string{out},
		Attribute: []*onnxpb.AttributeProto{
			intsAttr("kernel_shape", []int64{1, 1}),
			intsAttr("strides", []int64{1, 1}),
			intsAttr("pads", []int64{0, 0, 0, 0}),
		},
	})
	return out
}

// TestE2E_ResNet50_Block_BitExact verifies the bottleneck-stack architecture
// matches between direct and importer paths.
func TestE2E_ResNet50_Block_BitExact(t *testing.T) {
	spec := defaultResNet50StackSpec()
	arena := uop.NewArena(65536)
	w := newCNNWeights()
	rng := rand.New(rand.NewSource(1729))
	registerResNet50StackWeights(w, rng, spec)
	xLeaf, _ := makeFloatInput(arena, []int64{spec.N, spec.Cin, spec.H, spec.W}, rng, "x")
	direct := buildDirectResNet50Stack(arena, spec, w, xLeaf)
	model := buildModelProtoResNet50Stack(spec, w)
	imported := realizeViaImporter(t, arena, model,
		map[string]*tensor.Tensor{spec.InputName: xLeaf}, "y")
	assertBitExact(t, direct, imported, "ResNet50Stack")
}

// ── MobileNetV2 depthwise: punt loudly ───────────────────────────────────────

// TestE2E_MobileNetV2_DepthwiseBlock_PuntsLoudly feeds a depthwise convolution
// (Conv with group=in_channels, the central op of MobileNetV2's inverted
// residual block) and asserts the importer rejects it with a clear error.
// Conv group>1 is on the Phase 1.B punt list; supporting it requires extending
// tensor/nn.Conv2d to a grouped form. MobileNetV2 is deferred to v1.1; this
// test pins the punt boundary so we notice if it ever silently slips
// (graph-explosion bug class).
//
// Architecture: a 4-channel depthwise 3x3 conv on a [1,4,8,8] input. Each
// output channel sees only one input channel (group = Cin = Cout = 4).
func TestE2E_MobileNetV2_DepthwiseBlock_PuntsLoudly(t *testing.T) {
	const C = int64(4)
	b := &singleNodeBuilder{
		opType: "Conv",
		attrs: map[string]Attr{
			"kernel_shape": {Kind: AttrInts, Is: []int64{3, 3}},
			"strides":      {Kind: AttrInts, Is: []int64{1, 1}},
			"pads":         {Kind: AttrInts, Is: []int64{1, 1, 1, 1}},
			"group":        {Kind: AttrInt, I: C}, // depthwise
		},
		inputs: []nameInfo{
			{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, C, 8, 8}},
			// W shape is [Cout=C, Cin/group=1, 3, 3] for a depthwise conv.
			{Name: "w", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{C, 1, 3, 3}},
		},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{1, C, 8, 8}}},
		initializers: []*onnxpb.TensorProto{
			makeFloatInitializerForTests("x", []int64{1, C, 8, 8}, make([]float32, 1*int(C)*8*8)),
			makeFloatInitializerForTests("w", []int64{C, 1, 3, 3}, make([]float32, int(C)*1*3*3)),
		},
	}
	model := b.build(t)
	// The error must name the rejected attribute (group) and the offending
	// value (4). We assert both substrings.
	err := runSingleNodeExpectError(t, model, nil, "group", "4")
	if err == nil {
		t.Fatalf("expected depthwise Conv to be rejected, got nil")
	}
	t.Logf("MobileNetV2 depthwise rejected (as designed): %v", err)
}

// ── Classifier tail: older exporter vs clean tail ────────────────────────────

// TestE2E_ClassifierTail_BitExact builds two models that compute the same
// [N, NumClass] logits from a [N, C, H, W] tensor via two ONNX paths:
//
//	clean:   GlobalAveragePool -> Flatten(axis=1) -> Gemm
//	messy:   Shape -> Gather(axis=0, idx=[0]) -> Unsqueeze(axes=[0]) -> Concat
//	         with literal [Cout]-shape vec -> ReduceMean(axes=[2,3], keepdims=0)
//	         -> Reshape -> Gemm
//
// Both consume the same input + same fc weights; assert bit-exact agreement.
// This pins the older-PyTorch-export glue chain (host-tier Shape/Gather/
// Unsqueeze/Concat) against the canonical clean tail.
//
// Note: in real older exports the messy chain is more elaborate (includes the
// trailing Reshape's shape input built from Shape+Gather+Concat). The version
// below covers the same handler surface — every host op in the chain is
// exercised — without modelling the full PyTorch-internal glue verbatim.
func TestE2E_ClassifierTail_BitExact(t *testing.T) {
	const (
		N        = int64(2)
		C        = int64(8)
		H        = int64(4)
		W        = int64(4)
		NumClass = int64(5)
	)
	rng := rand.New(rand.NewSource(317))
	w := newCNNWeights()
	w.randomFill(rng, "fc_w", []int64{C, NumClass}, 0.3)

	// Both arenas must be the same so primitive-call interning unifies UOps.
	arena := uop.NewArena(16384)

	// Shared input tensor for both paths.
	xLeaf, _ := makeFloatInput(arena, []int64{N, C, H, W}, rng, "x")

	cleanModel := buildClassifierTailClean(N, C, H, W, NumClass, w)
	messyModel := buildClassifierTailMessy(N, C, H, W, NumClass, w)

	cleanOut := realizeViaImporter(t, arena, cleanModel,
		map[string]*tensor.Tensor{"x": xLeaf}, "y")
	messyOut := realizeViaImporter(t, arena, messyModel,
		map[string]*tensor.Tensor{"x": xLeaf}, "y")

	assertBitExact(t, cleanOut, messyOut, "ClassifierTail")
}

func buildClassifierTailClean(N, C, H, W, NumClass int64, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "tail_clean"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{makeVI("x", onnxpb.TensorProto_FLOAT, []int64{N, C, H, W})}
	g.Output = []*onnxpb.ValueInfoProto{makeVI("y", onnxpb.TensorProto_FLOAT, []int64{N, NumClass})}
	g.Node = append(g.Node,
		&onnxpb.NodeProto{
			Name: "gap", OpType: "GlobalAveragePool",
			Input: []string{"x"}, Output: []string{"gap"},
		},
		&onnxpb.NodeProto{
			Name: "flat", OpType: "Flatten",
			Input: []string{"gap"}, Output: []string{"flat"},
			Attribute: []*onnxpb.AttributeProto{intAttr("axis", 1)},
		},
		&onnxpb.NodeProto{
			Name: "fc", OpType: "Gemm",
			Input:  []string{"flat", "fc_w"},
			Output: []string{"y"},
		},
	)
	return &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph:       g,
	}
}

// buildClassifierTailMessy emits the older-PyTorch-style glue chain. The
// classification path is the same; the spatial reduction is expressed as
// ReduceMean(axes=[2,3], keepdims=0) yielding [N, C], and a Reshape with a
// shape vector built host-side from Shape/Gather/Unsqueeze/Concat collapses
// it to [N, C] explicitly. The Reshape's shape input is the concatenation of
// (Shape(x)[0:1]) and (Constant [C]).
//
// Constants (gather_idx, c_dim) are emitted as Constant nodes (not graph
// initializers) so the host-tier Constant handler picks up their integer
// dtype and routes them as HostInts. This matches the older-exporter pattern
// where the shape-arithmetic chain stays entirely host-side.
func buildClassifierTailMessy(N, C, H, W, NumClass int64, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "tail_messy"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{makeVI("x", onnxpb.TensorProto_FLOAT, []int64{N, C, H, W})}
	g.Output = []*onnxpb.ValueInfoProto{makeVI("y", onnxpb.TensorProto_FLOAT, []int64{N, NumClass})}
	g.Node = append(g.Node,
		// Constants for shape arithmetic — emitted as Constant nodes so the
		// host-tier Constant handler intercepts and produces HostInts.
		&onnxpb.NodeProto{
			Name: "const_gather_idx", OpType: "Constant",
			Output: []string{"gather_idx"},
			Attribute: []*onnxpb.AttributeProto{
				{Name: "value", Type: onnxpb.AttributeProto_TENSOR,
					T: makeIntInitializer("gather_idx_val", []int64{1}, []int64{0})},
			},
		},
		&onnxpb.NodeProto{
			Name: "const_c_dim", OpType: "Constant",
			Output: []string{"c_dim"},
			Attribute: []*onnxpb.AttributeProto{
				{Name: "value", Type: onnxpb.AttributeProto_TENSOR,
					T: makeIntInitializer("c_dim_val", []int64{1}, []int64{C}),
				},
			},
		},
		&onnxpb.NodeProto{
			Name: "shape", OpType: "Shape",
			Input: []string{"x"}, Output: []string{"x_shape"},
		},
		// Gather(axis=0): extract x_shape[0] = N.
		&onnxpb.NodeProto{
			Name: "gather", OpType: "Gather",
			Input:     []string{"x_shape", "gather_idx"},
			Output:    []string{"gathered"},
			Attribute: []*onnxpb.AttributeProto{intAttr("axis", 0)},
		},
		// Unsqueeze axes=[0] on the gathered scalar/vector — host op, no-op on
		// the int payload but exercises the opset 11-vs-13 migration path.
		&onnxpb.NodeProto{
			Name: "uns", OpType: "Unsqueeze",
			Input:     []string{"gathered"},
			Output:    []string{"uns_out"},
			Attribute: []*onnxpb.AttributeProto{intsAttr("axes", []int64{0})},
		},
		// Concat([uns, c_dim], axis=0) -> [N, C] as a 1-D int vector.
		&onnxpb.NodeProto{
			Name: "concat", OpType: "Concat",
			Input:     []string{"uns_out", "c_dim"},
			Output:    []string{"target_shape"},
			Attribute: []*onnxpb.AttributeProto{intAttr("axis", 0)},
		},
		// Spatial reduction: ReduceMean keepdims=0 over axes [2, 3] -> [N, C].
		&onnxpb.NodeProto{
			Name: "rm", OpType: "ReduceMean",
			Input:  []string{"x"},
			Output: []string{"rm_out"},
			Attribute: []*onnxpb.AttributeProto{
				intsAttr("axes", []int64{2, 3}),
				intAttr("keepdims", 0),
			},
		},
		// Reshape rm_out to target_shape (already [N, C], so this is a
		// shape-asserting identity).
		&onnxpb.NodeProto{
			Name: "reshape", OpType: "Reshape",
			Input:  []string{"rm_out", "target_shape"},
			Output: []string{"flat"},
		},
		&onnxpb.NodeProto{
			Name: "fc", OpType: "Gemm",
			Input:  []string{"flat", "fc_w"},
			Output: []string{"y"},
		},
	)
	return &onnxpb.ModelProto{
		IrVersion:   8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{{Domain: "", Version: 13}},
		Graph:       g,
	}
}

// ── Multi-batch ResNet-9 ─────────────────────────────────────────────────────

// TestE2E_ResNet9_MultiBatch runs the same ResNet-9 architecture at N in
// {1, 4, 8}. Each batch must be bit-exact between the direct and importer
// paths; max-abs-diff is logged for every N.
func TestE2E_ResNet9_MultiBatch(t *testing.T) {
	// Weights are shared across all three batch sizes; they're built once.
	wRng := rand.New(rand.NewSource(2027))
	w := newCNNWeights()
	registerResNet9Weights(w, wRng, defaultResNet9Spec(1))

	for _, N := range []int64{1, 4, 8} {
		spec := defaultResNet9Spec(N)
		arena := uop.NewArena(65536)
		rng := rand.New(rand.NewSource(1000 + N))
		xLeaf, _ := makeFloatInput(arena, []int64{N, spec.Cin, spec.H, spec.W}, rng, "x")
		direct := buildDirectResNet9(arena, spec, w, xLeaf)
		model := buildModelProtoResNet9(spec, w)
		imported := realizeViaImporter(t, arena, model,
			map[string]*tensor.Tensor{spec.InputName: xLeaf}, "y")
		dData, _, err := cpuEval(direct)
		if err != nil {
			t.Fatalf("N=%d: cpuEval(direct): %v", N, err)
		}
		iData, _, err := cpuEval(imported)
		if err != nil {
			t.Fatalf("N=%d: cpuEval(imported): %v", N, err)
		}
		if len(dData) != len(iData) {
			t.Fatalf("N=%d: length mismatch %d vs %d", N, len(dData), len(iData))
		}
		m := maxAbsDiff(dData, iData)
		t.Logf("ResNet9_MultiBatch N=%d: max-abs-diff = %g (n=%d)", N, m, len(dData))
		if m != 0 {
			t.Errorf("N=%d: expected bit-exact 0, got %g", N, m)
		}
	}
}

// ── Symbolic-batch ResNet-9 ──────────────────────────────────────────────────

// buildModelProtoResNet9Symbolic is the dim_param "N" variant of the
// ResNet-9 ModelProto. The input/output axis 0 carries the symbolic name.
func buildModelProtoResNet9Symbolic(spec resnet9Spec, w *cnnWeights, dimName string) *onnxpb.ModelProto {
	m := buildModelProtoResNet9(spec, w)
	in := m.Graph.Input[0]
	in.Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{
		Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: dimName},
	}
	out := m.Graph.Output[0]
	out.Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{
		Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: dimName},
	}
	return m
}

// TestE2E_ResNet9_SymbolicBatch imports the ResNet-9 model with dim_param
// "N", then runs at N in {1, 4, 8}, asserting bit-exact equality with a
// fresh-arena direct build at each concrete N (matching the pattern in
// e2e_cnn_test.go).
func TestE2E_ResNet9_SymbolicBatch(t *testing.T) {
	const dimName = "N"
	// Weights are shared.
	wRng := rand.New(rand.NewSource(2028))
	w := newCNNWeights()
	registerResNet9Weights(w, wRng, defaultResNet9Spec(1))

	// Import the symbolic model once.
	symArena := uop.NewArena(65536)
	symSpec := defaultResNet9Spec(1)
	symModel := buildModelProtoResNet9Symbolic(symSpec, w, dimName)
	bytesM := mustMarshalProto(t, symModel)
	r, err := Import(bytesM, symArena, "test")
	if err != nil {
		t.Fatalf("symbolic Import: %v", err)
	}
	if len(r.Inputs()) != 1 {
		t.Fatalf("expected 1 graph input, got %d", len(r.Inputs()))
	}
	if _, ok := r.Inputs()[0].Shape[0].ConstValue(); ok {
		t.Errorf("input axis 0 resolved to a const; expected symbolic")
	}
	if _, ok := symArena.FindDefineVar(dimName); !ok {
		t.Fatalf("arena.FindDefineVar(%q) missing", dimName)
	}

	for _, N := range []int64{1, 4, 8} {
		freshArena := uop.NewArena(65536)
		concreteSpec := defaultResNet9Spec(N)
		rng := rand.New(rand.NewSource(5000 + N))
		xShape := []int64{N, concreteSpec.Cin, concreteSpec.H, concreteSpec.W}
		xVals := make([]float32, xShape[0]*xShape[1]*xShape[2]*xShape[3])
		for i := range xVals {
			xVals[i] = rng.Float32()*2 - 1
		}
		freshLeaf := tensor.NewLeaf(freshArena, xShape, uop.Dtypes.Float32, "test")
		freshLeaf.SetData(append([]float32{}, xVals...))
		direct := buildDirectResNet9(freshArena, concreteSpec, w, freshLeaf)

		symLeaf := tensor.NewLeaf(symArena, xShape, uop.Dtypes.Float32, "test")
		symLeaf.SetData(append([]float32{}, xVals...))
		out, err := r.Run(map[string]*tensor.Tensor{concreteSpec.InputName: symLeaf})
		if err != nil {
			t.Fatalf("symbolic Run N=%d: %v", N, err)
		}
		imported, ok := out["y"]
		if !ok {
			t.Fatalf("symbolic Run N=%d: output y missing", N)
		}
		dData, _, derr := cpuEval(direct)
		if derr != nil {
			t.Fatalf("N=%d: cpuEval(direct): %v", N, derr)
		}
		iData, _, ierr := cpuEval(imported)
		if ierr != nil {
			t.Fatalf("N=%d: cpuEval(imported): %v", N, ierr)
		}
		if len(dData) != len(iData) {
			t.Fatalf("N=%d: length mismatch %d vs %d", N, len(dData), len(iData))
		}
		m := maxAbsDiff(dData, iData)
		t.Logf("ResNet9_SymbolicBatch N=%d: max-abs-diff = %g (n=%d)", N, m, len(dData))
		if m != 0 {
			t.Errorf("N=%d: expected bit-exact 0, got %g", N, m)
		}
	}
}
