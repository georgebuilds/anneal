package onnx

// Phase 1.C e2e helpers.
//
// The value-oracle gate compares two execution paths over identical weights
// and inputs:
//
//   path A — direct Tensor API: build the model via tensor / nn primitives,
//     get the output tensor.
//   path B — importer: build a ModelProto with the same weights as
//     initializers, the same op attributes, and identical opset; Import +
//     Run; collect the named graph output.
//
// Both paths build UOp graphs on the SAME arena. Identical primitive calls
// (FullSints for ReLU's zero, Maximum, Reshape, etc.) intern to identical
// UOps via the arena's deduplication. Evaluating both via cpuEval should
// produce *bit-exact* equal []float32 outputs. Any drift > 0 is a handler
// bug.
//
// We deliberately reuse the singleNodeBuilder Attr/onnxpb plumbing already in
// the package; this file only adds multi-node model construction.

import (
	"math/rand"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Model spec types ─────────────────────────────────────────────────────────

// tinyCNNSpec describes the TinyCNN architecture used by tests (a) and (d).
//
// Architecture: Conv(C_in→C_mid, 3×3, pad=1) → Relu → GlobalAveragePool →
//               Gemm(C_mid→numClasses).
type tinyCNNSpec struct {
	N         int64
	Cin       int64
	H, W      int64
	Cmid      int64
	NumClass  int64
	UseBias   bool
	InputName string
}

// tinyCNNMaxPoolSpec describes test (c): Conv → Relu → MaxPool(2x2,s=2) →
// Conv → GlobalAveragePool → Gemm.
type tinyCNNMaxPoolSpec struct {
	N           int64
	Cin         int64
	H, W        int64
	Cmid1, Cmid2 int64
	NumClass    int64
	InputName   string
}

// tinyResNetSpec describes test (b): a small residual block followed by a
// linear head. Conv→BN→Relu→Conv→BN→Add(skip)→Relu→GlobalAvgPool→Gemm.
type tinyResNetSpec struct {
	N         int64
	C         int64
	H, W      int64
	NumClass  int64
	InputName string
}

// ── Weight construction ──────────────────────────────────────────────────────

// cnnWeights is the shared weight registry by initializer name. Used by both
// the direct-API builder and the ModelProto builder so they consume *identical*
// bytes for each named parameter.
type cnnWeights struct {
	values map[string][]float32 // by initializer name
	shapes map[string][]int64
}

func newCNNWeights() *cnnWeights {
	return &cnnWeights{
		values: make(map[string][]float32),
		shapes: make(map[string][]int64),
	}
}

func (w *cnnWeights) set(name string, shape []int64, vals []float32) {
	w.values[name] = vals
	w.shapes[name] = shape
}

func (w *cnnWeights) randomFill(rng *rand.Rand, name string, shape []int64, scale float32) {
	n := int64(1)
	for _, s := range shape {
		n *= s
	}
	vals := make([]float32, n)
	for i := range vals {
		// Sample from a centred distribution scaled by `scale` to keep values
		// in a sensible range for downstream stability.
		vals[i] = (rng.Float32()*2 - 1) * scale
	}
	w.set(name, shape, vals)
}

// ── Path A: direct via Tensor API ─────────────────────────────────────────────

// buildDirectTinyCNN builds the TinyCNN forward graph using the tensor/nn API
// directly. Returns the output tensor node so caller can evaluate via cpuEval.
//
// The leaves are allocated on `arena`; weights are seeded from w.
func buildDirectTinyCNN(arena *uop.Arena, spec tinyCNNSpec, w *cnnWeights, x *tensor.Tensor) *tensor.Tensor {
	conv := &nn.Conv2d{
		Weight: leafParam(arena, w, "conv_w"),
		Stride: [2]int{1, 1},
		Pad:    [2]int{1, 1},
	}
	if spec.UseBias {
		conv.Bias = leafParam(arena, w, "conv_b")
	}
	h := conv.Forward(x)
	// Relu = Maximum(x, 0)
	h = nn.ReLU(h)
	// GlobalAveragePool: mean over spatial axes, keepdim=true.
	rank := h.Rank()
	axes := []int{}
	for i := 2; i < rank; i++ {
		axes = append(axes, i)
	}
	h = h.Mean(axes, true)
	// Flatten to [N, Cmid] before Gemm.
	h = h.Reshape([]int64{spec.N, spec.Cmid})
	// Gemm: y = x @ B + C  where B is [Cmid, numClass], transA=0, transB=0.
	bMat := leafParam(arena, w, "gemm_b").T
	out := h.Matmul(bMat)
	if spec.UseBias {
		biasT := leafParam(arena, w, "gemm_c").T
		biasB := tensor.BroadcastToSints(biasT, out.ShapeSints())
		out = out.Add(biasB)
	}
	return out
}

// buildDirectTinyCNNMaxPool builds the (c) variant.
func buildDirectTinyCNNMaxPool(arena *uop.Arena, spec tinyCNNMaxPoolSpec, w *cnnWeights, x *tensor.Tensor) *tensor.Tensor {
	conv1 := &nn.Conv2d{
		Weight: leafParam(arena, w, "conv1_w"),
		Stride: [2]int{1, 1},
		Pad:    [2]int{1, 1},
	}
	h := conv1.Forward(x)
	h = nn.ReLU(h)
	h = nn.MaxPool2D(h, 2, 2, 2, 2)
	conv2 := &nn.Conv2d{
		Weight: leafParam(arena, w, "conv2_w"),
		Stride: [2]int{1, 1},
		Pad:    [2]int{1, 1},
	}
	h = conv2.Forward(h)
	rank := h.Rank()
	axes := []int{}
	for i := 2; i < rank; i++ {
		axes = append(axes, i)
	}
	h = h.Mean(axes, true)
	h = h.Reshape([]int64{spec.N, spec.Cmid2})
	bMat := leafParam(arena, w, "gemm_b").T
	return h.Matmul(bMat)
}

// buildDirectTinyResNet builds the (b) variant: Conv→BN→Relu→Conv→BN→Add(skip)
// →Relu→GlobalAvgPool→Gemm.
func buildDirectTinyResNet(arena *uop.Arena, spec tinyResNetSpec, w *cnnWeights, x *tensor.Tensor) *tensor.Tensor {
	c1 := &nn.Conv2d{
		Weight: leafParam(arena, w, "c1_w"),
		Stride: [2]int{1, 1},
		Pad:    [2]int{1, 1},
	}
	h := c1.Forward(x)
	h = applyBatchNorm(arena, h, w, "bn1", spec.C)
	h = nn.ReLU(h)
	c2 := &nn.Conv2d{
		Weight: leafParam(arena, w, "c2_w"),
		Stride: [2]int{1, 1},
		Pad:    [2]int{1, 1},
	}
	h = c2.Forward(h)
	h = applyBatchNorm(arena, h, w, "bn2", spec.C)
	// Residual add: BN2 output + original x.
	h = h.Add(x)
	h = nn.ReLU(h)
	// Global average pool over spatial dims.
	axes := []int{}
	for i := 2; i < h.Rank(); i++ {
		axes = append(axes, i)
	}
	h = h.Mean(axes, true)
	h = h.Reshape([]int64{spec.N, spec.C})
	bMat := leafParam(arena, w, "gemm_b").T
	return h.Matmul(bMat)
}

// applyBatchNorm constructs the BN inference forward graph that exactly
// matches handleBatchNormalization: y = (x - mean) / sqrt(var + eps) * scale + B.
// All operands use the same Sint shape ([1, C, 1, 1]) so the broadcast pattern
// is identical to the importer path.
func applyBatchNorm(arena *uop.Arena, x *tensor.Tensor, w *cnnWeights, prefix string, c int64) *tensor.Tensor {
	scale := leafParam(arena, w, prefix+"_scale").T
	bias := leafParam(arena, w, prefix+"_bias").T
	mean := leafParam(arena, w, prefix+"_mean").T
	variance := leafParam(arena, w, prefix+"_var").T

	xSh := x.ShapeSints()
	// Reshape per-channel vectors to [1, C, 1, 1] — matches the handler's
	// broadcast layout.
	bc := make([]int64, len(xSh))
	for i := range bc {
		bc[i] = 1
	}
	bc[1] = c
	scale = scale.Reshape(bc)
	bias = bias.Reshape(bc)
	mean = mean.Reshape(bc)
	variance = variance.Reshape(bc)
	// eps tensor on bcShape using FullSints, matching handler.
	const epsilon = 1e-5
	eps := tensor.FullSints(arena, scale.ShapeSints(), epsilon, x.DType(), x.Device())
	denom := variance.Add(eps).Sqrt()
	norm := x.Sub(mean).Div(denom)
	return norm.Mul(scale).Add(bias)
}

// leafParam builds a Parameter whose underlying leaf tensor is on `arena` and
// whose data is the registered value from w. The returned *nn.Parameter is
// suitable for Conv2d.Weight / Conv2d.Bias / Linear.Weight slots.
func leafParam(arena *uop.Arena, w *cnnWeights, name string) *nn.Parameter {
	shape, ok := w.shapes[name]
	if !ok {
		panic("e2e_helper_test: weight not registered: " + name)
	}
	vals := w.values[name]
	leaf := tensor.NewLeaf(arena, shape, uop.Dtypes.Float32, "test")
	leaf.SetData(append([]float32{}, vals...))
	return &nn.Parameter{T: leaf, Name: name, Value: append([]float32{}, vals...)}
}

// ── Path B: ModelProto builders ──────────────────────────────────────────────

// buildModelProtoTinyCNN constructs the importer-fed ModelProto for the
// TinyCNN. Input `xName` is a graph input; weights are initializers seeded
// from w.
func buildModelProtoTinyCNN(spec tinyCNNSpec, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "tinycnn"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{
		makeVI(spec.InputName, onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.Cin, spec.H, spec.W}),
	}
	g.Output = []*onnxpb.ValueInfoProto{
		makeVI("y", onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.NumClass}),
	}
	convOut := "conv_out"
	reluOut := "relu_out"
	poolOut := "pool_out"
	flatOut := "flat_out"
	convInputs := []string{spec.InputName, "conv_w"}
	if spec.UseBias {
		convInputs = append(convInputs, "conv_b")
	}
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: "conv", OpType: "Conv",
		Input:  convInputs,
		Output: []string{convOut},
		Attribute: []*onnxpb.AttributeProto{
			intsAttr("kernel_shape", []int64{3, 3}),
			intsAttr("strides", []int64{1, 1}),
			intsAttr("pads", []int64{1, 1, 1, 1}),
		},
	})
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: "relu", OpType: "Relu",
		Input: []string{convOut}, Output: []string{reluOut},
	})
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: "gap", OpType: "GlobalAveragePool",
		Input: []string{reluOut}, Output: []string{poolOut},
	})
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: "flatten", OpType: "Flatten",
		Input: []string{poolOut}, Output: []string{flatOut},
		Attribute: []*onnxpb.AttributeProto{
			intAttr("axis", 1),
		},
	})
	gemmInputs := []string{flatOut, "gemm_b"}
	if spec.UseBias {
		gemmInputs = append(gemmInputs, "gemm_c")
	}
	g.Node = append(g.Node, &onnxpb.NodeProto{
		Name: "gemm", OpType: "Gemm",
		Input:  gemmInputs,
		Output: []string{"y"},
		// transA=0, transB=0, alpha=1, beta=1 (defaults).
	})
	return &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: g,
	}
}

// buildModelProtoTinyCNNSymbolic is the symbolic-batch variant of TinyCNN.
// Axis 0 of the graph input uses dim_param "N".
func buildModelProtoTinyCNNSymbolic(spec tinyCNNSpec, w *cnnWeights, dimParamName string) *onnxpb.ModelProto {
	m := buildModelProtoTinyCNN(spec, w)
	// Rewrite input dim 0 to symbolic.
	in := m.Graph.Input[0]
	in.Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{
		Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: dimParamName},
	}
	// Also rewrite output dim 0.
	out := m.Graph.Output[0]
	out.Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{
		Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: dimParamName},
	}
	return m
}

// buildModelProtoTinyCNNMaxPool: the (c) variant.
func buildModelProtoTinyCNNMaxPool(spec tinyCNNMaxPoolSpec, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "tinycnn_mp"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{
		makeVI(spec.InputName, onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.Cin, spec.H, spec.W}),
	}
	g.Output = []*onnxpb.ValueInfoProto{
		makeVI("y", onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.NumClass}),
	}
	g.Node = append(g.Node,
		&onnxpb.NodeProto{
			Name: "conv1", OpType: "Conv",
			Input:  []string{spec.InputName, "conv1_w"},
			Output: []string{"c1"},
			Attribute: []*onnxpb.AttributeProto{
				intsAttr("kernel_shape", []int64{3, 3}),
				intsAttr("strides", []int64{1, 1}),
				intsAttr("pads", []int64{1, 1, 1, 1}),
			},
		},
		&onnxpb.NodeProto{
			Name: "relu1", OpType: "Relu",
			Input: []string{"c1"}, Output: []string{"r1"},
		},
		&onnxpb.NodeProto{
			Name: "mp", OpType: "MaxPool",
			Input: []string{"r1"}, Output: []string{"p1"},
			Attribute: []*onnxpb.AttributeProto{
				intsAttr("kernel_shape", []int64{2, 2}),
				intsAttr("strides", []int64{2, 2}),
				intsAttr("pads", []int64{0, 0, 0, 0}),
			},
		},
		&onnxpb.NodeProto{
			Name: "conv2", OpType: "Conv",
			Input:  []string{"p1", "conv2_w"},
			Output: []string{"c2"},
			Attribute: []*onnxpb.AttributeProto{
				intsAttr("kernel_shape", []int64{3, 3}),
				intsAttr("strides", []int64{1, 1}),
				intsAttr("pads", []int64{1, 1, 1, 1}),
			},
		},
		&onnxpb.NodeProto{
			Name: "gap", OpType: "GlobalAveragePool",
			Input: []string{"c2"}, Output: []string{"gap_out"},
		},
		&onnxpb.NodeProto{
			Name: "flatten", OpType: "Flatten",
			Input: []string{"gap_out"}, Output: []string{"flat"},
			Attribute: []*onnxpb.AttributeProto{intAttr("axis", 1)},
		},
		&onnxpb.NodeProto{
			Name: "gemm", OpType: "Gemm",
			Input:  []string{"flat", "gemm_b"},
			Output: []string{"y"},
		},
	)
	return &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: g,
	}
}

// buildModelProtoTinyResNet: the (b) variant.
func buildModelProtoTinyResNet(spec tinyResNetSpec, w *cnnWeights) *onnxpb.ModelProto {
	g := &onnxpb.GraphProto{Name: "tinyresnet"}
	addInitsFromWeights(g, w)
	g.Input = []*onnxpb.ValueInfoProto{
		makeVI(spec.InputName, onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.C, spec.H, spec.W}),
	}
	g.Output = []*onnxpb.ValueInfoProto{
		makeVI("y", onnxpb.TensorProto_FLOAT, []int64{spec.N, spec.NumClass}),
	}
	g.Node = append(g.Node,
		&onnxpb.NodeProto{
			Name: "c1", OpType: "Conv",
			Input:  []string{spec.InputName, "c1_w"},
			Output: []string{"c1_out"},
			Attribute: []*onnxpb.AttributeProto{
				intsAttr("kernel_shape", []int64{3, 3}),
				intsAttr("strides", []int64{1, 1}),
				intsAttr("pads", []int64{1, 1, 1, 1}),
			},
		},
		&onnxpb.NodeProto{
			Name: "bn1", OpType: "BatchNormalization",
			Input:  []string{"c1_out", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"},
			Output: []string{"bn1_out"},
			Attribute: []*onnxpb.AttributeProto{
				floatAttr("epsilon", 1e-5),
			},
		},
		&onnxpb.NodeProto{
			Name: "relu1", OpType: "Relu",
			Input: []string{"bn1_out"}, Output: []string{"r1"},
		},
		&onnxpb.NodeProto{
			Name: "c2", OpType: "Conv",
			Input:  []string{"r1", "c2_w"},
			Output: []string{"c2_out"},
			Attribute: []*onnxpb.AttributeProto{
				intsAttr("kernel_shape", []int64{3, 3}),
				intsAttr("strides", []int64{1, 1}),
				intsAttr("pads", []int64{1, 1, 1, 1}),
			},
		},
		&onnxpb.NodeProto{
			Name: "bn2", OpType: "BatchNormalization",
			Input:  []string{"c2_out", "bn2_scale", "bn2_bias", "bn2_mean", "bn2_var"},
			Output: []string{"bn2_out"},
			Attribute: []*onnxpb.AttributeProto{
				floatAttr("epsilon", 1e-5),
			},
		},
		&onnxpb.NodeProto{
			Name: "add", OpType: "Add",
			Input: []string{"bn2_out", spec.InputName}, Output: []string{"add_out"},
		},
		&onnxpb.NodeProto{
			Name: "relu2", OpType: "Relu",
			Input: []string{"add_out"}, Output: []string{"r2"},
		},
		&onnxpb.NodeProto{
			Name: "gap", OpType: "GlobalAveragePool",
			Input: []string{"r2"}, Output: []string{"gap_out"},
		},
		&onnxpb.NodeProto{
			Name: "flatten", OpType: "Flatten",
			Input: []string{"gap_out"}, Output: []string{"flat"},
			Attribute: []*onnxpb.AttributeProto{intAttr("axis", 1)},
		},
		&onnxpb.NodeProto{
			Name: "gemm", OpType: "Gemm",
			Input:  []string{"flat", "gemm_b"},
			Output: []string{"y"},
		},
	)
	return &onnxpb.ModelProto{
		IrVersion: 8,
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: "", Version: 13},
		},
		Graph: g,
	}
}

// ── Small AttributeProto helpers ─────────────────────────────────────────────

func intsAttr(name string, vs []int64) *onnxpb.AttributeProto {
	cp := append([]int64{}, vs...)
	return &onnxpb.AttributeProto{Name: name, Type: onnxpb.AttributeProto_INTS, Ints: cp}
}

func intAttr(name string, v int64) *onnxpb.AttributeProto {
	return &onnxpb.AttributeProto{Name: name, Type: onnxpb.AttributeProto_INT, I: v}
}

func floatAttr(name string, v float32) *onnxpb.AttributeProto {
	return &onnxpb.AttributeProto{Name: name, Type: onnxpb.AttributeProto_FLOAT, F: v}
}

// addInitsFromWeights translates the registered weights map into TensorProto
// initializers attached to g. The FLOAT data format keeps bytes deterministic
// across runs.
func addInitsFromWeights(g *onnxpb.GraphProto, w *cnnWeights) {
	// Iterate in registration order is not needed for correctness — the
	// runner indexes by name — but stable iteration matters for diff. Go's
	// map iteration is undefined; we iterate but the produced order is
	// irrelevant since interning is by hash.
	for name, vals := range w.values {
		dims := w.shapes[name]
		g.Initializer = append(g.Initializer, &onnxpb.TensorProto{
			Name:      name,
			Dims:      dims,
			DataType:  int32(onnxpb.TensorProto_FLOAT),
			FloatData: append([]float32{}, vals...),
		})
	}
}

// ── Path execution + comparison ──────────────────────────────────────────────

// realizeViaImporter marshals the model, imports it on `arena`, and runs it
// with the named inputs. Returns the named output tensor.
func realizeViaImporter(t *testing.T, arena *uop.Arena, model *onnxpb.ModelProto, inputs map[string]*tensor.Tensor, outName string) *tensor.Tensor {
	t.Helper()
	bytes := mustMarshalProto(t, model)
	r, err := Import(bytes, arena, "test")
	if err != nil {
		t.Fatalf("realizeViaImporter: Import: %v", err)
	}
	out, err := r.Run(inputs)
	if err != nil {
		t.Fatalf("realizeViaImporter: Run: %v", err)
	}
	got, ok := out[outName]
	if !ok {
		t.Fatalf("realizeViaImporter: output %q not in result map (got %v)", outName, keysOf(out))
	}
	return got
}

func keysOf(m map[string]*tensor.Tensor) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// assertBitExact evaluates both tensors via cpuEval and asserts the resulting
// []float32 slices are *bit-exact* equal. The maximum absolute difference is
// always logged (will be 0 on success); on mismatch we log the first 8 element
// pairs.
func assertBitExact(t *testing.T, want, got *tensor.Tensor, label string) {
	t.Helper()
	wData, wSh, err := cpuEval(want)
	if err != nil {
		t.Fatalf("%s: cpuEval(want): %v", label, err)
	}
	gData, gSh, err := cpuEval(got)
	if err != nil {
		t.Fatalf("%s: cpuEval(got): %v", label, err)
	}
	if !shapeEq(wSh, gSh) {
		t.Fatalf("%s: shape mismatch want %v got %v", label, wSh, gSh)
	}
	if len(wData) != len(gData) {
		t.Fatalf("%s: length mismatch want %d got %d", label, len(wData), len(gData))
	}
	var maxAbs float32
	mismatchCount := 0
	for i := range wData {
		d := wData[i] - gData[i]
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
		if wData[i] != gData[i] {
			mismatchCount++
			if mismatchCount <= 8 {
				t.Logf("%s: mismatch at %d: want=%v got=%v diff=%v",
					label, i, wData[i], gData[i], d)
			}
		}
	}
	if mismatchCount > 0 {
		t.Fatalf("%s: %d/%d elements differ; max-abs-diff = %g (expected bit-exact 0)",
			label, mismatchCount, len(wData), maxAbs)
	}
	t.Logf("%s: bit-exact: max-abs-diff = 0 over %d elements (shape=%v)",
		label, len(wData), wSh)
}

// maxAbsDiff returns the maximum absolute element difference for two equal-length slices.
func maxAbsDiff(a, b []float32) float32 {
	var m float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

// makeFloatInput returns a randomly initialised input leaf on `arena` and the
// underlying []float32 (so callers can register the same data on a second
// arena for cross-arena comparisons).
func makeFloatInput(arena *uop.Arena, shape []int64, rng *rand.Rand, name string) (*tensor.Tensor, []float32) {
	n := int64(1)
	for _, s := range shape {
		n *= s
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = rng.Float32()*2 - 1
	}
	leaf := tensor.NewLeaf(arena, shape, uop.Dtypes.Float32, "test")
	leaf.SetData(append([]float32{}, vals...))
	_ = name
	return leaf, vals
}
