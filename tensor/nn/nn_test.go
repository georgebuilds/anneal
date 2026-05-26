package nn_test

import (
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func newArena() *uop.Arena { return uop.NewArena(1024) }

func shapeEq(t *testing.T, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape len %d != %d; got %v want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape[%d]: got %d want %d", i, got[i], want[i])
		}
	}
}

// ── Parameter ─────────────────────────────────────────────────────────────────

func TestParameter(t *testing.T) {
	a := newArena()
	p := nn.NewParameter(a, []int64{4, 8}, uop.Dtypes.Float32, "cpu")
	if !p.T.IsLeaf() {
		t.Fatal("parameter should be a leaf")
	}
	shapeEq(t, p.T.Shape(), []int64{4, 8})
}

// ── Activations ───────────────────────────────────────────────────────────────

func TestReLU(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	y := nn.ReLU(x)
	// ReLU = max(0, x) → root is Maximum (OpMax).
	if y.Node().Op() != uop.OpMax {
		t.Fatalf("ReLU root should be OpMax, got %s", y.Node().Op())
	}
	shapeEq(t, y.Shape(), []int64{3, 4})
}

func TestSigmoid(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{5}, uop.Dtypes.Float32, "cpu")
	y := nn.Sigmoid(x)
	shapeEq(t, y.Shape(), []int64{5})
	if !y.Node().Valid() {
		t.Fatal("sigmoid node should be valid")
	}
}

func TestTanh(t *testing.T) {
	a := newArena()
	x := tensor.NewLeaf(a, []int64{5}, uop.Dtypes.Float32, "cpu")
	y := nn.Tanh(x)
	shapeEq(t, y.Shape(), []int64{5})
}

// ── Linear ────────────────────────────────────────────────────────────────────

func TestLinearNoBias(t *testing.T) {
	a := newArena()
	l := nn.NewLinear(a, 8, 4, false, uop.Dtypes.Float32, "cpu")

	x := tensor.NewLeaf(a, []int64{3, 8}, uop.Dtypes.Float32, "cpu")
	y := l.Forward(x)

	shapeEq(t, y.Shape(), []int64{3, 4})
	if !y.Node().Valid() {
		t.Fatal("linear output node must be valid")
	}
}

func TestLinearWithBias(t *testing.T) {
	a := newArena()
	l := nn.NewLinear(a, 8, 4, true, uop.Dtypes.Float32, "cpu")

	x := tensor.NewLeaf(a, []int64{3, 8}, uop.Dtypes.Float32, "cpu")
	y := l.Forward(x)

	shapeEq(t, y.Shape(), []int64{3, 4})
}

func TestLinearParams(t *testing.T) {
	a := newArena()
	l := nn.NewLinear(a, 8, 4, true, uop.Dtypes.Float32, "cpu")
	if len(l.Params()) != 2 {
		t.Fatalf("linear with bias should have 2 params, got %d", len(l.Params()))
	}
	l2 := nn.NewLinear(a, 8, 4, false, uop.Dtypes.Float32, "cpu")
	if len(l2.Params()) != 1 {
		t.Fatalf("linear without bias should have 1 param, got %d", len(l2.Params()))
	}
}

// ── MLP forward ───────────────────────────────────────────────────────────────

func TestMLPForward(t *testing.T) {
	a := newArena()

	l1 := nn.NewLinear(a, 16, 8, true, uop.Dtypes.Float32, "cpu")
	l2 := nn.NewLinear(a, 8, 4, true, uop.Dtypes.Float32, "cpu")
	l3 := nn.NewLinear(a, 4, 2, false, uop.Dtypes.Float32, "cpu")

	x := tensor.NewLeaf(a, []int64{5, 16}, uop.Dtypes.Float32, "cpu")

	h1 := nn.ReLU(l1.Forward(x))
	h2 := nn.ReLU(l2.Forward(h1))
	out := l3.Forward(h2)

	shapeEq(t, out.Shape(), []int64{5, 2})
	if !out.Node().Valid() {
		t.Fatal("MLP output node must be valid")
	}

	// Verify no eager computation occurred — node types are graph ops, not data.
	if out.Node().Op() == uop.OpConst || out.Node().Op() == uop.OpBuffer {
		t.Fatal("MLP output should be a computed graph node, not a leaf")
	}
}

func TestMLPRealizes(t *testing.T) {
	a := newArena()
	l := nn.NewLinear(a, 4, 2, false, uop.Dtypes.Float32, "cpu")
	x := tensor.NewLeaf(a, []int64{3, 4}, uop.Dtypes.Float32, "cpu")
	out := l.Forward(x)

	err := tensor.Realize(out)
	if err == nil {
		t.Fatal("Realize should return an error (scheduler not implemented)")
	}
}

// ── Conv2d ────────────────────────────────────────────────────────────────────

func TestConv2dForward(t *testing.T) {
	a := newArena()
	// 3-channel input, 8-channel output, 3x3 kernel, no padding.
	conv := nn.NewConv2d(a, 3, 8, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "cpu")

	x := tensor.NewLeaf(a, []int64{1, 3, 8, 8}, uop.Dtypes.Float32, "cpu")
	out := conv.Forward(x)

	// Ho = Wo = 8 - 3 + 1 = 6
	shapeEq(t, out.Shape(), []int64{1, 8, 6, 6})
	if !out.Node().Valid() {
		t.Fatal("conv output must be valid")
	}
}

func TestConv2dWithBias(t *testing.T) {
	a := newArena()
	conv := nn.NewConv2d(a, 1, 4, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, uop.Dtypes.Float32, "cpu")

	// With padding=1, 3x3 kernel: Ho = Wo = (8+2-3)+1 = 8
	x := tensor.NewLeaf(a, []int64{2, 1, 8, 8}, uop.Dtypes.Float32, "cpu")
	out := conv.Forward(x)
	shapeEq(t, out.Shape(), []int64{2, 4, 8, 8})
}

func TestConv2dParams(t *testing.T) {
	a := newArena()
	conv := nn.NewConv2d(a, 3, 8, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, true, uop.Dtypes.Float32, "cpu")
	if len(conv.Params()) != 2 {
		t.Fatalf("conv with bias should have 2 params, got %d", len(conv.Params()))
	}
}

// ── Small conv net ────────────────────────────────────────────────────────────

func TestSmallConvNet(t *testing.T) {
	a := newArena()

	// 2 conv layers + 1 linear head
	conv1 := nn.NewConv2d(a, 1, 4, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, true, uop.Dtypes.Float32, "cpu")
	conv2 := nn.NewConv2d(a, 4, 8, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, "cpu")
	fc := nn.NewLinear(a, 8*2*2, 10, true, uop.Dtypes.Float32, "cpu")

	// Input: [N=1, C=1, H=8, W=8]
	x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "cpu")

	// conv1: [1,1,8,8] → [1,4,6,6]
	h1 := nn.ReLU(conv1.Forward(x))
	shapeEq(t, h1.Shape(), []int64{1, 4, 6, 6})

	// conv2: [1,4,6,6] → [1,8,4,4]
	h2 := nn.ReLU(conv2.Forward(h1))
	shapeEq(t, h2.Shape(), []int64{1, 8, 4, 4})

	// flatten: [1, 8*4*4=128] — but fc expects 8*2*2=32; use a 2x2 sub-region
	// to keep the test self-contained without a pool layer.
	h3 := h2.Shrink([][2]int64{{0, 1}, {0, 8}, {0, 2}, {0, 2}}).Flatten()
	shapeEq(t, h3.Shape(), []int64{32})
	h3 = h3.Reshape([]int64{1, 32})

	out := fc.Forward(h3)
	shapeEq(t, out.Shape(), []int64{1, 10})
	if !out.Node().Valid() {
		t.Fatal("conv net output must be valid")
	}
}
