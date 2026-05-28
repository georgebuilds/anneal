package schedule_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestFusionOracle(t *testing.T) {
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer dev.Close()
	tensor.DefaultExecutor = dev

	cases := []struct {
		name string
		build func(a *uop.Arena) *tensor.Tensor
	}{
		{
			name: "matmul + bias add",
			build: func(a *uop.Arena) *tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "webgpu")
				w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "webgpu")
				b := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float32, "webgpu")
				fillTensor(x, 1)
				fillTensor(w, 2)
				fillTensor(b, 3)
				return x.Matmul(w).Add(b)
			},
		},
		{
			name: "matmul + bias + relu",
			build: func(a *uop.Arena) *tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "webgpu")
				w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "webgpu")
				b := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float32, "webgpu")
				fillTensor(x, 4)
				fillTensor(w, 5)
				fillTensor(b, 6)
				return nn.ReLU(x.Matmul(w).Add(b))
			},
		},
		{
			name: "MLP forward",
			build: func(a *uop.Arena) *tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "webgpu")
				l1 := nn.NewLinear(a, 32, 64, true, uop.Dtypes.Float32, "webgpu")
				l2 := nn.NewLinear(a, 64, 10, true, uop.Dtypes.Float32, "webgpu")
				fillTensor(x, 7)
				fillParam(l1.Weight, 8)
				fillParam(l1.Bias, 9)
				fillParam(l2.Weight, 10)
				fillParam(l2.Bias, 11)
				l1.Weight.Load(a)
				l1.Bias.Load(a)
				l2.Weight.Load(a)
				l2.Bias.Load(a)
				return l2.Forward(nn.ReLU(l1.Forward(x)))
			},
		},
		{
			name: "conv forward",
			build: func(a *uop.Arena) *tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{1, 3, 32, 32}, uop.Dtypes.Float32, "webgpu")
				conv := nn.NewConv2d(a, 3, 16, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, uop.Dtypes.Float32, "webgpu")
				fillTensor(x, 12)
				fillParam(conv.Weight, 13)
				fillParam(conv.Bias, 14)
				conv.Weight.Load(a)
				conv.Bias.Load(a)
				return conv.Forward(x)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Run with fusion OFF
			schedule.FusionEnabled = false
			aOff := uop.NewArena(1024 * 1024)
			outOff := tc.build(aOff)
			if err := tensor.Realize(outOff); err != nil {
				t.Fatalf("Fusion OFF: Realize failed: %v", err)
			}
			dataOff := append([]float32{}, outOff.Data()...)

			// 2. Run with fusion ON
			schedule.FusionEnabled = true
			aOn := uop.NewArena(1024 * 1024)
			outOn := tc.build(aOn)
			if err := tensor.Realize(outOn); err != nil {
				t.Fatalf("Fusion ON: Realize failed: %v", err)
			}
			dataOn := outOn.Data()

			// 3. Compare
			if len(dataOff) != len(dataOn) {
				t.Fatalf("Output length mismatch: off=%d on=%d", len(dataOff), len(dataOn))
			}

			maxDiff := float32(0)
			for i := range dataOff {
				diff := float32(math.Abs(float64(dataOff[i] - dataOn[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			fmt.Printf("Graph %s: max-abs-diff = %e\n", tc.name, maxDiff)
			if maxDiff > 0 {
				t.Errorf("Graph %s: results differ! max-abs-diff = %e", tc.name, maxDiff)
				// If it fails, print some sample data to help debug (MLP case)
				if len(dataOn) > 0 {
					t.Logf("Fusion ON first 5 values: %v", dataOn[:min(5, len(dataOn))])
					t.Logf("Fusion OFF first 5 values: %v", dataOff[:min(5, len(dataOff))])
				}
			}
		})
	}
}

func fillTensor(t *tensor.Tensor, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	sh := t.Shape()
	size := int64(1)
	for _, s := range sh {
		size *= s
	}
	data := make([]float32, int(size))
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	t.SetData(data)
}

func fillParam(p *nn.Parameter, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	for i := range p.Value {
		p.Value[i] = rng.Float32()*2 - 1
	}
}

func TestFusionOracleGradients(t *testing.T) {
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer dev.Close()
	tensor.DefaultExecutor = dev

	xData, yData := toyDataset()

	// 1. Run with fusion OFF
	schedule.FusionEnabled = false
	aOff := uop.NewArena(1024 * 1024)
	mOff := newMLP(aOff, 2, 8, 1, uop.Dtypes.Float32)
	// Initialize with fixed values for reproducibility
	fillParam(mOff.l1.Weight, 42)
	fillParam(mOff.l1.Bias, 43)
	fillParam(mOff.l2.Weight, 44)
	fillParam(mOff.l2.Bias, 45)
	
	gradsOff := getMLPGradients(t, aOff, mOff, xData, yData)

	// 2. Run with fusion ON
	schedule.FusionEnabled = true
	aOn := uop.NewArena(1024 * 1024)
	mOn := newMLP(aOn, 2, 8, 1, uop.Dtypes.Float32)
	fillParam(mOn.l1.Weight, 42)
	fillParam(mOn.l1.Bias, 43)
	fillParam(mOn.l2.Weight, 44)
	fillParam(mOn.l2.Bias, 45)
	
	gradsOn := getMLPGradients(t, aOn, mOn, xData, yData)

	// 3. Compare
	params := []string{"l1.Weight", "l1.Bias", "l2.Weight", "l2.Bias"}
	for _, p := range params {
		off := gradsOff[p]
		on := gradsOn[p]
		if len(off) != len(on) {
			t.Fatalf("%s: length mismatch: off=%d on=%d", p, len(off), len(on))
		}
		maxDiff := float32(0)
		for i := range off {
			diff := float32(math.Abs(float64(off[i] - on[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		fmt.Printf("Gradient %s: max-abs-diff = %e\n", p, maxDiff)
		if maxDiff > 0 {
			t.Errorf("Gradient %s: results differ! max-abs-diff = %e", p, maxDiff)
		}
	}

	// 3. Conv2d backward
	// 3a. Fusion OFF
	schedule.FusionEnabled = false
	aConvOff := uop.NewArena(1024 * 1024)
	convOff := nn.NewConv2d(aConvOff, 3, 16, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, uop.Dtypes.Float32, "webgpu")
	fillParam(convOff.Weight, 46)
	fillParam(convOff.Bias, 47)
	gradsConvOff := getConvGradients(t, aConvOff, convOff)

	// 3b. Fusion ON
	schedule.FusionEnabled = true
	aConvOn := uop.NewArena(1024 * 1024)
	convOn := nn.NewConv2d(aConvOn, 3, 16, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, uop.Dtypes.Float32, "webgpu")
	fillParam(convOn.Weight, 46)
	fillParam(convOn.Bias, 47)
	gradsConvOn := getConvGradients(t, aConvOn, convOn)

	// 3c. Compare Conv
	paramsConv := []string{"conv.Weight", "conv.Bias"}
	for _, p := range paramsConv {
		off := gradsConvOff[p]
		on := gradsConvOn[p]
		maxDiff := float32(0)
		for i := range off {
			diff := float32(math.Abs(float64(off[i] - on[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		fmt.Printf("Gradient %s: max-abs-diff = %e\n", p, maxDiff)
		if maxDiff > 0 {
			t.Errorf("Gradient %s: results differ! max-abs-diff = %e", p, maxDiff)
		}
	}
}

func getConvGradients(t *testing.T, a *uop.Arena, conv *nn.Conv2d) map[string][]float32 {
	t.Helper()
	conv.Weight.Load(a)
	conv.Bias.Load(a)
	x := tensor.NewLeaf(a, []int64{1, 3, 32, 32}, uop.Dtypes.Float32, "webgpu")
	fillTensor(x, 48)

	out := conv.Forward(x)
	loss := out.Mul(out).Sum(nil, false)

	leaves := []*tensor.Tensor{conv.Weight.T, conv.Bias.T}
	grads := tensor.Backward(loss, leaves)

	results := make(map[string][]float32)
	labels := []string{"conv.Weight", "conv.Bias"}
	for i, l := range leaves {
		g := grads[l]
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("Realize conv gradient %s failed: %v", labels[i], err)
		}
		results[labels[i]] = append([]float32{}, g.Data()...)
	}
	return results
}

func getMLPGradients(t *testing.T, a *uop.Arena, m *mlp, xData, yData []float32) map[string][]float32 {
	t.Helper()
	for _, p := range m.mlpParams() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{16, 2}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{16, 1}, uop.Dtypes.Float32, "webgpu")
	tgt.SetData(append([]float32{}, yData...))

	pred := m.Forward(x)
	diff := pred.Sub(tgt)
	loss := diff.Mul(diff).Sum(nil, false)

	paramTensors := m.mlpParams()
	leaves := make([]*tensor.Tensor, len(paramTensors))
	for i, p := range paramTensors {
		leaves[i] = p.T
	}
	grads := tensor.Backward(loss, leaves)

	results := make(map[string][]float32)
	labels := []string{"l1.Weight", "l1.Bias", "l2.Weight", "l2.Bias"}
	for i, p := range paramTensors {
		g := grads[p.T]
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("Realize gradient %s failed: %v", labels[i], err)
		}
		results[labels[i]] = append([]float32{}, g.Data()...)
	}
	return results
}

// Reuse toyDataset and MLP structures from nn_test
type mlp struct {
	l1 *nn.Linear
	l2 *nn.Linear
}

func newMLP(a *uop.Arena, inSize, hiddenSize, outSize int64, dtype *uop.DType) *mlp {
	return &mlp{
		l1: nn.NewLinear(a, inSize, hiddenSize, true, dtype, "webgpu"),
		l2: nn.NewLinear(a, hiddenSize, outSize, true, dtype, "webgpu"),
	}
}

func (m *mlp) mlpParams() []*nn.Parameter {
	return append(m.l1.Params(), m.l2.Params()...)
}

func (m *mlp) Forward(x *tensor.Tensor) *tensor.Tensor {
	return m.l2.Forward(nn.ReLU(m.l1.Forward(x)))
}

func toyDataset() (xData, yData []float32) {
	pts := []float32{-0.75, -0.25, 0.25, 0.75}
	for _, x1 := range pts {
		for _, x2 := range pts {
			xData = append(xData, x1, x2)
			yData = append(yData, x1*x1+x2*x2)
		}
	}
	return
}
