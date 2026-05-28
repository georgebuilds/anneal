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

func min(a, b int) int {
	if a < b { return a }
	return b
}
