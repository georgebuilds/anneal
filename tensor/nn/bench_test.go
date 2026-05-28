package nn_test

import (
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestFusionBenchmark(t *testing.T) {
	requireGPU(t)

	// Workloads
	type workload struct {
		name string
		size string
		run  func(t *testing.T, nSteps int, seed int64) float32
	}

	workloads := []workload{
		{"MLP", "Large (Batch 256, 1024-wide, 3 layers)", runMLPBenchmark},
		{"Conv2d", "Large (Batch 16, 3->64, 64x64)", runConvBenchmark},
	}

	for _, wl := range workloads {
		t.Logf("=== Workload: %s (%s) ===", wl.name, wl.size)

		// Sanity check: verify ON and OFF produce matching results.
		lOff := wl.run(t, 1, 42)
		schedule.FusionEnabled = true
		lOn := wl.run(t, 1, 42)
		schedule.FusionEnabled = false // reset
		
		if absF32(lOn-lOff) > 1e-4 {
			t.Fatalf("Sanity check failed: Fusion ON/OFF mismatch: ON=%.6f OFF=%.6f", lOn, lOff)
		}
		t.Logf("Sanity check PASSED (Loss: %.6f)", lOff)

		results := make(map[bool]struct {
			median  time.Duration
			min     time.Duration
			max     time.Duration
			kernels int
		})

		for _, fusion := range []bool{false, true} {
			schedule.FusionEnabled = fusion
			
			// Warmup: 5 steps to prime caches and allocators.
			wl.run(t, 5, 42)

			const nIter = 10
			times := make([]time.Duration, nIter)
			var lastKernels int
			
			for i := 0; i < nIter; i++ {
				var stepKernels int
				schedule.StatsHook = func(s schedule.CompilerStats) {
					stepKernels += s.Kernels
				}
				
				start := time.Now()
				wl.run(t, 1, int64(i+100))
				times[i] = time.Since(start)
				
				schedule.StatsHook = nil
				lastKernels = stepKernels
			}

			sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
			
			results[fusion] = struct {
				median  time.Duration
				min     time.Duration
				max     time.Duration
				kernels int
			}{
				median:  times[nIter/2],
				min:     times[0],
				max:     times[nIter-1],
				kernels: lastKernels,
			}
		}

		off := results[false]
		on := results[true]

		diff := float64(on.median-off.median) / float64(off.median) * 100
		
		t.Logf("Fusion OFF: Kernels=%d, Median=%v, Min=%v, Max=%v", off.kernels, off.median, off.min, off.max)
		t.Logf("Fusion ON : Kernels=%d, Median=%v, Min=%v, Max=%v", on.kernels, on.median, on.min, on.max)
		
		verdict := "no change"
		if diff < -5 {
			verdict = "FASTER"
		} else if diff > 5 {
			verdict = "SLOWER"
		}
		
		t.Logf("Verdict: %s (%.2f%% change)", verdict, diff)
		t.Logf("--------------------------------------------------")
	}
}

// ── MLP Benchmark ─────────────────────────────────────────────────────────────

type mlpBench struct {
	layers []*nn.Linear
}

func newMLPBench(a *uop.Arena, inSize, hiddenSize, outSize int64, nHidden int) *mlpBench {
	m := &mlpBench{}
	m.layers = append(m.layers, nn.NewLinear(a, inSize, hiddenSize, true, uop.Dtypes.Float32, "webgpu"))
	for i := 0; i < nHidden-1; i++ {
		m.layers = append(m.layers, nn.NewLinear(a, hiddenSize, hiddenSize, true, uop.Dtypes.Float32, "webgpu"))
	}
	m.layers = append(m.layers, nn.NewLinear(a, hiddenSize, outSize, true, uop.Dtypes.Float32, "webgpu"))
	return m
}

func (m *mlpBench) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := x
	for i, l := range m.layers {
		h = l.Forward(h)
		if i < len(m.layers)-1 {
			h = nn.ReLU(h)
		}
	}
	return h
}

func (m *mlpBench) Params() []*nn.Parameter {
	var params []*nn.Parameter
	for _, l := range m.layers {
		params = append(params, l.Params()...)
	}
	return params
}

func runMLPBenchmark(t *testing.T, nSteps int, seed int64) float32 {
	const (
		batch      = 256
		inSize     = 1024
		hiddenSize = 1024
		outSize    = 1024
		nHidden    = 2 // total 3 layers
	)

	a0 := uop.NewArena(1024)
	model := newMLPBench(a0, inSize, hiddenSize, outSize, nHidden)
	
	rng := rand.New(rand.NewSource(seed))
	for _, p := range model.Params() {
		for i := range p.Value {
			p.Value[i] = rng.Float32() * 0.1
		}
	}

	opt := nn.NewSGD(model.Params(), 0.01)

	// Random data
	xData := make([]float32, batch*inSize)
	for i := range xData {
		xData[i] = rng.Float32()
	}
	yData := make([]float32, batch*outSize)
	for i := range yData {
		yData[i] = rng.Float32()
	}

	var lastLoss float32
	for step := 0; step < nSteps; step++ {
		a := uop.NewArena(65536)
		for _, p := range opt.Params {
			p.Load(a)
		}

		x := tensor.NewLeaf(a, []int64{batch, inSize}, uop.Dtypes.Float32, "webgpu")
		x.SetData(xData)
		tgt := tensor.NewLeaf(a, []int64{batch, outSize}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(yData)

		pred := model.Forward(x)
		diff := pred.Sub(tgt)
		loss := diff.Mul(diff).Sum(nil, false)

		leaves := make([]*tensor.Tensor, 0, len(opt.Params))
		for _, p := range opt.Params {
			leaves = append(leaves, p.T)
		}
		grads := tensor.Backward(loss, leaves)

		for _, p := range opt.Params {
			if g, ok := grads[p.T]; ok {
				if err := tensor.Realize(g); err != nil {
					t.Fatalf("Realize grad: %v", err)
				}
			}
		}

		opt.Step(grads)
		
		if step == nSteps-1 {
			if err := tensor.Realize(loss); err != nil {
				t.Fatalf("Realize loss: %v", err)
			}
			lastLoss = loss.Data()[0]
		}
	}
	return lastLoss
}

// ── Conv Benchmark ────────────────────────────────────────────────────────────

func runConvBenchmark(t *testing.T, nSteps int, seed int64) float32 {
	const (
		batch = 16
		cin   = 3
		imgH  = 64
		imgW  = 64
		cout  = 64
	)

	a0 := uop.NewArena(1024)
	model := nn.NewConv2d(a0, cin, cout, [2]int64{3, 3}, [2]int{1, 1}, [2]int{0, 0}, true, uop.Dtypes.Float32, "webgpu")
	
	rng := rand.New(rand.NewSource(seed))
	for _, p := range model.Params() {
		for i := range p.Value {
			p.Value[i] = rng.Float32() * 0.1
		}
	}

	opt := nn.NewSGD(model.Params(), 0.01)

	// Random data
	xData := make([]float32, batch*cin*imgH*imgW)
	for i := range xData {
		xData[i] = rng.Float32()
	}
	// Target: random vector
	yData := make([]float32, batch)
	for i := range yData {
		yData[i] = rng.Float32()
	}

	var lastLoss float32
	for step := 0; step < nSteps; step++ {
		a := uop.NewArena(65536)
		for _, p := range opt.Params {
			p.Load(a)
		}

		x := tensor.NewLeaf(a, []int64{batch, cin, imgH, imgW}, uop.Dtypes.Float32, "webgpu")
		x.SetData(xData)
		tgt := tensor.NewLeaf(a, []int64{batch, 1}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(yData)

		pred := model.Forward(x)
		// Reduce pred to [batch, 1] for loss
		lossInput := pred.Sum([]int{1, 2, 3}, false).Reshape([]int64{batch, 1})
		diff := lossInput.Sub(tgt)
		loss := diff.Mul(diff).Sum(nil, false)

		leaves := make([]*tensor.Tensor, 0, len(opt.Params))
		for _, p := range opt.Params {
			leaves = append(leaves, p.T)
		}
		grads := tensor.Backward(loss, leaves)

		for _, p := range opt.Params {
			if g, ok := grads[p.T]; ok {
				if err := tensor.Realize(g); err != nil {
					t.Fatalf("Realize grad: %v", err)
				}
			}
		}

		opt.Step(grads)
		
		if step == nSteps-1 {
			if err := tensor.Realize(loss); err != nil {
				t.Fatalf("Realize loss: %v", err)
			}
			lastLoss = loss.Data()[0]
		}
	}
	return lastLoss
}

