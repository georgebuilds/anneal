package examples

import (
	"math/rand"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "dynmlp",
		Summary: "2→8→1 MLP with symbolic batch dim; trains on y=x₁²+x₂² (same task as mlp)",
		Build:   buildDynMLP,
		Train:   trainDynMLP,
	})
}

const (
	dynmlpHidden   = int64(8)
	dynmlpBatchMin = int64(1)
	dynmlpBatchMax = int64(1024)
	dynmlpVarName  = "n"
	dynmlpDefault  = int64(16)
)

func buildDynMLP(device string) (*BuildResult, error) {
	seedArena := uop.NewArena(64)
	l1Seed := nn.NewLinear(seedArena, 2, dynmlpHidden, true, uop.Dtypes.Float32, device)
	l2Seed := nn.NewLinear(seedArena, dynmlpHidden, 1, true, uop.Dtypes.Float32, device)
	rng := rand.New(rand.NewSource(42))
	heInit(l1Seed.Weight, 2, rng)
	heInit(l2Seed.Weight, int(dynmlpHidden), rng)

	a := uop.NewArena(65536)
	l1 := nn.NewLinear(a, 2, dynmlpHidden, true, uop.Dtypes.Float32, device)
	l2 := nn.NewLinear(a, dynmlpHidden, 1, true, uop.Dtypes.Float32, device)
	copyParam(l1.Weight, l1Seed.Weight)
	copyParam(l1.Bias, l1Seed.Bias)
	copyParam(l2.Weight, l2Seed.Weight)
	copyParam(l2.Bias, l2Seed.Bias)
	l1.Weight.Load(a)
	l1.Bias.Load(a)
	l2.Weight.Load(a)
	l2.Bias.Load(a)

	xData, _ := toyDataset()
	x := tensor.NewSymbolicBatchInput(a, dynmlpVarName, dynmlpBatchMin, dynmlpBatchMax, []int64{2}, uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, xData...))

	h := nn.ReLU(l1.Forward(x))
	out := l2.Forward(h)

	return &BuildResult{
		Arena:  a,
		Output: out,
		Device: device,
		Leaves: []*tensor.Tensor{l1.Weight.T, l1.Bias.T, l2.Weight.T, l2.Bias.T},
	}, nil
}

func trainDynMLP(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	batch := cfg.Batch
	if batch <= 0 {
		batch = dynmlpDefault
	}

	seedArena := uop.NewArena(64)
	l1Seed := nn.NewLinear(seedArena, 2, dynmlpHidden, true, uop.Dtypes.Float32, device)
	l2Seed := nn.NewLinear(seedArena, dynmlpHidden, 1, true, uop.Dtypes.Float32, device)
	rng := rand.New(rand.NewSource(42))
	heInit(l1Seed.Weight, 2, rng)
	heInit(l2Seed.Weight, int(dynmlpHidden), rng)

	a0 := uop.NewArena(64)
	model := struct {
		l1 *nn.Linear
		l2 *nn.Linear
	}{
		l1: nn.NewLinear(a0, 2, dynmlpHidden, true, uop.Dtypes.Float32, device),
		l2: nn.NewLinear(a0, dynmlpHidden, 1, true, uop.Dtypes.Float32, device),
	}
	copyParam(model.l1.Weight, l1Seed.Weight)
	copyParam(model.l1.Bias, l1Seed.Bias)
	copyParam(model.l2.Weight, l2Seed.Weight)
	copyParam(model.l2.Bias, l2Seed.Bias)

	params := append(model.l1.Params(), model.l2.Params()...)
	opt := nn.NewSGD(params, cfg.LR)

	xFull, yFull := toyDataset()
	xBatch, yBatch := dynBatchSlice(xFull, yFull, batch)
	binding := map[string]int64{dynmlpVarName: batch}

	forward := func(x *tensor.Tensor) *tensor.Tensor {
		return model.l2.Forward(nn.ReLU(model.l1.Forward(x)))
	}

	newInputs := func(a *uop.Arena) (x, tgt *tensor.Tensor) {
		x = tensor.NewSymbolicBatchInput(a, dynmlpVarName, dynmlpBatchMin, dynmlpBatchMax, []int64{2}, uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, xBatch...))
		tgt = tensor.NewSymbolicBatchInput(a, dynmlpVarName, dynmlpBatchMin, dynmlpBatchMax, []int64{1}, uop.Dtypes.Float32, device)
		tgt.SetData(append([]float32{}, yBatch...))
		return
	}

	evalLoss := func() float32 {
		a := uop.NewArena(65536)
		for _, p := range params {
			p.Load(a)
		}
		x, tgt := newInputs(a)
		diff := forward(x).Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(batch), uop.Dtypes.Float32, device)
		loss := diff.Mul(diff).Sum(nil, false).Mul(scale)
		if err := tensor.RealizeWithBinding(binding, loss); err != nil {
			return 0
		}
		return loss.Data()[0]
	}

	if cfg.LogEvery > 0 {
		logFn(0, evalLoss())
	}

	for step := 1; step <= cfg.Steps; step++ {
		a := uop.NewArena(65536)
		for _, p := range opt.Params {
			p.Load(a)
		}

		x, tgt := newInputs(a)
		pred := forward(x)
		diff := pred.Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(batch), uop.Dtypes.Float32, device)
		loss := diff.Mul(diff).Sum(nil, false).Mul(scale)

		leaves := make([]*tensor.Tensor, len(opt.Params))
		for i, p := range opt.Params {
			leaves[i] = p.T
		}
		grads := tensor.Backward(loss, leaves)

		for _, p := range opt.Params {
			g, ok := grads[p.T]
			if !ok {
				continue
			}
			if err := tensor.RealizeWithBinding(binding, g); err != nil {
				return err
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			logFn(step, evalLoss())
		}
	}

	return nil
}

// dynBatchSlice returns x (n×2) and y (n×1 flat) data for the requested batch
// size by cycling through the 16-sample toyDataset.
func dynBatchSlice(xFull, yFull []float32, batch int64) (xOut, yOut []float32) {
	n := int(batch)
	base := len(yFull)
	xOut = make([]float32, n*2)
	yOut = make([]float32, n)
	for i := 0; i < n; i++ {
		j := i % base
		xOut[i*2] = xFull[j*2]
		xOut[i*2+1] = xFull[j*2+1]
		yOut[i] = yFull[j]
	}
	return
}
