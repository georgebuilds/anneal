package examples

import (
	"math"
	"math/rand"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "mlp",
		Summary: "2→8→1 multilayer perceptron; trains on y=x₁²+x₂²",
		Build:   buildMLP,
		Train:   trainMLP,
	})
}

const (
	mlpBatch  = int64(16)
	mlpHidden = int64(8)
)

// buildMLP constructs the forward graph for a 2→8→1 MLP and returns a BuildResult.
func buildMLP(device string) (*BuildResult, error) {
	// Seed arena: allocate parameter shapes, apply He init.
	seedArena := uop.NewArena(64)
	l1Seed := nn.NewLinear(seedArena, 2, mlpHidden, true, uop.Dtypes.Float32, device)
	l2Seed := nn.NewLinear(seedArena, mlpHidden, 1, true, uop.Dtypes.Float32, device)

	rng := rand.New(rand.NewSource(42))
	heInit(l1Seed.Weight, 2, rng)
	heInit(l2Seed.Weight, int(mlpHidden), rng)

	// Compute arena: load all params into fresh graph.
	a := uop.NewArena(65536)
	l1 := nn.NewLinear(a, 2, mlpHidden, true, uop.Dtypes.Float32, device)
	l2 := nn.NewLinear(a, mlpHidden, 1, true, uop.Dtypes.Float32, device)

	// Copy initialized values from seed layers.
	copyParam(l1.Weight, l1Seed.Weight)
	copyParam(l1.Bias, l1Seed.Bias)
	copyParam(l2.Weight, l2Seed.Weight)
	copyParam(l2.Bias, l2Seed.Bias)

	// Load params into compute arena.
	l1.Weight.Load(a)
	l1.Bias.Load(a)
	l2.Weight.Load(a)
	l2.Bias.Load(a)

	// Input tensor with toy dataset.
	xData, _ := toyDataset()
	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, xData...))

	// Forward pass: ReLU(l1.Forward(x)) → l2.Forward(...)
	h := nn.ReLU(l1.Forward(x))
	out := l2.Forward(h)

	return &BuildResult{
		Arena:  a,
		Output: out,
		Device: device,
		Leaves: []*tensor.Tensor{l1.Weight.T, l1.Bias.T, l2.Weight.T, l2.Bias.T},
	}, nil
}

// trainMLP runs the MLP training loop.
func trainMLP(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	// Seed arena for parameter initialization.
	seedArena := uop.NewArena(64)
	l1Seed := nn.NewLinear(seedArena, 2, mlpHidden, true, uop.Dtypes.Float32, device)
	l2Seed := nn.NewLinear(seedArena, mlpHidden, 1, true, uop.Dtypes.Float32, device)

	rng := rand.New(rand.NewSource(42))
	heInit(l1Seed.Weight, 2, rng)
	heInit(l2Seed.Weight, int(mlpHidden), rng)

	// Build persistent parameters (values survive arena resets).
	a0 := uop.NewArena(64)
	model := struct {
		l1 *nn.Linear
		l2 *nn.Linear
	}{
		l1: nn.NewLinear(a0, 2, mlpHidden, true, uop.Dtypes.Float32, device),
		l2: nn.NewLinear(a0, mlpHidden, 1, true, uop.Dtypes.Float32, device),
	}
	copyParam(model.l1.Weight, l1Seed.Weight)
	copyParam(model.l1.Bias, l1Seed.Bias)
	copyParam(model.l2.Weight, l2Seed.Weight)
	copyParam(model.l2.Bias, l2Seed.Bias)

	params := append(model.l1.Params(), model.l2.Params()...)
	opt := nn.NewSGD(params, cfg.LR)

	xData, yData := toyDataset()

	mlpForward := func(x *tensor.Tensor) *tensor.Tensor {
		return model.l2.Forward(nn.ReLU(model.l1.Forward(x)))
	}

	// Log initial loss.
	if cfg.LogEvery > 0 {
		l0 := evalMLPLoss(mlpForward, params, xData, yData, device)
		logFn(0, l0)
	}

	for step := 1; step <= cfg.Steps; step++ {
		a := uop.NewArena(65536)
		for _, p := range opt.Params {
			p.Load(a)
		}

		x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, xData...))
		tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float32, device)
		tgt.SetData(append([]float32{}, yData...))

		pred := mlpForward(x)
		diff := pred.Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(mlpBatch), uop.Dtypes.Float32, device)
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
			if err := tensor.Realize(g); err != nil {
				return err
			}
		}
		opt.Step(grads)

		if cfg.OnStep != nil {
			cfg.OnStep(step)
		}

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			l := evalMLPLoss(mlpForward, params, xData, yData, device)
			logFn(step, l)
		}
	}

	return nil
}

// evalMLPLoss runs a forward-only pass and returns the MSE-mean loss.
func evalMLPLoss(
	forward func(*tensor.Tensor) *tensor.Tensor,
	params []*nn.Parameter,
	xData, yData []float32,
	device string,
) float32 {
	a := uop.NewArena(65536)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{mlpBatch, 2}, uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, xData...))
	tgt := tensor.NewLeaf(a, []int64{mlpBatch, 1}, uop.Dtypes.Float32, device)
	tgt.SetData(append([]float32{}, yData...))
	diff := forward(x).Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(mlpBatch), uop.Dtypes.Float32, device)
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)
	if err := tensor.Realize(loss); err != nil {
		return 0
	}
	return loss.Data()[0]
}

// toyDataset returns 16 fixed samples for y = x₁² + x₂².
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

// heInit initializes a parameter with He initialization.
func heInit(p *nn.Parameter, fanIn int, rng *rand.Rand) {
	std := float32(math.Sqrt(2.0 / float64(fanIn)))
	for i := range p.Value {
		p.Value[i] = float32(rng.NormFloat64()) * std
	}
}

// copyParam copies the Value slice from src to dst.
func copyParam(dst, src *nn.Parameter) {
	if dst == nil || src == nil {
		return
	}
	copy(dst.Value, src.Value)
}
