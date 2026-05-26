package examples

import (
	"math/rand"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func init() {
	Register(&Example{
		Name:    "conv",
		Summary: "conv2d(1→4, 3×3)+relu+shrink+flatten+linear; trains on synthetic spatial task",
		Build:   buildConv,
		Train:   trainConv,
	})
}

// Conv net architecture constants (match tensor/nn/conv_test.go).
const (
	convBatch   = int64(8)
	convCin     = int64(1)
	convImgH    = int64(6)
	convImgW    = int64(6)
	convCout    = int64(4)
	convOutH    = int64(4) // 6 - 3 + 1 = 4
	convOutW    = int64(4)
	convCropH   = int64(2)
	convCropW   = int64(2)
	convFlatLen = convCout * convCropH * convCropW // 16
)

type convNetModel struct {
	conv *nn.Conv2d
	fc   *nn.Linear
}

func newConvNetModel(a *uop.Arena, device string) *convNetModel {
	return &convNetModel{
		conv: nn.NewConv2d(a, convCin, convCout, [2]int64{3, 3},
			[2]int{1, 1}, [2]int{0, 0}, false, uop.Dtypes.Float32, device),
		fc: nn.NewLinear(a, convFlatLen, 1, true, uop.Dtypes.Float32, device),
	}
}

func (m *convNetModel) convParams() []*nn.Parameter {
	return append(m.conv.Params(), m.fc.Params()...)
}

func (m *convNetModel) forward(x *tensor.Tensor) *tensor.Tensor {
	N := x.Shape()[0]
	h := nn.ReLU(m.conv.Forward(x))
	h = h.Shrink([][2]int64{
		{0, N}, {0, convCout}, {0, convCropH}, {0, convCropW},
	})
	h = h.Reshape([]int64{N, convFlatLen})
	return m.fc.Forward(h)
}

// buildConv constructs the forward graph for the conv net.
func buildConv(device string) (*BuildResult, error) {
	// Seed arena: initialize weights.
	seedArena := uop.NewArena(64)
	seedModel := newConvNetModel(seedArena, device)

	rng := rand.New(rand.NewSource(42))
	heInit(seedModel.conv.Weight, 9, rng) // fan-in = Cin*kH*kW = 1*3*3 = 9
	heInit(seedModel.fc.Weight, int(convFlatLen), rng)

	// Compute arena.
	a := uop.NewArena(131072)
	model := newConvNetModel(a, device)

	for _, pDst := range model.conv.Params() {
		for _, pSrc := range seedModel.conv.Params() {
			if pDst.Name == pSrc.Name {
				copyParam(pDst, pSrc)
			}
		}
	}
	// Copy by position since names may not be set.
	srcConvParams := seedModel.conv.Params()
	dstConvParams := model.conv.Params()
	for i := range srcConvParams {
		if i < len(dstConvParams) {
			copyParam(dstConvParams[i], srcConvParams[i])
		}
	}
	srcFCParams := seedModel.fc.Params()
	dstFCParams := model.fc.Params()
	for i := range srcFCParams {
		if i < len(dstFCParams) {
			copyParam(dstFCParams[i], srcFCParams[i])
		}
	}

	// Load params and collect leaf tensors for the backward pass.
	var leaves []*tensor.Tensor
	for _, p := range model.convParams() {
		p.Load(a)
		leaves = append(leaves, p.T)
	}

	images, _ := convDataset()
	x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
		uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, images...))

	out := model.forward(x)

	return &BuildResult{
		Arena:  a,
		Output: out,
		Device: device,
		Leaves: leaves,
	}, nil
}

// trainConv runs the conv net training loop.
func trainConv(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	// Seed arena for initialization.
	seedArena := uop.NewArena(64)
	seedModel := newConvNetModel(seedArena, device)

	rng := rand.New(rand.NewSource(42))
	heInit(seedModel.conv.Weight, 9, rng)
	heInit(seedModel.fc.Weight, int(convFlatLen), rng)

	// Persistent model.
	a0 := uop.NewArena(64)
	model := newConvNetModel(a0, device)

	srcConvParams := seedModel.conv.Params()
	dstConvParams := model.conv.Params()
	for i := range srcConvParams {
		if i < len(dstConvParams) {
			copyParam(dstConvParams[i], srcConvParams[i])
		}
	}
	srcFCParams := seedModel.fc.Params()
	dstFCParams := model.fc.Params()
	for i := range srcFCParams {
		if i < len(dstFCParams) {
			copyParam(dstFCParams[i], srcFCParams[i])
		}
	}

	params := model.convParams()
	opt := nn.NewSGD(params, cfg.LR)

	images, labels := convDataset()

	if cfg.LogEvery > 0 {
		l0 := evalConvLoss(model.forward, params, images, labels, device)
		logFn(0, l0)
	}

	for step := 1; step <= cfg.Steps; step++ {
		a := uop.NewArena(131072)
		for _, p := range opt.Params {
			p.Load(a)
		}

		x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
			uop.Dtypes.Float32, device)
		x.SetData(append([]float32{}, images...))
		tgt := tensor.NewLeaf(a, []int64{convBatch, 1}, uop.Dtypes.Float32, device)
		tgt.SetData(append([]float32{}, labels...))

		pred := model.forward(x)
		diff := pred.Sub(tgt)
		scale := tensor.ConstScalar(a, 1.0/float64(convBatch), uop.Dtypes.Float32, device)
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
			l := evalConvLoss(model.forward, params, images, labels, device)
			logFn(step, l)
		}
	}

	return nil
}

// evalConvLoss runs a forward-only pass and returns the MSE-mean loss.
func evalConvLoss(
	forward func(*tensor.Tensor) *tensor.Tensor,
	params []*nn.Parameter,
	images, labels []float32,
	device string,
) float32 {
	a := uop.NewArena(131072)
	for _, p := range params {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{convBatch, convCin, convImgH, convImgW},
		uop.Dtypes.Float32, device)
	x.SetData(append([]float32{}, images...))
	tgt := tensor.NewLeaf(a, []int64{convBatch, 1}, uop.Dtypes.Float32, device)
	tgt.SetData(append([]float32{}, labels...))
	diff := forward(x).Sub(tgt)
	scale := tensor.ConstScalar(a, 1.0/float64(convBatch), uop.Dtypes.Float32, device)
	loss := diff.Mul(diff).Sum(nil, false).Mul(scale)
	if err := tensor.Realize(loss); err != nil {
		return 0
	}
	return loss.Data()[0]
}

// convDataset returns 8 synthetic 1×6×6 images and scalar labels.
// Label for each image = mean pixel value in the top-left 3×3 region.
func convDataset() (images, labels []float32) {
	const (
		N = 8
		H = 6
		W = 6
	)
	images = make([]float32, N*H*W)
	labels = make([]float32, N)

	topLeftVals := [N]float32{0.9, 0.75, 0.6, 0.45, 0.8, 0.65, 0.5, 0.35}

	for n := 0; n < N; n++ {
		tlV := topLeftVals[n]
		for row := 0; row < H; row++ {
			for col := 0; col < W; col++ {
				var v float32
				if row < 3 && col < 3 {
					v = tlV
				} else {
					v = 1.0 - tlV
				}
				images[n*H*W+row*W+col] = v
			}
		}
		labels[n] = tlV
	}
	return
}
