package nn

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// ── BatchNorm2d ──────────────────────────────────────────────────────────────

// BatchNorm2d normalizes a 4-D input across the (N, H, W) axes for each channel.
// Mirrors torch.nn.BatchNorm2d.
//
// For input x of shape [N, C, H, W]:
//
//	training mode:
//	   mu_b   = mean(x, axes=[0, 2, 3])                       shape [C]
//	   var_b  = mean((x - mu_b)^2, axes=[0, 2, 3])            shape [C]
//	   xhat   = (x - mu_b) / sqrt(var_b + eps)
//	   y      = xhat * Weight + Bias
//	   (RunningMean, RunningVar update via EMA - host-side after Realize)
//
//	eval mode:
//	   xhat   = (x - RunningMean) / sqrt(RunningVar + eps)
//	   y      = xhat * Weight + Bias
//
// Weight (scale, init ones) and Bias (init zeros) are learnable parameters of
// shape [C]. RunningMean (init zeros) and RunningVar (init ones) are stateful
// non-differentiated buffers stored as []float32 alongside the module - they
// survive arena resets exactly like Adam's m and v moment buffers in optim.go.
//
// State-update contract (training mode): caller invokes PostStep on the live
// tensors after the training step's Realize completes. PostStep reads the
// batch statistics back from the realized tensors and updates RunningMean and
// RunningVar in-place via the EMA rule. Mirrors Adam.Step's "read-realized-
// gradient, mutate Go state" pattern.
type BatchNorm2d struct {
	Weight *Parameter // gamma, shape [C]; init ones
	Bias   *Parameter // beta,  shape [C]; init zeros

	RunningMean []float32 // shape [C]; init zeros; survives arena reset
	RunningVar  []float32 // shape [C]; init ones;  survives arena reset

	Momentum float32 // EMA factor; PyTorch default 0.1
	Eps      float32 // added inside sqrt; PyTorch default 1e-5
	Training bool    // toggle via Train() / Eval()

	channels int64

	// Pinned references to the last training-mode forward's batch statistics,
	// in [1, C, 1, 1] shape so PostStep can read their realized data. Cleared
	// in eval mode; cleared on PostStep after consumption.
	lastBatchMean *tensor.Tensor
	lastBatchVar  *tensor.Tensor
}

// NewBatchNorm2d constructs a BatchNorm2d module for C channels.
// Weight is initialised to 1.0, Bias to 0.0, RunningMean to 0.0, RunningVar to 1.0.
// Defaults match torch.nn.BatchNorm2d: momentum=0.1, eps=1e-5. The module starts
// in training mode; call Eval() before inference.
func NewBatchNorm2d(a *uop.Arena, channels int64, eps, momentum float32, dtype *uop.DType, device string) *BatchNorm2d {
	if channels <= 0 {
		panic(fmt.Sprintf("nn: NewBatchNorm2d: channels must be positive, got %d", channels))
	}
	bn := &BatchNorm2d{
		Weight:      NewParameter(a, []int64{channels}, dtype, device),
		Bias:        NewParameter(a, []int64{channels}, dtype, device),
		RunningMean: make([]float32, channels),
		RunningVar:  make([]float32, channels),
		Momentum:    momentum,
		Eps:         eps,
		Training:    true,
		channels:    channels,
	}
	for i := range bn.Weight.Value {
		bn.Weight.Value[i] = 1.0
	}
	// Bias.Value already zero.
	// RunningMean already zero; RunningVar starts at 1.0 (PyTorch convention).
	for i := range bn.RunningVar {
		bn.RunningVar[i] = 1.0
	}
	return bn
}

// Train switches the module into training mode (compute batch stats, EMA update).
func (bn *BatchNorm2d) Train() { bn.Training = true }

// Eval switches the module into evaluation mode (use RunningMean / RunningVar).
func (bn *BatchNorm2d) Eval() { bn.Training = false }

// Forward applies BatchNorm2d to x (shape [N, C, H, W]).
//
// In training mode, the batch mean and variance are computed from x and the
// module retains references to them so a subsequent PostStep can update the
// running statistics. In eval mode, the running statistics are loaded as fresh
// BUFFER leaves in x's arena and used to normalize.
func (bn *BatchNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x.Rank() != 4 {
		panic(fmt.Sprintf("nn: BatchNorm2d.Forward: input must be 4-D, got rank %d", x.Rank()))
	}
	xShape := x.Shape()
	C := xShape[1]
	if C != bn.channels {
		panic(fmt.Sprintf("nn: BatchNorm2d.Forward: input channels %d != module channels %d", C, bn.channels))
	}

	a := x.Arena()
	bcastShape := []int64{1, C, 1, 1}

	var mean, variance *tensor.Tensor
	if bn.Training {
		// batchMean over N, H, W → shape [C]; reshape to [1, C, 1, 1] for broadcast.
		// Stash the pre-Reshape Mean result: reductions are buffer-materialised by
		// the scheduler, so PostStep can realize-and-read that tensor directly.
		meanFlat := x.Mean([]int{0, 2, 3}, false)
		mean = meanFlat.Reshape(bcastShape)
		xc := x.Sub(mean)
		varFlat := xc.Mul(xc).Mean([]int{0, 2, 3}, false)
		variance = varFlat.Reshape(bcastShape)
		bn.lastBatchMean = meanFlat
		bn.lastBatchVar = varFlat
	} else {
		// Eval mode: load RunningMean / RunningVar as fresh BUFFER leaves.
		mLeaf := tensor.NewLeaf(a, []int64{C}, x.DType(), x.Device())
		mLeaf.SetData(append([]float32{}, bn.RunningMean...))
		vLeaf := tensor.NewLeaf(a, []int64{C}, x.DType(), x.Device())
		vLeaf.SetData(append([]float32{}, bn.RunningVar...))
		mean = mLeaf.Reshape(bcastShape)
		variance = vLeaf.Reshape(bcastShape)
		bn.lastBatchMean = nil
		bn.lastBatchVar = nil
	}

	epsT := tensor.FullSints(a, variance.ShapeSints(), float64(bn.Eps), x.DType(), x.Device())
	invStd := variance.Add(epsT).Sqrt().Recip()

	xhat := x.Sub(mean).Mul(invStd)

	gamma := bn.Weight.T.Reshape(bcastShape)
	beta := bn.Bias.T.Reshape(bcastShape)
	return xhat.Mul(gamma).Add(beta)
}

// PostStep updates RunningMean and RunningVar using EMA from the batch statistics
// of the most recent training-mode Forward. In eval mode or before any Forward,
// PostStep is a no-op.
//
// PostStep realizes the batch-mean and batch-var tensors itself so callers don't
// have to track them in their Realize lists. If the parent graph was already
// realized (typical: realize loss + grads, then PostStep), the cached buffers
// are picked up; otherwise the dependencies are scheduled on demand.
//
// The EMA rule mirrors PyTorch:
//
//	running = (1 - momentum) * running + momentum * batch
func (bn *BatchNorm2d) PostStep() error {
	if bn.lastBatchMean == nil || bn.lastBatchVar == nil {
		return nil
	}
	if err := tensor.Realize(bn.lastBatchMean); err != nil {
		return fmt.Errorf("nn: BatchNorm2d.PostStep: realize batch mean: %w", err)
	}
	if err := tensor.Realize(bn.lastBatchVar); err != nil {
		return fmt.Errorf("nn: BatchNorm2d.PostStep: realize batch var: %w", err)
	}
	mData := bn.lastBatchMean.Data()
	vData := bn.lastBatchVar.Data()
	if len(mData) != int(bn.channels) || len(vData) != int(bn.channels) {
		return fmt.Errorf("nn: BatchNorm2d.PostStep: batch-stat length %d/%d != channels %d",
			len(mData), len(vData), bn.channels)
	}
	m := bn.Momentum
	for c := int64(0); c < bn.channels; c++ {
		bn.RunningMean[c] = (1-m)*bn.RunningMean[c] + m*mData[c]
		bn.RunningVar[c] = (1-m)*bn.RunningVar[c] + m*vData[c]
	}
	bn.lastBatchMean = nil
	bn.lastBatchVar = nil
	return nil
}

// Params returns the trainable parameters (Weight, Bias) in deterministic order.
// RunningMean and RunningVar are NOT in this set - they are non-differentiated
// state buffers updated via PostStep.
func (bn *BatchNorm2d) Params() []*Parameter {
	return []*Parameter{bn.Weight, bn.Bias}
}
