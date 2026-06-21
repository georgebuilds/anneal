package nn

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/tensor"
)

// ── Adam Optimizer ────────────────────────────────────────────────────────────

// Adam implements the Adam optimization algorithm (Kingma and Ba, 2015) over a
// fixed parameter set. State is held on the optimizer (not the Parameter), so
// multiple optimizers can coexist for the same parameters during experiments.
//
// Per Parameter, Adam maintains:
//
//	m: exponential moving average of gradients     (first moment)
//	v: exponential moving average of squared grads (second moment)
//
// Both moments are stored as plain Go []float32 slices, mirroring p.Value, and
// they survive arena resets exactly like p.Value does. No tensor arena is touched.
//
// The update rule, given gradient g at step t:
//
//	m   = beta1 * m + (1 - beta1) * g
//	v   = beta2 * v + (1 - beta2) * g * g
//	m^  = m / (1 - beta1^t)
//	v^  = v / (1 - beta2^t)
//	val = val - lr * m^ / (sqrt(v^) + eps)
//
// Defaults match the paper: beta1=0.9, beta2=0.999, eps=1e-8.
type Adam struct {
	Params []*Parameter
	LR     float32
	Beta1  float32
	Beta2  float32
	Eps    float32
	// WeightDecay enables decoupled weight decay (AdamW, Loshchilov & Hutter):
	// each step also pulls every parameter toward 0 by lr*WeightDecay*value,
	// applied independently of the moment estimates. 0 disables it (plain Adam).
	// It is the direct stabiliser against weight explosion in fine-tuning (e.g.
	// the tied GPT-2 Wte, which gets a double gradient and otherwise grows until
	// the logits overflow f32).
	WeightDecay float32
	T           int // step counter; incremented on each Step call

	// Per-parameter first/second moment buffers. m[i] and v[i] are slices of
	// the same length as Params[i].Value; they are zero-initialised on construction.
	m []*[]float32
	v []*[]float32
}

// NewAdam constructs an Adam optimizer with the paper's default hyperparameters
// (beta1=0.9, beta2=0.999, eps=1e-8). lr is the only required tuning knob.
func NewAdam(params []*Parameter, lr float32) *Adam {
	return NewAdamWithBetas(params, lr, 0.9, 0.999, 1e-8)
}

// NewAdamW constructs an AdamW optimizer (Adam + decoupled weight decay) with
// the paper's default betas/eps. Use weightDecay ~0.1 for transformer
// fine-tuning to keep weights bounded.
func NewAdamW(params []*Parameter, lr, weightDecay float32) *Adam {
	a := NewAdamWithBetas(params, lr, 0.9, 0.999, 1e-8)
	a.WeightDecay = weightDecay
	return a
}

// NewAdamWithBetas constructs an Adam optimizer with caller-supplied betas and
// epsilon. Use this when sweeping hyperparameters or matching an external setup.
func NewAdamWithBetas(params []*Parameter, lr, beta1, beta2, eps float32) *Adam {
	ps := make([]*Parameter, len(params))
	copy(ps, params)

	ms := make([]*[]float32, len(ps))
	vs := make([]*[]float32, len(ps))
	for i, p := range ps {
		mBuf := make([]float32, len(p.Value))
		vBuf := make([]float32, len(p.Value))
		ms[i] = &mBuf
		vs[i] = &vBuf
	}
	return &Adam{
		Params: ps,
		LR:     lr,
		Beta1:  beta1,
		Beta2:  beta2,
		Eps:    eps,
		T:      0,
		m:      ms,
		v:      vs,
	}
}

// Step applies one Adam update to each parameter whose gradient is in grads.
// grads is the map returned by tensor.Backward, keyed by the step's leaf tensors
// (p.T after p.Load was called for this step). Every gradient tensor in grads must
// have been realized (tensor.Realize called) before Step is invoked.
//
// Mirrors SGD.Step's gradient-lookup contract: parameters whose gradient is
// absent from grads are silently skipped, leaving their moment buffers and step
// counter usage untouched for that iteration.
func (opt *Adam) Step(grads map[*tensor.Tensor]*tensor.Tensor) {
	opt.T++
	// Precompute bias-correction denominators once per Step call. Both 1-beta^t
	// values are scalar; they apply uniformly across all parameters at step T.
	bc1 := 1 - float32(math.Pow(float64(opt.Beta1), float64(opt.T)))
	bc2 := 1 - float32(math.Pow(float64(opt.Beta2), float64(opt.T)))

	for i, p := range opt.Params {
		g, ok := grads[p.T]
		if !ok {
			continue
		}
		opt.applyOne(i, p, g.Data(), bc1, bc2)
	}
}

// applyOne updates a single parameter's value and moment buffers in place.
// Split out from Step to keep the hot loop readable; mirrors SGD.SGDStep.
func (opt *Adam) applyOne(i int, p *Parameter, grad []float32, bc1, bc2 float32) {
	if len(grad) != len(p.Value) {
		panic(fmt.Sprintf("nn: Adam.Step: gradient length %d != parameter length %d for %q",
			len(grad), len(p.Value), p.Name))
	}
	m := *opt.m[i]
	v := *opt.v[i]
	if len(m) != len(p.Value) || len(v) != len(p.Value) {
		panic(fmt.Sprintf("nn: Adam.Step: moment buffer mismatch for %q (m=%d v=%d val=%d)",
			p.Name, len(m), len(v), len(p.Value)))
	}

	lr := opt.LR
	b1 := opt.Beta1
	b2 := opt.Beta2
	eps := opt.Eps
	wd := opt.WeightDecay

	for j := range p.Value {
		g := grad[j]
		m[j] = b1*m[j] + (1-b1)*g
		v[j] = b2*v[j] + (1-b2)*g*g
		mHat := m[j] / bc1
		vHat := v[j] / bc2
		// Decoupled weight decay (AdamW): pull toward 0 independently of the
		// adaptive moment step. Skipped when wd==0 (plain Adam).
		if wd != 0 {
			p.Value[j] -= lr * wd * p.Value[j]
		}
		p.Value[j] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
	}
}

// ZeroGrad is a no-op for Adam in the current design: gradients are produced by
// tensor.Backward into fresh tensors on every step and never accumulate across
// steps. The method exists for parity with optimizers in other frameworks; call
// it freely without semantic effect.
func (opt *Adam) ZeroGrad() {}
