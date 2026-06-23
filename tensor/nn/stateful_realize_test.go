package nn_test

import (
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// TestStatefulRealizeOracle is the correctness gate for stateful realize: the
// gradients produced by stateful one-by-one realize (each grad in its own Realize
// call, reusing the cached shared forward) must be BIT-IDENTICAL to stateless
// batched realize (one Realize for everything). A stale or aliased cached buffer
// would change a gradient, so equality here proves the cache is correct.
func TestStatefulRealizeOracle(t *testing.T) {
	requireGPU(t)

	build := func() ([]*nn.Parameter, *tensor.Tensor) {
		a := uop.NewArena(1 << 20)
		rng := rand.New(rand.NewSource(7))
		layers := []*nn.Linear{
			nn.NewLinear(a, 8, 16, true, uop.Dtypes.Float32, "webgpu"),
			nn.NewLinear(a, 16, 16, true, uop.Dtypes.Float32, "webgpu"),
			nn.NewLinear(a, 16, 4, true, uop.Dtypes.Float32, "webgpu"),
		}
		var params []*nn.Parameter
		for _, l := range layers {
			for _, p := range l.Params() {
				for i := range p.Value {
					p.Value[i] = float32(rng.NormFloat64()) * 0.1
				}
				p.Load(a)
				params = append(params, p)
			}
		}
		x := tensor.NewLeaf(a, []int64{2, 8}, uop.Dtypes.Float32, "webgpu")
		xd := make([]float32, 16)
		for i := range xd {
			xd[i] = float32(rng.NormFloat64())
		}
		x.SetData(xd)
		h := layers[2].Forward(layers[1].Forward(layers[0].Forward(x)))
		loss := h.Mul(h).Sum(nil, false)
		return params, loss
	}

	gradsOf := func(stateful bool) []float32 {
		prev := tensor.SetStatefulRealize(stateful)
		defer tensor.SetStatefulRealize(prev)
		params, loss := build()
		grads := tensor.Backward(loss, paramTensors(params))
		if stateful {
			// One Realize per output, reusing the cached shared forward.
			if err := tensor.Realize(loss); err != nil {
				t.Fatalf("realize loss: %v", err)
			}
			for _, p := range params {
				if err := tensor.Realize(grads[p.T]); err != nil {
					t.Fatalf("realize grad: %v", err)
				}
			}
		} else {
			all := []*tensor.Tensor{loss}
			for _, p := range params {
				all = append(all, grads[p.T])
			}
			if err := tensor.Realize(all...); err != nil {
				t.Fatalf("realize batched: %v", err)
			}
		}
		var out []float32
		for _, p := range params {
			out = append(out, grads[p.T].Data()...)
		}
		return out
	}

	stateless := gradsOf(false)
	stateful := gradsOf(true)
	if len(stateless) != len(stateful) {
		t.Fatalf("grad element count mismatch: stateless=%d stateful=%d", len(stateless), len(stateful))
	}
	for i := range stateless {
		if stateless[i] != stateful[i] {
			t.Fatalf("grad[%d] differs: stateless=%v stateful=%v (stateful realize served a wrong buffer)",
				i, stateless[i], stateful[i])
		}
	}
	t.Logf("stateful realize bit-identical to stateless across %d grad elements", len(stateless))
}

// TestStatefulRealizeInvalidation checks the hardest correctness case: changing a
// leaf's data mid-arena (which bumps RealizeGen) must invalidate the cache, so a
// re-realize reflects the new data rather than a stale cached buffer.
func TestStatefulRealizeInvalidation(t *testing.T) {
	requireGPU(t)
	prev := tensor.SetStatefulRealize(true)
	defer tensor.SetStatefulRealize(prev)

	a := uop.NewArena(1 << 18)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "webgpu")
	y := x.Mul(x) // a cacheable intermediate, then an output

	x.SetData([]float32{1, 2, 3, 4})
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("realize 1: %v", err)
	}
	got1 := append([]float32{}, y.Data()...) // expect 1,4,9,16

	// Change x: must invalidate the cache and recompute.
	x.SetData([]float32{2, 3, 4, 5})
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("realize 2: %v", err)
	}
	got2 := append([]float32{}, y.Data()...) // expect 4,9,16,25

	want1 := []float32{1, 4, 9, 16}
	want2 := []float32{4, 9, 16, 25}
	for i := range want1 {
		if got1[i] != want1[i] {
			t.Fatalf("realize 1 [%d]=%v want %v", i, got1[i], want1[i])
		}
		if got2[i] != want2[i] {
			t.Fatalf("realize 2 [%d]=%v want %v (cache not invalidated on SetData)", i, got2[i], want2[i])
		}
	}
}

// TestStatefulRealizeMultiStep trains a tiny net a few steps with stateful realize
// on and confirms the loss strictly decreases, i.e. params actually update from
// correct gradients across steps (each step a fresh arena, so a new cache scope).
func TestStatefulRealizeMultiStep(t *testing.T) {
	requireGPU(t)
	prev := tensor.SetStatefulRealize(true)
	defer tensor.SetStatefulRealize(prev)

	rng := rand.New(rand.NewSource(3))
	setupA := uop.NewArena(1 << 16)
	w := nn.NewLinear(setupA, 4, 4, true, uop.Dtypes.Float32, "webgpu")
	for _, p := range w.Params() {
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * 0.1
		}
	}
	target := []float32{0.5, -0.5, 0.5, -0.5}
	xd := []float32{1, 0.5, -0.5, 1}

	var first, last float32
	opt := nn.NewAdam(w.Params(), 0.05)
	for step := 0; step < 8; step++ {
		a := uop.NewArena(1 << 18)
		for _, p := range w.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Float32, "webgpu")
		x.SetData(xd)
		tgt := tensor.NewLeaf(a, []int64{1, 4}, uop.Dtypes.Float32, "webgpu")
		tgt.SetData(target)
		diff := w.Forward(x).Sub(tgt)
		loss := diff.Mul(diff).Sum(nil, false)
		grads := tensor.Backward(loss, paramTensors(w.Params()))
		// One Realize per output (the pattern stateful realize accelerates).
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("realize loss: %v", err)
		}
		for _, p := range w.Params() {
			if err := tensor.Realize(grads[p.T]); err != nil {
				t.Fatalf("realize grad: %v", err)
			}
		}
		if step == 0 {
			first = loss.Data()[0]
		}
		last = loss.Data()[0]
		opt.Step(grads)
	}
	if !(last < first) {
		t.Fatalf("loss did not decrease with stateful realize: first=%v last=%v", first, last)
	}
	t.Logf("stateful multi-step: loss %v -> %v", first, last)
}
