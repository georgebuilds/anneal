package nn_test

// Llama primitive correctness tests.
//
// Covers the modern small-LM stack added in tensor/nn: RMSNorm, RoPE, SwiGLU,
// GQAttention, and the Llama container that composes them. The harness mirrors
// block_test.go / gpt_test.go / vit_test.go exactly:
//
//   - requireGPU bootstrap (train_test.go) for the Metal/WebGPU executor.
//   - Central-difference fdXxxGradParam closures + a checkParam closure with the
//     tiered tolerance budget (tolTight=1e-3 for linear-only paths, tolSoftmax=
//     7e-2 for softmax/norm/rope chains).
//   - Every heavy softmax/norm-chain FD block is guarded by `if !testing.Short()`
//     so `go test -short` (lavapipe CI) skips the kernel-heavy backward passes
//     that OOM the runner — same discipline as gpt_test.go/vit_test.go.
//
// Tiny configs throughout (nEmbd=16, nHead=2, nKVHead=1, headDim=8, B=2, T=4,
// blockSize=4, vocab=8) keep the GPU work small.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── shared tolerances ─────────────────────────────────────────────────────────

const (
	llamaFDH        = float32(1e-3)
	llamaTolTight   = float32(1e-3)
	llamaTolSoftmax = float32(7e-2)
	llamaNCheck     = 3
)

// llamaHashFloats returns a sha256 hex digest of the float32 slice (little-endian
// bit patterns), matching the determinism-check convention in the sibling tests.
func llamaHashFloats(xs []float32) string {
	h := sha256.New()
	buf := make([]byte, 4)
	for _, v := range xs {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
		_, _ = h.Write(buf)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// llamaRelErr returns the scale-normalized relative error between analytic and
// FD gradients (scale floored at 1 so tiny gradients use absolute error).
func llamaRelErr(analytic, fd float32) float32 {
	diff := absF32(analytic - fd)
	scale := absF32(fd)
	if absF32(analytic) > scale {
		scale = absF32(analytic)
	}
	if scale < 1 {
		scale = 1
	}
	return diff / scale
}

// mustPanicLlama runs fn and fails the test unless fn panics. (The sibling
// coverage_extra_test.go mustPanic additionally asserts on the panic substring;
// here we only require that a panic occurred.)
func mustPanicLlama(t *testing.T, label string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("%s: expected panic, got none", label)
		}
	}()
	fn()
}

// idxBits encodes int32 indices as float32 bit patterns (the Int32-leaf upload
// convention shared by embedding_test.go / gpt_test.go).
func idxBits(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. RMSNorm
// ──────────────────────────────────────────────────────────────────────────────

// rmsNormHostRef computes the reference RMSNorm of a single [D] row:
//
//	y[d] = x[d] * rsqrt(mean(x^2) + eps) * w[d]
func rmsNormHostRef(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	meanSq := ss / float64(len(x))
	inv := float32(1.0 / math.Sqrt(meanSq+float64(eps)))
	out := make([]float32, len(x))
	for d := range x {
		out[d] = x[d] * inv * w[d]
	}
	return out
}

// TestRMSNormForward checks the forward pass against the host reference at a few
// elements, for a [B, T, D] input with a non-trivial Weight.
func TestRMSNormForward(t *testing.T) {
	requireGPU(t)

	const (
		B   = int64(2)
		T   = int64(4)
		D   = int64(16)
		eps = float32(1e-5)
	)

	a0 := uop.NewArena(4096)
	rn := nn.NewRMSNorm(a0, D, eps)
	rng := rand.New(rand.NewSource(3))
	for i := range rn.Weight.Value {
		rn.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*0.1
	}

	a := uop.NewArena(1 << 16)
	rn.Weight.Load(a)
	x := tensor.NewLeaf(a, []int64{B, T, D}, uop.Dtypes.Float32, "webgpu")
	xData := make([]float32, B*T*D)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}
	x.SetData(append([]float32{}, xData...))

	y := rn.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("RMSNorm Realize: %v", err)
	}
	got := y.Data()
	if int64(len(got)) != B*T*D {
		t.Fatalf("RMSNorm output length %d != %d", len(got), B*T*D)
	}

	// Compare every row against the host reference.
	maxErr := float32(0)
	for row := int64(0); row < B*T; row++ {
		xr := xData[row*D : (row+1)*D]
		want := rmsNormHostRef(xr, rn.Weight.Value, eps)
		for d := int64(0); d < D; d++ {
			e := absF32(got[row*D+d] - want[d])
			if e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 1e-4 {
		t.Fatalf("RMSNorm forward max-abs-diff %.3e exceeds 1e-4", maxErr)
	}
	t.Logf("RMSNorm forward matches host reference (max-abs-diff=%.3e)", maxErr)
}

// TestRMSNormFDGrad FD-checks the gradients on Weight and the input x. The
// rsqrt/mean chain amplifies central-difference truncation, so both run at the
// softmax-chain budget and are skipped under -short (the GPU backward kernels
// are kernel-heavy).
func TestRMSNormFDGrad(t *testing.T) {
	requireGPU(t)
	if testing.Short() {
		t.Skip("RMSNorm FD grad: skipped on -short (norm-chain backward is kernel-heavy)")
	}

	const (
		B   = int64(2)
		T   = int64(4)
		D   = int64(8)
		eps = float32(1e-5)
	)

	a0 := uop.NewArena(4096)
	rn := nn.NewRMSNorm(a0, D, eps)
	rng := rand.New(rand.NewSource(5))
	for i := range rn.Weight.Value {
		rn.Weight.Value[i] = 1.0 + float32(rng.NormFloat64())*0.1
	}
	xData := make([]float32, B*T*D)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}

	evalLoss := func() float32 {
		a := uop.NewArena(1 << 16)
		rn.Weight.Load(a)
		x := tensor.NewLeaf(a, []int64{B, T, D}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		loss := rn.Forward(x).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("RMSNorm evalLoss Realize: %v", err)
		}
		return loss.Data()[0]
	}

	// Analytic gradients on Weight and x.
	a := uop.NewArena(1 << 17)
	rn.Weight.Load(a)
	x := tensor.NewLeaf(a, []int64{B, T, D}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := rn.Forward(x).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{rn.Weight.T, x})
	for _, g := range grads {
		if err := tensor.Realize(g); err != nil {
			t.Fatalf("RMSNorm Realize grad: %v", err)
		}
	}
	wGrad := append([]float32{}, grads[rn.Weight.T].Data()...)
	xGrad := append([]float32{}, grads[x].Data()...)

	// FD on Weight.
	maxRelW := float32(0)
	for i := 0; i < llamaNCheck; i++ {
		orig := rn.Weight.Value[i]
		rn.Weight.Value[i] = orig + llamaFDH
		lp := evalLoss()
		rn.Weight.Value[i] = orig - llamaFDH
		lm := evalLoss()
		rn.Weight.Value[i] = orig
		fd := (lp - lm) / (2 * llamaFDH)
		rel := llamaRelErr(wGrad[i], fd)
		if rel > maxRelW {
			maxRelW = rel
		}
		t.Logf("RMSNorm.Weight[%d]: analytic=%+.6f fd=%+.6f rel=%.2e", i, wGrad[i], fd, rel)
		if rel > llamaTolSoftmax {
			t.Fatalf("RMSNorm.Weight[%d]: rel=%.2e > tol=%.2e", i, rel, llamaTolSoftmax)
		}
	}

	// FD on input x.
	maxRelX := float32(0)
	for i := 0; i < llamaNCheck; i++ {
		orig := xData[i]
		xData[i] = orig + llamaFDH
		lp := evalLoss()
		xData[i] = orig - llamaFDH
		lm := evalLoss()
		xData[i] = orig
		fd := (lp - lm) / (2 * llamaFDH)
		rel := llamaRelErr(xGrad[i], fd)
		if rel > maxRelX {
			maxRelX = rel
		}
		t.Logf("RMSNorm.x[%d]: analytic=%+.6f fd=%+.6f rel=%.2e", i, xGrad[i], fd, rel)
		if rel > llamaTolSoftmax {
			t.Fatalf("RMSNorm.x[%d]: rel=%.2e > tol=%.2e", i, rel, llamaTolSoftmax)
		}
	}
	t.Logf("RMSNorm FD ok (Weight max-rel=%.2e, x max-rel=%.2e)", maxRelW, maxRelX)
}

// TestRMSNormPanics covers the constructor and forward guards.
func TestRMSNormPanics(t *testing.T) {
	a0 := uop.NewArena(2048)

	mustPanicLlama(t, "NewRMSNorm(normalizedShape=0)", func() {
		nn.NewRMSNorm(a0, 0, 1e-5)
	})
	mustPanicLlama(t, "NewRMSNorm(normalizedShape<0)", func() {
		nn.NewRMSNorm(a0, -4, 1e-5)
	})

	// rank-0 input -> Forward must panic (rank < 1).
	rn := nn.NewRMSNorm(a0, 4, 1e-5)
	a := uop.NewArena(1 << 14)
	rn.Weight.Load(a)
	scalar := tensor.NewLeaf(a, []int64{}, uop.Dtypes.Float32, "webgpu")
	mustPanicLlama(t, "RMSNorm.Forward(rank-0)", func() {
		rn.Forward(scalar)
	})
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. RoPE
// ──────────────────────────────────────────────────────────────────────────────

// ropeHostRef applies the rotate_half RoPE convention to a single [T, D] head
// (B=H=1), returning the rotated [T, D] row-major buffer.
func ropeHostRef(x []float32, T, D int, base float64) []float32 {
	half := D / 2
	out := make([]float32, T*D)
	for p := 0; p < T; p++ {
		for d := 0; d < D; d++ {
			j := d % half
			invFreq := math.Pow(base, -float64(2*j)/float64(D))
			angle := float64(p) * invFreq
			c := float32(math.Cos(angle))
			s := float32(math.Sin(angle))
			// rotate_half(x) = concat(-x[half:], x[:half]).
			var rot float32
			if d < half {
				rot = -x[p*D+(d+half)]
			} else {
				rot = x[p*D+(d-half)]
			}
			out[p*D+d] = x[p*D+d]*c + rot*s
		}
	}
	return out
}

// TestRoPEForward checks Apply against a host re-implementation of rotate_half.
func TestRoPEForward(t *testing.T) {
	requireGPU(t)

	const (
		T    = 4
		D    = 8
		base = 10000.0
	)

	r := nn.NewRoPE(D, 16, base)

	a := uop.NewArena(1 << 16)
	x := tensor.NewLeaf(a, []int64{1, 1, T, D}, uop.Dtypes.Float32, "webgpu")
	rng := rand.New(rand.NewSource(11))
	xData := make([]float32, T*D)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}
	x.SetData(append([]float32{}, xData...))

	y := r.Apply(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("RoPE Realize: %v", err)
	}
	got := y.Data()
	want := ropeHostRef(xData, T, D, base)
	if len(got) != len(want) {
		t.Fatalf("RoPE output length %d != %d", len(got), len(want))
	}
	maxErr := float32(0)
	for i := range want {
		e := absF32(got[i] - want[i])
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("RoPE forward max-abs-diff %.3e exceeds 1e-5", maxErr)
	}
	t.Logf("RoPE forward matches host rotate_half reference (max-abs-diff=%.3e)", maxErr)
}

// TestRoPEFDGrad FD-checks the input gradient through a downstream Sum loss.
// RoPE has no trainable params; this exercises the Apply backward (concat /
// shrink / neg / broadcast-mul). Guarded on -short.
func TestRoPEFDGrad(t *testing.T) {
	requireGPU(t)
	if testing.Short() {
		t.Skip("RoPE FD grad: skipped on -short (Apply backward is kernel-heavy)")
	}

	const (
		B    = int64(2)
		H    = int64(1)
		T    = int64(4)
		D    = int64(8)
		base = 10000.0
	)

	r := nn.NewRoPE(int(D), 16, base)
	rng := rand.New(rand.NewSource(13))
	xData := make([]float32, B*H*T*D)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}

	evalLoss := func() float32 {
		a := uop.NewArena(1 << 16)
		x := tensor.NewLeaf(a, []int64{B, H, T, D}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		loss := r.Apply(x).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("RoPE evalLoss Realize: %v", err)
		}
		return loss.Data()[0]
	}

	a := uop.NewArena(1 << 17)
	x := tensor.NewLeaf(a, []int64{B, H, T, D}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := r.Apply(x).Sum(nil, false)
	grads := tensor.Backward(loss, []*tensor.Tensor{x})
	if err := tensor.Realize(grads[x]); err != nil {
		t.Fatalf("RoPE Realize grad: %v", err)
	}
	xGrad := append([]float32{}, grads[x].Data()...)

	maxRel := float32(0)
	for i := 0; i < llamaNCheck; i++ {
		orig := xData[i]
		xData[i] = orig + llamaFDH
		lp := evalLoss()
		xData[i] = orig - llamaFDH
		lm := evalLoss()
		xData[i] = orig
		fd := (lp - lm) / (2 * llamaFDH)
		rel := llamaRelErr(xGrad[i], fd)
		if rel > maxRel {
			maxRel = rel
		}
		t.Logf("RoPE.x[%d]: analytic=%+.6f fd=%+.6f rel=%.2e", i, xGrad[i], fd, rel)
		if rel > llamaTolSoftmax {
			t.Fatalf("RoPE.x[%d]: rel=%.2e > tol=%.2e", i, rel, llamaTolSoftmax)
		}
	}
	t.Logf("RoPE FD ok (x max-rel=%.2e)", maxRel)
}

// TestRoPEPanics covers the constructor and Apply guards.
func TestRoPEPanics(t *testing.T) {
	requireGPU(t)

	mustPanicLlama(t, "NewRoPE(odd headDim)", func() {
		nn.NewRoPE(7, 16, 10000.0)
	})
	mustPanicLlama(t, "NewRoPE(headDim<=0)", func() {
		nn.NewRoPE(0, 16, 10000.0)
	})
	mustPanicLlama(t, "NewRoPE(maxSeqLen<=0)", func() {
		nn.NewRoPE(8, 0, 10000.0)
	})
	mustPanicLlama(t, "NewRoPE(base<=0)", func() {
		nn.NewRoPE(8, 16, 0)
	})

	r := nn.NewRoPE(8, 4, 10000.0)
	a := uop.NewArena(1 << 14)

	// T > maxSeqLen.
	mustPanicLlama(t, "RoPE.Apply(T>maxSeqLen)", func() {
		x := tensor.NewLeaf(a, []int64{1, 1, 8, 8}, uop.Dtypes.Float32, "webgpu")
		r.Apply(x)
	})
	// headDim mismatch.
	mustPanicLlama(t, "RoPE.Apply(headDim mismatch)", func() {
		x := tensor.NewLeaf(a, []int64{1, 1, 4, 4}, uop.Dtypes.Float32, "webgpu")
		r.Apply(x)
	})
	// non-rank-4 input.
	mustPanicLlama(t, "RoPE.Apply(non-rank-4)", func() {
		x := tensor.NewLeaf(a, []int64{1, 4, 8}, uop.Dtypes.Float32, "webgpu")
		r.Apply(x)
	})
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. SwiGLU
// ──────────────────────────────────────────────────────────────────────────────

func swigluInitSmall(m *nn.SwiGLU, scale float32, rng *rand.Rand) {
	for i := range m.Gate.Weight.Value {
		m.Gate.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for i := range m.Up.Weight.Value {
		m.Up.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
	for i := range m.Down.Weight.Value {
		m.Down.Weight.Value[i] = float32(rng.NormFloat64()) * scale
	}
}

// TestSwiGLUForwardShape checks the forward output shape [B, T, nEmbd].
func TestSwiGLUForwardShape(t *testing.T) {
	requireGPU(t)

	const (
		B      = int64(2)
		T      = int64(4)
		nEmbd  = 16
		hidden = 32
	)

	a0 := uop.NewArena(1 << 14)
	m := nn.NewSwiGLU(a0, nEmbd, hidden, uop.Dtypes.Float32, "webgpu")
	rng := rand.New(rand.NewSource(17))
	swigluInitSmall(m, 0.1, rng)

	a := uop.NewArena(1 << 17)
	for _, p := range m.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, T, nEmbd}, uop.Dtypes.Float32, "webgpu")
	xData := make([]float32, B*T*nEmbd)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}
	x.SetData(xData)

	y := m.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("SwiGLU Realize: %v", err)
	}
	sh := y.Shape()
	if len(sh) != 3 || sh[0] != B || sh[1] != T || sh[2] != nEmbd {
		t.Fatalf("SwiGLU output shape %v != [%d,%d,%d]", sh, B, T, nEmbd)
	}
	t.Logf("SwiGLU forward shape [B=%d,T=%d,nEmbd=%d] OK", B, T, nEmbd)
}

// TestSwiGLUHidden checks the rounded-up multiple convention.
func TestSwiGLUHidden(t *testing.T) {
	// base = (2*4*nEmbd)/3. For nEmbd=16: base = 128/3 = 42 (int div).
	// Rounded up to multiple of 8 -> 48; to 16 -> 48; to 1 -> 42.
	cases := []struct {
		nEmbd, multipleOf, want int
	}{
		{16, 1, 42},
		{16, 8, 48},
		{16, 16, 48},
		{16, 64, 64},
		// multipleOf<=0 is treated as 1.
		{16, 0, 42},
		{16, -8, 42},
		{32, 256, 256}, // base = 256/3 = 85 -> round up to 256
	}
	for _, c := range cases {
		got := nn.SwiGLUHidden(c.nEmbd, c.multipleOf)
		if got != c.want {
			t.Fatalf("SwiGLUHidden(%d,%d)=%d, want %d", c.nEmbd, c.multipleOf, got, c.want)
		}
	}
	t.Logf("SwiGLUHidden rounds up to multiple correctly across %d cases", len(cases))
}

// TestSwiGLUFDGrad FD-checks the Gate/Up/Down weight gradients. The SiLU gate is
// a sigmoid chain, so the gated branches use the softmax-chain budget; Down is a
// plain linear and uses the tight budget. The sigmoid-chain checks are skipped on
// -short.
func TestSwiGLUFDGrad(t *testing.T) {
	requireGPU(t)

	const (
		B      = int64(2)
		T      = int64(4)
		nEmbd  = 8
		hidden = 16
	)

	a0 := uop.NewArena(1 << 14)
	m := nn.NewSwiGLU(a0, nEmbd, hidden, uop.Dtypes.Float32, "webgpu")
	rng := rand.New(rand.NewSource(19))
	swigluInitSmall(m, 0.1, rng)

	xData := make([]float32, B*T*nEmbd)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}

	evalLoss := func() float32 {
		a := uop.NewArena(1 << 17)
		for _, p := range m.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{B, T, nEmbd}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		loss := m.Forward(x).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("SwiGLU evalLoss Realize: %v", err)
		}
		return loss.Data()[0]
	}

	a := uop.NewArena(1 << 18)
	for _, p := range m.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, T, nEmbd}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := m.Forward(x).Sum(nil, false)

	leaves := []*tensor.Tensor{m.Gate.Weight.T, m.Up.Weight.T, m.Down.Weight.T}
	grads := tensor.Backward(loss, leaves)
	for _, leaf := range leaves {
		if err := tensor.Realize(grads[leaf]); err != nil {
			t.Fatalf("SwiGLU Realize grad: %v", err)
		}
	}
	gateGrad := append([]float32{}, grads[m.Gate.Weight.T].Data()...)
	upGrad := append([]float32{}, grads[m.Up.Weight.T].Data()...)
	downGrad := append([]float32{}, grads[m.Down.Weight.T].Data()...)

	checkParam := func(p *nn.Parameter, ag []float32, label string, useTol float32) {
		t.Helper()
		n := llamaNCheck
		if n > len(ag) {
			n = len(ag)
		}
		maxRel := float32(0)
		for i := 0; i < n; i++ {
			orig := p.Value[i]
			p.Value[i] = orig + llamaFDH
			lp := evalLoss()
			p.Value[i] = orig - llamaFDH
			lm := evalLoss()
			p.Value[i] = orig
			fd := (lp - lm) / (2 * llamaFDH)
			rel := llamaRelErr(ag[i], fd)
			if rel > maxRel {
				maxRel = rel
			}
			t.Logf("%s[%d]: analytic=%+.6f fd=%+.6f rel=%.2e (tol=%.0e)", label, i, ag[i], fd, rel, useTol)
			if rel > useTol {
				t.Fatalf("%s[%d]: rel=%.2e > tol=%.2e", label, i, rel, useTol)
			}
		}
		t.Logf("%s max-rel=%.2e", label, maxRel)
	}

	// Down is the final bias-free linear: tight budget, safe under -short.
	checkParam(m.Down.Weight, downGrad, "Down.Weight", llamaTolTight)

	// Gate/Up flow through the SiLU sigmoid chain: looser budget, kernel-heavy
	// backward -> skip on -short.
	if !testing.Short() {
		checkParam(m.Gate.Weight, gateGrad, "Gate.Weight", llamaTolSoftmax)
		checkParam(m.Up.Weight, upGrad, "Up.Weight", llamaTolSoftmax)
	}
}

// TestSwiGLUPanics covers the constructor guard.
func TestSwiGLUPanics(t *testing.T) {
	a0 := uop.NewArena(2048)
	mustPanicLlama(t, "NewSwiGLU(nEmbd<=0)", func() {
		nn.NewSwiGLU(a0, 0, 16, uop.Dtypes.Float32, "webgpu")
	})
	mustPanicLlama(t, "NewSwiGLU(hidden<=0)", func() {
		nn.NewSwiGLU(a0, 16, 0, uop.Dtypes.Float32, "webgpu")
	})
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. GQAttention
// ──────────────────────────────────────────────────────────────────────────────

func gqaInitSmall(m *nn.GQAttention, scale float32, rng *rand.Rand) {
	for _, p := range m.Params() {
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * scale
		}
	}
}

// gqaForwardShape builds a GQAttention with the given head counts, runs Forward,
// and asserts the [B,T,nEmbd] output shape. Returns nothing; fails the test on
// mismatch.
func gqaForwardShape(t *testing.T, nEmbd, nHead, nKVHead, blockSize int) {
	t.Helper()
	const (
		B = int64(2)
		T = int64(4)
	)
	headDim := nEmbd / nHead

	a0 := uop.NewArena(1 << 14)
	rope := nn.NewRoPE(headDim, blockSize, 10000.0)
	m := nn.NewGQAttention(a0, nEmbd, nHead, nKVHead, blockSize, rope)
	rng := rand.New(rand.NewSource(23))
	gqaInitSmall(m, 0.1, rng)

	a := uop.NewArena(1 << 18)
	for _, p := range m.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, T, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
	xData := make([]float32, B*T*int64(nEmbd))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}
	x.SetData(xData)

	y := m.Forward(x)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("GQA Realize: %v", err)
	}
	sh := y.Shape()
	if len(sh) != 3 || sh[0] != B || sh[1] != T || sh[2] != int64(nEmbd) {
		t.Fatalf("GQA(nHead=%d,nKVHead=%d) output shape %v != [%d,%d,%d]",
			nHead, nKVHead, sh, B, T, nEmbd)
	}
	t.Logf("GQA(nHead=%d,nKVHead=%d,g=%d) forward shape [%d,%d,%d] OK",
		nHead, nKVHead, nHead/nKVHead, B, T, nEmbd)
}

// TestGQAForwardShapeG1 exercises the g==1 path (nKVHead == nHead, no repeatKV).
func TestGQAForwardShapeG1(t *testing.T) {
	requireGPU(t)
	// nEmbd=16, nHead=2, nKVHead=2 -> g=1, headDim=8.
	gqaForwardShape(t, 16, 2, 2, 4)
}

// TestGQAForwardShapeGgt1 exercises the g>1 path (nKVHead < nHead, repeatKV via
// Expand).
func TestGQAForwardShapeGgt1(t *testing.T) {
	requireGPU(t)
	// nEmbd=16, nHead=2, nKVHead=1 -> g=2, headDim=8.
	gqaForwardShape(t, 16, 2, 1, 4)
}

// gqaFDGrad runs the Q/K/V/Proj weight FD check for a given head config. Q/K/V
// flow through the softmax chain (skipped on -short); Proj is the final linear
// (tight, runs unconditionally).
func gqaFDGrad(t *testing.T, nEmbd, nHead, nKVHead, blockSize int) {
	t.Helper()
	const (
		B = int64(2)
		T = int64(4)
	)
	headDim := nEmbd / nHead

	a0 := uop.NewArena(1 << 14)
	rope := nn.NewRoPE(headDim, blockSize, 10000.0)
	m := nn.NewGQAttention(a0, nEmbd, nHead, nKVHead, blockSize, rope)
	rng := rand.New(rand.NewSource(29))
	gqaInitSmall(m, 0.1, rng)

	xData := make([]float32, B*T*int64(nEmbd))
	for i := range xData {
		xData[i] = float32(rng.NormFloat64()) * 0.5
	}

	evalLoss := func() float32 {
		a := uop.NewArena(1 << 18)
		for _, p := range m.Params() {
			p.Load(a)
		}
		x := tensor.NewLeaf(a, []int64{B, T, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
		x.SetData(append([]float32{}, xData...))
		loss := m.Forward(x).Sum(nil, false)
		if err := tensor.Realize(loss); err != nil {
			t.Fatalf("GQA evalLoss Realize: %v", err)
		}
		return loss.Data()[0]
	}

	a := uop.NewArena(1 << 19)
	for _, p := range m.Params() {
		p.Load(a)
	}
	x := tensor.NewLeaf(a, []int64{B, T, int64(nEmbd)}, uop.Dtypes.Float32, "webgpu")
	x.SetData(append([]float32{}, xData...))
	loss := m.Forward(x).Sum(nil, false)

	leaves := []*tensor.Tensor{m.Q.Weight.T, m.K.Weight.T, m.V.Weight.T, m.Proj.Weight.T}
	grads := tensor.Backward(loss, leaves)
	for _, leaf := range leaves {
		if err := tensor.Realize(grads[leaf]); err != nil {
			t.Fatalf("GQA Realize grad: %v", err)
		}
	}
	qGrad := append([]float32{}, grads[m.Q.Weight.T].Data()...)
	kGrad := append([]float32{}, grads[m.K.Weight.T].Data()...)
	vGrad := append([]float32{}, grads[m.V.Weight.T].Data()...)
	projGrad := append([]float32{}, grads[m.Proj.Weight.T].Data()...)

	checkParam := func(p *nn.Parameter, ag []float32, label string, useTol float32) {
		t.Helper()
		n := llamaNCheck
		if n > len(ag) {
			n = len(ag)
		}
		maxRel := float32(0)
		for i := 0; i < n; i++ {
			orig := p.Value[i]
			p.Value[i] = orig + llamaFDH
			lp := evalLoss()
			p.Value[i] = orig - llamaFDH
			lm := evalLoss()
			p.Value[i] = orig
			fd := (lp - lm) / (2 * llamaFDH)
			rel := llamaRelErr(ag[i], fd)
			if rel > maxRel {
				maxRel = rel
			}
			t.Logf("%s[%d]: analytic=%+.6f fd=%+.6f rel=%.2e (tol=%.0e)", label, i, ag[i], fd, rel, useTol)
			if rel > useTol {
				t.Fatalf("%s[%d]: rel=%.2e > tol=%.2e", label, i, rel, useTol)
			}
		}
		t.Logf("%s max-rel=%.2e", label, maxRel)
	}

	// Proj is the final bias-free linear: tight budget, runs unconditionally.
	checkParam(m.Proj.Weight, projGrad, "Proj.Weight", llamaTolTight)

	// Q/K/V flow through the softmax + RoPE chain: looser budget, kernel-heavy
	// backward -> skip on -short.
	if !testing.Short() {
		checkParam(m.Q.Weight, qGrad, "Q.Weight", llamaTolSoftmax)
		checkParam(m.K.Weight, kGrad, "K.Weight", llamaTolSoftmax)
		checkParam(m.V.Weight, vGrad, "V.Weight", llamaTolSoftmax)
	}
}

// TestGQAFDGradG1 FD-checks gradients on the g==1 path.
func TestGQAFDGradG1(t *testing.T) {
	requireGPU(t)
	gqaFDGrad(t, 16, 2, 2, 4)
}

// TestGQAFDGradGgt1 FD-checks gradients on the g>1 path (repeatKV via Expand).
func TestGQAFDGradGgt1(t *testing.T) {
	requireGPU(t)
	gqaFDGrad(t, 16, 2, 1, 4)
}

// TestGQAPanics covers the constructor and Forward guards.
func TestGQAPanics(t *testing.T) {
	requireGPU(t)
	a0 := uop.NewArena(1 << 14)

	goodRope := nn.NewRoPE(8, 4, 10000.0) // headDim 8 = 16/2.

	mustPanicLlama(t, "nEmbd%nHead!=0", func() {
		nn.NewGQAttention(a0, 15, 2, 1, 4, goodRope)
	})
	mustPanicLlama(t, "nHead%nKVHead!=0", func() {
		nn.NewGQAttention(a0, 16, 4, 3, 4, nn.NewRoPE(4, 4, 10000.0))
	})
	mustPanicLlama(t, "nKVHead<=0", func() {
		nn.NewGQAttention(a0, 16, 2, 0, 4, goodRope)
	})
	mustPanicLlama(t, "blockSize<=0", func() {
		nn.NewGQAttention(a0, 16, 2, 1, 0, goodRope)
	})
	mustPanicLlama(t, "rope==nil", func() {
		nn.NewGQAttention(a0, 16, 2, 1, 4, nil)
	})
	mustPanicLlama(t, "rope.HeadDim mismatch", func() {
		// headDim should be 16/2=8, but rope built with 4.
		nn.NewGQAttention(a0, 16, 2, 1, 4, nn.NewRoPE(4, 4, 10000.0))
	})

	// Forward guards.
	m := nn.NewGQAttention(a0, 16, 2, 1, 4, goodRope)
	a := uop.NewArena(1 << 16)
	for _, p := range m.Params() {
		p.Load(a)
	}
	mustPanicLlama(t, "Forward(wrong rank)", func() {
		x := tensor.NewLeaf(a, []int64{2, 16}, uop.Dtypes.Float32, "webgpu")
		m.Forward(x)
	})
	mustPanicLlama(t, "Forward(nEmbd mismatch)", func() {
		x := tensor.NewLeaf(a, []int64{2, 4, 8}, uop.Dtypes.Float32, "webgpu")
		m.Forward(x)
	})
	mustPanicLlama(t, "Forward(T>blockSize)", func() {
		x := tensor.NewLeaf(a, []int64{2, 8, 16}, uop.Dtypes.Float32, "webgpu")
		m.Forward(x)
	})
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. Llama model
// ──────────────────────────────────────────────────────────────────────────────

func llamaInitSmall(m *nn.Llama, scale float32, rng *rand.Rand) {
	for _, p := range m.Params() {
		// RMSNorm weights are 1-D scale params init to 1.0; perturb around it.
		// Distinguish by name is unavailable, so seed everything from a small
		// normal; RMSNorm weights starting near 0 is fine for a forward/Params
		// smoke + determinism check (the FD-correctness of RMSNorm is proven
		// in TestRMSNormFDGrad).
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * scale
		}
	}
}

// evalLlamaOutput runs forward-only and returns a copy of the [B,T,vocab] logits.
func evalLlamaOutput(t *testing.T, m *nn.Llama, idxVals []int32, B, T int64) []float32 {
	t.Helper()
	a := uop.NewArena(1 << 19)
	for _, p := range m.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(idxBits(idxVals))
	y := m.Forward(idx)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Llama Realize: %v", err)
	}
	out := make([]float32, len(y.Data()))
	copy(out, y.Data())
	return out
}

// TestLlamaShape checks NewLlama builds and Forward produces [B,T,vocab].
func TestLlamaShape(t *testing.T) {
	requireGPU(t)

	const (
		vocab     = 8
		nLayer    = 2
		nHead     = 2
		nKVHead   = 1
		nEmbd     = 16
		hidden    = 32
		blockSize = 4
		B         = int64(2)
		T         = int64(4)
	)

	a0 := uop.NewArena(1 << 16)
	m := nn.NewLlama(a0, vocab, nLayer, nHead, nKVHead, nEmbd, hidden, blockSize, 10000.0)
	rng := rand.New(rand.NewSource(31))
	llamaInitSmall(m, 0.05, rng)

	idxVals := make([]int32, B*T)
	for i := range idxVals {
		idxVals[i] = int32(rng.Intn(vocab))
	}

	out := evalLlamaOutput(t, m, idxVals, B, T)
	want := int(B * T * vocab)
	if len(out) != want {
		t.Fatalf("Llama output length %d != %d (expected [B=%d,T=%d,vocab=%d])",
			len(out), want, B, T, vocab)
	}
	t.Logf("Llama forward output [B=%d,T=%d,vocab=%d] = %d elements OK", B, T, vocab, len(out))
}

// TestLlamaParamsCount verifies the tied-head param count (9*nLayer+2) and that
// LMHead.Weight is the SAME *Parameter pointer as Tok.Weight.
func TestLlamaParamsCount(t *testing.T) {
	const (
		vocab     = 8
		nLayer    = 2
		nHead     = 2
		nKVHead   = 1
		nEmbd     = 16
		hidden    = 32
		blockSize = 4
	)

	a0 := uop.NewArena(1 << 16)
	m := nn.NewLlama(a0, vocab, nLayer, nHead, nKVHead, nEmbd, hidden, blockSize, 10000.0)
	ps := m.Params()

	// Per block: Norm1 (1) + Q,K,V,Proj (4) + Norm2 (1) + Gate,Up,Down (3) = 9.
	// Plus Tok.Weight (1) and NormF.Weight (1). LMHead.Weight tied -> not counted.
	want := 9*nLayer + 2
	if len(ps) != want {
		t.Fatalf("Llama.Params(): got %d, want %d", len(ps), want)
	}

	// Tie-weight identity: LMHead.Weight must be the SAME pointer as Tok.Weight.
	if m.LMHead.Weight != m.Tok.Weight {
		t.Fatalf("expected LMHead.Weight to be the same *Parameter as Tok.Weight (weight tying)")
	}
	if !m.TieWeights {
		t.Fatalf("expected m.TieWeights == true")
	}
	// Tok.Weight is first in the param list.
	if ps[0] != m.Tok.Weight {
		t.Fatalf("Params()[0]: expected pointer-identity with Tok.Weight")
	}

	// Verify the full deterministic ordering.
	expected := []*nn.Parameter{m.Tok.Weight}
	for _, blk := range m.Blocks {
		expected = append(expected, blk.Params()...)
	}
	expected = append(expected, m.NormF.Weight)
	if len(expected) != len(ps) {
		t.Fatalf("expected-order length %d != Params() length %d", len(expected), len(ps))
	}
	for i, w := range expected {
		if ps[i] != w {
			t.Fatalf("Params()[%d]: pointer mismatch (parameter ordering changed)", i)
		}
	}
	t.Logf("Llama.Params() returns %d params (9*nLayer=%d + Tok+NormF=2); LMHead tied to Tok OK",
		len(ps), 9*nLayer)
}

// TestLlamaDeterminism runs the same forward three times from a fresh seed and
// asserts sha256-identical logits.
func TestLlamaDeterminism(t *testing.T) {
	requireGPU(t)

	const (
		vocab     = 8
		nLayer    = 2
		nHead     = 2
		nKVHead   = 1
		nEmbd     = 16
		hidden    = 32
		blockSize = 4
		B         = int64(2)
		T         = int64(4)
		seed      = int64(20260618)
	)

	runOnce := func() string {
		a0 := uop.NewArena(1 << 16)
		m := nn.NewLlama(a0, vocab, nLayer, nHead, nKVHead, nEmbd, hidden, blockSize, 10000.0)
		rng := rand.New(rand.NewSource(seed))
		llamaInitSmall(m, 0.05, rng)
		idxVals := make([]int32, B*T)
		for i := range idxVals {
			idxVals[i] = int32(rng.Intn(vocab))
		}
		out := evalLlamaOutput(t, m, idxVals, B, T)
		return llamaHashFloats(out)
	}

	hashes := make([]string, 3)
	for i := range hashes {
		hashes[i] = runOnce()
		t.Logf("run %d: sha256=%s", i+1, hashes[i])
	}
	if hashes[0] != hashes[1] || hashes[1] != hashes[2] {
		t.Fatalf("non-determinism across 3 runs:\n  run1=%s\n  run2=%s\n  run3=%s",
			hashes[0], hashes[1], hashes[2])
	}
	t.Logf("Llama determinism ok (3 runs bit-identical: %s)", hashes[0])
}

// TestLlamaPanics covers the constructor and Forward guards.
func TestLlamaPanics(t *testing.T) {
	requireGPU(t)
	a0 := uop.NewArena(1 << 16)

	mustPanicLlama(t, "NewLlama(nEmbd%nHead!=0)", func() {
		nn.NewLlama(a0, 8, 1, 2, 1, 15, 32, 4, 10000.0)
	})
	mustPanicLlama(t, "NewLlama(nHead%nKVHead!=0)", func() {
		nn.NewLlama(a0, 8, 1, 4, 3, 16, 32, 4, 10000.0)
	})
	mustPanicLlama(t, "NewLlama(vocab<=0)", func() {
		nn.NewLlama(a0, 0, 1, 2, 1, 16, 32, 4, 10000.0)
	})
	mustPanicLlama(t, "NewLlama(hidden<=0)", func() {
		nn.NewLlama(a0, 8, 1, 2, 1, 16, 0, 4, 10000.0)
	})

	// Forward guards.
	m := nn.NewLlama(a0, 8, 1, 2, 1, 16, 32, 4, 10000.0)
	a := uop.NewArena(1 << 17)
	for _, p := range m.Params() {
		p.Load(a)
	}

	mustPanicLlama(t, "Forward(non-rank-2 idx)", func() {
		idx := tensor.NewLeaf(a, []int64{2, 2, 2}, uop.Dtypes.Int32, "webgpu")
		idx.SetData(idxBits([]int32{0, 1, 2, 3, 0, 1, 2, 3}))
		m.Forward(idx)
	})
	mustPanicLlama(t, "Forward(wrong dtype)", func() {
		idx := tensor.NewLeaf(a, []int64{2, 4}, uop.Dtypes.Float32, "webgpu")
		idx.SetData(make([]float32, 8))
		m.Forward(idx)
	})
	mustPanicLlama(t, "Forward(nil idx data)", func() {
		idx := tensor.NewLeaf(a, []int64{2, 4}, uop.Dtypes.Int32, "webgpu")
		m.Forward(idx)
	})
	mustPanicLlama(t, "Forward(T>blockSize)", func() {
		idx := tensor.NewLeaf(a, []int64{1, 8}, uop.Dtypes.Int32, "webgpu")
		idx.SetData(idxBits([]int32{0, 1, 2, 3, 0, 1, 2, 3}))
		m.Forward(idx)
	})
}
