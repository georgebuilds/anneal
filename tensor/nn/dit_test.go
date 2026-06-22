package nn_test

// DiT primitive correctness (CPU interpreter, no GPU needed).
//
//   1. TestUnpatchifyInvertsFold  : Unpatchify exactly inverts PatchEmbed's
//                                   reshape/permute fold (bit-for-bit identity).
//   2. TestAdaLNNormZeroMeanUnitVar: norm-only output rows have mean ~0, var ~1.
//   3. TestModulateMatchesReference: x*(1+scale)+shift matches a host reference,
//                                    and scale=shift=0 is the identity.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ditCPU registers the pure-Go CPU interpreter as the realize backend for the
// duration of a test, so DiT primitive checks run deterministically with no GPU.
func ditCPU(t *testing.T) {
	t.Helper()
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })
}

func TestUnpatchifyInvertsFold(t *testing.T) {
	ditCPU(t)
	const (
		B = int64(2)
		C = int64(3)
		H = int64(8)
		W = int64(8)
		P = int64(4)
	)
	nH, nW := H/P, W/P
	N := nH * nW
	feat := C * P * P

	a := uop.NewArena(1 << 20)
	rng := rand.New(rand.NewSource(7))
	img := make([]float32, int(B*C*H*W))
	for i := range img {
		img[i] = float32(rng.NormFloat64())
	}
	x := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, "cpu")
	x.SetData(append([]float32{}, img...))

	// Fold to tokens [B, N, C*P*P] exactly as PatchEmbed.Forward does (minus the
	// Linear projection): reshape -> permute(0,2,4,1,3,5) -> reshape.
	tok := x.Reshape([]int64{B, C, nH, P, nW, P}).
		Permute([]int{0, 2, 4, 1, 3, 5}).
		Reshape([]int64{B, N, feat})

	back := nn.Unpatchify(tok, H, W, P, C)
	if err := tensor.Realize(back); err != nil {
		t.Fatalf("Realize(back): %v", err)
	}
	got := back.Data()
	if int64(len(got)) != B*C*H*W {
		t.Fatalf("output length %d != %d", len(got), B*C*H*W)
	}
	for i := range img {
		if got[i] != img[i] {
			t.Fatalf("Unpatchify(fold(x)) != x at index %d: got %v want %v", i, got[i], img[i])
		}
	}
}

func TestAdaLNNormZeroMeanUnitVar(t *testing.T) {
	ditCPU(t)
	const (
		B = int64(2)
		N = int64(3)
		D = int64(8)
	)
	a := uop.NewArena(1 << 20)
	rng := rand.New(rand.NewSource(9))
	data := make([]float32, int(B*N*D))
	for i := range data {
		data[i] = float32(rng.NormFloat64())*2.0 + 1.0 // non-zero mean, non-unit scale
	}
	x := tensor.NewLeaf(a, []int64{B, N, D}, uop.Dtypes.Float32, "cpu")
	x.SetData(append([]float32{}, data...))

	y := nn.AdaLNNorm(x, 1e-5)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}
	out := y.Data()
	rows := B * N
	for r := int64(0); r < rows; r++ {
		var s, s2 float64
		for j := int64(0); j < D; j++ {
			v := float64(out[r*D+j])
			s += v
			s2 += v * v
		}
		mean := s / float64(D)
		variance := s2/float64(D) - mean*mean
		if math.Abs(mean) > 1e-3 {
			t.Fatalf("row %d mean %v not ~0", r, mean)
		}
		if math.Abs(variance-1.0) > 1e-2 {
			t.Fatalf("row %d variance %v not ~1", r, variance)
		}
	}
}

func TestModulateMatchesReference(t *testing.T) {
	ditCPU(t)
	const (
		B = int64(2)
		N = int64(2)
		D = int64(4)
	)
	a := uop.NewArena(1 << 20)
	rng := rand.New(rand.NewSource(13))
	xd := make([]float32, int(B*N*D))
	for i := range xd {
		xd[i] = float32(rng.NormFloat64())
	}
	sd := make([]float32, int(B*D))
	hd := make([]float32, int(B*D))
	for i := range sd {
		sd[i] = float32(rng.NormFloat64())
		hd[i] = float32(rng.NormFloat64())
	}

	x := tensor.NewLeaf(a, []int64{B, N, D}, uop.Dtypes.Float32, "cpu")
	x.SetData(append([]float32{}, xd...))
	sc := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, "cpu")
	sc.SetData(append([]float32{}, sd...))
	sh := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, "cpu")
	sh.SetData(append([]float32{}, hd...))

	y := nn.Modulate(x, sc, sh)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}
	out := y.Data()
	for b := int64(0); b < B; b++ {
		for n := int64(0); n < N; n++ {
			for d := int64(0); d < D; d++ {
				want := xd[(b*N+n)*D+d]*(1.0+sd[b*D+d]) + hd[b*D+d]
				got := out[(b*N+n)*D+d]
				if math.Abs(float64(got-want)) > 1e-5 {
					t.Fatalf("Modulate mismatch at b%d n%d d%d: got %v want %v", b, n, d, got, want)
				}
			}
		}
	}
}

// TestDiTPrimitiveGuards exercises the shape/argument guards so a misuse fails
// loudly (and so the new code clears the coverage bar on its error paths).
func TestDiTPrimitiveGuards(t *testing.T) {
	a := uop.NewArena(1 << 16)
	leaf := func(shape ...int64) *tensor.Tensor {
		return tensor.NewLeaf(a, shape, uop.Dtypes.Float32, "cpu")
	}
	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("%s: expected panic, got none", name)
			}
		}()
		fn()
	}

	// Modulate guards.
	mustPanic("Modulate x not rank 3", func() { nn.Modulate(leaf(2, 4), leaf(2, 4), leaf(2, 4)) })
	mustPanic("Modulate scale not rank 2", func() { nn.Modulate(leaf(2, 3, 4), leaf(2, 3, 4), leaf(2, 4)) })
	mustPanic("Modulate cond D mismatch", func() { nn.Modulate(leaf(2, 3, 4), leaf(2, 5), leaf(2, 5)) })

	// Unpatchify guards.
	mustPanic("Unpatchify not rank 3", func() { nn.Unpatchify(leaf(2, 4), 8, 8, 4, 3) })
	mustPanic("Unpatchify patch <= 0", func() { nn.Unpatchify(leaf(2, 4, 48), 8, 8, 0, 3) })
	mustPanic("Unpatchify bad divisibility", func() { nn.Unpatchify(leaf(2, 4, 48), 8, 7, 4, 3) })
	mustPanic("Unpatchify N mismatch", func() { nn.Unpatchify(leaf(2, 5, 48), 8, 8, 4, 3) })
	mustPanic("Unpatchify feat mismatch", func() { nn.Unpatchify(leaf(2, 4, 50), 8, 8, 4, 3) })

	// Constructor guards.
	mustPanic("NewDiTBlock embedDim%nHead", func() { nn.NewDiTBlock(a, 15, 2, 4) })
	mustPanic("NewDiT embedDim%nHead", func() { nn.NewDiT(a, 8, 8, 4, 3, 3, 15, 8, 2, 2) })
	mustPanic("NewDiT bad divisibility", func() { nn.NewDiT(a, 8, 7, 4, 3, 3, 16, 8, 2, 2) })
	mustPanic("NewDiT nonpositive dim", func() { nn.NewDiT(a, 8, 8, 4, 0, 3, 16, 8, 2, 2) })
}

// ── DiT container (S2) ───────────────────────────────────────────────────────

func seedParamsRand(ps []*nn.Parameter, rng *rand.Rand, scale float32) {
	for _, p := range ps {
		for i := range p.Value {
			p.Value[i] = float32(rng.NormFloat64()) * scale
		}
	}
}

func randData(rng *rand.Rand, n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(rng.NormFloat64())
	}
	return d
}

// TestDiTBlockZeroInitIdentity proves the adaLN-zero contract: with the
// modulation projection left at zero (its init), every gate is 0, so the block
// is the EXACT identity regardless of how the attention/MLP weights are seeded.
func TestDiTBlockZeroInitIdentity(t *testing.T) {
	ditCPU(t)
	const (
		B     = int64(2)
		N     = int64(4)
		D     = int64(16)
		nHead = 2
	)
	a := uop.NewArena(1 << 22)
	blk := nn.NewDiTBlock(a, D, nHead, int(N))

	rng := rand.New(rand.NewSource(21))
	// Seed Attn + MLP non-trivially; leave Mod at its zero init (adaLN-zero).
	seedParamsRand(blk.Attn.Params(), rng, 0.1)
	seedParamsRand(blk.MLP.Params(), rng, 0.1)
	for _, p := range blk.Params() {
		p.Load(a)
	}

	xData := randData(rng, int(B*N*D))
	cData := randData(rng, int(B*D))
	x := tensor.NewLeaf(a, []int64{B, N, D}, uop.Dtypes.Float32, "cpu")
	x.SetData(xData)
	c := tensor.NewLeaf(a, []int64{B, D}, uop.Dtypes.Float32, "cpu")
	c.SetData(cData)

	y := blk.Forward(x, c)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}
	out := y.Data()
	for i := range xData {
		if math.Abs(float64(out[i]-xData[i])) > 1e-5 {
			t.Fatalf("adaLN-zero block not identity at init: index %d got %v want %v", i, out[i], xData[i])
		}
	}
}

// TestDiTForwardShape builds a small DiT, seeds every parameter, and checks the
// forward pass produces a finite [B, OutCh, H, W] noise prediction.
func TestDiTForwardShape(t *testing.T) {
	ditCPU(t)
	const (
		B        = int64(2)
		C        = int64(3)
		H        = int64(8)
		W        = int64(8)
		P        = int64(4)
		embedDim = int64(16)
		condDim  = int64(8)
		nLayer   = 2
		nHead    = 2
	)
	outCh := C
	a := uop.NewArena(1 << 23)
	m := nn.NewDiT(a, H, W, P, C, outCh, embedDim, condDim, nLayer, nHead)

	rng := rand.New(rand.NewSource(23))
	seedParamsRand(m.Params(), rng, 0.05)
	for _, p := range m.Params() {
		p.Load(a)
	}

	x := tensor.NewLeaf(a, []int64{B, C, H, W}, uop.Dtypes.Float32, "cpu")
	x.SetData(randData(rng, int(B*C*H*W)))
	cond := tensor.NewLeaf(a, []int64{B, condDim}, uop.Dtypes.Float32, "cpu")
	cond.SetData(randData(rng, int(B*condDim)))

	y := m.Forward(x, cond)
	if err := tensor.Realize(y); err != nil {
		t.Fatalf("Realize(y): %v", err)
	}
	got := y.Shape()
	want := []int64{B, outCh, H, W}
	if len(got) != len(want) {
		t.Fatalf("rank %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("shape[%d]=%d want %d", i, got[i], want[i])
		}
	}
	for i, v := range y.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite output at %d: %v", i, v)
		}
	}
}
