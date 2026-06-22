package tensor_test

// Forward-mode autodiff (JVP) correctness via central finite differences, on the
// pure-Go CPU interpreter. JVP(f, x; v) must match (f(x+eps*v) - f(x-eps*v))/(2eps).
// Lives in the external test package so it can import backend/cpu (which imports
// tensor) without an import cycle; it exercises JVP through the exported API only.

import (
	"math"
	"testing"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestJVPFiniteDifference(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1 << 22)
	const n = 6
	xd := []float32{0.5, -0.3, 1.2, 0.8, -1.1, 0.4}
	vd := []float32{0.2, 0.7, -0.5, 0.1, 0.9, -0.3}

	leaf := func(data []float32) *tensor.Tensor {
		x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, "cpu")
		x.SetData(append([]float32{}, data...))
		return x
	}
	kf := func(x *tensor.Tensor, v float64) *tensor.Tensor {
		return tensor.FullSints(a, x.ShapeSints(), v, x.DType(), "cpu")
	}

	// A composite over CPU-interpreter-supported ops exercising pointwise
	// (add/sub/mul/div/neg), unary (sqrt/recip/exp), and movement
	// (reshape/expand/permute) JVP rules:
	//   a1 = sqrt(x*x+1) - 1/(x+2) + exp(x)/(x+3) + x^3 + x          [6]
	//   r  = reshape a1 -> [6,1], expand -> [6,2], permute -> [2,6], then square.
	// (The Sin/Log2/Erf JVP rules mirror the FD-verified reverse-mode rules; the
	// CPU interpreter does not implement Sin, so they are not exercised here.)
	f := func(x *tensor.Tensor) *tensor.Tensor {
		a1 := x.Mul(x).Add(kf(x, 1)).Sqrt().
			Sub(x.Add(kf(x, 2)).Recip()).
			Add(x.Exp().Div(x.Add(kf(x, 3)))).
			Add(x.Mul(x).Mul(x)).
			Sub(x.Neg())
		r := tensor.BroadcastToSints(
			a1.ReshapeSints([]shape.Sint{shape.Const(n), shape.Const(1)}),
			[]shape.Sint{shape.Const(n), shape.Const(2)},
		)
		r = r.Permute([]int{1, 0}) // [2, 6]
		return r.Mul(r)
	}

	x := leaf(xd)
	out := f(x)
	v := leaf(vd)

	jt, err := tensor.JVP(out, []*tensor.Tensor{x}, []*tensor.Tensor{v})
	if err != nil {
		t.Fatalf("JVP: %v", err)
	}
	if err := tensor.Realize(jt); err != nil {
		t.Fatalf("realize JVP: %v", err)
	}
	got := jt.Data()

	eps := float32(1e-3)
	xp := make([]float32, n)
	xm := make([]float32, n)
	for i := range xd {
		xp[i] = xd[i] + eps*vd[i]
		xm[i] = xd[i] - eps*vd[i]
	}
	fp := f(leaf(xp))
	fm := f(leaf(xm))
	if err := tensor.Realize(fp, fm); err != nil {
		t.Fatalf("realize FD: %v", err)
	}
	fpd, fmd := fp.Data(), fm.Data()
	if len(got) != len(fpd) {
		t.Fatalf("length mismatch: jvp=%d fd=%d", len(got), len(fpd))
	}
	for i := range got {
		fd := (fpd[i] - fmd[i]) / (2 * eps)
		if d := math.Abs(float64(got[i] - fd)); d > 1e-2 {
			t.Fatalf("jvp[%d]=%v fd=%v (|diff|=%v)", i, got[i], fd, d)
		}
	}
}

// TestJVPUnknownOpErrors confirms JVP fails loudly (not silently wrong) on an op
// without a forward-mode rule, so callers know exactly what coverage is missing.
func TestJVPUnknownOpErrors(t *testing.T) {
	a := uop.NewArena(1 << 16)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	v := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	v.SetData([]float32{1, 1, 1, 1})
	// Flip has no JVP rule yet; JVP must error rather than silently produce a
	// wrong (zero) tangent.
	out := x.Flip([]bool{true})
	if _, err := tensor.JVP(out, []*tensor.Tensor{x}, []*tensor.Tensor{v}); err == nil {
		t.Fatal("expected JVP to error on an op with no forward-mode rule (flip)")
	}
}

// TestJVPMinMaxShrinkPad FD-checks the S3 rules on the CPU interpreter: binary
// max/min (the softmax and SiLU clamps on the DiT path), shrink (the modulation
// and QKV splits), and pad (the patch-embed conv). Constants are chosen so no
// element ties its compare operand, where the true derivative is undefined.
// TestKeepdimReduceGradientFD FD-checks the reverse-mode gradient through a
// keepdim reduce that feeds another reduce (the pattern that exposed the
// shapeOfNode keepdim bug), realized on the CPU interpreter.
func TestKeepdimReduceGradientFD(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1 << 22)
	const B, K = 2, 4
	xd := []float32{0.5, -0.3, 1.2, 0.8, -1.1, 0.4, 0.2, -0.7} // [B,K]

	leaf := func(data []float32) *tensor.Tensor {
		x := tensor.NewLeaf(a, []int64{B, K}, uop.Dtypes.Float32, "cpu")
		x.SetData(append([]float32{}, data...))
		return x
	}
	// f(x) = mean_all( (mean_k(x, keepdim=true))^2 ): keepdim reduce -> square -> reduce.
	f := func(x *tensor.Tensor) *tensor.Tensor {
		m := x.Mean([]int{1}, true) // [B,1] keepdim
		return m.Mul(m).Mean(nil, false)
	}

	x := leaf(xd)
	grads := tensor.Backward(f(x), []*tensor.Tensor{x})
	g := grads[x]
	if g == nil {
		t.Fatal("no gradient for x")
	}
	if err := tensor.Realize(g); err != nil {
		t.Fatalf("realize grad: %v", err)
	}
	got := g.Data()

	eps := float32(1e-3)
	fval := func(data []float32) float32 {
		l := f(leaf(data))
		if err := tensor.Realize(l); err != nil {
			t.Fatalf("realize fd: %v", err)
		}
		return l.Data()[0]
	}
	for i := range xd {
		xp := append([]float32{}, xd...)
		xp[i] += eps
		xm := append([]float32{}, xd...)
		xm[i] -= eps
		fd := (fval(xp) - fval(xm)) / (2 * eps)
		if d := math.Abs(float64(got[i] - fd)); d > 1e-2 {
			t.Fatalf("grad[%d]=%v fd=%v (|diff|=%v)", i, got[i], fd, d)
		}
	}
}

func TestJVPMinMaxShrinkPad(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1 << 22)
	const n = 6
	xd := []float32{0.5, -0.3, 1.2, 0.8, -1.1, 0.4}
	vd := []float32{0.2, 0.7, -0.5, 0.1, 0.9, -0.3}
	c1d := []float32{0.4, 0.1, 1.5, -0.2, -0.5, 0.9}  // max operand
	c2d := []float32{0.45, -0.5, 0.0, 1.0, -2.0, 0.2} // min operand

	leaf := func(data []float32) *tensor.Tensor {
		x := tensor.NewLeaf(a, []int64{n}, uop.Dtypes.Float32, "cpu")
		x.SetData(append([]float32{}, data...))
		return x
	}
	c1 := leaf(c1d) // constants w.r.t. JVP (not seeded)
	c2 := leaf(c2d)

	// f(x) = pad( shrink( max(x,c1) + min(x,c2) ) )^2
	f := func(x *tensor.Tensor) *tensor.Tensor {
		s := x.Maximum(c1).Add(x.Minimum(c2)) // [6]
		sh := s.Shrink([][2]int64{{1, 5}})    // [4]
		pd := sh.Pad([][2]int64{{1, 1}})      // [6] (zeros at the ends)
		return pd.Mul(pd)
	}

	x := leaf(xd)
	out := f(x)
	v := leaf(vd)

	jt, err := tensor.JVP(out, []*tensor.Tensor{x}, []*tensor.Tensor{v})
	if err != nil {
		t.Fatalf("JVP: %v", err)
	}
	if err := tensor.Realize(jt); err != nil {
		t.Fatalf("realize JVP: %v", err)
	}
	got := jt.Data()

	eps := float32(1e-3)
	xp := make([]float32, n)
	xm := make([]float32, n)
	for i := range xd {
		xp[i] = xd[i] + eps*vd[i]
		xm[i] = xd[i] - eps*vd[i]
	}
	fp := f(leaf(xp))
	fm := f(leaf(xm))
	if err := tensor.Realize(fp, fm); err != nil {
		t.Fatalf("realize FD: %v", err)
	}
	fpd, fmd := fp.Data(), fm.Data()
	for i := range got {
		fd := (fpd[i] - fmd[i]) / (2 * eps)
		if d := math.Abs(float64(got[i] - fd)); d > 2e-2 {
			t.Fatalf("jvp[%d]=%v fd=%v (|diff|=%v)", i, got[i], fd, d)
		}
	}
}

// TestJVPReduceMatmulWhere FD-checks the S2 rules on the CPU interpreter:
// matmul (which lowers to Mul + sum-reduce), explicit sum-reduce, where (as a
// relu), and max-reduce. Inputs are chosen so no element sits on the relu kink
// (h != 0) or a max tie, where the true derivative is undefined.
func TestJVPReduceMatmulWhere(t *testing.T) {
	dev, err := cpu.Open()
	if err != nil {
		t.Fatalf("cpu.Open: %v", err)
	}
	prev := tensor.DefaultExecutor
	tensor.DefaultExecutor = dev
	t.Cleanup(func() { tensor.DefaultExecutor = prev })

	a := uop.NewArena(1 << 22)
	const M, K, N = 2, 3, 4
	xd := []float32{0.5, -0.3, 1.2, 0.8, -1.1, 0.4} // [M,K]
	vd := []float32{0.2, 0.7, -0.5, 0.1, 0.9, -0.3} // tangent [M,K]
	wd := []float32{
		0.10, -0.20, 0.30, -0.10,
		0.40, 0.20, -0.30, 0.15,
		-0.25, 0.35, 0.10, -0.40,
	} // [K,N]

	leaf := func(sh []int64, data []float32) *tensor.Tensor {
		x := tensor.NewLeaf(a, sh, uop.Dtypes.Float32, "cpu")
		x.SetData(append([]float32{}, data...))
		return x
	}
	W := leaf([]int64{K, N}, wd) // constant w.r.t. JVP (not seeded)

	// f(x) = sum(relu(x @ W), axis=1) + max(x, axis=1)
	f := func(x *tensor.Tensor) *tensor.Tensor {
		h := x.Matmul(W) // [M,N]
		z := tensor.FullSints(a, h.ShapeSints(), 0, h.DType(), "cpu")
		relu := tensor.Where(h.CmpLt(z), z, h) // [M,N]
		s := relu.Sum([]int{1}, false)         // [M]
		mx := x.Max([]int{1}, false)           // [M]
		return s.Add(mx)
	}

	x := leaf([]int64{M, K}, xd)
	out := f(x)
	v := leaf([]int64{M, K}, vd)

	jt, err := tensor.JVP(out, []*tensor.Tensor{x}, []*tensor.Tensor{v})
	if err != nil {
		t.Fatalf("JVP: %v", err)
	}
	if err := tensor.Realize(jt); err != nil {
		t.Fatalf("realize JVP: %v", err)
	}
	got := jt.Data()

	eps := float32(1e-3)
	xp := make([]float32, len(xd))
	xm := make([]float32, len(xd))
	for i := range xd {
		xp[i] = xd[i] + eps*vd[i]
		xm[i] = xd[i] - eps*vd[i]
	}
	fp := f(leaf([]int64{M, K}, xp))
	fm := f(leaf([]int64{M, K}, xm))
	if err := tensor.Realize(fp, fm); err != nil {
		t.Fatalf("realize FD: %v", err)
	}
	fpd, fmd := fp.Data(), fm.Data()
	if len(got) != len(fpd) {
		t.Fatalf("length mismatch: jvp=%d fd=%d", len(got), len(fpd))
	}
	for i := range got {
		fd := (fpd[i] - fmd[i]) / (2 * eps)
		if d := math.Abs(float64(got[i] - fd)); d > 2e-2 {
			t.Fatalf("jvp[%d]=%v fd=%v (|diff|=%v)", i, got[i], fd, d)
		}
	}
}

// TestJVPDriverGuards covers the driver's input-validation paths.
func TestJVPDriverGuards(t *testing.T) {
	a := uop.NewArena(1 << 16)
	x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	x.SetData([]float32{1, 2, 3, 4})
	v := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	v.SetData([]float32{1, 1, 1, 1})
	out := x.Mul(x)

	if _, err := tensor.JVP(nil, []*tensor.Tensor{x}, []*tensor.Tensor{v}); err == nil {
		t.Error("expected an error for nil out")
	}
	if _, err := tensor.JVP(out, []*tensor.Tensor{x}, nil); err == nil {
		t.Error("expected an error for wrt/tangents length mismatch")
	}
	// A nil wrt/tangent entry is skipped (not seeded); out then has no seeded
	// dependency, so JVP returns a zero tangent rather than erroring.
	jt, err := tensor.JVP(out, []*tensor.Tensor{nil}, []*tensor.Tensor{nil})
	if err != nil {
		t.Fatalf("nil-seed entry: %v", err)
	}
	if jt == nil {
		t.Fatal("nil-seed entry: expected a zero tangent, got nil")
	}
}

// TestJVPRuleCoverage builds (does not realize) tangent graphs over the remaining
// rules so each rule closure executes: the transcendental unary rules (the CPU
// interpreter cannot realize Sin, but the forward tangent build still runs the
// rule), cast, contiguous, the nil-source-tangent branches of the binary rules
// (a constant operand has zero tangent), and the zero-tangent output path.
func TestJVPRuleCoverage(t *testing.T) {
	a := uop.NewArena(1 << 20)
	mk := func() (*tensor.Tensor, *tensor.Tensor) {
		x := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
		x.SetData([]float32{0.5, 1.0, 1.5, 2.0})
		v := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
		v.SetData([]float32{1, 1, 1, 1})
		return x, v
	}
	k := func(x *tensor.Tensor, c float64) *tensor.Tensor {
		return tensor.FullSints(a, x.ShapeSints(), c, x.DType(), "cpu")
	}
	nonNil := func(name string, out, x, v *tensor.Tensor) {
		jt, err := tensor.JVP(out, []*tensor.Tensor{x}, []*tensor.Tensor{v})
		if err != nil {
			t.Fatalf("%s: JVP error: %v", name, err)
		}
		if jt == nil {
			t.Fatalf("%s: nil tangent", name)
		}
	}

	// Unary rules (build-only):
	x, v := mk()
	nonNil("sin", x.Sin(), x, v)
	x, v = mk()
	nonNil("log2", x.Log2(), x, v)
	x, v = mk()
	nonNil("erf", x.Erf(), x, v)
	x, v = mk()
	nonNil("contiguous", x.Contiguous(), x, v)
	x, v = mk()
	nonNil("cast", x.Cast(uop.Dtypes.Float16), x, v)

	// Binary rules with one constant operand exercise the nil-source-tangent
	// branches (a constant carries no tangent):
	x, v = mk()
	nonNil("sub-constRHS", x.Sub(k(x, 3)), x, v)
	x, v = mk()
	nonNil("sub-constLHS", k(x, 3).Sub(x), x, v)
	x, v = mk()
	nonNil("mul-constRHS", x.Mul(k(x, 3)), x, v)
	x, v = mk()
	nonNil("mul-constLHS", k(x, 3).Mul(x), x, v)
	x, v = mk()
	nonNil("div-constRHS", x.Div(k(x, 3)), x, v)
	x, v = mk()
	nonNil("div-constLHS", k(x, 3).Div(x), x, v)

	// where with a constant false-branch exercises the nil-tangent path of the
	// third source.
	x, v = mk()
	nonNil("where-constFalse", tensor.Where(x.CmpLt(k(x, 1)), x, k(x, 0)), x, v)

	// max with a constant FIRST operand exercises the nil-tangent path of the
	// min/max rule's first source.
	x, v = mk()
	nonNil("max-constFirst", k(x, 1).Maximum(x), x, v)

	// A non-float intermediate (an int cast) carries no tangent: the only path to
	// the output runs through it, so the tangent is zero. Exercises the non-float
	// skip in the driver.
	x, v = mk()
	nonNil("nonfloat-skip", x.Cast(uop.Dtypes.Int32).Cast(uop.Dtypes.Float32), x, v)

	// Output independent of the seeded leaf -> a zero tangent (not nil, not error).
	x, v = mk()
	y := tensor.NewLeaf(a, []int64{4}, uop.Dtypes.Float32, "cpu")
	y.SetData([]float32{1, 2, 3, 4})
	jt, err := tensor.JVP(y.Mul(y), []*tensor.Tensor{x}, []*tensor.Tensor{v})
	if err != nil {
		t.Fatalf("zero-tangent: %v", err)
	}
	if jt == nil {
		t.Fatal("zero-tangent: expected a zero tensor, got nil")
	}
}
