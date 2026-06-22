package tensor

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// Forward-mode autodiff (Jacobian-vector products).
//
// Where Backward (gradient.go) walks the forward UOp DAG in REVERSE accumulating
// adjoints (vector-Jacobian products), JVP walks it FORWARD propagating tangents:
// for each op y = f(x1..xn) with input tangents dx_i, a per-op rule emits the
// output tangent dy. The tangent graph is just more UOps on the same arena, so it
// schedules, fuses, and realizes like anything else; no new IR, no mutation.
//
// This is the capability exact MeanFlow needs (the directional time-derivative of
// the network is one JVP). It is FD-checkable as its own oracle:
//
//	JVP(f, x; v) ≈ (f(x + eps*v) - f(x - eps*v)) / (2*eps).
//
// Coverage (this slice): pointwise ALU, the differentiable unary ops, cast, and
// the shape-movement ops (reshape, expand, permute). Reductions, matmul, where,
// gather, and pad are the next JVP slice; JVP returns a clear error on any op
// without a registered rule so callers know exactly what is missing.

// JVPRule computes the tangent of a single forward op from its sources' tangents.
// tanOf(i) returns the tangent of u.Src(i), or nil when that source's tangent is
// identically zero (rules treat nil as zero). A rule returns nil when the output
// tangent is identically zero.
type JVPRule func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, shapeCache map[uint32][]shape.Sint, device string) *Tensor

var jvpRules = buildJVP()

// JVP computes the Jacobian-vector product of out with respect to the leaves in
// wrt along the direction tangents (one tangent per wrt leaf, same shape). It
// returns the tangent of out, a zero tensor when out does not depend on any
// seeded leaf, or an error if the graph contains an op with no forward-mode rule.
func JVP(out *Tensor, wrt []*Tensor, tangents []*Tensor) (*Tensor, error) {
	if out == nil {
		return nil, fmt.Errorf("tensor: JVP: nil out")
	}
	if len(wrt) != len(tangents) {
		return nil, fmt.Errorf("tensor: JVP: wrt/tangents length mismatch (%d vs %d)", len(wrt), len(tangents))
	}
	device := out.device
	a := out.arena()

	topo := uop.TopoSort(out.node)
	shapeCache := make(map[uint32][]shape.Sint, len(topo))
	for _, u := range topo {
		shapeOfNode(u, shapeCache)
	}

	// Seed leaf tangents.
	tanMap := make(map[uint32]*Tensor, len(topo))
	for i, w := range wrt {
		if w == nil || tangents[i] == nil {
			continue
		}
		tanMap[w.node.Index()] = tangents[i]
	}

	for _, u := range topo {
		if _, seeded := tanMap[u.Index()]; seeded {
			continue // a seeded leaf keeps its tangent
		}
		// Non-float outputs (comparisons, indices) carry no tangent.
		if !u.DType().IsFloat() {
			continue
		}
		n := u.NSrc()
		has := false
		for j := 0; j < n; j++ {
			if tanMap[u.Src(j).Index()] != nil {
				has = true
				break
			}
		}
		if !has {
			continue // independent of all seeds -> zero tangent
		}
		rule, ok := jvpRules[u.Op()]
		if !ok {
			return nil, fmt.Errorf("tensor: JVP: no forward-mode rule for op %s", u.Op())
		}
		nodeT := wrapGradTensor(u, shapeCache[u.Index()], u.DType(), device)
		uu := u
		tanOf := func(i int) *Tensor {
			if i < 0 || i >= uu.NSrc() {
				return nil
			}
			return tanMap[uu.Src(i).Index()]
		}
		if t := rule(u, nodeT, tanOf, shapeCache, device); t != nil {
			tanMap[u.Index()] = t
		}
	}

	if t := tanMap[out.node.Index()]; t != nil {
		return t, nil
	}
	sh := shapeCache[out.node.Index()]
	if sh == nil {
		sh = []shape.Sint{}
	}
	return FullSints(a, sh, 0.0, out.dtype, device), nil
}

// jvpSrc reconstructs a Tensor handle for u.Src(i) (the primal value).
func jvpSrc(u uop.UOp, i int, shapeCache map[uint32][]shape.Sint, device string) *Tensor {
	s := u.Src(i)
	return wrapGradTensor(s, shapeCache[s.Index()], s.DType(), device)
}

// jvpK builds a constant of u's output shape.
func jvpK(u uop.UOp, v float64, shapeCache map[uint32][]shape.Sint, device string) *Tensor {
	return FullSints(u.Arena(), shapeCache[u.Index()], v, u.DType(), device)
}

// addTan sums two tangents, treating nil as zero.
func addTan(a, b *Tensor) *Tensor {
	switch {
	case a == nil:
		return b
	case b == nil:
		return a
	default:
		return a.Add(b)
	}
}

func buildJVP() map[uop.Op]JVPRule {
	m := map[uop.Op]JVPRule{}

	// Single-source rules use tanOf(0) directly: the driver only dispatches a rule
	// once at least one source carries a tangent, so for a one-source op tanOf(0)
	// is always non-nil (this mirrors how the gradient ruleset uses adj).

	// ── Identity-ish ──────────────────────────────────────────────────────────
	m[uop.OpContiguous] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0)
	}
	// A dispatched cast's source carries a tangent, hence is float; reinterpret it.
	castJVP := JVPRule(func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Cast(u.DType())
	})
	m[uop.OpCast] = castJVP

	// ── Unary ALU ─────────────────────────────────────────────────────────────
	m[uop.OpNeg] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Neg()
	}
	// d(2^x) = 2^x * ln2 * dx   (node IS 2^x)
	m[uop.OpExp2] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Mul(nodeT).Mul(jvpK(u, ln2, sc, dev))
	}
	// d(log2 x) = dx / (x * ln2)
	m[uop.OpLog2] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Div(jvpSrc(u, 0, sc, dev).Mul(jvpK(u, ln2, sc, dev)))
	}
	// d(sin x) = cos(x) * dx = sin(x + pi/2) * dx
	m[uop.OpSin] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Mul(jvpSrc(u, 0, sc, dev).Add(jvpK(u, math.Pi/2, sc, dev)).Sin())
	}
	// d(sqrt x) = dx / (2*sqrt(x)) = dx / (2*node)
	m[uop.OpSqrt] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Div(jvpK(u, 2.0, sc, dev).Mul(nodeT))
	}
	// d(1/x) = -dx / x^2 = -dx * node^2
	m[uop.OpReciprocal] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).Neg().Mul(nodeT).Mul(nodeT)
	}
	// d(erf x) = (2/sqrt(pi)) * exp(-x^2) * dx
	m[uop.OpErf] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		const twoOverSqrtPi = 1.1283791670955126
		x := jvpSrc(u, 0, sc, dev)
		return tanOf(0).Mul(jvpK(u, twoOverSqrtPi, sc, dev)).Mul(x.Mul(x).Neg().Exp())
	}

	// ── Binary ALU ────────────────────────────────────────────────────────────
	m[uop.OpAdd] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return addTan(tanOf(0), tanOf(1))
	}
	m[uop.OpSub] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		// At least one source carries a tangent (driver contract).
		tA, tB := tanOf(0), tanOf(1)
		switch {
		case tB == nil:
			return tA
		case tA == nil:
			return tB.Neg()
		default:
			return tA.Sub(tB)
		}
	}
	// d(a*b) = da*b + a*db
	m[uop.OpMul] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		tA, tB := tanOf(0), tanOf(1)
		var r *Tensor
		if tA != nil {
			r = tA.Mul(jvpSrc(u, 1, sc, dev))
		}
		if tB != nil {
			r = addTan(r, jvpSrc(u, 0, sc, dev).Mul(tB))
		}
		return r
	}

	// ── Movement (linear: apply the same movement to the tangent) ─────────────
	m[uop.OpReshape] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return tanOf(0).ReshapeSints(sc[u.Index()])
	}
	m[uop.OpExpand] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		return BroadcastToSints(tanOf(0), sc[u.Index()])
	}
	m[uop.OpPermute] = func(u uop.UOp, nodeT *Tensor, tanOf func(int) *Tensor, sc map[uint32][]shape.Sint, dev string) *Tensor {
		permArg := u.Arg().([]int64)
		perm := make([]int, len(permArg))
		for i, p := range permArg {
			perm[i] = int(p)
		}
		return tanOf(0).Permute(perm)
	}

	return m
}
