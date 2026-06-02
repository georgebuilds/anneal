package uop

import "fmt"

// TopoSort returns all nodes reachable from root in forward topological order:
// each node appears after all its sources. Iterative post-order DFS, safe for
// graphs of arbitrary depth. The traversal order on independent siblings is
// deterministic (source-index order), so callers that depend on a structural
// traversal can rely on it.
func TopoSort(root UOp) []UOp {
	seen := make(map[uint32]bool)
	var order []UOp

	type frame struct {
		u       UOp
		nextSrc int
	}
	stack := []frame{{root, 0}}

	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u
		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}
		pushed := false
		for f.nextSrc < u.NSrc() {
			child := u.Src(f.nextSrc)
			f.nextSrc++
			if !seen[child.Index()] {
				stack = append(stack, frame{child, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			order = append(order, u)
			stack = stack[:len(stack)-1]
		}
	}
	return order
}

// RebuildSymBound reconstructs the UOp bound expression for a symbolic ShapeDim
// from its (VarName, Mul) encoding. Interning ensures the rebuilt node aliases
// the original whenever the original was constructed in canonical orientation
// (DefineVar on the left of OpMul). Panics if VarName is not registered in a.
//
// Callers that need a shape.Sint wrapper around the result should use
// shape.SintFromShapeDim, which handles the concrete-vs-symbolic split.
func RebuildSymBound(a *Arena, d ShapeDim) UOp {
	defVar, ok := a.FindDefineVar(d.VarName)
	if !ok {
		panic(fmt.Sprintf("uop: RebuildSymBound: DefineVar %q not found in arena", d.VarName))
	}
	if d.Mul <= 1 {
		return defVar
	}
	mulConst := a.New(OpConst, Dtypes.Index, nil, d.Mul, nil)
	return a.New(OpMul, Dtypes.Index, []UOp{defVar, mulConst}, nil, nil)
}
