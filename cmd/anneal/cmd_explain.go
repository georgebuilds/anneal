package main

import (
	"fmt"
	"io"
	"os"
	"strings"
)

func explainCmd(args []string) int {
	return explainCmdW(args, os.Stdout)
}

func explainCmdW(args []string, w io.Writer) int {
	_, rest, err := parseFlags("explain", args)
	if err != nil {
		fmt.Fprintln(w, err)
		return 1
	}

	if len(rest) == 0 {
		fmt.Fprintln(w, "usage: anneal explain <op>")
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "examples: anneal explain add, anneal explain matmul.backward, anneal explain symbolic")
		return 1
	}

	query := strings.ToLower(rest[0])

	// Special case: "symbolic" keyword shows overview of all symbolic rules.
	if query == "symbolic" {
		fmt.Fprint(w, symbolicOverview())
		return 0
	}

	// Special case: matmul / matmul.backward
	if query == "matmul" || query == "matmul.backward" {
		fmt.Fprint(w, matmulExplain())
		return 0
	}

	// Look up in rule registry.
	var matched []ruleEntry
	for _, r := range allRules {
		for _, op := range r.ops {
			if op == query {
				matched = append(matched, r)
				break
			}
		}
	}

	if len(matched) == 0 {
		fmt.Fprintf(w, "op %q not found — try 'anneal explain symbolic' to list all rule groups\n", rest[0])
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "known ops: add, sub, mul, fdiv, neg, exp2, log2, sin, sqrt, reciprocal,")
		fmt.Fprintln(w, "           where, cast, reduceaxis, reshape, expand, permute, pad, shrink,")
		fmt.Fprintln(w, "           flip, max, matmul, matmul.backward, symbolic")
		return 1
	}

	// Group by kind: symbolic first, then gradient.
	var symRules, gradRules []ruleEntry
	for _, r := range matched {
		switch r.kind {
		case "symbolic":
			symRules = append(symRules, r)
		case "gradient":
			gradRules = append(gradRules, r)
		}
	}

	// Collect sources.
	sources := collectSources(matched)

	// Find canonical display name (Op with capital).
	displayName := canonicalOpName(query)
	fmt.Fprintf(w, "op: %s\n", displayName)
	if len(sources) > 0 {
		fmt.Fprintf(w, "sources: %s\n", strings.Join(sources, ", "))
	}
	fmt.Fprintln(w)

	if len(symRules) > 0 {
		fmt.Fprintln(w, "symbolic rules:")
		for _, r := range symRules {
			fmt.Fprintf(w, "  %-40s %s\n", r.pattern, r.description)
		}
		fmt.Fprintln(w)
	}

	if len(gradRules) > 0 {
		fmt.Fprintln(w, "gradient rules:")
		for _, r := range gradRules {
			fmt.Fprintf(w, "  %-40s %s\n", r.pattern, r.description)
		}
		fmt.Fprintln(w)
	}

	return 0
}

// ruleEntry represents one rewrite or gradient rule.
type ruleEntry struct {
	ops         []string // lowercase op names
	kind        string   // "symbolic" or "gradient"
	pattern     string   // short pattern like "x + 0 → x"
	description string   // human-readable explanation
	source      string   // file:function reference
	handler     string   // handler name in symbolic.upat (empty for gradient rules)
}

// collectSources deduplicates source references from matched rules.
func collectSources(rules []ruleEntry) []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range rules {
		if r.source != "" && !seen[r.source] {
			seen[r.source] = true
			out = append(out, r.source)
		}
	}
	return out
}

// canonicalOpName returns the display-cased name for a lowercase op query.
func canonicalOpName(q string) string {
	m := map[string]string{
		"add":        "Add",
		"sub":        "Sub",
		"mul":        "Mul",
		"fdiv":       "FDiv",
		"neg":        "Neg",
		"exp2":       "Exp2",
		"log2":       "Log2",
		"sin":        "Sin",
		"sqrt":       "Sqrt",
		"reciprocal": "Reciprocal",
		"where":      "Where",
		"cast":       "Cast",
		"reduceaxis": "ReduceAxis",
		"reshape":    "Reshape",
		"expand":     "Expand",
		"permute":    "Permute",
		"pad":        "Pad",
		"shrink":     "Shrink",
		"flip":       "Flip",
		"max":        "Max",
	}
	if n, ok := m[q]; ok {
		return n
	}
	return q
}

// allRules is the static rule registry.
//
// Symbolic entries are verified against rewrite/rules/symbolic.upat by
// TestUpatDriftCheck: the set of (op, handler) pairs here must exactly match
// the pairs the .upat file defines for each covered op.  Adding or removing a
// .upat rule without updating this table will make that test fail.
var allRules = []ruleEntry{
	// ── Add ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"add"},
		kind:        "symbolic",
		pattern:     "x + 0 → x",
		description: "additive identity (int and float)",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnX",
	},
	{
		ops:         []string{"add"},
		kind:        "symbolic",
		pattern:     "x + 0.0 → x",
		description: "additive identity (float)",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnX",
	},
	{
		ops:         []string{"add"},
		kind:        "symbolic",
		pattern:     "Const(a) + Const(b) → Const(a+b)",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldBinary",
	},
	{
		ops:         []string{"add"},
		kind:        "symbolic",
		pattern:     "bool + bool → bool | bool",
		description: "boolean algebra: addition becomes OR",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hBoolAdd",
	},
	{
		ops:         []string{"add"},
		kind:        "symbolic",
		pattern:     "x + y → y + x",
		description: "commutative canonicalization (const moves to src[1])",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hCanonicalize",
	},
	{
		ops:         []string{"add"},
		kind:        "gradient",
		pattern:     "∂(a+b)/∂a = adj",
		description: "adjoint passes through unchanged",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"add"},
		kind:        "gradient",
		pattern:     "∂(a+b)/∂b = adj",
		description: "adjoint passes through unchanged",
		source:      "tensor/gradient.go",
	},

	// ── Sub ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"sub"},
		kind:        "symbolic",
		pattern:     "x - 0 → x",
		description: "subtractive identity",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnX",
	},
	{
		ops:         []string{"sub"},
		kind:        "symbolic",
		pattern:     "x - x → 0",
		description: "self-cancellation",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hSubSelf",
	},
	{
		ops:         []string{"sub"},
		kind:        "symbolic",
		pattern:     "Const(a) - Const(b) → Const(a-b)",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldBinary",
	},
	{
		ops:         []string{"sub"},
		kind:        "gradient",
		pattern:     "∂(a-b)/∂a = adj",
		description: "adjoint passes through",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"sub"},
		kind:        "gradient",
		pattern:     "∂(a-b)/∂b = -adj",
		description: "adjoint negated for subtracted operand",
		source:      "tensor/gradient.go",
	},

	// ── Mul ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"mul"},
		kind:        "symbolic",
		pattern:     "x * 1 → x",
		description: "multiplicative identity",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnX",
	},
	{
		ops:         []string{"mul"},
		kind:        "symbolic",
		pattern:     "x * 0 → 0",
		description: "multiplicative absorbing element",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hMulZero",
	},
	{
		ops:         []string{"mul"},
		kind:        "symbolic",
		pattern:     "bool * bool → bool & bool",
		description: "boolean algebra: multiplication becomes AND",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hBoolMul",
	},
	{
		ops:         []string{"mul"},
		kind:        "symbolic",
		pattern:     "Const(a) * Const(b) → Const(a*b)",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldBinary",
	},
	{
		ops:         []string{"mul"},
		kind:        "symbolic",
		pattern:     "x * y → y * x",
		description: "commutative canonicalization (const moves to src[1])",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hCanonicalize",
	},
	{
		ops:         []string{"mul"},
		kind:        "gradient",
		pattern:     "∂(a·b)/∂a = adj·b",
		description: "product rule: multiply adj by the other operand",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"mul"},
		kind:        "gradient",
		pattern:     "∂(a·b)/∂b = adj·a",
		description: "product rule: multiply adj by the other operand",
		source:      "tensor/gradient.go",
	},

	// ── FDiv ─────────────────────────────────────────────────────────────────
	{
		ops:         []string{"fdiv"},
		kind:        "symbolic",
		pattern:     "Const(a) / Const(b) → Const(a/b)",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldBinary",
	},
	{
		ops:         []string{"fdiv"},
		kind:        "gradient",
		pattern:     "∂(a/b)/∂a = adj/b",
		description: "quotient rule numerator derivative",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"fdiv"},
		kind:        "gradient",
		pattern:     "∂(a/b)/∂b = -adj·a/b²",
		description: "quotient rule denominator derivative",
		source:      "tensor/gradient.go",
	},

	// ── Neg ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"neg"},
		kind:        "symbolic",
		pattern:     "Neg(Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldUnary",
	},
	{
		ops:         []string{"neg"},
		kind:        "gradient",
		pattern:     "∂(-x)/∂x = -adj",
		description: "negation flips the adjoint sign",
		source:      "tensor/gradient.go",
	},

	// ── Exp2 ─────────────────────────────────────────────────────────────────
	{
		ops:         []string{"exp2"},
		kind:        "symbolic",
		pattern:     "Exp2(Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldUnary",
	},
	{
		ops:         []string{"exp2"},
		kind:        "gradient",
		pattern:     "∂(2^x)/∂x = 2^x·ln2",
		description: "node IS 2^x; multiply adjoint by node and ln2",
		source:      "tensor/gradient.go",
	},

	// ── Log2 ─────────────────────────────────────────────────────────────────
	{
		ops:         []string{"log2"},
		kind:        "symbolic",
		pattern:     "Log2(Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldUnary",
	},
	{
		ops:         []string{"log2"},
		kind:        "gradient",
		pattern:     "∂(log₂x)/∂x = 1/(x·ln2)",
		description: "derivative of base-2 logarithm",
		source:      "tensor/gradient.go",
	},

	// ── Sin ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"sin"},
		kind:        "symbolic",
		pattern:     "Sin(Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldUnary",
	},
	{
		ops:         []string{"sin"},
		kind:        "gradient",
		pattern:     "∂(sin x)/∂x = sin(x+π/2)",
		description: "cos via phase-shift identity",
		source:      "tensor/gradient.go",
	},

	// ── Sqrt ─────────────────────────────────────────────────────────────────
	{
		ops:         []string{"sqrt"},
		kind:        "symbolic",
		pattern:     "Sqrt(Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldUnary",
	},
	{
		ops:         []string{"sqrt"},
		kind:        "gradient",
		pattern:     "∂(√x)/∂x = 1/(2√x)",
		description: "node IS √x; adjoint / (2·node)",
		source:      "tensor/gradient.go",
	},

	// ── Reciprocal ───────────────────────────────────────────────────────────
	{
		ops:         []string{"reciprocal"},
		kind:        "symbolic",
		pattern:     "Reciprocal(Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldUnary",
	},
	{
		ops:         []string{"reciprocal"},
		kind:        "gradient",
		pattern:     "∂(1/x)/∂x = -node²",
		description: "node IS 1/x; negate and square",
		source:      "tensor/gradient.go",
	},

	// ── Where ────────────────────────────────────────────────────────────────
	{
		ops:         []string{"where"},
		kind:        "symbolic",
		pattern:     "where(Const,Const,Const) → Const",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldTernary",
	},
	{
		ops:         []string{"where"},
		kind:        "symbolic",
		pattern:     "where(cond, v, v) → v",
		description: "both branches equal: condition irrelevant",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnV",
	},
	{
		ops:         []string{"where"},
		kind:        "symbolic",
		pattern:     "where(true, a, b) → a",
		description: "constant true condition: always true branch",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnA",
	},
	{
		ops:         []string{"where"},
		kind:        "symbolic",
		pattern:     "where(false, a, b) → b",
		description: "constant false condition: always false branch",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnB",
	},
	{
		ops:         []string{"where"},
		kind:        "gradient",
		pattern:     "grad_cond = nil",
		description: "condition has no gradient (non-differentiable boolean)",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"where"},
		kind:        "gradient",
		pattern:     "grad_x = where(cond, adj, 0)",
		description: "true-branch gradient: pass adj where condition is true",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"where"},
		kind:        "gradient",
		pattern:     "grad_y = where(cond, 0, adj)",
		description: "false-branch gradient: pass adj where condition is false",
		source:      "tensor/gradient.go",
	},

	// ── MulAcc ───────────────────────────────────────────────────────────────
	{
		ops:         []string{"mulacc"},
		kind:        "gradient",
		pattern:     "∂(a*b+c)/∂a = adj * b",
		description: "gradient w.r.t. first multiplier is adjoint times the second",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"mulacc"},
		kind:        "gradient",
		pattern:     "∂(a*b+c)/∂b = adj * a",
		description: "gradient w.r.t. second multiplier is adjoint times the first",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"mulacc"},
		kind:        "gradient",
		pattern:     "∂(a*b+c)/∂c = adj",
		description: "gradient w.r.t. addend is the incoming adjoint",
		source:      "tensor/gradient.go",
	},

	// ── Non-differentiable ───────────────────────────────────────────────────
	{
		ops:         []string{"cmplt", "cmpeq", "cmpne", "and", "or", "xor", "shl", "shr", "idiv", "mod", "pow", "trunc"},
		kind:        "gradient",
		pattern:     "no gradient",
		description: "this op is non-differentiable or is a leaf in the graph",
		source:      "tensor/gradient.go",
	},

	// ── Cast ─────────────────────────────────────────────────────────────────
	{
		ops:         []string{"cast"},
		kind:        "symbolic",
		pattern:     "Cast(Const) → Const",
		description: "cast a constant to a new typed constant",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hCastConstFold",
	},
	{
		ops:         []string{"cast"},
		kind:        "symbolic",
		pattern:     "Cast(x) → x",
		description: "identity cast: same dtype, drop the cast node",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hIdentityCast",
	},
	{
		ops:         []string{"cast", "bitcast"},
		kind:        "gradient",
		pattern:     "adj = Cast(adj, src_dtype)",
		description: "cast adjoint back to source dtype if source is float",
		source:      "tensor/gradient.go",
	},

	// ── ReduceAxis ───────────────────────────────────────────────────────────
	{
		ops:         []string{"reduceaxis"},
		kind:        "gradient",
		pattern:     "∂ReduceAxis(Add)/∂src: expand adj to src shape",
		description: "sum backward: broadcast adjoint back over reduced axes",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"reduceaxis"},
		kind:        "gradient",
		pattern:     "∂ReduceAxis(Max)/∂src: where(mask) / tie_count",
		description: "max backward: route adj to argmax positions, split ties equally",
		source:      "tensor/gradient.go",
	},

	// ── Reshape ──────────────────────────────────────────────────────────────
	{
		ops:         []string{"reshape"},
		kind:        "gradient",
		pattern:     "adj = Reshape(adj, src_shape)",
		description: "reshape adjoint back to source shape",
		source:      "tensor/gradient.go",
	},

	// ── Expand ───────────────────────────────────────────────────────────────
	{
		ops:         []string{"expand"},
		kind:        "gradient",
		pattern:     "adj = Sum(adj, broadcast_axes)",
		description: "undo broadcast: sum adjoint over expanded axes",
		source:      "tensor/gradient.go",
	},

	// ── Permute ──────────────────────────────────────────────────────────────
	{
		ops:         []string{"permute"},
		kind:        "gradient",
		pattern:     "adj = Permute(adj, inverse_perm)",
		description: "permute adjoint by the inverse permutation",
		source:      "tensor/gradient.go",
	},

	// ── Pad ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"pad"},
		kind:        "gradient",
		pattern:     "adj = Shrink(adj, remove_padding)",
		description: "shrink adjoint to strip the added padding",
		source:      "tensor/gradient.go",
	},

	// ── Shrink ───────────────────────────────────────────────────────────────
	{
		ops:         []string{"shrink"},
		kind:        "gradient",
		pattern:     "adj = Pad(adj, restore_size)",
		description: "pad adjoint with zeros to restore original size",
		source:      "tensor/gradient.go",
	},

	// ── Flip ─────────────────────────────────────────────────────────────────
	{
		ops:         []string{"flip"},
		kind:        "gradient",
		pattern:     "adj = Flip(adj, same_axes)",
		description: "flip adjoint along the same axes",
		source:      "tensor/gradient.go",
	},

	// ── Max ──────────────────────────────────────────────────────────────────
	{
		ops:         []string{"max"},
		kind:        "symbolic",
		pattern:     "max(x, x) → x",
		description: "idempotent max",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hReturnX",
	},
	{
		ops:         []string{"max"},
		kind:        "symbolic",
		pattern:     "Const(a) max Const(b) → Const(max(a,b))",
		description: "constant folding",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hFoldBinary",
	},
	{
		ops:         []string{"max"},
		kind:        "symbolic",
		pattern:     "bool max bool → bool | bool",
		description: "boolean algebra: max of booleans becomes OR",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hBoolMax",
	},
	{
		ops:         []string{"max"},
		kind:        "symbolic",
		pattern:     "max(x,y) → max(y,x)",
		description: "commutative canonicalization (const moves to src[1])",
		source:      "rewrite/rules/symbolic.upat",
		handler:     "hCanonicalize",
	},
	{
		ops:         []string{"max"},
		kind:        "gradient",
		pattern:     "∂max(a,b)/∂a = adj where a==max, ties split equally",
		description: "max backward: src[0] gets gradient where it equals the max output",
		source:      "tensor/gradient.go",
	},
	{
		ops:         []string{"max"},
		kind:        "gradient",
		pattern:     "∂max(a,b)/∂b = adj where b==max AND not tied",
		description: "max backward: src[1] gets gradient only when it's the unique argmax",
		source:      "tensor/gradient.go",
	},
}

// matmulExplain returns the detailed matmul / matmul.backward explanation.
func matmulExplain() string {
	var b strings.Builder
	b.WriteString("op: matmul (composed from primitives)\n")
	b.WriteString("source: tensor/reduce.go:Matmul\n")
	b.WriteString("\n")
	b.WriteString("matmul is not a primitive in anneal's IR. A[M,K] @ B[K,N] decomposes to:\n")
	b.WriteString("  Reshape(A, [M,K,1]) → Expand to [M,K,N]\n")
	b.WriteString("  Reshape(B, [1,K,N]) → Expand to [M,K,N]\n")
	b.WriteString("  Mul (element-wise) → [M,K,N]\n")
	b.WriteString("  ReduceAxis(Add, axis=K) → [M,N]\n")
	b.WriteString("\n")
	b.WriteString("gradient rules:\n")
	b.WriteString("  ∂L/∂A = ∂L/∂C @ Bᵀ   (via chain rule through ReduceAxis, Mul, Expand, Reshape)\n")
	b.WriteString("  ∂L/∂B = Aᵀ @ ∂L/∂C   (via chain rule through ReduceAxis, Mul, Expand, Reshape)\n")
	b.WriteString("\n")
	b.WriteString("contributing primitive rules:\n")
	b.WriteString("  Mul:             ∂(a·b)/∂a = adj·b, ∂(a·b)/∂b = adj·a\n")
	b.WriteString("  ReduceAxis(Add): expand adjoint back to pre-reduction shape\n")
	b.WriteString("  Expand:          sum adjoint over broadcast axes\n")
	b.WriteString("  Reshape:         reshape adjoint to source shape\n")
	b.WriteString("\n")
	b.WriteString("see: tensor/reduce.go:Matmul, tensor/gradient_ruleset.go:Gradient\n")
	return b.String()
}

// symbolicOverview returns the overview of all symbolic simplification rules.
func symbolicOverview() string {
	var b strings.Builder
	b.WriteString("symbolic simplification rules (rewrite/rules/symbolic.upat)\n")
	b.WriteString("\n")
	b.WriteString("12 rule groups fire bottom-up on every UOp node:\n")
	b.WriteString("   1. constant folding         — fold all-const nodes at compile time\n")
	b.WriteString("   2. cast/bitcast folding     — cast of a const → new const\n")
	b.WriteString("   3. identity cast            — Cast(x) → x when dtypes match\n")
	b.WriteString("   4. arithmetic identities    — x+0→x, x*1→x, x*0→0, x-0→x, x^0→x, x//1→x, x//-1→-x\n")
	b.WriteString("   5. boolean neutral/absorb   — x&true→x, x|false→x, x&false→false, x|true→true\n")
	b.WriteString("   6. self-cancellation        — x-x→0, x^x→0, x%x→0, x//x→1, x<x→false, x!=x→false\n")
	b.WriteString("   7. idempotent               — x|x→x, x&x→x, max(x,x)→x\n")
	b.WriteString("   8. boolean algebra          — bool*bool→bool&bool, bool+bool→bool|bool\n")
	b.WriteString("   9. structural               — (x^y)^y→x, (x%y)%y→x%y\n")
	b.WriteString("  10. where (ternary)          — cond.where(v,v)→v, true.where(a,b)→a, false.where(a,b)→b\n")
	b.WriteString("  11. bound-based comparisons  — fold CmpLt/CmpNe when intervals resolve result\n")
	b.WriteString("  12. commutative canon        — normalize operand order so consts land at src[1]\n")
	b.WriteString("\n")
	b.WriteString("run 'anneal explain <op>' for rules specific to one op.\n")
	return b.String()
}
