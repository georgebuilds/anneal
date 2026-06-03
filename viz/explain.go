// Explain view data (W3). Pure-Go: builds the per-op explain payload for the
// studio's explain view (anneal_web_spec §4 / §5.4). Two real sources of
// truth, one hand-curated piece:
//
//   - Symbolic rules: parsed at runtime from rewrite/rules/symbolic.upat (the
//     same file the .upat compiler reads at go-generate time). DD2: the rules
//     list is the real ruleset, not a mock.
//   - Gradient rule: looked up against tensor.Gradient.RegisteredOps() (the
//     parallel ruleset that powers Backward). DD2: registered iff the engine
//     has a registered handler.
//   - Op descriptions: one short line per op, hand-curated. This is the only
//     DD2 deviation the explain view ships; it is docs, not behaviour.
//
// Mini-graph: a tiny canonical before / after snippet illustrating the most
// recognisable rewrite for the op (e.g. Add: Add(x, 0) → x). The JSON is
// shape only; the studio's JS does the rendering and animation.
//
// This file is the same on native and js/wasm — no build tags, no
// backend/webgpu import. The compile target check in cmd_web_test.go pins it.

package viz

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// symbolicUpat is the verbatim contents of rewrite/rules/symbolic.upat,
// embedded at build time so the same rule list is available in both the
// native binary and the WASM target (WASM has no disk access).
//
//go:embed symbolic.upat
var symbolicUpat string

// ExplainSet is the top-level JSON payload returned by annealExplainOp.
// One ExplainSet per op; mirrors the contract in spec §4 / §5.4.
type ExplainSet struct {
	Op            string        `json:"op"`
	Description   string        `json:"description"`
	SymbolicRules []ExplainRule `json:"symbolic_rules"`
	GradientRule  *GradientRule `json:"gradient_rule,omitempty"`
	MiniGraph     MiniGraph     `json:"mini_graph"`
}

// ExplainRule is one symbolic rewrite rule attributed to the op. Pattern and
// rewrite are text representations of the LHS and RHS; Source is the file:line
// inside rewrite/rules/. Notes is the optional canonicalization / fold hint
// (e.g. "constant folding").
type ExplainRule struct {
	Name    string `json:"name"`
	Pattern string `json:"pattern"`
	Rewrite string `json:"rewrite"`
	Source  string `json:"source"`
	Notes   string `json:"notes,omitempty"`
}

// GradientRule is the lookup result against tensor.Gradient. Absent (nil) for
// ops without a registered gradient rule (non-differentiable, or pure
// frontend movement ops with a derived rule rather than a primitive one).
type GradientRule struct {
	Pattern string `json:"pattern"`
	Source  string `json:"source"`
}

// MiniGraph is a tiny before / after demonstration the studio's JS renders as
// two adjacent node trees. The animation between them is a JS concern; JSON
// carries only structure.
type MiniGraph struct {
	Before []ExplainNode `json:"before"`
	After  []ExplainNode `json:"after"`
	Edges  []ExplainEdge `json:"edges"`
}

// ExplainNode is one node in the mini-graph. Op selects an SVG shape in JS
// (Const → square, Var → circle, ALU → hexagon, …); Label is the display
// text rendered inside the node.
type ExplainNode struct {
	ID    string `json:"id"`
	Op    string `json:"op"`
	Label string `json:"label"`
}

// ExplainEdge is a directed edge from From → To. The studio renders edges
// inside Before and After independently; the From/To are node ids.
type ExplainEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// ToJSON serializes the ExplainSet to canonical JSON bytes.
func (e *ExplainSet) ToJSON() ([]byte, error) { return json.Marshal(e) }

// BuildExplain constructs the explain payload for the named op. opName is the
// canonical uop.Op String() name (case-insensitive); an unknown name returns
// an error so the WASM bridge can surface it to the user (spec §10 errors are
// blameless and actionable).
func BuildExplain(opName string) (*ExplainSet, error) {
	if opName == "" {
		return nil, fmt.Errorf("viz: BuildExplain: empty op name")
	}
	op, ok := uop.OpFromString(opName)
	if !ok {
		return nil, fmt.Errorf("viz: BuildExplain: unknown op %q", opName)
	}
	// Use the canonical String() form so callers get the same casing back
	// regardless of how they spelled the request.
	canonical := op.String()

	rules := symbolicRulesForOp(canonical)
	grad := gradientRuleForOp(op)
	mini := miniGraphForOp(canonical)
	desc := descriptionForOp(canonical)

	return &ExplainSet{
		Op:            canonical,
		Description:   desc,
		SymbolicRules: rules,
		GradientRule:  grad,
		MiniGraph:     mini,
	}, nil
}

// ── op descriptions (the one hand-curated piece; DD2 docs not behaviour) ──

// opDescriptions is the short one-line summary the studio shows above the
// rules list. Each entry is a human-readable explanation of what the op
// computes; absent ops fall back to a generic message derived from the op
// name (so the view always renders something useful).
var opDescriptions = map[string]string{
	// Unary ALU.
	"Neg":        "negate every element; adjoint flips sign.",
	"Exp2":       "elementwise base-2 exponential (2^x).",
	"Log2":       "elementwise base-2 logarithm.",
	"Sin":        "elementwise sine; cos via phase-shift identity in backward.",
	"Sqrt":       "elementwise square root; adjoint is 1/(2·node).",
	"Reciprocal": "elementwise 1/x; adjoint is -node².",
	"Trunc":      "elementwise truncation toward zero; non-differentiable.",
	"Erf":        "Gauss error function; backward uses (2/√π)·exp(-x²).",
	"Cast":       "change dtype; constant casts are folded at compile time.",
	"Bitcast":    "reinterpret bits as a new dtype without conversion.",

	// Binary ALU.
	"Add":  "elementwise sum; commutative, identity at 0, gradient passes through.",
	"Sub":  "elementwise difference; self-cancellation x-x→0 folds at compile time.",
	"Mul":  "elementwise product; identity at 1, absorbing at 0, product rule in backward.",
	"FDiv": "elementwise float division; quotient rule in backward.",
	"IDiv": "integer division (truncated toward zero); non-differentiable.",
	"Max":  "elementwise maximum; idempotent, gradient routes to argmax (ties split).",
	"Min":  "elementwise minimum; mirror of Max.",
	"Mod":  "integer remainder; non-differentiable.",
	"Pow":  "elementwise power; non-differentiable in v1.",
	"Shl":  "bit-shift left; non-differentiable.",
	"Shr":  "bit-shift right; non-differentiable.",

	// Comparison / boolean.
	"CmpLt":    "elementwise <; result is boolean, non-differentiable.",
	"CmpNe":    "elementwise ≠; result is boolean, non-differentiable.",
	"CmpEq":    "elementwise =; result is boolean, non-differentiable.",
	"And":      "elementwise bitwise/boolean AND; idempotent.",
	"Or":       "elementwise bitwise/boolean OR; idempotent.",
	"Xor":      "elementwise XOR; self-cancellation x⊕x→0 folds at compile time.",
	"ThreeFry": "counter-based RNG step; non-differentiable.",

	// Ternary.
	"Where":  "elementwise conditional select; cond is non-differentiable, adj routes per branch.",
	"MulAcc": "fused a·b+c; three-source gradient (product rule + pass-through).",

	// Movement (frontend-only, dissolved at rangeify).
	"Reshape": "change view shape without moving data; adjoint reshapes back.",
	"Expand":  "broadcast along singleton axes; adjoint sums over broadcast axes.",
	"Permute": "reorder axes; adjoint permutes by the inverse permutation.",
	"Pad":     "pad with constant; adjoint shrinks to remove padding.",
	"Shrink":  "slice a sub-range; adjoint pads with zeros to restore size.",
	"Flip":    "reverse along axes; adjoint flips back along the same axes.",

	// Indexing.
	"Gather":     "gather elements by index; adjoint is scatter-add into zeros.",
	"ScatterAdd": "scatter-add accumulation; backward of Gather.",

	// Reduction.
	"ReduceAxis": "reduce one axis with Add/Max; adjoint expands back over reduced axes.",
	"Reduce":     "schedule-level reduction primitive.",

	// Leaves / structural.
	"Const":     "compile-time constant; leaf of the graph, no gradient.",
	"DefineVar": "symbolic variable (dynamic shape); leaf.",
	"Bind":      "bind a DefineVar to a concrete value; folds to Const once bound.",
	"Buffer":    "external storage; leaf.",
	"Sink":      "graph aggregation point; structural.",
}

func descriptionForOp(name string) string {
	if d, ok := opDescriptions[name]; ok {
		return d
	}
	// Generic fallback: tag the kind so the user still gets something useful.
	op, ok := uop.OpFromString(name)
	if !ok {
		return name + ": no description available."
	}
	switch {
	case uop.GroupUnary.Has(op):
		return name + ": unary ALU op (elementwise)."
	case uop.GroupBinary.Has(op):
		return name + ": binary ALU op (elementwise)."
	case uop.GroupTernary.Has(op):
		return name + ": ternary ALU op."
	case uop.GroupMovement.Has(op):
		return name + ": movement op (frontend-only, dissolved at rangeify)."
	default:
		return name + ": structural / scheduling op."
	}
}

// ── symbolic rules: parsed from symbolic.upat at runtime ──────────────────

// noteForHandler maps a handler name to a one-line human note. Mirrors the
// "description" column in cmd_explain.go's allRules table without copying it
// (the explain view is the JSON shape; the CLI keeps its terse-text shape).
//
// Drift safety: if a new handler ships in symbolic.upat without an entry here
// the rule still renders, just with an empty Notes field. The view degrades
// gracefully; the test in cmd_explain_test.go pins the absence of a regression.
var noteForHandler = map[string]string{
	"hFoldUnary":     "constant folding (unary)",
	"hFoldBinary":    "constant folding (binary)",
	"hFoldTernary":   "constant folding (ternary)",
	"hBindFold":      "fold Bind(DefineVar, val) → Const(val)",
	"hCastConstFold": "cast a constant to a new typed constant",
	"hIdentityCast":  "identity cast: same dtype, drop the cast",
	"hReturnX":       "identity / idempotent: return x",
	"hReturnV":       "branches equal: return v",
	"hReturnA":       "constant-true condition: return a",
	"hReturnB":       "constant-false condition: return b",
	"hReturnBase":    "double modulo: return base",
	"hMulZero":       "multiplicative absorbing element: 0",
	"hSubSelf":       "self-cancellation: x - x = 0",
	"hXorSelf":       "self-cancellation: x ⊕ x = 0",
	"hModSelf":       "self-cancellation: x mod x = 0",
	"hIDivSelf":      "self-division: x // x = 1",
	"hCmpSelf":       "self-comparison: x < x = false",
	"hAndFalse":      "absorbing: x & false = false",
	"hOrTrue":        "absorbing: x | true = true",
	"hAndZeroInt":    "integer absorbing: x & 0 = 0",
	"hIDivNegOne":    "x // -1 = -x",
	"hBoolMul":       "boolean algebra: a·b = a & b",
	"hBoolAdd":       "boolean algebra: a + b = a | b",
	"hBoolMax":       "boolean algebra: max(a,b) = a | b",
	"hCmpLtBounds":   "bound-based < fold",
	"hCmpNeBounds":   "bound-based ≠ fold",
	"hCanonicalize":  "commutative canonicalization (const moves to src[1])",
}

// upatRule is one parsed entry from symbolic.upat. Internal to viz; consumed
// by symbolicRulesForOp to build the JSON-bound ExplainRule list.
type upatRule struct {
	ops     []string // canonical op names from the alternation (Add|Mul → [Add,Mul])
	pattern string   // verbatim LHS text (the "(...)" expression)
	handler string   // handler function name (RHS)
	line    int      // 1-based line number in symbolic.upat
}

// parsedUpat is the lazily-built rule list, parsed once on first access. The
// embed contents are immutable, so cache safety is trivial.
var parsedUpat = parseSymbolicUpat()

func parseSymbolicUpat() []upatRule {
	var out []upatRule
	scanner := bufio.NewScanner(strings.NewReader(symbolicUpat))
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, "=>")
		if idx < 0 {
			continue
		}
		pattern := strings.TrimSpace(line[:idx])
		handler := strings.TrimSpace(line[idx+2:])
		if handler == "" || !strings.HasPrefix(pattern, "(") {
			continue
		}
		// Extract OPS_EXPR: chars after "(" until a space, ":", or ")".
		inner := pattern[1:]
		end := strings.IndexAny(inner, " :\t)")
		if end < 0 {
			end = len(inner)
		}
		opsExpr := inner[:end]
		if opsExpr == "" || opsExpr == "*" {
			continue
		}
		ops := strings.Split(opsExpr, "|")
		out = append(out, upatRule{
			ops:     ops,
			pattern: pattern,
			handler: handler,
			line:    lineNo,
		})
	}
	return out
}

// symbolicRulesForOp returns every parsed .upat rule whose LHS mentions the
// given op (case-insensitive). The "rewrite" text is derived from the handler
// note where one exists; absent that, it is the handler name itself (so the
// user can still grep the source).
func symbolicRulesForOp(opName string) []ExplainRule {
	target := strings.ToLower(opName)
	var rules []ExplainRule
	for _, r := range parsedUpat {
		for _, op := range r.ops {
			if strings.ToLower(op) != target {
				continue
			}
			rules = append(rules, ExplainRule{
				Name:    r.handler,
				Pattern: r.pattern,
				Rewrite: rewriteTextForHandler(r.handler),
				Source:  fmt.Sprintf("rewrite/rules/symbolic.upat:%d", r.line),
				Notes:   noteForHandler[r.handler],
			})
			break
		}
	}
	return rules
}

// rewriteTextForHandler returns the short "→ RHS" text for a handler. Used as
// the Rewrite field on each ExplainRule; the JS renders Pattern + " → " +
// Rewrite as one row in the rules list. When no canonical RHS exists (e.g.
// hCanonicalize swaps operands), we return a recognisable placeholder so the
// rendered row is still readable.
func rewriteTextForHandler(h string) string {
	switch h {
	case "hReturnX":
		return "x"
	case "hReturnV":
		return "v"
	case "hReturnA":
		return "a"
	case "hReturnB":
		return "b"
	case "hReturnBase":
		return "base"
	case "hFoldUnary", "hFoldBinary", "hFoldTernary":
		return "Const"
	case "hBindFold":
		return "Const(val)"
	case "hCastConstFold":
		return "Const'"
	case "hIdentityCast":
		return "x"
	case "hMulZero":
		return "0"
	case "hSubSelf", "hXorSelf", "hModSelf":
		return "0"
	case "hIDivSelf":
		return "1"
	case "hCmpSelf":
		return "false"
	case "hAndFalse":
		return "false"
	case "hOrTrue":
		return "true"
	case "hAndZeroInt":
		return "0"
	case "hIDivNegOne":
		return "-x"
	case "hBoolMul":
		return "x & y"
	case "hBoolAdd", "hBoolMax":
		return "x | y"
	case "hCanonicalize":
		return "(canonicalized)"
	case "hCmpLtBounds", "hCmpNeBounds":
		return "Const (bound-resolved)"
	default:
		return h
	}
}

// ── gradient rule lookup against tensor.Gradient ──────────────────────────

// gradPatternForOp is the short pattern text the explain view shows when an
// op has a registered gradient rule. Mirrors the third column of
// cmd_explain.go's gradient rows but in symbolic notation.
var gradPatternForOp = map[uop.Op]string{
	uop.OpNeg:        "∂(-x)/∂x = -adj",
	uop.OpExp2:       "∂(2^x)/∂x = node · ln2",
	uop.OpLog2:       "∂(log₂x)/∂x = adj / (x · ln2)",
	uop.OpSin:        "∂(sin x)/∂x = adj · sin(x + π/2)",
	uop.OpSqrt:       "∂(√x)/∂x = adj / (2 · node)",
	uop.OpReciprocal: "∂(1/x)/∂x = -adj · node²",
	uop.OpErf:        "∂(erf x)/∂x = adj · (2/√π) · exp(-x²)",
	uop.OpCast:       "adj = Cast(adj, src_dtype) when src is float",
	uop.OpBitcast:    "adj = Bitcast(adj, src_dtype) when src is float",
	uop.OpAdd:        "∂(a+b)/∂a = adj; ∂(a+b)/∂b = adj",
	uop.OpSub:        "∂(a-b)/∂a = adj; ∂(a-b)/∂b = -adj",
	uop.OpMul:        "∂(a·b)/∂a = adj·b; ∂(a·b)/∂b = adj·a",
	uop.OpFDiv:       "∂(a/b)/∂a = adj/b; ∂(a/b)/∂b = -adj·a/b²",
	uop.OpMax:        "∂max/∂a = adj where a==max (ties split); src[1] only if unique argmax",
	uop.OpMin:        "∂min/∂a = adj where a==min (ties split); src[1] only if unique argmin",
	uop.OpMulAcc:     "∂(a·b+c)/∂a = adj·b; ∂/∂b = adj·a; ∂/∂c = adj",
	uop.OpWhere:      "grad_x = where(cond, adj, 0); grad_y = where(cond, 0, adj); cond has no grad",
	uop.OpReshape:    "adj = Reshape(adj, src_shape)",
	uop.OpExpand:     "adj = Sum(adj, broadcast_axes)",
	uop.OpPermute:    "adj = Permute(adj, inverse_perm)",
	uop.OpPad:        "adj = Shrink(adj, remove_padding)",
	uop.OpShrink:     "adj = Pad(adj, restore_size)",
	uop.OpFlip:       "adj = Flip(adj, same_axes)",
	uop.OpGather:     "dW = scatter-add(adj, idx) into zeros_like(W); idx is non-differentiable",
	uop.OpReduceAxis: "Add: expand adj to src shape; Max: where(mask) / tie_count",

	// Registered with a nil-returning rule (non-differentiable). The studio
	// still shows the gradient panel for these so a user looking up "why
	// doesn't CmpLt train?" sees the answer in-view, rather than blank.
	uop.OpCmpLt: "no gradient (comparison, non-differentiable)",
	uop.OpCmpNe: "no gradient (comparison, non-differentiable)",
	uop.OpCmpEq: "no gradient (comparison, non-differentiable)",
	uop.OpIDiv:  "no gradient (integer division)",
	uop.OpMod:   "no gradient (modulo)",
	uop.OpShl:   "no gradient (bit shift)",
	uop.OpShr:   "no gradient (bit shift)",
	uop.OpXor:   "no gradient (bitwise XOR)",
	uop.OpOr:    "no gradient (bitwise OR)",
	uop.OpAnd:   "no gradient (bitwise AND)",
	uop.OpPow:   "no gradient (Pow is non-differentiable in v1)",
	uop.OpTrunc: "no gradient (truncation is non-differentiable)",
}

// gradientRuleForOp looks up the op in tensor.Gradient and, if registered,
// returns a GradientRule. The source string is the canonical home of every
// gradient rule (the parallel ruleset that powers Backward). Unregistered ops
// (non-differentiable, or composed-from-primitives like matmul) return nil.
func gradientRuleForOp(op uop.Op) *GradientRule {
	registered := tensor.Gradient.RegisteredOps()
	hit := false
	for _, r := range registered {
		if r == op {
			hit = true
			break
		}
	}
	if !hit {
		return nil
	}
	pattern, ok := gradPatternForOp[op]
	if !ok {
		// Registered but no curated pattern; cite the source and use the op
		// name as a degenerate pattern so the view still renders.
		pattern = op.String() + ": registered (see source)"
	}
	return &GradientRule{
		Pattern: pattern,
		Source:  "tensor/gradient_ruleset.go",
	}
}

// ── mini-graph: hand-built canonical before/after per op ──────────────────

// miniGraphForOp returns the 2-3 node before/after snippet illustrating the
// canonical rewrite for the op. Hand-authored per op so it stays compact and
// recognisable; absent ops get a trivial 1-node pass-through so the view
// still renders something.
//
// Convention: node ids are stable across before/after when the node carries
// over (e.g. for Add(x, 0)→x the "x" node has id "x" in both sides). The
// studio's JS uses id-equality to animate the rewrite — same id stays put,
// removed ids fade out, new ids fade in.
func miniGraphForOp(opName string) MiniGraph {
	switch opName {
	case "Add":
		// Before: Add(x, Const(0)). After: x.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "zero", Op: "Const", Label: "0"},
				{ID: "add", Op: "Add", Label: "+"},
			},
			After: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "add"},
				{From: "zero", To: "add"},
			},
		}
	case "Sub":
		// Before: Sub(x, x). After: Const(0).
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "sub", Op: "Sub", Label: "-"},
			},
			After: []ExplainNode{
				{ID: "zero", Op: "Const", Label: "0"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "sub"},
			},
		}
	case "Mul":
		// Before: Mul(x, Const(1)). After: x.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "one", Op: "Const", Label: "1"},
				{ID: "mul", Op: "Mul", Label: "·"},
			},
			After: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "mul"},
				{From: "one", To: "mul"},
			},
		}
	case "FDiv":
		// Before: FDiv(Const, Const). After: Const folded.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "a", Op: "Const", Label: "a"},
				{ID: "b", Op: "Const", Label: "b"},
				{ID: "div", Op: "FDiv", Label: "/"},
			},
			After: []ExplainNode{
				{ID: "folded", Op: "Const", Label: "a/b"},
			},
			Edges: []ExplainEdge{
				{From: "a", To: "div"},
				{From: "b", To: "div"},
			},
		}
	case "Neg":
		// Before: Neg(Const). After: Const folded.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "c", Op: "Const", Label: "c"},
				{ID: "neg", Op: "Neg", Label: "-"},
			},
			After: []ExplainNode{
				{ID: "folded", Op: "Const", Label: "-c"},
			},
			Edges: []ExplainEdge{
				{From: "c", To: "neg"},
			},
		}
	case "Exp2", "Log2", "Sin", "Sqrt", "Reciprocal", "Trunc":
		// Generic unary const-fold mini-graph.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "c", Op: "Const", Label: "c"},
				{ID: "u", Op: opName, Label: opName},
			},
			After: []ExplainNode{
				{ID: "folded", Op: "Const", Label: opName + "(c)"},
			},
			Edges: []ExplainEdge{
				{From: "c", To: "u"},
			},
		}
	case "Where":
		// Before: Where(true, a, b). After: a.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "cond", Op: "Const", Label: "true"},
				{ID: "a", Op: "Var", Label: "a"},
				{ID: "b", Op: "Var", Label: "b"},
				{ID: "where", Op: "Where", Label: "?"},
			},
			After: []ExplainNode{
				{ID: "a", Op: "Var", Label: "a"},
			},
			Edges: []ExplainEdge{
				{From: "cond", To: "where"},
				{From: "a", To: "where"},
				{From: "b", To: "where"},
			},
		}
	case "Max":
		// Before: Max(x, x). After: x.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "max", Op: "Max", Label: "max"},
			},
			After: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "max"},
			},
		}
	case "Cast":
		// Before: Cast(x) with matching dtype. After: x.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "cast", Op: "Cast", Label: "Cast"},
			},
			After: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "cast"},
			},
		}
	case "Xor":
		// Before: Xor(x, x). After: 0.
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "xor", Op: "Xor", Label: "⊕"},
			},
			After: []ExplainNode{
				{ID: "zero", Op: "Const", Label: "0"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "xor"},
			},
		}
	case "And":
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "t", Op: "Const", Label: "true"},
				{ID: "and", Op: "And", Label: "&"},
			},
			After: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "and"},
				{From: "t", To: "and"},
			},
		}
	case "Or":
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
				{ID: "f", Op: "Const", Label: "false"},
				{ID: "or", Op: "Or", Label: "|"},
			},
			After: []ExplainNode{
				{ID: "x", Op: "Var", Label: "x"},
			},
			Edges: []ExplainEdge{
				{From: "x", To: "or"},
				{From: "f", To: "or"},
			},
		}
	case "Bind":
		// Bind(DefineVar, val) → Const(val).
		return MiniGraph{
			Before: []ExplainNode{
				{ID: "v", Op: "DefineVar", Label: "v"},
				{ID: "bind", Op: "Bind", Label: "Bind"},
			},
			After: []ExplainNode{
				{ID: "folded", Op: "Const", Label: "val"},
			},
			Edges: []ExplainEdge{
				{From: "v", To: "bind"},
			},
		}
	default:
		// Generic pass-through: a single node with no rewrite. The studio's
		// JS still renders it; the view says "no canonical rewrite" in this
		// case (handled in JS, since the rules list is the load-bearing piece).
		return MiniGraph{
			Before: []ExplainNode{{ID: "n", Op: opName, Label: opName}},
			After:  []ExplainNode{{ID: "n", Op: opName, Label: opName}},
			Edges:  nil,
		}
	}
}

// ── helpers for listing op names (used by tests) ──────────────────────────

// AllOpNames returns the sorted list of canonical op names (every entry in
// uop.opNames with a non-empty String()). The studio's op-list pane shows
// this set; the WASM bridge does not expose it directly today (the studio
// hard-codes the search index), but tests can use it to assert coverage.
func AllOpNames() []string {
	var names []string
	for i := uop.Op(0); int(i) < uop.OpCount; i++ {
		s := i.String()
		if s == "" || strings.HasPrefix(s, "Op(") {
			continue
		}
		names = append(names, s)
	}
	sort.Strings(names)
	return names
}
