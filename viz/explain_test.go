// Tests for viz.BuildExplain - the WASM-buildable path that drives the W3
// explain view. These tests run on native (no build tag); the same call
// applies in the WASM environment via the annealExplainOp bridge.

package viz

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// TestBuildExplain_Add pins the canonical Add payload: a known set of
// symbolic rules (additive identity, constant fold, commutative canon, …),
// the gradient pattern (∂(a+b)/∂a = adj), and a 2→1 mini-graph illustrating
// the additive-identity rewrite.
func TestBuildExplain_Add(t *testing.T) {
	e, err := BuildExplain("Add")
	if err != nil {
		t.Fatalf("BuildExplain(Add): %v", err)
	}
	if e.Op != "Add" {
		t.Errorf("Op = %q, want Add", e.Op)
	}
	if e.Description == "" {
		t.Errorf("Add: empty description")
	}
	if len(e.SymbolicRules) == 0 {
		t.Errorf("Add: no symbolic rules returned (expected >= 1)")
	}
	// Pin specific rule presence. The .upat file declares at least:
	//   - additive identity (hReturnX) - Add(*, Const !int64(0)) and float64(0)
	//   - constant folding (hFoldBinary)
	//   - commutative canonicalization (hCanonicalize)
	//   - boolean algebra (hBoolAdd)
	handlers := map[string]bool{}
	for _, r := range e.SymbolicRules {
		handlers[r.Name] = true
		if r.Source == "" {
			t.Errorf("Add rule %q missing source", r.Name)
		}
		if !strings.HasPrefix(r.Source, "rewrite/rules/symbolic.upat:") {
			t.Errorf("Add rule %q source %q does not point at the .upat", r.Name, r.Source)
		}
		if r.Pattern == "" {
			t.Errorf("Add rule %q missing pattern", r.Name)
		}
	}
	for _, want := range []string{"hReturnX", "hFoldBinary", "hCanonicalize", "hBoolAdd"} {
		if !handlers[want] {
			t.Errorf("Add: expected symbolic rule handler %q missing from %v", want, handlers)
		}
	}
	// Gradient rule is mandatory for Add: ∂(a+b)/∂a = adj passes through.
	if e.GradientRule == nil {
		t.Fatalf("Add: gradient rule should be registered (tensor.Gradient has OpAdd)")
	}
	if !strings.Contains(e.GradientRule.Pattern, "adj") {
		t.Errorf("Add gradient pattern %q should mention adj", e.GradientRule.Pattern)
	}
	if e.GradientRule.Source != "tensor/gradient_ruleset.go" {
		t.Errorf("Add gradient source = %q, want tensor/gradient_ruleset.go", e.GradientRule.Source)
	}
	// Mini-graph: 3-node Before (x, zero, add) → 1-node After (x).
	if len(e.MiniGraph.Before) != 3 {
		t.Errorf("Add mini-graph before has %d nodes, want 3", len(e.MiniGraph.Before))
	}
	if len(e.MiniGraph.After) != 1 {
		t.Errorf("Add mini-graph after has %d nodes, want 1", len(e.MiniGraph.After))
	}
	// JSON round-trip carries the documented top-level keys.
	b, err := e.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	for _, key := range []string{
		`"op"`, `"description"`, `"symbolic_rules"`, `"gradient_rule"`, `"mini_graph"`,
	} {
		if !strings.Contains(string(b), key) {
			t.Errorf("Add JSON missing key %s; payload: %s", key, string(b))
		}
	}
}

// TestBuildExplain_Sqrt pins the unary const-fold + the registered gradient
// rule (∂√x/∂x = 1/(2√x)) and the generic unary mini-graph (fold Const).
func TestBuildExplain_Sqrt(t *testing.T) {
	e, err := BuildExplain("Sqrt")
	if err != nil {
		t.Fatalf("BuildExplain(Sqrt): %v", err)
	}
	if e.Op != "Sqrt" {
		t.Errorf("Op = %q, want Sqrt", e.Op)
	}
	if len(e.SymbolicRules) == 0 {
		t.Errorf("Sqrt: expected at least the unary constant-fold rule")
	}
	// The unary constant-fold rule is the first match in .upat.
	foundFoldUnary := false
	for _, r := range e.SymbolicRules {
		if r.Name == "hFoldUnary" {
			foundFoldUnary = true
		}
	}
	if !foundFoldUnary {
		t.Errorf("Sqrt: expected hFoldUnary among symbolic rules")
	}
	// Gradient is registered.
	if e.GradientRule == nil {
		t.Fatalf("Sqrt: gradient rule should be registered")
	}
	// Mini-graph: 2 → 1.
	if len(e.MiniGraph.Before) != 2 {
		t.Errorf("Sqrt mini-graph before has %d nodes, want 2", len(e.MiniGraph.Before))
	}
	if len(e.MiniGraph.After) != 1 {
		t.Errorf("Sqrt mini-graph after has %d nodes, want 1", len(e.MiniGraph.After))
	}
}

// TestBuildExplain_UnknownOp pins the error path. The WASM bridge surfaces the
// error to the user as a blameless message ("unknown op X"); the renderer
// switches to an empty-state view without crashing.
func TestBuildExplain_UnknownOp(t *testing.T) {
	if _, err := BuildExplain("NotARealOp"); err == nil {
		t.Errorf("expected error for unknown op, got nil")
	}
	if _, err := BuildExplain(""); err == nil {
		t.Errorf("expected error for empty op name, got nil")
	}
}

// TestBuildExplain_JSONShape pins the field names so the JS renderer can rely
// on them without an out-of-band schema. Mirrors the same gate as
// TestBuildKernels_JSONShape.
func TestBuildExplain_JSONShape(t *testing.T) {
	e, err := BuildExplain("Mul")
	if err != nil {
		t.Fatalf("BuildExplain(Mul): %v", err)
	}
	b, err := e.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(b, &parsed); err != nil {
		t.Fatalf("explain JSON does not parse: %v", err)
	}
	for _, key := range []string{"op", "description", "symbolic_rules", "gradient_rule", "mini_graph"} {
		if _, ok := parsed[key]; !ok {
			t.Errorf("missing top-level key %q", key)
		}
	}
	// First symbolic rule has the documented sub-keys.
	rules, _ := parsed["symbolic_rules"].([]any)
	if len(rules) == 0 {
		t.Fatalf("Mul has no symbolic rules")
	}
	r0, _ := rules[0].(map[string]any)
	for _, key := range []string{"name", "pattern", "rewrite", "source"} {
		if _, ok := r0[key]; !ok {
			t.Errorf("first symbolic rule missing key %q", key)
		}
	}
	// Mini-graph has the documented sub-keys.
	mg, _ := parsed["mini_graph"].(map[string]any)
	for _, key := range []string{"before", "after", "edges"} {
		if _, ok := mg[key]; !ok {
			t.Errorf("mini_graph missing key %q", key)
		}
	}
}

// TestBuildExplain_NonDifferentiable pins that a known non-differentiable op
// surfaces its non-differentiability in the gradient pattern text. CmpLt is
// registered in tensor.Gradient with a nil-returning rule (so the dispatch is
// O(1) and the type system stays uniform); the explain view shows "no
// gradient" so the user understands why the op is a graph leaf in backward.
func TestBuildExplain_NonDifferentiable(t *testing.T) {
	e, err := BuildExplain("CmpLt")
	if err != nil {
		t.Fatalf("BuildExplain(CmpLt): %v", err)
	}
	if e.GradientRule == nil {
		t.Fatalf("CmpLt: expected a gradient rule entry indicating no-gradient (CmpLt is registered with a nil-returning handler in tensor.Gradient)")
	}
	if !strings.Contains(strings.ToLower(e.GradientRule.Pattern), "no gradient") {
		t.Errorf("CmpLt gradient pattern %q should mention 'no gradient'", e.GradientRule.Pattern)
	}
}

// TestBuildExplain_UnregisteredOp pins that a frontend-only / structural op
// with NO entry in tensor.Gradient (e.g. Sink, Buffer) returns nil for
// GradientRule and the JSON wire format omits the key.
func TestBuildExplain_UnregisteredOp(t *testing.T) {
	e, err := BuildExplain("Sink")
	if err != nil {
		t.Fatalf("BuildExplain(Sink): %v", err)
	}
	if e.GradientRule != nil {
		t.Errorf("Sink: gradient rule should be nil (unregistered), got %v", e.GradientRule)
	}
	b, _ := e.ToJSON()
	if strings.Contains(string(b), `"gradient_rule"`) {
		t.Errorf("Sink JSON should omit gradient_rule; payload: %s", string(b))
	}
}

// TestBuildExplain_CanonicalCasing pins that case-insensitive lookup still
// returns the canonical-cased op name in the payload (so the JS can rely on
// it for the URL contract /x/Add and for display).
func TestBuildExplain_CanonicalCasing(t *testing.T) {
	cases := map[string]string{
		"add": "Add", "ADD": "Add", "Add": "Add",
		"sqrt": "Sqrt", "SQRT": "Sqrt",
	}
	for input, want := range cases {
		e, err := BuildExplain(input)
		if err != nil {
			t.Errorf("BuildExplain(%q): %v", input, err)
			continue
		}
		if e.Op != want {
			t.Errorf("BuildExplain(%q) Op = %q, want %q", input, e.Op, want)
		}
	}
}

// TestBuildExplain_AllRegisteredGradientsCovered pins that every op registered
// in tensor.Gradient has a curated gradient pattern in gradPatternForOp (no
// op falls through to the degenerate "registered (see source)" fallback).
// Drift safety: if a new gradient lands without an entry, this test fails
// with the op name.
func TestBuildExplain_AllRegisteredGradientsCovered(t *testing.T) {
	var _ uop.Op // keep uop import used regardless of dispatch shape
	for _, op := range tensor.Gradient.RegisteredOps() {
		e, err := BuildExplain(op.String())
		if err != nil {
			t.Errorf("BuildExplain(%s): %v", op, err)
			continue
		}
		if e.GradientRule == nil {
			t.Errorf("op %s: registered in tensor.Gradient but BuildExplain returned no gradient rule", op)
			continue
		}
		if strings.Contains(e.GradientRule.Pattern, "registered (see source)") {
			t.Errorf("op %s: gradient pattern falls through to degenerate fallback; add an entry to gradPatternForOp", op)
		}
	}
}

// TestBuildExplain_UpatSourceLineMatchesFile pins that every rule's reported
// file:line points at a line in symbolic.upat that contains "=>" (i.e. it's a
// real rule, not a comment line). Drift safety: if the embedded copy of
// symbolic.upat ever falls out of sync with the upstream file, the parse
// offset would still produce a number but the line content would not be a
// rule; this test catches that.
func TestBuildExplain_UpatSourceLineMatchesFile(t *testing.T) {
	lines := strings.Split(symbolicUpat, "\n")
	for _, opName := range []string{"Add", "Mul", "Sub", "Sqrt", "Where"} {
		e, err := BuildExplain(opName)
		if err != nil {
			t.Fatalf("BuildExplain(%s): %v", opName, err)
		}
		for _, r := range e.SymbolicRules {
			// Source is "rewrite/rules/symbolic.upat:N"; extract N.
			parts := strings.Split(r.Source, ":")
			if len(parts) != 2 {
				t.Errorf("%s rule %s has malformed source %q", opName, r.Name, r.Source)
				continue
			}
			var n int
			if _, err := parseLineNo(parts[1], &n); err != nil {
				t.Errorf("%s rule %s: parse line in %q: %v", opName, r.Name, r.Source, err)
				continue
			}
			if n < 1 || n > len(lines) {
				t.Errorf("%s rule %s: line %d out of range (file has %d lines)", opName, r.Name, n, len(lines))
				continue
			}
			content := strings.TrimSpace(lines[n-1])
			if !strings.Contains(content, "=>") {
				t.Errorf("%s rule %s: line %d (%q) does not contain a rule arrow", opName, r.Name, n, content)
			}
		}
	}
}

// parseLineNo is a tiny strconv.Atoi wrapper that uses a *int sink so the
// callsite stays compact. Returns (n, err) where err names the bad input.
func parseLineNo(s string, out *int) (int, error) {
	n, err := atoi(s)
	if err == nil {
		*out = n
	}
	return n, err
}

func atoi(s string) (int, error) {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, errInvalidInt(s)
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}

type errInvalidInt string

func (e errInvalidInt) Error() string { return "invalid integer: " + string(e) }

// TestBuildExplain_EmbeddedUpatNonEmpty pins that the embedded copy of
// symbolic.upat has the same load-bearing rules the .upat compiler reads.
// We assert presence of canonical rule arrows (=>) and a representative
// handler (hReturnX). The byte-identity parity test lives in
// cmd/anneal/cmd_web_explain_test.go (it has access to both files via
// runtime.Caller).
func TestBuildExplain_EmbeddedUpatNonEmpty(t *testing.T) {
	if len(symbolicUpat) == 0 {
		t.Fatal("symbolicUpat is empty: //go:embed failed to populate the file")
	}
	if !strings.Contains(symbolicUpat, "=>") {
		t.Errorf("symbolicUpat has no rule arrows")
	}
	if !strings.Contains(symbolicUpat, "hReturnX") {
		t.Errorf("symbolicUpat missing canonical handler hReturnX")
	}
}

// TestBuildExplain_OpListSorted pins that AllOpNames returns a sorted set with
// the canonical-cased names (Add, Mul, ReduceAxis, …) and no synthetic
// "Op(N)" entries. The studio renders these in a search list; ordering and
// casing are part of the UX contract.
func TestBuildExplain_OpListSorted(t *testing.T) {
	names := AllOpNames()
	if len(names) == 0 {
		t.Fatal("AllOpNames returned empty")
	}
	for i := 1; i < len(names); i++ {
		if names[i] < names[i-1] {
			t.Errorf("op names not sorted: %q before %q", names[i-1], names[i])
		}
	}
	want := []string{"Add", "Mul", "Neg", "ReduceAxis", "Reshape", "Sqrt"}
	have := map[string]bool{}
	for _, n := range names {
		have[n] = true
	}
	for _, w := range want {
		if !have[w] {
			t.Errorf("op list missing canonical name %q", w)
		}
	}
}
