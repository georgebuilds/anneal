// Tests for the W3 explain CLI ↔ WASM coverage parity.
//
// Unlike W2's kernels view (where CLI text and WASM JSON are byte-identical
// for the WGSL block), the explain view CLI prints terse curated text while
// the WASM bridge returns structured JSON. Byte-equality is therefore NOT
// the W3 gate. Instead, this file asserts COVERAGE parity:
//
//   - Every op the CLI's `anneal explain <op>` resolves successfully (rc=0)
//     also resolves successfully via viz.BuildExplain.
//   - When the CLI lists a gradient section for an op, the WASM payload's
//     GradientRule is non-nil for that op.
//   - When the CLI lists symbolic rules for an op, the WASM payload's
//     SymbolicRules list is non-empty.
//
// The drift contract here is "same op resolves, same kinds of rules listed,
// same gradient cited." If the CLI ever drifts apart from the WASM payload
// on op coverage, this test catches it.

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/viz"
)

// TestExplainWASMMatchesCLI_Add pins the coverage equivalence for the
// canonical Add op. Both surfaces must resolve the op (CLI rc=0, WASM no
// error), and both must list at least one symbolic rule and a gradient
// rule.
func TestExplainWASMMatchesCLI_Add(t *testing.T) {
	testExplainCoverageParity(t, "add", "Add", true, true)
}

// TestExplainWASMMatchesCLI_Sqrt exercises a unary op with both a symbolic
// fold and a gradient.
func TestExplainWASMMatchesCLI_Sqrt(t *testing.T) {
	testExplainCoverageParity(t, "sqrt", "Sqrt", true, true)
}

// TestExplainWASMMatchesCLI_Where exercises a ternary with multiple
// symbolic rules and a per-source gradient set.
func TestExplainWASMMatchesCLI_Where(t *testing.T) {
	testExplainCoverageParity(t, "where", "Where", true, true)
}

// TestExplainWASMMatchesCLI_Reshape exercises a movement op: the CLI
// presents only a gradient rule (no symbolic rule lives under the op name);
// the WASM payload mirrors that.
func TestExplainWASMMatchesCLI_Reshape(t *testing.T) {
	testExplainCoverageParity(t, "reshape", "Reshape", false, true)
}

// TestExplainWASMMatchesCLI_BadOpNonzero pins the error path: CLI returns
// non-zero on an unknown op; WASM returns an error from BuildExplain.
func TestExplainWASMMatchesCLI_BadOpNonzero(t *testing.T) {
	var buf bytes.Buffer
	rc := explainCmdW([]string{"nope-not-a-real-op"}, &buf)
	if rc == 0 {
		t.Errorf("CLI explain nope returned rc=0, want non-zero")
	}
	if _, err := viz.BuildExplain("nope-not-a-real-op"); err == nil {
		t.Errorf("viz.BuildExplain(nope) returned nil error, want failure")
	}
}

// testExplainCoverageParity is the shared driver. cliQuery is the
// lowercased name the CLI accepts; wasmName is the canonical-cased name the
// WASM bridge returns. wantSym / wantGrad gate whether the assertion is
// for non-empty symbolic rules / non-nil gradient.
func testExplainCoverageParity(t *testing.T, cliQuery, wasmName string, wantSym, wantGrad bool) {
	t.Helper()

	// CLI side.
	var buf bytes.Buffer
	rc := explainCmdW([]string{cliQuery}, &buf)
	if rc != 0 {
		t.Fatalf("CLI explain %s returned rc=%d; output: %s", cliQuery, rc, buf.String())
	}
	cliOut := buf.String()

	// WASM side.
	e, err := viz.BuildExplain(wasmName)
	if err != nil {
		t.Fatalf("viz.BuildExplain(%q): %v", wasmName, err)
	}
	if e.Op != wasmName {
		t.Errorf("WASM Op = %q, want %q", e.Op, wasmName)
	}

	if wantSym {
		// CLI prints "symbolic rules:" when at least one symbolic rule is
		// curated for the op.
		if !strings.Contains(cliOut, "symbolic rules:") {
			t.Errorf("CLI explain %s missing 'symbolic rules:' section; output: %s", cliQuery, cliOut)
		}
		if len(e.SymbolicRules) == 0 {
			t.Errorf("WASM Op=%s has 0 symbolic rules; CLI lists them - coverage drift", wasmName)
		}
	}
	if wantGrad {
		if !strings.Contains(cliOut, "gradient rules:") {
			t.Errorf("CLI explain %s missing 'gradient rules:' section; output: %s", cliQuery, cliOut)
		}
		if e.GradientRule == nil {
			t.Errorf("WASM Op=%s has nil GradientRule; CLI lists a gradient - coverage drift", wasmName)
		}
	}
}

// TestExplainWASMMatchesCLI_AllCuratedOpsResolve sweeps every op the CLI's
// curated allRules table covers and asserts viz.BuildExplain resolves it
// (no error, canonical name back). Drift safety: a new entry in allRules
// without a matching uop.OpFromString lookup would break this test.
func TestExplainWASMMatchesCLI_AllCuratedOpsResolve(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range allRules {
		for _, opName := range r.ops {
			if seen[opName] {
				continue
			}
			seen[opName] = true
			e, err := viz.BuildExplain(opName)
			if err != nil {
				t.Errorf("CLI lists op %q in allRules but viz.BuildExplain rejects it: %v", opName, err)
				continue
			}
			if e.Op == "" {
				t.Errorf("WASM Op for %q is empty", opName)
			}
		}
	}
}
