// Tests for `anneal kernels` and its WASM-export sibling (viz.BuildKernels).
// The parity test below is the load-bearing one: it proves that the JSON the
// W2 kernels view receives is byte-for-byte the same WGSL the CLI prints.
// DD2: real compiler, one realize path, two readers (CLI + studio).

package main

import (
	"bytes"
	"strings"
	"testing"

	_ "github.com/georgebuilds/anneal/examples" // register mlp, conv, dynmlp
	"github.com/georgebuilds/anneal/viz"
)

// TestKernelsCmd_MLPSmoke pins the CLI command path on a small model: a
// non-zero exit code with no model would be a regression, and the basic
// kernels listing has a recognisable shape.
func TestKernelsCmd_MLPSmoke(t *testing.T) {
	var buf bytes.Buffer
	rc := kernelsCmdW([]string{"mlp"}, &buf)
	if rc != 0 {
		t.Fatalf("kernels mlp returned %d, want 0; output: %s", rc, buf.String())
	}
	out := buf.String()
	if !strings.Contains(out, "kernels: mlp") {
		t.Errorf("missing CLI header in kernels output")
	}
	if !strings.Contains(out, "@compute") {
		t.Errorf("CLI output missing WGSL @compute header")
	}
}

// TestKernelsCmd_BadModelExitsNonZero pins the error path: unknown model name
// must return non-zero and print an error message.
func TestKernelsCmd_BadModelExitsNonZero(t *testing.T) {
	var buf bytes.Buffer
	rc := kernelsCmdW([]string{"nope-not-a-real-model"}, &buf)
	if rc == 0 {
		t.Fatalf("kernels nope returned 0, want non-zero; output: %s", buf.String())
	}
}

// TestKernelsWASMMatchesCLI_MLP is the W2 parity gate: the WASM-export path
// (viz.BuildKernels, which the studio's kernels view consumes via the
// annealGetKernels JS bridge) and the CLI command (`anneal kernels mlp`)
// must agree on:
//
//   - kernel count
//   - kernel order (K0 first in WASM == "--- kernel 0 ---" first in CLI)
//   - WGSL text bytes (per kernel)
//
// If this test ever breaks, the two paths have diverged and the studio's
// kernels view is no longer showing what the CLI shows. That violates DD2
// (real compiler only) and is a merge blocker.
func TestKernelsWASMMatchesCLI_MLP(t *testing.T) {
	testKernelsParity(t, "mlp")
}

// TestKernelsWASMMatchesCLI_Conv exercises the parity gate on a deeper model.
func TestKernelsWASMMatchesCLI_Conv(t *testing.T) {
	testKernelsParity(t, "conv")
}

func testKernelsParity(t *testing.T, model string) {
	t.Helper()

	// WASM side: the same call studio.js will make via the worker RPC.
	k, err := viz.BuildKernels(model)
	if err != nil {
		t.Fatalf("BuildKernels(%q): %v", model, err)
	}

	// CLI side: capture exactly what `anneal kernels <model>` prints.
	var buf bytes.Buffer
	if rc := kernelsCmdW([]string{model}, &buf); rc != 0 {
		t.Fatalf("kernels %s returned %d; output: %s", model, rc, buf.String())
	}
	cliOut := buf.String()

	// Split the CLI output on "--- kernel N ---" boundaries; each chunk is one
	// kernel's textual rendering (header + WGSL).
	chunks := strings.Split(cliOut, "--- kernel ")
	// First chunk is the header; the rest are kernel sections.
	if len(chunks) < 2 {
		t.Fatalf("CLI output for %s has no kernel sections", model)
	}
	cliKernels := chunks[1:]

	if len(k.Kernels) != len(cliKernels) {
		t.Fatalf("%s kernel-count parity: WASM=%d, CLI=%d",
			model, len(k.Kernels), len(cliKernels))
	}

	for i, kr := range k.Kernels {
		cli := cliKernels[i]
		// Extract the WGSL from the CLI chunk. The CLI prints the kernel
		// header (type/output/inputs lines), then a blank line, then the
		// full WGSL. The first WGSL line is one of:
		//   - "enable f16;"            — f16-using kernel prelude
		//   - "@group(0) @binding(..." — storage buffer declarations
		// We anchor on these; @compute appears later (after the bindings).
		wgslStart := strings.Index(cli, "enable f16;")
		if wgslStart < 0 {
			wgslStart = strings.Index(cli, "@group(")
		}
		if wgslStart < 0 {
			// Last fallback: anchor on @compute (no bindings + no f16).
			wgslStart = strings.Index(cli, "@compute")
		}
		if wgslStart < 0 {
			t.Errorf("%s kernel %d (%s): could not locate WGSL start in CLI output", model, i, kr.ID)
			continue
		}
		cliWGSL := cli[wgslStart:]
		// The CLI appends one trailing newline (fmt.Fprintln after the WGSL
		// block); trim both sides for the byte comparison so a wandering
		// final \n doesn't trip the parity assertion.
		gotCLI := strings.TrimRight(cliWGSL, "\n")
		gotWASM := strings.TrimRight(kr.WGSL, "\n")
		if gotCLI != gotWASM {
			// Compute a small diff context to make a real failure debuggable
			// without dumping kilobytes.
			t.Errorf("%s kernel %d (%s): WGSL bytes differ between WASM and CLI",
				model, i, kr.ID)
			minLen := len(gotCLI)
			if len(gotWASM) < minLen {
				minLen = len(gotWASM)
			}
			for off := 0; off < minLen; off++ {
				if gotCLI[off] != gotWASM[off] {
					end := off + 80
					if end > minLen {
						end = minLen
					}
					t.Logf("  first diff at byte %d:\n    CLI: %q\n    WASM: %q",
						off, gotCLI[off:end], gotWASM[off:end])
					break
				}
			}
		}
	}
}
