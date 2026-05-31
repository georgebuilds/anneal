package webgpu_test

// Slice 3 — Reshape across a symbolic axis (Option B).
//
// Three deliverables:
//
//   (A) Round-trip correctness:
//       symbolic input [n,4] → merge to [n*4] → split back to [n,4]
//       must be bit-identical to the original for n ∈ {3, 7, 11}.
//       The merged buffer's flattened layout must equal the input's
//       row-major flattening (bit-identical).
//
//   (B) Sub-product mismatch rejection:
//       [n,4] reshape to [m,8] (two distinct symbolic vars) must panic
//       with a "size mismatch" message at View.Reshape rather than
//       silently producing a wrong-sized contiguous view.
//
//   (C) WGSL emission spot-check:
//       For the n=3 merge case the symbolic kernel's WGSL must contain
//       a loop / bounds-check expression referencing params_n built up
//       as a u32 ALU expression (e.g. "(params_n.n0 * 4u)").

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// build2DSymbolicInput constructs a leaf tensor of shape [n, 4] backed by
// a BUFFER node carrying a ShapeSintArg. Pre-fills data so the test can
// directly compare values without an additional copy.
func build2DSymbolicInput(t *testing.T, a *uop.Arena, name string, n int64) *tensor.Tensor {
	t.Helper()
	const device = "webgpu"
	ta := tensor.NewSymbolicBatchInput(a, name, 1, 1024, []int64{4}, uop.Dtypes.Float32, device)
	data := make([]float32, n*4)
	for i := int64(0); i < n; i++ {
		for j := int64(0); j < 4; j++ {
			data[i*4+j] = float32(i)*100.0 + float32(j)
		}
	}
	ta.SetData(data)
	return ta
}

// TestReshapeSymbolicRoundTrip — Slice 3 deliverable (A).
//
// For each n ∈ {3, 7, 11}:
//
//	a := input[n, 4]                  (a[i,j] = i*100 + j)
//	b := a.ReshapeSints([n*4])        (merge)
//	c := b.ReshapeSints([n, 4])       (split)
//
// Realize b and c. Assert:
//   - max-abs-diff(a, c) == 0.0
//   - b's flattened layout matches a's row-major flattening (bit-identical)
//
// All n values share the same compiled symbolic kernels: SymCompiledCount
// must stabilise at a small constant after every n binding.
func TestReshapeSymbolicRoundTrip(t *testing.T) {
	dev := requireDevice(t)
	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	ns := []int64{3, 7, 11}
	for _, n := range ns {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			a := uop.NewArena(64)
			ta := build2DSymbolicInput(t, a, "n", n)

			// Build the symbolic merge expression: out_dim = n*4
			nNode := ta.ShapeSints()[0].(shape.SymInt).Node
			mergedDim := shape.Mul(shape.SymInt{Node: nNode}, shape.ConstInt{V: 4})

			// Use Contiguous() to force each reshape to materialise a buffer.
			// Realize tb and tc in two separate calls so assignOutputs maps each
			// to its own final-output buffer; in a combined call, tb is read by
			// tc's kernel and would not appear in finalOutIdxes.
			tb := ta.ReshapeSints([]shape.Sint{mergedDim}).Contiguous()
			if err := tensor.RealizeWithBinding(map[string]int64{"n": n}, tb); err != nil {
				t.Fatalf("RealizeWithBinding tb: %v", err)
			}
			gotB := tb.Data()
			wantB := ta.Data() // SetData stored row-major already
			if int64(len(gotB)) != n*4 {
				t.Fatalf("b length = %d, want %d", len(gotB), n*4)
			}
			var maxAbsBA float64
			for i := range wantB {
				d := math.Abs(float64(gotB[i] - wantB[i]))
				if d > maxAbsBA {
					maxAbsBA = d
				}
			}

			// c must round-trip back to a, bit-identical.
			tc := tb.ReshapeSints([]shape.Sint{shape.SymInt{Node: nNode}, shape.ConstInt{V: 4}}).Contiguous()
			if err := tensor.RealizeWithBinding(map[string]int64{"n": n}, tc); err != nil {
				t.Fatalf("RealizeWithBinding tc: %v", err)
			}
			gotC := tc.Data()
			if int64(len(gotC)) != n*4 {
				t.Fatalf("c length = %d, want %d", len(gotC), n*4)
			}
			var maxAbsCA float64
			for i := range wantB {
				d := math.Abs(float64(gotC[i] - wantB[i]))
				if d > maxAbsCA {
					maxAbsCA = d
				}
			}

			t.Logf("n=%d: max|a - c| = %g; max|a - b_flat| = %g", n, maxAbsCA, maxAbsBA)

			if maxAbsBA != 0.0 {
				t.Errorf("n=%d: merge violates row-major flattening (max abs diff %g, want 0.0)", n, maxAbsBA)
			}
			if maxAbsCA != 0.0 {
				t.Errorf("n=%d: round-trip not bit-identical (max abs diff %g, want 0.0)", n, maxAbsCA)
			}
		})
	}

	// Compile-once stability: across n=3,7,11 the same set of WGSL programs
	// must serve every binding. The number of distinct symbolic compiled
	// programs depends on how many distinct kernels the reshape pair produces
	// (kernels with different bodies are different programs). Assert a tight
	// upper bound rather than a fragile exact value.
	count := dev.SymCompiledCount()
	t.Logf("symbolic kernels compiled across n=3,7,11: %d", count)
	if count > 4 {
		t.Errorf("SymCompiledCount = %d, expected a small constant (≤ 4) — compile-once broken", count)
	}
}

// TestReshapeSymbolicSubproductMismatch — Slice 3 deliverable (B).
//
// Constructs two distinct DefineVars n and m and attempts a Reshape from
// [n, 4] to [m, 8]. Pre-Slice-3 this silently succeeded; post-Slice-3 it
// must panic loudly so the corruption cannot land in execution. We catch
// the panic and report the message text so the brief's bar is met.
func TestReshapeSymbolicSubproductMismatch(t *testing.T) {
	a := uop.NewArena(64)
	const device = "webgpu"

	ta := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{4}, uop.Dtypes.Float32, device)
	mDef := a.DefineVar("m", 1, 1024)

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on [n,4] → [m,8] reshape; got none")
		}
		msg := fmt.Sprintf("%v", r)
		t.Logf("subproduct-mismatch panic message: %q", msg)
		if !strings.Contains(msg, "size mismatch") {
			t.Errorf("panic message lacks \"size mismatch\": %q", msg)
		}
	}()

	// This call must panic from View.Reshape's symbolic prod-equality check.
	_ = ta.ReshapeSints([]shape.Sint{shape.SymInt{Node: mDef}, shape.ConstInt{V: 8}})
}

// TestReshapeSymbolicWGSLSpotCheck — Slice 3 deliverable (C).
//
// For the [n,4]→[n*4] merge case at n=3, capture the emitted WGSL source
// and verify the symbolic dim's bound renders as a u32 ALU expression
// referencing params_n (i.e. not a baked literal 12u; not just a bare
// params_n.n0 because the merged dim's bound is 4*n, not n).
//
// The relevant line is logged verbatim for verification.
func TestReshapeSymbolicWGSLSpotCheck(t *testing.T) {
	a := uop.NewArena(64)
	const device = "webgpu"

	ta := tensor.NewSymbolicBatchInput(a, "n", 1, 1024, []int64{4}, uop.Dtypes.Float32, device)
	nNode := ta.ShapeSints()[0].(shape.SymInt).Node
	mergedDim := shape.Mul(shape.SymInt{Node: nNode}, shape.ConstInt{V: 4})
	tb := ta.ReshapeSints([]shape.Sint{mergedDim})

	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{tb.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, device)
	if len(items) == 0 {
		t.Fatal("schedule produced no items")
	}

	// Find the symbolic kernel whose output has a non-trivial multiplier.
	var sym schedule.ExecItem
	found := false
	for _, item := range items {
		if !itemHasSymDim(item) {
			continue
		}
		// The merged output buffer carries SymDimMul=[4].
		if len(item.Bufs) > 0 && len(item.Bufs[0].SymDimMul) > 0 && item.Bufs[0].SymDimMul[0] != 1 {
			sym = item
			found = true
			break
		}
		if !found {
			sym = item
		}
	}
	if !sym.Ast.Valid() && sym.WGSL == "" {
		t.Fatal("no symbolic kernel found")
	}

	wgsl := sym.WGSL
	if wgsl == "" {
		wgsl = codegen.RenderWGSL(sym).WGSL
	}

	// Locate the load-bearing lines: bounds-check or loop bound referencing
	// params_n with a multiplier.
	var hits []string
	for _, line := range strings.Split(wgsl, "\n") {
		if !strings.Contains(line, "params_n") {
			continue
		}
		hits = append(hits, strings.TrimSpace(line))
	}
	if len(hits) == 0 {
		t.Fatalf("no line referencing params_n in WGSL\n--- WGSL ---\n%s", wgsl)
	}
	for _, h := range hits {
		t.Logf("WGSL line referencing params_n: %s", h)
	}

	// Must contain at least one occurrence of a multiplier-on-params_n form
	// (either "params_n.n0 * 4u" or "4u * params_n.n0" or "(params_n.n0 * 4u)").
	hasALUForm := false
	for _, h := range hits {
		if strings.Contains(h, "params_n.n0 * 4u") || strings.Contains(h, "4u * params_n.n0") {
			hasALUForm = true
			break
		}
	}
	if !hasALUForm {
		t.Errorf("FAIL: expected a u32 ALU expression like \"params_n.n0 * 4u\" in WGSL; got none\n--- WGSL ---\n%s", wgsl)
	}
}
