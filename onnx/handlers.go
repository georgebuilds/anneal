package onnx

// Handler registry. Stage-1 CNN and Stage-2 transformer handlers live here.
// Phase 1.B will fill the body; Phase 1.A only needs the registry seam so
// that Runner construction can call RegisterAll without dangling references.
//
// New handlers must:
//   - Match the Handler signature (see runner.go).
//   - Inspect ctx.Opset to branch on opset semantics where required (see
//     plan §4 / §6 — Squeeze/Unsqueeze/Split/ReduceSum axes-as-input at
//     opset 13, Softmax axis semantics at opset 13, Clip min/max at opset
//     11, etc.).
//   - Surface internal errors via the returned error rather than panicking.
//   - Never silently produce wrong output: a corner case the handler
//     doesn't yet support is a punt-loudly error, not a fallback.

// RegisterAll installs every canonical handler on r. Empty in Phase 1.A;
// each Phase 1.B / Phase 3 op pulls in one line here.
func RegisterAll(r *Runner) {
	// Phase 1.B handlers go here, e.g.:
	//   r.RegisterHandler("Conv", handleConv)
	//   r.RegisterHandler("Relu", handleRelu)
	//   ...
	_ = r
}
