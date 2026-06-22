package tui

// TokenSnapshot is the wire format for the W7 /sse/generate stream. It is a
// parallel type to Snapshot (which is loss-shaped): TokenSnapshot is
// token-shaped - one value pushed per emitted token plus a final PhaseDone
// terminator. The two types share Phase encoding so the studio's SSE
// vocabulary stays uniform.
//
// Design intent:
//   - One frame per generated token; the SSE handler throttles to ~60 Hz on
//     the wire (the same min-interval floor as the train channel).
//   - RefMatch is a pointer so the absence of a reference (no ?compare=1 or
//     no oracle configured) is distinguishable from a recorded "false". JSON
//     omits the field entirely when nil; the studio reads it as undefined
//     and hides the ref-match column.
//   - Compiler stats are carried per-frame so the kernel-pulse animation
//     can fire on every dispatched token, matching W6's train view UX.
//
// Stable JSON tags pinned by TestTokenSnapshotJSONRoundtrip.
type TokenSnapshot struct {
	// Step is the 0-based token index inside the current generation. The
	// initial PhaseInit frame (when the runner sends one) uses Step=0 and
	// no TokenText; the first real emitted token is Step=0 with TokenText
	// set and PhaseGenerating.
	Step int `json:"step"`
	// MaxTokens is the user-requested cap on tokens to generate. Carried
	// per-frame so a late-joining client can compute a progress percentage
	// without a separate "open" event.
	MaxTokens int `json:"max_tokens"`

	// TokenID is the BPE token id (or char id for nanogpt) the model
	// emitted at this step. Always set on a PhaseGenerating frame.
	TokenID int `json:"token_id"`
	// TokenText is the decoded text fragment for TokenID. For GPT-2 BPE
	// this is the byte-decoded subword; for nanogpt it is a single
	// character. May be empty for token ids that decode to nothing
	// renderable (rare, but allowed).
	TokenText string `json:"token_text"`

	// LogitArgmax is the argmax over the logit vector at this step. For a
	// greedy decode this equals TokenID; for stochastic sampling the two
	// may differ.
	LogitArgmax int `json:"logit_argmax"`
	// LogitSummary is a human-readable summary of the last-token logit
	// vector. Format is "max=X.YZ at idx N" (single-line, ASCII-safe so
	// the SSE framing can carry it without escaping).
	LogitSummary string `json:"logit_summary,omitempty"`

	// WallMs is milliseconds elapsed since the generation started. Lets
	// the studio compute tokens/sec without a wall-clock import on the
	// client side.
	WallMs int64 `json:"wall_ms,omitempty"`

	// Phase mirrors Snapshot's Phase enum so consumers can route on the
	// same string vocabulary ("init" / "generating" / "done" / "error").
	// PhaseTraining is reused as PhaseGenerating on the wire - see the
	// Phase.String comment in snapshot.go; we don't add a new phase value
	// because the studio's CSS / a11y states already key off these four.
	Phase Phase `json:"phase"`

	// RefMatch is set when ?compare=1 enables the oracle path. The
	// pointer-vs-bool distinction is load-bearing: nil means "no oracle",
	// false means "oracle ran and disagreed", true means "matched".
	// JSON omits the field entirely when nil so the wire is forward-
	// compatible with clients that don't know about the field.
	RefMatch *bool `json:"ref_match,omitempty"`

	// Error carries the human-readable failure when Phase == PhaseError.
	Error string `json:"error,omitempty"`

	// Compiler stats: carried per-frame for the fused-kernel pulse
	// animation (same plumbing as Snapshot). The studio reads
	// LastKernelID to drive the gold pulse on the kernel-thumb SVG.
	DispatchCount        int    `json:"dispatch_count,omitempty"`
	LastKernelID         string `json:"last_kernel_id,omitempty"`
	LastDispatchWasFused bool   `json:"last_dispatch_was_fused,omitempty"`
}
