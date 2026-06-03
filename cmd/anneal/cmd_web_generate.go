//go:build !js

// W7 — /sse/generate: stream TokenSnapshot frames from a real generation
// run to the browser generate view. Spec: notes/anneal_web_spec.md §5.6 +
// §7.
//
// Design notes (direct copy of cmd_web_train.go's playbook):
//   - TokenSnapshots flow through a buffered channel; the SSE writer
//     dequeues + flushes one frame per token.
//   - Server-side throttle: a min-interval floor (~60 Hz) on the wire so a
//     pathologically chatty model can't flood the browser; for generate
//     the natural rate is one frame per token, which is already well
//     under 60 Hz for any model on this stack — the floor is defensive.
//   - Backpressure: when the client disconnects (r.Context() cancelled)
//     the token send is dropped; the generation goroutine continues so a
//     bundle finalises cleanly even if the tab closed mid-run.
//   - Bundle tee: per spec §6 the web tier wires bundle UNCONDITIONALLY
//     (?bundle=0 is the override to skip). GenerationRows flow into the
//     bundle.Writer alongside the SSE stream.
//   - Tests can inject a custom generateRunner (no GPU dependency) so the
//     wire contract (headers, framing, done event, cancellation, the
//     compare-to-reference toggle) is pinned in pure Go without a real
//     gpt2.Sample invocation.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/georgebuilds/anneal/internal/bundle"
	"github.com/georgebuilds/anneal/tui"
)

// generateRunner is the seam tests use to swap the real GPU generation
// pipeline (gpt2.Sample or examples nanogpt generate) for a stub.
// Production wires it to runGenerateNative (which opens WebGPU). Tests
// replace it with a token-emitting fake.
//
// The runner receives a context (for cancellation), a model name, prompt,
// max tokens, a compare flag (1 = include ref_match per token), and a
// callback it should invoke per token. The runner returns only after the
// run finishes (cleanly or with an error). The handler closes the token
// channel after the runner returns.
type generateRunner func(
	ctx context.Context,
	model, prompt string,
	maxTokens int,
	compare bool,
	emit func(tui.TokenSnapshot),
) error

// generateRunnerFn is the currently-installed runner. Tests overwrite it
// via withStubGenerateRunner; production wires runGenerateNative inside
// the registered handler.
var generateRunnerFn generateRunner = runGenerateNative

// sseGenerateMaxTokens caps the tokens query so a single SSE request can't
// kick off a 1M-token run. Matches the HTML control's max attribute.
const sseGenerateMaxTokens = 256

// sseGeneratePromptMaxLen caps the prompt length so an absurdly large
// query string can't OOM the encoder. Matches the studio's maxlength.
const sseGeneratePromptMaxLen = 2048

// validGenerateModels is the set of models the generate view exposes.
// Kept tight (gpt2 + nanogpt) because the production runner has bespoke
// wiring for each; widening it without wiring breaks at runtime.
var validGenerateModels = map[string]bool{
	"gpt2":    true,
	"nanogpt": true,
}

// handleSSEGenerate is the /sse/generate endpoint. Spec §5.6 + §7.
//
// Query params:
//   - model:   "gpt2" or "nanogpt" (required)
//   - prompt:  URL-encoded prompt string (required, non-empty)
//   - tokens:  max tokens to generate (optional; default 32; 1..256)
//   - compare: "1" enables ref-match per token (optional; default off)
//   - bundle:  "0" disables the bundle tee; anything else (or absent)
//     enables it (web tier default = ON per spec §6)
func handleSSEGenerate(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" {
		writeJSONError(w, http.StatusBadRequest, "missing model",
			"the model query parameter is required")
		return
	}
	if !validGenerateModels[model] {
		writeJSONError(w, http.StatusBadRequest, "unknown model",
			fmt.Sprintf("generate supports gpt2 and nanogpt; got %q", model))
		return
	}

	prompt := r.URL.Query().Get("prompt")
	if strings.TrimSpace(prompt) == "" {
		writeJSONError(w, http.StatusBadRequest, "missing prompt",
			"the prompt query parameter is required and must be non-empty")
		return
	}
	if len(prompt) > sseGeneratePromptMaxLen {
		writeJSONError(w, http.StatusBadRequest, "prompt too long",
			fmt.Sprintf("prompt length %d exceeds cap %d", len(prompt), sseGeneratePromptMaxLen))
		return
	}

	tokens := 32
	if v := r.URL.Query().Get("tokens"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 || n > sseGenerateMaxTokens {
			writeJSONError(w, http.StatusBadRequest, "invalid tokens",
				fmt.Sprintf("tokens must be an integer in 1..%d, got %q", sseGenerateMaxTokens, v))
			return
		}
		tokens = n
	}

	compare := false
	if v := r.URL.Query().Get("compare"); v == "1" || v == "true" {
		compare = true
	}

	// Bundle: ON by default (spec §6, web tier). ?bundle=0 opts out.
	enableBundle := true
	if v := r.URL.Query().Get("bundle"); v == "0" || v == "false" {
		enableBundle = false
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONError(w, http.StatusInternalServerError, "flush unsupported",
			"the response writer does not implement http.Flusher (SSE requires it)")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ctx := r.Context()

	// Optional bundle sink. Same shape as the train handler: opening
	// errors are non-fatal; the SSE stream still flows.
	var bw *bundle.Writer
	if enableBundle {
		if root, err := bundle.EnvOrDefault(); err == nil {
			bw, _ = bundle.NewWriter(root, model, bundle.KindGenerate)
			if bw != nil {
				_ = bw.SetProvenance(version, "", "", "", "", nil)
				_ = bw.WriteConfig(bundle.Config{
					Model: model,
					Hyperparams: map[string]any{
						"prompt":     prompt,
						"max_tokens": tokens,
						"compare":    compare,
					},
				})
			}
		}
	}

	tokCh := make(chan tui.TokenSnapshot, 32)
	startWall := time.Now()

	go func() {
		defer close(tokCh)
		_ = generateRunnerFn(ctx, model, prompt, tokens, compare, func(s tui.TokenSnapshot) {
			select {
			case tokCh <- s:
			case <-ctx.Done():
			}
		})
	}()

	var lastTokenWrite time.Time
	totalTokens := 0
	for tok := range tokCh {
		// Server-side throttle: drop token frames that arrive within
		// 16 ms of the previous token write. Lifecycle frames (init /
		// done / error) bypass the throttle AND do not advance the
		// throttle clock, so an init frame can land immediately before
		// the first token without delaying it. For generation the
		// natural per-token cadence is far below 60 Hz so this is
		// defensive against a pathological emitter.
		isLifecycle := tok.Phase == tui.PhaseInit ||
			tok.Phase == tui.PhaseDone ||
			tok.Phase == tui.PhaseError
		if !isLifecycle && !lastTokenWrite.IsZero() && time.Since(lastTokenWrite) < sseMinIntervalMs*time.Millisecond {
			continue
		}

		// Bundle tee: persist the per-token row alongside the wire.
		// Skip the lifecycle-only frames (PhaseInit, PhaseDone) since
		// they carry no token text.
		if bw != nil && tok.Phase == tui.PhaseTraining && tok.TokenText != "" {
			_ = bw.AppendGeneration(bundle.GenerationRow{
				Step:         tok.Step,
				TokenID:      tok.TokenID,
				TokenText:    tok.TokenText,
				LogitArgmax:  tok.LogitArgmax,
				LogitSummary: tok.LogitSummary,
				RefMatch:     tok.RefMatch,
			})
		}
		if tok.Phase == tui.PhaseTraining {
			totalTokens++
		}

		body, err := json.Marshal(tok)
		if err != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n",
				jsonString("token snapshot marshal: "+err.Error()))
			flusher.Flush()
			break
		}
		fmt.Fprintf(w, "data: %s\n\n", body)
		flusher.Flush()
		if !isLifecycle {
			lastTokenWrite = time.Now()
		}

		if ctx.Err() != nil {
			break
		}
	}

	if ctx.Err() == nil {
		// `event: done` terminator carries the run totals so the client
		// can flip the "save run" link without re-counting tokens.
		wallMs := time.Since(startWall).Milliseconds()
		donePayload := map[string]any{
			"total_tokens": totalTokens,
			"wall_ms":      wallMs,
		}
		body, _ := json.Marshal(donePayload)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", body)
		flusher.Flush()
	}

	if bw != nil {
		_ = bw.Finalize(time.Since(startWall).Milliseconds())
		_ = bw.Close()
	}
}
