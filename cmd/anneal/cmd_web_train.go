//go:build !js

// W6 — /sse/train: stream Snapshot frames from a real training run to the
// browser dashboard. Spec: notes/anneal_web_spec.md §5.5 + §7.
//
// Design notes:
//   - Snapshots flow through cfg.SnapshotFn (the W5 channel) into a buffered
//     channel; the SSE writer dequeues + flushes one frame per snapshot.
//   - Server-side throttle: a min-interval floor (~60 Hz) on the wire so a
//     pathologically chatty trainer can't flood the browser; LogEvery sets
//     the real cadence.
//   - Backpressure: when the client disconnects (r.Context() cancelled) the
//     snapshot send is dropped; the training goroutine continues to its end
//     so a bundle finalises cleanly even if the tab closed mid-run.
//   - Bundle tee: per spec §6 the web tier wires bundle UNCONDITIONALLY
//     (?bundle=0 is the override to skip). Loss rows + events flow into the
//     bundle.Writer alongside the SSE stream.
//   - Tests can inject a custom trainRunner (no GPU dependency) so the
//     wire contract (headers, framing, done event, cancellation) is
//     pinned in pure Go without a real Train invocation.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/internal/bundle"
	"github.com/georgebuilds/anneal/tui"
)

// trainRunner is the seam tests use to swap the real GPU training loop for
// a stub. Production wires it to runTrainNative (which opens WebGPU). Tests
// replace it with a snapshot-emitting fake.
//
// The runner receives a context (for cancellation), a model name + parsed
// steps + a SnapshotFn it should call with each step's Snapshot. It returns
// only after the run finishes (cleanly or with an error). The handler closes
// the snapshot channel after the runner returns.
type trainRunner func(ctx context.Context, model string, steps int, snap func(tui.Snapshot)) error

// trainRunnerFn is the currently-installed runner. Tests overwrite it via
// withStubTrainRunner; production wires runTrainNative inside the registered
// handler.
var trainRunnerFn trainRunner = runTrainNative

// sseTrainMaxSteps caps the steps query so a single SSE request can't
// kick off a 10M-step run. Matches the HTML control's max attribute.
const sseTrainMaxSteps = 10000

// sseMinIntervalMs is the server-side wire floor: at least this many
// milliseconds elapse between frames pushed to the browser. ~60 Hz.
const sseMinIntervalMs = 16

// handleSSETrain is the /sse/train endpoint. Spec §5.5 + §7.
//
// Query params:
//   - model:  example name (required; must be in the examples registry)
//   - steps:  optional override (must be 1..sseTrainMaxSteps)
//   - bundle: "0" disables the bundle tee; anything else (or absent) enables
//     it (web tier default = ON per spec §6)
func handleSSETrain(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" {
		writeJSONError(w, http.StatusBadRequest, "missing model",
			"the model query parameter is required")
		return
	}
	if _, err := examples.Get(model); err != nil {
		writeJSONError(w, http.StatusBadRequest, "unknown model", err.Error())
		return
	}

	steps := 100
	if v := r.URL.Query().Get("steps"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 || n > sseTrainMaxSteps {
			writeJSONError(w, http.StatusBadRequest, "invalid steps",
				fmt.Sprintf("steps must be an integer in 1..%d, got %q", sseTrainMaxSteps, v))
			return
		}
		steps = n
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
	// Disable Nginx-style buffering for proxies in front of `anneal web`.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ctx := r.Context()

	// Optional bundle sink (web tier default = ON). Opening errors are
	// non-fatal: the SSE stream still flows even if the bundle directory
	// is unreachable; the client just gets a one-line "bundle: skip" note.
	var bw *bundle.Writer
	if enableBundle {
		if root, err := bundle.EnvOrDefault(); err == nil {
			bw, _ = bundle.NewWriter(root, model, bundle.KindTrain)
			if bw != nil {
				_ = bw.SetProvenance(version, "", "", "", "", nil)
				_ = bw.WriteConfig(bundle.Config{
					Model: model,
					Hyperparams: map[string]any{
						"steps": steps,
					},
				})
			}
		}
	}

	// Buffered snapshot channel: a small buffer lets a fast trainer emit
	// without blocking on the main goroutine's wire writes; the throttle
	// below decides when each one lands on the SSE socket.
	snapCh := make(chan tui.Snapshot, 32)
	startWall := time.Now()

	// Train in a separate goroutine. SnapshotFn drops snapshots when the
	// client has disconnected (ctx done) so the trainer doesn't block on a
	// full channel after we stop reading.
	go func() {
		defer close(snapCh)
		_ = trainRunnerFn(ctx, model, steps, func(s tui.Snapshot) {
			select {
			case snapCh <- s:
			case <-ctx.Done():
			}
		})
	}()

	// Stream loop: dequeue snapshots, throttle to ~60 Hz, write SSE frames.
	var lastWrite time.Time
	for snap := range snapCh {
		// Server-side throttle: drop frames that arrive within 16 ms of the
		// last write. The CLI's LogEvery already coarse-grains; this is a
		// belt-and-braces floor on the wire.
		if !lastWrite.IsZero() && time.Since(lastWrite) < sseMinIntervalMs*time.Millisecond {
			continue
		}

		// Bundle tee: persist the per-snapshot loss row alongside the wire.
		if bw != nil && snap.HasLoss {
			_ = bw.AppendLoss(bundle.LossRow{
				Step:   snap.Step,
				Loss:   snap.Loss,
				WallMs: snap.WallMs,
			})
		}

		body, err := json.Marshal(snap)
		if err != nil {
			// Marshalling Snapshot should never fail for normal float32
			// data; if it does we abort the stream cleanly.
			_, _ = fmt.Fprintf(w, "event: error\ndata: %s\n\n",
				jsonString("snapshot marshal: "+err.Error()))
			flusher.Flush()
			break
		}
		// SSE frame: data: <json>\n\n  (one data line, blank-line terminated)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
		flusher.Flush()
		lastWrite = time.Now()

		if ctx.Err() != nil {
			break
		}
	}

	// Always emit the done event so the client knows to close the
	// EventSource cleanly; the client's `event: done` handler resolves
	// pending promises and enables the open-in-viz / save-run buttons.
	if ctx.Err() == nil {
		_, _ = fmt.Fprintf(w, "event: done\ndata: {}\n\n")
		flusher.Flush()
	}

	if bw != nil {
		_ = bw.Finalize(time.Since(startWall).Milliseconds())
		_ = bw.Close()
	}
}

// runTrainNative is the production trainRunner: opens the WebGPU device,
// wires cfg.SnapshotFn = snap, runs examples.Get(model).Train. It exists in
// a //go:build !js file so the SSE handler stays GPU-free for tests; the
// real implementation lives in cmd_web_train_runner.go.

// jsonString returns a JSON-quoted string. Used for the error event payload
// so newlines and quotes in err.Error() can't break the SSE framing.
func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
