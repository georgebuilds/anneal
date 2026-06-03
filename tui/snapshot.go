package tui

import (
	"encoding/json"
	"fmt"
)

// Phase is the trainer lifecycle phase carried inside a Snapshot. It is encoded
// as a string in JSON so the SSE wire format stays human-readable; consumers
// (the studio train view, history bundles) compare strings, never integers.
type Phase int

const (
	// PhaseInit is the trainer's initial state before any step has fired. The
	// step counter is at zero, no loss has been observed.
	PhaseInit Phase = iota
	// PhaseTraining is the steady state: at least one step has been emitted
	// and training has not yet completed.
	PhaseTraining
	// PhaseDone is set after the trainer's final step completes successfully.
	PhaseDone
	// PhaseError is set when the trainer aborts with an error; the Error
	// field on the Snapshot carries the message.
	PhaseError
)

// String returns the canonical wire spelling of a Phase. The strings are the
// JSON serialization and are stable across versions.
func (p Phase) String() string {
	switch p {
	case PhaseInit:
		return "init"
	case PhaseTraining:
		return "training"
	case PhaseDone:
		return "done"
	case PhaseError:
		return "error"
	}
	return "unknown"
}

// MarshalJSON encodes Phase as a string. Treating Phase as a string in JSON
// keeps the SSE event payload trivially readable and decouples the wire
// format from the integer ordering of the iota block (which is free to grow).
func (p Phase) MarshalJSON() ([]byte, error) {
	return json.Marshal(p.String())
}

// UnmarshalJSON accepts the same string spellings MarshalJSON produces.
// Unknown spellings are rejected with a clear error so a wire-format drift
// surfaces at the consumer rather than silently mapping to PhaseInit.
func (p *Phase) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case "init":
		*p = PhaseInit
	case "training":
		*p = PhaseTraining
	case "done":
		*p = PhaseDone
	case "error":
		*p = PhaseError
	default:
		return fmt.Errorf("tui: unknown phase %q", s)
	}
	return nil
}

// Snapshot is the unified value type that the Trainer emits once per logged
// step. It captures everything the dashboard displays and is the wire format
// for the future SSE train view (spec §5.5). All fields are JSON-marshalable;
// the structure is value-only so it crosses the goroutine/SSE boundary cheaply.
//
// Design intent:
//   - One type, two consumers: the bubbletea TUI's Model.View and the future
//     SSE writer both read from the same Snapshot. Adding a third consumer is
//     additive (zero coupling to the renderer).
//   - Optional fields stay zero-valued for trainers that don't compute them.
//     The renderer treats zero values as "unset" and falls back to dashes /
//     placeholders, identical to the legacy code paths.
//   - LossHistory is a bounded slice (the Model trims it to maxSparkHistory
//     before rendering); the Snapshot carries whatever the producer chose to
//     send and the renderer keeps the sparkline window honest.
type Snapshot struct {
	// Core training progress.
	Step        int       `json:"step"`
	MaxSteps    int       `json:"max_steps"`
	Loss        float32   `json:"loss"`
	HasLoss     bool      `json:"has_loss"`
	LossHistory []float32 `json:"loss_history,omitempty"`

	// Tokens-per-step and throughput. Optional; zero means unset.
	Tokens       int     `json:"tokens,omitempty"`
	WallMs       int64   `json:"wall_ms,omitempty"`
	LearningRate float32 `json:"learning_rate,omitempty"`
	BatchSize    int64   `json:"batch_size,omitempty"`

	// Compiler stats: the scheduler's running counts. Populated by the
	// StatsHook bridge from schedule.CompilerStats; carried here so the SSE
	// stream is self-contained.
	UOpsCount            int    `json:"uops_count,omitempty"`
	KernelsCount         int    `json:"kernels_count,omitempty"`
	FusedCount           int    `json:"fused_count,omitempty"`
	DispatchCount        int    `json:"dispatch_count,omitempty"`
	Pass                 string `json:"pass,omitempty"`
	LastKernelID         string `json:"last_kernel_id,omitempty"`
	LastDispatchMs       int64  `json:"last_dispatch_ms,omitempty"`
	LastDispatchWasFused bool   `json:"last_dispatch_was_fused,omitempty"`

	// Provenance: device + backend + example name. These are constant across
	// the run and could be carried on a separate "open" SSE event, but we
	// pack them into every snapshot so a late-joining client gets a complete
	// picture from the next frame without replaying history.
	AdapterName string `json:"adapter_name,omitempty"`
	BackendName string `json:"backend_name,omitempty"`
	DeviceTag   string `json:"device_tag,omitempty"`

	// Lifecycle + free-form text.
	NoteText   string `json:"note_text,omitempty"`
	Phase      Phase  `json:"phase"`
	Error      string `json:"error,omitempty"`
	SampleText string `json:"sample_text,omitempty"`
}

// SnapshotMsg carries a Snapshot through the bubbletea program message
// stream. Model.Update consumes this and stores the Snapshot as the source
// of truth for rendering; legacy StepMsg/LossMsg/StatsMsg paths still work
// (they patch the same Snapshot field by field) so callers can migrate
// incrementally.
type SnapshotMsg struct{ Snapshot Snapshot }
