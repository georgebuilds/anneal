//go:build !js

package tui_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tui"
)

// ── Snapshot value-type tests ─────────────────────────────────────────────────

// TestSnapshotJSONRoundtrip verifies that a populated Snapshot survives a
// JSON encode/decode without losing fields. This is the contract the future
// SSE writer relies on (spec §5.5): the same Snapshot the TUI renders is
// the one the browser receives over the wire.
func TestSnapshotJSONRoundtrip(t *testing.T) {
	orig := tui.Snapshot{
		Step:                 42,
		MaxSteps:             100,
		Loss:                 0.012345,
		HasLoss:              true,
		LossHistory:          []float32{1.0, 0.5, 0.25, 0.125},
		Tokens:               1024,
		WallMs:               1500,
		LearningRate:         3e-4,
		BatchSize:            16,
		UOpsCount:            312,
		KernelsCount:         8,
		FusedCount:           2,
		DispatchCount:        20,
		Pass:                 "memory plan",
		LastKernelID:         "k7",
		LastDispatchMs:       3,
		LastDispatchWasFused: true,
		AdapterName:          "Apple M3 Pro",
		BackendName:          "Metal",
		DeviceTag:            "webgpu",
		NoteText:             "step 42 logged",
		Phase:                tui.PhaseTraining,
		Error:                "",
		SampleText:           "ROMEO: ...",
	}

	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got tui.Snapshot
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	// Reflect-equality check via re-marshal: the second marshal must equal
	// the first byte-for-byte (deterministic field order from struct tags).
	b2, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("re-Marshal: %v", err)
	}
	if string(b) != string(b2) {
		t.Errorf("roundtrip drift:\nfirst:  %s\nsecond: %s", b, b2)
	}
}

// TestPhaseJSONStrings verifies the Phase enum encodes as the documented
// string spellings ("init", "training", "done", "error"). This is the wire
// format that consumers compare against, so it must be pinned.
func TestPhaseJSONStrings(t *testing.T) {
	cases := []struct {
		p    tui.Phase
		want string
	}{
		{tui.PhaseInit, `"init"`},
		{tui.PhaseTraining, `"training"`},
		{tui.PhaseDone, `"done"`},
		{tui.PhaseError, `"error"`},
	}
	for _, c := range cases {
		b, err := json.Marshal(c.p)
		if err != nil {
			t.Errorf("Marshal(%v): %v", c.p, err)
			continue
		}
		if string(b) != c.want {
			t.Errorf("Marshal(%v) = %s, want %s", c.p, b, c.want)
		}
		var back tui.Phase
		if err := json.Unmarshal(b, &back); err != nil {
			t.Errorf("Unmarshal(%s): %v", b, err)
			continue
		}
		if back != c.p {
			t.Errorf("roundtrip Phase: got %v, want %v", back, c.p)
		}
	}
}

// TestPhaseJSONUnknownRejected verifies that an unknown phase string fails
// loudly. The wire format must surface drift at the boundary, never silently
// degrade to PhaseInit.
func TestPhaseJSONUnknownRejected(t *testing.T) {
	var p tui.Phase
	if err := json.Unmarshal([]byte(`"nope"`), &p); err == nil {
		t.Error("Unmarshal of unknown phase string should error, got nil")
	}
}

// ── Model.Snapshot accessor test ──────────────────────────────────────────────

// TestModelSnapshotAccessor verifies that Model.Snapshot returns the
// accumulated state after the legacy per-field messages, which is how the
// future SSE writer will scrape a Model rendered by the legacy path.
func TestModelSnapshotAccessor(t *testing.T) {
	m := baseModel()
	final := viewWithModel(m,
		tui.LossMsg{Step: 10, Loss: 0.5},
		tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 100, Kernels: 3, Pass: "rangeify"}},
	)
	snap := final.Snapshot()
	if snap.Step != 10 {
		t.Errorf("Snapshot.Step = %d, want 10", snap.Step)
	}
	if snap.Loss != 0.5 {
		t.Errorf("Snapshot.Loss = %g, want 0.5", snap.Loss)
	}
	if !snap.HasLoss {
		t.Error("Snapshot.HasLoss = false, want true after LossMsg")
	}
	if snap.UOpsCount != 100 {
		t.Errorf("Snapshot.UOpsCount = %d, want 100", snap.UOpsCount)
	}
	if snap.KernelsCount != 3 {
		t.Errorf("Snapshot.KernelsCount = %d, want 3", snap.KernelsCount)
	}
	if snap.Pass != "rangeify" {
		t.Errorf("Snapshot.Pass = %q, want %q", snap.Pass, "rangeify")
	}
	if snap.Phase != tui.PhaseTraining {
		t.Errorf("Snapshot.Phase = %v, want PhaseTraining", snap.Phase)
	}
}

// viewWithModel returns the final Model after applying msgs (the test helper
// in dashboard_test.go returns the rendered string; this returns the typed
// Model so we can call Snapshot()).
func viewWithModel(m tui.Model, msgs ...tea.Msg) tui.Model {
	var cur tea.Model = m
	for _, msg := range msgs {
		cur, _ = cur.Update(msg)
	}
	return cur.(tui.Model)
}

// ── Path equivalence: legacy vs SnapshotMsg ───────────────────────────────────

// TestLegacyVsSnapshotPathEquivalent is the W5 §11.5 byte-identical guarantee:
// the SnapshotMsg path must render bit-for-bit identical output to the legacy
// LossMsg+StatsMsg+DoneMsg path for the same logical state. If this drifts,
// the refactor changed visible behavior and the renderer must be patched
// back to match.
func TestLegacyVsSnapshotPathEquivalent(t *testing.T) {
	cases := []struct {
		name           string
		legacyMsgs     []tea.Msg
		equivalentSnap tui.Snapshot
		extraStats     *schedule.CompilerStats
	}{
		{
			name: "step+loss only (mlp-shaped)",
			legacyMsgs: []tea.Msg{
				tui.LossMsg{Step: 1, Loss: 0.875},
			},
			equivalentSnap: tui.Snapshot{
				Step:    1,
				Loss:    0.875,
				HasLoss: true,
				Phase:   tui.PhaseTraining,
			},
		},
		{
			name: "loss + stats (conv-shaped)",
			legacyMsgs: []tea.Msg{
				tui.LossMsg{Step: 5, Loss: 0.234},
				tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 256, Kernels: 6, Fused: 0, Pass: "memory plan"}},
			},
			equivalentSnap: tui.Snapshot{
				Step:    5,
				Loss:    0.234,
				HasLoss: true,
				Phase:   tui.PhaseTraining,
			},
			extraStats: &schedule.CompilerStats{UOps: 256, Kernels: 6, Fused: 0, Pass: "memory plan"},
		},
		{
			name: "done state (mlp 5/5)",
			legacyMsgs: []tea.Msg{
				tui.LossMsg{Step: 5, Loss: 0.001},
				tui.DoneMsg{},
			},
			equivalentSnap: tui.Snapshot{
				Step:    5,
				Loss:    0.001,
				HasLoss: true,
				Phase:   tui.PhaseDone,
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			legacy := viewWith(baseModel(), c.legacyMsgs...)

			// Build the SnapshotMsg path with the same initial Model and
			// the same companion StatsMsg if any (the snapshot shim does
			// not carry stats; they flow through the parallel StatsHook
			// channel as before).
			snapMsgs := []tea.Msg{}
			if c.extraStats != nil {
				snapMsgs = append(snapMsgs, tui.StatsMsg{Stats: *c.extraStats})
			}
			snapMsgs = append(snapMsgs, tui.SnapshotMsg{Snapshot: c.equivalentSnap})

			newPath := viewWith(baseModel(), snapMsgs...)

			if legacy != newPath {
				t.Errorf("legacy vs snapshot path differ - refactor changed visible bytes\nlegacy:\n%s\nsnapshot:\n%s",
					stripANSI(legacy), stripANSI(newPath))
			}
		})
	}
}

// ── Captured testdata: byte-identical regression ──────────────────────────────

// testdataCases lists the (legacy-message-sequence, golden-file) pairs that
// pin the dashboard's rendered output. Refactors that touch the renderer
// must either reproduce these bytes exactly or explicitly bless the diff by
// regenerating the testdata under ANNEAL_BLESS_TUI=1.
//
// The cases mirror the shape of mlp / conv / nanogpt at step 5, scaled down
// so the testdata is hand-readable.
func testdataCases() []struct {
	name string
	msgs []tea.Msg
	file string
} {
	return []struct {
		name string
		msgs []tea.Msg
		file string
	}{
		{
			name: "mlp_step5",
			file: "mlp_step5.golden",
			msgs: []tea.Msg{
				tea.WindowSizeMsg{Width: 80, Height: 24},
				tui.LossMsg{Step: 1, Loss: 0.875},
				tui.LossMsg{Step: 2, Loss: 0.621},
				tui.LossMsg{Step: 3, Loss: 0.402},
				tui.LossMsg{Step: 4, Loss: 0.251},
				tui.LossMsg{Step: 5, Loss: 0.134},
				tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 88, Kernels: 5, Fused: 0, Pass: "memory plan"}},
				tui.DoneMsg{},
			},
		},
		{
			name: "conv_step5",
			file: "conv_step5.golden",
			msgs: []tea.Msg{
				tea.WindowSizeMsg{Width: 80, Height: 24},
				tui.LossMsg{Step: 1, Loss: 1.500},
				tui.LossMsg{Step: 2, Loss: 1.100},
				tui.LossMsg{Step: 3, Loss: 0.800},
				tui.LossMsg{Step: 4, Loss: 0.600},
				tui.LossMsg{Step: 5, Loss: 0.450},
				tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 312, Kernels: 8, Fused: 0, Pass: "memory plan"}},
				tui.DoneMsg{},
			},
		},
		{
			name: "nanogpt_step5",
			file: "nanogpt_step5.golden",
			msgs: []tea.Msg{
				tea.WindowSizeMsg{Width: 80, Height: 24},
				tui.LossMsg{Step: 0, Loss: 4.1700},
				tui.LossMsg{Step: 1, Loss: 4.0500},
				tui.LossMsg{Step: 2, Loss: 3.9000},
				tui.LossMsg{Step: 3, Loss: 3.7500},
				tui.LossMsg{Step: 4, Loss: 3.5800},
				tui.LossMsg{Step: 5, Loss: 3.4200},
				tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 1024, Kernels: 24, Fused: 0, Pass: "memory plan"}},
			},
		},
		{
			name: "waiting_initial",
			file: "waiting_initial.golden",
			msgs: []tea.Msg{
				tea.WindowSizeMsg{Width: 80, Height: 24},
			},
		},
	}
}

// TestRenderGolden pins the rendered TUI output for representative cases.
// Set ANNEAL_BLESS_TUI=1 to regenerate the .golden files when an intentional
// renderer change lands.
//
// The test runs in NO_COLOR mode so the goldens are pure UTF-8 text and a
// diff is human-readable. The NO_COLOR path is the lossless carrier (DD1 §9),
// so a renderer that passes this test on NO_COLOR also preserves shape +
// label semantics in colored mode.
func TestRenderGolden(t *testing.T) {
	t.Setenv("NO_COLOR", "1")

	for _, c := range testdataCases() {
		t.Run(c.name, func(t *testing.T) {
			got := viewWith(baseModel(), c.msgs...)
			path := filepath.Join("testdata", c.file)

			if os.Getenv("ANNEAL_BLESS_TUI") == "1" {
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatalf("bless write %s: %v", path, err)
				}
				t.Logf("blessed %s (%d bytes)", path, len(got))
				return
			}

			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read golden %s: %v (set ANNEAL_BLESS_TUI=1 to create)", path, err)
			}
			if string(want) != got {
				t.Errorf("rendered output differs from %s\n"+
					"want (%d bytes):\n%s\n"+
					"got (%d bytes):\n%s\n"+
					"(set ANNEAL_BLESS_TUI=1 to bless)",
					path, len(want), string(want), len(got), got)
			}
		})
	}
}

// TestRenderGoldenSnapshotPathMatches verifies that driving the renderer
// through the new SnapshotMsg path produces the same bytes as the legacy
// per-field path captured in the goldens. This is the W5 §11.5 guarantee
// asserted end-to-end.
func TestRenderGoldenSnapshotPathMatches(t *testing.T) {
	t.Setenv("NO_COLOR", "1")

	for _, c := range testdataCases() {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join("testdata", c.file)
			want, err := os.ReadFile(path)
			if err != nil {
				t.Skipf("no golden for %s yet (%v)", c.name, err)
			}

			// Translate the legacy message sequence into a single
			// SnapshotMsg, plus any independent StatsMsg (stats flow on
			// a parallel channel in both paths).
			var (
				lastLoss float32
				hasLoss  bool
				step     int
				history  []float32
				phase    = tui.PhaseInit
				stats    schedule.CompilerStats
				width    = 80
				height   = 24
			)
			for _, m := range c.msgs {
				switch m := m.(type) {
				case tea.WindowSizeMsg:
					width, height = m.Width, m.Height
				case tui.LossMsg:
					step = m.Step
					lastLoss = m.Loss
					hasLoss = true
					history = append(history, m.Loss)
					if phase == tui.PhaseInit {
						phase = tui.PhaseTraining
					}
				case tui.StepMsg:
					step = m.Step
					if phase == tui.PhaseInit {
						phase = tui.PhaseTraining
					}
				case tui.StatsMsg:
					stats = m.Stats
				case tui.DoneMsg:
					phase = tui.PhaseDone
				case tui.ErrMsg:
					phase = tui.PhaseError
				}
			}
			_ = height
			snap := tui.Snapshot{
				Step:         step,
				Loss:         lastLoss,
				HasLoss:      hasLoss,
				LossHistory:  history,
				UOpsCount:    stats.UOps,
				KernelsCount: stats.Kernels,
				FusedCount:   stats.Fused,
				Pass:         stats.Pass,
				Phase:        phase,
			}
			got := viewWith(baseModel(),
				tea.WindowSizeMsg{Width: width, Height: 24},
				tui.SnapshotMsg{Snapshot: snap},
			)
			if string(want) != got {
				t.Errorf("SnapshotMsg path drifted from golden %s\nwant:\n%s\ngot:\n%s",
					path, string(want), got)
			}
		})
	}
}

// TestSnapshotShimBackcompat verifies that a Snapshot synthesized only from
// (step, loss) - the contract the cmd_train.go back-compat shim implements -
// drives the renderer to a sensible state, even with all the optional
// fields blank.
func TestSnapshotShimBackcompat(t *testing.T) {
	got := stripANSI(viewWith(baseModel(),
		tui.SnapshotMsg{Snapshot: tui.Snapshot{
			Step:    7,
			Loss:    0.42,
			HasLoss: true,
			Phase:   tui.PhaseTraining,
		}},
	))
	if !strings.Contains(got, "0.420000") {
		t.Errorf("shim-shaped snapshot did not surface loss; view:\n%s", got)
	}
	if !strings.Contains(got, "7/100") {
		t.Errorf("shim-shaped snapshot did not surface step; view:\n%s", got)
	}
}
