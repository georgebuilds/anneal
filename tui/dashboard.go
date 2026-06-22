//go:build !js

package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/georgebuilds/anneal/schedule"
)

// ── Messages ──────────────────────────────────────────────────────────────────

// StepMsg is sent after every training step (step counter update without loss).
//
// Legacy: kept so the per-field message path that predates the Snapshot
// refactor (W5) keeps working. Internally these messages mutate the Model's
// Snapshot field; the renderer reads only from the Snapshot.
type StepMsg struct{ Step int }

// LossMsg is sent when a loss evaluation completes.
//
// Legacy: see StepMsg.
type LossMsg struct {
	Step int
	Loss float32
}

// StatsMsg is sent by the schedule.StatsHook after each Realize call.
//
// Legacy: see StepMsg.
type StatsMsg struct{ Stats schedule.CompilerStats }

// DoneMsg is sent when training completes successfully.
//
// Legacy: see StepMsg.
type DoneMsg struct{}

// ErrMsg is sent when training fails.
//
// Legacy: see StepMsg.
type ErrMsg struct{ Err error }

// ── Config and Model ──────────────────────────────────────────────────────────

// Config holds one-time dashboard configuration set at startup.
type Config struct {
	Device     string // adapter name, e.g. "Apple M3 Pro"
	Backend    string // backend name, e.g. "Metal"
	ModelName  string // example name, e.g. "mlp"
	TotalSteps int
}

// Model is the bubbletea model for the anneal train dashboard.
//
// W5 refactor: rendering reads from a single Snapshot value (`snap`). The
// legacy per-message paths (StepMsg/LossMsg/StatsMsg/DoneMsg/ErrMsg) mutate
// the Snapshot in place, so the byte-for-byte output is unchanged but a
// future SSE consumer can construct a Snapshot directly and skip the
// per-field plumbing entirely.
type Model struct {
	cfg    Config
	snap   Snapshot
	width  int
	height int
	theme  *theme
}

const maxSparkHistory = 40

// New returns an initialized dashboard model.
func New(cfg Config) Model {
	return Model{
		cfg: cfg,
		snap: Snapshot{
			MaxSteps:    cfg.TotalSteps,
			AdapterName: cfg.Device,
			BackendName: cfg.Backend,
			Phase:       PhaseInit,
		},
		theme:  newTheme(),
		width:  80,
		height: 24,
	}
}

// Snapshot returns a copy of the Model's current Snapshot. Callers (the
// future SSE writer, test code) read from this to serialize state without
// poking the renderer internals.
func (m Model) Snapshot() Snapshot { return m.snap }

// SetStatsHook wires schedule.StatsHook to push StatsMsg to p.
// Call before starting the training goroutine; defer ClearStatsHook.
func SetStatsHook(p *tea.Program) {
	schedule.StatsHook = func(s schedule.CompilerStats) {
		p.Send(StatsMsg{Stats: s})
	}
}

// ClearStatsHook removes the schedule.StatsHook set by SetStatsHook.
func ClearStatsHook() {
	schedule.StatsHook = nil
}

// ── tea.Model interface ───────────────────────────────────────────────────────

func (m Model) Init() tea.Cmd { return nil }

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		}
	case StepMsg:
		m.snap.Step = msg.Step
		if m.snap.Phase == PhaseInit {
			m.snap.Phase = PhaseTraining
		}
	case LossMsg:
		m.snap.Step = msg.Step
		m.snap.Loss = msg.Loss
		m.snap.HasLoss = true
		m.snap.LossHistory = appendBounded(m.snap.LossHistory, msg.Loss, maxSparkHistory)
		if m.snap.Phase == PhaseInit {
			m.snap.Phase = PhaseTraining
		}
	case StatsMsg:
		applyStats(&m.snap, msg.Stats)
	case DoneMsg:
		m.snap.Phase = PhaseDone
	case ErrMsg:
		if msg.Err != nil {
			m.snap.Error = msg.Err.Error()
		}
		m.snap.Phase = PhaseError
	case SnapshotMsg:
		// New code path: the trainer hands us a fully-formed Snapshot.
		// Merge it into m.snap with two rules:
		//
		//   1. Loss history is the renderer's responsibility. If the
		//      producer didn't supply a history, derive it from the
		//      previous history plus the new loss; if it did, trim to
		//      the renderer's window. This keeps the sparkline stable
		//      across the legacy LossMsg path and the new path.
		//
		//   2. The compiler stats and config-derived fields flow through
		//      independent channels (StatsHook, New(cfg) at startup). A
		//      Snapshot that arrives with these at zero should not blow
		//      away accumulated state. We treat zero/empty as "unset"
		//      and keep the prior value; producers wanting to clear them
		//      should send an explicit reset, not rely on zeroing.
		next := msg.Snapshot
		if next.HasLoss && len(next.LossHistory) == 0 {
			next.LossHistory = appendBounded(m.snap.LossHistory, next.Loss, maxSparkHistory)
		} else if len(next.LossHistory) > maxSparkHistory {
			next.LossHistory = next.LossHistory[len(next.LossHistory)-maxSparkHistory:]
		}
		// Preserve the static config-derived header fields if the
		// producer left them blank, so a minimal Snapshot from the
		// LogFn back-compat shim still renders the header correctly.
		if next.MaxSteps == 0 {
			next.MaxSteps = m.snap.MaxSteps
		}
		if next.AdapterName == "" {
			next.AdapterName = m.snap.AdapterName
		}
		if next.BackendName == "" {
			next.BackendName = m.snap.BackendName
		}
		// Preserve compiler stats accumulated via the parallel StatsHook
		// channel. The shim does not know these so it always sends them
		// as zero; without this merge the renderer would oscillate
		// between "waiting for first step…" and the live numbers.
		if next.UOpsCount == 0 {
			next.UOpsCount = m.snap.UOpsCount
		}
		if next.KernelsCount == 0 {
			next.KernelsCount = m.snap.KernelsCount
		}
		if next.FusedCount == 0 {
			next.FusedCount = m.snap.FusedCount
		}
		if next.Pass == "" {
			next.Pass = m.snap.Pass
		}
		m.snap = next
	}
	return m, nil
}

// appendBounded appends v to hist and trims the result to at most n entries
// from the tail. Pulled out so the legacy LossMsg path and the SnapshotMsg
// path share the exact same window discipline.
func appendBounded(hist []float32, v float32, n int) []float32 {
	hist = append(hist, v)
	if len(hist) > n {
		hist = hist[len(hist)-n:]
	}
	return hist
}

// applyStats copies a schedule.CompilerStats into the Snapshot's compiler
// fields. Shared by the legacy StatsMsg path and any future direct producer.
func applyStats(s *Snapshot, c schedule.CompilerStats) {
	s.UOpsCount = c.UOps
	s.KernelsCount = c.Kernels
	s.FusedCount = c.Fused
	s.Pass = c.Pass
}

// View renders the complete dashboard.
func (m Model) View() string {
	w := m.width
	if w < 40 {
		w = 40
	}
	t := m.theme

	var sb strings.Builder

	// ── Header bar ────────────────────────────────────────────────────────
	// sentence case; device · model · backend (all human-readable state in one bar)
	header := fmt.Sprintf("  anneal train  ·  %s  ·  %s  ·  %s", m.cfg.Device, m.cfg.ModelName, m.cfg.Backend)
	sb.WriteString(t.header.Width(w).Render(header))
	sb.WriteString("\n\n")

	// ── Progress region ───────────────────────────────────────────────────
	sb.WriteString(m.renderProgress(w))
	sb.WriteString("\n\n")

	// ── Metrics (loss sparkline) ──────────────────────────────────────────
	sb.WriteString(m.renderMetrics(w))
	sb.WriteString("\n\n")

	// ── Legend: DD1 color semantics (always present; lossless in NO_COLOR) ─
	sb.WriteString(m.renderLegend(w))
	sb.WriteString("\n\n")

	// ── Compiler region ───────────────────────────────────────────────────
	sb.WriteString(m.renderCompiler(w))
	sb.WriteString("\n\n")

	// ── Footer ────────────────────────────────────────────────────────────
	sb.WriteString(m.renderFooter(w))
	sb.WriteString("\n")

	return sb.String()
}

// ── Section renderers ─────────────────────────────────────────────────────────

func (m Model) renderProgress(w int) string {
	t := m.theme
	total := m.cfg.TotalSteps
	if total == 0 {
		total = 1
	}

	// Step counter: "step 42/100"
	counter := fmt.Sprintf("  step %d/%d", m.snap.Step, total)

	// Progress bar width: leave room for counter (~15), bar brackets (2),
	// percentage (~5), separating spaces (4). Cap at 50, floor at 10.
	barW := w - len(counter) - 13
	if barW < 10 {
		barW = 10
	}
	if barW > 50 {
		barW = 50
	}

	filled := 0
	if total > 0 && m.snap.Step > 0 {
		filled = (m.snap.Step * barW) / total
	}
	if filled > barW {
		filled = barW
	}

	// Build bar: filled portion in teal (forward pass = progress), empty in faint
	fillChar := t.barFill.Render(strings.Repeat("█", filled))
	emptyChar := t.barEmpty.Render(strings.Repeat("░", barW-filled))
	bar := "[" + fillChar + emptyChar + "]"

	pct := 0
	if total > 0 {
		pct = (m.snap.Step * 100) / total
	}

	status := ""
	switch m.snap.Phase {
	case PhaseDone:
		status = "  " + t.forward.Render("done")
	case PhaseError:
		status = "  error"
	}

	return fmt.Sprintf("%s  %s  %d%%%s", counter, bar, pct, status)
}

func (m Model) renderMetrics(w int) string {
	t := m.theme
	_ = w

	if !m.snap.HasLoss {
		return "  " + t.muted.Render("loss  -")
	}

	// Loss value in teal: it is a forward-pass metric (forward=teal per DD1).
	// In NO_COLOR mode lipgloss strips the color; the label "loss" identifies it.
	lossStr := t.forward.Render(fmt.Sprintf("%.6f", m.snap.Loss))
	spark := sparkline(m.snap.LossHistory, 20)
	return fmt.Sprintf("  %s  %s  %s", t.muted.Render("loss"), lossStr, t.muted.Render(spark))
}

// renderLegend renders the DD1 color semantics inline.
// This section is the lossless NO_COLOR carrier: shapes (-, ╌, ▪) plus labels
// always distinguish the three states regardless of color support.
func (m Model) renderLegend(w int) string {
	t := m.theme
	_ = w

	// Each entry: shape (colored in mode; label always present) + description.
	// forward: solid line - teal
	fwd := t.forward.Render(SymForward) + " " + t.muted.Render("forward")
	// backward: dashed line - ember
	bwd := t.backward.Render(SymBackward) + " " + t.muted.Render("backward")
	// fused: filled box - gold
	fus := t.fused.Render(SymFused) + " " + t.muted.Render("fused")

	sep := t.faint.Render("   ")
	return "  " + t.faint.Render("legend:") + "  " + fwd + sep + bwd + sep + fus
}

func (m Model) renderCompiler(w int) string {
	t := m.theme
	_ = w

	if m.snap.KernelsCount == 0 && m.snap.UOpsCount == 0 {
		return "  " + t.muted.Render("compiler") + "  " + t.faint.Render("waiting for first step…")
	}

	// uops → kernels → fused counts show the scheduler's work in real numbers.
	// fused=0 is honest: Pass 5 (cross-boundary fusion) is not yet live in v1.
	dot := t.faint.Render(" · ")
	uops := fmt.Sprintf("%s %s", t.faint.Render("uops"), t.muted.Render(fmt.Sprintf("%d", m.snap.UOpsCount)))
	kernels := fmt.Sprintf("%s %s", t.faint.Render("kernels"), t.muted.Render(fmt.Sprintf("%d", m.snap.KernelsCount)))
	fused := fmt.Sprintf("%s %s", t.faint.Render("fused"), t.fused.Render(fmt.Sprintf("%d", m.snap.FusedCount)))
	pass := fmt.Sprintf("%s %s", t.faint.Render("pass:"), t.muted.Render(m.snap.Pass))

	return "  " + t.faint.Render("compiler:") + "  " + uops + dot + kernels + dot + fused + dot + pass
}

func (m Model) renderFooter(w int) string {
	t := m.theme
	_ = w
	// v → viz is Phase 11; present but stubbed per DESIGN.md §8 (prototype).
	return "  " + t.faint.Render("q → quit") + "   " + t.faint.Render("v → viz (phase 11)")
}

// ── Sparkline ─────────────────────────────────────────────────────────────────

// sparkline renders values as a Unicode block sparkline of the given width.
// Higher loss = taller bar (natural mapping: high value → tall bar).
// As training improves, the sparkline shows a decreasing trend.
func sparkline(values []float32, width int) string {
	if len(values) == 0 {
		return strings.Repeat(" ", width)
	}

	// Take last `width` values.
	start := 0
	if len(values) > width {
		start = len(values) - width
	}
	vals := values[start:]

	// Normalize over the visible window.
	mn, mx := vals[0], vals[0]
	for _, v := range vals {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}

	var sb strings.Builder
	// Left-pad with spaces when fewer values than width.
	for i := 0; i < width-len(vals); i++ {
		sb.WriteRune(' ')
	}

	span := mx - mn
	for _, v := range vals {
		idx := 0
		if span > 1e-10 {
			idx = int(float64(v-mn) / float64(span) * 7)
		} else {
			idx = 3 // middle level when all values are equal
		}
		if idx < 0 {
			idx = 0
		}
		if idx > 7 {
			idx = 7
		}
		sb.WriteRune(sparkBlocks[idx])
	}
	return sb.String()
}
