package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/georgebuilds/anneal/schedule"
)

// ── Messages ──────────────────────────────────────────────────────────────────

// StepMsg is sent after every training step (step counter update without loss).
type StepMsg struct{ Step int }

// LossMsg is sent when a loss evaluation completes.
type LossMsg struct {
	Step int
	Loss float32
}

// StatsMsg is sent by the schedule.StatsHook after each Realize call.
type StatsMsg struct{ Stats schedule.CompilerStats }

// DoneMsg is sent when training completes successfully.
type DoneMsg struct{}

// ErrMsg is sent when training fails.
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
type Model struct {
	cfg         Config
	step        int
	loss        float32
	hasLoss     bool
	lossHistory []float32
	stats       schedule.CompilerStats
	done        bool
	err         error
	width       int
	height      int
	theme       *theme
}

const maxSparkHistory = 40

// New returns an initialized dashboard model.
func New(cfg Config) Model {
	return Model{
		cfg:   cfg,
		theme: newTheme(),
		width: 80,
		height: 24,
	}
}

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
		m.step = msg.Step
	case LossMsg:
		m.step = msg.Step
		m.loss = msg.Loss
		m.hasLoss = true
		m.lossHistory = append(m.lossHistory, msg.Loss)
		if len(m.lossHistory) > maxSparkHistory {
			m.lossHistory = m.lossHistory[len(m.lossHistory)-maxSparkHistory:]
		}
	case StatsMsg:
		m.stats = msg.Stats
	case DoneMsg:
		m.done = true
	case ErrMsg:
		m.err = msg.Err
	}
	return m, nil
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
	counter := fmt.Sprintf("  step %d/%d", m.step, total)

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
	if total > 0 && m.step > 0 {
		filled = (m.step * barW) / total
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
		pct = (m.step * 100) / total
	}

	status := ""
	if m.done {
		status = "  " + t.forward.Render("done")
	} else if m.err != nil {
		status = "  error"
	}

	return fmt.Sprintf("%s  %s  %d%%%s", counter, bar, pct, status)
}

func (m Model) renderMetrics(w int) string {
	t := m.theme
	_ = w

	if !m.hasLoss {
		return "  " + t.muted.Render("loss  —")
	}

	// Loss value in teal: it is a forward-pass metric (forward=teal per DD1).
	// In NO_COLOR mode lipgloss strips the color; the label "loss" identifies it.
	lossStr := t.forward.Render(fmt.Sprintf("%.6f", m.loss))
	spark := sparkline(m.lossHistory, 20)
	return fmt.Sprintf("  %s  %s  %s", t.muted.Render("loss"), lossStr, t.muted.Render(spark))
}

// renderLegend renders the DD1 color semantics inline.
// This section is the lossless NO_COLOR carrier: shapes (—, ╌, ▪) plus labels
// always distinguish the three states regardless of color support.
func (m Model) renderLegend(w int) string {
	t := m.theme
	_ = w

	// Each entry: shape (colored in mode; label always present) + description.
	// forward: solid line — teal
	fwd := t.forward.Render(SymForward) + " " + t.muted.Render("forward")
	// backward: dashed line — ember
	bwd := t.backward.Render(SymBackward) + " " + t.muted.Render("backward")
	// fused: filled box — gold
	fus := t.fused.Render(SymFused) + " " + t.muted.Render("fused")

	sep := t.faint.Render("   ")
	return "  " + t.faint.Render("legend:") + "  " + fwd + sep + bwd + sep + fus
}

func (m Model) renderCompiler(w int) string {
	t := m.theme
	_ = w

	s := m.stats
	if s.Kernels == 0 && s.UOps == 0 {
		return "  " + t.muted.Render("compiler") + "  " + t.faint.Render("waiting for first step…")
	}

	// uops → kernels → fused counts show the scheduler's work in real numbers.
	// fused=0 is honest: Pass 5 (cross-boundary fusion) is not yet live in v1.
	dot := t.faint.Render(" · ")
	uops := fmt.Sprintf("%s %s", t.faint.Render("uops"), t.muted.Render(fmt.Sprintf("%d", s.UOps)))
	kernels := fmt.Sprintf("%s %s", t.faint.Render("kernels"), t.muted.Render(fmt.Sprintf("%d", s.Kernels)))
	fused := fmt.Sprintf("%s %s", t.faint.Render("fused"), t.fused.Render(fmt.Sprintf("%d", s.Fused)))
	pass := fmt.Sprintf("%s %s", t.faint.Render("pass:"), t.muted.Render(s.Pass))

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

