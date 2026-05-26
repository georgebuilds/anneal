package tui_test

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tui"
)

// baseModel returns a dashboard model configured for rendering tests.
func baseModel() tui.Model {
	return tui.New(tui.Config{
		Device:     "Apple M3 Pro",
		Backend:    "Metal",
		ModelName:  "mlp",
		TotalSteps: 100,
	})
}

// viewWith feeds msgs into m and returns the rendered view.
func viewWith(m tui.Model, msgs ...tea.Msg) string {
	var cur tea.Model = m
	for _, msg := range msgs {
		cur, _ = cur.Update(msg)
	}
	return cur.View()
}

// stripANSI removes ANSI escape sequences from s for plain-text comparison.
func stripANSI(s string) string {
	in := false
	var b strings.Builder
	for _, r := range s {
		if r == '\033' {
			in = true
			continue
		}
		if in {
			if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				in = false
			}
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// maxLineWidth returns the maximum visible line length (after stripping ANSI).
func maxLineWidth(s string) int {
	lines := strings.Split(stripANSI(s), "\n")
	max := 0
	for _, l := range lines {
		n := len([]rune(l))
		if n > max {
			max = n
		}
	}
	return max
}

// ── Color token tests ─────────────────────────────────────────────────────────

// TestColorTokenValues verifies the exact hexes match DESIGN.md §3.3.
func TestColorTokenValues(t *testing.T) {
	cases := []struct{ name, got, want string }{
		{"ink", tui.ColorInk, "#14110F"},
		{"surface", tui.ColorSurface, "#1F1A17"},
		{"teal", tui.ColorTeal, "#00ADD8"},
		{"ember", tui.ColorEmber, "#FF7A45"},
		{"gold", tui.ColorGold, "#F2C57C"},
		{"text", tui.ColorText, "#E8E2DA"},
		{"muted", tui.ColorMuted, "#8A817A"},
		{"faint", tui.ColorFaint, "#5C544D"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("color %s = %q, want %q (DESIGN.md §3.3)", c.name, c.got, c.want)
		}
	}
}

// ── Shape symbol tests ────────────────────────────────────────────────────────

// TestShapeSymbolsDistinct verifies the three DD1 shape symbols are present,
// non-empty, and distinct — they are the lossless NO_COLOR carriers (§9).
func TestShapeSymbolsDistinct(t *testing.T) {
	syms := map[string]string{
		"forward":  tui.SymForward,
		"backward": tui.SymBackward,
		"fused":    tui.SymFused,
	}
	for name, sym := range syms {
		if sym == "" {
			t.Errorf("Sym%s is empty — must be a non-empty shape character", name)
		}
	}
	if tui.SymForward == tui.SymBackward || tui.SymForward == tui.SymFused || tui.SymBackward == tui.SymFused {
		t.Errorf("shape symbols not all distinct: forward=%q backward=%q fused=%q",
			tui.SymForward, tui.SymBackward, tui.SymFused)
	}
}

// ── NO_COLOR lossless legend tests ────────────────────────────────────────────

// TestNoColorLegendLossless verifies that in NO_COLOR mode the legend still
// contains all three shape symbols AND labels — color loss is truly lossless
// (DESIGN.md §9, §3.3).
func TestNoColorLegendLossless(t *testing.T) {
	t.Setenv("NO_COLOR", "1")

	view := stripANSI(baseModel().View())

	for _, sym := range []string{tui.SymForward, tui.SymBackward, tui.SymFused} {
		if !strings.Contains(view, sym) {
			t.Errorf("NO_COLOR view missing shape symbol %q — shape must survive color removal", sym)
		}
	}
	for _, label := range []string{"forward", "backward", "fused"} {
		if !strings.Contains(view, label) {
			t.Errorf("NO_COLOR view missing label %q — label must pair with shape (§9)", label)
		}
	}
}

// TestNoColorNoANSI verifies that NO_COLOR produces no ANSI escape codes.
func TestNoColorNoANSI(t *testing.T) {
	t.Setenv("NO_COLOR", "1")

	if strings.Contains(baseModel().View(), "\033[") {
		t.Error("NO_COLOR mode produced ANSI escape codes — color must be fully suppressed")
	}
}

// ── Sentence case tests ───────────────────────────────────────────────────────

// TestSentenceCase verifies labels use sentence case — no Title Case, no ALL
// CAPS (DESIGN.md §9).
func TestSentenceCase(t *testing.T) {
	view := stripANSI(baseModel().View())

	banned := []string{
		"LOSS", "STEP", "FORWARD", "BACKWARD", "FUSED", "LEGEND",
		"COMPILER", "QUIT", "VIZ", "DONE", "PASS",
		"Loss", "Step", "Legend", "Compiler", "Quit", "Viz", "Pass:",
	}
	for _, phrase := range banned {
		if strings.Contains(view, phrase) {
			t.Errorf("view contains title-case/ALL-CAPS phrase %q — want sentence case", phrase)
		}
	}
}

// ── 80-column layout test ─────────────────────────────────────────────────────

// TestLayout80Col verifies no line exceeds 80 columns when width=80
// (DESIGN.md §6 degradation hard requirement).
func TestLayout80Col(t *testing.T) {
	view := viewWith(
		baseModel(),
		tea.WindowSizeMsg{Width: 80, Height: 24},
		tui.LossMsg{Step: 42, Loss: 0.012345},
		tui.StatsMsg{Stats: schedule.CompilerStats{
			UOps: 312, Kernels: 8, Fused: 0, Pass: "memory plan",
		}},
	)

	if w := maxLineWidth(view); w > 80 {
		t.Errorf("at width=80, longest line is %d characters (want ≤80)\nview:\n%s", w, stripANSI(view))
	}
}

// ── Compiler stats wiring tests ───────────────────────────────────────────────

// TestCompilerStatsInView verifies that real stats from a StatsMsg appear in
// the rendered output — counts must not be hardcoded constants.
func TestCompilerStatsInView(t *testing.T) {
	view := stripANSI(viewWith(
		baseModel(),
		tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 312, Kernels: 8, Pass: "memory plan"}},
	))

	for _, want := range []string{"312", "8", "memory plan"} {
		if !strings.Contains(view, want) {
			t.Errorf("compiler stats missing %q in view:\n%s", want, view)
		}
	}
}

// TestCompilerStatsNotHardcoded verifies that different stat values produce
// different output — rules out any hardcoded constant in the renderer.
func TestCompilerStatsNotHardcoded(t *testing.T) {
	view1 := stripANSI(viewWith(baseModel(),
		tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 100, Kernels: 3, Pass: "rangeify"}},
	))
	view2 := stripANSI(viewWith(baseModel(),
		tui.StatsMsg{Stats: schedule.CompilerStats{UOps: 999, Kernels: 17, Pass: "memory plan"}},
	))

	if view1 == view2 {
		t.Error("different CompilerStats produced identical views — stats may be hardcoded")
	}
	if !strings.Contains(view1, "100") {
		t.Errorf("UOps=100 not found in view1:\n%s", view1)
	}
	if !strings.Contains(view2, "999") {
		t.Errorf("UOps=999 not found in view2:\n%s", view2)
	}
}

// ── Content correctness tests ─────────────────────────────────────────────────

// TestLossValueVisible verifies that a LossMsg makes the loss value appear.
func TestLossValueVisible(t *testing.T) {
	view := stripANSI(viewWith(baseModel(), tui.LossMsg{Step: 5, Loss: 0.123456}))
	if !strings.Contains(view, "0.123456") {
		t.Errorf("loss 0.123456 not found in view:\n%s", view)
	}
}

// TestStepCounterVisible verifies the step counter shows current/total.
func TestStepCounterVisible(t *testing.T) {
	view := stripANSI(viewWith(baseModel(), tui.StepMsg{Step: 42}))
	if !strings.Contains(view, "42/100") {
		t.Errorf("step counter 42/100 not found in view:\n%s", view)
	}
}

// TestDoneVisible verifies DoneMsg produces a done indicator.
func TestDoneVisible(t *testing.T) {
	view := stripANSI(viewWith(baseModel(), tui.DoneMsg{}))
	if !strings.Contains(view, "done") {
		t.Errorf("DoneMsg did not produce 'done' in view:\n%s", view)
	}
}

// TestHeaderContent verifies device, model, backend, and "anneal train" appear.
func TestHeaderContent(t *testing.T) {
	view := stripANSI(baseModel().View())
	for _, want := range []string{"Apple M3 Pro", "mlp", "Metal", "anneal train"} {
		if !strings.Contains(view, want) {
			t.Errorf("header missing %q; view:\n%s", want, view)
		}
	}
}

// TestFooterKeyBindings verifies footer shows q→quit and v→viz.
func TestFooterKeyBindings(t *testing.T) {
	view := stripANSI(baseModel().View())
	for _, want := range []string{"q →", "quit", "v →", "viz"} {
		if !strings.Contains(view, want) {
			t.Errorf("footer missing %q; view:\n%s", want, view)
		}
	}
}
