// Package tui provides the bubbletea/lipgloss training dashboard for anneal.
// Colors follow DESIGN.md §3.3 and DD1: teal=forward, ember=backward, gold=fused.
// Color is never the sole carrier — shapes/labels pair with every color (§9).
package tui

import "github.com/charmbracelet/lipgloss"

// Color token constants — exact hexes from DESIGN.md §3.3.
// Exported for tests so they can verify the values match the spec precisely.
const (
	ColorInk     = "#14110F" // canvas / terminal background
	ColorSurface = "#1F1A17" // raised surface, header bars
	ColorTeal    = "#00ADD8" // forward pass (DD1)
	ColorEmber   = "#FF7A45" // backward pass (DD1)
	ColorGold    = "#F2C57C" // fused (DD1)
	ColorText    = "#E8E2DA" // primary text
	ColorMuted   = "#8A817A" // secondary text
	ColorFaint   = "#5C544D" // separators, tertiary marks
)

// Shape symbols for the NO_COLOR path — paired with labels so no information
// is lost when color is unavailable (DESIGN.md §3.3, §9).
// Exported for tests that verify the lossless-no-color invariant.
const (
	SymForward  = "—" // solid line → forward pass
	SymBackward = "╌" // dashed line → backward pass
	SymFused    = "▪" // filled box → fused
)

// sparkBlocks is the Unicode block progression for sparklines (low→high).
var sparkBlocks = []rune{'▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'}

// theme holds the lipgloss styles for the dashboard.
type theme struct {
	header   lipgloss.Style // surface background, text foreground
	text     lipgloss.Style // primary text
	muted    lipgloss.Style // secondary text (labels, counts)
	faint    lipgloss.Style // separators, tertiary marks
	forward  lipgloss.Style // teal — forward pass metric
	backward lipgloss.Style // ember — backward pass metric
	fused    lipgloss.Style // gold — fused kernel
	barFill  lipgloss.Style // teal — progress bar filled portion
	barEmpty lipgloss.Style // faint — progress bar empty portion
}

func newTheme() *theme {
	return &theme{
		header:   lipgloss.NewStyle().Background(lipgloss.Color(ColorSurface)).Foreground(lipgloss.Color(ColorText)),
		text:     lipgloss.NewStyle().Foreground(lipgloss.Color(ColorText)),
		muted:    lipgloss.NewStyle().Foreground(lipgloss.Color(ColorMuted)),
		faint:    lipgloss.NewStyle().Foreground(lipgloss.Color(ColorFaint)),
		forward:  lipgloss.NewStyle().Foreground(lipgloss.Color(ColorTeal)),
		backward: lipgloss.NewStyle().Foreground(lipgloss.Color(ColorEmber)),
		fused:    lipgloss.NewStyle().Foreground(lipgloss.Color(ColorGold)),
		barFill:  lipgloss.NewStyle().Foreground(lipgloss.Color(ColorTeal)),
		barEmpty: lipgloss.NewStyle().Foreground(lipgloss.Color(ColorFaint)),
	}
}
