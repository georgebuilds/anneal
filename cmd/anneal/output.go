package main

import (
	"fmt"
	"os"
	"strings"
)

// noColor reports whether ANSI escape codes should be suppressed.
// The NO_COLOR convention (https://no-color.org) is: any non-empty value disables color.
func noColor() bool {
	return os.Getenv("NO_COLOR") != ""
}

// bold wraps s in ANSI bold escape codes unless NO_COLOR is set.
func bold(s string) string {
	if noColor() {
		return s
	}
	return "\033[1m" + s + "\033[0m"
}

// formatError formats a blameless, actionable error message.
// summary is the first line (what happened). details are optional follow-up lines.
func formatError(summary string, details ...string) string {
	var b strings.Builder
	b.WriteString(summary)
	if len(details) > 0 {
		b.WriteString("\n")
		for _, d := range details {
			b.WriteString(d)
			b.WriteString("\n")
		}
	}
	return b.String()
}

// noAdapterError is the canonical blameless no-GPU error message per DESIGN.md §4.
// Other commands use this when they fail to open the WebGPU adapter.
func noAdapterError() string {
	return formatError(
		"no WebGPU adapter found — anneal needs a Vulkan, Metal, or D3D12 backend at runtime.",
		"",
		"anneal requires one of the following GPU backends to be available:",
		"  Metal    — macOS 10.14+ with a Metal-capable GPU (most Apple hardware)",
		"  Vulkan   — Linux/Windows with a Vulkan-capable GPU and drivers installed",
		"  D3D12    — Windows 10+ with a DirectX 12-capable GPU",
		"",
		fmt.Sprintf("run '%s' to see what's available.", bold("anneal doctor")),
	)
}

// doctorFailureMsg returns the doctor-specific failure message (used when
// doctorCmd itself cannot open a WebGPU adapter). Unlike noAdapterError, it
// does NOT end with "run anneal doctor" since the user is already in doctor.
func doctorFailureMsg() string {
	return formatError(
		"no WebGPU adapter found",
		"anneal needs a Vulkan, Metal, or D3D12 backend — none was detected.",
		"",
		"to get started:",
		"  macOS    ensure Xcode command-line tools are installed (provides Metal)",
		"  Linux    install Vulkan drivers (e.g. mesa-vulkan-drivers or your GPU vendor's package)",
		"  Windows  install DirectX 12-capable drivers from your GPU vendor",
	)
}
