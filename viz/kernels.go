// Kernels view data (W2). Pure-Go: builds the kernel set for a named example
// by walking the same realize path that `anneal kernels` uses on the CLI
// (frontend graph build + Backward + scheduler + WGSL codegen), and returns a
// JSON document the studio's kernels view renders.
//
// DD2: the WGSL text is the real compiler's output, not a mock. This file is
// the same on native and js/wasm - no build tags, no backend/webgpu import.
// The compile target check in TestWebKernels_WASMBuildable (cmd_web_test.go)
// pins this.

package viz

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// KernelSet is the top-level JSON payload returned by annealGetKernels.
type KernelSet struct {
	Model   string       `json:"model"`
	Kernels []KernelData `json:"kernels"`
}

// KernelData is one entry in KernelSet.Kernels. Field shape mirrors the
// contract in anneal_web_spec §4 / §5.3 and is consumed by studio.js.
type KernelData struct {
	ID          string       `json:"id"`
	OpCount     int          `json:"op_count"`
	BuffersIn   int          `json:"buffers_in"`
	BuffersOut  int          `json:"buffers_out"`
	Shape       []int64      `json:"shape"`
	WGSL        string       `json:"wgsl"`
	FusionSpans []FusionSpan `json:"fusion_spans"`
}

// FusionSpan is one contiguous run of WGSL source lines attributed to a single
// compilation phase (forward, backward, fused). Line numbers are 1-based,
// inclusive on both ends, matching what an editor would show.
type FusionSpan struct {
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Label     string `json:"label"`
}

// Span labels.
const (
	FusionLabelForward  = "fwd"
	FusionLabelBackward = "bwd"
	FusionLabelFused    = "fused"
)

// ToJSON serializes the KernelSet to canonical JSON bytes.
func (k *KernelSet) ToJSON() ([]byte, error) { return json.Marshal(k) }

// BuildKernels constructs the full forward + backward graph for the named
// example, runs the scheduler, renders WGSL for every kernel, and returns the
// kernel set. It walks the same realize path as cmd/anneal/cmd_kernels.go so
// the JSON kernel order and WGSL bytes match the CLI byte-for-byte (the test
// TestWebKernels_MatchesCLI in cmd/anneal/ pins this).
func BuildKernels(name string) (*KernelSet, error) {
	ex, err := examples.Get(name)
	if err != nil {
		return nil, err
	}
	result, err := ex.Build("webgpu")
	if err != nil {
		return nil, fmt.Errorf("viz: build %q: %w", name, err)
	}

	a := result.Arena
	out := result.Output

	// Match `anneal kernels` exactly: SINK over the forward output. The CLI
	// does NOT include the backward graph in its schedule (see cmd_kernels.go);
	// matching that order is the only way the bytewise CLI parity test holds.
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{out.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")

	// Build per-kernel phase attribution by walking the post-Backward graph.
	// We need to call Backward so the arena has phase metadata for backward
	// nodes that fusion spans can attribute. Skip when no leaves are exposed
	// (some examples are forward-only).
	if len(result.Leaves) > 0 {
		loss := out.Sum(nil, false)
		_, _ = tensor.BackwardWithTrace(loss, result.Leaves)
	}

	kernels := make([]KernelData, 0, len(items))
	for i, item := range items {
		kernels = append(kernels, buildKernelData(i, item))
	}

	return &KernelSet{Model: name, Kernels: kernels}, nil
}

// buildKernelData renders one kernel's WGSL and derives its summary fields.
func buildKernelData(idx int, item schedule.ExecItem) KernelData {
	wgsl := codegen.RenderWGSL(item).WGSL
	opCount := countOps(item)
	bufIn, bufOut := countBuffers(item)
	shape := outputShape(item)
	spans := fusionSpans(wgsl, item.Ast.Arena())

	return KernelData{
		ID:          fmt.Sprintf("K%d", idx),
		OpCount:     opCount,
		BuffersIn:   bufIn,
		BuffersOut:  bufOut,
		Shape:       shape,
		WGSL:        wgsl,
		FusionSpans: spans,
	}
}

// countOps walks the kernel AST and counts non-leaf UOps. Buffers, Const,
// Range, DefineVar, Param, and the kernel SINK itself are excluded so the
// number reflects "operations performed" rather than "nodes in the tree".
func countOps(item schedule.ExecItem) int {
	if !item.Ast.Valid() {
		return 0
	}
	seen := make(map[uint32]bool)
	count := 0

	var visit func(u uop.UOp)
	visit = func(u uop.UOp) {
		if seen[u.Index()] {
			return
		}
		seen[u.Index()] = true
		if isOpKind(u.Op()) {
			count++
		}
		for i := 0; i < u.NSrc(); i++ {
			visit(u.Src(i))
		}
	}
	visit(item.Ast)
	return count
}

// isOpKind reports whether the given Op represents a counted operation
// (everything other than leaves, ranges, parameter bindings, and structural
// nodes). The list mirrors what RenderWGSL turns into a real WGSL expression.
func isOpKind(op uop.Op) bool {
	switch op {
	case uop.OpBuffer, uop.OpConst, uop.OpRange, uop.OpDefineVar,
		uop.OpSpecial, uop.OpParam, uop.OpSink, uop.OpDefineLocal,
		uop.OpDefineReg, uop.OpNoop:
		return false
	}
	return true
}

// countBuffers returns (in, out) using the kernel's Bufs descriptor: Bufs[0]
// is always the output; Bufs[1..] are the inputs. Mirrors codegen's bind
// numbering so the studio's reported counts match the WGSL @binding indices.
func countBuffers(item schedule.ExecItem) (int, int) {
	if len(item.Bufs) == 0 {
		return 0, 0
	}
	return len(item.Bufs) - 1, 1
}

// outputShape returns the shape of the kernel's output buffer (Bufs[0]).
// Returns nil for kernels with no buffers (shouldn't happen in practice).
func outputShape(item schedule.ExecItem) []int64 {
	if len(item.Bufs) == 0 {
		return nil
	}
	// Defensive copy so a JSON marshal never holds the schedule's slice.
	shape := make([]int64, len(item.Bufs[0].Shape))
	copy(shape, item.Bufs[0].Shape)
	return shape
}

// letPattern matches a WGSL `let t<idx>:` binding. idx is the source UOp's
// arena index; codegen names every fused intermediate this way (see
// emitExpr in codegen/lower.go which writes `t%d` for InstrLet with NodeIdx).
// Named lets (emitted with explicit Name=...) do not match this pattern and
// are attributed via the "carry the previous phase" rule below.
var letPattern = regexp.MustCompile(`\blet\s+t(\d+)\s*:`)

// fusionSpans walks the rendered WGSL line by line, attributing each line to
// a phase by looking up the arena's Provenance for any `let t<idx>` binding
// on that line. Consecutive lines with the same attribution are coalesced
// into one span. Lines without a `let t<idx>` binding inherit the previous
// span's phase (the line is part of the same fused subexpression).
//
// Phases produced:
//   - "fwd"   - line belongs to a forward-pass UOp
//   - "bwd"   - line belongs to a backward-pass UOp
//   - "fused" - line is structural (storage bindings, @compute header, loop
//     control, the @binding declarations, etc.) AND no prior span
//     has been seen yet.
//
// A "fused" span covers the WGSL prologue (the boilerplate before the first
// `let t<idx>`); the rest of the kernel is split by phase boundaries. If a
// kernel is fully forward or fully backward, exactly one non-prologue span
// is emitted.
//
// The arena lookup is bounded by len(arena.provenance); out-of-range indices
// (extremely unlikely for a well-formed kernel) fall back to "fused".
func fusionSpans(wgsl string, arena *uop.Arena) []FusionSpan {
	if wgsl == "" {
		return nil
	}
	lines := strings.Split(wgsl, "\n")
	if len(lines) == 0 {
		return nil
	}

	// Drop a trailing empty line introduced by a final "\n" (typical of
	// RenderWGSL output) so the end_line of the last span matches what a
	// user sees in an editor.
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	// Determine the phase of each line. -1 marks "no attribution from a let
	// binding on this line" (inherit from the previous span).
	const phaseUnknown = -1
	const phaseFwd = 0
	const phaseBwd = 1
	const phaseFused = 2

	linePhase := make([]int, len(lines))
	for i, line := range lines {
		linePhase[i] = phaseUnknown
		// Look up every `let t<idx>:` occurrence on the line. A single line
		// rarely contains more than one let, but tolerate it.
		matches := letPattern.FindAllStringSubmatch(line, -1)
		if len(matches) == 0 {
			continue
		}
		for _, m := range matches {
			idx64, err := strconv.ParseUint(m[1], 10, 32)
			if err != nil {
				continue
			}
			idx := uint32(idx64)
			if arena == nil || idx >= uint32(arena.Len()) {
				continue
			}
			switch arena.Provenance(idx) {
			case uop.PhaseForward:
				linePhase[i] = phaseFwd
			case uop.PhaseBackward:
				linePhase[i] = phaseBwd
			}
			// First match on the line wins; codegen never mixes phases
			// inside a single rendered statement.
			break
		}
	}

	// Build spans by carrying forward the most recent known phase. Lines
	// before the first known phase form a "fused" prologue.
	var spans []FusionSpan
	cur := phaseFused
	startLine := 1
	emit := func(end int) {
		if end < startLine {
			return
		}
		label := FusionLabelFused
		switch cur {
		case phaseFwd:
			label = FusionLabelForward
		case phaseBwd:
			label = FusionLabelBackward
		}
		spans = append(spans, FusionSpan{
			StartLine: startLine,
			EndLine:   end,
			Label:     label,
		})
	}
	for i := 0; i < len(lines); i++ {
		phase := linePhase[i]
		// Inherit from previous when this line has no own attribution.
		if phase == phaseUnknown {
			continue
		}
		if phase != cur {
			// Close the previous span ending on the previous line.
			emit(i) // span ends on (i-1)+1 = i (1-based end of previous line)
			startLine = i + 1
			cur = phase
		}
	}
	emit(len(lines))
	return spans
}
