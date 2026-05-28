package tensor

import (
	"github.com/georgebuilds/anneal/uop"
)

// GradTraceEvent records that a single gradient rule fired during backward.
// One event per call to Gradient.Dispatch in the reverse-topo driver loop.
type GradTraceEvent struct {
	Seq            int      // 0-indexed firing order (reverse-topo)
	ForwardNodeIdx uint32   // arena index of the forward UOp whose rule fired
	ForwardOp      uop.Op   // op of the forward node (= rule's key in Gradient)
	AdjointIdx     uint32   // arena index of the incoming adjoint (the `adj` tensor's node)
	ProducedIdx    []uint32 // arena indices of the adjoint UOps the rule produced
	// (one entry per non-nil src contribution)
}

// GradTrace is the captured sequence of rule firings from one BackwardWithTrace call.
// Events are in firing order (reverse topological order over the forward graph).
type GradTrace struct {
	Events []GradTraceEvent
}

// TraceSentinel is the value used in ProducedIdx to indicate a nil contribution
// (e.g. for non-differentiable sources or rules that skip a source).
const TraceSentinel uint32 = 0xFFFFFFFF
