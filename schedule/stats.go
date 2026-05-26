package schedule

// CompilerStats holds per-Realize compiler metrics emitted to the dashboard.
// UOps and Kernels are always live numbers from the current schedule run.
// Fused is 0 in v1 — it becomes meaningful once Pass 5 (removeBufferize) elides
// cross-boundary BUFFERIZE nodes; see CreateSchedule and removeBufferize.
type CompilerStats struct {
	UOps    int    // arena node count at SINK construction (tensor-level graph size)
	Kernels int    // number of kernel dispatches produced by the schedule
	Fused   int    // cross-boundary fused kernels (reserved; 0 until Pass 5 is live)
	Pass    string // last completed pipeline pass name
}

// StatsHook, if set, is called by CreateSchedule after each successful run.
// The hook runs synchronously in the same goroutine that called CreateSchedule.
// Callers must set this before training and clear it (nil) when done.
// Not safe for concurrent modification.
var StatsHook func(CompilerStats)
