package tensor

import (
	"fmt"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// DefaultExecutor is the backend used by Realize. Set it before calling
// Realize; typically done in main or test setup:
//
//	dev, err := webgpu.Open()
//	tensor.DefaultExecutor = dev
var DefaultExecutor backend.Executor

// Realize executes the computation graphs rooted at each tensor, materialising
// concrete float32 data. Leaf tensors must have data attached via SetData()
// before Realize is called.
//
// Realized data is stored in each tensor's Data() field. If multiple tensors
// are passed, they must be independent or form a linear chain (the order of
// output assignment follows the schedule's Kahn sort order).
func Realize(tensors ...*Tensor) error {
	if len(tensors) == 0 {
		return nil
	}

	a := tensors[0].arena()
	device := tensors[0].device

	// Build the tensor-level SINK (§7.4 role a).
	// Always added to the arena regardless of whether an executor is registered,
	// so callers that inspect the arena can observe it even on error returns.
	srcs := make([]uop.UOp, len(tensors))
	for i, t := range tensors {
		srcs[i] = t.node
	}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)

	if DefaultExecutor == nil {
		return fmt.Errorf("tensor: no backend registered — set tensor.DefaultExecutor before calling Realize")
	}

	// Run all 10 scheduler passes. outBufBySrc[i] is the output BUFFER arena
	// index for the i-th requested tensor (caller order) — the durable
	// attribution that survives the scheduler's structural-key reordering.
	items, outBufBySrc := schedule.CreateScheduleWithOutputs(sink, device)
	if len(items) == 0 {
		return nil
	}

	// Apply beam-cached optimizations. Default mode: O(1) map lookup per kernel,
	// identity on miss. Search mode (ANNEAL_BEAM=1): runs BeamSearch on misses.
	var bench backend.Benchmarker
	if b, ok := DefaultExecutor.(backend.Benchmarker); ok {
		bench = b
	}
	items = codegen.BeamApplyToItems(items, DefaultExecutor, bench)

	// Run any host-side preprocessors registered against this arena (Slice D
	// scatter-add: sort the idx tensor and populate the sortedIdx / perm
	// leaves before executor.Run reads them).
	RunScatterPreprocessors(a)

	// Collect input data for leaf buffers.
	// Leaf tensors are BUFFER nodes; their node.Index() == ExecItem.Bufs[j].UOpIdx
	// for the kernels that read them (confirmed by splitKernels: input buffers are
	// the original BUFFER nodes, not renamed).
	inputs := leafInputs(tensors)

	// Execute.
	outputs, err := DefaultExecutor.Run(items, inputs)
	if err != nil {
		return fmt.Errorf("tensor: realize: %w", err)
	}

	// Map GPU outputs back to the requested tensors by node identity: tensor[i]
	// gets the buffer the scheduler attributed to ITS sink src (outBufBySrc[i]).
	assignOutputs(tensors, outBufBySrc, outputs)
	return nil
}

// RealizeWithBinding executes the computation graphs with at least one symbolic
// dim bound to a concrete value. binding maps DefineVar name → int64 value.
// The registered DefaultExecutor must implement backend.SymbolicExecutor.
func RealizeWithBinding(binding map[string]int64, tensors ...*Tensor) error {
	if len(tensors) == 0 {
		return nil
	}
	exec, ok := DefaultExecutor.(backend.SymbolicExecutor)
	if !ok {
		return fmt.Errorf("tensor: registered executor does not implement SymbolicExecutor")
	}
	a := tensors[0].arena()
	device := tensors[0].device
	srcs := make([]uop.UOp, len(tensors))
	for i, t := range tensors {
		srcs[i] = t.node
	}
	sink := a.New(uop.OpSink, uop.Dtypes.Void, srcs, nil, nil)
	items, outBufBySrc := schedule.CreateScheduleWithOutputs(sink, device)
	if len(items) == 0 {
		return nil
	}
	var bench backend.Benchmarker
	if b, ok := DefaultExecutor.(backend.Benchmarker); ok {
		bench = b
	}
	items = codegen.BeamApplyToItems(items, DefaultExecutor, bench)
	RunScatterPreprocessors(a)
	inputs := leafInputs(tensors)
	outputs, err := exec.RunSymbolic(items, inputs, binding)
	if err != nil {
		return fmt.Errorf("tensor: realize with binding: %w", err)
	}
	assignOutputs(tensors, outBufBySrc, outputs)
	return nil
}

// leafInputs DFS-walks the UOp graph rooted at each tensor's node and collects
// data for every OpBuffer node that has had SetData called (registered in
// leafRegistry). This handles the common case where Realize(output) is called
// but the actual data is on input leaf tensors deeper in the graph.
func leafInputs(tensors []*Tensor) map[uint32][]float32 {
	inputs := make(map[uint32][]float32)
	seen := make(map[uint32]bool)
	var walk func(u uop.UOp)
	walk = func(u uop.UOp) {
		idx := u.Index()
		if seen[idx] {
			return
		}
		seen[idx] = true
		if u.Op() == uop.OpBuffer {
			if v, ok := u.Arena().Leaf(idx); ok {
				inputs[idx] = v
			}
		}
		for i := 0; i < u.NSrc(); i++ {
			walk(u.Src(i))
		}
	}
	for _, t := range tensors {
		walk(t.node)
	}
	return inputs
}

// assignOutputs maps realized output-buffer data back into tensor.data fields by
// NODE IDENTITY. outBufBySrc[i] is the output BUFFER arena index attributed to
// the i-th requested tensor (caller order, parallel to `tensors`), as captured
// by schedule.CreateScheduleWithOutputs before the scheduler's structural-key
// reordering. tensor[i].data = outputs[outBufBySrc[i]].
//
// This is the durable fix for the multi-output scramble: genuinely isomorphic
// kernels (same shape AND structure) cannot be told apart by shape or structure,
// so the only reliable disambiguator is the original tensor node threaded through
// the schedule — which outBufBySrc carries.
//
// Behaviour by case:
//   - leaf tensors keep their caller-provided data (outBufBySrc[i] == 0 sentinel);
//   - duplicate tensors passed twice both resolve to the same buffer index;
//   - a buffer absent from the outputs map (e.g. consumed only as a later
//     kernel's input — should not happen for a requested output) leaves the
//     tensor's data unchanged.
func assignOutputs(tensors []*Tensor, outBufBySrc []uint32, outputs map[uint32][]float32) {
	for i, t := range tensors {
		if i >= len(outBufBySrc) {
			break
		}
		if t.IsLeaf() {
			continue // leaf data was provided by the caller, not produced by a kernel
		}
		bufIdx := outBufBySrc[i]
		if bufIdx == 0 {
			continue // no kernel output attributed to this src
		}
		if data, ok := outputs[bufIdx]; ok {
			t.data = data
		}
	}
}
