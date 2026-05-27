package tensor

import (
	"fmt"

	"github.com/georgebuilds/anneal/backend"
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

	// Run all 10 scheduler passes.
	items := schedule.CreateSchedule(sink, device)
	if len(items) == 0 {
		return nil
	}

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

	// Map GPU outputs back to the requested tensors.
	// Final output buffers (Slot=-1 and not read as input by any later kernel)
	// appear in the `outputs` map. They are matched to tensors in Kahn order
	// (ascending output buffer arena index), which is the order `items` was
	// constructed — for independent tensors, tensor[i]→items[i]→outputs[i].
	assignOutputs(tensors, items, outputs)
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
	items := schedule.CreateSchedule(sink, device)
	if len(items) == 0 {
		return nil
	}
	inputs := leafInputs(tensors)
	outputs, err := exec.RunSymbolic(items, inputs, binding)
	if err != nil {
		return fmt.Errorf("tensor: realize with binding: %w", err)
	}
	assignOutputs(tensors, items, outputs)
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

// assignOutputs maps final output buffer data back into tensor.data fields.
// The match is by Kahn-sorted schedule order (same ordering used by createSchedule
// when it broke ties by output buffer arena index). For a single-output call this
// is exact. For multiple independent outputs the i-th tensor gets the i-th final
// output in sorted-arena-index order.
func assignOutputs(tensors []*Tensor, items []schedule.ExecItem, outputs map[uint32][]float32) {
	// Collect final output buffer UOpIdxes in schedule order.
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}
	var finalOutIdxes []uint32
	for _, item := range items {
		uopIdx := item.Bufs[0].UOpIdx
		if !readByAny[uopIdx] {
			finalOutIdxes = append(finalOutIdxes, uopIdx)
		}
	}

	// For requested tensors that are not leaves (they need GPU output), assign
	// in parallel with finalOutIdxes.
	outSlot := 0
	for _, t := range tensors {
		if t.IsLeaf() {
			continue // leaf data was provided by the caller, not produced by a kernel
		}
		if outSlot >= len(finalOutIdxes) {
			break
		}
		t.data = outputs[finalOutIdxes[outSlot]]
		outSlot++
	}
}
