package schedule

import (
	"sort"

	"github.com/georgebuilds/anneal/uop"
)

// collectBuffers does a DFS from u, stopping (not recursing) at OpBuffer nodes
// but adding each one to the result list. Encounter order is preserved; seen
// deduplicates by arena index.
func collectBuffers(u uop.UOp, seen map[uint32]bool, out *[]uop.UOp) {
	if !u.Valid() || seen[u.Index()] {
		return
	}
	seen[u.Index()] = true
	if u.Op() == uop.OpBuffer {
		*out = append(*out, u)
		return // do not recurse into BUFFER's own srcs (LUNIQUE, DEVICE)
	}
	for i := 0; i < u.NSrc(); i++ {
		collectBuffers(u.Src(i), seen, out)
	}
}

// rebuildWithParams topo-sorts inner and returns a rebuilt copy where every
// BUFFER node is replaced by the corresponding PARAM from paramMap.
// PARAM nodes are leaves and are not recursed into.
func rebuildWithParams(a *uop.Arena, inner uop.UOp, paramMap map[uint32]uop.UOp) uop.UOp {
	topo := topoSort(inner)
	rebuild := make(map[uint32]uint32, len(topo))

	for _, u := range topo {
		if u.Op() == uop.OpBuffer {
			param, ok := paramMap[u.Index()]
			if ok {
				rebuild[u.Index()] = param.Index()
			} else {
				rebuild[u.Index()] = u.Index()
			}
			continue
		}

		srcs := make([]uop.UOp, u.NSrc())
		childChanged := false
		for i := 0; i < u.NSrc(); i++ {
			ch := u.Src(i)
			if newIdx, ok := rebuild[ch.Index()]; ok {
				srcs[i] = a.At(newIdx)
				if newIdx != ch.Index() {
					childChanged = true
				}
			} else {
				srcs[i] = ch
			}
		}

		var node uop.UOp
		if childChanged {
			node = a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag())
		} else {
			node = u
		}
		rebuild[u.Index()] = node.Index()
	}

	if newIdx, ok := rebuild[inner.Index()]; ok {
		return a.At(newIdx)
	}
	return inner
}

// findAfterNodes scans arena nodes in [startIdx, a.Len()) for AFTER nodes that
// represent scheduler kernels (AFTER.Src(1) == OpEnd, signalling a full kernel
// boundary). startIdx must be the arena length before the current GetKernelGraph
// call so that AFTER nodes from prior Realize calls on the same arena are excluded.
//
// Intermediate AFTER nodes are not reachable from SINK via graph edges — they are
// referenced only through the BUFFER nodes inside downstream kernels' bodies,
// which is why an arena scan is necessary rather than a graph traversal.
func findAfterNodes(a *uop.Arena, startIdx uint32) []uop.UOp {
	var afters []uop.UOp
	for i := startIdx; i < uint32(a.Len()); i++ {
		u := a.At(i)
		if u.Op() == uop.OpAfter && u.NSrc() == 2 && u.Src(1).Op() == uop.OpEnd {
			afters = append(afters, u)
		}
	}
	// Sort by output buffer arena index for determinism.
	sort.Slice(afters, func(i, j int) bool {
		return afters[i].Src(0).Index() < afters[j].Src(0).Index()
	})
	return afters
}

// splitKernels converts each AFTER node in the kernel graph into a CALL node
// whose first src is a kernel-level SINK AST with PARAM leaves instead of BUFFERs.
// Returns the rebuilt top-level SINK and the list of CALL nodes.
//
// startIdx is the arena length before GetKernelGraph ran; findAfterNodes uses it
// to exclude AFTER nodes from any prior Realize call on the same arena.
func splitKernels(a *uop.Arena, sink uop.UOp, startIdx uint32) (uop.UOp, []uop.UOp) {
	afters := findAfterNodes(a, startIdx)
	calls := make([]uop.UOp, len(afters))

	for i, after := range afters {
		outBuf := after.Src(0) // BUFFER node — the kernel's output
		inner := after.Src(1)  // END node

		// Collect all BUFFER nodes reachable from inner (DFS, stop at BUFFER).
		var allBufs []uop.UOp
		seen := make(map[uint32]bool)
		collectBuffers(inner, seen, &allBufs)

		// Separate input buffers (exclude the output buffer).
		inputBufs := allBufs[:0:0]
		for _, b := range allBufs {
			if b.Index() != outBuf.Index() {
				inputBufs = append(inputBufs, b)
			}
		}

		// Sort input buffers by arena index for determinism.
		sort.Slice(inputBufs, func(x, y int) bool {
			return inputBufs[x].Index() < inputBufs[y].Index()
		})

		// Build paramMap: BUFFER.Index() → PARAM UOp.
		paramMap := make(map[uint32]uop.UOp, 1+len(inputBufs))
		paramMap[outBuf.Index()] = a.New(uop.OpParam, outBuf.DType(), nil, int64(0), nil)
		for j, inBuf := range inputBufs {
			paramMap[inBuf.Index()] = a.New(uop.OpParam, inBuf.DType(), nil, int64(j+1), nil)
		}

		// Rebuild the inner subtree with BUFFER → PARAM substitutions.
		rebuiltInner := rebuildWithParams(a, inner, paramMap)

		// Kernel SINK: SINK(rebuiltInner, arg=KernelInfo{NumParams: 1+len(inputBufs)})
		kernelSink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{rebuiltInner},
			uop.KernelInfo{NumParams: 1 + len(inputBufs)}, nil)

		// CALL: CALL(kernelSink, outBuf, inputBuf0, inputBuf1, ...)
		callSrcs := make([]uop.UOp, 2+len(inputBufs))
		callSrcs[0] = kernelSink
		callSrcs[1] = outBuf
		for j, inBuf := range inputBufs {
			callSrcs[2+j] = inBuf
		}
		calls[i] = a.New(uop.OpCall, uop.Dtypes.Void, callSrcs, nil, nil)
	}

	// Rebuild top-level SINK with CALL nodes replacing AFTER nodes.
	newSink := a.New(uop.OpSink, uop.Dtypes.Void, calls, nil, nil)
	return newSink, calls
}

// createSchedule performs a Kahn topological sort of CALL nodes based on
// buffer producer-consumer dependencies. Returns CALL nodes in execution order.
// Ties are broken by the arena index of the CALL's output buffer (ascending).
func createSchedule(calls []uop.UOp) []uop.UOp {
	n := len(calls)
	if n == 0 {
		return nil
	}

	// writerOf maps outputBuf.Index() → call list index.
	writerOf := make(map[uint32]int, n)
	for i, call := range calls {
		outBuf := call.Src(1)
		writerOf[outBuf.Index()] = i
	}

	// Build adjacency: edges[i] = set of call indices that depend on i.
	edges := make([]map[int]struct{}, n)
	inDegree := make([]int, n)
	for i := range edges {
		edges[i] = make(map[int]struct{})
	}

	for k, call := range calls {
		// Inputs are call.Src(2..N).
		for s := 2; s < call.NSrc(); s++ {
			buf := call.Src(s)
			if i, ok := writerOf[buf.Index()]; ok && i != k {
				if _, dup := edges[i][k]; !dup {
					edges[i][k] = struct{}{}
					inDegree[k]++
				}
			}
		}
	}

	// Kahn: start with zero-inDegree nodes sorted by output buffer arena index.
	frontier := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if inDegree[i] == 0 {
			frontier = append(frontier, i)
		}
	}
	sortFrontier := func() {
		sort.Slice(frontier, func(a, b int) bool {
			return calls[frontier[a]].Src(1).Index() < calls[frontier[b]].Src(1).Index()
		})
	}
	sortFrontier()

	ordered := make([]uop.UOp, 0, n)
	for len(frontier) > 0 {
		idx := frontier[0]
		frontier = frontier[1:]
		ordered = append(ordered, calls[idx])

		newNodes := false
		for dep := range edges[idx] {
			inDegree[dep]--
			if inDegree[dep] == 0 {
				frontier = append(frontier, dep)
				newNodes = true
			}
		}
		if newNodes {
			sortFrontier()
		}
	}

	return ordered
}

// bufSize extracts the element count from a BUFFER node's arg.
// Leaf buffers carry []int64 shape (product gives size); scheduler-allocated
// buffers carry int64 directly.
func bufSize(bufNode uop.UOp) int64 {
	switch v := bufNode.Arg().(type) {
	case int64:
		return v
	case []int64:
		size := int64(1)
		for _, s := range v {
			size *= s
		}
		return size
	case string:
		// Special buffers (e.g. "randn") — return 1 as a safe fallback.
		return 1
	default:
		return 1
	}
}

// linearToSchedule converts an ordered list of CALL nodes into ExecItems.
// Each ExecItem carries the kernel SINK AST and a Buffer descriptor for every
// PARAM (Bufs[0]=output, Bufs[1..N-1]=inputs). Slot is initialised to -1.
func linearToSchedule(ordered []uop.UOp) []ExecItem {
	items := make([]ExecItem, len(ordered))
	for i, call := range ordered {
		ki := call.Src(0).Arg().(uop.KernelInfo)
		bufs := make([]Buffer, ki.NumParams)
		for n := 0; n < ki.NumParams; n++ {
			bufNode := call.Src(1 + n)
			bufs[n] = Buffer{
				UOpIdx: bufNode.Index(),
				Size:   bufSize(bufNode),
				Shape:  bufShape(bufNode),
				DType:  bufNode.DType(),
				Slot:   -1,
			}
		}
		items[i] = ExecItem{
			Ast:  call.Src(0),
			Bufs: bufs,
		}
	}
	return items
}

// bufShape extracts the per-dimension shape from a BUFFER node's arg.
// Leaf buffers carry []int64 shape; scheduler-allocated buffers carry []int64
// after the Phase 7b fix (old int64 total-size is treated as [n]).
func bufShape(bufNode uop.UOp) []int64 {
	switch v := bufNode.Arg().(type) {
	case []int64:
		return v
	case int64:
		return []int64{v}
	case string:
		return []int64{1}
	default:
		return []int64{1}
	}
}

// memoryPlan assigns Slot values for intermediate (scheduler-allocated) buffers
// using greedy interval-graph colouring. Leaf inputs and final outputs always
// keep Slot=-1.
func memoryPlan(items []ExecItem) []ExecItem {
	// writtenAt[bufUOpIdx] = kernel index that produces this buffer.
	writtenAt := make(map[uint32]int, len(items))
	for i, item := range items {
		writtenAt[item.Bufs[0].UOpIdx] = i
	}

	// readByAny[bufUOpIdx] = true if any kernel reads this buffer as input.
	readByAny := make(map[uint32]bool)
	for _, item := range items {
		for _, buf := range item.Bufs[1:] {
			readByAny[buf.UOpIdx] = true
		}
	}

	// doNotAlias: buffers that are outputs (in writtenAt) but never read as
	// input (final outputs). They must not be aliased.
	doNotAlias := make(map[uint32]bool)
	for id := range writtenAt {
		if !readByAny[id] {
			doNotAlias[id] = true
		}
	}

	// Build lifetimes for intermediate buffers.
	type lifetime struct {
		firstWrite int
		lastRead   int
	}
	lts := make(map[uint32]*lifetime)
	for id, first := range writtenAt {
		if doNotAlias[id] {
			continue
		}
		lts[id] = &lifetime{firstWrite: first, lastRead: first}
	}

	// Extend lastRead for each intermediate buffer consumed as input.
	for i, item := range items {
		for _, buf := range item.Bufs[1:] {
			if lt, ok := lts[buf.UOpIdx]; ok {
				if i > lt.lastRead {
					lt.lastRead = i
				}
			}
		}
	}

	// Sort intermediate buffers by firstWrite, break ties by UOpIdx.
	type bufEntry struct {
		id  uint32
		lt  *lifetime
	}
	intermediates := make([]bufEntry, 0, len(lts))
	for id, lt := range lts {
		intermediates = append(intermediates, bufEntry{id, lt})
	}
	sort.Slice(intermediates, func(a, b int) bool {
		if intermediates[a].lt.firstWrite != intermediates[b].lt.firstWrite {
			return intermediates[a].lt.firstWrite < intermediates[b].lt.firstWrite
		}
		return intermediates[a].id < intermediates[b].id
	})

	// Greedy slot assignment.
	// slots[s] = lastRead of the most-recently-assigned buffer in slot s.
	slots := []int{}
	slotOf := make(map[uint32]int, len(intermediates))

	for _, be := range intermediates {
		assigned := -1
		for s, lastRead := range slots {
			if lastRead < be.lt.firstWrite {
				assigned = s
				slots[s] = be.lt.lastRead
				break
			}
		}
		if assigned == -1 {
			assigned = len(slots)
			slots = append(slots, be.lt.lastRead)
		}
		slotOf[be.id] = assigned
	}

	// Apply slot assignments to output buffers only (Bufs[0] of each item).
	// Input buffer entries keep Slot=-1 regardless; the slot identifies where
	// to allocate the output, and the runtime reads inputs from where they were written.
	for i := range items {
		if slot, ok := slotOf[items[i].Bufs[0].UOpIdx]; ok {
			items[i].Bufs[0].Slot = slot
		}
	}

	return items
}

// CreateSchedule runs all 10 passes of the rangeify pipeline on a SINK-rooted
// tensor-level graph and returns an ordered, executable schedule.
// NOTE: the schedule's correctness is not fully verified until Phase 8 codegen
// can run it end-to-end against a value oracle; this function verifies structure only.
func CreateSchedule(sink uop.UOp, device string) []ExecItem {
	a := sink.Arena()
	// Capture arena length before GetKernelGraph so:
	//   (a) findAfterNodes excludes AFTER nodes from prior Realize calls on the same arena
	//   (b) StatsHook receives the tensor-level graph size (pre-scheduler expansion)
	uopsCount := a.Len()
	startIdx := uint32(uopsCount)
	sink = GetKernelGraph(sink, device)
	_, calls := splitKernels(a, sink, startIdx)
	ordered := createSchedule(calls)
	items := linearToSchedule(ordered)
	items = memoryPlan(items)
	if h := StatsHook; h != nil {
		h(CompilerStats{
			UOps:    uopsCount,
			Kernels: len(items),
			Fused:   0,
			Pass:    "memory plan",
		})
	}
	return items
}
