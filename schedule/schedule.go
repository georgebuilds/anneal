// Package schedule implements the ten-pass rangeify pipeline: realize-map,
// bufferize, kernel split, toposort, and memory plan.
//
// schedule.go implements passes 1-6 (GetKernelGraph): from a SINK-rooted tensor
// graph to a kernel-segmented graph with explicit BUFFER+STORE+AFTER boundaries.
// kernels.go implements passes 7-10 (split_kernels, create_schedule,
// linear_to_schedule, memory_planner) - see CreateSchedule.
package schedule

import (
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/georgebuilds/anneal/uop"
)

// DebugBufRanges captures range info for BUFFERIZE nodes with 2 outRanges and a symbolic redRange.
// Key = BUFFERIZE arena index. Call DebugBufRangesFlush to write to a file.
var DebugBufRanges []string

// DebugBufRangesEnabled enables range capture in addBuffers.
var DebugBufRangesEnabled bool

// DebugBufRangesFlush writes captured info to the given path and clears the buffer.
func DebugBufRangesFlush(path string) {
	var out string
	for _, s := range DebugBufRanges {
		out += s + "\n"
	}
	if err := os.WriteFile(path, []byte(out), 0644); err != nil {
		log.Printf("schedule: debug write failed: %v", err)
	}
	DebugBufRanges = nil
}

// GetKernelGraph runs passes 1-6 of the rangeify scheduler.
//
// Input:  SINK-rooted tensor-level UOp graph as produced by tensor.Realize.
// Output: Kernel-segmented graph where each realized buffer is represented by
//
//	BUFFER(LUNIQUE, DEVICE).after(END(STORE(INDEX(buf,*out), body), *out, *red)).
//	The body contains INDEX(BUFFER, *arithmetic_indices) at every leaf
//	access with all movement ops dissolved into range arithmetic.
//
// device is the target device string (e.g. "cpu", "webgpu").
func GetKernelGraph(sink uop.UOp, device string) uop.UOp {
	// Pass 1: early rewrites (no-op in v1 - identity ops already eliminated
	// at tensor-construction time).
	sink = earlyRewrites(sink)

	// Pass 1b: auto-Contiguous budget pass. Inserts OpContiguous nodes when
	// a future kernel boundary would exceed MaxBuffersPerKernel storage-buffer
	// reads (WebGPU/Metal cap is 8 bindings per kernel). Operates pre-rangeify
	// so the inserted nodes are picked up naturally by hardRealizeOps in
	// buildRealizeMap. Generalises the manual Contiguous() calls historically
	// sprinkled through tensor/nn/ to dodge the same limit.
	sink = enforceBufferBudget(sink)

	// Passes 2-4: compute realize map, thread range indices through each kernel
	// subgraph (dissolving movement ops into range arithmetic), insert BUFFERIZE.
	sink = runRangeify(sink)

	// Pass 5: remove cheap BUFFERIZE (cost-based fusion). Folds a Removable
	// matmul-reduce producer into its sole elementwise-epilogue consumer,
	// subject to the WebGPU 8-buffer-per-kernel budget. Gated by
	// FusionEnabled for differential testing.
	sink = removeBufferize(sink)

	// Pass 6: rewrite surviving BUFFERIZE → BUFFER(LUNIQUE,DEVICE)+STORE+END+AFTER.
	sink = addBuffers(sink.Arena(), sink, device)

	return sink
}

// ── Pass 1: early rewrites ────────────────────────────────────────────────────

func earlyRewrites(sink uop.UOp) uop.UOp { return sink }

// ── Pass 5: remove cheap BUFFERIZE ───────────────────────────────────────────

// kernelTopoSort returns all nodes reachable from root within the same kernel
// (stopping at OpBuffer and OpBufferize boundaries).
func kernelTopoSort(root uop.UOp) []uop.UOp {
	seen := make(map[uint32]bool)
	var order []uop.UOp

	type frame struct {
		u       uop.UOp
		nextSrc int
	}
	stack := []frame{{root, 0}}

	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u
		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}

		// Boundaries: stop traversal.
		if u.Op() == uop.OpBuffer || u.Op() == uop.OpBufferize {
			seen[u.Index()] = true
			order = append(order, u)
			stack = stack[:len(stack)-1]
			continue
		}

		pushed := false
		for f.nextSrc < u.NSrc() {
			child := u.Src(f.nextSrc)
			f.nextSrc++
			if !seen[child.Index()] {
				stack = append(stack, frame{child, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			order = append(order, u)
			stack = stack[:len(stack)-1]
		}
	}
	return order
}

// removeBufferize elides Removable=true BUFFERIZE nodes by folding each into
// its sole consumer when that consumer is a non-reduce BUFFERIZE and the
// combined kernel stays within the 8-buffer WebGPU budget. Gated by
// FusionEnabled for differential testing.
var FusionEnabled = true

func removeBufferize(sink uop.UOp) uop.UOp {
	if !FusionEnabled {
		return sink
	}
	a := sink.Arena()
	topo := uop.TopoSort(sink)

	// 1. Analyze all BUFFERIZE nodes.
	bufzNodes := make([]uop.UOp, 0)
	for _, u := range topo {
		if u.Op() == uop.OpBufferize {
			bufzNodes = append(bufzNodes, u)
		}
	}

	bufzReads := make(map[uint32]map[uint32]bool)
	hasReduce := make(map[uint32]bool)
	readsInReduce := make(map[uint32]map[uint32]bool)

	for _, bfz := range bufzNodes {
		body := bfz.Src(0)
		bodyTopo := kernelTopoSort(body)
		reads := make(map[uint32]bool)

		var reduceNodes []uop.UOp
		for _, u := range bodyTopo {
			if u.Op() == uop.OpReduce {
				hasReduce[bfz.Index()] = true
				reduceNodes = append(reduceNodes, u)
			}
			if u.Op() == uop.OpBuffer || u.Op() == uop.OpBufferize {
				reads[u.Index()] = true
			}
		}
		bufzReads[bfz.Index()] = reads

		for _, r := range reduceNodes {
			elemTopo := kernelTopoSort(r.Src(0))
			for _, u := range elemTopo {
				if u.Op() == uop.OpBufferize {
					if readsInReduce[bfz.Index()] == nil {
						readsInReduce[bfz.Index()] = make(map[uint32]bool)
					}
					readsInReduce[bfz.Index()][u.Index()] = true
				}
			}
		}
	}

	// Build a map of BUFFERIZE consumers.
	// bufzConsumers[bfz_A] = list of indices (BUFFERIZE or SINK) that read bfz_A.
	bufzConsumers := make(map[uint32][]uint32)
	for bfzIdx, reads := range bufzReads {
		for prodIdx := range reads {
			if a.At(prodIdx).Op() == uop.OpBufferize {
				bufzConsumers[prodIdx] = append(bufzConsumers[prodIdx], bfzIdx)
			}
		}
	}
	for i := 0; i < sink.NSrc(); i++ {
		prodIdx := sink.Src(i).Index()
		bufzConsumers[prodIdx] = append(bufzConsumers[prodIdx], sink.Index())
	}

	// 2. Decide fusions.
	fusedInto := make(map[uint32]uint32)
	currentBufs := make(map[uint32]map[uint32]bool)
	for k, v := range bufzReads {
		m := make(map[uint32]bool)
		for b := range v {
			m[b] = true
		}
		currentBufs[k] = m
	}

	for _, bfz := range bufzNodes {
		arg := bfz.Arg().(uop.BufferizeArg)
		if !arg.Removable {
			continue
		}
		consList := bufzConsumers[bfz.Index()]
		if len(consList) != 1 {
			continue
		}
		bfzBIdx := consList[0]
		if a.At(bfzBIdx).Op() != uop.OpBufferize {
			continue
		}
		if hasReduce[bfzBIdx] {
			continue
		}
		if readsInReduce[bfzBIdx][bfz.Index()] {
			continue
		}

		// Budget check (WebGPU limit 8).
		combined := make(map[uint32]bool)
		for b := range currentBufs[bfzBIdx] {
			combined[b] = true
		}
		delete(combined, bfz.Index())
		for b := range currentBufs[bfz.Index()] {
			combined[b] = true
		}
		// Combined inputs + 1 output must be <= 8.
		if len(combined)+1 > 8 {
			continue
		}

		fusedInto[bfz.Index()] = bfzBIdx
		currentBufs[bfzBIdx] = combined
	}

	if len(fusedInto) == 0 {
		return sink
	}

	// 3. Rebuild graph with fusions.
	fr := &fusedRebuilder{
		a:               a,
		fusedInto:       fusedInto,
		memoBody:        make(map[uint32]uop.UOp),
		memoRanges:      make(map[uint32][]uop.UOp),
		externalRebuild: make(map[uint32]uint32),
	}

	rebuild := fr.externalRebuild
	for _, u := range topo {
		if _, ok := fusedInto[u.Index()]; ok {
			continue
		}

		if u.Op() == uop.OpBufferize {
			newBody := fr.getFusedBody(u.Index())
			newRanges := fr.getFusedRanges(u.Index())

			bfzSrcs := make([]uop.UOp, 1+len(newRanges))
			bfzSrcs[0] = newBody
			copy(bfzSrcs[1:], newRanges)
			rebuild[u.Index()] = a.New(uop.OpBufferize, u.DType(), bfzSrcs, u.Arg(), u.Tag()).Index()
			continue
		}

		// Standard rebuild.
		srcs := make([]uop.UOp, u.NSrc())
		changed := false
		for i := 0; i < u.NSrc(); i++ {
			oldSrc := u.Src(i)
			if newIdx, ok := rebuild[oldSrc.Index()]; ok {
				srcs[i] = a.At(newIdx)
				if newIdx != oldSrc.Index() {
					changed = true
				}
			} else {
				srcs[i] = oldSrc
			}
		}
		if changed {
			rebuild[u.Index()] = a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag()).Index()
		} else {
			rebuild[u.Index()] = u.Index()
		}
	}

	return a.At(rebuild[sink.Index()])
}

type fusedRebuilder struct {
	a               *uop.Arena
	fusedInto       map[uint32]uint32
	memoBody        map[uint32]uop.UOp
	memoRanges      map[uint32][]uop.UOp
	externalRebuild map[uint32]uint32
}

func (fr *fusedRebuilder) getFusedBody(bfzIdx uint32) uop.UOp {
	if b, ok := fr.memoBody[bfzIdx]; ok {
		return b
	}
	bfz := fr.a.At(bfzIdx)
	body := bfz.Src(0)

	topo := kernelTopoSort(body)
	localRebuild := make(map[uint32]uint32)

	for _, u := range topo {
		if u.Op() == uop.OpIndex {
			prodIdx := u.Src(0).Index()
			if _, ok := fr.fusedInto[prodIdx]; ok {
				// Producer is fused into this consumer.
				prod := fr.a.At(prodIdx)

				// Recursively get the producer's already-fused body.
				prodBody := fr.getFusedBody(prodIdx)
				// And its original ranges (for substitution mapping).
				// Note: AxisLoop ranges are ALWAYS first in BUFFERIZE.Src(1:).
				numProdRanges := prod.NSrc() - 1
				prodRanges := make([]uop.UOp, numProdRanges)
				for i := 0; i < numProdRanges; i++ {
					prodRanges[i] = prod.Src(i + 1)
				}

				// Substitute AxisLoop ranges.
				// Now that runRangeify includes all dimensions (including OpConst(0))
				// in the BUFFERIZE range list, positional mapping is robust.
				subs := make(map[uint32]uop.UOp)
				numLoopRanges := u.NSrc() - 1
				for i := 0; i < numLoopRanges; i++ {
					prodRange := prodRanges[i]
					if prodRange.Op() == uop.OpRange {
						idx := u.Src(i + 1).Index()
						if remapped, ok := localRebuild[idx]; ok {
							subs[prodRange.Index()] = fr.a.At(remapped)
						} else if remapped, ok := fr.externalRebuild[idx]; ok {
							subs[prodRange.Index()] = fr.a.At(remapped)
						} else {
							subs[prodRange.Index()] = u.Src(i + 1)
						}
					}
				}

				// Perform substitution.
				fused := fr.substituteRanges(prodBody, subs)
				localRebuild[u.Index()] = fused.Index()
				continue
			}
		}

		// Standard rebuild within body.
		srcs := make([]uop.UOp, u.NSrc())
		changed := false
		for i := 0; i < u.NSrc(); i++ {
			oldSrc := u.Src(i)
			if newIdx, ok := localRebuild[oldSrc.Index()]; ok {
				srcs[i] = fr.a.At(newIdx)
				if newIdx != oldSrc.Index() {
					changed = true
				}
			} else if newIdx, ok := fr.externalRebuild[oldSrc.Index()]; ok {
				srcs[i] = fr.a.At(newIdx)
				if newIdx != oldSrc.Index() {
					changed = true
				}
			} else {
				srcs[i] = oldSrc
			}
		}
		if changed {
			localRebuild[u.Index()] = fr.a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag()).Index()
		} else {
			localRebuild[u.Index()] = u.Index()
		}
	}
	res := fr.a.At(localRebuild[body.Index()])
	fr.memoBody[bfzIdx] = res
	return res
}

func (fr *fusedRebuilder) getFusedRanges(bfzIdx uint32) []uop.UOp {
	if r, ok := fr.memoRanges[bfzIdx]; ok {
		return r
	}
	bfz := fr.a.At(bfzIdx)

	rangeMap := make(map[uint32]uop.UOp)
	for i := 1; i < bfz.NSrc(); i++ {
		rangeMap[bfz.Src(i).Index()] = bfz.Src(i)
	}

	// Find all producers fused into this bfz (directly or indirectly).
	for prodIdx, targetIdx := range fr.fusedInto {
		if targetIdx == bfzIdx {
			// Producer fused into this target.
			// Add its AxisReduce ranges (recursively, if producer itself has fusions).
			prodRanges := fr.getFusedRanges(prodIdx)
			for _, r := range prodRanges {
				if r.Op() == uop.OpRange && r.Arg().(uop.RangeArg).Type == uop.AxisReduce {
					rangeMap[r.Index()] = r
				}
			}
		}
	}

	newRanges := make([]uop.UOp, 0, len(rangeMap))
	rangeIndices := make([]uint32, 0, len(rangeMap))
	for idx := range rangeMap {
		rangeIndices = append(rangeIndices, idx)
	}
	sort.Slice(rangeIndices, func(i, j int) bool { return rangeIndices[i] < rangeIndices[j] })
	for _, idx := range rangeIndices {
		newRanges = append(newRanges, rangeMap[idx])
	}
	fr.memoRanges[bfzIdx] = newRanges
	return newRanges
}

func (fr *fusedRebuilder) substituteRanges(root uop.UOp, subs map[uint32]uop.UOp) uop.UOp {
	topo := kernelTopoSort(root)
	localRebuild := make(map[uint32]uint32)

	for _, u := range topo {
		if s, ok := subs[u.Index()]; ok {
			localRebuild[u.Index()] = s.Index()
			continue
		}

		srcs := make([]uop.UOp, u.NSrc())
		changed := false
		for i := 0; i < u.NSrc(); i++ {
			oldSrc := u.Src(i)
			if newIdx, ok := localRebuild[oldSrc.Index()]; ok {
				srcs[i] = fr.a.At(newIdx)
				if newIdx != oldSrc.Index() {
					changed = true
				}
			} else if newIdx, ok := fr.externalRebuild[oldSrc.Index()]; ok {
				srcs[i] = fr.a.At(newIdx)
				if newIdx != oldSrc.Index() {
					changed = true
				}
			} else {
				srcs[i] = oldSrc
			}
		}
		if changed {
			localRebuild[u.Index()] = fr.a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag()).Index()
		} else {
			localRebuild[u.Index()] = u.Index()
		}
	}
	return fr.a.At(localRebuild[root.Index()])
}

// ── Pass 6: add_buffers ───────────────────────────────────────────────────────

// addBuffers converts each surviving OpBufferize node into a materialised
// buffer representation:
//
//	BUFFERIZE(indexed_body, r_out0…, r_red0…, arg=BufferizeArg) →
//	    BUFFER(LUNIQUE(id), DEVICE(dev))
//	        .after(END(STORE(INDEX(buf, r_out0…), indexed_body), r_out0…, r_red0…))
//
// The STORE destination is INDEX(buf, *outRanges) following tinygrad's
// representation where the write address is an indexed pointer.  The body
// (src[0] of BUFFERIZE) is the result of runRangeify: INDEX nodes at every
// leaf buffer access, REDUCE nodes for accumulations.
//
// Any INDEX(BUFFERIZE, *indices) in the body is rewritten to INDEX(BUFFER, *indices)
// so that downstream reads target the allocated buffer, not the wrapper.
func addBuffers(a *uop.Arena, sink uop.UOp, device string) uop.UOp {
	topo := uop.TopoSort(sink)
	rebuild := make(map[uint32]uint32, len(topo))
	// bufForBufz maps a BUFFERIZE node's arena index → the BUFFER node's arena
	// index.  INDEX(BUFFERIZE, *i) nodes use this to become INDEX(BUFFER, *i).
	bufForBufz := make(map[uint32]uint32)

	// Assign LUNIQUE counters by structural-key order of BUFFERIZE nodes so that
	// the counter (and therefore BUFFER identity) is a pure function of graph
	// structure, independent of arena construction order.
	keys := uop.StructuralKeys(a)
	var bufzList []uop.UOp
	for _, u := range topo {
		if u.Op() == uop.OpBufferize {
			bufzList = append(bufzList, u)
		}
	}
	sort.SliceStable(bufzList, func(i, j int) bool {
		return keys[bufzList[i].Index()] < keys[bufzList[j].Index()]
	})
	counterMap := make(map[uint32]int64, len(bufzList))
	for i, u := range bufzList {
		counterMap[u.Index()] = int64(i)
	}

	for _, u := range topo {
		// Build remapped srcs, with special handling for INDEX nodes:
		// the indexed buffer (src[0]) must point to BUFFER, not AFTER.
		srcs := make([]uop.UOp, u.NSrc())
		childChanged := false
		for i := 0; i < u.NSrc(); i++ {
			ch := u.Src(i)
			var newIdx uint32
			found := false

			if u.Op() == uop.OpIndex && i == 0 {
				// Buffer being accessed: use the underlying BUFFER (not AFTER).
				if bIdx, ok := bufForBufz[ch.Index()]; ok {
					newIdx = bIdx
					found = true
				}
			}
			if !found {
				if nIdx, ok := rebuild[ch.Index()]; ok {
					newIdx = nIdx
					found = true
				}
			}
			if found {
				srcs[i] = a.At(newIdx)
				if newIdx != ch.Index() {
					childChanged = true
				}
			} else {
				srcs[i] = ch
			}
		}

		if u.Op() != uop.OpBufferize {
			var node uop.UOp
			if childChanged {
				node = a.New(u.Op(), u.DType(), srcs, u.Arg(), u.Tag())
			} else {
				node = u
			}
			rebuild[u.Index()] = node.Index()
			continue
		}

		// BUFFERIZE: src[0]=indexed_body, src[1..]=range nodes (AxisLoop + AxisReduce)
		body := srcs[0]
		ranges := srcs[1:]

		// Separate output (AxisLoop) from reduce (AxisReduce) range vars.
		// runRangeify now provides Loop ranges at the start, one per dimension.
		var outRanges, redRanges []uop.UOp
		for _, r := range ranges {
			if r.Op() == uop.OpRange {
				if r.Arg().(uop.RangeArg).Type == uop.AxisReduce {
					redRanges = append(redRanges, r)
				} else {
					outRanges = append(outRanges, r)
				}
			} else if r.Op() == uop.OpConst {
				// Size-1 loop dimension.
				outRanges = append(outRanges, r)
			}
		}

		// Output buffer shape = per-dim sizes of the AxisLoop ranges.
		// Symbolic ranges contribute 0 - the actual size is resolved at dispatch
		// time from the binding (see backend/webgpu/executor.go:symElemCount).
		outShape := make([]int64, len(outRanges))
		for ri, r := range outRanges {
			switch {
			case r.Op() != uop.OpRange:
				outShape[ri] = 1
			case uop.RangeIsSymbolic(r):
				outShape[ri] = 0
			default:
				outShape[ri] = uop.RangeSize(r)
			}
		}
		// Slice 3 derived bounds: if any symbolic range carries an ALU bound
		// (e.g. Mul(n,4) from a reshape-merge), promote outShape to a
		// ShapeSintArg so the buffer carries the bound expression. bufSymDimMul
		// downstream resolves it to (multiplier, var-name) for runtime sizing.
		// Bare-DefineVar bounds stay on the []int64 path so Slice 1/2 buffers
		// remain byte-identical.
		hasDerivedBound := false
		for _, r := range outRanges {
			if r.Op() == uop.OpRange && uop.RangeIsSymbolic(r) {
				if r.Src(0).Op() != uop.OpDefineVar {
					hasDerivedBound = true
					break
				}
			}
		}
		var bufArg any = outShape
		if hasDerivedBound {
			// Decide between ShapeSintArg (single-term encoding from Slice 4)
			// and the richer BoundExprArg (Option B Slice 5 - affine sums for
			// the pad/shrink output case where bounds are Add(var, const) or
			// Add(var, var)). Try the narrow encoding first; if any symbolic
			// range carries a bound shape SymBoundFactor can't decompose
			// (i.e. needs the affine sum), fall through to BoundExprArg.
			canNarrow := true
			for _, r := range outRanges {
				if r.Op() != uop.OpRange || !uop.RangeIsSymbolic(r) {
					continue
				}
				b := r.Src(0)
				if b.Op() == uop.OpDefineVar {
					continue
				}
				if b.Op() == uop.OpMul && b.NSrc() == 2 {
					a0, a1 := b.Src(0), b.Src(1)
					if (a0.Op() == uop.OpDefineVar && a1.Op() == uop.OpConst) ||
						(a0.Op() == uop.OpConst && a1.Op() == uop.OpDefineVar) {
						continue
					}
				}
				canNarrow = false
				break
			}
			if canNarrow {
				arg := make(uop.ShapeSintArg, len(outRanges))
				for ri, r := range outRanges {
					switch {
					case r.Op() != uop.OpRange:
						arg[ri] = uop.ShapeDim{V: 1}
					case uop.RangeIsSymbolic(r):
						name, mul := uop.SymBoundFactor(r.Src(0))
						arg[ri] = uop.ShapeDim{Sym: true, VarName: name, Mul: mul}
					default:
						arg[ri] = uop.ShapeDim{V: uop.RangeSize(r)}
					}
				}
				bufArg = arg
			} else {
				arg := make(uop.BoundExprArg, len(outRanges))
				for ri, r := range outRanges {
					switch {
					case r.Op() != uop.OpRange:
						arg[ri] = uop.BoundDim{V: 1}
					case uop.RangeIsSymbolic(r):
						terms, off, ok := uop.BoundToAffine(r.Src(0))
						if !ok {
							panic(fmt.Sprintf("schedule: BoundExprArg: bound expression too complex to decompose into an affine sum (op=%s)", r.Src(0).Op()))
						}
						arg[ri] = uop.BoundDim{Sym: true, Terms: terms, Offset: off}
					default:
						arg[ri] = uop.BoundDim{V: uop.RangeSize(r)}
					}
				}
				bufArg = arg
			}
		}

		// Debug: log kernels with 2 outRanges and a symbolic redRange.
		if DebugBufRangesEnabled && len(outRanges) == 2 && len(redRanges) == 1 {
			if redRanges[0].Op() == uop.OpRange && uop.RangeIsSymbolic(redRanges[0]) {
				var ra0desc, ra1desc string
				if outRanges[0].Op() == uop.OpRange {
					ra0 := outRanges[0].Arg().(uop.RangeArg)
					ra0desc = fmt.Sprintf("id=%d,sym=%v,arenaIdx=%d", ra0.ID, uop.RangeIsSymbolic(outRanges[0]), outRanges[0].Index())
				} else {
					ra0desc = fmt.Sprintf("OpConst,arenaIdx=%d", outRanges[0].Index())
				}
				if outRanges[1].Op() == uop.OpRange {
					ra1 := outRanges[1].Arg().(uop.RangeArg)
					ra1desc = fmt.Sprintf("id=%d,sym=%v,arenaIdx=%d", ra1.ID, uop.RangeIsSymbolic(outRanges[1]), outRanges[1].Index())
				} else {
					ra1desc = fmt.Sprintf("OpConst,arenaIdx=%d", outRanges[1].Index())
				}
				DebugBufRanges = append(DebugBufRanges, fmt.Sprintf(
					"addBuffers BUFFERIZE idx=%d: outRanges=[%s  %s] outShape=%v",
					u.Index(), ra0desc, ra1desc, outShape))
			}
		}

		// BUFFER(LUNIQUE, DEVICE, arg=outShape or ShapeSintArg if derived bounds)
		lunique := a.New(uop.OpLUnique, uop.Dtypes.Void, nil, counterMap[u.Index()], nil)
		deviceNode := a.New(uop.OpDevice, uop.Dtypes.Void, nil, device, nil)
		buf := a.New(uop.OpBuffer, u.DType(), []uop.UOp{lunique, deviceNode}, bufArg, nil)
		bufForBufz[u.Index()] = buf.Index()

		// Store destination: INDEX(buf, r_out0, r_out1, …)
		dstSrcs := make([]uop.UOp, 1+len(outRanges))
		dstSrcs[0] = buf
		copy(dstSrcs[1:], outRanges)
		storeDst := a.New(uop.OpIndex, uop.Dtypes.Void, dstSrcs, nil, nil)

		// STORE(INDEX(buf, *outRanges), body)
		store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{storeDst, body}, nil, nil)

		// END(store, *outRanges, *redRanges)
		endSrcs := make([]uop.UOp, 1+len(outRanges)+len(redRanges))
		endSrcs[0] = store
		copy(endSrcs[1:], outRanges)
		copy(endSrcs[1+len(outRanges):], redRanges)
		end := a.New(uop.OpEnd, uop.Dtypes.Void, endSrcs, nil, nil)

		// AFTER(buf, end)
		after := a.New(uop.OpAfter, buf.DType(), []uop.UOp{buf, end}, nil, nil)
		rebuild[u.Index()] = after.Index()
	}

	if nIdx, ok := rebuild[sink.Index()]; ok {
		return a.At(nIdx)
	}
	return sink
}
