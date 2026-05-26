// Package schedule implements the ten-pass rangeify pipeline: realize-map,
// bufferize, kernel split, toposort, and memory plan.
//
// This file implements passes 1–6 (GetKernelGraph): from a SINK-rooted tensor
// graph to a kernel-segmented graph with explicit BUFFER+STORE+AFTER boundaries.
// Passes 7–10 (split_kernels, create_schedule, linear_to_schedule,
// memory_planner) are Phase 7b and are not yet implemented.
package schedule

import (
	"sort"

	"github.com/georgebuilds/anneal/uop"
)

// GetKernelGraph runs passes 1–6 of the rangeify scheduler.
//
// Input:  SINK-rooted tensor-level UOp graph as produced by tensor.Realize.
// Output: Kernel-segmented graph where each realized buffer is represented by
//         BUFFER(LUNIQUE, DEVICE).after(END(STORE(INDEX(buf,*out), body), *out, *red)).
//         The body contains INDEX(BUFFER, *arithmetic_indices) at every leaf
//         access with all movement ops dissolved into range arithmetic.
//
// device is the target device string (e.g. "cpu", "webgpu").
func GetKernelGraph(sink uop.UOp, device string) uop.UOp {
	// Pass 1: early rewrites (no-op in v1 — identity ops already eliminated
	// at tensor-construction time).
	sink = earlyRewrites(sink)

	// Passes 2–4: compute realize map, thread range indices through each kernel
	// subgraph (dissolving movement ops into range arithmetic), insert BUFFERIZE.
	sink = runRangeify(sink)

	// Pass 5: remove cheap BUFFERIZE (cost-based fusion). All BUFFERIZE nodes
	// are Removable=false in v1, so this is a no-op placeholder.
	sink = removeBufferize(sink)

	// Pass 6: rewrite surviving BUFFERIZE → BUFFER(LUNIQUE,DEVICE)+STORE+END+AFTER.
	sink = addBuffers(sink.Arena(), sink, device)

	return sink
}

// ── Pass 1: early rewrites ────────────────────────────────────────────────────

func earlyRewrites(sink uop.UOp) uop.UOp { return sink }

// ── Pass 5: remove cheap BUFFERIZE ───────────────────────────────────────────

// removeBufferize elides Removable=true BUFFERIZE nodes when the cost function
// allows fusion. v1 only creates Removable=false nodes, so this is a no-op.
func removeBufferize(sink uop.UOp) uop.UOp { return sink }

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
	topo := topoSort(sink)
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
		var outRanges, redRanges []uop.UOp
		for _, r := range ranges {
			if r.Op() == uop.OpRange {
				if r.Arg().(uop.RangeArg).Type == uop.AxisReduce {
					redRanges = append(redRanges, r)
				} else {
					outRanges = append(outRanges, r)
				}
			}
			// Const(0) ranges (size-1 dims) are not looped over.
		}

		// Output buffer shape = per-dim sizes of the AxisLoop ranges.
		// Stored as []int64 so paramShape in the codegen can reconstruct strides
		// for multi-dimensional intermediate buffers (e.g. a [2,2] reduce output
		// read as a 2D input by a downstream elementwise kernel).
		outShape := make([]int64, len(outRanges))
		for ri, r := range outRanges {
			outShape[ri] = r.Arg().(uop.RangeArg).Size
		}

		// BUFFER(LUNIQUE, DEVICE, arg=outShape)
		lunique := a.New(uop.OpLUnique, uop.Dtypes.Void, nil, counterMap[u.Index()], nil)
		deviceNode := a.New(uop.OpDevice, uop.Dtypes.Void, nil, device, nil)
		buf := a.New(uop.OpBuffer, u.DType(), []uop.UOp{lunique, deviceNode}, outShape, nil)
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

	if newIdx, ok := rebuild[sink.Index()]; ok {
		return a.At(newIdx)
	}
	return sink
}
