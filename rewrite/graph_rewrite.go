package rewrite

import "github.com/georgebuilds/anneal/uop"

// Option configures a GraphRewrite call.
type Option func(*rewriteOpts)

type rewriteOpts struct {
	bpm Matcher
	ctx any
}

// WithBPM adds a pre-order Matcher that is applied to each node in a per-node
// fixpoint loop before the node's children are visited. This enables rewrites that must
// complete before the post-order (main) PM sees the rebuilt subtree.
func WithBPM(bpm Matcher) Option {
	return func(o *rewriteOpts) { o.bpm = bpm }
}

// WithCtx passes a context value through to all match handlers.
func WithCtx(ctx any) Option {
	return func(o *rewriteOpts) { o.ctx = ctx }
}

// GraphRewrite walks the DAG rooted at sink and rewrites each node using pm (post-order,
// applied once per node after its children are resolved). Returns the rewritten root.
//
// The driver is an iterative slice-based state machine rather than a tree-recursive
// walk, so it handles arbitrarily deep graphs without stack overflow
//
// Rewrites never mutate nodes; replacements are new interned nodes in the same arena.
// Unchanged subtrees are automatically shared because interning maps identical structures
// to the same index.
//
// Optional: WithBPM(bpm) runs bpm in a per-node fixpoint before descending. This
// supports rules that need to fire before children are processed. WithCtx(ctx) forwards
// a value to every match handler.
func GraphRewrite(sink uop.UOp, pm Matcher, opts ...Option) uop.UOp {
	o := &rewriteOpts{}
	for _, opt := range opts {
		opt(o)
	}
	return runRewrite(sink, pm, o.bpm, o.ctx)
}

// ── iterative state machine ───────────────────────────────────────────────────

// frame is one entry on the explicit DFS stack.
// stage 0: apply bpm fixpoint, push children.
// stage 1: all children resolved — rebuild srcs, apply pm.
// stage 2: pm fired — wait for the replacement node to be resolved.
type frame struct {
	node    uop.UOp
	stage   int
	newNode uop.UOp // set in stage 0; the bpm-fixed node (may equal node)
	replN   uop.UOp // set in stage 1 when pm fires; stage 2 waits for it
}

// runRewrite is the iterative state-machine implementation of graph_rewrite.
//
// Sharing and correctness invariant: shared nodes (same arena index appearing as a child
// of multiple parents) may be pushed onto the stack multiple times. The fast-path check
// at the top of the loop ("already in replace → pop immediately") makes duplicates O(1).
// This avoids the subtle bug where shared nodes get buried below their parents in the
// LIFO stack and cause an infinite wait in stage 1.
func runRewrite(sink uop.UOp, pm, bpm Matcher, ctx any) uop.UOp {
	a := sink.Arena()

	// replace maps an old arena index to its final arena index after all rewrites.
	// A node with replace[i] = i was not rewritten.
	replace := make(map[uint32]uint32, a.Len()*2)

	stack := make([]frame, 0, a.Len())
	stack = append(stack, frame{node: sink, stage: 0})

	for len(stack) > 0 {
		top := len(stack) - 1
		n := stack[top].node

		// Fast-path: already resolved (shared subgraph encountered again).
		if _, done := replace[n.Index()]; done {
			stack = stack[:top]
			continue
		}

		switch stack[top].stage {
		case 0:
			// Apply bpm to fixpoint.
			cur := n
			if bpm != nil {
				seen := map[uint32]bool{cur.Index(): true}
				for {
					r, ok := bpm.Rewrite(cur, ctx)
					if !ok {
						break
					}
					if seen[r.Index()] {
						break // cycle guard
					}
					seen[r.Index()] = true
					cur = r
				}
			}
			stack[top].newNode = cur
			stack[top].stage = 1

			// Push children of cur (post-order: children before parent).
			// Reverse push so first child ends up at top and is processed first.
			// We skip children already in replace; we do NOT guard against children
			// already on the stack elsewhere — shared nodes may be pushed multiple
			// times and are deduped by the fast-path at the top of the loop.
			for i := cur.NSrc() - 1; i >= 0; i-- {
				child := cur.Src(i)
				if _, done := replace[child.Index()]; !done {
					stack = append(stack, frame{node: child, stage: 0})
				}
			}

		case 1:
			newN := stack[top].newNode

			// Verify all children of newN are resolved. Push any that are not.
			// (Should rarely trigger; mainly a safety net for bpm-introduced children.)
			allDone := true
			for i := 0; i < newN.NSrc(); i++ {
				child := newN.Src(i)
				if _, done := replace[child.Index()]; !done {
					allDone = false
					stack = append(stack, frame{node: child, stage: 0})
					break
				}
			}
			if !allDone {
				continue
			}

			// Rebuild the node with resolved-replacement children.
			changed := false
			newSrcs := make([]uop.UOp, newN.NSrc())
			for i := 0; i < newN.NSrc(); i++ {
				child := newN.Src(i)
				newIdx := replace[child.Index()]
				newSrcs[i] = a.At(newIdx)
				if newIdx != child.Index() {
					changed = true
				}
			}

			var candidate uop.UOp
			if changed {
				candidate = a.New(newN.Op(), newN.DType(), newSrcs, newN.Arg(), newN.Tag())
			} else {
				candidate = newN
			}

			// Short-circuit if candidate was already processed (interning may return
			// an already-resolved node when srcs didn't actually change structurally).
			if finalIdx, done := replace[candidate.Index()]; done {
				replace[n.Index()] = finalIdx
				stack = stack[:top]
				continue
			}

			// Apply pm (post-order, once per node).
			if pm != nil {
				if r, ok := pm.Rewrite(candidate, ctx); ok && r.Index() != candidate.Index() {
					if finalIdx, done := replace[r.Index()]; done {
						replace[n.Index()] = finalIdx
						stack = stack[:top]
						continue
					}
					stack[top].stage = 2
					stack[top].replN = r
					stack = append(stack, frame{node: r, stage: 0})
					continue
				}
			}

			replace[n.Index()] = candidate.Index()
			stack = stack[:top]

		case 2:
			repl := stack[top].replN
			if finalIdx, done := replace[repl.Index()]; done {
				replace[n.Index()] = finalIdx
				stack = stack[:top]
			} else {
				stack = append(stack, frame{node: repl, stage: 0})
			}
		}
	}

	if finalIdx, ok := replace[sink.Index()]; ok {
		return a.At(finalIdx)
	}
	return sink
}
