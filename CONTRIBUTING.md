# Contributing to anneal

Thanks for your interest in anneal. This is largely a from-scratch Go port of [tinygrad](https://github.com/tinygrad/tinygrad)'s modern, rangeify-era compiler core, built deliberately in phases. This guide covers how to get set up, the conventions I hold to, and most importantly, the invariants you must not break.

If you're new to the architecture, read **[SPEC.md](SPEC.md)** first. It is the source of truth, and it takes precedence over your priors about how tinygrad works.

## Philosophy

A few framing points that explain most of the rules below:

- **anneal is a compiler, not an autodiff library.** Gradients are a rewrite pass over the UOp graph, not a tape. If your change makes autodiff feel more "idiomatic Go" at the cost of the graph staying a single, schedulable, fusable IR — it's the wrong change.
- **It's a faithful port of a *specific* commit,** not of tinygrad-in-general. We pin to a known tinygrad master commit (see SPEC.md). Most tutorials and blog posts describe the obsolete LazyBuffer/Linearizer architecture — ignore them. When a question touches current tinygrad internals, **read the actual source at the pinned commit** rather than recalling it.
- **Three corrections worth internalizing,** because external sources get them wrong:
  1. There is **no Z3/SMT solver** in the core indexing path — it's graph rewrite plus interval arithmetic.
  2. Upstream's rewrite driver is **recursive**; ours is **iterative** by design (a deliberate improvement, not an accident to "fix").
  3. The IR memory model is an **integer-indexed arena with interning**, never a `*UOp` pointer graph.

## Getting set up

```bash
git clone https://github.com/georgebuilds/anneal && cd anneal
go build ./...
go test ./...
go run ./cmd/anneal doctor   # confirm a WebGPU device is reachable
```

You'll need a recent Go toolchain (see `go.mod`) and a platform with a WebGPU-capable driver. anneal links the driver at runtime via zero-CGO, so you do **not** need a C compiler, CUDA toolkit, or Xcode at build time.

## The invariants (don't break these)

These are non-negotiable. A PR that violates one will be sent back regardless of how good it otherwise is. They come straight from SPEC §10:

- **UOps are immutable.** Rewrites produce *new* nodes; they never mutate existing ones.
- **Identity equality is only valid because of interning.** Never compare UOps structurally in hot paths — and never assume two structurally-equal graphs share arena indices.
- **No reflection in the rewrite hot path.** This is a hard performance invariant for the rewrite engine.
- **Movement ops never copy.** reshape/permute/expand/pad/shrink/flip are index arithmetic only. The *scheduler* is the only thing that materializes a buffer.
- **No SMT solver in the core indexing path.** Indexing is graph rewrite + interval arithmetic.
- **Nothing holds an arena index across a training step.** Within a step, indices accumulate freely and are valid until the arena is abandoned at step end. The only legitimate cross-step state is `nn.Parameter.Value`.
- **Hard correctness boundaries are not negotiable;** performance heuristics are. Know which is which before you touch the scheduler (SPEC §7.3).

### The recurring bug class

We've been bitten by the same mistake four times, so it gets its own heading:

> **Using a construction-order or allocation artifact as if it were a stable structural identity.**

Arena indices reflect *the order nodes were built*, not *what they are*. When you key anything off an arena index — schedule ordering, an ID counter, a cache key, provenance tracking — ask yourself: *does this result need to be a function of graph structure, or just of this particular build?* If it must be structural, derive it from a content hash (op, arg, dtype, sorted child hashes), not from an index. This is the single most common source of silent correctness bugs in the codebase.

## Testing

- **Table-driven tests** are the default. Follow the existing style.
- **For anything that executes, a passing test count is not a sufficient report.** We require a *value oracle* — actual numbers. For autodiff: finite-difference agreement. For training: a loss trajectory that goes down. "All tests pass" has hidden real correctness gaps more than once; numeric oracles caught them. If your change touches gradients, the scheduler, or codegen, show the numbers.
- **Slice risky work so the novel, correctness-critical part is proven before mechanical work layers on top of it.** Don't build the easy 80% on an unverified core.

## Code style

- `gofmt` / `goimports`, standard Go conventions.
- Strict typing. Prefer concrete types and exhaustive switches over `interface{}` and reflection — and reflection is *banned* in the rewrite hot path.
- Keep the IR immutable and the arena model intact (see invariants above).
- Match the surrounding file's structure; this codebase mirrors tinygrad's domains by package.

## Submitting changes

1. **Stay in scope.** Some features are intentionally deferred: general symbolic movement (splitting/merging/padding a symbolic axis, dynamic seq-len), multi-device, and image dtypes — see the README status table. PRs that pull deferred features forward will likely be declined unless they've been discussed first. BEAM autotuning and epilogue fusion have shipped; new kernel Opts belong in `codegen/opt.go` as additional `OptKind` variants following the existing pattern.
2. **Keep the docs honest.** If your change alters the architecture, update SPEC.md (and DESIGN.md if it touches a surface) in the same PR. Stale design docs are worse than none.
3. **Show your oracle.** Include the finite-difference / loss-trajectory numbers for anything that executes.
4. **One focused change per PR.** Easier to review, easier to bisect.
5. Open an issue first for anything large or architectural — it's faster than finding out in review that it conflicts with a locked decision.

## Questions

If something in SPEC.md is ambiguous or looks wrong against the pinned tinygrad commit, that's worth an issue on its own — the spec is meant to be precise and source-grounded.
