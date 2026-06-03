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
- **Adding a new backend.** A backend implements `backend.Renderer`, `backend.Compiler`, `backend.Allocator`, `backend.Program`, and `backend.DeviceBuffer`; the orchestrator pattern in `backend/webgpu/executor.go` is the reference for how to compose them. Threading discipline (a locked OS-thread GPU-owner goroutine, see `backend/webgpu/open.go`) is required if the target driver, like Metal, has thread-affine state; WebGPU's `onGPU` funnel is the canonical example.

## Getting set up

```bash
git clone https://github.com/georgebuilds/anneal && cd anneal
go build ./...
go test ./...
go run ./cmd/anneal doctor   # confirm a WebGPU device is reachable
```

You'll need a recent Go toolchain (see `go.mod`) and a platform with a WebGPU-capable driver. anneal links the driver at runtime via zero-CGO, so you do **not** need a C compiler, CUDA toolkit, or Xcode at build time.

### Optional: regenerating `docs/og-image.png`

The repo's social preview PNG (`docs/og-image.png`) is rasterized from `docs/og-image.svg`. If you edit the SVG, regenerate the PNG with the Node helper declared in `package.json`:

```bash
npm install
node -e "const {Resvg}=require('@resvg/resvg-js'); const fs=require('fs'); const svg=fs.readFileSync('docs/og-image.svg'); fs.writeFileSync('docs/og-image.png', new Resvg(svg).render().asPng())"
```

`node_modules/` is gitignored; only `package.json` and `package-lock.json` are tracked. Skip this step if you're not touching the SVG.

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

We've been bitten by the same mistake many times, so it gets its own heading:

> **Using a construction-order or allocation artifact as if it were a stable structural identity.**

Arena indices reflect *the order nodes were built*, not *what they are*. The same applies to positional slot indices, allocation order in `Buffer.Shape`, and any per-build counter used as a name. When you key anything off such a quantity — schedule ordering, an ID counter, a cache key, provenance tracking, a sym-dim slot number — ask yourself: *does this result need to be a function of graph structure, or just of this particular build?* If it must be structural, derive it from a content hash (op, arg, dtype, sorted child hashes) or from a name, not from an index. This is the single most common source of silent correctness bugs in the codebase, and each new instance has surfaced when a test combined structural elements previous tests kept separate — bare pass-counts hide it; value oracles catch it.

## Testing

- **Table-driven tests** are the default. Follow the existing style.
- **For anything that executes, a passing test count is not a sufficient report.** We require a *value oracle* — actual numbers. For autodiff: finite-difference agreement. For training: a loss trajectory that goes down. "All tests pass" has hidden real correctness gaps more than once; numeric oracles caught them. If your change touches gradients, the scheduler, or codegen, show the numbers.
- **Slice risky work so the novel, correctness-critical part is proven before mechanical work layers on top of it.** Don't build the easy 80% on an unverified core.
- **Coverage.** Run `make coverage` to see per-package and total coverage; CI runs the same target on every PR. Coverage is a hygiene signal, not the oracle: it complements (does not replace) the finite-difference / loss-trajectory numbers above.
- **Lint is enforced.** `make lint` (golangci-lint) runs on every PR and fails the build on any violation. The local golangci-lint cache can report stale counts after a `.golangci.yml` change, so `golangci-lint cache clean && make lint` is the authoritative local check.

## Code style

- `gofmt` / `goimports`, standard Go conventions.
- Strict typing. Prefer concrete types and exhaustive switches over `interface{}` and reflection — and reflection is *banned* in the rewrite hot path.
- Keep the IR immutable and the arena model intact (see invariants above).
- Match the surrounding file's structure; this codebase mirrors tinygrad's domains by package.

## ONNX importer (onnx/)

The importer lowers ONNX nodes onto the same UOp arena as the rest of the compiler. There is no shadow IR. Handler files are grouped by op surface (`onnx/handlers_const.go`, `_elementwise.go`, `_reduction.go`, `_movement.go`, `_conv.go`, `_pool.go`, `_norm.go`, `_linear.go`, `_transformer.go`) and a new handler lands in the file that owns its surface.

- **Coverage policy.** New handlers map ONNX ops onto anneal Tensor ops. If a primitive doesn't exist (the Phase 3 work surfaced `OpErf` and `OpMin` this way), the primitive lands first (new `Op` iota + WGSL lowerer + gradient rule + drift-table entry in `cmd/anneal/cmd_explain.go`), and the handler dispatches to it.
- **Value-oracle test per handler.** Every new handler ships a numeric assertion, not a pass-count. The pattern is in `onnx/handlers_*_test.go`: build the same ONNX subgraph two ways on the same arena, evaluate both via the tree-walking CPU evaluator (`onnx/cpu_eval_test.go`), assert `[]float32` slice equality. The Strategy A bit-exact gate (`max-abs-diff = 0`) is what surfaced the Phase 4 `Expand` rank-broadcast bug.
- **Punts must be loud.** Unsupported attribute combinations (`Conv group>1`, `MaxPool ceil_mode`, `Slice |step|>1`, etc.) reject at handler dispatch with a clear error citing the documented v1 boundary; no silent wrong outputs.
- **Conformance corpus.** `onnx/testdata/node/` ships 234 cases drawn from the ONNX 1.17.0 backend node-test suite. Regenerate via `notes/scripts/copy_node_corpus.py` (pins onnx 1.17.0, filters `_expanded` / `_int8` / quantized / control-flow / string / float8 / sequence / map / optional). Any case not in the skip list (`onnx/conformance_skip.go`) and not passing is a real bug. The skip list is the exclusion contract; entries must cite the plan section.
- **Structure-only mode.** Visualization callers pass `onnx.WithStructureOnly()` to skip initializer payload materialisation. The runner still resolves shapes and dtypes correctly; `Run` fails closed because payloads aren't there. The studio's WASM dropzone is the canonical caller.

## anneal web (web/ + cmd/anneal/cmd_web*.go)

The studio is a single-binary local browser surface. Five files compose it: `web/studio.html`, `web/studio.css`, `web/studio.js`, `web/worker.js`, and `web/anneal.wasm` (built from `viz/wasm/`, embedded via `web/embed.go`).

- **WASM hosted in a Web Worker.** The compiler frontend runs off the main thread; the main thread stays responsive. The Worker exposes the `annealGetGraph` / `annealGetKernels` / `annealExplainOp` / `annealNodeDetail` / `annealImportONNX` / `annealInspectTensor` bridge. The same artifact powers the visualizer-demo page.
- **WASM/native split is load-bearing.** Every view that *compiles* runs as WASM in the Worker. Every view that *executes* (train, generate) streams over SSE from a native handler in `cmd/anneal/cmd_web_*.go`. The doctor view is both: native device card next to the browser's `navigator.gpu` probe.
- **DD1 through DD4 are binding.** Colour is never alone (every coloured element carries shape, label, or weight as a second channel); the real compiler only (no mocks, the WASM bridge is the production path); the library is the product (the studio is a view, not a fork); restraint (no notebook, no model hub, no telemetry, no auth, no third-party JS). See [DESIGN.md](DESIGN.md) §2.
- **Accessibility is a gate, not a goal.** Every new view passes the per-view checklist in [web/A11Y.md](web/A11Y.md) §1 and the 20 foundation a11y tests in `cmd/anneal/cmd_web_test.go` (TestWebA11Y_*). The matching DESIGN invariants (skip link, semantic landmarks, single h1 per active view, focus visible, keyboard reachable, ?-modal chord help, polite live region, aria-current, touch target 44x44, reduced-motion, forced-colors mapping, lang attribute) are pinned source-side; adding a view without updating those tests is a merge blocker. See [DESIGN.md](DESIGN.md) §11.
- **Bundle format contract.** `internal/bundle/` writes runs to `~/.cache/anneal/runs/<ts>-<model>-<6hex>/` with a `bundle_version` integer in `manifest.json`. `BundleVersion = 1` is current. Additive fields that older readers can ignore are fine; breaking changes bump the version and the reader refuses older versions with a documented message. The bundle directory name shape is regex-enforced. The web tier defaults to ON (`?bundle=0` disables); the CLI defaults to OFF (`--bundle` or `ANNEAL_BUNDLE=1` opts in).

## Submitting changes

1. **Stay in scope.** General symbolic axis movement (split/merge a symbolic axis, sym pad/shrink, multi-dim sym dispatch) and the dynamic seq-length tensor-input frontend (`tensor.Variable` + `tensor.NewSymbolicShape`) have shipped; ONNX import and `anneal web` have shipped; remaining intentional deferrals are multi-device and image dtypes (see the README status table) plus the documented ONNX v1.1 punt list (see LIMITATIONS.md). PRs that pull deferred features forward will likely be declined unless they've been discussed first. BEAM autotuning and epilogue fusion have shipped; new kernel Opts belong in `codegen/opt.go` as additional `OptKind` variants following the existing pattern. Note that the four shipped opts (`LOCAL` / `TILE` / `UPCAST` / `VECTORIZE`) currently blanket-bail on symbolic axes pending a separate slice that drops the lowerer's 1D-flattened sym dispatch model. Relaxing those bails without that prerequisite produces wrong indices, not OOB.
2. **Keep the docs honest.** If your change alters the architecture, update SPEC.md (and DESIGN.md if it touches a surface) in the same PR. Stale design docs are worse than none.
3. **Show your oracle.** Include the finite-difference / loss-trajectory numbers for anything that executes.
4. **One focused change per PR.** Easier to review, easier to bisect.
5. Open an issue first for anything large or architectural — it's faster than finding out in review that it conflicts with a locked decision.

## Questions

If something in SPEC.md is ambiguous or looks wrong against the pinned tinygrad commit, that's worth an issue on its own — the spec is meant to be precise and source-grounded.
