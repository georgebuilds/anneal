# SPEC.md — Anneal: A Go tensor compiler

## 1. Goal and non-goals

### 1.1 Goal

Build a from-scratch Go implementation of tinygrad's architecture: an immutable-UOp graph, transformed by a pattern-rewrite engine, scheduled into kernels, lowered, rendered to device source, and executed. The defining property I am replicating is that **autodiff and optimization are compiler passes over a single IR**, which lets the scheduler fuse kernels across the forward/backward boundary.

### 1.2 The two decisions this spec is built on

- **(D1) Graph-rewrite compiler, not an idiomatic Go autodiff library.** Gradients are produced by a pass over the UOp graph, not by tape/closure backward methods. This is the more expensive path (arena + interning + pattern-matcher codegen are all in scope), and it is justified *only* by fusion-through-backward — which is the whole point. A tape-based autograd lib was considered and rejected because it hides the backward pass from the scheduler and forecloses tinygrad's reason to exist.

  **D1 status: VERIFIED.** Every backward node is an ordinary interned `UOp` on the same arena as the forward graph. The scheduler sees the complete forward + backward DAG and fuses across the boundary. See §5 for the as-built implementation mechanism.

- **(D2) Rangeify indexing model, not view-merge.** Movement ops become index arithmetic on ranges (the current scheduler), not stacked `View` composition resolved via `View.__add__`/`simplify`. I inherit a simpler model precisely because I carry no legacy. The `View`/stride/offset/mask *data structure* is still used as the per-op representation; what I do **not** build is the cross-view merge/simplify machinery — rangeify dissolves it into range substitution.

### 1.3 Scope status (current)

Original v1 was deliberately bounded; several items have shipped past that bound. Current honest status:

**Shipped beyond original v1:**
- **Symbolic shapes — dynamic batch dimension.** A net trains with the batch dim NOT baked into the kernel. The seam (§6.4) is now load-bearing for this use case. **General symbolic axis movement** (split/merge a symbolic dim, pad/shrink on a symbolic axis, dynamic seq-length) remains deferred — see §6.4 and §11 for the Option-A/B line.
- **JIT replay / `TinyJit`-style capture.** `tensor.JIT` captures the frozen execution plan on first call; subsequent calls skip sink construction, scheduling, and the leaf walk. The match guard is keyed on the captured output expression's structural key (§7.5c).
- **Schedule cache.** `CreateSchedule` memoized on a structural key (in-process, single-arena). The former determinism BLOCKER is resolved (§7.6 pass-7b notes).
- **`.upat` DSL codegen migration.** The symbolic ruleset is compiled from `rewrite/rules/symbolic.upat` (§4.1).
- **Migration I/O.** `tensor/npy` (load `.npy`/`.npz`) and `tensor/safetensors` (save/load HuggingFace checkpoints, bidirectionally compatible with the real Python library). Both pure-Go, no cgo, no runtime Python.
- **Low-precision dtypes.** f16 via `shader-f16` extension (fail-closed); bf16 as storage-only with f32 compute. Reduction accumulators stay f32 (§11 Q3).

**Still deferred / dropped:**
- General symbolic axis movement (Option B) — see §6.4 / §11. Picked up only when LLM seq-len reshaping is actually the goal.
- Multi-device / sharding / `ALLREDUCE`. Dropped for v1; `OpCopy`'s hard-boundary role is conditionally dormant and rejoins when this lands (§7.3 note).
- `ImageDType` and image-specific codegen paths. Dropped.
- BEAM search / autotuning + cost-based fusion (Pass 5 stub). The known v1 performance ceiling — every reduce is a hard kernel boundary. Real fusion + BEAM both relax it.
- Backends beyond the first. v1 targets one backend (§7).

---

## 2. Target surface (current)

| Dimension | Status |
|---|---|
| Shapes — static | ✅ |
| Shapes — dynamic batch (symbolic) | ✅ `NewSymbolicBatchInput` + `RealizeWithBinding` |
| Symbolic shapes — general movement (split/merge/pad a symbolic axis, seq-len) | ⛔ Deferred (Option B; see §6.4) |
| Devices | Single device (multi-device deferred) |
| Autodiff | ✅ Full reverse-mode via typed graph traversal (D1 verified — §5) |
| Backend | ✅ WebGPU (native + WASM; §7.8) |
| JIT | ✅ Capture/replay (`tensor.JIT`; §7.5c) |
| Schedule cache | ✅ Memoized on structural key (§7.6) |
| Migration I/O | ✅ `.npy`/`.npz` load; `.safetensors` save/load |
| Dtypes | `float32`, `int32`, `bool` runtime-verified; f16 ✅ via `shader-f16` (fail-closed if unavailable); bf16 ✅ storage-only (f32 compute); fp8 ⛔ Deferred. |
| Image dtype | ⛔ Not supported |
| BEAM autotuning / cost-based fusion | ⛔ Deferred |

The original v1 milestone — train a small MLP and a small conv net end to end with gradients produced by the graph traversal and kernels fused across the forward/backward boundary — was met. Subsequent additions extend it without re-litigating it.

---

## 3. Core IR — the UOp

The single IR node used across the entire stack (tensor graph, kernel graph, linearized instructions).

### 3.1 Structure

A UOp is an immutable record of five fields: `(op, dtype, src, arg, tag)`.

- `op` — operation enum.
- `dtype` — output dtype (`void` for control ops).
- `src` — ordered tuple of source UOps (the DAG edges).
- `arg` — static metadata (const value, axis, var name, kernel info, etc.). Must be one of the supported types listed in §3.3; passing an unsupported type panics at construction time.
- `tag` — classification tag used by lowering.

Equality is **by identity**, made correct by structural interning (§3.3).

### 3.2 Memory model — integer-indexed arena (load-bearing)

UOps are **not** Go pointers in a graph. They are `uint32` indices into a contiguous, pre-allocated arena of UOp structs. Rationale: the compiler allocates millions of short-lived UOps; a `*UOp` pointer graph would batter Go's GC. The arena gives cache locality and O(1) bulk reclamation.

**Lifecycle (as built):** The arena's lifecycle is bound to one **training step**, not one `Realize()` call. Within a step, multiple `Realize()` calls share one arena and freely accumulate nodes. Between steps, training code allocates a fresh `uop.NewArena()` and abandons the old one. The canonical training pattern is **fresh-arena-per-step**.

Nothing holds UOp arena indices across step boundaries — the step N leaf may be at index 7, step N+1 at index 3 in a fresh arena. The only legitimate cross-step state is `nn.Parameter.Value` (§7.5b), and JIT's captured plan, which uses structural keys to survive arena churn (§7.5c).

**Forward/backward provenance** is recorded as an out-of-band per-node phase tag on the arena (a parallel slice, scoped via `SetPhase`/`defer` and reset on `Arena.Reset`). It does NOT participate in interning — provenance is a property of *when* a node was first built, not *what* it is. First-construction-wins: a node interned during the forward pass stays tagged forward even if the backward pass later asks for the same structure.

### 3.3 Interning / hash-consing

Before constructing a UOp, hash its `(op, dtype, src, arg, tag)` signature and look it up in a cache; identical signatures resolve to the same arena index.

Go uses an **arena-bound cache**: the cache lives and dies with the arena, so dead nodes vanish when the arena is abandoned at step end.

**Bypass intern set:** `{OpUnique, OpLUnique, OpBuffer}` carry intrinsic identity that must not dedup.

**Structural key:** `uop.StructuralKeys(a *Arena) []uint64` computes a bottom-up content hash per node — `H(op, dtype_struct_hash, arg, [child_key for child in srcs])`. Children contribute by their *structural keys*, not by arena indices, so two structurally-identical subtrees built in different arena order get the same key. `DType.StructuralHash()` is itself a pure function of dtype fields (not a pointer address), so the key is process-portable. This is the foundation for deterministic scheduling (§7.6 pass-7b) and the JIT match guard (§7.5c).

---

## 4. The rewrite engine

### 4.1 PatternMatcher and the rule DSL

Transformations are `graph_rewrite` passes driven by a `PatternMatcher`: an op-indexed dispatch table of `(pattern, handler)` rules.

**Status: the symbolic ruleset has been migrated to `.upat` codegen.** `rewrite/rules/symbolic.upat` is the source of truth; `rewrite/gen/main.go` is the generator; `rewrite/rules/symbolic_gen.go` is the generated output (DO NOT EDIT, no reflection). The live exported `var Symbolic rewrite.Matcher = symbolicGenerated` is the generated matcher; the v0 runtime matcher is retained as `SymbolicV0` to serve as a differential-testing oracle.

**Drift enforcement:** `TestUpatDriftCheck` parses `symbolic.upat` and asserts exact bidirectional (op, handler) correspondence with `anneal explain`'s curated rule table. Adding/removing/changing a `.upat` symbolic rule without updating the curated table fails the build.

**Gradient rules are not (yet) a PatternMatcher ruleset.** See §5 for the as-built implementation; migration is optional and is the natural next user of the `.upat` DSL.

Reflection in the hot rewrite path is **prohibited**.

### 4.2 The driver

`graph_rewrite` walks the DAG applying rules. **The iterative driver was built from the start** — a slice-based state machine that handles arbitrarily deep graphs without stack overflow.

Two composable matchers in one driver: `bpm` (pre-order) and `pm` (post-order). Rule order matters — earlier rules win.

### 4.3 Rulesets

`symbolic` (arithmetic identity, constant folding, canonicalization). Scheduler and codegen passes are defined in their sections.

### 4.4 Bounds (interval arithmetic, not SMT)

`rewrite/rules/bounds.go` provides `BoundsOf(u uop.UOp) Bounds` — SMT-free integer interval arithmetic over the UOp graph. Composes through ALU ops (`OpAdd`/`Sub`/`Mul`/`IDiv`/`Mod`/`CmpLT`/`CmpNE`/`CmpEQ`/`Cast`), handles `OpDefineVar`, and is the foundation that load-bears symbolic-shape work without an external solver.

Tested directly via table-driven cases in `bounds_test.go` (90% line coverage on `BoundsOf` as of the hardening pass). A latent `OpMod` bug surfaced by that pass — the same-period guard used Go's truncating `/` where `floorDiv` is correct, producing too-tight bounds for dividend ranges straddling zero — has been fixed.

---

## 5. Autodiff

Gradients are computed by `tensor.Backward()` — a typed iterative reverse traversal of the forward UOp DAG. The per-op derivative rules live in `applyGradRule()`, a Go `switch` over `uop.Op` in `tensor/gradient.go`. Adjoints converging from multiple consumers are summed by injecting `OpAdd` nodes. The output is an augmented graph containing forward + backward UOps on the same arena, which the scheduler sees whole and fuses across.

`gradient.go`'s shape handling carries `[]shape.Sint` (not `[]int64`) so a symbolic batch dim flows through the backward pass as an opaque passthrough axis. See §6.4.

**Adjoint precision through Cast backward.** To preserve precision through narrowing/widening forwards, the backward rule for `OpCast`/`OpBitcast` uses `LeastUpperDType(adj.dtype, src.dtype)` for the resulting adjoint. This ensures that a forward narrowing (e.g. f32 → bf16) does not silently truncate the backward gradient to the narrow type; it stays f32. `TestGradientThroughCastBF16ToF32` serves as the value oracle (calculating `y = (1 + 1/256)·x` where the gradient `1.00390625` would truncate to `1.0` if narrowed).

**This was a deliberate implementation choice.** D1 is fully achieved: every backward node is an ordinary interned `UOp` with no tape, no closures, and no `requires_grad` flag on the execution path. **D1 has been VERIFIED end-to-end:** the scheduler fuses forward and gradient kernels, and finite-difference checks confirm correctness across static and dynamic-batch nets.

A PatternMatcher / `.upat` gradient ruleset is an option (the `.upat` generator now exists). It would be an ergonomics and introspectability improvement, not a correctness requirement. The current typed traversal is the correct, permanent v1 design.

**Visualizer consequence:** the visualizer cannot show the backward pass as "rules firing" until gradients migrate to a ruleset. It can show the resulting augmented graph (forward nodes in teal, backward nodes in ember; provenance is read from the per-node phase tag, §3.2) with correct coloring, and the fusion boundary in gold.

---

## 6. Shapes — View, ShapeTracker, and the symbolic seam

### 6.1 View and ShapeTracker

A `View` carries `(shape, strides, offset, mask, contiguous)`. A `ShapeTracker` is a stack of Views.

```go
type View struct {
    Shape      []Sint
    Strides    []Sint
    Offset     Sint
    Mask       [][2]Sint
    Contiguous bool
}
type ShapeTracker struct {
    Views []View
}
```

`Tensor`'s public API stays `[]int64` (`Tensor.Shape() []int64`) for the static path; `Tensor.ShapeSints()` is the symbolic-aware variant for callers that may hold symbolic dims.

### 6.2 Movement ops as view math (no copies)

All six movement primitives are pure stride/offset/mask edits on the last view; they never copy. Correctness rests on the flat index being affine: `idx = offset + Σ idx_i·stride_i`. A copy is never forced by a movement op — only by the **scheduler** (§7) when it inserts a realize boundary.

### 6.3 What rangeify changes (D2)

Under rangeify, cross-view composition is replaced by **range substitution**: movement ops swizzle index ranges rather than stacking views that later get merged. **I do not build `View.__add__` / `ShapeTracker.simplify`.** The View struct remains the per-op representation; the merge layer is the thing rangeify obviates.

### 6.4 Symbolic seam — Option A (dynamic batch) shipped; Option B deferred

The seam is `Sint = int | UOp`:

```go
type Sint interface { isSint(); ConstValue() (int64, bool) }
type ConstInt struct { V int64 }
type SymInt   struct { Node uop.UOp }
```

`SymInt` arithmetic builds real UOp expressions (`Add`/`Sub`/`Mul`/`Neg`/`IDiv`/`Mod`); `ConstInt`×`ConstInt` stays off-arena so the static path is bit-identical. `arena.DefineVar(name, min, max)` creates the symbolic dim. There is **no SMT/Z3 dependency** — bound reasoning is `BoundsOf` interval arithmetic over the same UOp graph (§4.4).

**Option A — dynamic batch (shipped):** a symbolic dim rides through ops as an **opaque passthrough axis**: matched by node identity (`SintShapesEqual`), moved whole by `Reshape`/`Expand`/`Permute`/`broadcast`/`Matmul`, never split or merged or compared arithmetically against another symbolic value. Reshape validation compares only the *concrete* sub-products, treating a symbolic dim as a matching token that must appear (by node identity) on both sides. Symbolic comparisons (`Lt`/`Le`/`Eq` on `SymInt`) deliberately **panic** as a fence: if the compiler ever reaches a path that needs to compare two symbolic values arithmetically, that is Option B territory and the panic catches it immediately rather than silently producing a wrong bound.

The symbolic dim's value reaches the GPU via a trailing WGSL **uniform** buffer keyed at dispatch time, so one compiled WGSL kernel runs at any batch size without recompiling (`Device.RunSymbolic` + a compile-once cache keyed on WGSL source). Binding is by point-substitution: `tensor.RealizeWithBinding(map[string]int64, tensors...)` substitutes each `DefineVar` with its `Const` value into the graph *before* scheduling, and the existing symbolic ruleset folds the result.

**Option B — split/merge a symbolic dim (deferred, own slice).** Reshapes that split or merge a symbolic axis (`[n*4]→[n,4]`), pad/shrink on a symbolic axis, dynamic seq-len reshaping. Requires building `Lt/Le/Eq`-on-`SymInt` as predicate UOps, symbolic-product reshape arithmetic, and symbolic masks — re-enters real symbolic-indexing correctness risk. Not needed for dynamic batch; it is what LLM seq-len reshaping eventually wants.

---

## 7. The scheduler (rangeify) and codegen

### 7.1 Shape of it

Two stages over one UOp graph: (1) `GetKernelGraph` splits the tensor-level graph into per-kernel `CALL` nodes by inserting `BUFFERIZE` boundaries and removing them where fusion pays; (2) `createSchedule` toposorts kernels into an ordered list of `ExecItem`s. There is no separate "kernel AST" type — it is all UOps.

### 7.2 Fusion mental model (inverted from PyTorch/JAX)

**Fuse by default; bufferize on realize; remove bufferize if cheap.** The grouper does not decide what to fuse — it decides what *not* to fuse by inserting `BUFFERIZE`, then a cost pass opportunistically deletes removable bufferizes.

### 7.3 Hard boundaries vs. tunable heuristics

**Hard correctness boundaries** (must separate kernels), in `buildRealizeMap`:

`CONTIGUOUS`, `ASSIGN`, `BUFFER_VIEW`, `ENCDEC`, `REDUCE_AXIS`, plus SINK srcs.

- **`REDUCE_AXIS` as hard boundary (v1 deliberate simplification):** every reduce — including every matmul and Linear — materialises to a buffer rather than fusing into adjacent elementwise ops. Known v1 performance ceiling, not a bug. Relaxing this is tied to deferred Pass 5 cost-fusion work.
- **`COPY` is conditionally deferred:** `OpCopy` was in the design's `ALWAYS_CONTIGUOUS` hard set and will rejoin `buildRealizeMap` when multi-device lands. Dormant — not deleted.

**Tunable performance choices** (not correctness): multi-consumer splitting, per-kernel buffer-count limits (Metal 31, WebGPU 8), reduce-reads-from-buffer keep.

**Conv2d im2col:** `Conv2d.Forward` uses im2col-as-single-matmul. A single `ReduceAxis` (matmul over K = Cin·kH·kW) materialises `conv_out` as **one buffer**, staying within WebGPU's 8-buffer-per-kernel limit in the backward pass.

### 7.4 SINK (three roles)

`SINK` is overloaded by stage: (a) tensor sink wrapping all live tensors at `realize()`; (b) kernel sink carrying `KernelInfo`, which becomes each kernel's AST; (c) traversal gate that stops schedule-level walks at kernel boundaries.

### 7.5 Buffer numbering (two stages)

`LUNIQUE` UID per bufferize (global, per-arena counter in `addBuffers`), then per-kernel `PARAM(arg=N)` numbering inside `splitKernels`. The renderer turns `PARAM(arg=N)` into `data{N}`. `ExecItem.bufs[N]` is the runtime buffer for `data{N}`.

Ordering at every step is driven by `uop.StructuralKeys` (§3.3), not by arena allocation order — see §7.6 pass-7b notes.

### 7.5b Parameter persistence

`nn.Parameter` is the only legitimately cross-step value state:

- **`Parameter.Value []float32`** — the canonical weight vector. Lives on the `Parameter` struct, not in any arena. Outlives every arena abandonment.
- **`Load(a *Arena) *Tensor`** — creates a fresh `OpBuffer` leaf in arena `a`, copies `p.Value` into `a.leaves[newLeaf.Index()]`, and sets `p.T` to the new leaf.
- **`SGDStep(grad []float32, lr float32)`** — applies `p.Value[i] -= lr * grad[i]` in-place after `Realize(gradTensor)` materialises the gradient.

The step N leaf may have arena index 7; the step N+1 leaf in a fresh arena may have index 3. Both are found via `p.Value` through `Load`. `TestCrossResetPersistence` verifies this end-to-end on GPU.

### 7.5c JIT capture/replay

`tensor.JIT` captures the frozen execution plan on the first call and replays subsequent calls without rebuilding the graph or rescheduling. The captured plan holds the `[]ExecItem` schedule plus a leaf table; replay re-uploads current `Parameter.Value` data into the recorded buffer slots, applies the current symbolic binding (if any), and dispatches.

**Crux — arena-index instability across step boundaries.** Per §3.2 / §10, arena indices are not stable across the fresh per-step arena. So a captured `ExecItem.Bufs[*].UOpIdx` cannot be naively reused on a later step. Resolution (Design B — structural remap): leaves are matched by preorder-DFS ordinal — a function of graph *topology*, not arena index — and data is re-resolved from the current step's `Parameter.Value` each replay.

**Match guard.** Replay is only valid if the current call is structurally the same as the captured one. The guard checks (i) leaf count, (ii) per-slot leaf sizes, and **(iii) the structural key of the captured output expression(s)** (`capSK = subgraphSK(tensors)` via `uop.StructuralKeys`). A mismatch on any of these forces a re-capture. The output-expression-SK is load-bearing — two same-shape `OpBuffer` leaves have identical leaf-level structural keys (no srcs, identical `arg`, off-graph `Value`), so leaf-SK alone could not discriminate them; the output-SK at the expression level catches any structural difference between capture and replay.

**Invariant to preserve:** the harmlessly-passing case is structurally-identical graphs with permuted *leaves* (e.g. `W1*x+W2 → W2*x+W1`). The DFS remap's permutation cancels the expression's permutation, so those correctly bypass the guard. This is intentional; do not over-tighten the guard.

JIT dispatch funnels through the same `onGPU` owner-goroutine path as `Run`/`RunSymbolic` (§7.8) — bypassing it would reintroduce the Metal autorelease-pool race.

### 7.6 Minimum viable scheduler — 10 ordered passes

Each pass is either a PatternMatcher of ~5–15 rules or a direct Go function.

1. **`earlyRewrites`**: clean movement-op chains, fold ASSIGN chains, fix self-assign hazards. *v1: no-op identity.*

2–4. **`runRangeify` (fused)**: computes the realize map, threads index arithmetic through each kernel subgraph, and inserts BUFFERIZE.

   **2–4a. Realize map:** `buildRealizeMap` marks SINK srcs and hard-boundary ops as realize points.

   **2–4b. Range propagation via `indexExprNode`:** for each realize point, fresh `AxisLoop RANGE` nodes are created for each output dimension. `indexExprNode` recurses through the kernel body, dissolving all six movement ops into index arithmetic. Symbolic dims (§6.4) flow through as symbolic `RANGE` bounds.

5. **`removeBufferize` cost pass.** Deliberate v1 no-op stub — every reduce is a hard kernel boundary. Real fusion is BEAM/autotuning future work.

6. **`addBuffers`** — assigns `LUNIQUE` ids to each surviving bufferize. **Ordered by structural key**, not by DFS visit order, so the same logical graph produces the same id sequence regardless of construction order.

7. **`splitKernels`** — splits the post-rangeify graph into per-kernel subgraphs at BUFFERIZE boundaries; assigns per-kernel `PARAM(arg=N)` numbering **ordered by structural key**.

7b. **`createSchedule`** — toposorts kernels via Kahn with a deterministic tiebreak. **Frontier tiebreak is by structural key**, not arena index.

   *Determinism fix (resolved blocker):* Earlier, `createSchedule`, `splitKernels`, and `addBuffers` all keyed sort/numbering on arena allocation order, so a structurally-equal graph built differently produced a different-but-valid schedule. The schedule cache (§1.3) could not work without this being fixed — the cache keys on structure but would return a schedule with mismatched PARAM numbering on a hit. Fix: ordering everywhere is now a function of `StructuralKeys`.

8. **`fixIndexDtype`** — narrows index expressions from int64 where the range bound proves int32 fits.

9. **`finalRewrites`** — codegen-ready cleanup.

10. **`Schedule cache`.** `CreateSchedule` is memoized on the structural key of its `sink` argument (+ device) via an arena-local cache. Hit returns the cached `[]ExecItem` directly; miss computes and stores. The cache is per-arena (arena indices in `ExecItem.Bufs` are only valid in their build arena), so it is correct within a step; JIT (§7.5c) is the across-step counterpart.

### 7.7 Codegen

Each kernel's SINK-rooted UOp tree is lowered to a linear `[]Instr` sequence by `Lower()`, then rendered to WGSL source by `RenderWGSL()`. The lowerer/renderer split is the current architecture; codegen is a pure function of `ExecItem` and never reaches back into the arena.

**Codegen targets WGSL exclusively** (`codegen/lower.go` + `codegen/wgsl.go`). The C-style renderer family (for CUDA/Metal Phase 2/3) is deferred.

**Low-precision dtypes (f16/bf16).**
- `enable f16;` directive emitted when any f16 buffer is in scope.
- f16 native types and arithmetic for elementwise ops.
- **Reduction accumulators widen to f32 even when operands are f16** (load-bearing correctness — pure FP16/FP16 is a deferred opt-in, not the default).
- bf16 as `array<u32>` storage with bitcast/shift widening at boundaries; no `enable f16;` required for bf16-only kernels.

For symbolic kernels (§6.4), the symbolic dim is rendered as a WGSL **uniform** buffer (not a storage buffer — uniforms don't count against Metal's 8-storage-buffer limit, restoring full data-buffer budget on symbolic kernels). The uniform must be a struct of `u32` fields, not `array<u32,N>` — array element stride is 16 bytes in the uniform address space; struct fields pack at 4 bytes. A prior memory-corruption-at-step-1200 bug came from this exact pitfall and has been fixed.

### 7.8 Backend strategy

**Use zero-CGO dynamic linking** (purego/goffi-style) at the driver boundary.

**As built:** the WebGPU backend uses `github.com/gogpu/wgpu`, `github.com/gogpu/naga`, `github.com/gogpu/gputypes`, `github.com/go-webgpu/webgpu`, and `github.com/go-webgpu/goffi`. The zero-CGO dynamic-linking strategy is confirmed for WebGPU.

**Metal threading discipline (load-bearing).** A single permanently-`runtime.LockOSThread`'d **GPU-owner goroutine** owns all Metal entry points. Every public call (`Run`, `RunSymbolic`, `DispatchSymKernel`, `CompileSymKernel`, `Open`, `Close`) funnels through it via `onGPU`; `*Locked` helpers assume they're already on the owner thread.

`readBuffer` does NOT use `wgpu.Buffer.Map` (which internally spawns an unpinned goroutine whose blocking `waitUntilCompleted` lets Go migrate it off the OS thread that created the `NSAutoreleasePool` → SIGSEGV in pool drain). Instead it uses `MapAsync` + an explicit `Poll(PollWait)` driven *on* the owner thread, so the library never spawns its internal goroutine and every pool create/drain pair shares one OS thread. This eliminated a ~70% crash rate under load.

Phasing:
- **Phase 1 — WebGPU (native via wgpu, + WASM via `syscall/js`).** v1 target, shipped.
- **Phase 2 — CUDA driver API** (`libcuda.so.1` + NVRTC → PTX) for datacenter throughput.
- **Phase 3 — Metal** (`objc_msgSend` FFI) for Apple-silicon unified-memory specialisation.

---

## 8. Module layout (as built)

```
uop/                UOp struct, arena, interning, Ops enum, dtype, KernelInfo,
                      StructuralKeys, DefineVar/Bind constructors
rewrite/            PatternMatcher, UPat DSL, graph_rewrite driver
  gen/              .upat → .go codegen
  rules/            symbolic.upat (source of truth), symbolic_gen.go (generated),
                      symbolic.go, alu.go, bounds.go
                    NOTE: gradient rules live in tensor/gradient.go (§5), not here
                    NOTE: scheduler passes use direct Go functions, not PatternMatcher rulesets
shape/              View, ShapeTracker, movement ops, Sint seam (sint.go)
schedule/           rangeify, realize-map, bufferize, kernel split, toposort,
                      memory plan, stats hook, structural-key-ordered scheduling,
                      schedule cache (cache.go)
codegen/            lower.go (→ []Instr linearisation), wgsl.go (WGSL renderer)
backend/            device abstraction
  webgpu/           open.go (locked GPU-owner goroutine), executor.go,
                      symbolic dispatch path (RunSymbolic + compile-once cache)
tensor/             Tensor API, ops, movement, gradient, realize, jit
  nn/               Linear, Conv2d, activations, SGD, Parameter
  npy/              .npy/.npz ingestion (pure Go)
  safetensors/      .safetensors save+load (pure Go, bidirectional w/ Python lib)
cmd/anneal/         CLI (run/train/graph/kernels/explain/doctor/viz verbs)
tui/                bubbletea/lipgloss train dashboard
viz/                visualizer
examples/           mlp.go, conv.go, dynmlp.go (dynamic-batch)
docs/               GitHub Pages site (bilingual en/es)

Module path: github.com/georgebuilds/anneal
```

---

## 9. Build order (history)

**Phases 0–12 complete; the post-v1 work below is also shipped.**

1. UOp + arena + interning + dtype + a v0 runtime PatternMatcher with named captures. ✓
2. `graph_rewrite` driver (iterative state machine) + gated toposort. ✓
3. `symbolic` ruleset; constant folding / canonicalization. ✓
4. Tensor frontend → UOp graph. ✓
5. shape/ (View + movement ops as range swizzles). ✓
6. Gradient traversal (`tensor.Backward`); verified vs finite differences. ✓
7. Scheduler 10-pass pipeline. ✓
8. Codegen + WebGPU backend. ✓
9. Train an MLP, then a conv net. v1 milestone. ✓
10. CLI verbs + TUI + visualizer scaffold. ✓
11. Visualizer real-compiler-via-WASM. ✓
12. `.upat` DSL codegen migration (symbolic ruleset). ✓

**Post-Phase-12 work (also shipped):**
- Structural-hash determinism fix (resolved the schedule-cache blocker).
- Schedule cache.
- JIT capture/replay (`tensor.JIT`) with output-SK match guard (§7.5c).
- Metal AutoreleasePool race fix — single locked GPU-owner goroutine (§7.8).
- Symbolic shapes — dynamic batch (Option A, §6.4); `Lt/Le/Eq`-on-`SymInt` deliberately panic as the Option-B fence.
- Migration I/O — `tensor/npy` (load) and `tensor/safetensors` (save+load, bidirectional with the real Python library).
- `explain` symbolic-rule drift check vs `symbolic.upat`.
- Bounds-system test coverage + `OpMod` floor-div fix.
- f16 in WGSL codegen with f32 accumulator (Slice 1).
- shader-f16 extension negotiation + bf16 storage-only (Slice 2).
- Tensor/nn f16/bf16 surface; implicit mixed precision via f32 master weights (Slice 3).
- FD gradient checks for f16/bf16 with literature-calibrated tolerances; OpCast adjoint precision fix (Slice 4).

---

## 10. Invariants (don't violate these)

- UOps are immutable; rewrites produce new nodes.
- Identity equality is only valid because of interning — never compare structurally in hot paths.
- Nothing holds UOp arena indices across step boundaries. Within a step, indices accumulate freely and are valid until the arena is abandoned at step end.
- `nn.Parameter.Value` is the only legitimate cross-step value state; JIT's captured plan is the only legitimate cross-step *plan* state, and survives via structural keys + per-step value re-resolution (§7.5c).
- Forward/backward provenance is recorded out-of-band on the arena (per-node phase tag), NOT in the interning key. First-construction-wins (§3.2).
- No reflection in the rewrite hot path.
- Movement ops never copy; only the scheduler materialises.
- Hard correctness boundaries (§7.3) are not negotiable; performance heuristics are.
- No SMT solver in the core indexing path; bound reasoning is interval arithmetic (`BoundsOf`, §4.4).
- Scheduling is a pure function of graph structure — kernel order, PARAM numbering, and LUNIQUE id assignment are driven by `StructuralKeys`, never by arena allocation order (§3.3, §7.6).
- Dtype structural identity goes through `DType.StructuralHash()` (a function of dtype fields), not pointer address — so structural keys are portable across processes.
- All Metal-touching calls go through the `onGPU` owner-goroutine funnel (§7.8). Bypassing it reintroduces the autorelease-pool race.
- JIT replay's match guard is keyed on the captured output expression's structural key, not on per-leaf identity. Same-shape sibling `OpBuffer` leaves cannot be discriminated at the leaf level (§7.5c).
- Symbolic comparisons (`Lt/Le/Eq` on `SymInt`) panic by design — they are the Option-A/Option-B fence (§6.4). If a code path needs one, that path is Option B and must be scoped as such, not silently unfenced.
- Reduction accumulators are f32 even when operands are f16; never accumulate in f16. The FP16/FP16 fast path is a deferred opt-in, not the default.
- Adjoint precision through OpCast/OpBitcast backward uses `LeastUpperDType(adj.dtype, src.dtype)`. A forward dtype narrowing must never silently narrow the backward adjoint.

---

## 11. Open questions still to resolve in design (not research)

1. **Iterative vs. recursive driver. RESOLVED.** The iterative slice-based state machine was built from the start. Anneal's iterative driver is the deliberate choice from §4.2.

2. **v0 matcher → codegen migration trigger. RESOLVED.** The `.upat` DSL was built and the symbolic ruleset migrated (§4.1). Drift check (`TestUpatDriftCheck`) enforces curated-prose ↔ `.upat` correspondence. The gradient ruleset remains the natural next user of the DSL — that migration is optional (D1 holds either way) and would unlock live derivation of `explain`'s gradient half and a "rules firing" backward animation in the visualizer.

3. **Dtype breadth — runtime lowering. RESOLVED.**
   - **f16** via `shader-f16` WGSL extension; reductions use an f32 accumulator for precision (narrows at write).
   - **bf16** as storage-only with f32 compute (`bitcast<u32>(expr) & 0xFFFF0000u` on store, `bitcast<f32>(u32_buffer[i])` on load).
   - **Fail-closed:** the engine fails before any GPU allocation if a requested f16/bf16 surface is unavailable; no silent fp32 fallback. `anneal doctor` reports availability per device.
   - **Tolerances:** elementwise atol=1e-3 (f16 ε ≈ 9.77e-4); small matmul atol=1e-2 rtol=1e-3 (TVM `test_to_mixed_precision`); chained atol=5e-2 rtol=1e-2 (tinygrad PR #7973: rtol=atol=2e-3 violated after 7 convs); bf16 FD atol=rtol=0.3 (7-bit mantissa quantization noise/h analysis).
   - **Deferred:** FP16/FP16 accumulation (fast-path opt-in); explicit mixed-precision training with loss scaling (Parameter.Value as f32 master gives implicit mixed precision; named API is deferred).

4. **Renderer target. RESOLVED: WGSL-only.** Codegen is WGSL-only (`codegen/lower.go` + `codegen/wgsl.go`). The C-style renderer family (for CUDA/Metal Phase 2/3) is deferred.

5. **Symbolic Option B (split/merge a symbolic dim) trigger.** When does the project actually need general symbolic axis movement? Recommendation: when LLM seq-len reshaping becomes a real target. Until then, the Option-A fence in `SymInt` comparisons (§6.4) holds.

---

## 12. Provenance and confidence

This spec is calibrated against the as-built implementation. Original Phases 1–10 were verified end-to-end by GPU training of MLP and conv net; subsequent shipped work (listed in §9) is each value-proven against an oracle appropriate to its surface:

- **Symbolic batch (Option A):** forward 1.19e-7 vs CPU (compile-once across batch sizes); backward FD gradient check 2.43e-4; symbolic-vs-static gradients identical (0.0 max-abs-diff); learnable-task training matches static MLP trajectory.
- **Schedule cache:** hit returns identical GPU results (max-abs-diff 0); one symbolic schedule serves multiple bindings; static path byte-identical.
- **JIT capture/replay:** replay vs non-JIT max-abs-diff 0 over many steps with changing weights; training converges (ratio 0.0011, Pearson 0.9752); adversarial output-SK guard test demonstrates the count-only guard's failure mode (+9 vs −9) and the structural-key fix.
- **Metal AutoreleasePool fix:** 0 crashes / 60 runs (30 + 30 with test-side pinning removed); value oracle byte-identical.
- **Determinism fix:** verified by two-build-order test producing identical schedule.
- **Migration I/O:** `.npy` round-trip vs real `np.save()` fixtures (big-endian + Fortran-order); safetensors BIDIRECTIONALLY round-tripped against the real Python library; save→load→identical-forward (max-abs-diff 0).
- **`OpMod` floor-div fix:** surfaced by bounds-system table-driven coverage (28% → 90%); adversarial cases for `[-3,3] mod 4` and `[-1,1] mod 2` now produce the correct `[0,3]` and `[0,1]` rather than the buggy `[1,3]` and `[1,1]`.
- **`.upat` drift check:** bidirectional (op, handler) correspondence enforced as a build-failing test.
- **f16/bf16 support:** f16 elementwise atol=1e-3 (ε ≈ 9.77e-4); bf16 FD atol=rtol=0.3 (calibrated to 7-bit mantissa noise floor); bf16-storage MLP trains at f32 quality (Pearson 0.9810 vs f32 baseline 0.9735).
- **OpCast adjoint fix:** `TestGradientThroughCastBF16ToF32` precision oracle proves the fix (`y = (1 + 1/256)·x` — gradient `1.00390625` would have narrowed to `1.0` pre-fix).

For any claim about current behavior, the code is the source of truth; this spec describes intent and invariants, not surface details that may evolve.
