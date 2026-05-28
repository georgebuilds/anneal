<div align="center">

<img src="docs/icon.svg" alt="anneal" width="128" height="128" />

# anneal

**A tensor compiler in Go — autodiff is a graph rewrite, and kernels fuse across the forward/backward seam.**

[![status](https://img.shields.io/badge/status-v1-14b8a6)](#status)
[![backend](https://img.shields.io/badge/backend-WebGPU-0d9488)](#backend)
[![go](https://img.shields.io/badge/Go-1.22%2B-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![model](https://img.shields.io/badge/it's%20a-compiler-f59e0b)](#what-anneal-is)
[![license](https://img.shields.io/badge/license-AGPL3-blue)](LICENSE)

[Visualizer](https://georgebuilds.github.io/anneal/) · [Architecture (SPEC)](SPEC.md) · [Contributing](CONTRIBUTING.md)

</div>

---

anneal is a from-scratch Go port of [tinygrad](https://github.com/tinygrad/tinygrad)'s modern, *rangeify-era* core. It takes tensor programs, lowers them through a graph-rewrite compiler, and emits fused GPU kernels. It currently trains a small MLP and a small convolutional network end-to-end on real GPU hardware via WebGPU.

It is a research project and a learning vehicle, built deliberately in phases. It is not (yet) a drop-in replacement for a production framework — see [Status](#status) for exactly what v1 does and doesn't do.

## What anneal is

Most autodiff libraries record a tape and replay it. anneal doesn't.

- **It's a compiler, not an autodiff library.** Everything — forward ops, gradients, movement ops — is a single immutable IR node (the `UOp`). Computation is suspended until you `Realize()`, at which point the whole program is one graph the compiler can rewrite, schedule, and fuse.
- **Gradients are a rewrite pass.** `Backward()` doesn't build closures; it injects gradient `UOp`s into the *same* graph as the forward pass. The scheduler then fuses kernels across the forward/backward boundary — an optimization that's structurally impossible with a tape.
- **Movement ops are range arithmetic, not copies.** reshape, permute, expand, pad, shrink, and flip never move data. They become index math (the *rangeify* model), and the only thing that ever materializes a buffer is the scheduler.
- **It runs in the browser.** The same compiler builds to WASM and powers the [live visualizer](https://georgebuilds.github.io/anneal/) — which runs the *real* compiler, not a mock.

In the visualizer (and throughout the project) color encodes architecture:

![forward](https://img.shields.io/badge/forward-teal-14b8a6) &nbsp; ![backward](https://img.shields.io/badge/backward-ember-f97316) &nbsp; ![fused](https://img.shields.io/badge/fused-gold-f59e0b)

## Quickstart

anneal ships a single CLI, `anneal`, which is the fastest way to see it work.

```bash
# install the CLI
go install github.com/georgebuilds/anneal/cmd/anneal@latest

# or, from a clone:
git clone https://github.com/georgebuilds/anneal && cd anneal
go build ./cmd/anneal
```

Then:

```bash
anneal doctor      # check your environment can reach a WebGPU device
anneal train mlp   # train the MLP with a live TUI dashboard (also: conv, dynmlp --batch=N)
anneal graph       # dump the UOp graph for a program
anneal kernels     # show the scheduled, fused kernels and their WGSL
anneal explain add  # explain the rewrite/gradient rules for an op
```

`anneal doctor` is the right first command: anneal links the platform WebGPU driver at runtime (zero-CGO), so `doctor` tells you whether a usable device is present before anything else.

## Using anneal as a library

The tensor API will feel familiar if you've used tinygrad or numpy. The key difference is the lazy/realize boundary:

```go
import "github.com/georgebuilds/anneal/tensor"

// ... build a model and a forward pass producing `loss` ...

loss.Backward()   // injects gradient UOps into the same graph (teal → ember)
loss.Realize()    // schedule, fuse across the seam (gold), compile to WGSL, run
```

For runnable, end-to-end code, including parameter setup, the training loop, and an SGD step — see [`examples/`](examples) (`mlp.go`, `conv.go`, `dynmlp.go`). Those are the canonical reference for the current API surface.

## Project layout

```
uop/         UOp IR: arena, interning, ops enum, dtype
rewrite/     PatternMatcher, graph-rewrite driver, symbolic rules
shape/       View, ShapeTracker, movement ops
schedule/    rangeify, realize-map, bufferize, kernel split
codegen/     UOp tree → linear instrs → WGSL
backend/     device abstraction; webgpu/ first
tensor/      Tensor API, ops, autodiff (gradient.go), realize
  nn/        Linear, Conv2d, activations, SGD, Parameter
cmd/anneal/  the CLI
viz/         the WASM visualizer
examples/    mlp.go, conv.go, dynmlp.go
```

The full architecture — the UOp arena and interning model, the rewrite driver, the rangeify indexing model, the 10-pass scheduler, and the design decisions behind them — lives in **[SPEC.md](SPEC.md)**. Read it before making non-trivial changes.

## Status

The line between shipped capabilities and deferred ones is intentional, not accidental. That line has moved since the project started — dynamic-batch training and JIT have landed — but the harder items remain deliberate non-goals for now.

| Capability | Status |
|---|---|
| Reverse-mode autodiff | ✅ Full, via graph rewrite |
| Backend | ✅ WebGPU (native + WASM) |
| Shapes — static | ✅ |
| Shapes — dynamic batch (symbolic) | ✅ `NewSymbolicBatchInput` + `RealizeWithBinding` |
| Symbolic shapes — general movement (split/merge/pad a symbolic axis, seq-len) | ⛔ Deferred |
| JIT | ✅ Capture/replay (`tensor.JIT`) |
| Schedule cache | ✅ Memoized on structural key |
| Devices | Single device |
| Dtypes | f16 ✅ (with shader-f16); bf16 ✅ storage-only (f32 compute); fp8 ⛔ Deferred |
| Multi-device | ⛔ Deferred |
| Image dtypes | ⛔ Deferred |
| BEAM autotuning | ⛔ Deferred |

The original milestone — train a small MLP and a small conv net end-to-end on GPU, with gradients produced by the rewrite pass and kernels fused across the forward/backward boundary — is met. Since then: dynamic-batch training (`dynmlp`, symbolic batch dim), JIT capture/replay, and a schedule cache have all shipped. The harder deferrals listed above remain intentional.

## Contributing

Contributions are welcome, but anneal has a small set of hard invariants (immutable IR, identity equality via interning, no reflection in the rewrite hot path, no copies from movement ops, no SMT solver in indexing) that keep the design coherent. Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** before opening a PR.

## Credits

anneal is largely a port of, and owes its architecture to, [tinygrad](https://github.com/tinygrad/tinygrad) by the tinygrad authors. The reference target is a pinned tinygrad commit (see [CONTRIBUTING.md](CONTRIBUTING.md)); blog-era LazyBuffer/Linearizer descriptions of tinygrad do *not* describe this design.

GPU access is via [`gogpu/wgpu`](https://github.com/gogpu/wgpu) and `goffi` (zero-CGO).

## License
