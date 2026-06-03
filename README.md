<div align="center">

<img src="docs/favicon.svg" alt="anneal" width="128" height="128" />

# anneal

**A tensor compiler in Go — autodiff is a graph rewrite, and kernels fuse across the forward/backward seam.**

[![status](https://img.shields.io/badge/status-v1-14b8a6)](#status)
[![backend](https://img.shields.io/badge/backend-WebGPU-0d9488)](#backend)
[![go](https://img.shields.io/badge/Go-1.26%2B-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![model](https://img.shields.io/badge/it's%20a-compiler-f59e0b)](#what-anneal-is)
[![codecov](https://codecov.io/github/georgebuilds/anneal/branch/main/graph/badge.svg?token=1S9OUTWWG8)](https://codecov.io/github/georgebuilds/anneal)
[![license](https://img.shields.io/badge/license-AGPL3-blue)](LICENSE)

[Visualizer](https://georgebuilds.github.io/anneal/visualizer-demo/) · [Architecture (SPEC)](SPEC.md) · [Limitations](LIMITATIONS.md) · [Contributing](CONTRIBUTING.md)

</div>

---

anneal is a from-scratch Go port of [tinygrad](https://github.com/tinygrad/tinygrad)'s modern, *rangeify-era* core. It takes tensor programs, lowers them through a graph-rewrite compiler, and emits fused GPU kernels. It trains a small MLP, a small convolutional network, and a char-level nanoGPT end-to-end on real GPU hardware via WebGPU, and runs GPT-2-small forward with bit-identical output to HuggingFace's reference implementation.

It is a research project and a learning vehicle, built deliberately in phases. It is not (yet) a drop-in replacement for a production framework — see [Status](#status) for exactly what v1 does and doesn't do.

## What anneal is

Most autodiff libraries record a tape and replay it. anneal doesn't.

- **It's a compiler, not an autodiff library.** Everything — forward ops, gradients, movement ops — is a single immutable IR node (the `UOp`). Computation is suspended until you `Realize()`, at which point the whole program is one graph the compiler can rewrite, schedule, and fuse.
- **Gradients are a rewrite pass.** `Backward()` doesn't build closures; it injects gradient `UOp`s into the *same* graph as the forward pass. The scheduler then fuses kernels across the forward/backward boundary — an optimization that's structurally impossible with a tape.
- **Movement ops are range arithmetic, not copies.** reshape, permute, expand, pad, shrink, and flip never move data. They become index math (the *rangeify* model), and the only thing that ever materializes a buffer is the scheduler.
- **It runs in the browser.** The same compiler builds to WASM and powers the [live visualizer](https://georgebuilds.github.io/anneal/visualizer-demo/), which runs the *real* compiler, not a mock.

In the visualizer (and throughout the project) color encodes architecture:

![forward](https://img.shields.io/badge/forward-teal-14b8a6) &nbsp; ![backward](https://img.shields.io/badge/backward-ember-FF7A45) &nbsp; ![fused](https://img.shields.io/badge/fused-gold-f59e0b)

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
anneal doctor               # check your environment can reach a WebGPU device
anneal train mlp            # train the MLP with a live TUI dashboard (also: conv, dynmlp --batch=N)
anneal train nanogpt        # char-level transformer trained end to end on Shakespeare
anneal gpt2 sample "Hello"  # forward GPT-2-small from HuggingFace weights, sample text
anneal graph                # dump the UOp graph for a program
anneal kernels              # show the scheduled, fused kernels and their WGSL
anneal explain add          # explain the rewrite/gradient rules for an op
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

For symbolic / dynamic-shape inputs, compose `Variable` values into the shape list and bind concrete values at realize time. The same compiled kernel runs at any bound value in `[min, max]`:

```go
seq := tensor.NewVariable(a, "seq_len", 1, 1024)
x   := tensor.NewSymbolicShape(a, []shape.Sint{
        shape.Const(batch), seq.Sint(), shape.Const(dim),
}, uop.Dtypes.Float32, "webgpu")
// ... build forward pass producing y ...
tensor.RealizeWithBinding(seq.Bind(64), y)
```

For runnable, end-to-end code, including parameter setup, the training loop, optimizer steps, and generation, see [`examples/`](examples): `mlp.go`, `conv.go`, `dynmlp.go`, `nanogpt.go` (char-level transformer training), and `gpt2/` (HF safetensors load + BPE + autoregressive sample). Those are the canonical reference for the current API surface.

## Project layout

```
uop/         UOp IR: arena, interning, ops enum, dtype
rewrite/     PatternMatcher, graph-rewrite driver, symbolic rules
shape/       View, ShapeTracker, movement ops
schedule/    rangeify, realize-map, bufferize, kernel split
codegen/     UOp tree → linear instrs → WGSL; opt.go (Opt seam, four kernel transforms), beam.go (BEAM autotuning)
backend/     Renderer/Compiler/Allocator/Program/DeviceBuffer interfaces; webgpu/ first
tensor/      Tensor API, ops, autodiff (gradient.go), realize
  nn/        Linear, Conv2d, MaxPool2D, Embedding, LayerNorm, CausalSelfAttention,
             MLP, Block, GPT, activations, SGD, Adam, Parameter
cmd/anneal/  the CLI
viz/         the WASM visualizer
examples/    mlp.go, conv.go, dynmlp.go, nanogpt.go, gpt2/
internal/
  assets/    SHA-pinned downloader for Shakespeare corpus and HF GPT-2 weights
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
| Symbolic shapes — split/merge a symbolic axis, sym pad/shrink, multi-dim sym dispatch | ✅ Shipped |
| Dynamic seq-length tensor API | ✅ `tensor.NewVariable` + `tensor.NewSymbolicShape` (non-outermost sym, multiple Variables per shape) |
| JIT | ✅ Capture/replay (`tensor.JIT`) |
| Schedule cache | ✅ Memoized on structural key |
| Devices | Single device |
| Dtypes | f16 ✅ (RTNE, requires shader-f16); bf16 ✅ storage + RTNE narrowing, f32 compute, any adapter; fp8 ⛔ Deferred |
| Multi-device | ⛔ Deferred |
| Image dtypes | ⛔ Deferred |
| BEAM autotuning | ✅ Env-gated (ANNEAL_BEAM=1 to search); persistent disk cache |

For the specific shape of each deferral and the platform ceilings behind them (8-buffer-per-kernel WGSL limit, single-adapter WebGPU constraint, non-matmul `OptUpcast`/`OptVectorize`, the WGSL `var<workgroup>` ceiling that gates `OptTile` on symbolic axes), see [LIMITATIONS.md](LIMITATIONS.md).

The original milestone — train a small MLP and a small conv net end-to-end on GPU, with gradients produced by the rewrite pass and kernels fused across the forward/backward boundary — is met. Since then: dynamic-batch training (`dynmlp`, symbolic batch dim), general symbolic axis movement (split/merge a symbolic dim, sym pad/shrink, multi-dim sym dispatch with the symbolic axis in any position on both kernel-output and input buffers), JIT capture/replay, a schedule cache, epilogue fusion (Pass 5 now elides a reduce-output BUFFERIZE into a single downstream elementwise consumer), and BEAM autotuning (env-gated, disk-cached) have all shipped. The remaining deferrals listed above are intentional. Kernel autotuning: `LOCAL` applies to multi-dim symbolic kernels; `TILE` stays unavailable on symbolic axes because WGSL forbids non-const workgroup sizes, a hard platform ceiling; `UPCAST` and `VECTORIZE` are matmul-only by lowerer design (only `emitTiledReduce` handles their per-lane positions); both are now fail-loud at opt-application time when composed without `OptTile`, and BEAM's `ActionSpace` pre-filters them on non-tiled kernels. Symbolic kernels still run correctly via the identity codegen path.

## Contributing

Contributions are welcome, but anneal has a small set of hard invariants (immutable IR, identity equality via interning, no reflection in the rewrite hot path, no copies from movement ops, no SMT solver in indexing) that keep the design coherent. Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** before opening a PR.

## Credits

anneal is largely a port of, and owes its architecture to, [tinygrad](https://github.com/tinygrad/tinygrad) by the tinygrad authors. The reference target is a pinned tinygrad commit (see [CONTRIBUTING.md](CONTRIBUTING.md)); blog-era LazyBuffer/Linearizer descriptions of tinygrad do *not* describe this design.

GPU access is via [`gogpu/wgpu`](https://github.com/gogpu/wgpu) and `goffi` (zero-CGO).

## License
