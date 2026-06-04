<div align="center">

<img src="docs/favicon.svg" alt="anneal" width="128" height="128" />

# anneal

**A tensor compiler in Go. Autodiff is a graph rewrite, kernels fuse across the forward/backward seam, ONNX import is zero-CGO, and `anneal web` is a local browser studio.**

[![backend](https://img.shields.io/badge/backend-WebGPU-0d9488)](#backend)
[![go](https://img.shields.io/badge/Go-1.26%2B-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![model](https://img.shields.io/badge/it's%20a-compiler-f59e0b)](#what-anneal-is)
[![codecov](https://codecov.io/github/georgebuilds/anneal/branch/main/graph/badge.svg?token=1S9OUTWWG8)](https://codecov.io/github/georgebuilds/anneal)
[![license](https://img.shields.io/badge/license-AGPL3-blue)](LICENSE)

[Visualizer](https://georgebuilds.github.io/anneal/visualizer-demo/) · [Architecture (SPEC)](SPEC.md) · [Design (web)](DESIGN.md) · [Accessibility](web/A11Y.md) · [Limitations](LIMITATIONS.md) · [Contributing](CONTRIBUTING.md)

</div>

---

anneal is a from-scratch Go port of [tinygrad](https://github.com/tinygrad/tinygrad)'s modern, *rangeify-era* core. It takes tensor programs, lowers them through a graph-rewrite compiler, and emits fused GPU kernels. It trains a small MLP, a small convolutional network, a char-level nanoGPT, and a tiny Vision Transformer end-to-end on real GPU hardware via WebGPU, and runs GPT-2-small forward with bit-identical output to HuggingFace's reference implementation.

It is a research project and a learning vehicle, built deliberately in phases. It is not (yet) a drop-in replacement for a production framework — see [Status](#status) for exactly what v1 does and doesn't do.

## What anneal is

Most autodiff libraries record a tape and replay it. anneal doesn't.

- **It's a compiler, not an autodiff library.** Everything (forward ops, gradients, movement ops) is a single immutable IR node (the `UOp`). Computation is suspended until you `Realize()`, at which point the whole program is one graph the compiler can rewrite, schedule, and fuse.
- **Gradients are a rewrite pass.** `Backward()` doesn't build closures; it injects gradient `UOp`s into the *same* graph as the forward pass. The scheduler then fuses kernels across the forward/backward boundary, an optimization that's structurally impossible with a tape.
- **Movement ops are range arithmetic, not copies.** reshape, permute, expand, pad, shrink, and flip never move data. They become index math (the *rangeify* model), and the only thing that ever materializes a buffer is the scheduler.
- **It runs in the browser.** The same compiler builds to WASM and powers the [live visualizer](https://georgebuilds.github.io/anneal/visualizer-demo/), which runs the *real* compiler, not a mock.
- **It imports ONNX, zero-CGO.** `onnx.Import(bytes, arena, device)` parses ONNX 1.17 models via pure-Go protobuf bindings and lowers them onto the same UOp arena as everything else. About 100 op handlers cover the Stage-1 CNN core and the Stage-2 transformer core; symbolic `dim_param` axes ride through as anneal `Variable`s. See [Importing ONNX](#importing-onnx).
- **It has a local browser studio.** `anneal web` serves a single-binary studio at `:3001` with eight deep-linkable views (visualize, kernels, explain, train, generate, history, doctor, plus the home pane). Zero telemetry, zero accounts, model bytes never leave your machine. See [`anneal web`](#anneal-web).

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
anneal train vit            # vision transformer on a synthetic 32x32 RGB classification task
anneal gpt2 sample "Hello"  # forward GPT-2-small from HuggingFace weights, sample text
anneal graph                # dump the UOp graph for a program
anneal kernels              # show the scheduled, fused kernels and their WGSL
anneal explain add          # explain the rewrite/gradient rules for an op
anneal web                  # serve the local studio (deep-linkable views, no telemetry)
```

`anneal doctor` is the right first command: anneal links the platform WebGPU driver at runtime (zero-CGO), so `doctor` tells you whether a usable device is present before anything else.

## `anneal web`

`anneal web [addr]` serves a single-binary studio at `:3001` by default (use `:0` to auto-allocate). The studio is local-only, ships with zero telemetry and zero accounts, and is split along a single load-bearing axis: **every view that compiles runs in-browser as WASM in a Web Worker; every view that executes streams from a native SSE handler.** Eight deep-linkable views land at stable URLs:

| Route | View | Tier |
|---|---|---|
| `/` | studio (device card, model cards, recent runs, ONNX + tensor dropzones) | WASM |
| `/v/<model>` | visualize, with node inspector | WASM |
| `/k/<model>` | kernels (WGSL with fusion boundaries, tokenizer-annotated) | WASM |
| `/x/<op>` | explain (rewrite + gradient rules for one op) | WASM |
| `/t/<model>` | train (live dashboard over SSE, loss sparkline, kernel thumbnail) | native (SSE) |
| `/g/<model>` | generate (token-by-token streaming, click-through to producing kernel) | native (SSE) |
| `/r/<id>` | history (sortable table over `~/.cache/anneal/runs/`, resurrect any run) | native disk + WASM re-render |
| `/d` | doctor (native device card + browser `navigator.gpu` probe, side by side) | native + browser probe |

Drop a `.onnx` file on the studio's home and it imports via the WASM-buildable `onnx.Import(... WithStructureOnly())` path. **Model bytes never reach the server**; the topology is decoded in the Worker, the model card appears next to the bundled examples, and visualize/kernels/explain open as on any registered example. Unsupported ops surface as a first-class panel, not a silent failure.

WCAG 2.x AA is a binding requirement, not an aspiration. The brand tokens carry verified contrast (forward teal `#00ADD8` 7.14:1 on dark surface, ember and gold likewise tabulated), the OS theme listener tracks `prefers-color-scheme` live, every chord shortcut is discoverable via `?`, `prefers-reduced-motion` is honoured, and `forced-colors: active` maps brand tokens to system color keywords. The full per-view checklist lives in [`web/A11Y.md`](web/A11Y.md) and is binding.

The web studio default-writes a run bundle to `~/.cache/anneal/runs/<ts>-<model>-<6hex>/` (`manifest.json`, `schedule.json`, `kernels/*.wgsl`, `graph.json`, `loss.csv`, `generation.ndjson`, `events.ndjson`, `config.json`). `?bundle=0` disables. From the CLI, `anneal train` writes no bundle unless `--bundle` or `ANNEAL_BUNDLE=1` is set: CLI runs stay disk-side-effect-free by default.

GPT-2-small forward fits cleanly on M3 through the generate view: a verified 47s cold-start wall for a 3-token completion and 3.82 GB peak RSS for a 10-token run.

The full set of design invariants (DD1 through DD4: colour is never alone; real compiler only; the library is the product; restraint) lives in [`DESIGN.md`](DESIGN.md).

## Importing ONNX

```go
import "github.com/georgebuilds/anneal/onnx"

arena := uop.NewArena()
r, err := onnx.Import(modelBytes, arena, "webgpu")
// r.Inputs(), r.Outputs(), r.Nodes() expose the lowered graph;
// dim_param symbolic axes ride through as anneal Variables.
out, err := r.Run(inputs)
```

The importer covers ~100 ONNX ops across the Stage-1 CNN core and the Stage-2 transformer core. Two new UOps shipped to close coverage gaps: `OpErf` (Abramowitz-Stegun 7.1.26 polynomial in WGSL, max abs error 1.68e-07 over [-4, 4]) and `OpMin`. Symbolic `dim_param` axes ride through as anneal `Variable`s; the importer reuses the symbolic seam that already shipped on `main`.

The correctness gate is intentionally tighter than onnxruntime goldens. **Strategy A bit-exact gate:** every E2E test builds the model twice on the same arena (direct Tensor API + via the importer) and asserts `[]float32` slice equality (`max-abs-diff = 0`). Arena interning guarantees structurally identical primitive calls share UOp identity. **Strategy B cross-check:** a committed ResNet-9 onnxruntime golden at `onnx/testdata/scripts/` lands at max-abs-diff 8.2e-08, four orders of magnitude inside the 1e-3 tolerance. **Conformance harness:** Phase 4 runs the full ONNX 1.17.0 backend node corpus (234 committed cases, filtered from 1288 upstream). 174 pass, 0 fail, 60 documented skips, worst max-abs-diff 7.324e-04. The skip list is the documented exclusion contract; any case not in it and not passing is a real bug.

For the studio dropzone, pass `onnx.WithStructureOnly()`: the importer creates correctly-shaped, correctly-typed initializer leaves with empty payloads, so the WASM tier can visualize topology without ever materialising weight bytes. Runner.Run fails loudly in that mode.

Documented v1.1 deferrals are enumerated in [LIMITATIONS.md](LIMITATIONS.md) (notably Conv group>1, Resize, control flow, quantization, FLOAT8 and STRING dtypes, Slice |step|>1).

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

For runnable, end-to-end code, including parameter setup, the training loop, optimizer steps, and generation, see [`examples/`](examples): `mlp.go`, `conv.go`, `dynmlp.go`, `nanogpt.go` (char-level transformer training), `vit.go` (vision transformer on a synthetic image-classification task), and `gpt2/` (HF safetensors load + BPE + autoregressive sample). Those are the canonical reference for the current API surface.

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
             SelfAttention (non-causal), MLP, Block, GPT, PatchEmbed, ViTBlock, ViT,
             activations, SGD, Adam, Parameter
cmd/anneal/  the CLI (includes `anneal web` and its SSE handlers)
viz/         the WASM visualizer
web/         studio.html / studio.css / studio.js / worker.js, embedded into the CLI binary; A11Y.md is the binding per-view a11y checklist
onnx/        ONNX importer; onnxpb/ holds the pure-Go protobuf bindings;
             testdata/ holds the 234-case ONNX 1.17.0 conformance corpus
examples/    mlp.go, conv.go, dynmlp.go, nanogpt.go, vit.go, gpt2/
internal/
  assets/    SHA-pinned downloader for Shakespeare corpus and HF GPT-2 weights
  bundle/    on-disk run bundle format (manifest.json + schedule.json + kernels/ + loss.csv etc.)
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
| ONNX import | ✅ `onnx.Import(bytes, arena, device)`, ~100 op handlers, zero-CGO; Strategy A bit-exact gate + Strategy B onnxruntime cross-check; 174/234 conformance pass, 0 fail; `WithStructureOnly()` for WASM dropzone |
| `anneal web` (local studio) | ✅ Single binary, 8 deep-linkable views, WASM/native split, zero telemetry, WCAG 2.x AA |
| Run bundle persistence | ✅ `~/.cache/anneal/runs/<ts>-<model>-<6hex>/`; CLI default OFF (`--bundle` / `ANNEAL_BUNDLE=1`), web default ON (`?bundle=0` disables) |

For the specific shape of each deferral and the platform ceilings behind them (8-buffer-per-kernel WGSL limit, single-adapter WebGPU constraint, non-matmul `OptUpcast`/`OptVectorize`, the WGSL `var<workgroup>` ceiling that gates `OptTile` on symbolic axes, the ONNX v1.1 punt list, the studio's `~20 MB` WASM artifact), see [LIMITATIONS.md](LIMITATIONS.md).

### Hardware compatibility

`anneal train`, `anneal gpt2`, and the `anneal web` train/generate/doctor views all require **native WebGPU** (Metal on macOS, Vulkan on Linux, D3D12 or Vulkan on Windows). `anneal doctor` reports adapter capabilities and whether `shader-f16` is available.

The visualize, kernels, explain, history, and ONNX dropzone views compile to WASM and work in **any modern browser**: importing a model and inspecting its UOp graph + scheduled WGSL needs no GPU at all. The doctor view shows the native side and the browser's `navigator.gpu` probe side by side so the gap is visible.

The original milestone — train a small MLP and a small conv net end-to-end on GPU, with gradients produced by the rewrite pass and kernels fused across the forward/backward boundary — is met. Since then: dynamic-batch training (`dynmlp`, symbolic batch dim), general symbolic axis movement (split/merge a symbolic dim, sym pad/shrink, multi-dim sym dispatch with the symbolic axis in any position on both kernel-output and input buffers), JIT capture/replay, a schedule cache, epilogue fusion (Pass 5 now elides a reduce-output BUFFERIZE into a single downstream elementwise consumer), and BEAM autotuning (env-gated, disk-cached) have all shipped. The remaining deferrals listed above are intentional. Kernel autotuning: `LOCAL` applies to multi-dim symbolic kernels; `TILE` stays unavailable on symbolic axes because WGSL forbids non-const workgroup sizes, a hard platform ceiling; `UPCAST` and `VECTORIZE` are matmul-only by lowerer design (only `emitTiledReduce` handles their per-lane positions); both are now fail-loud at opt-application time when composed without `OptTile`, and BEAM's `ActionSpace` pre-filters them on non-tiled kernels. Symbolic kernels still run correctly via the identity codegen path.

## Contributing

Contributions are welcome, but anneal has a small set of hard invariants (immutable IR, identity equality via interning, no reflection in the rewrite hot path, no copies from movement ops, no SMT solver in indexing) that keep the design coherent. Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** before opening a PR.

## Credits

anneal is largely a port of, and owes its architecture to, [tinygrad](https://github.com/tinygrad/tinygrad) by the tinygrad authors. The reference target is a pinned tinygrad commit (see [CONTRIBUTING.md](CONTRIBUTING.md)); blog-era LazyBuffer/Linearizer descriptions of tinygrad do *not* describe this design.

GPU access is via [`gogpu/wgpu`](https://github.com/gogpu/wgpu) and `goffi` (zero-CGO).

## License
