# Limitations

The line between what anneal does and what it doesn't is intentional. This page is the v1 user's reference for the real boundaries: things you will trip on if you try them, even though the surrounding capability looks like it should cover the case. Architectural rationale lives in [SPEC.md](SPEC.md); the table in [README.md](README.md#status) is the high-level capability map. This page is the gap list.

No item here is a bug. Each is either a deliberate scope cut, a platform ceiling that anneal cannot remove unilaterally, or a piece of carried debt with a known shape and a known cost.

## Platform: WebGPU

anneal's only shipped backend is WebGPU (native via wgpu, browser via WASM). Several v1 limits are downstream of that choice, not of anneal's code.

### Single adapter per device

WebGPU's `requestAdapter` returns one adapter. There is no in-spec way to address two GPUs from the same WebGPU context. Practical consequence: multi-device is not available in v1, in browser or native. A future CUDA backend retires this constraint (the CUDA driver API addresses devices directly); WebGPU itself never will, at least without spec-level changes.

If you have two discrete GPUs on a Linux or Windows host, anneal cannot use both at once today. On a single-GPU Mac, multi-device was never a meaningful target.

### 8 storage buffers per compute kernel

WGSL compute shaders bind at most 8 storage buffers (the WebGPU standard limit, not Apple-specific). Inputs plus outputs plus any constant or uniform buffers all count. anneal defends this limit at three points:

1. The scheduler's epilogue-fusion pass refuses to elide a removable bufferize if the resulting fused kernel would exceed 8 buffers.
2. The tile-loading opt rejects a tile fuse that would bust the budget at opt-application time.
3. Specific gradient rules (notably `ReduceAxis(OpMax)` backward, used by `MaxPool2D`) insert a materialization barrier so that deep adjoint chains do not inline their leaf buffers into the consuming kernel.

If you build a graph whose worst-case kernel exceeds 8 buffers in a path none of the defenses cover, you will see a `CreateBindGroupLayout` or `CreateShaderModule` failure at realize time, not at codegen time. The failure is intermittent across small graph changes because scheduling decides whether the problematic kernel actually gets fused on a given run. See [SPEC §10](SPEC.md) for the invariant and the three workaround patterns.

### WGSL is the only renderer

Codegen targets WGSL exclusively. CUDA and Metal-direct paths are deferred (CUDA is on the roadmap; Metal-direct is not). Anneal links the platform's WebGPU driver at runtime (zero-CGO), so on macOS you are running through Apple's WebGPU implementation, not Metal directly.

### No `atomic<f32>` in WGSL

WGSL does not support floating-point atomics (`gpuweb#4894` was still in the proposal stage as of mid-2026). This is why anneal's scatter-add backward (used by `Embedding` gradient) routes through a host-side sort plus a deterministic segment-sum kernel rather than an atomic accumulation. The current path is correct and deterministic; a future backend with float atomics could trade determinism for fewer host-device round trips.

## Performance ceilings

### WGSL non-const `var<workgroup>` rules out `OptTile` on symbolic axes

`OptTile` declares a `var<workgroup>` tile buffer whose size is part of the kernel SINK at codegen time. WGSL requires `var<workgroup>` sizes to be compile-time constants. A symbolic axis is, by construction, not compile-time constant inside one compiled kernel. So `OptTile` bails on a symbolic axis. `OptLocal` does apply to multi-dim symbolic kernels (the `targetDim=0` collapse landed in the symbolic Slice 7d work); `OptTile` does not, and will not on WGSL.

This is a hard platform ceiling, not a missing slice. A CUDA backend retires it because PTX permits dynamic shared-memory sizes.

### `OptUpcast` and `OptVectorize` bail outside matmul shapes

In v1, `OptUpcast` and `OptVectorize` are only wired through the tiled-reduce lowerer (the matmul shape). On a non-matmul kernel, an `OptUpcast` would lower outside `emitTiledReduce` and produce an `exprOf[AxisUpcast]="0"` substitution, which causes each thread to write only lane 0 of its `factor`-wide stripe and silently drop the other `factor - 1` lanes.

Both `OptUpcast` and `OptVectorize` are now **fail-loud at opt-application time**: `applyUpcast` and `applyVectorize` panic with a diagnostic when targeting a kernel whose body lacks an `OpReduce` tagged by `OptTile` (the only path that activates `emitTiledReduce`). Compose `OptTile` before either opt, or skip them on the kernel. Use `codegen.KernelHasTiledReduce(sink)` to check eligibility ahead of time when applying opts across a mixed kernel list. BEAM's `ActionSpace` uses the same predicate to pre-filter both kinds, so the search never reaches either assertion; BEAM's existing value-identity guard remains as belt-and-suspenders for any other wrong-output candidate.

### Per-thread scalar f32 throughput on M3

Large compute-bound matmul (1024 cubed and above) on an M3 saturates at roughly 85 GFLOP/s, flat across identity, `OptLocal`, `OptTile`, `OptTile + OptUpcast`, and the full `OptTile + OptUpcast + OptVectorize` stack. A four-experiment diagnostic falsified the three obvious candidate ceilings (uncoalesced loads, occupancy, L2 bandwidth). The binding cost is per-thread scalar f32 load and FMA throughput in WGSL on M3, independent of cache topology, workgroup count, tile size, or register-block factor.

Two future levers (typed `array<vec4<f32>>` buffer bindings; WGSL subgroup-matrix ops) would in principle change the picture; both are out of v1 scope. Small or dispatch-bound kernels do see BEAM wins (an isolated 1x1x8x8 conv with 3x3 kernel goes 489us to 195us, a 2.5x speedup); the ceiling is specific to large compute-bound matmul.

## Capability deferrals

These are tracked as deferred in [README's Status table](README.md#status); they are listed here with the specific shape of the gap so you can tell whether your workload hits them.

### Dynamic seq-length tensor API

Symbolic shapes are fully shipped (Option A dynamic batch, Option B general axis movement including split or merge across a symbolic axis, symbolic pad and shrink amounts, multi-dim symbolic dispatch with the symbolic axis in any position, cross-arena structural-key portability). What is not yet shipped is the general-purpose tensor-surface constructor that admits a non-outermost symbolic axis as an input shape. The current `NewSymbolicBatchInput` constructor places the symbolic axis at position 0. The internal machinery can express the more general case; the input ergonomics catch up next.

### Multi-device

Deferred regardless of backend. Even when CUDA lands, the v1 cut is copy-op-only first, then data parallelism; tensor parallelism is further deferred.

### Image dtypes

Deferred. anneal's dtype set is `float32`, `int32`, `bool`, `f16` (via the `shader-f16` WGSL extension), and `bf16` (storage-only with f32 compute). Image dtypes (the tinygrad concept of `imagef32` and related) are not in v1.

### fp8

Deferred. f16 and bf16 are the smallest dtypes in v1.

### f16 requires device support and fails closed

f16 requires the WebGPU `shader-f16` device feature. If a device does not advertise it, anneal fails before any GPU allocation rather than silently falling back to f32. `anneal doctor` reports which features your device supports. bf16 has no such requirement (storage is `array<u32>`, compute runs in f32) and runs on any WebGPU adapter.

### JIT and schedule cache are single-arena

The JIT capture and the in-process schedule cache are both keyed within a single arena. Cross-arena structural-key portability is shipped for symbolic graphs (so the persistent BEAM disk cache survives arena churn), but the in-process JIT path itself does not cross arenas. In practice this means JIT replay is bounded to the lifetime of the arena it captured against.

## Browser specifics

anneal's WASM build runs in any WebGPU-enabled browser. Two extra constraints apply there:

- The browser owns adapter selection; anneal cannot pick a specific GPU.
- Storage buffer limits in browser implementations can be lower than the WebGPU spec floor. GPT-2-small in particular touches a buffer size limit on some browsers; CI skips GPT-2 scale tests when the device's `maxStorageBufferBindingSize` is below the kernel's requirement.

## Determinism notes

anneal aims for run-to-run bit identity. Two places this is not free:

- BEAM autotuning runs only on `ANNEAL_BEAM=1`. The default path is a pure lookup against the persistent disk cache; on a cache miss it returns identity. So bit identity holds across runs as long as the cache is shared and the WGSL hash matches.
- The scatter-add backward uses a host-side sort plus segment-sum specifically so that the same `(indices, grad_out)` input produces the same `dW` output every run, even though distinct segments dispatch in parallel.

Anything else that should hold bit-identical (the schedule cache, structural keys, `normalizeWGSL` placeholder substitution) is covered by load-bearing invariants in [SPEC §10](SPEC.md). Violations of those have shown up as recurring bug-class incidents during development and are guarded accordingly.
