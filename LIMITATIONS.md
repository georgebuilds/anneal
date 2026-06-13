# Limitations

The line between what anneal does and what it doesn't is intentional. This page is the v1 user's reference for the real boundaries: things you will trip on if you try them, even though the surrounding capability looks like it should cover the case. Architectural rationale lives in [SPEC.md](SPEC.md); the table in [README.md](README.md#status) is the high-level capability map. This page is the gap list.

No item here is a bug. Each is either a deliberate scope cut, a platform ceiling that anneal cannot remove unilaterally, or a piece of carried debt with a known shape and a known cost.

## Platform: WebGPU

The shipped backend remains WebGPU (native via wgpu, browser via WASM). The `Renderer`, `Compiler`, `Allocator`, `Program`, and `DeviceBuffer` interfaces in `backend/` define the contract a CUDA or Metal-direct backend would satisfy; none has shipped yet, so every v1 limit below is the WebGPU shape of the world.

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

### Multi-device

Deferred regardless of backend. Even when CUDA lands, the v1 cut is copy-op-only first, then data parallelism; tensor parallelism is further deferred.

### Image dtypes

Shipped as a storage-layout sibling. `Dtypes.ImageFloat32` packs four logical f32 elements into one WGSL `vec4<f32>` storage slot (binding type `array<vec4<f32>>`); compute stays scalar f32, gradients and autodiff are unchanged, and the promotion lattice treats image as a storage-only variant of `Float32` (`LeastUpperDType(ImageFloat32, Float32) == Float32`). Static image-output kernels lower via a deterministic vec4 slot dispatch (one thread per output slot, four logical outputs per thread, whole slot written once), which removed the former carried constraint: output row strides that are not a multiple of 4 (e.g. matmul N=30, N=17, M=17) are bit-exact and covered by the value oracle. Remaining scope limits: SYMBOLIC image-output kernels keep the legacy per-lane store cascade and therefore the old aligned-row-stride constraint (unaligned symbolic image strides were never supported); image-output kernels are excluded from the BEAM action space (the slot dispatch is keyed in the lowerer, and every Opt would reshape the kernel back onto the cascade — hand-applying an Opt to an image-output kernel panics at Lower time, fail-loud by design).

### fp8

Shipped as storage-only dtypes with f32 compute, on the bf16 decoded-storage scheme. `Dtypes.FP8E4M3` (OCP e4m3fn: bias 7, no infinities, max finite ±448) and `Dtypes.FP8E5M2` (IEEE-style: bias 15, ±Inf/NaN, max finite ±57344) store the fp8-quantized value's full f32 bit pattern per u32 slot, so loads are a free `bitcast<f32>`, reduce accumulators stay f32, and the RTNE narrowing happens once at the store boundary (`_fp8e4m3_rtne_bits` / `_fp8e5m2_rtne_bits`, mirroring `uop.Float32ToFP8E4M3/E5M2` bit for bit — GPU-vs-host storage comparisons are exact, not tolerance-based). e4m3fn conversion uses the CUDA satfinite convention (finite overflow and ±Inf saturate to ±448); e5m2 overflow rounds to ±Inf. No device feature is required; fp8 runs on any WebGPU adapter. Carried scope limits: 4 bytes/elem storage (precision semantics, not memory savings — a packed 4-per-u32 layout would hit the WGSL read-modify-write store race, the same class as the image-dtype stride constraint above); mid-kernel `Cast` to fp8 computes in f32 without re-quantizing (quantization is a storage-boundary property, same as bf16); ONNX FLOAT8 wire decode stays on the v1.1 punt list; fp8 compute / scaled-matmul recipes (loss scaling, amax tracking) are out of scope.

### f16 requires device support and fails closed

f16 requires the WebGPU `shader-f16` device feature. If a device does not advertise it, anneal fails before any GPU allocation rather than silently falling back to f32. `anneal doctor` reports which features your device supports. bf16 and fp8 have no such requirement (storage is `array<u32>`, compute runs in f32) and run on any WebGPU adapter.

### JIT and schedule cache are single-arena

The JIT capture and the in-process schedule cache are both keyed within a single arena. Cross-arena structural-key portability is shipped for symbolic graphs (so the persistent BEAM disk cache survives arena churn), but the in-process JIT path itself does not cross arenas. In practice this means JIT replay is bounded to the lifetime of the arena it captured against.

## Browser specifics

anneal's WASM build runs in any WebGPU-enabled browser. Two extra constraints apply there:

- The browser owns adapter selection; anneal cannot pick a specific GPU.
- Storage buffer limits in browser implementations can be lower than the WebGPU spec floor. GPT-2-small in particular touches a buffer size limit on some browsers; CI skips GPT-2 scale tests when the device's `maxStorageBufferBindingSize` is below the kernel's requirement.

## ONNX importer (v1.1 punt list)

The importer covers the Stage-1 CNN and Stage-2 transformer cores end-to-end (`onnx.Import(bytes, arena, device)`), with the Phase 4 conformance harness at 174/234 passing (0 failures, 60 documented skips, worst max-abs-diff 7.324e-04 against a 1e-3 tolerance). The skip list in `onnx/conformance_skip.go` is the documented exclusion contract; the entries below correspond to those reasons and are all targeted for v1.1.

### Op surface deferrals

- **Conv group>1.** Depthwise / grouped convolution is the MobileNetV2 v1.1 blocker. Adding it touches `tensor/nn.Conv2d`'s im2col layout. ResNet-9 and ResNet-50 bottleneck shapes work; MobileNetV2 rejects loudly.
- **Conv 3-D pads (`pads` attr length 6) and `auto_pad`.** Both need a pad-axis rebroadcast in the handler. None of the v1 targets use them.
- **Resize and Upsample.** Deferred to v1.1 as the nastiest common op; no v1 surface relies on them.
- **NonZero, Unique.** Data-dependent shapes; deferred.
- **Control flow (`If`, `Loop`, `Scan`).** All three deferred to v1.1.
- **Scatter family (`Scatter`, `ScatterND`, `ScatterElements`) and `GatherElements`, `GatherND`.** Deferred to v1.1 (the `ScatterND` graph-explosion trap is documented in the plan).
- **Quantization (`QLinearMatMul`, `QLinearConv`, `QuantizeLinear`, `DequantizeLinear`).** Out of v1 scope.
- **`Sequence*`, `Map*`, `Optional*` containers.** Out of v1 scope.
- **MaxPool variants.** `ceil_mode=1`, `dilations>1`, `auto_pad`, `storage_order`, 1-D, 3-D, uint8 inputs, explicit `pads` (anneal zero-pads then pools, ONNX pads with -inf semantically), and the 5x5/stride-3 output-trim path are all punted; the v1 surface is MaxPool2D with the canonical kernel/stride/padding configuration.
- **Slice with `|step|>1`.** `step=-1` is supported (flips the axis and shrinks); larger absolute steps reject loudly.
- **`Pow` with integer-typed bases or exponents.** Handler is f32-only.
- **uint8 elementwise (`Add`, `Sub`, `Mul`, `Div`) and Clip int8.** Handlers are float-typed; integer overflow wrap is not modelled.
- **Reduce empty-set semantics and `noop_with_empty_axes`.** Identity-element behaviour for empty reductions is not pinned for v1.
- **`Shape`, `Size`, `Range` as graph outputs.** Host-tier int vectors cannot be returned as graph outputs in v1; real models use them as intermediate values feeding `Reshape` / `Slice`, not as terminal outputs. Lifting a host int vector to a device int32 tensor at materialise time is the v1.1 follow-up.
- **`Shape` start/end attributes (opset 15+).** Handler returns the full shape; the start/end slice is a clean follow-up.

### Dtype deferrals

- **STRING, COMPLEX, FLOAT8 (`E4M3FN`/`E5M2`/`*FNUZ`), INT4, UINT4.** Out of v1 scope; no WGSL lowering exists.
- **BFLOAT16 wire-encoded as UINT16.** ONNX 1.17 stores BFLOAT16 payloads under `TensorProto.data_type=UINT16` in the `.pb` fixtures while the model declares BFLOAT16. The pb decoder honours the on-disk dtype and reads them as uint16 integers (correct per protobuf, off vs. the model's bf16 semantic). Real anneal pipelines use `SetData([]float32)` directly, so this only affects the conformance corpus. Handler-level dtype coercion against the graph input declaration is v1.1.

### Concat along a symbolic axis (Phase 3 carry)

`tensor.Concat` composes `Pad` and `Add`. With a symbolic axis, the structural `Sint` canonicalisation under `tensor/movement.go::PadSints` doesn't fold cleanly. Neither GPT-2 nor BERT inference graphs hit this (they sum token + position embeddings elementwise rather than concatenating across the symbolic seq/batch dim), and no CNN path requires it either. Targeted for v1.1.

## `anneal web` limitations

- **Single-adapter native, same as the base.** The studio inherits the WebGPU single-adapter constraint; the doctor view surfaces the active adapter only.
- **WebGPU required for `train` / `generate` / `doctor`.** The native SSE views need a working WebGPU adapter exactly the same way `anneal train` does. The browser-card half of the doctor view requires the browser to expose `navigator.gpu`, which not every modern browser does yet.
- **WASM artifact size.** `web/anneal.wasm` is currently around 20 MB after W8 (the ONNX importer adds the protobuf bindings to the bridge). The studio loads it lazily off the main thread, but the first-paint cost on a slow connection is real. No tree-shaking pass yet.
- **studio.js module split.** `web/studio.js` is around 100 KB. The current per-file budget (set at W6) is HTML 32 KB, CSS 48 KB, JS 70 KB, worker 8 KB; studio.js already exceeded the JS budget at W8 and is flagged for an ES-module split before the next major view lands.
- **No telemetry, no auth, no hosted mode.** By design, not a limitation, but called out so it isn't misread as a missing feature.

## Determinism notes

anneal aims for run-to-run bit identity. Two places this is not free:

- BEAM autotuning runs only on `ANNEAL_BEAM=1`. The default path is a pure lookup against the persistent disk cache; on a cache miss it returns identity. So bit identity holds across runs as long as the cache is shared and the WGSL hash matches.
- The scatter-add backward uses a host-side sort plus segment-sum specifically so that the same `(indices, grad_out)` input produces the same `dW` output every run, even though distinct segments dispatch in parallel.

Anything else that should hold bit-identical (the schedule cache, structural keys, `normalizeWGSL` placeholder substitution) is covered by load-bearing invariants in [SPEC §10](SPEC.md). Violations of those have shown up as recurring bug-class incidents during development and are guarded accordingly.
