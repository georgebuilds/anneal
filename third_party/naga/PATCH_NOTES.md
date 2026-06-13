# naga MSL workgroup-memory fix (smem-fix branch)

Patch for anneal's WebGPU stack: WGSL `var<workgroup>` memory was silently
non-functional on Metal through the pure-Go gogpu stack (all reads 0.0, writes
dropped, no error).

## Base

- Upstream: https://github.com/gogpu/naga
- Base commit: `17522fbc2e959ded51c135e46740a5d35ab24fc0` (tag `v0.17.13`)
- Verified: `diff -rq` of the checkout vs `~/go/pkg/mod/github.com/gogpu/naga@v0.17.13/`
  is empty — the base matches exactly what anneal builds against.
- Fix branch: `smem-fix` (commit `5e9899f`), local only — never pushed.

## Root cause

`msl/internal/codegen/functions.go` emitted WGSL `var<workgroup>` globals as
Metal **entry-point parameters** in the threadgroup address space:

```
kernel void main_(..., threadgroup type_2& sm, ...)
```

This mirrors Rust naga — but Rust wgpu-hal's Metal encoder compensates by
calling `setThreadgroupMemoryLength:atIndex:` for each threadgroup parameter
before dispatch. The pure-Go `gogpu/wgpu` Metal HAL never makes that call
(`hal/metal/encoder.go` `Dispatch`, ~line 694 — no such msgSend anywhere in
`hal/metal/`). Metal then treats the parameter as unsized/unbacked: reads
return 0, writes are dropped, and no API error is raised.

## Fix (naga-only; gogpu/wgpu untouched)

Declare workgroup variables at **function-body scope** inside the kernel
entry point instead of as parameters:

```
kernel void main_(...) {
    threadgroup type_2 sm;
    ...
}
```

Function-scope threadgroup declarations are legal MSL (MSL spec §4.4, the
threadgroup address space is allowed for variables declared inside kernel
functions) and require **no host-side `setThreadgroupMemoryLength` call** —
the compiler sizes the memory statically. Names are unchanged, so all body
references and helper-function call sites resolve exactly as before.

### Files changed (vs v0.17.13, 15 files, +81/−39)

1. `msl/internal/codegen/functions.go` (the only production change):
   - `writeEntryPointFunction`: the `SpaceWorkGroup` branch of the
     global-parameter loop no longer emits a parameter; instead, used
     workgroup vars are declared at the top of the kernel body
     (`threadgroup <type> <name>;`), immediately before the zero-init
     prologue (which references them).
   - `epUsedGlobals` computation hoisted earlier so it can gate both the
     zero-init decision and the declarations.
   - `needWorkgroupInit` + `writeWorkgroupZeroInit` now filter by
     per-entry-point usage (matching Rust naga's
     `!fun_info[handle].is_empty()`). This also fixes a latent bug: a
     workgroup var NOT used by an entry point previously produced a
     zero-init statement referencing an undeclared parameter name
     (invalid MSL).
2. `snapshot/snapshot_test.go`: 6 shaders added to `referenceAllowList`
   under a new documented reason `msl-threadgroup-function-scope`
   (abstract-types-operators, globals, interface,
   overrides-atomicCompareExchangeWeak, policy-mix, workgroup-uniform-load).
3. `snapshot/testdata/golden/msl/*.msl`: 13 golden files regenerated via
   `UPDATE_GOLDEN=1` — every diff is exactly "parameter removed, body-scope
   declaration added"; no other text changes.

### What is unchanged

- Helper (non-entry) functions still receive `threadgroup T& name`
  pass-through parameters (`writePassThroughParam` untouched); the entry
  point passes its body-scope variable by the same name. Verified in the
  `abstract-types-operators` golden, which covers the transitive case (a
  workgroup var used *only* inside a called helper).
- SPIR-V, HLSL, GLSL, DXIL backends: zero changes (diff touches only
  `msl/internal/codegen/functions.go` + snapshot test data). Vulkan path
  unaffected.
- Kernels without workgroup vars: byte-identical MSL (verified below).
- Mesh/task entry points (`mesh-shader` golden) go through a different
  emission path and are byte-identical to baseline.

## Zero-init decision

WGSL requires workgroup variables to be zero-initialized; MSL threadgroup
memory is NOT zero-initialized by hardware. The Go port **already** had the
Rust-naga-equivalent zero-init prologue
(`if (metal::all(__local_invocation_id == metal::uint3(0u))) { var = {}; ... }`
followed by `threadgroup_barrier(mem_flags::mem_threadgroup)`, with
atomic_store_explicit handling for atomics), gated on
`Options.ZeroInitializeWorkgroupMemory`. `msl.DefaultOptions()` sets it to
`true` and `gogpu/wgpu`'s Metal HAL compiles with `msl.DefaultOptions()`
(`hal/metal/device.go:489`), so zero-init is active in anneal's pipeline.
It was previously writing into the broken unsized memory (no-op); the fix
makes it effective. No new zero-init code was needed — only the per-EP usage
filter to match Rust semantics. Empirically confirmed by Probe D below.

## Test results

- Baseline (v0.17.13, unpatched): `go test ./...` → **9547 passed, 33
  packages, 0 failures**.
- Patched: **9547 passed, 33 packages, 0 failures** (after golden
  regeneration + allow-list). Rust-reference summary on MSL moved from
  "83 pass / 3 allow / 6 fail" mid-patch to "pass + 9 allow-listed, 0 fail"
  in the final run; the 6 new allow entries are the deliberate divergence.

## Empirical GPU verification (M3, Metal, gogpu/wgpu v0.29.1 + patched naga)

Probe module: `/tmp/sgprobe` with
`replace github.com/gogpu/naga => /Users/george/Code/naga-smem-fix`.
Probes: `/tmp/sgprobe/smemverify/main.go` (output buffers poisoned with
−999 so stale zeros can't masquerade as results), plus the original
`/tmp/sgprobe/lidprobe`.

| Probe | Stock v0.17.13 | Patched |
|---|---|---|
| A: smem passthrough, 64 lanes write lid+1 / barrier / read neighbor | 0/64 correct (all 0.0) | **64/64 correct** (out=[2,3,...,64,1]) |
| B: tiled matmul 32×32×32, two 8×8 `var<workgroup>` tiles, vs Go CPU reference | max-abs-diff 6.602e+00 (all zeros) | **max-abs-diff 0.000e+00** (bit-exact) |
| C: TWO workgroup vars (array<f32,32> + array<u32,32>), cross-lane reads | 0/32 correct | **32/32 correct** |
| D: zero-init — read smem before any write, expect 0.0 | 64/64 (vacuous: broken memory reads as 0) | **64/64 read 0.0** (real memory, prologue works) |
| No-workgroup kernel (storage+uniform+helper fn) MSL stock vs patched | — | **byte-identical** |

The patched MSL for probe A was also inspected by eye: parameter list loses
`threadgroup type_2& sm`, body gains `threadgroup type_2 sm;` before the
zero-init prologue; nothing else changes.

## Doubts / edge cases / follow-ups

- **Deliberate divergence from Rust upstream**: Rust naga emits threadgroup
  parameters and relies on its HAL making the
  `setThreadgroupMemoryLength:atIndex:` call. The "fully upstream-faithful"
  alternative would be a gogpu/wgpu HAL fix (reflect threadgroup sizes from
  the compile, call the msgSend per dispatch). The function-scope approach
  was chosen per brief: single-repo patch, legal MSL, no host plumbing.
  If gogpu upstream prefers HAL parity instead, this patch is still safe to
  carry — the two approaches cannot conflict (after this patch there are no
  threadgroup parameters left to size).
- **Threadgroup memory budget**: function-scope declarations are statically
  sized into the pipeline; same 32 KB/threadgroup Apple-GPU budget as
  parameter-based memory. `maxTotalThreadgroupMemory` validation behavior is
  identical.
- **Override-sized workgroup arrays**: sizes are resolved by the
  pipeline-constants pass before codegen, so the body-scope declaration is
  always statically sized. The `overrides-atomicCompareExchangeWeak` golden
  covers this.
- **Non-compute stages**: WGSL validation forbids `var<workgroup>` outside
  compute; the declaration loop keys off usage, not stage, same as the old
  parameter loop. Mesh/task path untouched.
- **xcrun metal unavailable on this machine** (no full Xcode), so no offline
  MSL compile check — but every probe creates a real `MTLComputePipelineState`
  at runtime, which compiles the MSL through the Metal framework; pipeline
  creation succeeded for all probes, which is the stronger check.
- **gogpu/wgpu v0.29.1 unchanged** — anneal's go.mod only needs a `replace`
  for naga (orchestrator wires this).
