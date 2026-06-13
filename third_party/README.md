# third_party

## naga

A build-only copy (non-test Go source) of [gogpu/naga](https://github.com/gogpu/naga)
v0.17.13 carrying one fix: the MSL backend emits WGSL `var<workgroup>`
variables as function-body-scope `threadgroup` declarations instead of
threadgroup entry-point parameters. The gogpu/wgpu Metal HAL never calls
`setThreadgroupMemoryLength:atIndex:` for threadgroup parameters, so without
this fix all workgroup-memory reads silently return zero on Metal (every
tiled kernel computes wrong results). See `naga/PATCH_NOTES.md` for the full
analysis and verification.

Wired in via `replace github.com/gogpu/naga => ./third_party/naga` in the
root `go.mod`. Temporary: an upstream PR to gogpu/naga is planned; when it
merges and a release is tagged, delete this directory and the replace
directive.
