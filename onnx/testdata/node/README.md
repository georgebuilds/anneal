# ONNX Backend Node Corpus (Phase 4)

This directory is a **curated subset** of the ONNX 1.17.0 backend node tests
shipped in `onnx.backend.test.data.node` from the upstream `onnx` Python
package. It is the on-disk corpus consumed by
`onnx/conformance_test.go::TestConformance_NodeCorpus`.

Phase 4 uses Strategy B (precomputed `.pb` goldens shipped with the
upstream package) — **no Python or onnxruntime is invoked at test
time**. The harness loads `model.onnx`, feeds the `input_*.pb`
TensorProtos through the anneal importer, and asserts the max-abs-diff
against the corresponding `output_*.pb` is below 1e-3.

## How to regenerate

```
python3 -m venv /tmp/onnx_phase4_venv
/tmp/onnx_phase4_venv/bin/pip install onnx==1.17.0 numpy
/tmp/onnx_phase4_venv/bin/python notes/scripts/copy_node_corpus.py
```

The pin to onnx 1.17.0 matches tinygrad's reference (per
`notes/onnx_implementation_plan.md` §7).

## What's in scope

Anything whose test directory base name (after stripping the `test_`
prefix) matches one of `SUPPORTED_PREFIXES` in
`copy_node_corpus.py`. The prefixes mirror the device-tier handlers
in `onnx/handlers.go` and the host-tier handlers in `onnx/host_ops.go`.

## What's excluded at copy time

Hard-coded `HARD_EXCLUDE_TOKENS` in `copy_node_corpus.py`: quantized
ops (QLinear/Quantize/Dequantize), string/sequence/float8/int4/uint4
dtypes, training-mode variants, control flow (handled by not matching
any prefix), data-dependent shapes, Scatter / GatherElements / GatherND,
Resize / Upsample, ConvTranspose / DeformConv, MaxUnpool / argmax pool,
CastLike, the `_expanded` decomposed variants (they re-prove the
decomposed elementwise paths the bit-exact e2e tests already cover).

The exclusion is coarse-grained to keep the corpus small. The
fine-grained per-case skip list with reasons (citing the plan section)
lives in `onnx/conformance_skip.go` and is consumed by the conformance
test — that file is the documented exclusion contract.

## Stats from the most recent copy

- Upstream corpus: 1288 test_* directories
- Kept: 234 (in this directory)
- Dropped (hard exclude):    459
- Dropped (unsupported op):  595
- Total committed bytes: 1.10 MB
