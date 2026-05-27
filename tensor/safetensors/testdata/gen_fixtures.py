#!/usr/bin/env python3
"""Generate safetensors fixture files for tensor/safetensors round-trip tests.

Run from anywhere:
    python3 path/to/tensor/safetensors/testdata/gen_fixtures.py

Requires: safetensors, numpy
    pip install safetensors numpy
"""

import os
import struct
import numpy as np
from safetensors.numpy import save_file

here = os.path.dirname(os.path.abspath(__file__))


def save(name, tensors, metadata=None):
    path = os.path.join(here, name)
    kw = {}
    if metadata:
        kw["metadata"] = metadata
    save_file(tensors, path, **kw)
    for k, v in tensors.items():
        print(f"  {name}  [{k}]: shape={v.shape} dtype={v.dtype}  "
              f"values={v.flatten()[:6]}")


print("Generating safetensors fixtures in:", here)

# --- F32: 3×4 arange ----------------------------------------------------------
# Expected float32 values: 0..11.
save("f32_3x4.safetensors", {
    "w": np.arange(12, dtype=np.float32).reshape(3, 4),
})

# --- F64: 1D, exact values (powers of 2 so float32 cast is lossless) ----------
# Expected after float32 cast: [0.0, 1.0, 2.0, 100.0].
save("f64_exact.safetensors", {
    "v": np.array([0.0, 1.0, 2.0, 100.0], dtype=np.float64),
})

# --- Multi-dtype in one file --------------------------------------------------
# All values chosen to be exact in float32 after conversion.
save("multi_dtype.safetensors", {
    "i8":   np.array([-5, 0, 10, 127], dtype=np.int8),
    "u8":   np.array([0, 1, 255], dtype=np.uint8),
    "i16":  np.array([-32768, 0, 32767], dtype=np.int16),
    "i32":  np.array([1, 2, 3], dtype=np.int32),
    "i64":  np.array([10, 20, 30], dtype=np.int64),
    "bool": np.array([True, False, True], dtype=np.bool_),
    "f32":  np.array([1.0, 2.0, 3.0], dtype=np.float32),
})

# --- With __metadata__ block --------------------------------------------------
save("with_metadata.safetensors", {
    "weights": np.arange(6, dtype=np.float32).reshape(2, 3),
}, metadata={"model": "test-v1", "framework": "anneal"})

# --- F16 (IEEE-754 half precision) -------------------------------------------
# Use values exactly representable in F16: powers of 2.
save("f16.safetensors", {
    "h": np.array([1.0, 2.0, 0.5, 4.0], dtype=np.float16),
})

# --- BF16 (brain float16) ---------------------------------------------------
# numpy does not have bfloat16; build the raw safetensors bytes manually.
# BF16 = upper 16 bits of float32 IEEE-754 representation.
# Values: 1.0 = 0x3F80, -1.0 = 0xBF80, 0.5 = 0x3F00, 2.0 = 0x4000
def write_bf16_fixture(path, name, f32_values):
    """Write a minimal safetensors file containing one BF16 tensor."""
    import json, struct
    # BF16 bits = upper 16 bits of float32
    bf16_bytes = b""
    for v in f32_values:
        bits = struct.unpack(">I", struct.pack(">f", v))[0]
        bf16_bits = (bits >> 16) & 0xFFFF
        bf16_bytes += struct.pack("<H", bf16_bits)  # LE

    shape = [len(f32_values)]
    n_bytes = len(bf16_bytes)
    header = json.dumps({name: {"dtype": "BF16", "shape": shape,
                                "data_offsets": [0, n_bytes]}}).encode()
    # Pad header to multiple of 8
    extra = (8 - len(header) % 8) % 8
    header += b" " * extra
    hdr_len = struct.pack("<Q", len(header))
    with open(path, "wb") as fh:
        fh.write(hdr_len + header + bf16_bytes)
    print(f"  {os.path.basename(path)}  [{name}]: shape={shape} dtype=BF16  "
          f"values={f32_values}")

write_bf16_fixture(os.path.join(here, "bf16.safetensors"), "b",
                   [1.0, -1.0, 0.5, 2.0])

# --- I64 out-of-range (contains 2^24+1 = 16,777,217) -------------------------
# Loading should produce a clear error.
save("i64_oor.safetensors", {
    "v": np.array([1, 2, 16_777_217], dtype=np.int64),
})

# --- NPZ-style multi-tensor file for NPZ-like usage --------------------------
save("multi_tensor.safetensors", {
    "weights": np.arange(6, dtype=np.float32).reshape(2, 3),
    "bias":    np.array([0.0, 0.5, 1.0], dtype=np.float32),
    "mask":    np.array([True, False, True], dtype=np.bool_),
})

print("\nDone. All fixtures written.")
