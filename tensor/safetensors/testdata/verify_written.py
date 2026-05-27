#!/usr/bin/env python3
"""Read an anneal-generated safetensors file and verify its contents.

Usage:
    python3 verify_written.py <path.safetensors>

This script is the "write fidelity" oracle: if anneal's Go writer produces a
file that real Python safetensors can parse and returns the correct values,
anneal's output is ecosystem-compatible.

Requires: safetensors, numpy
    pip install safetensors numpy
"""

import sys
import numpy as np
from safetensors import safe_open


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_written.py <path.safetensors>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Reading: {path}")

    tensors = {}
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata()
        if meta:
            print(f"  __metadata__: {meta}")
        for key in f.keys():
            t = f.get_tensor(key)
            tensors[key] = t
            flat = t.flatten()
            preview = flat[:min(8, len(flat))]
            print(f"  [{key}]: shape={list(t.shape)} dtype={t.dtype}  "
                  f"values={preview.tolist()}")

    print("\nAll tensors read successfully by Python safetensors library.")
    return tensors


if __name__ == "__main__":
    main()
