#!/usr/bin/env python3
"""Generate NumPy fixture files for the tensor/npy round-trip tests.

Run from anywhere:
    python3 path/to/tensor/npy/testdata/gen_fixtures.py

Requires: numpy (pip install numpy)
"""

import os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def save(name, arr):
    path = os.path.join(here, name)
    np.save(path, arr)
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")


print("Generating fixtures in:", here)

# --- float32: 3×4 arange -------------------------------------------------------
# Expected data: [0, 1, 2, ..., 11] as float32.
save("float32_3x4.npy", np.arange(12, dtype=np.float32).reshape(3, 4))

# --- float64: 1D, exact values (powers of 2 / integers so float32 cast is lossless)
# Expected data after float32 cast: [0.0, 1.0, 2.0, 100.0]
save("float64_exact.npy", np.array([0.0, 1.0, 2.0, 100.0], dtype=np.float64))

# --- int32: 2×3 matrix ---------------------------------------------------------
# Expected data: [1, 2, 3, 4, 5, 6] as float32 by value.
save("int32_2x3.npy", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

# --- int8: 1D with negative values ---------------------------------------------
# Expected data: [-5, 0, 10, 127] as float32 by value.
save("int8_1d.npy", np.array([-5, 0, 10, 127], dtype=np.int8))

# --- uint8: 1D with boundary values --------------------------------------------
# Expected data: [0, 1, 255] as float32 by value.
save("uint8_1d.npy", np.array([0, 1, 255], dtype=np.uint8))

# --- bool: 1D ------------------------------------------------------------------
# Expected data: [1.0, 0.0, 1.0, 1.0] (true→1.0, false→0.0).
save("bool_1d.npy", np.array([True, False, True, True], dtype=np.bool_))

# --- Fortran order: 2×3 --------------------------------------------------------
# Logical C-order values: [[0,1,2],[3,4,5]].
# Stored in .npy as Fortran (column-major): bytes = [0, 3, 1, 4, 2, 5].
# After loading and transposing to C order, expected data: [0,1,2,3,4,5].
save("fortran_2x3.npy",
     np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)))

# --- scalar float32 ------------------------------------------------------------
# Shape () → anneal shape []. Expected single value: 42.0.
np.save(os.path.join(here, "scalar_f32.npy"), np.float32(42.0))
print("  scalar_f32.npy: shape=() dtype=float32")

# --- 3D float32: 2×3×4 ---------------------------------------------------------
# Expected data: [0, 1, ..., 23] as float32.
save("float32_2x3x4.npy", np.arange(24, dtype=np.float32).reshape(2, 3, 4))

# --- big-endian float32 --------------------------------------------------------
# The descr in the .npy header will be '>f4'; logical values are [1,2,3,4].
save("bigendian_f32.npy",
     np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype(">f4")))

# --- int64 out-of-range (contains 2^24+1 = 16,777,217) ------------------------
# Loading should return an error about the out-of-range value.
save("int64_oor.npy", np.array([1, 2, 16_777_217], dtype=np.int64))

# --- complex64: unsupported dtype ----------------------------------------------
# Loading should return an error naming "complex".
save("complex64.npy", np.array([1 + 2j, 3 + 4j], dtype=np.complex64))

# --- NPZ: multiple named arrays ------------------------------------------------
# Keys: "weights" (2×3 float32), "bias" (3 float32), "mask" (3 bool).
# All values chosen to be exactly representable in float32.
np.savez(
    os.path.join(here, "multi.npz"),
    weights=np.arange(6, dtype=np.float32).reshape(2, 3),
    bias=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    mask=np.array([True, False, True], dtype=np.bool_),
)
print("  multi.npz: weights(2×3 f32), bias(3 f32), mask(3 bool)")

print("\nDone. All fixture files written.")
