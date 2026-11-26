"""Embedding (de)serialisation helpers."""

from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 512
_BLOCK = 8192  # rows per block for in-place normalisation


def normalize(v: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """L2-normalise a vector or the rows of a matrix.

    ``inplace=True`` divides a float32 matrix in place, in row blocks, so a large matrix is
    not copied (the row norms are computed per row either way: same result).
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
    if inplace and v.flags.writeable:
        for start in range(0, len(v), _BLOCK):
            rows = v[start : start + _BLOCK]
            n = np.linalg.norm(rows, axis=1, keepdims=True)
            n[n == 0] = 1.0
            rows /= n
        return v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b, dtype=np.float32)


def stack(blobs: list[bytes]) -> np.ndarray:
    if not blobs:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.vstack([from_blob(b) for b in blobs]).astype(np.float32, copy=False)


def centroid(m: np.ndarray) -> np.ndarray:
    return normalize(m.mean(axis=0)) if len(m) else np.zeros(EMBEDDING_DIM, dtype=np.float32)
