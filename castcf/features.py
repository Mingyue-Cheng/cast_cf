from __future__ import annotations

import numpy as np


def standardize_by_reference(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize query/reference arrays using reference-only statistics."""
    query_arr = np.asarray(query, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if query_arr.ndim != 2 or ref_arr.ndim != 2:
        raise ValueError("query and reference must be 2D arrays")
    if query_arr.shape[1] != ref_arr.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: query={query_arr.shape[1]}, reference={ref_arr.shape[1]}"
        )

    mean = ref_arr.mean(axis=0, keepdims=True)
    std = ref_arr.std(axis=0, keepdims=True)
    safe_std = np.where(std > 0, std, 1.0)
    return (query_arr - mean) / safe_std, (ref_arr - mean) / safe_std

