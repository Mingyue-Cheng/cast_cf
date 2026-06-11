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


def overlap_exclusion_mask(
    query_series: np.ndarray,
    query_anchors: np.ndarray,
    corpus_series: np.ndarray,
    corpus_anchors: np.ndarray,
    min_anchor_gap: int,
) -> np.ndarray:
    """Mark query-corpus pairs whose future windows can overlap in time.

    A corpus case is excluded for a query when both belong to the same series
    and their anchors are closer than `min_anchor_gap` days (typically the
    forecast horizon), because their `y_future` windows would share days and
    leak the query's own future into retrieval or pair supervision.
    """
    q_series = np.asarray(query_series)
    c_series = np.asarray(corpus_series)
    q_anchor = np.asarray(query_anchors, dtype=np.int64)
    c_anchor = np.asarray(corpus_anchors, dtype=np.int64)
    if q_series.ndim != 1 or c_series.ndim != 1 or q_anchor.ndim != 1 or c_anchor.ndim != 1:
        raise ValueError("series and anchor arrays must be 1D")
    if len(q_series) != len(q_anchor):
        raise ValueError("query series and anchors must have the same length")
    if len(c_series) != len(c_anchor):
        raise ValueError("corpus series and anchors must have the same length")
    if min_anchor_gap < 1:
        raise ValueError("min_anchor_gap must be positive")

    same_series = q_series[:, None] == c_series[None, :]
    near_anchor = np.abs(q_anchor[:, None] - c_anchor[None, :]) < min_anchor_gap
    return same_series & near_anchor

