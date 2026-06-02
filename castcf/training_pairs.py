from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairBatch:
    """Forecast-utility supervision pairs for metric learning."""

    query_indices: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    positive_distances: np.ndarray
    negative_distances: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(len(self.query_indices))


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return arr


def _candidate_iter(candidate_indices: Iterable[np.ndarray] | np.ndarray) -> Iterable[np.ndarray]:
    if isinstance(candidate_indices, np.ndarray):
        if candidate_indices.ndim != 2:
            raise ValueError(f"candidate_indices must be 2D, got shape {candidate_indices.shape}")
        return (candidate_indices[row_id] for row_id in range(candidate_indices.shape[0]))
    return candidate_indices


def build_future_utility_pairs(
    y_future: np.ndarray,
    candidate_indices: Iterable[np.ndarray] | np.ndarray,
    min_distance_margin: float = 0.0,
) -> PairBatch:
    """Build ranking pairs using observed future-trajectory utility.

    For each query case, the closest candidate future trajectory becomes the
    positive item, and the farthest candidate future trajectory becomes the
    negative item. The query itself is removed when present.
    """
    y = _as_2d(y_future)
    if min_distance_margin < 0:
        raise ValueError("min_distance_margin must be non-negative")

    query_ids: list[int] = []
    positive_ids: list[int] = []
    negative_ids: list[int] = []
    positive_distances: list[float] = []
    negative_distances: list[float] = []

    for query_id, raw_candidates in enumerate(_candidate_iter(candidate_indices)):
        if query_id >= len(y):
            break
        candidates = np.asarray(raw_candidates, dtype=int)
        candidates = candidates[(candidates >= 0) & (candidates < len(y))]
        candidates = np.unique(candidates[candidates != query_id])
        if len(candidates) < 2:
            continue

        distances = np.mean(np.abs(y[candidates] - y[query_id]), axis=1)
        finite_mask = np.isfinite(distances)
        candidates = candidates[finite_mask]
        distances = distances[finite_mask]
        if len(candidates) < 2:
            continue

        positive_pos = int(np.argmin(distances))
        negative_pos = int(np.argmax(distances))
        positive_distance = float(distances[positive_pos])
        negative_distance = float(distances[negative_pos])
        if negative_distance - positive_distance < min_distance_margin:
            continue

        query_ids.append(query_id)
        positive_ids.append(int(candidates[positive_pos]))
        negative_ids.append(int(candidates[negative_pos]))
        positive_distances.append(positive_distance)
        negative_distances.append(negative_distance)

    return PairBatch(
        query_indices=np.asarray(query_ids, dtype=int),
        positive_indices=np.asarray(positive_ids, dtype=int),
        negative_indices=np.asarray(negative_ids, dtype=int),
        positive_distances=np.asarray(positive_distances, dtype=float),
        negative_distances=np.asarray(negative_distances, dtype=float),
    )
