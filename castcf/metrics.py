from __future__ import annotations

import numpy as np
import pandas as pd


def _same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    return true, pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _same_shape(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _same_shape(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def nfd_at_k(y_future: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Neighbor Future Distance for each query case."""
    return query_nfd_at_k(y_future, y_future, neighbors)


def query_nfd_at_k(
    y_query_future: np.ndarray,
    y_corpus_future: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Neighbor Future Distance when query cases retrieve from a separate corpus."""
    y_query = np.asarray(y_query_future, dtype=float)
    y_corpus = np.asarray(y_corpus_future, dtype=float)
    nbr = np.asarray(neighbors, dtype=int)
    if y_query.ndim != 2 or y_corpus.ndim != 2 or nbr.ndim != 2:
        raise ValueError("y_query_future, y_corpus_future, and neighbors must be 2D arrays")
    if len(y_query) != len(nbr):
        raise ValueError("y_query_future and neighbors must have the same row count")
    if y_query.shape[1] != y_corpus.shape[1]:
        raise ValueError("query and corpus futures must have the same horizon length")

    neighbor_futures = y_corpus[nbr]  # (queries, k, horizon)
    return np.mean(np.abs(neighbor_futures - y_query[:, None, :]), axis=(1, 2))


def subset_metric_table(
    cases: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subset_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Report MAE/MSE over all cases and selected boolean subsets."""
    true, pred = _same_shape(y_true, y_pred)
    if len(cases) != len(true):
        raise ValueError("cases and prediction arrays must have the same row count")

    results: dict[str, dict[str, float]] = {
        "all": {
            "count": int(len(cases)),
            "mae": mae(true, pred),
            "mse": mse(true, pred),
        }
    }
    for column in subset_columns:
        if column not in cases.columns:
            continue
        mask = cases[column].astype(bool).to_numpy()
        count = int(mask.sum())
        if count == 0:
            results[column] = {"count": 0, "mae": float("nan"), "mse": float("nan")}
        else:
            results[column] = {
                "count": count,
                "mae": mae(true[mask], pred[mask]),
                "mse": mse(true[mask], pred[mask]),
            }
    return results
