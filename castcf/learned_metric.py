from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from castcf.retrieval import cosine_similarity_matrix, shape_knn_search


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _validate_row_count(*arrays: np.ndarray) -> None:
    expected = len(arrays[0])
    if any(len(arr) != expected for arr in arrays[1:]):
        raise ValueError("feature arrays must have matching row counts")


def _pair_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    dot = np.sum(left * right, axis=1)
    return np.divide(dot, denom, out=np.zeros_like(dot, dtype=float), where=denom > 0)


def _linear_scores(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", np.ascontiguousarray(matrix, dtype=float), weights)


def pair_feature_matrix(
    query_x: np.ndarray,
    query_context: np.ndarray,
    query_meta: np.ndarray,
    corpus_x: np.ndarray,
    corpus_context: np.ndarray,
    corpus_meta: np.ndarray,
    query_indices: np.ndarray,
    corpus_indices: np.ndarray,
) -> np.ndarray:
    """Construct query-candidate features used by the learned metric scorer."""
    qx = _as_2d(query_x, "query_x")
    qctx = _as_2d(query_context, "query_context")
    qmeta = _as_2d(query_meta, "query_meta")
    cx = _as_2d(corpus_x, "corpus_x")
    cctx = _as_2d(corpus_context, "corpus_context")
    cmeta = _as_2d(corpus_meta, "corpus_meta")
    _validate_row_count(qx, qctx, qmeta)
    _validate_row_count(cx, cctx, cmeta)

    qids = np.asarray(query_indices, dtype=int)
    cids = np.asarray(corpus_indices, dtype=int)
    if qids.ndim != 1 or cids.ndim != 1 or len(qids) != len(cids):
        raise ValueError("query_indices and corpus_indices must be 1D arrays with equal length")
    if np.any(qids < 0) or np.any(qids >= len(qx)):
        raise ValueError("query_indices contain out-of-range values")
    if np.any(cids < 0) or np.any(cids >= len(cx)):
        raise ValueError("corpus_indices contain out-of-range values")

    qx_rows = qx[qids]
    qctx_rows = qctx[qids]
    qmeta_rows = qmeta[qids]
    cx_rows = cx[cids]
    cctx_rows = cctx[cids]
    cmeta_rows = cmeta[cids]

    shape_cos = _pair_cosine(qx_rows, cx_rows)
    context_cos = _pair_cosine(qctx_rows, cctx_rows)
    meta_cos = _pair_cosine(qmeta_rows, cmeta_rows)
    shape_distance = -np.mean(np.abs(qx_rows - cx_rows), axis=1)
    context_distance = -np.mean(np.abs(qctx_rows - cctx_rows), axis=1)
    meta_distance = -np.mean(np.abs(qmeta_rows - cmeta_rows), axis=1)

    return np.column_stack(
        [
            shape_cos,
            context_cos,
            meta_cos,
            shape_distance,
            context_distance,
            meta_distance,
        ]
    )


@dataclass(frozen=True)
class LearnedMetricScorer:
    """Linear scorer trained with forecast-aware pairwise ranking loss."""

    weights: np.ndarray
    loss_history: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    @classmethod
    def fit_pairwise(
        cls,
        positive_features: np.ndarray,
        negative_features: np.ndarray,
        epochs: int = 200,
        learning_rate: float = 0.1,
        l2: float = 1e-4,
        seed: int | None = None,
    ) -> "LearnedMetricScorer":
        positives = _as_2d(positive_features, "positive_features")
        negatives = _as_2d(negative_features, "negative_features")
        if positives.shape != negatives.shape:
            raise ValueError(
                f"positive_features and negative_features shape mismatch: {positives.shape} vs {negatives.shape}"
            )
        if len(positives) == 0:
            raise ValueError("At least one training pair is required")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")

        combined = np.vstack([positives, negatives])
        feature_mean = combined.mean(axis=0)
        feature_scale = combined.std(axis=0)
        feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
        positives_scaled = (positives - feature_mean) / feature_scale
        negatives_scaled = (negatives - feature_mean) / feature_scale
        if not np.all(np.isfinite(positives_scaled)) or not np.all(np.isfinite(negatives_scaled)):
            raise ValueError("Training features contain non-finite values")

        rng = np.random.default_rng(seed)
        weights = rng.normal(loc=0.0, scale=0.01, size=positives.shape[1])
        diff = positives_scaled - negatives_scaled
        losses: list[float] = []

        for _ in range(epochs):
            logits = _linear_scores(diff, weights)
            clipped = np.clip(logits, -50.0, 50.0)
            sigmoid_negative = 1.0 / (1.0 + np.exp(clipped))
            gradient = -np.mean(sigmoid_negative[:, None] * diff, axis=0) + l2 * weights
            weights = weights - learning_rate * gradient

            loss_logits = _linear_scores(diff, weights)
            loss = np.mean(np.log1p(np.exp(-np.clip(loss_logits, -50.0, 50.0))))
            loss += 0.5 * l2 * float(np.sum(weights * weights))
            losses.append(float(loss))

        return cls(
            weights=weights,
            loss_history=np.asarray(losses, dtype=float),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = _as_2d(features, "features")
        if matrix.shape[1] != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} features, got {matrix.shape[1]}")
        if self.feature_mean is not None and self.feature_scale is not None:
            matrix = (matrix - self.feature_mean) / self.feature_scale
        return _linear_scores(matrix, self.weights)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": self.weights}
        if self.loss_history is not None:
            payload["loss_history"] = self.loss_history
        if self.feature_mean is not None and self.feature_scale is not None:
            payload["feature_mean"] = self.feature_mean
            payload["feature_scale"] = self.feature_scale
        np.savez(output_path, **payload)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "LearnedMetricScorer":
        with np.load(Path(path), allow_pickle=False) as payload:
            loss_history = payload["loss_history"] if "loss_history" in payload.files else None
            feature_mean = payload["feature_mean"] if "feature_mean" in payload.files else None
            feature_scale = payload["feature_scale"] if "feature_scale" in payload.files else None
            return cls(
                weights=payload["weights"],
                loss_history=loss_history,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
            )


def multiroute_candidate_indices(
    query_x: np.ndarray,
    query_context: np.ndarray,
    query_meta: np.ndarray,
    corpus_x: np.ndarray,
    corpus_context: np.ndarray,
    corpus_meta: np.ndarray,
    route_k: int,
    exclude_self: bool = False,
) -> list[np.ndarray]:
    """Return merged shape/context/meta candidate ids for each query."""
    qx = _as_2d(query_x, "query_x")
    qctx = _as_2d(query_context, "query_context")
    qmeta = _as_2d(query_meta, "query_meta")
    cx = _as_2d(corpus_x, "corpus_x")
    cctx = _as_2d(corpus_context, "corpus_context")
    cmeta = _as_2d(corpus_meta, "corpus_meta")
    _validate_row_count(qx, qctx, qmeta)
    _validate_row_count(cx, cctx, cmeta)
    if route_k < 1:
        raise ValueError("route_k must be positive")
    if len(cx) == 0:
        raise ValueError("corpus must contain at least one row")

    k_eff = min(route_k, len(cx))
    shape_candidates, _ = shape_knn_search(qx, cx, k=k_eff)
    context_candidates, _ = shape_knn_search(qctx, cctx, k=k_eff)
    meta_candidates, _ = shape_knn_search(qmeta, cmeta, k=k_eff)

    candidate_lists: list[np.ndarray] = []
    for query_id in range(len(qx)):
        merged = np.unique(
            np.concatenate(
                [
                    shape_candidates[query_id],
                    context_candidates[query_id],
                    meta_candidates[query_id],
                ]
            )
        )
        if exclude_self and len(qx) == len(cx):
            merged = merged[merged != query_id]
        candidate_lists.append(merged.astype(int, copy=False))
    return candidate_lists


def learned_metric_search(
    query_x: np.ndarray,
    query_context: np.ndarray,
    query_meta: np.ndarray,
    corpus_x: np.ndarray,
    corpus_context: np.ndarray,
    corpus_meta: np.ndarray,
    scorer: LearnedMetricScorer,
    k: int,
    route_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve corpus cases with the learned forecast-aware metric scorer."""
    qx = _as_2d(query_x, "query_x")
    cx = _as_2d(corpus_x, "corpus_x")
    if k < 1:
        raise ValueError("k must be positive")
    output_k = min(k, len(cx))
    candidates_by_query = multiroute_candidate_indices(
        qx,
        query_context,
        query_meta,
        cx,
        corpus_context,
        corpus_meta,
        route_k=max(route_k, output_k),
    )

    neighbors = np.empty((len(qx), output_k), dtype=int)
    scores = np.empty_like(neighbors, dtype=float)
    for query_id, candidates in enumerate(candidates_by_query):
        if len(candidates) == 0:
            fallback_scores = cosine_similarity_matrix(qx[[query_id]], cx).ravel()
            order = np.argsort(-fallback_scores)[:output_k]
            selected = order
            selected_scores = fallback_scores[selected]
        else:
            query_ids = np.full(len(candidates), query_id, dtype=int)
            features = pair_feature_matrix(
                qx,
                query_context,
                query_meta,
                cx,
                corpus_context,
                corpus_meta,
                query_ids,
                candidates,
            )
            candidate_scores = scorer.predict(features)
            order = np.argsort(-candidate_scores)[:output_k]
            selected = candidates[order]
            selected_scores = candidate_scores[order]

        if len(selected) < output_k:
            pad = output_k - len(selected)
            selected = np.pad(selected, (0, pad), mode="edge")
            selected_scores = np.pad(selected_scores, (0, pad), mode="edge")
        neighbors[query_id] = selected
        scores[query_id] = selected_scores
    return neighbors, scores
