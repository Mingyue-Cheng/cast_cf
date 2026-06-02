from __future__ import annotations

import numpy as np


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return arr


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = _as_2d(values)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr, dtype=float), where=norms > 0)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosine similarities with zero-vector protection."""
    left_norm = np.ascontiguousarray(_normalize_rows(left), dtype=float)
    right_norm = np.ascontiguousarray(_normalize_rows(right), dtype=float)
    return np.einsum("id,jd->ij", left_norm, right_norm)


def _topk_desc(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be positive")
    k_eff = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=np.arange(k_eff), axis=1)[:, :k_eff]
    row = np.arange(scores.shape[0])[:, None]
    unsorted_scores = scores[row, idx]
    order = np.argsort(-unsorted_scores, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    return top_idx, top_scores


def shape_knn_neighbors(x_past: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve nearest cases by past-shape cosine similarity."""
    x = _as_2d(x_past)
    if len(x) < 2:
        raise ValueError("At least two cases are required for neighbor retrieval")
    sims = cosine_similarity_matrix(x, x)
    np.fill_diagonal(sims, -np.inf)
    return _topk_desc(sims, k)


def shape_knn_search(
    query_x_past: np.ndarray,
    corpus_x_past: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve corpus cases for each query by past-shape cosine similarity."""
    query_x = _as_2d(query_x_past)
    corpus_x = _as_2d(corpus_x_past)
    sims = cosine_similarity_matrix(query_x, corpus_x)
    return _topk_desc(sims, k)


def castcf_lite_neighbors(
    x_past: np.ndarray,
    context: np.ndarray,
    meta: np.ndarray,
    k: int,
    candidate_k: int,
    shape_weight: float = 0.5,
    context_weight: float = 0.4,
    meta_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank shape candidates using context and entity metadata similarity."""
    x = _as_2d(x_past)
    ctx = _as_2d(context)
    ent = _as_2d(meta)
    if len(x) != len(ctx) or len(x) != len(ent):
        raise ValueError("x_past, context, and meta must have the same row count")

    candidate_k = max(candidate_k, k)
    candidates, shape_scores = shape_knn_neighbors(x, k=candidate_k)
    ctx_sims = cosine_similarity_matrix(ctx, ctx)
    meta_sims = cosine_similarity_matrix(ent, ent)

    reranked = np.empty((len(x), min(k, candidates.shape[1])), dtype=int)
    reranked_scores = np.empty_like(reranked, dtype=float)
    for row_id in range(len(x)):
        cand = candidates[row_id]
        combined = (
            shape_weight * shape_scores[row_id]
            + context_weight * ctx_sims[row_id, cand]
            + meta_weight * meta_sims[row_id, cand]
        )
        order = np.argsort(-combined)[:k]
        reranked[row_id] = cand[order]
        reranked_scores[row_id] = combined[order]
    return reranked, reranked_scores


def castcf_lite_search(
    query_x_past: np.ndarray,
    query_context: np.ndarray,
    query_meta: np.ndarray,
    corpus_x_past: np.ndarray,
    corpus_context: np.ndarray,
    corpus_meta: np.ndarray,
    k: int,
    candidate_k: int,
    shape_weight: float = 0.5,
    context_weight: float = 0.4,
    meta_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve corpus cases for each query with CastCF-lite reranking."""
    query_x = _as_2d(query_x_past)
    query_ctx = _as_2d(query_context)
    query_ent = _as_2d(query_meta)
    corpus_x = _as_2d(corpus_x_past)
    corpus_ctx = _as_2d(corpus_context)
    corpus_ent = _as_2d(corpus_meta)
    if len(query_x) != len(query_ctx) or len(query_x) != len(query_ent):
        raise ValueError("query arrays must have the same row count")
    if len(corpus_x) != len(corpus_ctx) or len(corpus_x) != len(corpus_ent):
        raise ValueError("corpus arrays must have the same row count")

    candidate_k = max(candidate_k, k)
    candidates, shape_scores = shape_knn_search(query_x, corpus_x, k=candidate_k)
    ctx_sims = cosine_similarity_matrix(query_ctx, corpus_ctx)
    meta_sims = cosine_similarity_matrix(query_ent, corpus_ent)

    reranked = np.empty((len(query_x), min(k, candidates.shape[1])), dtype=int)
    reranked_scores = np.empty_like(reranked, dtype=float)
    for row_id in range(len(query_x)):
        cand = candidates[row_id]
        combined = (
            shape_weight * shape_scores[row_id]
            + context_weight * ctx_sims[row_id, cand]
            + meta_weight * meta_sims[row_id, cand]
        )
        order = np.argsort(-combined)[:k]
        reranked[row_id] = cand[order]
        reranked_scores[row_id] = combined[order]
    return reranked, reranked_scores


def castcf_multiroute_search(
    query_x_past: np.ndarray,
    query_context: np.ndarray,
    query_meta: np.ndarray,
    corpus_x_past: np.ndarray,
    corpus_context: np.ndarray,
    corpus_meta: np.ndarray,
    k: int,
    route_k: int,
    shape_weight: float = 0.35,
    context_weight: float = 0.55,
    meta_weight: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve from shape/context/meta routes, then rerank the merged candidate pool."""
    query_x = _as_2d(query_x_past)
    query_ctx = _as_2d(query_context)
    query_ent = _as_2d(query_meta)
    corpus_x = _as_2d(corpus_x_past)
    corpus_ctx = _as_2d(corpus_context)
    corpus_ent = _as_2d(corpus_meta)
    if len(query_x) != len(query_ctx) or len(query_x) != len(query_ent):
        raise ValueError("query arrays must have the same row count")
    if len(corpus_x) != len(corpus_ctx) or len(corpus_x) != len(corpus_ent):
        raise ValueError("corpus arrays must have the same row count")
    if route_k < 1:
        raise ValueError("route_k must be positive")

    route_k = max(route_k, k)
    shape_sims = cosine_similarity_matrix(query_x, corpus_x)
    ctx_sims = cosine_similarity_matrix(query_ctx, corpus_ctx)
    meta_sims = cosine_similarity_matrix(query_ent, corpus_ent)

    shape_candidates, _ = _topk_desc(shape_sims, route_k)
    ctx_candidates, _ = _topk_desc(ctx_sims, route_k)
    meta_candidates, _ = _topk_desc(meta_sims, route_k)

    reranked = np.empty((len(query_x), min(k, len(corpus_x))), dtype=int)
    reranked_scores = np.empty_like(reranked, dtype=float)
    for row_id in range(len(query_x)):
        candidates = np.unique(
            np.concatenate(
                [
                    shape_candidates[row_id],
                    ctx_candidates[row_id],
                    meta_candidates[row_id],
                ]
            )
        )
        combined = (
            shape_weight * shape_sims[row_id, candidates]
            + context_weight * ctx_sims[row_id, candidates]
            + meta_weight * meta_sims[row_id, candidates]
        )
        order = np.argsort(-combined)[:k]
        selected = candidates[order]
        selected_scores = combined[order]
        if len(selected) < reranked.shape[1]:
            pad = reranked.shape[1] - len(selected)
            selected = np.pad(selected, (0, pad), mode="edge")
            selected_scores = np.pad(selected_scores, (0, pad), mode="edge")
        reranked[row_id] = selected
        reranked_scores[row_id] = selected_scores
    return reranked, reranked_scores


def aggregate_neighbor_futures(
    y_future: np.ndarray,
    neighbors: np.ndarray,
    scores: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """Aggregate neighbor future trajectories with softmax weights."""
    y = _as_2d(y_future)
    nbr = np.asarray(neighbors, dtype=int)
    scr = _as_2d(scores)
    if nbr.shape != scr.shape:
        raise ValueError("neighbors and scores must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    stable_scores = scr / temperature
    stable_scores = stable_scores - np.max(stable_scores, axis=1, keepdims=True)
    weights = np.exp(stable_scores)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predictions = np.zeros((nbr.shape[0], y.shape[1]), dtype=float)
    for row_id in range(nbr.shape[0]):
        predictions[row_id] = weights[row_id] @ y[nbr[row_id]]
    return predictions
