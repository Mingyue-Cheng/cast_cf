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


def meta_match_matrix(query_meta: np.ndarray, corpus_meta: np.ndarray) -> np.ndarray:
    """Return the fraction of matching entity fields for each query-corpus pair.

    Entity ids are nominal: numeric closeness between two store or product ids
    carries no similarity information, so equality per field is the only
    meaningful signal. Scores are in [0, 1].
    """
    query = _as_2d(query_meta)
    corpus = _as_2d(corpus_meta)
    if query.shape[1] != corpus.shape[1]:
        raise ValueError(
            f"Meta dimension mismatch: query={query.shape[1]}, corpus={corpus.shape[1]}"
        )
    sims = np.zeros((len(query), len(corpus)), dtype=float)
    for col in range(query.shape[1]):
        sims += query[:, col][:, None] == corpus[:, col][None, :]
    return sims / query.shape[1]


def _apply_exclusion(sims: np.ndarray, exclusion_mask: np.ndarray | None) -> np.ndarray:
    if exclusion_mask is None:
        return sims
    mask = np.asarray(exclusion_mask, dtype=bool)
    if mask.shape != sims.shape:
        raise ValueError(
            f"exclusion_mask shape {mask.shape} does not match similarity shape {sims.shape}"
        )
    sims[mask] = -np.inf
    return sims


def _topk_desc(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be positive")
    k_eff = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    unsorted_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-unsorted_scores, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_scores = np.take_along_axis(unsorted_scores, order, axis=1)
    return top_idx, top_scores


def shape_knn_neighbors(
    x_past: np.ndarray,
    k: int,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve nearest cases by past-shape cosine similarity."""
    x = _as_2d(x_past)
    if len(x) < 2:
        raise ValueError("At least two cases are required for neighbor retrieval")
    sims = cosine_similarity_matrix(x, x)
    np.fill_diagonal(sims, -np.inf)
    _apply_exclusion(sims, exclusion_mask)
    return _topk_desc(sims, k)


def shape_knn_search(
    query_x_past: np.ndarray,
    corpus_x_past: np.ndarray,
    k: int,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve corpus cases for each query by past-shape cosine similarity."""
    query_x = _as_2d(query_x_past)
    corpus_x = _as_2d(corpus_x_past)
    sims = cosine_similarity_matrix(query_x, corpus_x)
    _apply_exclusion(sims, exclusion_mask)
    return _topk_desc(sims, k)


def _rerank_candidates(
    candidates: np.ndarray,
    shape_scores: np.ndarray,
    query_ctx_norm: np.ndarray,
    corpus_ctx_norm: np.ndarray,
    query_meta: np.ndarray,
    corpus_meta: np.ndarray,
    k: int,
    shape_weight: float,
    context_weight: float,
    meta_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Context/meta similarities are computed per candidate row instead of as a
    # full query-by-corpus matrix, keeping memory linear in the candidate count.
    output_k = min(k, candidates.shape[1])
    reranked = np.empty((len(candidates), output_k), dtype=int)
    reranked_scores = np.empty_like(reranked, dtype=float)
    for row_id in range(len(candidates)):
        cand = candidates[row_id]
        cand_shape_scores = shape_scores[row_id]
        valid = np.isfinite(cand_shape_scores)
        if valid.any():
            cand = cand[valid]
            cand_shape_scores = cand_shape_scores[valid]
        # einsum instead of matmul: Accelerate BLAS raises spurious FP warnings.
        ctx_sims = np.einsum("kd,d->k", corpus_ctx_norm[cand], query_ctx_norm[row_id])
        meta_sims = np.mean(corpus_meta[cand] == query_meta[row_id], axis=1)
        combined = (
            shape_weight * cand_shape_scores
            + context_weight * ctx_sims
            + meta_weight * meta_sims
        )
        order = np.argsort(-combined)[:output_k]
        selected = cand[order]
        selected_scores = combined[order]
        if len(selected) < output_k:
            pad = output_k - len(selected)
            selected = np.pad(selected, (0, pad), mode="edge")
            selected_scores = np.pad(selected_scores, (0, pad), mode="edge")
        reranked[row_id] = selected
        reranked_scores[row_id] = selected_scores
    return reranked, reranked_scores


def castcf_lite_neighbors(
    x_past: np.ndarray,
    context: np.ndarray,
    meta: np.ndarray,
    k: int,
    candidate_k: int,
    shape_weight: float = 0.5,
    context_weight: float = 0.4,
    meta_weight: float = 0.1,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank shape candidates using context similarity and entity-field matches."""
    x = _as_2d(x_past)
    ctx = _as_2d(context)
    ent = _as_2d(meta)
    if len(x) != len(ctx) or len(x) != len(ent):
        raise ValueError("x_past, context, and meta must have the same row count")

    candidate_k = max(candidate_k, k)
    candidates, shape_scores = shape_knn_neighbors(x, k=candidate_k, exclusion_mask=exclusion_mask)
    ctx_norm = _normalize_rows(ctx)
    return _rerank_candidates(
        candidates,
        shape_scores,
        ctx_norm,
        ctx_norm,
        ent,
        ent,
        k=k,
        shape_weight=shape_weight,
        context_weight=context_weight,
        meta_weight=meta_weight,
    )


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
    exclusion_mask: np.ndarray | None = None,
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
    candidates, shape_scores = shape_knn_search(
        query_x, corpus_x, k=candidate_k, exclusion_mask=exclusion_mask
    )
    return _rerank_candidates(
        candidates,
        shape_scores,
        _normalize_rows(query_ctx),
        _normalize_rows(corpus_ctx),
        query_ent,
        corpus_ent,
        k=k,
        shape_weight=shape_weight,
        context_weight=context_weight,
        meta_weight=meta_weight,
    )


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
    exclusion_mask: np.ndarray | None = None,
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
    shape_sims = _apply_exclusion(cosine_similarity_matrix(query_x, corpus_x), exclusion_mask)
    ctx_sims = _apply_exclusion(cosine_similarity_matrix(query_ctx, corpus_ctx), exclusion_mask)
    meta_sims = _apply_exclusion(meta_match_matrix(query_ent, corpus_ent), exclusion_mask)

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
        if exclusion_mask is not None:
            candidates = candidates[~np.asarray(exclusion_mask, dtype=bool)[row_id, candidates]]
        if len(candidates) == 0:
            candidates = np.argsort(-shape_sims[row_id])[: reranked.shape[1]]
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
    normalize_scores: bool = False,
) -> np.ndarray:
    """Aggregate neighbor future trajectories with softmax weights.

    Duplicate neighbor ids within a row (introduced by edge padding when a
    query has fewer than k valid candidates) keep only their first occurrence,
    so a padded neighbor is never double-counted.

    Set `normalize_scores=True` when scores come from an unbounded scorer
    (e.g. the learned metric): each row is standardized before the softmax so
    `temperature` keeps a comparable meaning across scoring functions.
    """
    y = _as_2d(y_future)
    nbr = np.asarray(neighbors, dtype=int)
    scr = np.array(_as_2d(scores), dtype=float, copy=True)
    if nbr.shape != scr.shape:
        raise ValueError("neighbors and scores must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if normalize_scores:
        finite = np.isfinite(scr)
        counts = np.maximum(finite.sum(axis=1, keepdims=True), 1)
        means = np.where(finite, scr, 0.0).sum(axis=1, keepdims=True) / counts
        variances = np.where(finite, (scr - means) ** 2, 0.0).sum(axis=1, keepdims=True) / counts
        stds = np.sqrt(variances)
        stds = np.where(stds > 1e-12, stds, 1.0)
        scr = np.where(finite, (scr - means) / stds, scr)

    for row_id in range(nbr.shape[0]):
        _, first_positions = np.unique(nbr[row_id], return_index=True)
        duplicate = np.ones(nbr.shape[1], dtype=bool)
        duplicate[first_positions] = False
        scr[row_id, duplicate] = -np.inf

    stable_scores = scr / temperature
    stable_scores = stable_scores - np.max(stable_scores, axis=1, keepdims=True)
    weights = np.exp(stable_scores)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predictions = np.zeros((nbr.shape[0], y.shape[1]), dtype=float)
    for row_id in range(nbr.shape[0]):
        predictions[row_id] = weights[row_id] @ y[nbr[row_id]]
    return predictions
