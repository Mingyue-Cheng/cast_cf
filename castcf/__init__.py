"""CastCF-lite experiment utilities."""

from castcf.data import (
    CONTEXT_COLUMNS,
    ENTITY_COLUMNS,
    build_daily_cases,
    case_overlap_exclusion_mask,
    load_freshretail_split,
    sample_series_ids,
    save_cases,
)
from castcf.features import overlap_exclusion_mask, standardize_by_reference
from castcf.learned_metric import (
    LearnedMetricScorer,
    learned_metric_search,
    multiroute_candidate_indices,
    pair_feature_matrix,
)
from castcf.metrics import mae, mse, nfd_at_k, query_nfd_at_k, subset_metric_table
from castcf.retrieval import (
    aggregate_neighbor_futures,
    castcf_lite_neighbors,
    castcf_lite_search,
    castcf_multiroute_search,
    cosine_similarity_matrix,
    meta_match_matrix,
    shape_knn_neighbors,
    shape_knn_search,
)
from castcf.training_pairs import PairBatch, build_future_utility_pairs

__all__ = [
    "CONTEXT_COLUMNS",
    "ENTITY_COLUMNS",
    "LearnedMetricScorer",
    "PairBatch",
    "aggregate_neighbor_futures",
    "build_daily_cases",
    "build_future_utility_pairs",
    "case_overlap_exclusion_mask",
    "castcf_lite_neighbors",
    "castcf_lite_search",
    "castcf_multiroute_search",
    "cosine_similarity_matrix",
    "learned_metric_search",
    "load_freshretail_split",
    "mae",
    "meta_match_matrix",
    "mse",
    "multiroute_candidate_indices",
    "nfd_at_k",
    "overlap_exclusion_mask",
    "pair_feature_matrix",
    "query_nfd_at_k",
    "sample_series_ids",
    "save_cases",
    "shape_knn_neighbors",
    "shape_knn_search",
    "standardize_by_reference",
    "subset_metric_table",
]
