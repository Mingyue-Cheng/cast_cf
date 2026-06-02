from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from castcf.features import standardize_by_reference
from castcf.learned_metric import LearnedMetricScorer, learned_metric_search
from castcf.metrics import query_nfd_at_k, subset_metric_table
from castcf.retrieval import (
    aggregate_neighbor_futures,
    castcf_lite_search,
    castcf_multiroute_search,
    shape_knn_search,
)
from experiments.prepare_freshretail_cases import load_config


DEFAULT_SUBSETS = [
    "future_discounted",
    "future_holiday",
    "future_activity",
    "past_stockout",
    "future_stockout",
]


def _stack_column(rows: pd.DataFrame, column: str, dtype: type = float) -> np.ndarray:
    values = [np.asarray(value, dtype=dtype) for value in rows[column].to_list()]
    return np.stack(values, axis=0)


def _combined_context(rows: pd.DataFrame) -> np.ndarray:
    return np.hstack([_stack_column(rows, "context_past"), _stack_column(rows, "context_future")])


def _recent_forecast(x_past: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(x_past[:, [-1]], repeats=horizon, axis=1)


def _context_knn_search(
    query_context: np.ndarray,
    corpus_context: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    return shape_knn_search(query_context, corpus_context, k=k)


def _method_report(
    query_cases: pd.DataFrame,
    y_query: np.ndarray,
    y_pred: np.ndarray,
    subset_columns: list[str],
    y_memory: np.ndarray | None = None,
    neighbors: np.ndarray | None = None,
) -> dict[str, Any]:
    subsets = subset_metric_table(query_cases, y_query, y_pred, subset_columns=subset_columns)
    report: dict[str, Any] = {
        "mae": subsets["all"]["mae"],
        "mse": subsets["all"]["mse"],
        "subsets": subsets,
    }
    if y_memory is not None and neighbors is not None:
        nfd = query_nfd_at_k(y_query, y_memory, neighbors)
        report["nfd_at_k"] = float(np.mean(nfd))
        report["nfd_at_k_std"] = float(np.std(nfd))
    return report


def run_baselines_from_config(config: dict[str, Any]) -> dict[str, int | str]:
    data_cfg = config["data"]
    retrieval_cfg = config["retrieval"]
    cases_path = Path(data_cfg["cases_path"])
    if not cases_path.exists():
        raise FileNotFoundError(f"Prepared cases file not found: {cases_path}")

    cases = pd.read_parquet(cases_path)
    memory = cases[cases["split"] == "memory"].reset_index(drop=True)
    query = cases[cases["split"] == "query"].reset_index(drop=True)
    if memory.empty or query.empty:
        raise ValueError(f"Need non-empty memory and query cases, got memory={len(memory)}, query={len(query)}")

    x_memory = _stack_column(memory, "x_past")
    y_memory = _stack_column(memory, "y_future")
    ctx_memory = _combined_context(memory)
    meta_memory = _stack_column(memory, "meta")

    x_query = _stack_column(query, "x_past")
    y_query = _stack_column(query, "y_future")
    ctx_query = _combined_context(query)
    meta_query = _stack_column(query, "meta")

    standardize_features = bool(retrieval_cfg.get("standardize_features", True))
    if standardize_features:
        x_query_retrieval, x_memory_retrieval = standardize_by_reference(x_query, x_memory)
        ctx_query_retrieval, ctx_memory_retrieval = standardize_by_reference(ctx_query, ctx_memory)
        meta_query_retrieval, meta_memory_retrieval = standardize_by_reference(meta_query, meta_memory)
    else:
        x_query_retrieval, x_memory_retrieval = x_query, x_memory
        ctx_query_retrieval, ctx_memory_retrieval = ctx_query, ctx_memory
        meta_query_retrieval, meta_memory_retrieval = meta_query, meta_memory

    k = int(retrieval_cfg["k"])
    candidate_k = int(retrieval_cfg.get("candidate_k", max(k, 20)))
    route_k = int(retrieval_cfg.get("route_k", candidate_k))
    temperature = float(retrieval_cfg.get("temperature", 0.1))
    subset_columns = list(data_cfg.get("subset_columns", DEFAULT_SUBSETS))

    recent_pred = _recent_forecast(x_query, horizon=y_query.shape[1])

    shape_neighbors, shape_scores = shape_knn_search(x_query_retrieval, x_memory_retrieval, k=k)
    shape_pred = aggregate_neighbor_futures(y_memory, shape_neighbors, shape_scores, temperature=temperature)

    context_neighbors, context_scores = _context_knn_search(ctx_query_retrieval, ctx_memory_retrieval, k=k)
    context_pred = aggregate_neighbor_futures(y_memory, context_neighbors, context_scores, temperature=temperature)

    castcf_neighbors, castcf_scores = castcf_lite_search(
        x_query_retrieval,
        ctx_query_retrieval,
        meta_query_retrieval,
        x_memory_retrieval,
        ctx_memory_retrieval,
        meta_memory_retrieval,
        k=k,
        candidate_k=candidate_k,
        shape_weight=float(retrieval_cfg.get("shape_weight", 0.4)),
        context_weight=float(retrieval_cfg.get("context_weight", 0.5)),
        meta_weight=float(retrieval_cfg.get("meta_weight", 0.1)),
    )
    castcf_pred = aggregate_neighbor_futures(y_memory, castcf_neighbors, castcf_scores, temperature=temperature)

    multiroute_neighbors, multiroute_scores = castcf_multiroute_search(
        x_query_retrieval,
        ctx_query_retrieval,
        meta_query_retrieval,
        x_memory_retrieval,
        ctx_memory_retrieval,
        meta_memory_retrieval,
        k=k,
        route_k=route_k,
        shape_weight=float(retrieval_cfg.get("shape_weight", 0.4)),
        context_weight=float(retrieval_cfg.get("context_weight", 0.5)),
        meta_weight=float(retrieval_cfg.get("meta_weight", 0.1)),
    )
    multiroute_pred = aggregate_neighbor_futures(
        y_memory,
        multiroute_neighbors,
        multiroute_scores,
        temperature=temperature,
    )

    method_reports = {
        "recent": _method_report(query, y_query, recent_pred, subset_columns),
        "shape_knn": _method_report(query, y_query, shape_pred, subset_columns, y_memory, shape_neighbors),
        "context_knn": _method_report(query, y_query, context_pred, subset_columns, y_memory, context_neighbors),
        "castcf_lite": _method_report(query, y_query, castcf_pred, subset_columns, y_memory, castcf_neighbors),
        "castcf_multiroute": _method_report(
            query,
            y_query,
            multiroute_pred,
            subset_columns,
            y_memory,
            multiroute_neighbors,
        ),
    }

    learned_model_path = retrieval_cfg.get("learned_model_path")
    if learned_model_path and Path(learned_model_path).exists():
        learned_scorer = LearnedMetricScorer.load(learned_model_path)
        learned_neighbors, learned_scores = learned_metric_search(
            x_query_retrieval,
            ctx_query_retrieval,
            meta_query_retrieval,
            x_memory_retrieval,
            ctx_memory_retrieval,
            meta_memory_retrieval,
            learned_scorer,
            k=k,
            route_k=route_k,
        )
        learned_pred = aggregate_neighbor_futures(
            y_memory,
            learned_neighbors,
            learned_scores,
            temperature=temperature,
        )
        method_reports["learned_metric"] = _method_report(
            query,
            y_query,
            learned_pred,
            subset_columns,
            y_memory,
            learned_neighbors,
        )

    metrics = {
        "cases_path": str(cases_path),
        "memory_cases": int(len(memory)),
        "query_cases": int(len(query)),
        "retrieval": {
            "k": k,
            "candidate_k": candidate_k,
            "route_k": route_k,
            "temperature": temperature,
            "standardize_features": standardize_features,
            "shape_weight": float(retrieval_cfg.get("shape_weight", 0.4)),
            "context_weight": float(retrieval_cfg.get("context_weight", 0.5)),
            "meta_weight": float(retrieval_cfg.get("meta_weight", 0.1)),
            "learned_model_path": str(learned_model_path) if learned_model_path else None,
        },
        "methods": method_reports,
    }

    metrics_path = Path(data_cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "memory_cases": int(len(memory)),
        "query_cases": int(len(query)),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CastCF-lite retrieval baselines.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    summary = run_baselines_from_config(load_config(args.config))
    print(summary)


if __name__ == "__main__":
    main()
