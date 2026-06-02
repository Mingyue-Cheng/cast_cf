from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from castcf.features import standardize_by_reference
from castcf.learned_metric import LearnedMetricScorer, multiroute_candidate_indices, pair_feature_matrix
from castcf.training_pairs import build_future_utility_pairs
from experiments.prepare_freshretail_cases import load_config


def _stack_column(rows: pd.DataFrame, column: str, dtype: type = float) -> np.ndarray:
    values = [np.asarray(value, dtype=dtype) for value in rows[column].to_list()]
    return np.stack(values, axis=0)


def _combined_context(rows: pd.DataFrame) -> np.ndarray:
    return np.hstack([_stack_column(rows, "context_past"), _stack_column(rows, "context_future")])


def _standardized_memory_features(
    x_memory: np.ndarray,
    ctx_memory: np.ndarray,
    meta_memory: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        return x_memory, ctx_memory, meta_memory
    x_retrieval, _ = standardize_by_reference(x_memory, x_memory)
    ctx_retrieval, _ = standardize_by_reference(ctx_memory, ctx_memory)
    meta_retrieval, _ = standardize_by_reference(meta_memory, meta_memory)
    return x_retrieval, ctx_retrieval, meta_retrieval


def train_metric_from_config(config: dict[str, Any]) -> dict[str, int | float | str]:
    data_cfg = config["data"]
    retrieval_cfg = config.get("retrieval", {})
    training_cfg = config.get("training", {})
    cases_path = Path(data_cfg["cases_path"])
    if not cases_path.exists():
        raise FileNotFoundError(f"Prepared cases file not found: {cases_path}")

    model_path = Path(retrieval_cfg.get("learned_model_path", "artifacts/freshretail_learned_metric.npz"))
    cases = pd.read_parquet(cases_path)
    memory = cases[cases["split"] == "memory"].reset_index(drop=True)
    if len(memory) < 3:
        raise ValueError(f"Need at least three memory cases for pairwise training, got {len(memory)}")

    x_memory = _stack_column(memory, "x_past")
    y_memory = _stack_column(memory, "y_future")
    ctx_memory = _combined_context(memory)
    meta_memory = _stack_column(memory, "meta")

    standardize_features = bool(retrieval_cfg.get("standardize_features", True))
    x_retrieval, ctx_retrieval, meta_retrieval = _standardized_memory_features(
        x_memory,
        ctx_memory,
        meta_memory,
        enabled=standardize_features,
    )

    route_k = int(
        training_cfg.get(
            "route_k",
            retrieval_cfg.get("route_k", retrieval_cfg.get("candidate_k", min(100, len(memory) - 1))),
        )
    )
    route_k = max(2, min(route_k, len(memory)))
    candidate_lists = multiroute_candidate_indices(
        x_retrieval,
        ctx_retrieval,
        meta_retrieval,
        x_retrieval,
        ctx_retrieval,
        meta_retrieval,
        route_k=route_k,
        exclude_self=True,
    )
    pairs = build_future_utility_pairs(
        y_memory,
        candidate_lists,
        min_distance_margin=float(training_cfg.get("min_distance_margin", 0.0)),
    )
    if pairs.pair_count == 0:
        raise ValueError("No forecast-utility training pairs were generated; lower min_distance_margin or increase route_k")

    positive_features = pair_feature_matrix(
        x_retrieval,
        ctx_retrieval,
        meta_retrieval,
        x_retrieval,
        ctx_retrieval,
        meta_retrieval,
        pairs.query_indices,
        pairs.positive_indices,
    )
    negative_features = pair_feature_matrix(
        x_retrieval,
        ctx_retrieval,
        meta_retrieval,
        x_retrieval,
        ctx_retrieval,
        meta_retrieval,
        pairs.query_indices,
        pairs.negative_indices,
    )
    scorer = LearnedMetricScorer.fit_pairwise(
        positive_features,
        negative_features,
        epochs=int(training_cfg.get("epochs", 200)),
        learning_rate=float(training_cfg.get("learning_rate", 0.1)),
        l2=float(training_cfg.get("l2", 1e-4)),
        seed=int(training_cfg.get("seed", data_cfg.get("seed", 42))),
    )
    scorer.save(model_path)

    loss_history = scorer.loss_history if scorer.loss_history is not None else np.array([], dtype=float)
    return {
        "memory_cases": int(len(memory)),
        "route_k": int(route_k),
        "pair_count": int(pairs.pair_count),
        "initial_loss": float(loss_history[0]) if len(loss_history) else float("nan"),
        "final_loss": float(loss_history[-1]) if len(loss_history) else float("nan"),
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CastCF forecast-aware metric reranker.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    summary = train_metric_from_config(load_config(args.config))
    print(summary)


if __name__ == "__main__":
    main()
