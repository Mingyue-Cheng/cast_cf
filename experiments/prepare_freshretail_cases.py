from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from castcf.data import CONTEXT_COLUMNS, ENTITY_COLUMNS, build_daily_cases, sample_series_ids, save_cases


DATA_COLUMNS = ENTITY_COLUMNS + CONTEXT_COLUMNS + [
    "dt",
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing FreshRetailNet split: {path}")
    return pd.read_parquet(path, columns=DATA_COLUMNS)


def _with_series_id(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["series_id"] = work["store_id"].astype(str) + "_" + work["product_id"].astype(str)
    return work


def _filter_series(rows: pd.DataFrame, series_ids: list[str]) -> pd.DataFrame:
    work = _with_series_id(rows)
    return work[work["series_id"].isin(set(series_ids))].drop(columns=["series_id"])


def prepare_cases_from_config(config: dict[str, Any]) -> dict[str, int | str]:
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]
    lookback_days = int(data_cfg["lookback_days"])
    horizon_days = int(data_cfg["horizon_days"])
    stride_days = int(data_cfg.get("stride_days", horizon_days))
    max_series = data_cfg.get("max_series")
    max_series = None if max_series is None else int(max_series)
    seed = int(data_cfg.get("seed", 42))

    train = _read_split(data_dir, "train")
    eval_rows = _read_split(data_dir, "eval")
    series_ids = sample_series_ids(train, max_series=max_series, seed=seed)

    train_sample = _filter_series(train, series_ids)
    eval_sample = _filter_series(eval_rows, series_ids)

    memory_cases = build_daily_cases(
        train_sample,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        stride_days=stride_days,
    )
    memory_cases["split"] = "memory"

    train_tail = (
        _with_series_id(train_sample)
        .sort_values(["series_id", "dt"])
        .groupby("series_id", sort=False)
        .tail(lookback_days)
        .drop(columns=["series_id"])
    )
    query_input = pd.concat([train_tail, eval_sample], ignore_index=True)
    query_cases = build_daily_cases(
        query_input,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        stride_days=horizon_days,
        series_ids=series_ids,
    )
    query_cases["split"] = "query"

    cases = pd.concat([memory_cases, query_cases], ignore_index=True)
    if memory_cases.empty or query_cases.empty:
        raise ValueError(
            f"Prepared empty cases: memory={len(memory_cases)}, query={len(query_cases)}. "
            "Check lookback_days, horizon_days, and available train/eval length."
        )

    output_path = save_cases(cases, data_cfg["cases_path"])
    return {
        "series": len(series_ids),
        "memory_cases": int(len(memory_cases)),
        "query_cases": int(len(query_cases)),
        "cases_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FreshRetailNet cases for CastCF-lite.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    summary = prepare_cases_from_config(load_config(args.config))
    print(summary)


if __name__ == "__main__":
    main()
