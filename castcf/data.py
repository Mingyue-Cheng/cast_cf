from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ENTITY_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]

CONTEXT_COLUMNS = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


def load_freshretail_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load a FreshRetailNet parquet split."""
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"FreshRetailNet split not found: {path}")
    return pd.read_parquet(path)


def _series_id_frame(df: pd.DataFrame) -> pd.Series:
    return df["store_id"].astype(str) + "_" + df["product_id"].astype(str)


def sample_series_ids(df: pd.DataFrame, max_series: int | None, seed: int = 42) -> list[str]:
    """Return deterministic store-product series ids."""
    series_ids = np.array(sorted(_series_id_frame(df).unique()))
    if max_series is None or max_series >= len(series_ids):
        return series_ids.tolist()
    rng = np.random.default_rng(seed)
    picked = rng.choice(series_ids, size=max_series, replace=False)
    return sorted(picked.tolist())


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).mean())


def _max_bool(values: pd.Series) -> bool:
    return bool((pd.to_numeric(values, errors="coerce").fillna(0) > 0).any())


def _context_summary(window: pd.DataFrame) -> np.ndarray:
    """Build context features available before the horizon executes."""
    discount = pd.to_numeric(window["discount"], errors="coerce").fillna(1.0)
    return np.array(
        [
            float(discount.mean()),
            float(discount.min()),
            float((window["holiday_flag"] > 0).mean()),
            float((window["activity_flag"] > 0).mean()),
            _mean(window["precpt"]),
            _mean(window["avg_temperature"]),
            _mean(window["avg_humidity"]),
            _mean(window["avg_wind_level"]),
        ],
        dtype=float,
    )


def _past_context_summary(window: pd.DataFrame) -> np.ndarray:
    stockout = pd.to_numeric(window["stock_hour6_22_cnt"], errors="coerce").fillna(0)
    return np.concatenate(
        [
            _context_summary(window),
            np.array(
                [
                    float(stockout.mean()),
                    float((stockout > 0).mean()),
                ],
                dtype=float,
            ),
        ]
    )


def _list_mean(values: Iterable[object]) -> float:
    flattened: list[float] = []
    for value in values:
        if isinstance(value, np.ndarray):
            flattened.extend(value.astype(float).tolist())
        elif isinstance(value, list):
            flattened.extend(float(x) for x in value)
    if not flattened:
        return 0.0
    return float(np.mean(flattened))


def build_daily_cases(
    rows: pd.DataFrame,
    lookback_days: int,
    horizon_days: int,
    stride_days: int = 1,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Convert daily FreshRetailNet rows into CastCF forecasting cases.

    `future_stockout` is emitted only as a diagnostic flag and is deliberately
    excluded from `context_future` to avoid leakage.
    """
    if lookback_days < 1 or horizon_days < 1 or stride_days < 1:
        raise ValueError("lookback_days, horizon_days, and stride_days must be positive")

    required = set(ENTITY_COLUMNS + CONTEXT_COLUMNS + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = rows.copy()
    work["series_id"] = _series_id_frame(work)
    if series_ids is not None:
        allowed = set(series_ids)
        work = work[work["series_id"].isin(allowed)]

    case_rows: list[dict[str, object]] = []
    for series_id, group in work.sort_values(["series_id", "dt"]).groupby("series_id", sort=False):
        group = group.reset_index(drop=True)
        total = len(group)
        if total < lookback_days + horizon_days:
            continue
        for future_start in range(lookback_days, total - horizon_days + 1, stride_days):
            past = group.iloc[future_start - lookback_days : future_start]
            future = group.iloc[future_start : future_start + horizon_days]
            meta = group.iloc[0][ENTITY_COLUMNS].to_numpy(dtype=np.int64)
            stockout_future = pd.to_numeric(future["stock_hour6_22_cnt"], errors="coerce").fillna(0)
            hours_sale_mean = _list_mean(future["hours_sale"]) if "hours_sale" in future.columns else 0.0
            case_rows.append(
                {
                    "case_id": f"{series_id}@{past.iloc[-1]['dt']}",
                    "series_id": series_id,
                    "anchor_dt": str(past.iloc[-1]["dt"]),
                    "x_past": past["sale_amount"].to_numpy(dtype=float),
                    "y_future": future["sale_amount"].to_numpy(dtype=float),
                    "context_past": _past_context_summary(past),
                    "context_future": _context_summary(future),
                    "meta": meta,
                    "future_discounted": bool((future["discount"] < 1.0).any()),
                    "future_holiday": _max_bool(future["holiday_flag"]),
                    "future_activity": _max_bool(future["activity_flag"]),
                    "past_stockout": _max_bool(past["stock_hour6_22_cnt"]),
                    "future_stockout": bool((stockout_future > 0).any()),
                    "future_hours_sale_mean": hours_sale_mean,
                }
            )

    return pd.DataFrame(case_rows)


def save_cases(cases: pd.DataFrame, output_path: str | Path) -> Path:
    """Save cases to parquet, creating the parent directory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(path, index=False)
    return path

