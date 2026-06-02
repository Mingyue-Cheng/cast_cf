import json
from pathlib import Path

import pandas as pd

from experiments.prepare_freshretail_cases import prepare_cases_from_config
from experiments.run_retrieval_baselines import run_baselines_from_config


def _toy_split(days: int, start_day: int, series_count: int = 3) -> pd.DataFrame:
    rows = []
    for idx in range(series_count):
        store_id = 10 + idx
        product_id = 100 + idx
        for offset in range(days):
            day = start_day + offset
            rows.append(
                {
                    "city_id": idx,
                    "store_id": store_id,
                    "management_group_id": 1,
                    "first_category_id": 2,
                    "second_category_id": 3,
                    "third_category_id": 4,
                    "product_id": product_id,
                    "dt": f"2024-06-{day:02d}",
                    "sale_amount": float(day + idx),
                    "hours_sale": [float(day + idx)] * 24,
                    "stock_hour6_22_cnt": 1 if day % 4 == 0 else 0,
                    "hours_stock_status": [0] * 24,
                    "discount": 0.9 if idx % 2 == 0 else 1.0,
                    "holiday_flag": 1 if day % 5 == 0 else 0,
                    "activity_flag": 1 if day % 3 == 0 else 0,
                    "precpt": float(day % 2),
                    "avg_temperature": 20.0 + day,
                    "avg_humidity": 60.0 + day,
                    "avg_wind_level": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_and_run_baselines_on_toy_freshretail_files(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _toy_split(days=6, start_day=1).to_parquet(data_dir / "train.parquet", index=False)
    _toy_split(days=2, start_day=7).to_parquet(data_dir / "eval.parquet", index=False)

    config = {
        "data": {
            "data_dir": str(data_dir),
            "cases_path": str(tmp_path / "cases.parquet"),
            "metrics_path": str(tmp_path / "metrics.json"),
            "max_series": 3,
            "seed": 1,
            "lookback_days": 3,
            "horizon_days": 2,
            "stride_days": 1,
        },
        "retrieval": {
            "k": 2,
            "candidate_k": 3,
            "temperature": 1.0,
            "shape_weight": 0.2,
            "context_weight": 0.7,
            "meta_weight": 0.1,
        },
    }

    prepare_summary = prepare_cases_from_config(config)
    run_summary = run_baselines_from_config(config)

    assert prepare_summary["memory_cases"] == 6
    assert prepare_summary["query_cases"] == 3
    assert run_summary["query_cases"] == 3

    metrics = json.loads(Path(config["data"]["metrics_path"]).read_text())
    assert set(metrics["methods"]) == {
        "recent",
        "shape_knn",
        "context_knn",
        "castcf_lite",
        "castcf_multiroute",
    }
    assert metrics["methods"]["shape_knn"]["nfd_at_k"] >= 0.0
    assert metrics["methods"]["castcf_multiroute"]["nfd_at_k"] >= 0.0
