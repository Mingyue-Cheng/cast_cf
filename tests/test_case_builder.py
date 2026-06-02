import numpy as np
import pandas as pd

from castcf.data import build_daily_cases, sample_series_ids


def _toy_rows() -> pd.DataFrame:
    rows = []
    for store_id, product_id in [(10, 100), (20, 200)]:
        for day in range(6):
            rows.append(
                {
                    "city_id": store_id // 10,
                    "store_id": store_id,
                    "management_group_id": 1,
                    "first_category_id": 2,
                    "second_category_id": 3,
                    "third_category_id": 4,
                    "product_id": product_id,
                    "dt": f"2024-06-{day + 1:02d}",
                    "sale_amount": float(day + 1 + (product_id // 100)),
                    "hours_sale": [float(day)] * 24,
                    "stock_hour6_22_cnt": 1 if day == 3 else 0,
                    "hours_stock_status": [0] * 24,
                    "discount": 0.9 if day >= 4 else 1.0,
                    "holiday_flag": 1 if day == 5 else 0,
                    "activity_flag": 1 if day == 4 else 0,
                    "precpt": float(day),
                    "avg_temperature": 20.0 + day,
                    "avg_humidity": 60.0 + day,
                    "avg_wind_level": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_daily_cases_maps_retail_rows_to_forecasting_cases():
    cases = build_daily_cases(_toy_rows(), lookback_days=3, horizon_days=2, stride_days=1)

    assert len(cases) == 4
    first = cases.iloc[0]

    np.testing.assert_array_equal(first["x_past"], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(first["y_future"], np.array([5.0, 6.0]))
    np.testing.assert_array_equal(first["context_future"], np.array([0.95, 0.9, 0.0, 0.5, 3.5, 23.5, 63.5, 2.0]))
    np.testing.assert_array_equal(first["meta"], np.array([1, 10, 1, 2, 3, 4, 100]))

    assert first["series_id"] == "10_100"
    assert first["anchor_dt"] == "2024-06-03"
    assert bool(first["future_discounted"]) is True
    assert bool(first["future_holiday"]) is False
    assert bool(first["future_activity"]) is True
    assert bool(first["past_stockout"]) is False


def test_sample_series_ids_is_deterministic_and_limited():
    rows = _toy_rows()

    sample_a = sample_series_ids(rows, max_series=1, seed=7)
    sample_b = sample_series_ids(rows, max_series=1, seed=7)

    assert sample_a == sample_b
    assert len(sample_a) == 1
    assert sample_a[0] in {"10_100", "20_200"}
