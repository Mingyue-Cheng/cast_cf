import numpy as np
import pandas as pd

from castcf.metrics import mae, mse, nfd_at_k, query_nfd_at_k, subset_metric_table


def test_mae_and_mse_average_over_all_horizon_values():
    y_true = np.array([[1.0, 3.0], [2.0, 4.0]])
    y_pred = np.array([[2.0, 1.0], [2.0, 7.0]])

    assert mae(y_true, y_pred) == 1.5
    assert mse(y_true, y_pred) == 3.5


def test_nfd_at_k_measures_neighbor_future_distance():
    y_true = np.array(
        [
            [10.0, 10.0],
            [11.0, 10.0],
            [20.0, 20.0],
        ]
    )
    neighbors = np.array([[1, 2], [0, 2], [1, 0]])

    distances = nfd_at_k(y_true, neighbors)

    np.testing.assert_allclose(distances[0], np.mean([0.5, 10.0]))


def test_query_nfd_at_k_measures_query_to_corpus_neighbor_distance():
    y_query = np.array([[10.0, 10.0]])
    y_corpus = np.array([[11.0, 10.0], [20.0, 20.0]])
    neighbors = np.array([[0, 1]])

    distances = query_nfd_at_k(y_query, y_corpus, neighbors)

    np.testing.assert_allclose(distances[0], np.mean([0.5, 10.0]))


def test_subset_metric_table_reports_all_and_named_subsets():
    cases = pd.DataFrame(
        {
            "future_discounted": [True, False, True],
            "future_holiday": [False, True, False],
        }
    )
    y_true = np.array([[1.0], [3.0], [6.0]])
    y_pred = np.array([[2.0], [1.0], [4.0]])

    table = subset_metric_table(
        cases,
        y_true,
        y_pred,
        subset_columns=["future_discounted", "future_holiday"],
    )

    assert table["all"]["count"] == 3
    assert table["all"]["mae"] == 5.0 / 3.0
    assert table["future_discounted"]["count"] == 2
    assert table["future_discounted"]["mae"] == 1.5
    assert table["future_holiday"]["count"] == 1
    assert table["future_holiday"]["mae"] == 2.0
