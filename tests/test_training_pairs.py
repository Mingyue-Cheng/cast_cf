import numpy as np

from castcf.training_pairs import build_future_utility_pairs


def test_build_future_utility_pairs_selects_future_closer_positive_and_farther_negative():
    y_future = np.array(
        [
            [10.0, 10.0],
            [10.2, 10.1],
            [30.0, 30.0],
        ]
    )
    candidates = [
        np.array([1, 2]),
        np.array([0, 2]),
        np.array([0, 1]),
    ]

    pairs = build_future_utility_pairs(y_future, candidates, min_distance_margin=0.5)

    assert pairs.query_indices.tolist()[:2] == [0, 1]
    assert pairs.positive_indices.tolist()[:2] == [1, 0]
    assert pairs.negative_indices.tolist()[:2] == [2, 2]
    assert np.all(pairs.positive_distances < pairs.negative_distances)

