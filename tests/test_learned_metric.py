import numpy as np

from castcf.learned_metric import LearnedMetricScorer, pair_feature_matrix


def test_pair_feature_matrix_contains_similarity_and_difference_signals():
    query_x = np.array([[1.0, 2.0]])
    corpus_x = np.array([[1.0, 2.0], [5.0, 5.0]])
    query_context = np.array([[0.9, 1.0]])
    corpus_context = np.array([[0.9, 1.0], [0.1, 0.0]])
    query_meta = np.array([[1.0]])
    corpus_meta = np.array([[1.0], [2.0]])

    features = pair_feature_matrix(
        query_x,
        query_context,
        query_meta,
        corpus_x,
        corpus_context,
        corpus_meta,
        np.array([0, 0]),
        np.array([0, 1]),
    )

    assert features.shape == (2, 6)
    assert features[0, 0] > features[1, 0]
    assert features[0, 1] > features[1, 1]
    assert features[0, 3] > features[1, 3]


def test_learned_metric_scorer_learns_to_rank_positive_above_negative(tmp_path):
    positive_features = np.array(
        [
            [0.9, 0.8, 0.7, -0.1, -0.2, -0.1],
            [0.8, 0.9, 0.6, -0.2, -0.1, -0.2],
        ]
    )
    negative_features = np.array(
        [
            [0.1, 0.2, 0.3, -2.0, -1.5, -1.0],
            [0.2, 0.1, 0.2, -1.8, -1.4, -1.2],
        ]
    )

    scorer = LearnedMetricScorer.fit_pairwise(
        positive_features,
        negative_features,
        epochs=200,
        learning_rate=0.5,
        l2=0.0,
    )

    assert np.all(scorer.predict(positive_features) > scorer.predict(negative_features))

    path = tmp_path / "metric.npz"
    scorer.save(path)
    loaded = LearnedMetricScorer.load(path)
    np.testing.assert_allclose(loaded.predict(positive_features), scorer.predict(positive_features))

