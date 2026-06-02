import numpy as np

from castcf.retrieval import (
    aggregate_neighbor_futures,
    castcf_lite_neighbors,
    castcf_lite_search,
    castcf_multiroute_search,
    cosine_similarity_matrix,
    shape_knn_neighbors,
    shape_knn_search,
)


def test_cosine_similarity_handles_zero_vectors_without_nan():
    left = np.array([[1.0, 0.0], [0.0, 0.0]])
    right = np.array([[1.0, 0.0], [0.0, 1.0]])

    sims = cosine_similarity_matrix(left, right)

    np.testing.assert_allclose(sims, np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_shape_knn_excludes_self_and_returns_most_similar_shape():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [10.0, 1.0, 0.0],
        ]
    )

    neighbors, scores = shape_knn_neighbors(x, k=1)

    assert neighbors.shape == (3, 1)
    assert scores.shape == (3, 1)
    assert neighbors[0, 0] == 1
    assert neighbors[1, 0] == 0


def test_shape_knn_search_retrieves_corpus_indices_for_queries():
    query_x = np.array([[1.0, 2.0, 3.0]])
    corpus_x = np.array([[9.0, 0.0, 0.0], [1.1, 2.1, 3.1]])

    neighbors, scores = shape_knn_search(query_x, corpus_x, k=1)

    assert neighbors.tolist() == [[1]]
    assert scores.shape == (1, 1)


def test_castcf_lite_search_reranks_corpus_candidates_with_context():
    query_x = np.array([[1.0, 2.0, 3.0]])
    corpus_x = np.array([[1.01, 2.01, 3.01], [1.3, 2.3, 3.3], [9.0, 0.0, 0.0]])
    query_context = np.array([[1.0, 0.0]])
    corpus_context = np.array([[0.0, 1.0], [1.0, 0.1], [0.0, 1.0]])
    query_meta = np.ones((1, 1))
    corpus_meta = np.ones((3, 1))

    neighbors, _ = castcf_lite_search(
        query_x,
        query_context,
        query_meta,
        corpus_x,
        corpus_context,
        corpus_meta,
        k=1,
        candidate_k=2,
        shape_weight=0.1,
        context_weight=0.9,
        meta_weight=0.0,
    )

    assert neighbors.tolist() == [[1]]


def test_castcf_multiroute_search_recovers_context_candidate_outside_shape_top1():
    query_x = np.array([[1.0, 2.0, 3.0]])
    corpus_x = np.array(
        [
            [1.01, 2.01, 3.01],
            [9.0, 8.0, 7.0],
            [4.0, 0.0, 0.0],
        ]
    )
    query_context = np.array([[1.0, 0.0]])
    corpus_context = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query_meta = np.ones((1, 1))
    corpus_meta = np.ones((3, 1))

    lite_neighbors, _ = castcf_lite_search(
        query_x,
        query_context,
        query_meta,
        corpus_x,
        corpus_context,
        corpus_meta,
        k=1,
        candidate_k=1,
        shape_weight=0.1,
        context_weight=0.9,
        meta_weight=0.0,
    )
    multi_neighbors, _ = castcf_multiroute_search(
        query_x,
        query_context,
        query_meta,
        corpus_x,
        corpus_context,
        corpus_meta,
        k=1,
        route_k=1,
        shape_weight=0.1,
        context_weight=0.9,
        meta_weight=0.0,
    )

    assert lite_neighbors.tolist() == [[0]]
    assert multi_neighbors.tolist() == [[1]]


def test_castcf_lite_reranks_shape_candidates_with_context():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.3, 2.3, 3.3],
            [1.01, 2.01, 3.01],
            [9.0, 8.0, 7.0],
        ]
    )
    context = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    meta = np.ones((4, 1))

    shape_neighbors, _ = shape_knn_neighbors(x, k=2)
    castcf_neighbors, _ = castcf_lite_neighbors(
        x,
        context,
        meta,
        k=1,
        candidate_k=2,
        shape_weight=0.1,
        context_weight=0.9,
        meta_weight=0.0,
    )

    assert list(shape_neighbors[0]) == [2, 1]
    assert castcf_neighbors[0, 0] == 1


def test_aggregate_neighbor_futures_uses_softmax_scores():
    y_future = np.array(
        [
            [0.0, 0.0],
            [10.0, 20.0],
            [20.0, 40.0],
        ]
    )
    neighbors = np.array([[1, 2]])
    scores = np.array([[2.0, 0.0]])

    pred = aggregate_neighbor_futures(y_future, neighbors, scores, temperature=1.0)

    weight_1 = np.exp(2.0) / (np.exp(2.0) + np.exp(0.0))
    weight_2 = 1.0 - weight_1
    np.testing.assert_allclose(pred[0], weight_1 * y_future[1] + weight_2 * y_future[2])
