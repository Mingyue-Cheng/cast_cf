import numpy as np

from castcf.features import overlap_exclusion_mask, standardize_by_reference


def test_standardize_by_reference_uses_reference_statistics_and_handles_constant_columns():
    query = np.array([[12.0, 5.0], [8.0, 5.0]])
    reference = np.array([[10.0, 5.0], [14.0, 5.0]])

    query_scaled, reference_scaled = standardize_by_reference(query, reference)

    np.testing.assert_allclose(reference_scaled[:, 0], np.array([-1.0, 1.0]))
    np.testing.assert_allclose(reference_scaled[:, 1], np.array([0.0, 0.0]))
    np.testing.assert_allclose(query_scaled[:, 0], np.array([0.0, -2.0]))
    np.testing.assert_allclose(query_scaled[:, 1], np.array([0.0, 0.0]))
    assert np.isfinite(query_scaled).all()
    assert np.isfinite(reference_scaled).all()


def test_overlap_exclusion_mask_blocks_same_series_overlapping_futures():
    query_series = np.array([0, 0, 1])
    query_anchors = np.array([10, 10, 10])
    corpus_series = np.array([0, 0, 0, 1])
    corpus_anchors = np.array([9, 12, 10, 10])

    mask = overlap_exclusion_mask(
        query_series, query_anchors, corpus_series, corpus_anchors, min_anchor_gap=2
    )

    # Same series: anchors 9 and 10 (gap 1 < horizon 2) overlap; gap 2 does not.
    assert mask[0].tolist() == [True, False, True, False]
    # Different series never overlaps, even at the same anchor.
    assert mask[2].tolist() == [False, False, False, True]

