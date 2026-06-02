import numpy as np

from castcf.features import standardize_by_reference


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

