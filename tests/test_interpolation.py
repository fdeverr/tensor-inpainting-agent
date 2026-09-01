import numpy as np
import pytest

from inpainting_research_agent.core.interpolation import nearest_neighbor_fill


def test_nearest_neighbor_fill_preserves_observations_and_fills_hole():
    observed = np.zeros((5, 5, 3), dtype=np.float32)
    mask = np.ones((5, 5), dtype=np.bool_)
    mask[1:4, 1:4] = False
    observed[mask] = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    reconstruction = nearest_neighbor_fill(observed, mask)

    assert np.array_equal(reconstruction[mask], observed[mask])
    assert np.allclose(reconstruction[~mask], [0.2, 0.5, 0.8])
    assert np.isfinite(reconstruction).all()


def test_nearest_neighbor_fill_rejects_fully_missing_image():
    observed = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.bool_)

    with pytest.raises(ValueError, match="at least one observed"):
        nearest_neighbor_fill(observed, mask)
