import numpy as np

from inpainting_research_agent.core.masks import split_observed_mask


def test_observed_split_is_disjoint_complete_and_reproducible():
    observed = np.ones((10, 12), dtype=np.bool_)
    observed[2:5, 4:8] = False

    first_train, first_validation = split_observed_mask(observed, 0.2, seed=13)
    second_train, second_validation = split_observed_mask(observed, 0.2, seed=13)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_validation, second_validation)
    assert not np.logical_and(first_train, first_validation).any()
    assert np.array_equal(np.logical_or(first_train, first_validation), observed)
    assert not first_train[~observed].any()
    assert not first_validation[~observed].any()
    assert first_train.any()
    assert first_validation.any()

