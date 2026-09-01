import numpy as np

from inpainting_research_agent.core.masks import generate_observation_mask


def test_random_mask_is_exact_and_reproducible():
    first = generate_observation_mask(20, 30, 0.4, "random", seed=7)
    second = generate_observation_mask(20, 30, 0.4, "random", seed=7)

    assert first.dtype == np.bool_
    assert np.array_equal(first, second)
    assert np.count_nonzero(~first) == round(20 * 30 * 0.4)
    assert first.any()
    assert (~first).any()


def test_block_mask_is_exact_and_seeded():
    first = generate_observation_mask(25, 31, 0.3, "block", seed=11)
    second = generate_observation_mask(25, 31, 0.3, "block", seed=11)

    assert np.array_equal(first, second)
    assert np.count_nonzero(~first) == round(25 * 31 * 0.3)
