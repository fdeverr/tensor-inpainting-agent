import numpy as np
import pytest
import torch
from torch import nn

from inpainting_research_agent.core.models import create_model


MODEL_CASES = (
    (
        "matrix",
        {"rank": 4, "init_scale": 0.1},
        {"left_factor", "right_factor", "channel_bias"},
    ),
    (
        "cp",
        {"rank": 4, "init_scale": 0.2},
        {"height_factor", "width_factor", "channel_factor", "channel_bias"},
    ),
    (
        "tucker",
        {"rank_h": 4, "rank_w": 3, "rank_c": 2, "init_scale": 0.15},
        {"core", "height_factor", "width_factor", "channel_factor", "channel_bias"},
    ),
)


@pytest.mark.parametrize("model_name,hyperparameters,expected_parameters", MODEL_CASES)
def test_tensor_model_forward_backward_and_parameters(
    model_name,
    hyperparameters,
    expected_parameters,
):
    torch.manual_seed(3)
    image_shape = (8, 7, 3)
    model = create_model(
        model_name=model_name,
        image_shape=image_shape,
        initial_channel_mean=[0.4, 0.5, 0.6],
        hyperparameters=hyperparameters,
    )
    prediction = model()
    observed = torch.rand(image_shape)
    train_mask = torch.ones(image_shape[:2], dtype=torch.bool)
    train_mask[2:4, 3:5] = False
    loss = sum(model.loss_terms(prediction, observed, train_mask).values())
    loss.backward()

    named_parameters = dict(model.named_parameters())
    assert tuple(prediction.shape) == image_shape
    assert expected_parameters == set(named_parameters)
    assert all(isinstance(parameter, nn.Parameter) for parameter in named_parameters.values())
    assert all(parameter.grad is not None for parameter in named_parameters.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in named_parameters.values())


def test_tucker_rejects_rank_larger_than_image_dimension():
    with pytest.raises(ValueError, match="Tucker ranks"):
        create_model(
            model_name="tucker",
            image_shape=(8, 7, 3),
            initial_channel_mean=np.full(3, 0.5),
            hyperparameters={"rank_h": 9, "rank_w": 3, "rank_c": 2},
        )
