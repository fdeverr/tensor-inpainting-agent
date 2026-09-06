import torch


class CandidateTensorInpaintingModel(TuckerDecomposition):
    """Add an image-space TV prior while retaining the learned tensor factors."""

    def __init__(self, image_shape, initial_channel_mean, tv_weight=0.0001, **kwargs):
        super().__init__(
            image_shape=image_shape,
            initial_channel_mean=initial_channel_mean,
            **kwargs,
        )
        if tv_weight < 0.0:
            raise ValueError("tv_weight must be non-negative")
        self.tv_weight = float(tv_weight)

    def loss_terms(self, prediction, observed, train_mask):
        terms = super().loss_terms(prediction, observed, train_mask)
        vertical_tv = torch.abs(prediction[1:, :, :] - prediction[:-1, :, :]).mean()
        horizontal_tv = torch.abs(prediction[:, 1:, :] - prediction[:, :-1, :]).mean()
        terms["tv_regularization"] = self.tv_weight * (vertical_tv + horizontal_tv)
        return terms

    @classmethod
    def search_space(cls, image_shape):
        space = dict(TuckerDecomposition.search_space(image_shape))
        space["tv_weight"] = [0.0, 0.0001, 0.0005, 0.001]
        return space
