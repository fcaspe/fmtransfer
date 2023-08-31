"""
Helpers for loss functions.
"""
import torch


class NormalizedL1Loss(torch.nn.Module):
    """
    Loss function that normalizes the specified dimension
    before computing L1 distance. This allows that all components
    of the specified dimension have the same weighting in the loss
    computation.
    The normalization is done on the target.
    """

    def __init__(self, normalized_dimension: int):
        super().__init__()
        self.normalized_dimension = normalized_dimension
        self.loss = torch.nn.L1Loss(reduction="mean")

    def forward(self, prediction, target):
        norm = torch.max(target, dim=self.normalized_dimension, keepdim=True)[0]
        norm = norm.detach()
        return self.loss(prediction / norm, target / norm)
