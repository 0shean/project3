"""
Some loss functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch

def mse(predictions, targets):
    """
    Compute the MSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The MSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    return loss_per_sample_and_seq.mean()

def mpjpe(predict, target):
    return torch.mean(torch.norm(predict - target, dim=-1))

def angle_loss(predict, target):
    return torch.mean(torch.acos(
        torch.clamp(
            ((predict * target).sum(dim=-1)) /
            (predict.norm(dim=-1) * target.norm(dim=-1) + 1e-8),
            -1.0, 1.0
        )
    ))


