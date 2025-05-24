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
    return torch.mean(torch.norm(predict - target, dim=-1))  # must return a torch.Tensor

def angle_loss(predict, target):
    cos_sim = (predict * target).sum(dim=-1) / (
        predict.norm(dim=-1) * target.norm(dim=-1) + 1e-8)
    return torch.mean(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))

def geodesic_loss(pred_mat, targ_mat, eps=1e-7):
    # pred_mat, targ_mat: [B, T, J, 3, 3]
    R_err = pred_mat @ targ_mat.transpose(-1, -2)      # [B, T, J, 3, 3]
    cos_theta = (R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2] - 1) / 2
    cos_theta = cos_theta.clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)
    return theta.mean()


