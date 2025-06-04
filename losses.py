"""
Some loss functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn.functional as F
from fk import compute_forward_kinematics

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

def mpjpe(pred_mats, targ_mats):
    """
    Mean Per-Joint Position Error (MPJPE).

    Args:
        pred_mats: Tensor of shape (B, T, J, 3, 3), predicted rotation matrices for each joint.
        targ_mats: Tensor of shape (B, T, J, 3, 3), ground-truth rotation matrices.

    Returns:
        Scalar tensor: the average L2 distance between predicted and target joint positions over all batches, frames, and joints.
    """
    # Compute 3D joint positions via forward kinematics
    # Assumes compute_forward_kinematics returns tensor of shape (B, T, J, 3)
    pred_xyz = compute_forward_kinematics(pred_mats)
    targ_xyz = compute_forward_kinematics(targ_mats)
    # Compute L2 distance per joint
    dist = torch.norm(pred_xyz - targ_xyz, dim=-1)  # shape: (B, T, J)
    return torch.mean(dist)


def geodesic_loss(pred_mats, targ_mats, eps=1e-7):
    """
    Geodesic loss on SO(3): average rotation angle between predicted and target.

    Args:
        pred_mats: Tensor of shape (B, T, J, 3, 3).
        targ_mats: Tensor of shape (B, T, J, 3, 3).
        eps: small constant for numerical stability.

    Returns:
        Scalar tensor: average geodesic distance (radians).
    """
    # Relative rotation: R_rel = pred * targ^T
    targ_transpose = targ_mats.transpose(-1, -2)
    R_rel = torch.matmul(pred_mats, targ_transpose)  # (B, T, J, 3, 3)
    # Compute trace
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B, T, J)
    # Compute the angle via arccos((trace - 1)/2)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta_clamped = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta_clamped)
    return torch.mean(theta)


def angle_loss(pred_mats, targ_mats, eps=1e-7):
    """
    Squared-angle loss: squares the geodesic distance on SO(3).

    Args:
        pred_mats: Tensor of shape (B, T, J, 3, 3).
        targ_mats: Tensor of shape (B, T, J, 3, 3).
        eps: small constant for numerical stability.

    Returns:
        Scalar tensor: average squared rotation angle.
    """
    # Relative rotation
    targ_transpose = targ_mats.transpose(-1, -2)
    R_rel = torch.matmul(pred_mats, targ_transpose)  # (B, T, J, 3, 3)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B, T, J)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta_clamped = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta_clamped)
    return torch.mean(theta ** 2)
