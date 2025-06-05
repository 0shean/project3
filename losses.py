"""
Some loss functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch


# --- safe SO(3) log map ----------------------------------------------
def mat_to_axis_angle(R, eps=1e-6):
    """
    R: (..., 3, 3) rotation matrices  (not necessarily perfect)
    returns (..., 3) axis-angle vectors
    """
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)

    angle = torch.acos(cos)                          # (...,)

    # ‖sin‖ might be tiny when angle≈0 → avoid /0
    sin = torch.sin(angle).clamp(min=eps)

    # off-diagonal “vee” operator
    vx = R[..., 2, 1] - R[..., 1, 2]
    vy = R[..., 0, 2] - R[..., 2, 0]
    vz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack((vx, vy, vz), dim=-1) / (2 * sin.unsqueeze(-1))

    return axis * angle.unsqueeze(-1)                # axis-angle


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

def velocity_loss(seq):
    vel = seq[:,1:] - seq[:,:-1]        # finite difference
    return vel.abs().mean()

def velocity_diff_loss(pred, targ):
    """
    Finite-difference velocity L1 between pred & target.
    pred, targ: [B, T, D] (rotation-mat or xyz)
    """
    v_pred = pred[:, 1:] - pred[:, :-1]
    v_targ = targ[:, 1:] - targ[:, :-1]
    return (v_pred - v_targ).abs().mean()

def local_to_global(rot_local, parents):
    """
    Converts local joint rotations to global rotations using the kinematic chain.
    Args:
        rot_local: (B, T, J, 3, 3) rotation matrices in local frame
        parents: list of int, where -1 indicates root joint
    Returns:
        (B, T, J, 3, 3) global rotation matrices
    """
    B, T, J, _, _ = rot_local.shape
    glob = [None] * J
    for j in range(J):
        p = parents[j]
        if p == -1:
            glob[j] = rot_local[..., j, :, :]
        else:
            glob[j] = torch.matmul(glob[p], rot_local[..., j, :, :])
    return torch.stack(glob, dim=2)

def joint_angle_loss(pred_mat, targ_mat, parents, eps=1e-6):
    """
    Computes mean angular error (in radians) between predicted and ground-truth joint rotations.
    Args:
        pred_mat: (B, T, J, 3, 3) predicted rotation matrices
        targ_mat: (B, T, J, 3, 3) ground-truth rotation matrices
        parents: list[int] of joint parent indices
    Returns:
        scalar: average joint angle error (radians)
    """
    parents = parents[1:]
    pred_mat = pred_mat[:, :, 1:]
    targ_mat = targ_mat[:, :, 1:]

    Rg_pred = local_to_global(pred_mat, parents)
    Rg_targ = local_to_global(targ_mat, parents)

    u, _, v = torch.linalg.svd(pred_mat)
    pred_mat = torch.matmul(u, v.transpose(-1, -2))

    R_err = torch.matmul(Rg_pred, Rg_targ.transpose(-1, -2))

    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)
    return angle.mean()

def bone_length_loss(pred, parents):
    B, T, J, _, _ = pred.shape
    diffs = []
    for j, p in enumerate(parents):
        if p == -1:
            continue
        pred_j = pred[..., j, :, 2]  # z-axis
        pred_p = pred[..., p, :, 2]
        dist = (pred_j - pred_p).norm(dim=-1)
        diffs.append(dist)
    diffs = torch.stack(diffs, dim=-1)
    diffs_var = diffs.var(dim=-1).mean()  # variance across joints
    return diffs_var