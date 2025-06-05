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
    assert len(parents) == J, f"Parent list length {len(parents)} must match joint count {J}"
    glob = [None] * J
    for j in range(J):
        p = parents[j]
        if p == -1:
            glob[j] = rot_local[..., j, :, :]
        else:
            # By construction, glob[p] has already been set because p < j in a valid kinematic tree
            assert glob[p] is not None, f"Parent index {p} not initialized before child {j}"
            glob[j] = torch.matmul(glob[p], rot_local[..., j, :, :])
    return torch.stack(glob, dim=2)


def joint_angle_loss(pred_mat, targ_mat, parents, eps=1e-6):
    """
    Computes mean angular error (in radians) between predicted and ground-truth joint rotations.
    Excludes the root joint and remaps parent indices accordingly.
    Args:
        pred_mat: (B, T, J, 3, 3) predicted rotation matrices (including root)
        targ_mat: (B, T, J, 3, 3) ground-truth rotation matrices (including root)
        parents: list[int] of length J, where -1 indicates root joint
    Returns:
        scalar: average joint angle error (radians)
    """
    # 1) Remove root joint (index 0) from both prediction and target
    pred_mat = pred_mat[:, :, 1:]   # now shape (B, T, J-1, 3, 3)
    targ_mat = targ_mat[:, :, 1:]   # same shape
    kept_parents = parents[1:]      # length J-1

    # 2) Build a map old_index → new_index after slicing out root.
    #    Original joints were [0, 1, 2, ..., J-1]. After removing 0, the new indices are [0←1, 1←2, …].
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(range(1, len(parents)))}

    # 3) Remap each parent to the new index, or -1 if parent was root or not in kept range.
    remapped_parents = []
    for old_p in kept_parents:
        if old_p == -1 or old_p not in old_to_new:
            remapped_parents.append(-1)
        else:
            remapped_parents.append(old_to_new[old_p])
    # Now remapped_parents has length J-1, and each entry is in [-1..(J-2)].

    # 4) (Optional but recommended) Project pred_mat into SO(3) to ensure valid rotations.
    #    Compute SVD per joint, then reconstruct: R_closest = U @ Vᵀ
    u, _, v = torch.linalg.svd(pred_mat)
    pred_mat = torch.matmul(u, v.transpose(-1, -2))

    # 5) Compute global rotations for both pred and target using the remapped parents.
    Rg_pred = local_to_global(pred_mat, remapped_parents)
    Rg_targ = local_to_global(targ_mat, remapped_parents)

    # 6) Compute the geodesic angle between Rg_pred and Rg_targ at each joint:
    #    R_err = Rg_pred @ Rg_targᵀ, then angle = arccos((trace(R_err) - 1) / 2).
    R_err = torch.matmul(Rg_pred, Rg_targ.transpose(-1, -2))
    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)

    # 7) Return mean over batch, time, and joints (in radians).
    return angle.mean()


def bone_length_loss(pred, parents):
    """
    Penalizes variance in bone lengths (distance between each child joint and its parent) across joints.
    Args:
        pred: (B, T, J, 3, 3) predicted rotation matrices (global or local)
        parents: list[int] of length J, where -1 indicates root joint
    Returns:
        scalar: mean variance of bone lengths across the batch and time
    """
    B, T, J, _, _ = pred.shape
    assert len(parents) == J, f"Parent list length {len(parents)} must match joint count {J}"

    diffs = []
    for j, p in enumerate(parents):
        if p == -1:
            continue
        # Use the z-axis of each joint’s rotation matrix as a proxy for joint position
        pred_j = pred[..., j, :, 2]   # shape (B, T)
        pred_p = pred[..., p, :, 2]   # shape (B, T)
        dist = (pred_j - pred_p).norm(dim=-1)  # L2 distance (B, T)
        diffs.append(dist)
    diffs = torch.stack(diffs, dim=-1)  # shape (B, T, num_bones)
    diffs_var = diffs.var(dim=-1).mean()  # var across bones → shape (B, T), then mean over B,T
    return diffs_var

