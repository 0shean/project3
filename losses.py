# losses.py
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

# --- new: bone-length and joint-limit --------------------------------
# Assumes input rotation matrices: [B, T, J, 3, 3]

def bone_length_loss(pred_mat, targ_mat):
    """
    Penalize differences in bone lengths: the distance between joint centers.
    Simplest: compare the distance of each joint transformed by pred vs targ.
    Here we assume joint positions in rest pose are known. For simplicity,
    we approximate bone length by taking the norms of the first column of R.
    """
    # pred_mat/targ_mat shape: [B, T, J, 3, 3]
    # Use column 0 as a proxy for bone direction * length (unit bone lengths).
    p0 = pred_mat[..., :, 0]  # [B, T, J, 3]
    t0 = targ_mat[..., :, 0]  # [B, T, J, 3]
    # bone length proxy: norm of p0, t0
    lp = torch.norm(p0, dim=-1)  # [B, T, J]
    lt = torch.norm(t0, dim=-1)  # [B, T, J]
    return (lp - lt).abs().mean()


def joint_limit_loss(pred_mat, limit=0.7854):  # default ~45° limit
    """
    Penalize rotations exceeding a joint limit (in radians) about any axis.
    We compute axis-angle for each pred_mat and clamp angle at 'limit'.
    """
    # pred_mat: [B, T, J, 3, 3]
    # Compute axis-angle via mat_to_axis_angle helper (B*T*J,3)
    B, T, J, _, _ = pred_mat.shape
    flat = pred_mat.reshape(-1, 3, 3)        # [B*T*J, 3, 3]
    aa   = mat_to_axis_angle(flat)           # [B*T*J, 3]
    angle = torch.norm(aa, dim=-1)           # [B*T*J]
    # compute amount over limit
    over = torch.relu(angle - limit)
    return over.mean()
