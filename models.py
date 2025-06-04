"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import AMASSBatch
from losses import mpjpe, angle_loss, geodesic_loss
from fk import SMPL_MAJOR_JOINTS, SMPL_JOINTS, compute_forward_kinematics
import random

def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return BaseModel(config)

def rot6d_to_rotmat(x):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Input: x of shape (..., 6)
    Output: R of shape (..., 3, 3)
    Follows Zhou et al. "On the Continuity of Rotation Representations in Neural Networks".
    """
    # Split into two 3D vectors
    x = x.view(*x.shape[:-1], 3, 2)  # (..., 3, 2)
    a1 = x[..., :, 0]                # (..., 3)
    a2 = x[..., :, 1]                # (..., 3)

    # Normalize first vector
    b1 = F.normalize(a1, dim=-1)     # (..., 3)
    # Make a2 orthogonal to b1
    dot = (b1 * a2).sum(dim=-1, keepdim=True)  # (..., 1)
    a2_ortho = a2 - dot * b1                  # (..., 3)
    b2 = F.normalize(a2_ortho, dim=-1)        # (..., 3)
    # Third vector via cross product
    b3 = torch.cross(b1, b2, dim=-1)          # (..., 3)

    # Stack as columns
    R = torch.stack([b1, b2, b3], dim=-1)     # (..., 3, 3)
    return R
class StructuredPrediction6D(nn.Module):
    def __init__(self, in_dim, joint_dim, joint_names):
        super().__init__()
        self.joint_names = joint_names
        # joint_dim now = 6 for 6D representation
        self.joint_mlps = nn.ModuleDict({
            j: nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, joint_dim)
            ) for j in joint_names
        })

    def forward(self, x):
        # x: [B, in_dim]
        outs = [ self.joint_mlps[j](x) for j in self.joint_names ]
        # concatenate: [B, num_joints * 6]
        return torch.cat(outs, dim=-1)


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.input_dim         # e.g., 135 (15 joints * 9 dims)
        self.hidden_size = config.hidden_size      # e.g., 512
        self.num_layers = config.num_layers        # e.g., 2
        self.pred_frames = config.output_n         # e.g., 24
        self.dropout = config.dropout              # e.g., 0.1
        self.sched_sampling_prob = 1.0             # initial teacher-forcing ratio
        self.sched_sampling_decay = config.sched_sampling_decay  # e.g., 0.05 per epoch

        # GRU for autoregressive modeling (we add dropout on inputs manually)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        # Two-layer residual head → hidden to hidden
        self.mlp_pre = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        # DCT-based motion attention (unchanged)
        self.motion_att = DCTMotionAttention(input_dim=self.input_size, hidden_dim=self.hidden_size, heads=4)

        # Structured prediction: now outputs 6D per joint
        joint_names = [SMPL_JOINTS[i] for i in SMPL_MAJOR_JOINTS]
        joint_dim = 6
        self.spl = StructuredPrediction6D(in_dim=self.hidden_size, joint_dim=joint_dim, joint_names=joint_names)

    def forward(self, batch, epoch=None):
        seq = batch.poses                        # shape (B, T_total, D)
        input_seq = seq[:, :120]                 # seed frames
        target_seq = seq[:, 120:]                # future frames for loss

        B, T_in, D = input_seq.shape  # D = 135

        # Compute motion context via DCT attention
        motion_ctx = self.motion_att(input_seq)   # [B, hidden_size]

        # Encode seed sequence via GRU
        _, h = self.gru(input_seq, None)          # h: (num_layers, B, H)
        # Last ground truth frame
        x_t = input_seq[:, -1]                    # [B, D]
        outputs_6d = []

        # For tracking velocity regularization, store previous two 6D preds
        prev6d_1 = None
        prev6d_2 = None

        for t in range(self.pred_frames):
            # Optionally apply dropout on input to GRU at each step
            x_t_dropped = F.dropout(x_t, p=self.dropout, training=self.training)
            x_t_input = x_t_dropped.unsqueeze(1)  # [B,1,D]

            # One GRU step
            out, h = self.gru(x_t_input, h)       # out: [B,1,H]
            hidden = self.mlp_pre(out.squeeze(1))  # [B, H]
            hidden = hidden + motion_ctx           # add motion context

            # Structured prediction to 6D per joint
            delta6d = self.spl(hidden)             # [B, J*6]

            # Add delta in 6D space: x_t6d = x_t6d_prev + delta6d
            # First, convert last predicted 3x3 rotations to 6D if t==0
            if t == 0:
                # Convert x_t from 3x3 mat to 6D rep by taking first two columns
                # We assume seed is valid rotation matrices
                x_t6d = self.mat_to_6d(x_t)
            else:
                x_t6d = prev6d_1

            # Current 6D prediction
            new6d = x_t6d + delta6d                # [B, J*6]
            outputs_6d.append(new6d)

            # Convert new6d to 3x3 for next input
            # reshape to (B, J, 6)
            new6d_reshaped = new6d.view(B, len(SMPL_MAJOR_JOINTS), 6)
            R_pred = rot6d_to_rotmat(new6d_reshaped)  # [B, J, 3,3]

            # Flatten back to (B, D)
            R_flat = R_pred.view(B, -1)             # [B, J*9]

            # If some joints not in major, copy identity for them (optional)
            # Here we assume D covers only major joints
            x_t = R_flat

            # Update previous 6D history for velocity loss
            prev6d_2 = prev6d_1
            prev6d_1 = new6d

        # Stack all predicted 6D outputs: list of (B, J*6) → (B, T_pred, J*6)
        pred_seq_6d = torch.stack(outputs_6d, dim=1)

        if self.training:
            return {
                'pred6d': pred_seq_6d,   # predicted 6D representations
                'target': target_seq,     # ground truth 3x3 rotations
                'seed_last_6d': self.mat_to_6d(input_seq[:, -1])  # last ground truth 6D
            }
        else:
            return {
                'pred6d': pred_seq_6d,
                'seed_last_6d': self.mat_to_6d(input_seq[:, -1])
            }

    def mat_to_6d(self, mat_flat):
        """
        Convert flattened 3x3 rotations (shape [B, J*9]) to 6D by taking first two columns.
        Assumes mat_flat is exact rotation matrices.
        """
        B = mat_flat.shape[0]
        J = len(SMPL_MAJOR_JOINTS)
        R = mat_flat.view(B, J, 3, 3)
        # take first two columns for each joint
        col1 = R[..., :, 0]  # [B, J, 3]
        col2 = R[..., :, 1]  # [B, J, 3]
        sixd = torch.cat([col1, col2], dim=-1)  # [B, J, 6]
        return sixd.view(B, J * 6)

    def backward(self, batch, model_out, do_backward=True):
        B = batch.batch_size
        # Extract predictions and targets
        pred6d = model_out['pred6d']                 # [B, T_pred, J*6]
        target_mat = model_out['target']             # [B, T_pred, J*9]
        seed_last_6d = model_out['seed_last_6d']     # [B, J*6]

        T = pred6d.shape[1]
        J = len(SMPL_MAJOR_JOINTS)

        # Convert predicted 6D -> rotation matrices per joint per frame
        pred6d_reshaped = pred6d.view(B, T, J, 6)
        predR = rot6d_to_rotmat(pred6d_reshaped)     # [B, T, J, 3, 3]

        # Reshape target to (B, T, J, 3,3)
        targ_mat = target_mat.view(B, T, J, 3, 3)

        # Geodesic loss on rotations
        loss_geo = geodesic_loss(predR, targ_mat)

        # MPJPE in 3D joint space (requires forward kinematics to get joint positions)
        # Compute 3D joint positions from rotation matrices
        # seed root for t=0 is from last seed frame
        # Build full poses array (B, T, J, 3,3)
        fullR = predR
        # Convert rotations to 3D joint positions (calls a user-provided FK function)
        # Suppose compute_forward_kinematics takes R of shape [B, T, J, 3,3] and returns positions [B, T, J,3]
        pred_xyz = compute_forward_kinematics(fullR)
        targ_xyz = compute_forward_kinematics(targ_mat)
        loss_mpjpe = F.mse_loss(pred_xyz, targ_xyz)

        # Velocity regularization: encourage smooth changes in 6D space
        vel_loss = 0.0
        for t in range(2, T):
            v_t = pred6d[:, t] - pred6d[:, t-1]
            v_t1 = pred6d[:, t-1] - pred6d[:, t-2]
            vel_loss = vel_loss + F.mse_loss(v_t, v_t1)
        vel_loss = vel_loss / (T - 2)

        # Bone-length consistency: ensure each bone length matches ground truth
        # pred_xyz and targ_xyz: [B, T, J, 3]
        bone_loss = 0.0
        # Suppose compute_forward_kinematics returns joint positions with indices matching SMPL_MAJOR_JOINTS
        parent_indices = self._get_parent_indices()
        for t in range(T):
            for j, p in enumerate(parent_indices):
                if p < 0:
                    continue
                pred_len = torch.norm(pred_xyz[:, t, j] - pred_xyz[:, t, p], dim=-1)
                targ_len = torch.norm(targ_xyz[:, t, j] - targ_xyz[:, t, p], dim=-1)
                bone_loss = bone_loss + F.mse_loss(pred_len, targ_len)
        bone_loss = bone_loss / (T * J)

        # Total loss: weighted sum
        lambda_geo = 1.0
        lambda_mpjpe = 1.0
        lambda_vel = 0.1
        lambda_bone = 0.5

        total_loss = lambda_geo * loss_geo + lambda_mpjpe * loss_mpjpe + lambda_vel * vel_loss + lambda_bone * bone_loss

        if do_backward:
            total_loss.backward()

        loss_dict = {
            'geodesic_loss': loss_geo.item(),
            'mpjpe_xyz': loss_mpjpe.item(),
            'vel_loss': vel_loss.item(),
            'bone_loss': bone_loss.item(),
            'total_loss': total_loss.item()
        }
        return loss_dict, targ_xyz

    def model_name(self):
        return '{}-lr{}-6D'.format(self.__class__.__name__, self.config.lr)

    def _get_parent_indices(self):
        """
        Return list of parent indices for each SMPL_MAJOR_JOINTS joint.
        Assumes SMPL_JOINTS gives full list; SMPL_MAJOR_JOINTS gives indices of major joints.
        """
        # Build a map from joint name to parent name (you may have a dictionary somewhere)
        # For demo, assume a simple chain
        parent_map = {
            'root': -1,
            'spine': 0,
            'chest': 1,
            'neck': 2,
            'head': 3,
            'left_shoulder': 1,
            'left_elbow': 5,
            'left_wrist': 6,
            'right_shoulder': 1,
            'right_elbow': 8,
            'right_wrist': 9,
            'left_hip': 0,
            'left_knee': 12,
            'left_ankle': 13,
            'right_hip': 0,
            'right_knee': 15,
            'right_ankle': 16
        }
        joint_names = [SMPL_JOINTS[i] for i in SMPL_MAJOR_JOINTS]
        indices = []
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        for name in joint_names:
            parent_name = parent_map.get(name, None)
            if parent_name is None or parent_name not in name_to_idx:
                indices.append(-1)
            else:
                indices.append(name_to_idx[parent_name])
        return indices


# Keep the original DCTMotionAttention unchanged
class DCTMotionAttention(nn.Module):
    """
    Compute a motion context by taking a real-FFT (as a proxy for DCT-II)
    over the seed sequence for each joint-dimension, projecting into hidden
    space, running a small self-attention, and averaging over frequency.
    """
    def __init__(self, input_dim, hidden_dim, heads=4):
        super().__init__()
        self.freq_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)

    def forward(self, seed_seq):
        freq = torch.fft.rfft(seed_seq, dim=1).real
        feat = self.freq_proj(freq)
        attn_out, _ = self.attn(feat, feat, feat)
        context = attn_out.mean(dim=1)
        return context