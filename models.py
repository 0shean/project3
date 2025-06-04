"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import AMASSBatch
from losses import mse
from losses import mpjpe, angle_loss
from data import AMASSBatch
from losses import mpjpe, geodesic_loss, angle_loss, velocity_diff_loss
from fk import SMPL_MAJOR_JOINTS, SMPL_JOINTS
import torch.fft

import math
import random

# fk.py already contains SMPL_PARENTS and SMPL_MAJOR_JOINTS
from fk import SMPL_PARENTS            # list[int] len = n_joints
from fk import SMPL_JOINTS             # list[str] joint-name lookup
from losses import mat_to_axis_angle

def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return BaseModel(config)
def local_to_global(rot_local, parents):
    B, T, J, _, _ = rot_local.shape
    glob = [None] * J
    for j in range(J):
        p = parents[j]
        if p == -1:
            glob[j] = rot_local[..., j, :, :]
        else:
            glob[j] = torch.matmul(glob[p], rot_local[..., j, :, :])
    return torch.stack(glob, dim=2)

def so3_log(R, eps=1e-6):
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos)
    sin_theta = torch.sin(theta)
    mask = sin_theta.abs() < 1e-4
    scale = torch.where(mask, 0.5 + (theta ** 2) / 12, theta / (2 * sin_theta))
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    return scale.unsqueeze(-1) * w

def joint_angle_loss(pred_mat, targ_mat, parents, eps=1e-6):
    Rg_pred = local_to_global(pred_mat, parents)
    Rg_targ = local_to_global(targ_mat, parents)
    R_err = torch.matmul(Rg_pred, Rg_targ.transpose(-1, -2))
    cos = ((R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2] - 1) / 2).clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)
    return angle.mean()

# Placeholder loss functions that should exist elsewhere in your repo
mpjpe = lambda p, t: ((p - t) ** 2).sum(dim=-1).sqrt().mean()
geodesic_loss = lambda p, t: ((p - t) ** 2).mean()
velocity_diff_loss = lambda p, t: ((p[:, 1:] - p[:, :-1]) - (t[:, 1:] - t[:, :-1])).abs().mean()

# ───────────────────────────────────────────────────────────────────────────────
#  Model components
# ───────────────────────────────────────────────────────────────────────────────

class DCTMotionAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=4):
        super().__init__()
        self.freq_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)

    def forward(self, seed_seq):
        freq = torch.fft.rfft(seed_seq, dim=1).real
        feat = self.freq_proj(freq)
        attn_out, _ = self.attn(feat, feat, feat)
        return attn_out.mean(dim=1)

class HierarchicalSPL(nn.Module):
    def __init__(self, in_dim, joint_dim, parents, mode="dense", hidden_per_joint=64):
        super().__init__()
        self.parents = parents
        self.joint_dim = joint_dim
        self.mode = mode
        self.mlps = nn.ModuleList()
        for j, p in enumerate(parents):
            anc_mult = 0 if p == -1 else (1 if mode == "sparse" else self._depth(j))
            inp = in_dim + anc_mult * joint_dim
            self.mlps.append(nn.Sequential(
                nn.Linear(inp, hidden_per_joint), nn.ReLU(), nn.Linear(hidden_per_joint, joint_dim)
            ))

    def _depth(self, j):
        d = 0
        while self.parents[j] != -1:
            j = self.parents[j]
            d += 1
        return d

    def forward(self, h):
        B = h.size(0)
        preds = [None] * len(self.parents)
        for j, mlp in enumerate(self.mlps):
            parent = self.parents[j]
            if parent == -1:
                inp = h
            else:
                if self.mode == "sparse":
                    inp = torch.cat([h, preds[parent]], dim=-1)
                else:
                    chain = []
                    p = parent
                    while p != -1:
                        chain.append(preds[p])
                        p = self.parents[p]
                    inp = torch.cat([h] + chain[::-1], dim=-1)
            preds[j] = torch.tanh(mlp(inp))
        return torch.cat(preds, dim=-1)

# ───────────────────────────────────────────────────────────────────────────────
#  BaseModel
# ───────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.input_dim        # e.g. 135
        self.hidden_size = config.hidden_size     # e.g. 512
        self.num_layers = config.num_layers       # e.g. 2
        self.pred_frames = config.output_n        # e.g. 24

        self.ss_k = 1500

        # ─── core GRU ──────────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=config.dropout,
            batch_first=True,
        )

        # residual MLP + motion context gate
        self.mlp_pre = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.motion_att = DCTMotionAttention(self.input_size, self.hidden_size, heads=4)
        self.gate_fc = nn.Linear(self.hidden_size, 1)

        # Build 15‑joint hierarchical SPL (dense mode, smaller MLPs)
        joint_dim = self.input_size // len(SMPL_MAJOR_JOINTS)  # 9
        major_parents = []
        for j in SMPL_MAJOR_JOINTS:
            p = SMPL_PARENTS[j]
            major_parents.append(SMPL_MAJOR_JOINTS.index(p) if p in SMPL_MAJOR_JOINTS else -1)
        self.major_parents = major_parents

        self.spl = HierarchicalSPL(
            in_dim=self.hidden_size,
            joint_dim=joint_dim,
            parents=major_parents,
            mode="dense",
            hidden_per_joint=64,
        )

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, batch):
        seq = batch.poses
        input_seq = seq[:, : self.config.seed_seq_len]  # 20
        target_seq = seq[:, self.config.seed_seq_len :]  # 24

        motion_ctx = self.motion_att(input_seq)
        _, h = self.gru(input_seq, None)
        x_t = input_seq[:, -1]
        outputs = []

        for t in range(self.pred_frames):
            out, h = self.gru(x_t.unsqueeze(1), h)
            hidden = self.mlp_pre(out.squeeze(1))
            # gated addition of motion context
            gate = torch.tanh(self.gate_fc(hidden))
            hidden = hidden + gate * motion_ctx
            delta = self.spl(hidden)
            # ------------------------------------------------------------------
            # decide what to feed next: ground-truth vs. model prediction
            # ------------------------------------------------------------------
            pred_frame = x_t + delta  # model’s proposal

            if self.training:
                step = getattr(self, "training_step", 0)
                p_tf = self.ss_k / (self.ss_k + math.exp(step / self.ss_k))
                # draw Bernoulli mask once per sample in the batch
                mask = torch.rand(pred_frame.size(0), device=pred_frame.device) < p_tf
                # broadcast mask over feature dim
                mask = mask.float().unsqueeze(-1).repeat(1, pred_frame.size(1))
                gt_frame = target_seq[:, t]  # B×D
                x_t = mask * gt_frame + (1 - mask) * pred_frame
            else:
                x_t = pred_frame
            outputs.append(x_t)

        pred_seq = torch.stack(outputs, dim=1)

        if self.training:
            return {"predictions": pred_seq, "target": target_seq}
        else:
            return {"predictions": pred_seq, "seed": input_seq[:, -1:]}  # (B,1,135)

    # ────────────────────────────────────────────────────────────────────────
    def backward(self, batch, model_out, do_backward=True):
        pred_seq = model_out["predictions"]
        target_seq = model_out["target"]

        # Use only first `horizon` frames for all supervised losses
        horizon = 12
        pred_short = pred_seq[:, :horizon]
        targ_short = target_seq[:, :horizon]

        B, T, D = pred_short.shape
        J = D // 9
        pred_mat = pred_short.view(B, T, J, 3, 3)
        targ_mat = targ_short.view(B, T, J, 3, 3)

        loss_jangle = joint_angle_loss(pred_mat, targ_mat, parents=self.major_parents)
        loss_mpjpe = mpjpe(pred_short, targ_short)
        loss_geo = geodesic_loss(pred_mat, targ_mat)

        # Velocity loss (prepend last seed frame)
        last_seed = batch.poses[:, self.config.seed_seq_len - 1 : self.config.seed_seq_len]
        vel_pred = torch.cat([last_seed, pred_short], dim=1)
        vel_targ = torch.cat([last_seed, targ_short], dim=1)
        loss_vel = velocity_diff_loss(vel_pred, vel_targ)

        # --- first-frame continuity ----------------------------------------------
        last_seed_mat = last_seed.view(B, 1, J, 3, 3)  # reshape once
        first_pred_mat = pred_mat[:, 0:1]  # first predicted frame
        loss_cont = geodesic_loss(first_pred_mat, last_seed_mat)
        # --------------------------------------------------------------------------

        # Weighting (joint‑angle loss weight lowered)
        total_loss = (
                1.0 * loss_mpjpe
                + 1.0 * loss_geo
                + 0.25 * loss_vel
                + 0.50 * loss_cont  # NEW
                + 0.05 * loss_jangle
        )

        if do_backward:
            total_loss.backward()

        return (
            {
                "mpjpe": loss_mpjpe.item(),
                "geodesic_loss": loss_geo.item(),
                "velocity_loss": loss_vel.item(),
                "joint_angle": loss_jangle.item(),
                "continuity loss": loss_cont.item(),
                "total_loss": total_loss.item(),
            },
            target_seq,
        )

    def model_name(self):
        return f"{self.__class__.__name__}-lr{self.config.lr}"






class DummyModel(BaseModel):
    """
    This is a dummy model. It provides basic implementations to demonstrate how more advanced models can be built.
    """

    def __init__(self, config):
        self.n_history = 10
        super(DummyModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        # In this model we simply feed the last time steps of the seed to a dense layer and
        # predict the targets directly.
        self.dense = nn.Linear(in_features=self.n_history * self.pose_size,
                               out_features=self.config.target_seq_len * self.pose_size)

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}
        batch_size = batch.batch_size
        model_in = batch.poses[:, self.config.seed_seq_len-self.n_history:self.config.seed_seq_len]
        pred = self.dense(model_in.reshape(batch_size, -1))
        model_out['predictions'] = pred.reshape(batch_size, self.config.target_seq_len, -1)
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        targets = batch.poses[:, self.config.seed_seq_len:]

        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
