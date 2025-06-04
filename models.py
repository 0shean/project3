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

# fk.py already contains SMPL_PARENTS and SMPL_MAJOR_JOINTS
from fk import SMPL_PARENTS            # list[int] len = n_joints
from fk import SMPL_JOINTS             # list[str] joint-name lookup
from losses import mat_to_axis_angle


def so3_log(R, eps=1e-6):
    """
    Log-map of a batch of rotation matrices.
    R: (..., 3, 3)  torch tensor
    Returns: (..., 3) axis-angle vector
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1) / 2
    cos = cos.clamp(-1 + eps, 1 - eps)          # numerical safety
    theta = torch.acos(cos)

    # When sin(theta) is small use first-order expansion
    sin_theta = torch.sin(theta)
    mask = sin_theta.abs() < 1e-4
    scale = torch.where(mask,
                        0.5 + (theta**2)/12,     # series expansion
                        theta / (2 * sin_theta))
    # skew-symmetric part
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return scale.unsqueeze(-1) * w


# ─── forward-kinematics helpers ──────────────────────────────────────────
def local_to_global(rot_local, parents):
    """
    rot_local : (B,T,J,3,3)  local rotations
    parents   : list[int] length-J, −1 for root
    returns   : (B,T,J,3,3)  global rotations
    """
    rot_global = rot_local.clone()
    for j in range(1, len(parents)):          # root (0) already global
        p = parents[j]
        if p != -1:
            rot_global[..., j, :, :] = torch.matmul(
                rot_global[..., p, :, :], rot_global[..., j, :, :])
    return rot_global

def joint_angle_loss(pred_mat, targ_mat, parents, eps=1e-6):
    """
    Mean global joint-angle error (radians) over all joints & frames.
    """
    Rg_pred = local_to_global(pred_mat, parents)
    Rg_targ = local_to_global(targ_mat, parents)
    R_err   = torch.matmul(Rg_pred, Rg_targ.transpose(-1, -2))
    cos = (R_err[...,0,0] + R_err[...,1,1] + R_err[...,2,2] - 1) / 2
    cos = cos.clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)
    return angle.mean()
# ─────────────────────────────────────────────────────────────────────────




def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return BaseModel(config)

class DCTMotionAttention(nn.Module):
    """
    Compute a motion context by taking a real-FFT (as a proxy for DCT-II)
    over the seed sequence for each joint-dimension, projecting into hidden
    space, running a small self-attention, and averaging over frequency.
    """
    def __init__(self, input_dim, hidden_dim, heads=4):
        super().__init__()
        # project each frequency‐bin of size input_dim into hidden_dim
        self.freq_proj = nn.Linear(input_dim, hidden_dim)
        # a tiny Transformer‐style self‐attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
    def forward(self, seed_seq):
        # seed_seq: [B, T_in, D]  where D = input_dim
        # 1) compute real‐FFT over time dim
        freq = torch.fft.rfft(seed_seq, dim=1).real        # [B, T′, D], where T′ = T_in//2+1
        # 2) lift into hidden space
        feat = self.freq_proj(freq)                        # [B, T′, hidden_dim]
        # 3) self‐attend over frequencies
        attn_out, _ = self.attn(feat, feat, feat)          # [B, T′, hidden_dim]
        # 4) pool over the frequency axis
        context = attn_out.mean(dim=1)                     # [B, hidden_dim]
        return context
class StructuredPredictionLayer(nn.Module):
    def __init__(self, in_dim, joint_dim, joint_names):
        super().__init__()
        self.joint_names = joint_names
        # one MLP per joint
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
        return torch.cat(outs, dim=-1)  # [B, num_joints*joint_dim]

class HierarchicalSPL(nn.Module):
    """
    Hierarchical Structured Prediction Layer (SPL)
    that follows the human kinematic chain.
    Each joint MLP receives the global context vector h_t
    *and* the prediction of its parent joint.
    mode='sparse' : only immediate parent
    mode='dense'  : concatenation of all ancestor predictions
    """
    def __init__(self, in_dim, joint_dim, parents, mode='sparse',
                 hidden_per_joint=128):
        super().__init__()
        self.parents  = parents                      # list[int]
        self.joint_dim = joint_dim
        self.mode = mode

        # build one MLP for every joint
        self.mlps = nn.ModuleList()
        for j, p in enumerate(parents):
            # input = context + parent pred(s)
            anc_mult = 0 if p == -1 else (1 if mode=='sparse'
                                          else self._depth(j))
            inp = in_dim + anc_mult * joint_dim
            self.mlps.append(nn.Sequential(
                nn.Linear(inp, hidden_per_joint),
                nn.ReLU(),
                nn.Linear(hidden_per_joint, joint_dim)
            ))

    def _depth(self, j):
        """number of ancestors of joint j (exclusive)"""
        d = 0
        while self.parents[j] != -1:
            j = self.parents[j]; d += 1
        return d

    def forward(self, h):
        """
        h : (B, in_dim) — same for *all* joints at this time-step.
        Returns concatenated predictions          (B, K*joint_dim)
        """
        B = h.size(0)
        preds = [None] * len(self.parents)
        for j, mlp in enumerate(self.mlps):
            parent = self.parents[j]
            if parent == -1:                          # root
                inp = h
            else:
                if self.mode == 'sparse':
                    inp = torch.cat([h, preds[parent]], dim=-1)
                else:                                # dense: all ancestors
                    chain = []
                    p = parent
                    while p != -1:
                        chain.append(preds[p]); p = self.parents[p]
                    inp = torch.cat([h] + chain[::-1], dim=-1)
            preds[j] = torch.tanh(mlp(inp))     # instead of mlp(inp)
        return torch.cat(preds, dim=-1)


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.input_dim        # e.g., 135
        self.hidden_size = config.hidden_size     # e.g., 512
        self.num_layers = config.num_layers       # e.g., 2
        self.pred_frames = config.output_n        # e.g., 24

        # GRU for autoregressive modeling
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=config.dropout,
            batch_first=True
        )

        # two-layer residual head → split per‐joint final mapping
        self.mlp_pre = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU())

        # DCT‐based motion attention over the seed window
        # DCT‐based motion attention over the seed window
        # --------------------------------------------------------------------------
        # DCT-based motion attention over the seed window
        self.motion_att = DCTMotionAttention(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            heads=4)

        # --------------------------------------------------------------------------
        # >>> NEW CODE – Hierarchical SPL for the 15 *major* SMPL joints <<<
        #
        # We keep the 15-joint pose vector (135 numbers) that the dataloader gives.
        # So: joint_dim = 9 and parents = length-15 list aligned with those joints.
        #
        joint_dim = self.input_size // len(SMPL_MAJOR_JOINTS)  # 135 // 15 = 9

        # Build a parent index list *restricted to the 15 major joints*
        major_parents = []
        for j in SMPL_MAJOR_JOINTS:  # iterate over the 15 joint IDs
            p = SMPL_PARENTS[j]  # parent in the full 24-joint tree
            major_parents.append(
                SMPL_MAJOR_JOINTS.index(p) if p in SMPL_MAJOR_JOINTS else -1)
            self.major_parents = major_parents

        self.spl = HierarchicalSPL(
            in_dim=self.hidden_size,
            joint_dim=joint_dim,  # 9 numbers per joint
            parents=major_parents,  # length-15 parent list
            mode='sparse')  # or 'dense' if you prefer
        # --------------------------------------------------------------------------

    def forward(self, batch):
        seq = batch.poses
        input_seq = seq[:, :self.config.seed_seq_len]  # 20
        target_seq = seq[:, self.config.seed_seq_len:]  # 1

        B, T_in, D = input_seq.shape

        motion_ctx = self.motion_att(input_seq)  # [B, hidden_size]

        _, h = self.gru(input_seq, None)
        x_t = input_seq[:, -1]
        outputs = []

        for _ in range(self.pred_frames):
            x_t_input = x_t.unsqueeze(1)
            out, h = self.gru(x_t_input, h)
            hidden = self.mlp_pre(out.squeeze(1))
            hidden = hidden + motion_ctx
            delta = self.spl(hidden)
            x_t = x_t + delta
            outputs.append(x_t)

        pred_seq = torch.stack(outputs, dim=1)

        if self.training:
            return {
                'predictions': pred_seq,
                'target': target_seq
            }
        else:
            return {
                'predictions': pred_seq,
                'seed': input_seq[:, -1:]  # shape (B, 1, 135)
            }

    def backward(self, batch, model_out, do_backward=True):
        pred_seq = model_out['predictions']
        target_seq = model_out['target']

        # --- use only the first GT frame for the loss ---
        # --- use the FULL 24-frame target ---
        pred_used = pred_seq  # shape (B, 24, D)
        targ_used = target_seq

        B, T, D = pred_used.shape
        J = D // 9
        pred_mat = pred_used.view(B, T, J, 3, 3)
        targ_mat = targ_used.view(B, T, J, 3, 3)

        loss_jangle = joint_angle_loss(
        pred_mat, targ_mat, parents=self.major_parents)

        loss_mpjpe = mpjpe(pred_used, targ_used)
        loss_geo = geodesic_loss(pred_mat, targ_mat)

        # --- velocity loss ---
        last_seed = batch.poses[:, self.config.seed_seq_len - 1:self.config.seed_seq_len]  # (B,1,D)
        vel_pred = torch.cat([last_seed, pred_used], dim=1)
        vel_targ = torch.cat([last_seed, targ_used], dim=1)
        loss_vel = velocity_diff_loss(vel_pred, vel_targ)  # make sure losses.py has this fn

        # total_loss                                              AFTER
        total_loss = (1.0 * loss_mpjpe +
                      1.0 * loss_geo +
                      0.25 * loss_vel +
                      0.2 * loss_jangle)

        if do_backward:
            total_loss.backward()

        loss_dict = {
            'mpjpe': loss_mpjpe.item(),
            'geodesic_loss': loss_geo.item(),
            'velocity_loss': loss_vel.item(),
            'joint_angle': loss_jangle.item(),  # ← new
            'total_loss': total_loss.item(),
        }

        return loss_dict, target_seq

    def model_name(self):
        """A summary string of this model. Override this if desired."""
        return '{}-lr{}'.format(self.__class__.__name__, self.config.lr)






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
