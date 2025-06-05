# -*- coding: utf-8 -*-
"""
GRU‑TC Motion Forecast Model  (einops‑free)
==========================================

This version removes all einops dependencies by replacing tensor rearrangements
with native PyTorch `view`, `reshape`, and `permute` calls.  It is therefore
compatible with the course cluster environment where additional site‑packages
cannot be installed.

Key components
--------------
* **TransformerContext** – encodes the 120‑frame seed once and produces a
  fixed‑length context vector using `nn.TransformerEncoder` (batch_first=True
  so no external rearrange is needed).
* **GRUTCMotionForecast** – GRU decoder that receives the context vector at
  every step.
* **SPLHead** – lightweight structured prediction layer emitting 6‑D rotation
  vectors for each joint.
* **rot6d_to_matrix / matrix_to_rot6d** – minimal SO(3) helpers (copied here to
  avoid extra imports).

Usage
-----
```python
from gru_tc_model import GRUTCMotionForecast
model = GRUTCMotionForecast(config)  # config must include input_dim, d_model, n_head, n_layer
pred = model(seed)  # seed shape: (B, 120, J, 6)
```
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
#  Rotation representation helpers (6‑D rep; Zhou et al. 2019)
# ----------------------------------------------------------------------------
def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6‑D rotation rep to 3×3 matrix (B,*,3,3)."""
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    """Inverse of rot6d_to_matrix.
    mat shape: (B,*,3,3) → returns (B,*,6)
    """
    return mat[..., :2, :].clone().reshape(*mat.shape[:-2], 6)

# ----------------------------------------------------------------------------
#  Transformer context conditioner
# ----------------------------------------------------------------------------
class TransformerContext(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, nlayers: int):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (B, T, D)
        h = self.enc(seq)                    # (B, T, D)
        ctx = h.mean(dim=1)                  # (B, D)
        return self.proj_out(ctx)

# ----------------------------------------------------------------------------
#  Structured Prediction Layer (minimal; can be swapped for your own SPL)
# ----------------------------------------------------------------------------
class SPLHead(nn.Module):
    def __init__(self, in_dim: int, num_joints: int, out_dim_per_joint: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_joints * out_dim_per_joint)
        self.num_joints = num_joints
        self.out_dim = out_dim_per_joint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H)
        return self.fc(x)

# ----------------------------------------------------------------------------
#  Main model
# ----------------------------------------------------------------------------
class GRUTCMotionForecast(nn.Module):
    """GRU decoder with a Transformer seed conditioner (einops‑free)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Dataset uses 9‑D (3×3) per joint → derive joint count from that
        self.in_rot_dim = 6                                         # we output 6‑D
        self.num_joints = cfg.input_dim // 9                        # e.g. 135 → 15 joints

        embed_dim   = self.num_joints * self.in_rot_dim             # 15 × 6 = 90
        nhead       = cfg.n_head
        nlayers     = cfg.n_layer
        ctx_dim     = cfg.d_model                                   # projected context + GRU hidden size
        hidden_size = ctx_dim

        # context conditioner
        self.ctx_net = TransformerContext(embed_dim, nhead, nlayers)
        self.proj_ctx = nn.Linear(embed_dim, ctx_dim)

        # decoder
        self.gru = nn.GRU(embed_dim + ctx_dim, hidden_size, batch_first=True)
        # output head
        self.spl = SPLHead(hidden_size, self.num_joints, self.in_rot_dim)
(
        self,
        cfg: object,  # Configuration containing input_dim, d_model, n_head, n_layer
    ):
        super().__init__()
        # Fixed 6-D rotation representation
        in_rot_dim = 6
        # Derive number of joints from input dimension (e.g., 135 = 22*6 + remainder)
        num_joints = cfg.input_dim // in_rot_dim
        hidden_size = cfg.d_model
        ctx_dim = cfg.d_model
        nhead = cfg.n_head
        nlayers = cfg.n_layer

        self.num_joints = num_joints
        self.in_rot_dim = in_rot_dim
        # context conditioner
        self.ctx_net = TransformerContext(num_joints * in_rot_dim, nhead, nlayers)
        # decoder: input is (joint*6 + ctx_dim), hidden is hidden_size
        self.gru = nn.GRU(num_joints * in_rot_dim + ctx_dim, hidden_size, batch_first=True)
        # output head: maps hidden_size → joint*6
        self.spl = SPLHead(hidden_size, num_joints, in_rot_dim)

    # ---------------------------------------------------------------------
    def forward(self, seed: torch.Tensor, future: int = 24) -> torch.Tensor:
        """Predict `future` frames given `seed` of shape (B, T_seed, J, 6)."""
        B, T_seed, J, D = seed.shape  # D should be 6
        assert J == self.num_joints and D == self.in_rot_dim, "seed shape mismatch"

        # Flatten joints for the conditioner: (B, T, J*D)
        seq_flat = seed.reshape(B, T_seed, J * D)
        ctx = self.ctx_net(seq_flat)                # (B, ctx_dim)

        # Prepare initial hidden state
        h = torch.zeros(1, B, self.gru.hidden_size, device=seed.device, dtype=seed.dtype)

        # Use last frame as initial decoder input
        inp = seed[:, -1].reshape(B, J * D)         # (B, J*D)
        outputs = []
        for _ in range(future):
            # concat context every step
            dec_in = torch.cat([inp, ctx], dim=-1).unsqueeze(1)  # (B,1, J*D + ctx_dim)
            out_t, h = self.gru(dec_in, h)                       # out_t: (B,1,hidden_size)
            rot6d = self.spl(out_t.squeeze(1))                   # (B, J*D)
            outputs.append(rot6d)
            inp = rot6d                                          # autoregressive

        pred = torch.stack(outputs, dim=1)                       # (B, future, J*D)
        return pred.view(B, future, J, D)

# ----------------------------------------------------------------------------
#  Model creation helper
# ----------------------------------------------------------------------------
def create_model(config: object) -> GRUTCMotionForecast:
    return GRUTCMotionForecast(config)

# ----------------------------------------------------------------------------
#  Quick sanity check (executed only when run as a script)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, J = 2, 120, 22
    seed = torch.randn(B, T, J, 6)
    # Dummy config with necessary fields
    class Cfg: pass
    cfg = Cfg()
    cfg.input_dim = J * 6
    cfg.d_model = 256
    cfg.n_head = 4
    cfg.n_layer = 2
    model = GRUTCMotionForecast(cfg)
    out = model(seed, future=24)
    print("Output shape:", out.shape)  # should be (2, 24, 22, 6)
