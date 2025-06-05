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
model = GRUTCMotionForecast(num_joints=22)
pred = model(seed)  # seed shape: (B, 120, 22, 6)
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
        h = self.enc(seq)                  # (B, T, D)
        ctx = h.mean(dim=1)               # (B, D)
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
    def __init__(
        self,
        num_joints: int = 22,
        in_rot_dim: int = 6,
        hidden_size: int = 512,
        ctx_dim: int = 256,
        nhead: int = 4,
        nlayers: int = 2,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.in_rot_dim = in_rot_dim
        # context conditioner
        self.ctx_net = TransformerContext(num_joints * in_rot_dim, nhead, nlayers)
        # decoder
        self.gru = nn.GRU(num_joints * in_rot_dim + ctx_dim, hidden_size, batch_first=True)
        # output head
        self.spl = SPLHead(hidden_size, num_joints, in_rot_dim)

    # ---------------------------------------------------------------------
    def forward(self, seed: torch.Tensor, future: int = 24) -> torch.Tensor:
        """Predict `future` frames given 120‑frame `seed`.

        Parameters
        ----------
        seed : Tensor, shape (B, 120, J, 6)
        future : int, number of frames to forecast
        """
        B, T_seed, J, D = seed.shape  # D should be 6
        assert J == self.num_joints and D == self.in_rot_dim, "seed shape mismatch"

        # Flatten joints for the conditioner: (B, T, J*D)
        seq_flat = seed.reshape(B, T_seed, J * D)
        ctx = self.ctx_net(seq_flat)              # (B, ctx_dim)

        # Prepare GRU hidden state
        h = torch.zeros(1, B, self.gru.hidden_size, device=seed.device, dtype=seed.dtype)

        # Use last frame as the initial decoder input
        inp = seed[:, -1].reshape(B, J * D)       # (B, J*D)
        outputs = []
        for _ in range(future):
            # concat context every step
            dec_in = torch.cat([inp, ctx], dim=-1).unsqueeze(1)  # (B,1, J*D+ctx)
            out_t, h = self.gru(dec_in, h)                       # out_t: (B,1,H)
            rot6d = self.spl(out_t.squeeze(1))                   # (B, J*D)
            outputs.append(rot6d)
            inp = rot6d                                          # autoregressive

        pred = torch.stack(outputs, dim=1)                       # (B, F, J*D)
        return pred.view(B, future, J, D)

# ----------------------------------------------------------------------------
#  Quick sanity check (executed only when run as a script)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, J = 2, 120, 22
    seed = torch.randn(B, T, J, 6)
    model = GRUTCMotionForecast(num_joints=J)
    out = model(seed, future=24)
    print("Output shape:", out.shape)  # should be (2, 24, 22, 6)
