# -*- coding: utf-8 -*-
"""
Einops‑free GRU‑TC Motion‑Forecast model (clean version)
=======================================================

* TransformerContext – encodes the 120‑frame seed to a fixed‑length context.
* GRUTCMotionForecast – GRU decoder conditioned on that context.
* SPLHead – linear layer that emits 6‑D rotations (Zhou et al. 2019) per joint.

The model expects a `Configuration` object **cfg** that exposes at least:

    cfg.input_dim   # e.g. 135  (9 × J for seed data in 3×3 matrices)
    cfg.d_model     # hidden / context dimension, also GRU hidden size
    cfg.n_head      # Transformer heads
    cfg.n_layer     # Transformer encoder layers

```python
seed6 = torch.randn(B, 120, J, 6)         # 6‑D rotation seed
model  = GRUTCMotionForecast(cfg)
pred6  = model(seed6, future=24)           # (B,24,J,6)
```
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
#  Rotation helpers (6‑D rep; Zhou et al. 2019)
# ----------------------------------------------------------------------------

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6‑D rotation rep to 3×3 matrix (… ,3,3)."""
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    """Inverse of `rot6d_to_matrix`. Returns (…,6)."""
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)

# ----------------------------------------------------------------------------
#  Transformer seed conditioner
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

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B,T,D)
        ctx_seq = self.enc(seq)           # (B,T,D)
        ctx = ctx_seq.mean(dim=1)         # (B,D)
        return self.proj_out(ctx)

# ----------------------------------------------------------------------------
#  Structured‑prediction head (linear)
# ----------------------------------------------------------------------------
class SPLHead(nn.Module):
    def __init__(self, in_dim: int, num_joints: int, out_dim_per_joint: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_joints * out_dim_per_joint)
        self.num_joints = num_joints
        self.out_dim_per_joint = out_dim_per_joint

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,H)
        return self.fc(x)                                # (B,J*D)

# ----------------------------------------------------------------------------
#  Main model
# ----------------------------------------------------------------------------
class GRUTCMotionForecast(nn.Module):
    """GRU decoder wrapped with a Transformer context conditioner."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- sizes ---------------------------------------------------
        self.in_rot_dim = 6                                 # 6‑D Zhou repr.
        self.num_joints = cfg.input_dim // 9                # input dim is 9·J
        embed_dim = self.num_joints * self.in_rot_dim       # J·6, seed flat
        ctx_dim   = cfg.d_model                             # also GRU hidden

        # --- modules -------------------------------------------------
        self.ctx_net = TransformerContext(embed_dim, cfg.n_head, cfg.n_layer)
        self.gru     = nn.GRU(embed_dim + ctx_dim, ctx_dim, batch_first=True)
        self.spl     = SPLHead(ctx_dim, self.num_joints, self.in_rot_dim)

    # ----------------------------------------------------------------
    def forward(self, seed: torch.Tensor, future: int = 24) -> torch.Tensor:
        """Forecast `future` frames.

        Parameters
        ----------
        seed : Tensor (B, T_seed, J, 6)
        future : int, prediction horizon
        """
        B, T_seed, J, D = seed.shape
        assert J == self.num_joints and D == self.in_rot_dim, "seed shape mismatch"

        # ---- encode context ----------------------------------------
        seq_flat = seed.reshape(B, T_seed, -1)              # (B,T,J*6)
        ctx = self.ctx_net(seq_flat)                        # (B,ctx_dim)

        # ---- decode autoregressively -------------------------------
        h = torch.zeros(1, B, self.gru.hidden_size, device=seed.device, dtype=seed.dtype)
        inp = seed[:, -1].reshape(B, -1)                    # last frame, (B,J*6)
        outs = []
        for _ in range(future):
            dec_in = torch.cat([inp, ctx], dim=-1).unsqueeze(1)  # (B,1,J*6+ctx)
            out_t, h = self.gru(dec_in, h)                      # (B,1,H)
            rot6d = self.spl(out_t.squeeze(1))                  # (B,J*6)
            outs.append(rot6d)
            inp = rot6d                                         # feed back

        pred = torch.stack(outs, dim=1)                         # (B,F,J*6)
        return pred.view(B, future, J, D)

# ----------------------------------------------------------------------------
#  Factory helper
# ----------------------------------------------------------------------------

def create_model(config):
    return GRUTCMotionForecast(config)

# ----------------------------------------------------------------------------
#  Sanity check (executed when run directly)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, J = 2, 120, 22
    seed = torch.randn(B, T, J, 6)
    # Dummy cfg
    class Cfg: pass
    cfg = Cfg(); cfg.input_dim = J * 9; cfg.d_model = 256; cfg.n_head = 4; cfg.n_layer = 2
    model = GRUTCMotionForecast(cfg)
    out = model(seed, future=24)
    print("Output shape:", out.shape)  # expected (2, 24, 22, 6)
