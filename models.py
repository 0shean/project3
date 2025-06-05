# -*- coding: utf-8 -*-
"""
Einops‑free GRU‑TC Motion‑Forecast model (clean version)
=======================================================

This revision fixes the GRU input‑size mismatch (Expected 388 vs. Got 264)
by introducing an **input projection** so that the Transformer and GRU both
operate in the configurable `cfg.d_model` dimensionality.

Pipeline
--------
seed (B,T,J,6) ─reshape→ (B,T,J*6)
               └─in_proj (J*6→d_model)──▶ TransformerEncoder (d_model)
                                          └─mean→ ctx (B,d_model)

GRU sees `[last_frame(J*6) ‖ ctx]` → hidden size =`d_model`.

Required `cfg` fields
~~~~~~~~~~~~~~~~~~~~~
```
cfg.input_dim   # e.g. 198   (9 × J)
cfg.d_model     # context & hidden size, e.g. 256
cfg.n_head      # Transformer heads
cfg.n_layer     # Transformer layers
```
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Rotation helpers (6‑D rep; Zhou et al. 2019)
# ---------------------------------------------------------------------------

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)

# ---------------------------------------------------------------------------
#  Transformer seed conditioner
# ---------------------------------------------------------------------------
class TransformerContext(nn.Module):
    def __init__(self, d_model: int, nhead: int, nlayers: int):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B,T,d_model)
        return self.enc(seq)                               # (B,T,d_model)

# ---------------------------------------------------------------------------
#  Structured‑prediction head
# ---------------------------------------------------------------------------
class SPLHead(nn.Module):
    def __init__(self, in_dim: int, num_joints: int, out_dim_per_joint: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_joints * out_dim_per_joint)
        self.num_joints = num_joints
        self.out_dim_per_joint = out_dim_per_joint

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B,H)
        return self.fc(x)                                  # (B,J*D)

# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------
class GRUTCMotionForecast(nn.Module):
    """GRU decoder wrapped with a Transformer context conditioner."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ----- dimensions ---------------------------------------------------
        self.in_rot_dim = 6                                 # 6‑D Zhou repr.
        self.num_joints = cfg.input_dim // 9                # derive J from 9·J
        self.embed_in   = self.num_joints * self.in_rot_dim # J·6  (e.g. 22*6=132)
        self.d_model    = cfg.d_model                       # ctx & hidden size

        # ----- modules ------------------------------------------------------
        self.in_proj = nn.Linear(self.embed_in, self.d_model)
        self.ctx_net = TransformerContext(self.d_model, cfg.n_head, cfg.n_layer)
        self.gru     = nn.GRU(self.embed_in + self.d_model, self.d_model, batch_first=True)
        self.spl     = SPLHead(self.d_model, self.num_joints, self.in_rot_dim)

    # ---------------------------------------------------------------------
    def forward(self, seed: torch.Tensor, future: int = 24) -> torch.Tensor:
        """Forecast `future` frames from `seed` (B,T_seed,J,6)."""
        B, T_seed, J, D = seed.shape
        assert J == self.num_joints and D == self.in_rot_dim, "seed shape mismatch"

        # ---- encode context --------------------------------------------
        seq_flat = seed.reshape(B, T_seed, -1)           # (B,T,J*6)
        seq_proj = self.in_proj(seq_flat)                # (B,T,d_model)
        ctx_seq  = self.ctx_net(seq_proj)                # (B,T,d_model)
        ctx      = ctx_seq.mean(dim=1)                   # (B,d_model)

        # ---- autoregressive decoding -----------------------------------
        h   = torch.zeros(1, B, self.d_model, device=seed.device, dtype=seed.dtype)
        inp = seed[:, -1].reshape(B, -1)                 # (B,J*6)
        outs = []
        for _ in range(future):
            dec_in = torch.cat([inp, ctx], dim=-1).unsqueeze(1)  # (B,1,J*6+d_model)
            out_t, h = self.gru(dec_in, h)                       # (B,1,d_model)
            rot6d = self.spl(out_t.squeeze(1))                   # (B,J*6)
            outs.append(rot6d)
            inp = rot6d                                          # feedback

        pred = torch.stack(outs, dim=1)                          # (B,F,J*6)
        return pred.view(B, future, J, D)

# ---------------------------------------------------------------------------
#  Factory helper
# ---------------------------------------------------------------------------

def create_model(config):
    return GRUTCMotionForecast(config)

# ---------------------------------------------------------------------------
#  Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, J = 2, 120, 22
    seed = torch.randn(B, T, J, 6)
    class Cfg: pass
    cfg = Cfg(); cfg.input_dim = J * 9; cfg.d_model = 256; cfg.n_head = 4; cfg.n_layer = 2
    model = GRUTCMotionForecast(cfg)
    out = model(seed, future=24)
    print("Output shape:", out.shape)  # expected (2,24,22,6)
