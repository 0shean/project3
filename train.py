# -*- coding: utf-8 -*-
"""
train.py — cluster‑ready pipeline for the GRU‑TC model (einops‑free)
===================================================================

• Handles 6‑D rotations, bone‑length & joint‑limit penalties, PS‑KLD self‑distillation
• Adds curriculum noise and a Transformer LR schedule (Vaswani et al.)
• Compatible with the ETH MP project3 cluster environment (no external deps)
"""
from __future__ import annotations
import collections, os, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter

# project modules -----------------------------------------------------
from configuration import Configuration, CONSTANTS as C
from data import LMDBDataset, AMASSBatch
from data_transforms import ExtractWindow, ToTensor
from evaluate import evaluate_test
from motion_metrics import MetricsEngine
import utils as U

from models import GRUTCMotionForecast, matrix_to_rot6d, rot6d_to_matrix
import losses as L

# --------------------------------------------------------------------
class CurriculumNoiseScheduler:
    """Linear σ ramp for rotation‑noise augmentation (6‑D)."""
    def __init__(self, cfg):
        self.start_std   = cfg.curriculum_start_std
        self.end_std     = cfg.curriculum_end_std
        self.total_steps = cfg.curriculum_steps
        self.global_step = 0

    def add_noise(self, rot6d: torch.Tensor) -> torch.Tensor:
        if self.global_step >= self.total_steps:
            return rot6d
        t = self.global_step / self.total_steps
        sigma = (1 - t) * self.start_std + t * self.end_std
        return rot6d + torch.randn_like(rot6d) * sigma

    def step(self):
        self.global_step += 1

class TransformerLRScheduler(optim.lr_scheduler._LRScheduler):
    """Transformer LR schedule: d_model^{-0.5}·min(step^{-0.5}, step·warmup^{-1.5})"""
    def __init__(self, optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup  = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]

# --------------------------------------------------------------------

def build_dataloaders(cfg):
    win = cfg.seed_seq_len + cfg.target_seq_len
    rng = np.random.RandomState(4313)
    train_tf = transforms.Compose([ExtractWindow(win, rng, mode="random"), ToTensor()])
    valid_tf = transforms.Compose([ToTensor()])

    train_ds = LMDBDataset(os.path.join(C.DATA_DIR, "training"), transform=train_tf, filter_seq_len=win)
    valid_ds = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=valid_tf)

    dl_train = DataLoader(train_ds, batch_size=cfg.bs_train, shuffle=True,
                          num_workers=cfg.data_workers, collate_fn=AMASSBatch.from_sample_list)
    dl_valid = DataLoader(valid_ds, batch_size=cfg.bs_eval, shuffle=False,
                          num_workers=cfg.data_workers, collate_fn=AMASSBatch.from_sample_list)
    return dl_train, dl_valid

# --------------------------------------------------------------------

def compute_loss(model_out: Dict[str, torch.Tensor], batch, cfg):
    """Compute composite loss.
    model_out["predictions"] : (B, T, J, 6)
    batch.poses              : (B, seed+T, J*9)  – 9‑D rotation matrices
    """
    from models import rot6d_to_matrix, matrix_to_rot6d

    # ------------------------------------------------------------------
    # Predictions: 6‑D per joint → matrices + flattened 6‑D
    pred6  = model_out["predictions"]                 # (B,T,J,6)
    B, T, J, _ = pred6.shape
    pred6_flat = pred6.reshape(B, T, J * 6)          # (B,T,J*6) for velocity‑loss
    pred_mat   = rot6d_to_matrix(pred6.reshape(-1, 6)).view(B, T, J, 3, 3)

    # ------------------------------------------------------------------
    # Targets: stored in LMDB as 9‑D (3×3) per joint
    targ9  = batch.poses[:, cfg.seed_seq_len:]        # (B,T,J*9)
    _, _, JD9 = targ9.shape
    J_targ = JD9 // 9
    targ_mat = targ9.view(B, T, J_targ, 3, 3)
    target6_flat = matrix_to_rot6d(targ_mat.reshape(-1,3,3)).view(B, T, J_targ * 6)

    assert J == J_targ, "Joint count mismatch between pred and target"

    # ------------------------------------------------------------------
    # Loss components
    loss_geo  = L.geodesic_loss(pred_mat, targ_mat)
    loss_vel  = L.velocity_diff_loss(pred6_flat, target6_flat)
    loss_bone = L.bone_length_loss(pred_mat, targ_mat)
    loss_limit= L.joint_limit_loss(pred_mat)

    # PS‑KLD (may be absent early in training)
    loss_ps = model_out.get("ps_kld", torch.tensor(0.0, device=pred6.device))

    total = (cfg.loss_geodesic * loss_geo +
             cfg.loss_vel      * loss_vel +
             cfg.loss_bone     * loss_bone +
             cfg.loss_limit    * loss_limit +
             cfg.loss_pskld    * loss_ps)

    return {
        "total_loss": total,
        "geodesic"  : loss_geo.detach(),
        "velocity"  : loss_vel.detach(),
        "bone"      : loss_bone.detach(),
        "limit"     : loss_limit.detach(),
        "pskld"     : loss_ps.detach()
    }

# --------------------------------------------------------------------

def main(cfg: Configuration):
    torch.manual_seed(cfg.seed or int(time.time()))

    dl_train, dl_valid = build_dataloaders(cfg)
    net = GRUTCMotionForecast(cfg).to(C.DEVICE)
    print("Params:", U.count_parameters(net))

    opt = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = TransformerLRScheduler(opt, cfg.d_model, cfg.lr_warmup_steps)

    cur_noise = CurriculumNoiseScheduler(cfg)
    metrics   = MetricsEngine(C.METRIC_TARGET_LENGTHS)
    log_dir   = U.create_model_dir(C.EXPERIMENT_DIR, int(time.time()), "gru_tc")
    writer    = SummaryWriter(log_dir=log_dir)

    best_val, gstep = float("inf"), 0
    for epoch in range(cfg.n_epochs):
        net.train(); ep_loss, nsmp = collections.defaultdict(float), 0
        for batch in dl_train:
            opt.zero_grad(); batch_gpu = batch.to_gpu()

            # 9‑D → 6‑D seed conversion + noise --------------------------------
            seed9 = batch_gpu.poses[:, :cfg.seed_seq_len]        # (B,120,J*9)
            B,T9,D9 = seed9.shape; J = D9 // 9
            seed_mat = seed9.view(B,T9,J,3,3)
            seed6 = matrix_to_rot6d(seed_mat.reshape(-1,3,3)).view(B,T9,J,6)
            seed6 = cur_noise.add_noise(seed6); cur_noise.step()

            # forward + loss ---------------------------------------------------
            out = net(seed6)
            losses = compute_loss(out, batch_gpu, cfg)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()

            for k,v in losses.items(): ep_loss[k] += v.item()*batch_gpu.batch_size
            nsmp += batch_gpu.batch_size
            if gstep % cfg.print_every == 0:
                writer.add_scalar("loss/train", losses["total_loss"].item(), gstep)
                writer.add_scalar("lr", opt.param_groups[0]["lr"], gstep)
            gstep += 1

        print(f"[EPOCH {epoch+1:03d}] train total {ep_loss['total_loss']/nsmp:.6f}")

        # ---------------- validation -----------------------------------------
        net.eval(); metrics.reset(); vloss, vsmp = collections.defaultdict(float), 0
        with torch.no_grad():
            for batch in dl_valid:
                b_gpu = batch.to_gpu()
                targ9 = b_gpu.poses[:, cfg.seed_seq_len:]

                # prepare seed as above
                seed9 = b_gpu.poses[:, :cfg.seed_seq_len]
                B,T9,D9 = seed9.shape; J = D9//9
                seed6 = matrix_to_rot6d(seed9.view(B,T9,J,3,3).reshape(-1,3,3)).view(B,T9,J,6)

                out = net(seed6)
                ld = compute_loss(out, b_gpu, cfg)
                for k,v in ld.items(): vloss[k] += v.item()*b_gpu.batch_size
                vsmp += b_gpu.batch_size
                metrics.compute_and_aggregate(out["predictions"], targ9)

        vloss = {k:v/vsmp for k,v in vloss.items()}
        print(f"[EPOCH {epoch+1:03d}] valid total {vloss['total_loss']:.6f}")
        writer.add_scalar("loss/valid", vloss["total_loss"], epoch)

        # checkpoint ----------------------------------------------------------
        if vloss["total_loss"] < best_val:
            best_val = vloss["total_loss"]
            torch.save({
                "epoch":epoch, "model":net.state_dict(), "opt":opt.state_dict(), "cfg":vars(cfg)
            }, os.path.join(log_dir,"model.pth"))

    print("Training finished — running test evaluation…")
    evaluate_test(Path(log_dir).name.split("-")[0])

# --------------------------------------------------------------------
if __name__ == "__main__":
    main(Configuration.parse_cmd())
