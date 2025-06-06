# -*- coding: utf-8 -*-
"""
train.py — final cluster‑ready pipeline for the GRU‑TC model (einops‑free)
=======================================================================

Key points
----------
* 9‑D LMDB rotations → convert to 6‑D before feeding the model.
* The model outputs a **tensor** `(B,T,J,6)` not a dict.
* Losses needing rotation matrices convert 6‑D → 3×3 via `rot6d_to_matrix`.
* Velocity loss works directly in 6‑D feature space.
* Curriculum noise, Transformer LR schedule, checkpoints, TensorBoard logging.
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

# project modules ------------------------------------------------------
from configuration import Configuration, CONSTANTS as C
from data import LMDBDataset, AMASSBatch
from data_transforms import ExtractWindow, ToTensor
from evaluate import evaluate_test
from motion_metrics import MetricsEngine
import utils as U

from models import GRUTCMotionForecast, rot6d_to_matrix, matrix_to_rot6d
import losses as L
from models import rot6d_to_matrix, matrix_to_rot6d

# ────────────────────────────────────────────────────────────────────
class CurriculumNoiseScheduler:
    def __init__(self, cfg):
        self.start_std   = cfg.curriculum_start_std
        self.end_std     = cfg.curriculum_end_std
        self.total_steps = cfg.curriculum_steps
        self.global_step = 0

    def add(self, rot6d: torch.Tensor) -> torch.Tensor:
        if self.global_step >= self.total_steps:
            return rot6d
        t = self.global_step / self.total_steps
        sigma = (1 - t) * self.start_std + t * self.end_std
        return rot6d + torch.randn_like(rot6d) * sigma

    def step(self):
        self.global_step += 1

class TransformerLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimzier, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup  = warmup_steps
        super().__init__(optimzier)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [b * scale for b in self.base_lrs]

# ────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg):
    win = cfg.seed_seq_len + cfg.target_seq_len
    rng = np.random.RandomState(4313)
    train_tf = transforms.Compose([ExtractWindow(win, rng, 'random'), ToTensor()])
    valid_tf = transforms.Compose([ToTensor()])
    train_ds = LMDBDataset(os.path.join(C.DATA_DIR, 'training'), transform=train_tf, filter_seq_len=win)
    valid_ds = LMDBDataset(os.path.join(C.DATA_DIR, 'validation'), transform=valid_tf)
    dl_train = DataLoader(train_ds, cfg.bs_train, True, num_workers=cfg.data_workers,
                          collate_fn=AMASSBatch.from_sample_list)
    dl_valid = DataLoader(valid_ds, cfg.bs_eval, False, num_workers=cfg.data_workers,
                          collate_fn=AMASSBatch.from_sample_list)
    return dl_train, dl_valid

# ────────────────────────────────────────────────────────────────────

def compute_loss(pred6: torch.Tensor, batch, cfg):
    """
    pred6 : (B, T, J, 6)   – model output (6-D rot per joint)
    batch : AMASSBatch     – provides 9-D GT in batch.poses
    """

    # ------------ prepare target ---------------------------------------
    target9 = batch.poses[:, cfg.seed_seq_len:]          # (B,T,J*9)
    B, T, D9 = target9.shape
    J = pred6.shape[2]                                   # joint count from model
    assert D9 == J * 9, "GT dimensionality mismatch"

    targ_mat = target9.reshape(B, T, J, 3, 3)            # (B,T,J,3,3)
    targ6    = matrix_to_rot6d(targ_mat.reshape(-1,3,3)).reshape(B, T, J, 6)

    # ------------ predictions to matrices ------------------------------
    pred_mat = rot6d_to_matrix(pred6.reshape(-1, 6)).reshape(B, T, J, 3, 3)

    # ------------ losses -----------------------------------------------
    loss_geo  = L.geodesic_loss(pred_mat, targ_mat)
    loss_vel = L.velocity_diff_loss(
        pred6.reshape(B, T, J * 6),
        matrix_to_rot6d(targ_mat.reshape(-1, 3, 3)).reshape(B, T, J * 6)
    )
    loss_bone  = L.bone_length_loss(pred_mat, targ_mat)
    loss_limit = L.joint_limit_loss(pred_mat)
    loss_ps    = torch.tensor(0.0, device=pred6.device)  # placeholder if unused

    total = (cfg.loss_geodesic * loss_geo +
             cfg.loss_vel      * loss_vel +
             cfg.loss_bone     * loss_bone +
             cfg.loss_limit    * loss_limit +
             cfg.loss_pskld    * loss_ps)

    return dict(total_loss=total,
                geodesic=loss_geo, velocity=loss_vel,
                bone=loss_bone,  limit=loss_limit,
                pskld=loss_ps)

# ────────────────────────────────────────────────────────────────────

# ─── helper ─────────────────────────────────────────────────────────────
def to_seed6d(seed9: torch.Tensor, cfg) -> torch.Tensor:
    """
    Convert a (B, T, J*9) batch of 3×3 rotation-matrices to 6-D rep
    and reshape to (B, T, J, 6) for the model.
    """
    from models import matrix_to_rot6d                # keep import local
    B, T, D = seed9.shape
    J = D // 9

    # (B,T,J*9) → (B,T,J,3,3)
    mat = seed9.reshape(B, T, J, 3, 3)                # ←  use .reshape, not .view
    # flatten to (B*T*J,3,3) so matrix_to_rot6d works
    mat_flat = mat.reshape(-1, 3, 3)
    seed6 = matrix_to_rot6d(mat_flat).reshape(B, T, J, 6)
    return seed6


# ────────────────────────────────────────────────────────────────────

def main(cfg: Configuration):
    torch.manual_seed(cfg.seed or int(time.time()))
    dl_train, dl_valid = build_dataloaders(cfg)

    net = GRUTCMotionForecast(cfg).to(C.DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = TransformerLRScheduler(optimizer, cfg.d_model, cfg.lr_warmup_steps)

    cur_noise = CurriculumNoiseScheduler(cfg)
    metrics   = MetricsEngine(C.METRIC_TARGET_LENGTHS)
    log_dir   = U.create_model_dir(C.EXPERIMENT_DIR, int(time.time()), 'gru_tc')
    writer    = SummaryWriter(log_dir)

    best_val, gstep = float('inf'), 0

    for epoch in range(cfg.n_epochs):
        net.train()
        eloss, nsamp = collections.defaultdict(float), 0
        for batch in dl_train:
            optimizer.zero_grad()
            b_gpu = batch.to_gpu()

            # seed → 6‑D
            seed6 = to_seed6d(b_gpu.poses[:, :cfg.seed_seq_len], cfg)
            seed6 = cur_noise.add(seed6); cur_noise.step()

            pred6 = net(seed6)                                   # (B,T,J,6)
            loss_d = compute_loss(pred6, b_gpu, cfg)
            loss_d['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            for k,v in loss_d.items(): eloss[k]+=v.item()*b_gpu.batch_size
            nsamp += b_gpu.batch_size
            if gstep % cfg.print_every==0:
                writer.add_scalar('train/total', loss_d['total_loss'].item(), gstep)
            gstep+=1

        print(f"[E{epoch+1:03d}] train total {(eloss['total_loss']/nsamp):.4f}")

        # ---------------- validation ------------------------------
        net.eval(); vloss, vsamp = collections.defaultdict(float), 0; metrics.reset()
        with torch.no_grad():
            for batch in dl_valid:
                b_gpu = batch.to_gpu()

                # seed → 6-D and forward
                seed6 = to_seed6d(b_gpu.poses[:, :cfg.seed_seq_len], cfg)
                pred6 = net(seed6)  # (B,T,J,6)

                # -------- losses --------
                ldict = compute_loss(pred6, b_gpu, cfg)
                for k, v in ldict.items():
                    vloss[k] += v.item() * b_gpu.batch_size
                vsamp += b_gpu.batch_size

                # -------- metrics expect 9-D (J×9) flattened --------
                B, T, J, _ = pred6.shape
                pred9 = (
                    rot6d_to_matrix(pred6.reshape(-1, 6))  # (B*T*J,3,3)
                    .reshape(B, T, J * 9)  # (B,T,J*9)
                )
                target9 = b_gpu.poses[:, cfg.seed_seq_len:]  # (B,T,J*9)

                metrics.compute_and_aggregate(pred9, target9)

        vtot=vloss['total_loss']/vsamp
        print(f"[E{epoch+1:03d}] valid total {vtot:.4f}")
        writer.add_scalar('valid/total', vtot, epoch)
        if vtot<best_val:
            best_val=vtot
            torch.save({'epoch':epoch,'model_state':net.state_dict(),'opt_state':optimizer.state_dict(),'cfg':vars(cfg)},
                       os.path.join(log_dir,'model.pth'))

    print('Training finished – evaluating on test set …')
    evaluate_test(Path(log_dir).name.split('-')[0])

# --------------------------------------------------------------------
if __name__ == "__main__":
    main(Configuration.parse_cmd())