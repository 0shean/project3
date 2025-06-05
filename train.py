# -*- coding: utf-8 -*-
"""
train.py — cluster-ready pipeline for the GRU-TC model
=====================================================

• Uses the upgraded **GRU-TC MotionForecastModel** (einops-free)
• Handles 6-D rotations, bone-length & joint-limit penalties, PS-KLD self-distillation
• Adds curriculum noise and a Transformer LR schedule (à la Vaswani 2017)
• Invokes diffusion refinement after coarse prediction once warm-up is over
"""
from __future__ import annotations
import argparse, collections, glob, os, sys, time, math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter

# -- project modules ---------------------------------------------------
from configuration import Configuration, CONSTANTS as C
from data import LMDBDataset, AMASSBatch
from data_transforms import ExtractWindow, ToTensor
from evaluate import evaluate_test
from motion_metrics import MetricsEngine
import utils as U

# upgraded model + losses
from models import GRUTCMotionForecast
import losses as L

# ----------------------------------------------------------------------
class CurriculumNoiseScheduler:
    """Linearly increases Gaussian rotation noise σ from start → end over N steps."""
    def __init__(self, cfg):
        # Read the flat CLI flags instead of a nested dict
        self.start_std   = cfg.curriculum_start_std
        self.end_std     = cfg.curriculum_end_std
        self.total_steps = cfg.curriculum_steps
        self.global_step = 0

    def add_noise(self, rot6d: torch.Tensor) -> torch.Tensor:
        if self.global_step >= self.total_steps:
            return rot6d
        t = self.global_step / self.total_steps
        sigma = (1 - t) * self.start_std + t * self.end_std
        noise = torch.randn_like(rot6d) * sigma
        return rot6d + noise

    def step(self):
        self.global_step += 1

class TransformerLRScheduler(optim.lr_scheduler._LRScheduler):
    """d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

# ----------------------------------------------------------------------

def build_dataloaders(cfg):
    rng = np.random.RandomState(4313)
    win = cfg.seed_seq_len + cfg.target_seq_len
    train_tf = transforms.Compose([ExtractWindow(win, rng, mode="random"), ToTensor()])
    valid_tf = transforms.Compose([ToTensor()])
    train_ds = LMDBDataset(os.path.join(C.DATA_DIR, "training"), transform=train_tf, filter_seq_len=win)
    valid_ds = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=valid_tf)
    dl_train = DataLoader(
        train_ds,
        batch_size=cfg.bs_train,
        shuffle=True,
        num_workers=cfg.data_workers,
        collate_fn=AMASSBatch.from_sample_list
    )
    dl_valid = DataLoader(
        valid_ds,
        batch_size=cfg.bs_eval,
        shuffle=False,
        num_workers=cfg.data_workers,
        collate_fn=AMASSBatch.from_sample_list
    )
    return dl_train, dl_valid

# full composite loss
def compute_loss(model_out: Dict[str, torch.Tensor], batch, cfg):
    pred   = model_out["predictions"]                # [B, T, D]
    target = batch.poses[:, cfg.seed_seq_len:]        # [B, T, D]

    B, T, D = pred.shape
    J = D // 9

    pred_mat = pred.view(B, T, J, 3, 3)
    targ_mat = target.view(B, T, J, 3, 3)

    # basic losses
    loss_geo = L.geodesic_loss(pred_mat, targ_mat)
    loss_vel = L.velocity_diff_loss(pred, target)

    # bone-length & joint-limit
    loss_bone  = L.bone_length_loss(pred_mat, targ_mat)
    loss_limit = L.joint_limit_loss(pred_mat)

    # PS-KLD self-distillation (assumed computed inside model forward())
    loss_ps = model_out.get("ps_kld", torch.tensor(0.0, device=pred.device))

    total = (
        cfg.loss_geodesic * loss_geo +
        cfg.loss_vel      * loss_vel +
        cfg.loss_bone     * loss_bone +
        cfg.loss_limit    * loss_limit +
        cfg.loss_pskld    * loss_ps
    )

    return {
        "total_loss": total,
        "geodesic":   loss_geo.detach(),
        "velocity":   loss_vel.detach(),
        "bone":       loss_bone.detach(),
        "limit":      loss_limit.detach(),
        "pskld":      loss_ps.detach()
    }

# ----------------------------------------------------------------------

def main(cfg: Configuration):
    if cfg.seed is None:
        cfg.seed = int(time.time())
    torch.manual_seed(cfg.seed)

    # dataloaders
    dl_train, dl_valid = build_dataloaders(cfg)

    # model ------------------------------------------------------------
    net = GRUTCMotionForecast(cfg)
    net.to(C.DEVICE)
    print("Params:", U.count_parameters(net))

    # opt + sched ------------------------------------------------------
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = TransformerLRScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.lr_warmup_steps)

    # helpers
    cur_noise = CurriculumNoiseScheduler(cfg)
    metrics   = MetricsEngine(C.METRIC_TARGET_LENGTHS)
    # Replace net.model_name() with a fixed string "gru_tc"
    log_dir = U.create_model_dir(C.EXPERIMENT_DIR, int(time.time()), "gru_tc")
    writer  = SummaryWriter(log_dir=log_dir)

    best_val    = float("inf")
    global_step = 0
    net.train()

    for epoch in range(cfg.n_epochs):
        epoch_loss = collections.defaultdict(float)
        nsmp       = 0

        for batch in dl_train:
            optimizer.zero_grad()
            batch_gpu = batch.to_gpu()

            # curriculum noise on seed (in-place)
            with torch.no_grad():
                noisy_seed = cur_noise.add_noise(batch_gpu.poses[:, :cfg.seed_seq_len])
                batch_gpu.poses[:, :cfg.seed_seq_len] = noisy_seed
            cur_noise.step()

            # forward pass
            seed = batch_gpu.poses[:, :cfg.seed_seq_len]  # (B, 120, J*6)
            seed = seed.view(seed.size(0), cfg.seed_seq_len, -1, 6)  # (B,120,J,6)
            model_out = net(seed)
            loss_dict = compute_loss(model_out, batch_gpu, cfg)
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            net.training_step = global_step  # for scheduled sampling inside model

            # logging --------------------------------------------------
            for k, v in loss_dict.items():
                epoch_loss[k] += v.item() * batch_gpu.batch_size
            nsmp += batch_gpu.batch_size

            if global_step % cfg.print_every == 0:
                writer.add_scalar("loss/train", loss_dict["total_loss"].item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

        # epoch-level logging ------------------------------------------
        avg_loss = {k: v / nsmp for k, v in epoch_loss.items()}
        print(f"[EPOCH {epoch+1:03d}] train total {avg_loss['total_loss']:.6f}")

        # ------- validation ------------------------------------------
        net.eval()
        metrics.reset()
        val_loss = collections.defaultdict(float)
        vsamp    = 0
        with torch.no_grad():
            for batch in dl_valid:
                batch_gpu = batch.to_gpu()
                out = net(batch_gpu)
                ldict = compute_loss(out, batch_gpu, cfg)
                for k, v in ldict.items():
                    val_loss[k] += v.item() * batch_gpu.batch_size
                vsamp += batch_gpu.batch_size
                metrics.compute_and_aggregate(out["predictions"], batch_gpu.poses[:, cfg.seed_seq_len:])

        val_loss = {k: v / vsamp for k, v in val_loss.items()}
        print(f"[EPOCH {epoch+1:03d}] valid total {val_loss['total_loss']:.6f}")
        writer.add_scalar("loss/valid", val_loss["total_loss"], epoch)

        # checkpoint ---------------------------------------------------
        ckpt_path = os.path.join(log_dir, "model.pth")
        if val_loss["total_loss"] < best_val:
            best_val = val_loss["total_loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": net.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "cfg":         vars(cfg)
            }, ckpt_path)
        net.train()

    # ---------------------------------------------------------------
    print("Training finished. Evaluating test set…")
    evaluate_test(Path(log_dir).name.split("-")[0])

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main(Configuration.parse_cmd())
