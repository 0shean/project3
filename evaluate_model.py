#!/usr/bin/env python3
import os
import sys
import time
import math
import torch
import collections
import numpy as np
from torch.utils.data import DataLoader

# Ensure the project root is on PYTHONPATH so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration import Configuration
from configuration import CONSTANTS as C
from models import create_model
from data import AMASSBatch, LMDBDataset
from data_transforms import ToTensor
from losses import mpjpe, geodesic_loss, velocity_diff_loss, joint_angle_loss, bone_length_loss
from motion_metrics import MetricsEngine


def evaluate_model(config, checkpoint_path):
    """
    Loads a PoseTransformer from checkpoint_path, runs it on the validation set, and prints average losses
    and joint-angle metrics.
    """

    # 1) Instantiate model and load checkpoint
    device = C.DEVICE
    net = create_model(config)
    net.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # The checkpoint may store a dict with 'model_state_dict'
    if "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    # 2) Build validation DataLoader (same as in train.py)
    window_size = config.seed_seq_len + config.target_seq_len
    valid_transform = ToTensor()  # No ExtractWindow for validation
    valid_data = LMDBDataset(
        os.path.join(C.DATA_DIR, "validation"),
        transform=valid_transform
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config.bs_eval,
        shuffle=False,
        num_workers=config.data_workers,
        collate_fn=AMASSBatch.from_sample_list
    )

    # 3) Prepare metric accumulators
    loss_vals_agg = collections.defaultdict(float)
    n_samples = 0
    metrics_engine = MetricsEngine(C.METRIC_TARGET_LENGTHS)
    metrics_engine.reset()

    # 4) Loop over validation set
    with torch.no_grad():
        for abatch in valid_loader:
            batch_gpu = abatch.to_gpu()

            # Forward pass
            model_out = net(batch_gpu)
            pred_seq = model_out["predictions"]         # (B, T_out, D)
            target_seq = model_out["target"]            # (B, T_out, D)

            B, T, D = pred_seq.shape
            J = D // 9

            # Reshape to rotation matrices
            pred_mat = pred_seq.view(B, T, J, 3, 3)
            targ_mat = target_seq.view(B, T, J, 3, 3)

            # Compute individual losses
            loss_mpjpe = mpjpe(pred_seq, target_seq)
            loss_geo = geodesic_loss(pred_mat, targ_mat)

            # Velocity loss: prepend last seed frame
            seed_len = config.seed_seq_len
            last_seed = batch_gpu.poses[:, seed_len - 1:seed_len]  # (B,1,D)
            vel_pred = torch.cat([last_seed, pred_seq], dim=1)
            vel_targ = torch.cat([last_seed, target_seq], dim=1)
            loss_vel = velocity_diff_loss(vel_pred, vel_targ)

            # Joint-angle loss (in radians) and convert to degrees for logging
            loss_jangle = joint_angle_loss(pred_mat, targ_mat, parents=net.major_parents)
            loss_jangle_deg = loss_jangle * (180.0 / math.pi)

            # Bone-length loss
            loss_bone = bone_length_loss(pred_mat, parents=net.major_parents)

            # Total weighted loss
            total_loss = (
                0.75 * loss_mpjpe
                + 0.5 * loss_geo
                + 0.5 * loss_vel
                + 1.0 * loss_jangle
                + 0.3 * loss_bone
            )

            # Aggregate weighted by batch size
            loss_vals = {
                "mpjpe": loss_mpjpe.item(),
                "geodesic_loss": loss_geo.item(),
                "velocity_loss": loss_vel.item(),
                "joint_angle_deg": loss_jangle_deg.item(),
                "bone_length": loss_bone.item(),
                "total_loss": total_loss.item(),
            }
            for k, v in loss_vals.items():
                loss_vals_agg[k] += v * B
            n_samples += B

            # Also compute evaluation metrics (joint-angle metric from motion_metrics)
            metrics_engine.compute_and_aggregate(pred_seq, target_seq)

    # 5) Compute average losses over validation set
    avg_losses = {k: v / n_samples for k, v in loss_vals_agg.items()}

    # 6) Fetch final aggregated joint-angle metrics
    final_metrics = metrics_engine.get_final_metrics()  # dictionary e.g. {"joint_angle": array of length T_out}

    # Print results
    print("=== Validation Results ===")
    print(f"MPJPE               : {avg_losses['mpjpe']:.6f}")
    print(f"Geodesic loss       : {avg_losses['geodesic_loss']:.6f}")
    print(f"Velocity loss       : {avg_losses['velocity_loss']:.6f}")
    print(f"Joint-angle (deg)   : {avg_losses['joint_angle_deg']:.6f}")
    print(f"Bone-length loss    : {avg_losses['bone_length']:.6f}")
    print(f"Total weighted loss : {avg_losses['total_loss']:.6f}")
    print()
    # Print the “metrics until 24” summary string
    print("=== motion_metrics -joint_angle (sum over frames) ===")
    print(metrics_engine.get_summary_string(final_metrics))  # e.g. “metrics until 24:   joint_angle: XX.XXX”


if __name__ == "__main__":
    # Parse the same CLI arguments as train.py so we can reuse Configuration
    config = Configuration.parse_cmd()
    # Hardcode the checkpoint path as requested:
    checkpoint_path = "/home/sergejsz/project3/experiments/1749130067-PoseTransformer-lr0.0003/model.pth"
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    start_time = time.time()
    evaluate_model(config, checkpoint_path)
    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f} seconds.")
