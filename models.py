import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from losses import mpjpe, geodesic_loss, joint_angle_loss, velocity_diff_loss, bone_length_loss
from fk import SMPL_MAJOR_JOINTS, SMPL_PARENTS

def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return PoseTransformer(config)



class PoseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.input_dim
        self.hidden_size = config.hidden_size
        self.pred_frames = config.output_n
        self.ss_k = 400

        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.transformer = nn.Transformer(  # simple encoder-decoder style
            d_model=self.hidden_size,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(self.hidden_size, self.input_size)

        self.joint_dim = self.input_size // len(SMPL_MAJOR_JOINTS)
        self.major_parents = [
            SMPL_MAJOR_JOINTS.index(SMPL_PARENTS[j]) if SMPL_PARENTS[j] in SMPL_MAJOR_JOINTS else -1
            for j in SMPL_MAJOR_JOINTS
        ]

    def forward(self, batch):
        x = batch.poses
        input_seq = x[:, :self.config.seed_seq_len]  # (B, T_in, D)
        target_seq = x[:, self.config.seed_seq_len:]  # (B, T_out, D)

        memory = self.encoder(input_seq)
        tgt = torch.zeros_like(target_seq)

        out = self.transformer(src=memory, tgt=self.encoder(tgt))
        pred_seq = self.decoder(out)

        return {"predictions": pred_seq, "target": target_seq}

    def backward(self, batch, model_out, do_backward=True):
        pred_seq = model_out["predictions"]
        target_seq = model_out["target"]

        B, T, D = pred_seq.shape
        J = D // 9
        pred_mat = pred_seq.view(B, T, J, 3, 3)
        targ_mat = target_seq.view(B, T, J, 3, 3)

        loss_jangle = joint_angle_loss(pred_mat, targ_mat, parents=self.major_parents)
        loss_mpjpe = mpjpe(pred_seq, target_seq)
        loss_geo = geodesic_loss(pred_mat, targ_mat)
        loss_vel = velocity_diff_loss(pred_seq, target_seq)
        loss_bone = bone_length_loss(pred_mat, self.major_parents)

        total_loss = (
            0.75 * loss_mpjpe
            + 0.5 * loss_geo
            + 0.5 * loss_vel
            + 1.0 * loss_jangle
            + 0.3 * loss_bone
        )

        if do_backward:
            total_loss.backward()

        return {
            "mpjpe": loss_mpjpe.item(),
            "geodesic_loss": loss_geo.item(),
            "velocity_loss": loss_vel.item(),
            "joint_angle": loss_jangle.item(),
            "bone_length": loss_bone.item(),
            "total_loss": total_loss.item(),
        }, target_seq

    def model_name(self):
        return f"PoseTransformer-lr{self.config.lr}"
