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
from losses import mpjpe, angle_loss
from fk import SMPL_MAJOR_JOINTS, SMPL_JOINTS

def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return BaseModel(config)


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
            batch_first=True
        )

        # two-layer residual head → split per‐joint final mapping
        self.mlp_pre = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU())
        #assume config.joint_names is a list of your 15 SMPL joints
        joint_names = [SMPL_JOINTS[i] for i in SMPL_MAJOR_JOINTS]
        joint_dim = self.input_size // len(joint_names)
        self.spl = StructuredPredictionLayer(in_dim = self.hidden_size, joint_dim = joint_dim, joint_names = joint_names)

    def forward(self, batch):
        seq = batch.poses
        input_seq = seq[:, :120]
        target_seq = seq[:, 120:]

        B, T_in, D = input_seq.shape
        h = None
        x_t = input_seq[:, -1]
        outputs = []

        for _ in range(self.pred_frames):
            x_t_input = x_t.unsqueeze(1)
            out, h = self.gru(x_t_input, h)
            hidden = self.mlp_pre(out.squeeze(1))
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

        loss_mpjpe = mpjpe(pred_seq, target_seq)
        loss_angle = angle_loss(pred_seq, target_seq)
        total_loss = loss_mpjpe + loss_angle

        if do_backward:
            total_loss.backward()

        loss_dict = {
            'mpjpe': loss_mpjpe.item(),
            'angle_loss': loss_angle.item(),
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
