"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn

from data import AMASSBatch
from losses import mse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import idct

def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return DummyModel(config)

class MotionAttentionModel(nn.Module):
    def __init__(self, num_joints=15, joint_dim=9, dct_n=20, hidden_dim=256, num_future_frames=24):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.dct_n = dct_n
        self.num_future = num_future_frames

        # Embedding networks for keys, queries, and values
        self.query_net = nn.Sequential(
            nn.Linear(dct_n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.key_net = nn.Sequential(
            nn.Linear(dct_n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(dct_n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_future_frames * joint_dim)
        )

    def forward(self, batch):
        x = batch.dct_input  # shape: [B, 15, 9, 20]
        B, J, D, K = x.shape

        x = x.view(B, J * D, K)  # [B, 135, 20]

        Q = self.query_net(x)  # [B, 135, H]
        K_ = self.key_net(x)   # [B, 135, H]
        V = self.value_net(x)  # [B, 135, H]

        attn_weights = torch.bmm(Q, K_.transpose(1, 2)) / (Q.shape[-1] ** 0.5)
        attn = torch.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn, V)

        features = torch.cat([Q, context], dim=-1)  # [B, 135, 2H]
        out = self.predictor(features)  # [B, 135, 24*9]
        out = out.view(B, self.num_joints, self.joint_dim, self.num_future).permute(0, 3, 1, 2)  # [B, 24, 15, 9]

        # Inverse DCT and reshape to [B, 24, 15, 3, 3]
        output_np = out.detach().cpu().numpy()
        reconstructed = np.zeros((B, self.num_future, self.num_joints, 3, 3), dtype=np.float32)

        for b in range(B):
            for j in range(self.num_joints):
                for d in range(self.joint_dim):
                    signal = output_np[b, :, j, d]  # (24,)
                    time_series = idct(signal, n=self.num_future, norm='ortho')
                    output_np[b, :, j, d] = time_series
                for t in range(self.num_future):
                    reconstructed[b, t, j] = output_np[b, t, j].reshape(3, 3)

        # Return output in expected format
        pred = torch.tensor(reconstructed, dtype=torch.float32, device=x.device)  # [B, 24, 15, 3, 3]
        return {'predictions': pred}

    def backward(self, batch, model_out):
        """
        Compute the loss between model_out['predictions'] and ground-truth in batch.
        """
        pred = model_out['predictions']  # [B, 24, 15, 3, 3]
        gt = batch.poses[:, -self.num_future:]  # [B, 24, 15, 3, 3]

        # Compute per-frame MSE
        loss = F.mse_loss(pred, gt, reduction='mean')

        return {'total_loss': loss}, gt


class BaseModel(nn.Module):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the model, called automatically by the initializer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, batch: AMASSBatch):
        """The forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch: AMASSBatch, model_out):
        """The backward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

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
