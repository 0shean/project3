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
    return MotionAttentionModel(
        num_joints=15, joint_dim=9, dct_n=config.dct_n, hidden_dim=256, num_future_frames=config.target_seq_len
    )



class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)  # [J, J]
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: [B, J, C]
        A = self.A.to(x.device)  # move to same device as input
        x = torch.einsum('ij,bjc->bic', A, x)
        x = self.fc(x)
        return x

def get_adjacency_matrix(num_joints):
    A = np.eye(num_joints)
    bone_connections = [
        (0, 1), (1, 2), (2, 3),  # spine to head
        (1, 4), (4, 5),  # right arm
        (1, 6), (6, 7),  # left arm
        (0, 8), (8, 9), (9, 10),  # right leg
        (0, 11), (11, 12), (12, 13)  # left leg
    ]
    for i, j in bone_connections:
        A[i, j] = 1
        A[j, i] = 1
    return A


class MotionAttentionModel(nn.Module):

    def model_name(self):
        return f"MotionAttentionModel-dct{self.dct_n}"
    def __init__(self, num_joints=15, joint_dim=9, dct_n=20, hidden_dim=256, num_future_frames=24):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.dct_n = dct_n
        self.num_future = num_future_frames
        self.hidden_dim = hidden_dim

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
        A = get_adjacency_matrix(self.num_joints)
        self.gcn1 = GCNLayer(in_channels=4608, out_channels=hidden_dim, A=A)
        self.gcn2 = GCNLayer(in_channels=self.hidden_dim, out_channels=self.num_future * self.joint_dim, A=A)

    def forward(self, batch):
        """
        batch.dct_input: [B, 15, 9, 40]
        batch.dct_history: [B, N, 15, 9, 40]
        """
        x = batch.dct_input  # [B, 15, 9, 40]
        h = batch.dct_history  # [B, N, 15, 9, 40]

        B, J, D, K = x.shape
        N = h.shape[1]

        # Flatten joints and axes: [B, 135, 40]
        x_flat = x.view(B, J * D, K)  # query: [B, 135, 40]
        h_flat = h.view(B, N, J * D, K)  # history: [B, N, 135, 40]

        # Encode query: [B, 135, H]
        Q = self.query_net(x_flat)  # [B, 135, H]

        # Encode history as keys and values
        K_enc = self.key_net(h_flat)  # [B, N, 135, H]
        V_enc = self.value_net(h_flat)  # [B, N, 135, H]

        # Compute attention scores: [B, 135, N]
        scores = torch.einsum('bij, bnij -> bin', Q, K_enc) / (Q.shape[-1] ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, 135, N]

        # Aggregate over values: weighted sum → [B, 135, H]
        context = torch.einsum('bin, bnij -> bij', attn_weights, V_enc)

        # Concatenate query and attended context: [B, 135, 2H]
        features = torch.cat([Q, context], dim=-1)  # [B, 135, 2H]
        features = features.view(B, self.num_joints, self.joint_dim, -1)  # [B, 15, 9, 2H]
        features = features.reshape(B, self.num_joints, -1)  # [B, 15, 4608]


        x = self.gcn1(features)  # [B, 15, H]
        x = F.relu(x)
        x = self.gcn2(x)  # [B, 15, 24*9]

        out = x.view(B, self.num_joints, self.num_future, self.joint_dim).permute(0, 2, 1, 3)  # [B, 24, 15, 9]

        # Inverse DCT
        output_np = out.detach().cpu().numpy()
        reconstructed = np.zeros((B, self.num_future, self.num_joints, 3, 3), dtype=np.float32)

        for b in range(B):
            for j in range(self.num_joints):
                for d in range(self.joint_dim):
                    signal = output_np[b, :, j, d]
                    time_series = idct(signal, n=self.num_future, norm='ortho')
                    output_np[b, :, j, d] = time_series
                for t in range(self.num_future):
                    reconstructed[b, t, j] = output_np[b, t, j].reshape(3, 3)

        pred = torch.tensor(reconstructed, dtype=torch.float32, device=x.device)  # [B, 24, 15, 3, 3]
        return {'predictions': pred}

    def backward(self, batch, model_out):
        """
        Compute the loss between model_out['predictions'] and ground-truth in batch.
        """
        pred = model_out['predictions']  # [B, 24, 15, 3, 3]
        gt = batch.poses[:, -self.num_future:].reshape(-1, self.num_future, 15, 3, 3)


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
