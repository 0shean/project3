import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import idct

from data import AMASSBatch
from losses import mse


def create_model(config):
    return MotionAttentionModel(
        num_joints=15,
        joint_dim=9,
        dct_n=config.dct_n,
        hidden_dim=config.hidden_dim if hasattr(config, 'hidden_dim') else 256,
        num_future_frames=config.target_seq_len
    )


class TimeEmbedding(nn.Module):
    def __init__(self, num_heads, emb_dim=64, max_windows=500):
        super().__init__()
        self.emb = nn.Embedding(max_windows, emb_dim)
        self.proj = nn.Linear(emb_dim, num_heads)

    def forward(self, idxs):
        h = self.emb(idxs)  # [B, N, emb_dim]
        return self.proj(h)  # [B, N, num_heads]


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        A = self.A.to(x.device)
        x = torch.einsum('ij,bjc->bic', A, x)
        return self.fc(x)


def get_adjacency_matrix(num_joints):
    A = np.eye(num_joints)
    bone_connections = [
        (0, 1), (1, 2), (2, 3),
        (1, 4), (4, 5),
        (1, 6), (6, 7),
        (0, 8), (8, 9), (9, 10),
        (0, 11), (11, 12), (12, 13)
    ]
    for i, j in bone_connections:
        A[i, j] = 1
        A[j, i] = 1
    return A


class MotionAttentionModel(nn.Module):
    def model_name(self):
        return f"MotionAttentionModel-dct{self.dct_n}-mh{self.num_heads}"

    def __init__(
        self, num_joints=15, joint_dim=9, dct_n=20,
        hidden_dim=256, num_future_frames=24
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.dct_n = dct_n
        self.num_future = num_future_frames
        self.hidden_dim = hidden_dim
        self.num_heads = 8
        self.d_model = hidden_dim

        # Project DCT inputs to model dimension
        self.input_proj = nn.Linear(self.dct_n, self.d_model)

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            batch_first=True
        )

        # Time embeddings for memory windows
        self.time_emb = TimeEmbedding(self.num_heads, emb_dim=64)

        # Graph convolutional predictor
        A = get_adjacency_matrix(self.num_joints)
        in_ch = self.joint_dim * 2 * self.d_model
        self.gcn1 = GCNLayer(in_channels=in_ch, out_channels=self.hidden_dim, A=A)
        self.gcn2 = GCNLayer(in_channels=self.hidden_dim, out_channels=self.num_future * self.joint_dim, A=A)

    def forward(self, batch):
        x = batch.dct_input  # [B, J, D, K]
        h = batch.dct_history  # [B, N, J, D, K]

        B, J, D, K = x.shape
        N = h.shape[1]

        # Flatten to sequences
        x_flat = x.view(B, J*D, K)      # [B, 135, dct_n]
        h_flat = h.view(B, N, J*D, K)   # [B, N, 135, dct_n]

        # Project
        q = self.input_proj(x_flat)     # [B, 135, d_model]
        hp = self.input_proj(h_flat)    # [B, N, 135, d_model]

        # Time bias
        idxs = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)  # [B, N]
        tb = self.time_emb(idxs)                                           # [B, N, num_heads]
        tb = tb.view(B*N, self.num_heads)
        tb = self.input_proj(tb).view(B, N, 1, self.d_model)
        hp = hp + tb  # add to history embeddings

        # Reshape history for attention
        h_seq = hp.view(B, N * J * D, self.d_model)  # [B, N*135, d_model]

        # Attention
        attn_out, _ = self.mha(query=q, key=h_seq, value=h_seq)  # [B, 135, d_model]

        # Combine
        features = torch.cat([q, attn_out], dim=-1)  # [B, 135, 2*d_model]
        features = features.view(B, J, D, 2*self.d_model)
        features = features.reshape(B, J, -1)  # [B, J, joint_dim*2*d_model]

        # GCN prediction
        x = self.gcn1(features)
        x = F.relu(x)
        x = self.gcn2(x)

        # Reshape to DCT coefficients
        out = x.view(B, J, self.num_future, self.joint_dim).permute(0, 2, 1, 3)  # [B, T, J, 9]

        # Inverse DCT to rotation matrices
        out_np = out.detach().cpu().numpy()
        rec = np.zeros((B, self.num_future, self.num_joints, 3, 3), dtype=np.float32)
        for b in range(B):
            for j in range(self.num_joints):
                for d in range(self.joint_dim):
                    coeffs = out_np[b, :, j, d]
                    ts = idct(coeffs, n=self.num_future, norm='ortho')
                    out_np[b, :, j, d] = ts
                rec[b, :, j] = out_np[b, :, j].reshape(self.num_future, 3, 3)

        pred = torch.tensor(rec, dtype=torch.float32, device=x.device)
        return {'predictions': pred}

    def backward(self, batch, model_out):
        pred = model_out['predictions']
        gt = batch.poses[:, -self.num_future:].reshape(-1, self.num_future, self.num_joints, 3, 3)
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
