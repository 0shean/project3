import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import dct, idct


class MotionAttentionEncoder(nn.Module):
    """
    Splits the input sequence into overlapping windows, encodes via DCT,
    and applies self-attention between a query (recent window) and past windows.
    """

    def __init__(self, seq_len, window_size, stride, dct_size, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.stride = stride
        self.dct_size = dct_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections for Q, K, V
        dim_C = dct_size * embed_dim
        self.lin_q = nn.Linear(dim_C, embed_dim)
        self.lin_k = nn.Linear(dim_C, embed_dim)
        self.lin_v = nn.Linear(dim_C, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # x: (B, T, D) where T = seq_len, D = embed_dim
        B, T, D = x.size()
        # 1) split into windows
        windows = [x[:, i:i + self.window_size, :] for i in range(0, T - self.window_size + 1, self.stride)]
        w = len(windows)
        assert w >= 2, "Need at least one past and one query window"
        # 2) DCT encode each window along time dim
        coeffs = []
        for win in windows:
            wperm = win.permute(0, 2, 1)  # (B, D, W)
            d = dct(wperm, norm='ortho')  # (B, D, W)
            dsub = d[:, :, :self.dct_size]  # (B, D, dct_size)
            coeffs.append(dsub.reshape(B, -1))  # (B, dct_size*D)
        # stack: (w, B, C)
        coeffs = torch.stack(coeffs, dim=0)
        past, query = coeffs[:-1], coeffs[-1]
        # project
        Q = self.lin_q(query)  # (B, E)
        K = self.lin_k(past)  # (w-1, B, E)
        V = self.lin_v(past)  # (w-1, B, E)
        # reshape for attention: (L, N, E)
        Q = Q.unsqueeze(0)  # (1, B, E)
        attn_out, _ = self.attn(Q, K, V)
        attn_out = attn_out.squeeze(0)  # (B, E)
        return self.out_proj(attn_out)


class StructuredPredictionLayer(nn.Module):
    """
    Predicts joint rotations sequentially along the kinematic tree.
    """

    def __init__(self, feature_dim, joint_dim, parent_indices, hidden_size, share_weights=False):
        super().__init__()
        self.joint_dim = joint_dim
        self.parent_indices = parent_indices
        self.num_joints = len(parent_indices)
        self.share_weights = share_weights
        if share_weights:
            self.shared_mlp = nn.Sequential(
                nn.Linear(feature_dim + joint_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, joint_dim)
            )
        else:
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim + joint_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, joint_dim)
                ) for _ in range(self.num_joints)
            ])

    def forward(self, features):
        B = features.size(0)
        device = features.device
        preds = [None] * self.num_joints
        for i, parent in enumerate(self.parent_indices):
            parent_rot = torch.zeros(B, self.joint_dim, device=device) if parent < 0 else preds[parent]
            inp = torch.cat([features, parent_rot], dim=-1)
            out = self.shared_mlp(inp) if self.share_weights else self.mlps[i](inp)
            preds[i] = out
        return torch.cat(preds, dim=-1)


class MotionAttentionSPLModel(nn.Module):
    """
    Embeds input sequence, applies motion-attention,
    and predicts future frames via SPL.
    """

    def __init__(self, config):
        super().__init__()
        in_dim = config.input_dim
        seq_len = config.seed_seq_len
        self.future_len = config.target_seq_len
        emb_dim = config.model_embed_dim
        self.frame_embed = nn.Linear(in_dim, emb_dim)
        self.attn = MotionAttentionEncoder(
            seq_len=seq_len,
            window_size=config.ma_window_size,
            stride=config.ma_stride,
            dct_size=config.ma_dct_size,
            embed_dim=emb_dim,
            num_heads=config.ma_num_heads,
            dropout=config.ma_dropout
        )
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True) if config.model_use_gru else None
        feat_dim = emb_dim * (1 + (1 if self.gru else 0))
        self.spl = StructuredPredictionLayer(
            feature_dim=feat_dim,
            joint_dim=config.joint_dim,
            parent_indices=config.parent_indices,
            hidden_size=config.spl_hidden_size,
            share_weights=config.spl_share_weights
        )

    def forward(self, seq):
        x = self.frame_embed(seq)  # (B, seq_len, emb_dim)
        attn_feat = self.attn(x)  # (B, emb_dim)
        feats = [attn_feat]
        if self.gru:
            _, h = self.gru(x[:, -self.gru.input_size:], None)
            feats.append(h.squeeze(0))
        feat = torch.cat(feats, dim=-1)  # (B, feat_dim)
        pred_frame = self.spl(feat)  # (B, num_joints*joint_dim)
        return pred_frame.unsqueeze(1).repeat(1, self.future_len, 1)
