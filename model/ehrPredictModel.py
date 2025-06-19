# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class PatientEncoder(nn.Module):
    """双向 LSTM，取 (h_fwd_last || h_bwd_last) 作为患者向量"""
    def __init__(self, dim_in: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(
            dim_in, hidden, n_layers,
            batch_first=True, bidirectional=True
        )

    def forward(self, x, mask):
        lengths = mask.sum(1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.rnn(packed)        # [2*n_layers, B, H]
        # 连接最后一层的正向 + 反向
        return torch.cat((h_n[-2], h_n[-1]), dim=-1)   # [B, 2H]


class MortalityClassifier(nn.Module):
    """PatientEncoder → MLP → logit/值"""
    def __init__(self,
                 dim_in: int,
                 hidden: int = 128,
                 mlp_hidden: int = 64,
                 dropout: float = 0.3,
                 out_dim: int = 1):
        super().__init__()
        self.encoder = PatientEncoder(dim_in, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim)    # out_dim 1(单任务) | 3(多任务)
        )

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        out = self.mlp(z)
        # 单输出直接 squeeze，三输出保持 [B,3]
        return out.squeeze(-1) if out.shape[-1] == 1 else out
