import torch
import torch.nn as nn

class PatientEncoder(nn.Module):
    """Bidirectional LSTM, using (h_fwd_last || h_bwd_last) as the patient representation"""
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
        return torch.cat((h_n[-2], h_n[-1]), dim=-1)   # [B, 2H]


class EHRPredictor(nn.Module):
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
            nn.Linear(mlp_hidden, out_dim)    # out_dim 1(single task) | 3(multitask)
        )

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        out = self.mlp(z)
        # single task: squeeze, multitask: [B,3]
        return out.squeeze(-1) if out.shape[-1] == 1 else out
