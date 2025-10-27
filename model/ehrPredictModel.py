import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)  # 打分向量

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # raw attention scores
        attn = self.score(h).squeeze(-1)
        attn = attn.masked_fill(mask == 0, -1e9)
        w = torch.softmax(attn, dim=-1)
        pooled = torch.sum(w.unsqueeze(-1) * h, dim=1)
        return pooled


class PatientEncoder(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=dim_in,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = AttnPool(dim=2 * hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(out_packed, batch_first=True)

        if h.size(1) < mask.size(1):
            pad_len = mask.size(1) - h.size(1)
            h = torch.cat([h, h.new_zeros(h.size(0), pad_len, h.size(2))], dim=1)

        return self.pool(h, mask)      # [B, 2H]


class EHRPredictor(nn.Module):

    def __init__(
        self,
        dim_in: int,
        hidden: int = 128,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        self.encoder = PatientEncoder(dim_in, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, mask)
        logits = self.mlp(z)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits
