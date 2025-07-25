import torch
import torch.nn as nn

from typing import Optional


# NOTE(Nic@10)


class OutputNetwork(nn.Module):
    def __init__(
        self, d_model: int, vocab_size: int, max_seq_len: int, stablemax: bool = False
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_stablemax = stablemax

        self.proj = nn.Linear(d_model, self.max_seq_len * self.vocab_size, bias=False)

    def forward(self, z_h: torch.Tensor, seq_len: Optional[int] = None):
        B, D = z_h.shape
        assert D == self.d_model
        seq_len = seq_len if seq_len is not None else self.max_seq_len

        out: torch.Tensor = self.proj(z_h)
        out = out.view(B, self.max_seq_len, self.vocab_size)
        out = out[:, :seq_len, :]

        # return logits
        return out
