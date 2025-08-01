import math
import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding


class InputNetwork(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_len: int):
        super().__init__()

        # Definition
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Initialization
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape

        # NOTE(Nic): sqrt scaling is not explicitly mentioned in the paper
        x = self.emb(tokens) * math.sqrt(self.d_model)
        x += self.pos(torch.arange(S, dtype=torch.long)) * math.sqrt(self.d_model)
        x = self.proj(x)

        return x
