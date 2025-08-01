import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from typing import Optional


class RotaryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(dim=self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        assert D == self.d_model

        qkv: torch.Tensor = self.w_qkv(x)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (Q,K,V), batch, n_heads, seq_len, d_head
        q, k, v = qkv[0], qkv[1], qkv[2]  # NOTE(Nic): qkv.chunk(3, dim=0)

        # NOTE(Nic): we might be able to use rotate_queries_and_keys for efficiency.
        q = self.rope.rotate_queries_or_keys(q, seq_dim=2)
        k = self.rope.rotate_queries_or_keys(k, seq_dim=2)

        # NOTE(Nic): not causal, by default
        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(B, S, D)

        return self.w_o(attn)


class GLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()

        d_ff = d_ff or d_model * 4

        self.w_1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_and_value: torch.Tensor = self.w_1(x)  # B, S, 2*d_ff
        gate, value = gate_and_value.chunk(2, dim=-1)
        gated = F.silu(gate) * value

        return self.w_2(gated)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None) -> None:
        super().__init__()
        self.attn = RotaryAttention(d_model, n_heads)
        self.ff = GLUFeedForward(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-8, elementwise_affine=True)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-8, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class RecurrentModule(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, n_layers: int, d_ff: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, z: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:

        combined_seq = z + z_other

        for layer in self.layers:
            combined_seq = layer(combined_seq)

        return combined_seq
