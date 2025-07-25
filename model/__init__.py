import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from model.input import InputNetwork
from model.recurrence import RecurrentModule
from model.output import OutputNetwork


@dataclass
class HRMConfig:
    input_vocab_size: int
    output_vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: Optional[int] = None

    N: int = 2  # number of high-level module cycles
    T: int = 4  # number of low-level module cycles

    max_seq_len: int = 256


class HRM(nn.Module):
    def __init__(self, config: HRMConfig):
        super().__init__()

        self.config = config

        self.input_net = InputNetwork(
            self.config.input_vocab_size, self.config.d_model, self.config.max_seq_len
        )

        self.H_net = RecurrentModule(
            self.config.d_model,
            self.config.n_heads,
            self.config.n_layers,
            self.config.d_ff,
        )

        self.L_net = RecurrentModule(
            self.config.d_model,
            self.config.n_heads,
            self.config.n_layers,
            self.config.d_ff,
        )

        self.output_net = OutputNetwork(
            self.config.d_model, self.config.output_vocab_size, self.config.max_seq_len
        )

        # Initial states of L and H nets.
        self.z0_L = nn.Parameter(torch.randn(1, self.config.d_model) * 0.02)
        self.z0_H = nn.Parameter(torch.randn(1, self.config.d_model) * 0.02)
        self.zeros = torch.zeros((1, 1, self.config.d_model), requires_grad=False)

    def forward(self, tokens: torch.Tensor, seq_len: Optional[int]):
        B, S = tokens.shape
        seq_len = seq_len or self.config.max_seq_len

        x_tilde = self.input_net(tokens)
        z_L = self.z0_L.expand(B, -1)
        z_H = self.z0_H.expand(B, -1)

        with torch.no_grad():
            for i in range(self.config.N * self.config.T - 1):

                z_L = self.L_net(z_L, z_H, x_tilde)

                if (i + 1) % self.config.T == 0:
                    z_H = self.H_net(z_H, z_L, self.zeros)

        z_L = self.L_net(z_L.detach(), z_H.detach(), x_tilde)
        z_H = self.H_net(z_H.detach(), z_L, self.zeros)

        return self.output_net(z_H, seq_len)

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return F.cross_entropy(
            y_pred.reshape(-1, self.config.output_vocab_size), y_true.reshape(-1)
        )
