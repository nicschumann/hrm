import torch
import torch.nn as nn

from typing import Optional, Tuple
from dataclasses import dataclass

from model.input import InputNetwork
from model.recurrence import RecurrentModule


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

        # NOTE(Nic): In the reconfigured codebase, the output function really doesn't
        # need any complex structure – it's just a matrix we map across each z_H Hidden state,
        # and we have one z_H for each input token.
        # self.output_net = OutputNetwork(
        #     self.config.d_model, self.config.output_vocab_size, self.config.max_seq_len
        # )

        self.lm_head = nn.Linear(
            self.config.d_model, self.config.output_vocab_size, bias=False
        )
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

        # NOTE(Nic): Hidden states of L and H nets; one for each sequence position up to max_seq_len
        self.z0_L = nn.Parameter(
            torch.randn(1, self.config.max_seq_len, self.config.d_model) * 0.02
        )
        self.z0_H = nn.Parameter(
            torch.randn(1, self.config.max_seq_len, self.config.d_model) * 0.02
        )

    def segment(
        self, z: Tuple[torch.Tensor, torch.Tensor], tokens: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # NOTE(Nic): this runs a single HRM training segment.
        # Returns
        # - (z_H, z_L): new hidden states produced from this segment,
        # - y_pred: logits predicted for the segment
        B, S = tokens.shape

        z_H, z_L = z  # (B, S, D_MODEL), (B, S, D_MODEL)
        x_tilde = self.input_net(tokens)  # (B, S, D_MODEL)

        with torch.no_grad():
            for i in range(self.config.N * self.config.T - 1):
                z_L = self.L_net(z_L, z_H + x_tilde)

                if (i + 1) % self.config.T == 0:
                    z_H = self.H_net(z_H, z_L)

        # NOTE(Nic): final step in the segment, which accumulates a gradient.
        z_L = self.L_net(z_L, z_H + x_tilde)
        z_H = self.H_net(z_H, z_L)

        # NOTE(Nic): in our case, the output seq and input seq have the same length.
        # In cases of other output lengths, this will need different handling. The original implementation
        # appends the input sequence with a set of "puzzle_identifier" embeddings (one for each required output token)
        # and then treats the predictions for these puzzle identifiers as the output to train agains.
        y_pred = self.lm_head(z_H)

        return (z_H, z_L), y_pred

    def initial_state(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE(Nic): given a batch of token sequences, returns a pair of initial hidden states
        # for that batch.
        B, S = tokens.shape

        return (
            self.z0_H.expand(B, -1, -1)[:, :S].contiguous(),
            self.z0_L.expand(B, -1, -1)[:, :S].contiguous(),
        )

    def forward(self, tokens: torch.Tensor, M: int = 1):
        # Straightforward wrapper around segment. Creates an initial hidden state,
        # runs M segments, and then returns the final prediction from the trajectory
        z = self.initial_state(tokens)

        for m in range(M):
            z, y_pred = self.segment(z, tokens)

        return y_pred
