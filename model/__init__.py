import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from typing import Optional, Tuple
from dataclasses import dataclass

from data.maze import MazeDataset
from model.tokenizers.maze import MazeTokenizer, OutputTokens
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

        # NOTE(Nic): this doesn't really need any complex structure, so
        self.output_net = OutputNetwork(
            self.config.d_model, self.config.output_vocab_size, self.config.max_seq_len
        )

        self.lm_head = nn.Linear(
            self.config.d_model, self.config.output_vocab_size, bias=False
        )
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

        # Initial states of L and H nets.
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

        z_L = self.L_net(z_L, z_H + x_tilde)
        z_H = self.H_net(z_H, z_L)

        # NOTE(Nic): in our case, the output seq and input seq have the same length.
        # In cases of other output lengths, this will need different handling. The original implementation
        # appends the input sequence with a set of "puzzle_identifier" embeddings (one for each required output token)
        # and then treats the predictions for these puzzle identifiers as the output to train agains.
        y_pred = self.lm_head(z_H)

        return (z_H, z_L), y_pred

    def forward(self, tokens: torch.Tensor, M: int = 1):
        B, S = tokens.shape

        z = (
            self.z0_H.expand(B, -1, -1)[:, :S].contiguous(),
            self.z0_L.expand(B, -1, -1)[:, :S].contiguous(),
        )

        for m in range(M):
            z, y_pred = self.segment(z, tokens)

        return y_pred

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return F.cross_entropy(
            y_pred.reshape(-1, self.config.output_vocab_size), y_true.reshape(-1)
        )


def train_loop(
    hrm: HRM,
    tokenizer: MazeTokenizer,
    train_samples: MazeDataset,
    test_samples: MazeDataset,
    M: int = 4,
    epochs: int = 1000,
    train_split: float = 0.75,
):
    batch_size = 128
    opt = optim.AdamW(hrm.parameters(), lr=1e-4)

    train_split_subset, val_split_subset = random_split(
        train_samples, [train_split, 1 - train_split]
    )

    train_loader = DataLoader(train_split_subset, batch_size=batch_size)
    val_loader = DataLoader(val_split_subset, batch_size=batch_size)
    test_loader = DataLoader(test_samples, batch_size=batch_size)

    print(
        f"train samples: ~{len(train_loader) * batch_size:,} | val samples: ~{len(val_loader) * batch_size:,} | test samples: ~{len(test_loader) * batch_size:,}"
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    supervision_losses: list[float] = []

    for epoch in range(epochs):

        train_losses.clear()
        hrm.train()

        num_batches = len(train_loader)
        vis_every = math.floor(num_batches * 0.05)
        for i, (x_img, y_img) in enumerate(train_loader):
            x_seq, y_seq = tokenizer(x_img, y_img)

            B, S = x_seq.shape
            z = (
                hrm.z0_H.expand(B, -1, -1)[:, :S].contiguous(),
                hrm.z0_L.expand(B, -1, -1)[:, :S].contiguous(),
            )

            supervision_losses.clear()

            for m in range(M):
                z, y_pred = hrm.segment(z, x_seq)

                loss = F.cross_entropy(
                    y_pred.reshape(-1, len(OutputTokens)),
                    y_seq.reshape(-1),
                    # ignore_index=OutputTokens.IGNORE,
                )

                # NOTE(Nic): need to detach the hidden state before the next segment.
                z = (z[0].detach(), z[1].detach())

                opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(hrm.parameters(), 1.0)
                opt.step()

                supervision_losses.append(loss.item())
                loss_list = ", ".join(
                    list(map(lambda x: f"{x:.4f}", supervision_losses))
                )
                loss_bg = " " * (8 * M - 2 - len(loss_list))

                print(
                    f"[trn] epoch: {epoch}/{epochs} | iter: ({i}/{num_batches})[{m}/{M}] | norm: {grad_norm:.3f} | seg losses: {loss_list + loss_bg}"
                    + " " * 10,
                    end="\r",
                )

            train_losses.extend(supervision_losses)

            if (i + 1) % vis_every == 0:
                num_samples = 5
                y_pred_indices = torch.argmax(y_pred, dim=-1)
                y_pred_imgs = tokenizer.untokenize(y_pred_indices)
                debug_image_arr = torch.concat(
                    (x_img[:num_samples], y_pred_imgs[:num_samples]), dim=0
                )
                save_image(
                    debug_image_arr,
                    f"./test-data/test-{epoch}-{i}.png",
                    nrow=num_samples,
                    padding=1,
                    pad_value=0.5,
                )

        avg_trn_loss = (
            sum(train_losses) / len(train_losses) if len(train_losses) > 0 else 0
        )
        print(f"\n[trn] epoch {epoch}/{epochs} | avg loss: {avg_trn_loss:.4f}")

        # val_losses.clear()
        # hrm.eval()

        # with torch.no_grad():
        #     for i, (x_val_img, y_val_img) in enumerate(val_loader):
        #         x_val_seq, y_val_seq = tokenizer(x_val_img, y_val_img)
        #         seq_len = y_val_seq.size(1)

        #         y_pred = hrm(x_val_seq, seq_len)
        #         loss = hrm.loss(y_pred, y_val_seq)
        #         print(
        #             f"[val] epoch: {epoch}/{epochs} | iter: {i}/{len(val_loader)} | loss: {loss.item():.4f}"
        #             + " " * 10,
        #             end="\r",
        #         )

        #         val_losses.append(loss.item())

        #     avg_val_loss = (
        #         sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0
        #     )
        #     print(f"\n[val] epoch {epoch}/{epochs} | avg loss: {avg_val_loss:.4f}")
