import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from typing import Optional
from dataclasses import dataclass

from data.maze import MazeDataset
from model.tokenizers.maze import MazeTokenizer
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

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return F.cross_entropy(
            y_pred.reshape(-1, self.config.output_vocab_size), y_true.reshape(-1)
        )


def test_train(hrm: HRM, input: torch.Tensor, target: torch.Tensor):
    opt = optim.AdamW(hrm.parameters(), lr=1e-4)
    for i in range(10_000):
        opt.zero_grad()

        y_pred = hrm(input, target.size(1))
        loss = hrm.loss(target, y_pred)

        print(f"iter: {i} | loss: {loss.item()}")
        loss.backward()

        opt.step()


def train_loop(
    hrm: HRM,
    tokenizer: MazeTokenizer,
    train_samples: MazeDataset,
    test_samples: MazeDataset,
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

    for epoch in range(epochs):

        train_losses.clear()
        hrm.train()

        for i, (x_img, y_img) in enumerate(train_loader):
            x_seq, y_seq = tokenizer(x_img, y_img)
            seq_len = y_seq.size(1)

            opt.zero_grad()

            y_pred = hrm(x_seq, seq_len)
            loss = hrm.loss(y_pred, y_seq)

            train_losses.append(loss.item())

            print(
                f"[trn] epoch: {epoch}/{epochs} | iter: {i}/{len(train_loader)} | loss: {loss.item():.4f}"
                + " " * 10,
                end="\r",
            )

            loss.backward()

            opt.step()

        avg_trn_loss = (
            sum(train_losses) / len(train_losses) if len(train_losses) > 0 else 0
        )
        print(f"\n[trn] epoch {epoch}/{epochs} | avg loss: {avg_trn_loss:.4f}")

        val_losses.clear()
        hrm.eval()

        with torch.no_grad():
            for i, (x_val_img, y_val_img) in enumerate(val_loader):
                x_val_seq, y_val_seq = tokenizer(x_val_img, y_val_img)
                seq_len = y_seq.size(1)

                y_pred = hrm(x_val_seq, seq_len)
                loss = hrm.loss(y_pred, y_val_seq)
                print(
                    f"[val] epoch: {epoch}/{epochs} | iter: {i}/{len(val_loader)} | loss: {loss.item():.4f}"
                    + " " * 10,
                    end="\r",
                )

                val_losses.append(loss.item())

            avg_val_loss = (
                sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0
            )
            print(f"\n[val] epoch {epoch}/{epochs} | avg loss: {avg_val_loss:.4f}")
