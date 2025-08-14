import math
import torch
import wandb
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from data.maze import MazeDataset
from model.tokenizers.maze import MazeTokenizer, OutputTokens


from model.hrm import HRM, HRMConfig


SplitType = Literal["train"] | Literal["val"]
MetricType = Literal["acc"] | Literal["loss"]


class SegmentMetrics:
    # NOTE(Nic): segment_* members contain M arrays, one for each segment.
    # each array carries a list of losses for that segment in a given epoch.
    M: int
    segments: dict[MetricType, dict[SplitType, list[list[float]]]]

    def __init__(self, M: int = 4):
        self.M = M
        self.segments = {
            "loss": {
                "train": [[] for _ in range(M)],
                "val": [[] for _ in range(M)],
            },
            "acc": {
                "train": [[] for _ in range(M)],
                "val": [[] for _ in range(M)],
            },
        }

    def add_segment_values(
        self,
        values: list[float],
        split: SplitType = "train",
        metric: MetricType = "loss",
    ):
        assert (
            len(values) == self.M
        ), f"expected length={self.M} segment metrics, but got {len(values)}"

        for m, value in enumerate(values):
            self.segments[metric][split][m].append(value)

    def get_segments(
        self, index: int = -1, split: SplitType = "train", metric: MetricType = "loss"
    ) -> list[float]:
        data = self.segments[metric][split]
        return [data[m][index] for m in range(self.M) if len(data[m]) > index]

    def get_metrics(
        self, index: int = -1, split: SplitType = "train", metric: MetricType = "loss"
    ) -> dict[str, float]:
        segs_at_index = self.get_segments(index, split, metric)
        assert len(segs_at_index) == self.M
        return {
            f"{split}/segment-{m+1}-{metric}": segs_at_index[m] for m in range(self.M)
        }

    def get_average_metrics(
        self, split: SplitType = "train", metric: MetricType = "loss"
    ) -> dict[str, float]:
        return {
            f"{split}/segment-{m+1}-avg-{metric}": sum(self.segments[metric][split][m])
            / len(self.segments[metric][split][m])
            for m in range(self.M)
        }

    def clear(self, split: SplitType = "train", metric: MetricType = "loss"):
        for m in range(self.M):
            self.segments[metric][split][m].clear()

    def clear_all(self):
        splits: list[SplitType] = ["train", "val"]
        metrics: list[MetricType] = ["loss", "acc"]
        for split in splits:
            for metric in metrics:
                self.clear(split, metric)


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    epochs: int


def accuracies(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # to calculate the accuracy, we want the number of route tokens correctly predicted
    # out of the total number of false route tokens predicted by the model + the total number
    # of true route tokens.

    # NOTE(Nic): this will contain both predicted and true route indices
    pred_tokens = y_pred.argmax(dim=-1)

    all_route_tokens = pred_tokens.clone()
    y_true_mask = y_true == OutputTokens.ROUTE
    all_route_tokens[y_true_mask] = OutputTokens.ROUTE

    total_route_tokens = (
        (all_route_tokens == OutputTokens.ROUTE)
        .type(torch.int32)
        .sum(dim=-1)
        .type(torch.float32)
    )

    correct_route_tokens = (
        ((pred_tokens == y_true) & (y_true == OutputTokens.ROUTE))
        .type(torch.int32)
        .sum(dim=-1)
        .type(torch.float32)
    )

    accuracies = correct_route_tokens / total_route_tokens

    # print("true route tokens: ", true_route_tokens[0])
    # print("total route tokens: ", total_route_tokens[0])
    # print("correct route tokens: ", correct_route_tokens[0])
    # print(accuracies)

    return accuracies


def get_device() -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    return device


def train_loop(
    hrm: HRM,
    tokenizer: MazeTokenizer,
    train_samples: MazeDataset,
    test_samples: MazeDataset,
    M: int = 4,  # NOTE(Nic): number of supervision steps
    epochs: int = 100,
    train_split: float = 0.75,
):

    device = get_device()
    if device == "cuda":
        print("compiling model...")
        hrm = torch.compile(hrm)  # type: ignore

    train_config = TrainConfig(lr=1e-4, batch_size=128, epochs=epochs)

    opt = optim.AdamW(hrm.parameters(), lr=train_config.lr)

    train_split_subset, val_split_subset, _ = random_split(
        train_samples, [1024, 8192, len(train_samples) - 9216]
    )

    train_loader = DataLoader(train_split_subset, batch_size=train_config.batch_size)
    val_loader = DataLoader(val_split_subset, batch_size=train_config.batch_size)
    test_loader = DataLoader(test_samples, batch_size=train_config.batch_size)
    train_loader_length = len(train_loader)
    num_train_samples = train_loader_length * train_config.batch_size
    num_val_samples = len(val_loader) * train_config.batch_size
    num_test_samples = len(test_loader) * train_config.batch_size

    target_train_samples = 3_112_704
    samples_per_epoch = num_train_samples
    target_epochs = math.ceil(target_train_samples / samples_per_epoch)

    wandb_run = wandb.init(
        entity="type-tools",
        project="hrm",
        config={
            "lr": train_config.lr,
            "batch-size": train_config.batch_size,
            "hrm/params": sum(p.numel() for p in hrm.parameters()),
            "hrm/N": hrm.config.N,
            "hrm/T": hrm.config.T,
            "hrm/M": M,
            "hrm/n_layers": hrm.config.n_layers,
            "hrm/n_heads": hrm.config.d_model,
            "hrm/d_ff": hrm.config.d_ff,
            "maze-dims": (9, 9),
            "train-samples": num_train_samples,
            "val-samples": num_val_samples,
            "test-samples": num_test_samples,
            "epochs": target_epochs,
            "device": device,
        },
    )

    print(
        f"train samples: ~{num_train_samples:,} | val samples: ~{num_val_samples:,} | test samples: ~{num_test_samples:,}"
    )

    segment_metrics = SegmentMetrics(M=M)
    segment_losses: list[float] = []
    segment_accuracies: list[float] = []

    hrm = hrm.to(device)

    for epoch in range(target_epochs):

        hrm.train()
        # NOTE(Nic): clear old segments from the tracker.
        segment_metrics.clear_all()
        vis_every_train = max(math.floor(len(train_loader) * 0.05), 1)
        vis_every_val = max(math.floor(len(val_loader) * 0.05), 1)

        for i, (x_img, y_img) in enumerate(train_loader):
            # NOTE(Nic): We use the sample_index as the global step for indexing our metrics
            # NOTE(Nic): We report batch statistics for the last segment in each train batch, but only validation averages.
            sample_index = epoch * train_loader_length + i
            x_seq, y_seq = tokenizer(x_img, y_img)
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)

            z = hrm.initial_state(x_seq)

            segment_losses.clear()
            segment_accuracies.clear()

            for m in range(M):
                z, y_pred = hrm.segment(z, x_seq)

                loss = F.cross_entropy(
                    y_pred.reshape(-1, len(OutputTokens)),
                    y_seq.reshape(-1),
                    # ignore_index=OutputTokens.IGNORE,
                )

                acc = accuracies(y_pred, y_seq).mean()

                # NOTE(Nic): need to detach the hidden state before the next segment.
                z = (z[0].detach(), z[1].detach())

                opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(hrm.parameters(), 1.0)
                opt.step()

                # NOTE(Nic): add results of this segment on this batch to the tracker.

                segment_losses.append(loss.item())
                segment_accuracies.append(acc.item())

                # NOTE(Nic): log out the results of this segment to stdout

                loss_list = ", ".join(
                    list(
                        map(
                            lambda x: f"{x[1]:.4f} ({segment_accuracies[x[0]] * 100:.1f}%)",
                            enumerate(segment_losses),
                        )
                    )
                )
                loss_bg = " " * (14 * M - 2 - len(loss_list))

                print(
                    f"[trn] epoch: {epoch}/{target_epochs} | iter: ({i}/{len(train_loader)})[{m}/{M}] | norm: {grad_norm:.3f} | segs: {loss_list + loss_bg}"
                    + " " * 10,
                    end="\r",
                )

                # end stdout log

            segment_metrics.add_segment_values(
                segment_losses, split="train", metric="loss"
            )
            segment_metrics.add_segment_values(
                segment_accuracies, split="train", metric="acc"
            )

            # NOTE(Nic): Log a single batch's worth of segments to wandb...

            base_log_data = {
                "epoch": epoch,
                "sample_step": sample_index,
                "train/grad-norm": grad_norm.item(),
            }

            segments_loss_data = segment_metrics.get_metrics(
                split="train", metric="loss"
            )
            segments_acc_data = segment_metrics.get_metrics(split="train", metric="acc")

            log_data = base_log_data | segments_loss_data | segments_acc_data

            wandb_run.log(
                data=log_data,
                step=sample_index,
                # NOTE(Nic): if we're on the last train sample, don't commit yet, cause we have val data to add later.
                commit=sample_index < train_loader_length - 1,
            )

            # end wandb log

            # NOTE(Nic): periodically log some images out for reference.

            if (i + 1) % vis_every_train == 0:

                num_samples = 5
                y_pred_indices = torch.argmax(y_pred, dim=-1)
                y_pred_imgs = tokenizer.untokenize(y_pred_indices, y_seq).cpu()
                debug_image_arr = torch.concat(
                    (x_img[:num_samples], y_pred_imgs[:num_samples]), dim=0
                )
                save_image(
                    debug_image_arr,
                    f"./test-data/test-{epoch}-{i}.png",
                    nrow=num_samples,
                    padding=1,
                    pad_value=0.25,
                )

        avg_trn_loss = segment_metrics.get_average_metrics(
            split="train", metric="loss"
        )[f"train/segment-{M}-avg-loss"]

        avg_trn_acc = segment_metrics.get_average_metrics(split="train", metric="acc")[
            f"train/segment-{M}-avg-acc"
        ]

        print(
            f"\n[trn] epoch: {epoch}/{target_epochs} | seg {M}:  {avg_trn_loss:.4f} ({avg_trn_acc*100:.1f}%)"
        )

        hrm.eval()

        with torch.no_grad():
            for i, (x_val_img, y_val_img) in enumerate(val_loader):
                x_val_seq, y_val_seq = tokenizer(x_val_img, y_val_img)
                x_val_seq = x_val_seq.to(device)
                y_val_seq = y_val_seq.to(device)
                z_val = hrm.initial_state(x_val_seq)

                segment_losses.clear()
                segment_accuracies.clear()

                for m in range(M):

                    z_val, y_val_pred = hrm.segment(z_val, x_val_seq)
                    val_loss = F.cross_entropy(
                        y_val_pred.reshape(-1, len(OutputTokens)),
                        y_val_seq.reshape(-1),
                        # ignore_index=OutputTokens.IGNORE,
                    )
                    val_acc = accuracies(y_val_pred, y_val_seq).mean()

                    segment_losses.append(val_loss.item())
                    segment_accuracies.append(val_acc.item())

                segment_metrics.add_segment_values(
                    segment_losses, split="val", metric="loss"
                )

                segment_metrics.add_segment_values(
                    segment_accuracies, split="val", metric="acc"
                )

                print(
                    f"[val] epoch: {epoch}/{target_epochs} | iter: {i}/{len(val_loader)} | seg {M}: {val_loss.item():.4f} ({val_acc.item()*100:.1f}%)"
                    + " " * 10,
                    end="\r",
                )

                # NOTE(Nic): occasionally save some val images
                if (i + 1) % vis_every_val == 0:
                    num_samples = 5
                    y_val_pred_indices = torch.argmax(y_val_pred, dim=-1)
                    y_val_pred_imgs = tokenizer.untokenize(
                        y_val_pred_indices, y_val_seq
                    ).cpu()
                    val_debug_image_arr = torch.concat(
                        (x_val_img[:num_samples], y_val_pred_imgs[:num_samples]), dim=0
                    )
                    save_image(
                        val_debug_image_arr,
                        f"./test-data/test-{epoch}-val-{i}.png",
                        nrow=num_samples,
                        padding=1,
                        pad_value=0.25,
                    )

            avg_val_loss = segment_metrics.get_average_metrics(
                split="val", metric="loss"
            )

            avg_val_acc = segment_metrics.get_average_metrics(split="val", metric="acc")

            avg_train_loss = segment_metrics.get_average_metrics(
                split="train", metric="loss"
            )

            avg_train_acc = segment_metrics.get_average_metrics(
                split="train", metric="acc"
            )

            print(
                f"\n[val] epoch: {epoch}/{target_epochs} | avg loss: {avg_val_loss[f"val/segment-{M}-avg-loss"]:.4f} | avg acc: {avg_val_acc[f"val/segment-{M}-avg-acc"]*100:.1f}%"
            )

            # NOTE(Nic): send the epoch summary stats over for both validation and training
            val_log_data = avg_val_loss | avg_val_acc | avg_train_loss | avg_train_acc

            wandb_run.log(
                data=val_log_data,
                step=sample_index,
                commit=True,
            )

    wandb_run.finish()
