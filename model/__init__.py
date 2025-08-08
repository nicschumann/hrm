import math
import torch
import wandb
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from data.maze import MazeDataset
from model.tokenizers.maze import MazeTokenizer, OutputTokens


from model.hrm import HRM, HRMConfig


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
        .type(torch.uint32)
        .sum(dim=-1)
        .type(torch.float32)
    )

    correct_route_tokens = (
        ((pred_tokens == y_true) & (y_true == OutputTokens.ROUTE))
        .type(torch.uint32)
        .sum(dim=-1)
        .type(torch.float32)
    )

    accuracies = correct_route_tokens / total_route_tokens

    # print("true route tokens: ", true_route_tokens[0])
    # print("total route tokens: ", total_route_tokens[0])
    # print("correct route tokens: ", correct_route_tokens[0])
    # print(accuracies)

    return accuracies


def train_loop(
    hrm: HRM,
    tokenizer: MazeTokenizer,
    train_samples: MazeDataset,
    test_samples: MazeDataset,
    M: int = 4,  # NOTE(Nic): number of supervision steps
    epochs: int = 1000,
    train_split: float = 0.75,
):

    train_config = TrainConfig(lr=1e-4, batch_size=128, epochs=100)

    opt = optim.AdamW(hrm.parameters(), lr=1e-4)

    train_split_subset, val_split_subset = random_split(
        train_samples, [train_split, 1 - train_split]
    )

    train_loader = DataLoader(train_split_subset, batch_size=train_config.batch_size)
    val_loader = DataLoader(val_split_subset, batch_size=train_config.batch_size)
    test_loader = DataLoader(test_samples, batch_size=train_config.batch_size)
    train_loader_length = len(train_loader)
    num_train_samples = train_loader_length * train_config.batch_size
    num_val_samples = len(val_loader) * train_config.batch_size
    num_test_samples = len(test_loader) * train_config.batch_size

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
            "maze-dims": (9, 9),
            "train-samples": num_train_samples,
            "val-samples": num_val_samples,
            "test-samples": num_test_samples,
            "epochs": 100,
        },
    )

    print(
        f"train samples: ~{num_train_samples:,} | val samples: ~{num_val_samples:,} | test samples: ~{num_test_samples:,}"
    )

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    supervision_losses: list[float] = []
    supervision_accuracies: list[float] = []
    zH_deltas: list[float] = []
    zL_deltas: list[float] = []

    for epoch in range(train_config.epochs):

        train_losses.clear()
        train_accuracies.clear()
        hrm.train()

        num_batches = len(train_loader)
        vis_every = math.floor(num_batches * 0.05)

        for i, (x_img, y_img) in enumerate(train_loader):
            # NOTE(Nic): We use the sample_index as the global step for indexing our metrics
            # NOTE(Nic): We report batch statistics for the last segment in each train batch, but only validation averages.
            sample_index = epoch * train_loader_length + i
            x_seq, y_seq = tokenizer(x_img, y_img)

            z = hrm.initial_state(x_seq)

            supervision_losses.clear()
            supervision_accuracies.clear()

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

                supervision_losses.append(loss.item())
                supervision_accuracies.append(acc.item())
                loss_list = ", ".join(
                    list(
                        map(
                            lambda x: f"{x[1]:.4f} ({supervision_accuracies[x[0]] * 100:.1f}%)",
                            enumerate(supervision_losses),
                        )
                    )
                )
                loss_bg = " " * (14 * M - 2 - len(loss_list))

                print(
                    f"[trn] epoch: {epoch}/{epochs} | iter: ({i}/{num_batches})[{m}/{M}] | norm: {grad_norm:.3f} | segs: {loss_list + loss_bg}"
                    + " " * 10,
                    end="\r",
                )

            log_data = {
                "epoch": epoch,
                "sample_step": sample_index,
                "train/loss": supervision_losses[-1],
                "train/acc": supervision_accuracies[-1],
                "train/grad-norm": grad_norm,
            }

            # NOTE(Nic): record all the segment deltas so we can see
            # how the indiviudal segments are converging.
            for i in range(1, M):
                log_data[f"train/segment-{i + 1}-loss-delta"] = (
                    supervision_losses[i] - supervision_losses[0]
                )
                log_data[f"train/segment-{i + 1}-acc-delta"] = (
                    supervision_accuracies[i] - supervision_accuracies[0]
                )

            wandb_run.log(
                data=log_data,
                step=sample_index,
                # NOTE(Nic): if we're on the last train sample, don't commit yet, cause we have val data to add later.
                commit=sample_index < train_loader_length - 1,
            )

            train_losses.extend(supervision_losses)
            train_accuracies.extend(supervision_accuracies)

            if (i + 1) % vis_every == 0:

                num_samples = 5
                y_pred_indices = torch.argmax(y_pred, dim=-1)
                y_pred_imgs = tokenizer.untokenize(y_pred_indices, y_seq)
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

        avg_trn_loss = (
            sum(train_losses) / len(train_losses) if len(train_losses) > 0 else 0
        )
        avg_trn_acc = (
            sum(train_accuracies) / len(train_accuracies)
            if len(train_accuracies) > 0
            else 0
        )
        print(
            f"\n[trn] epoch: {epoch}/{epochs} | avg loss: {avg_trn_loss:.4f} | avg acc: {avg_trn_acc*100:.1f}%"
        )

        val_losses.clear()
        val_accuracies.clear()
        hrm.eval()

        with torch.no_grad():
            for i, (x_val_img, y_val_img) in enumerate(val_loader):
                x_val_seq, y_val_seq = tokenizer(x_val_img, y_val_img)
                y_val_pred = hrm.forward(x_val_seq, M=M)  # NOTE(Nic): does M segments
                val_loss = F.cross_entropy(
                    y_val_pred.reshape(-1, len(OutputTokens)),
                    y_val_seq.reshape(-1),
                    # ignore_index=OutputTokens.IGNORE,
                )
                val_acc = accuracies(y_val_pred, y_val_seq).mean()
                val_accuracies.append(val_acc.item())
                val_losses.append(val_loss.item())

                print(
                    f"[val] epoch: {epoch}/{epochs} | iter: {i}/{len(val_loader)} | final seg: {val_loss.item():.4f} ({val_acc.item()*100:.1f}%)"
                    + " " * 10,
                    end="\r",
                )

                if (i + 1) % vis_every == 0:
                    num_samples = 5
                    y_val_pred_indices = torch.argmax(y_val_pred, dim=-1)
                    y_val_pred_imgs = tokenizer.untokenize(
                        y_val_pred_indices, y_val_seq
                    )
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

            avg_val_loss = (
                sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0
            )
            avg_val_acc = (
                sum(val_accuracies) / len(val_accuracies)
                if len(val_accuracies) > 0
                else 0
            )
            print(
                f"\n[val] epoch: {epoch}/{epochs} | avg loss: {avg_val_loss:.4f} | avg acc: {avg_val_acc*100:.1f}%"
            )

            wandb_run.log(
                data={
                    "val/loss": avg_val_loss,
                    "val/acc": avg_val_acc,
                },
                step=sample_index,
                # NOTE(Nic): if we're on the last train sample, don't commit yet, cause we have val data to add later.
                commit=True,
            )

    wandb_run.finish()
