import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from data.maze import MazeDataset
from model.tokenizers.maze import MazeTokenizer, OutputTokens


from model.hrm import HRM


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
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    supervision_losses: list[float] = []
    supervision_accuracies: list[float] = []

    for epoch in range(epochs):

        train_losses.clear()
        train_accuracies.clear()
        hrm.train()

        num_batches = len(train_loader)
        vis_every = math.floor(num_batches * 0.05)

        for i, (x_img, y_img) in enumerate(train_loader):
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
