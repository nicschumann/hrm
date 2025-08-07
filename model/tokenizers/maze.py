import torch
import torch.nn as nn
from enum import IntEnum, auto


class InputTokens(IntEnum):
    PAD = 0
    WALL = auto()
    PATH = auto()
    SOURCE_POS = auto()
    TARGET_POS = auto()
    NEW_LINE = auto()


class OutputTokens(IntEnum):
    PAD = 0
    IGNORE = auto()
    ROUTE = auto()
    NEW_LINE = auto()


class MazeTokenizer(nn.Module):

    # This function takes an output sequence of tokens produced by the model
    # and converts it back into a 2D image that can be viewed as a maze.
    def untokenize(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        grid_size=(9, 9),
    ):
        assert (
            tokens.dim() == 2
        ), f"expecting a batched token sequence B, S, got a tensor with dim={tokens.dim()}"
        B, S = tokens.shape
        W, H = grid_size

        # NOTE(Nic): We need to account for 1px of padding all around the maze
        # So a 9x9 maze is represented as an 11x11 image, due to the border.
        # We represent new lines with explicit newline tokens in tokenization,
        # so we need to reshape into W,H = 9 + 2, 9 + 3.
        tokens_2d = tokens.reshape(B, W + 2, H + 3)
        targets_2d = targets.reshape(B, W + 2, H + 3)
        # Now that we have newlines structurally, we can drop the last column.
        tokens_2d = tokens_2d[:, :, :-1]
        targets_2d = targets_2d[:, :, :-1]

        images = torch.zeros(
            3, B, W + 2, H + 2
        )  # (C, B, W, H), channels before batch for ease of indexing...

        true_mask = torch.where(targets_2d == OutputTokens.ROUTE)
        images[0][true_mask] = 0.65  # set the true route to white
        images[1][true_mask] = 0.65  # set the true route to white
        images[2][true_mask] = 0.65  # set the true route to white

        true_pos_mask = torch.where(
            (tokens_2d == targets_2d) & (tokens_2d == OutputTokens.ROUTE)
        )
        images[0][true_pos_mask] = 0.0
        images[1][true_pos_mask] = 1.0
        images[2][true_pos_mask] = 1.0  # set correct preditions to green

        false_pos_mask = torch.where(
            (tokens_2d != targets_2d) & (tokens_2d == OutputTokens.ROUTE)
        )
        images[0][false_pos_mask] = 1.0
        images[1][false_pos_mask] = 0.5
        images[2][false_pos_mask] = 0.0  # set incorrect preditions to red

        images = images.permute(1, 0, 2, 3)  # B, C, W, H

        return images

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        assert x.dim() == 4, f"expecting B, C, W, H, got a tensor with dim={x.dim()}"
        B, C, W, H = x.shape
        assert C == 3, f"expecting 3-channel maze images, but got ({B}, {C}, {W}, {H})"

        input_grid = torch.full((B, W, H + 1), fill_value=InputTokens.NEW_LINE)
        wall_mask = torch.where(
            (x[:, 0, :, :] == 0) & (x[:, 1, :, :] == 0) & (x[:, 2, :, :] == 0)
        )
        path_mask = torch.where(
            (x[:, 0, :, :] == 1) & (x[:, 1, :, :] == 1) & (x[:, 2, :, :] == 1)
        )
        start_mask = torch.where(
            (x[:, 0, :, :] == 1) & (x[:, 1, :, :] == 0) & (x[:, 2, :, :] == 0)
        )
        end_mask = torch.where(
            (x[:, 0, :, :] == 0) & (x[:, 1, :, :] == 1) & (x[:, 2, :, :] == 0)
        )

        input_grid[wall_mask] = InputTokens.WALL
        input_grid[path_mask] = InputTokens.PATH
        input_grid[start_mask] = InputTokens.SOURCE_POS
        input_grid[end_mask] = InputTokens.TARGET_POS

        if y is not None:
            assert y.dim() == 3
            B_out, W_out, H_out = y.shape
            assert B_out == B and W_out == W and H_out == H

            output_grid = torch.full_like(input_grid, fill_value=OutputTokens.NEW_LINE)
            ignore_mask = torch.where(y == 0)
            route_mask = torch.where(y == 1)
            output_grid[ignore_mask] = OutputTokens.IGNORE
            output_grid[route_mask] = OutputTokens.ROUTE

            output_grid = output_grid.reshape(B, -1)
        else:
            output_grid = None

        return input_grid.reshape(B, -1), output_grid
