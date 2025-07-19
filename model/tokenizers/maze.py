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

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        assert x.dim() == 4, f"expecting B, C, W, H, got a tensor with dim={x.dim()}"
        B, C, W, H = x.shape
        assert C == 3, f"expecting 3-channel maze images, but got ({B}, {C}, {W}, {H})"

        # input_seqs = torch.full((B, W, H + 1), fill_value=InputTokens.PAD)
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
