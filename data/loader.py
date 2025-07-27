from typing import Tuple

import torch
from torch.utils.data import DataLoader, default_collate

from model.tokenizers.maze import MazeTokenizer
from data.maze import MazeDataset


def tokenizing_collate_fn(batch):
    xs, ys = default_collate(batch)
    print(xs.shape, ys.shape)

    #
    return batch


def get_tokenizing_dataloader(
    batch_size: int = 32, train: bool = True, shuffle: bool = True
) -> Tuple[DataLoader, MazeDataset]:
    data = MazeDataset("./cache", train=train)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader, data
