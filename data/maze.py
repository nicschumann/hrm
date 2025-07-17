"""easy_to_hard_data.py
Python package with datasets for studying generalization from
    easy training data to hard test examples.
Developed as part of easy-to-hard (github.com/aks2203/easy-to-hard).
Avi Schwarzschild
June 2021
"""

import os
import os.path
from typing import Optional, Callable

import numpy as np
import torch

from data.utils import download_url, extract_zip


class MazeDataset(torch.utils.data.Dataset):
    """This is a dataset class for mazes.
    padding and cropping is done correctly within this class for small and large mazes.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        size: int = 9,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):

        self.root = root
        self.train = train
        self.size = size
        self.transform = transform

        self.folder_name = f"maze_data_{'train' if self.train else 'test'}_{size}"
        url = (
            f"https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/"
            f"{self.folder_name}.tar.gz"
        )

        if download:
            self.download(url)

        print(f"Loading mazes of size {size} x {size}.")

        inputs_path = os.path.join(root, self.folder_name, "inputs.npy")
        solutions_path = os.path.join(root, self.folder_name, "solutions.npy")
        inputs_np = np.load(inputs_path)
        targets_np = np.load(solutions_path)

        self.inputs = torch.from_numpy(inputs_np).float()
        self.targets = torch.from_numpy(targets_np).long()

    def __getitem__(self, index):
        img, target = self.inputs[index], self.targets[index]

        if self.transform is not None:
            stacked = torch.cat([img, target.unsqueeze(0)], dim=0)
            stacked = self.transform(stacked)
            img = stacked[:3].float()
            target = stacked[3].long()

        return img, target

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.folder_name)
        if not os.path.exists(fpath):
            return False
        return True

    def download(self, url) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        path = download_url(url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)


if __name__ == "__main__":
    d = MazeDataset("./cache")
