from pathlib import Path

import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "raw"
)

SPLIT_DIR = (
    PROJECT_ROOT /
    "data" /
    "splits"
)


class GeoVisionDataset(Dataset):

    def __init__(
        self,
        split="train"
    ):

        if split not in [
            "train",
            "val",
            "test"
        ]:

            raise ValueError(
                "split must be "
                "'train', 'val', or 'test'"
            )

        split_file = (
            SPLIT_DIR /
            f"{split}.txt"
        )

        if not split_file.exists():

            raise FileNotFoundError(
                f"Split file not found:\n"
                f"{split_file}"
            )

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.files = [
                line.strip()
                for line in f
                if line.strip()
            ]

        print(
            f"{split} dataset: "
            f"{len(self.files)} tiles"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(
        self,
        index
    ):

        filename = self.files[index]

        path = (
            DATA_DIR /
            filename
        )

        data = tifffile.imread(
            path
        )

        # ========================================
        # Expected:
        # (256, 256, 7)
        # ========================================

        if data.shape != (
            256,
            256,
            7
        ):

            raise RuntimeError(
                f"Unexpected shape for "
                f"{filename}: "
                f"{data.shape}"
            )

        # ========================================
        # Image bands
        # ========================================

        image = data[
            :,
            :,
            :6
        ]

        # ========================================
        # Label
        # ========================================

        label = data[
            :,
            :,
            6
        ]

        # ========================================
        # Handle invalid image values
        # ========================================

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ========================================
        # Handle invalid labels
        # ========================================

        label = np.nan_to_num(
            label,
            nan=255
        )

        label = label.astype(
            np.int64
        )

        # ========================================
        # Channels first
        # ========================================

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ========================================
        # Sentinel-2 scaling
        # ========================================

        image = (
            image.astype(
                np.float32
            )
            / 10000.0
        )

        # ========================================
        # Tensor conversion
        # ========================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label