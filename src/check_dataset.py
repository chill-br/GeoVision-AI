from pathlib import Path

import tifffile
import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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


CLASS_NAMES = [
    "Vegetation",
    "Built-up",
    "Water",
    "Agriculture",
    "Bare land"
]

NUM_CLASSES = 5
IGNORE_INDEX = 255


for split in [
    "train",
    "val",
    "test"
]:

    split_file = (
        SPLIT_DIR /
        f"{split}.txt"
    )

    if not split_file.exists():

        print(
            f"Missing split file: "
            f"{split_file}"
        )

        continue


    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as f:

        files = [
            line.strip()
            for line in f
            if line.strip()
        ]


    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    ignore = 0

    all_water = 0


    for filename in files:

        path = (
            DATA_DIR /
            filename
        )

        data = tifffile.imread(
            path
        )

        if data.shape != (
            256,
            256,
            7
        ):

            raise RuntimeError(
                f"Unexpected shape "
                f"for {filename}: "
                f"{data.shape}"
            )


        label = np.nan_to_num(
            data[:, :, 6],
            nan=IGNORE_INDEX
        ).astype(
            np.int64
        )


        unique, class_counts = (
            np.unique(
                label,
                return_counts=True
            )
        )


        for value, count in zip(
            unique,
            class_counts
        ):

            if value == IGNORE_INDEX:

                ignore += int(
                    count
                )

            elif (
                0 <= value < NUM_CLASSES
            ):

                counts[value] += int(
                    count
                )


        valid = (
            label != IGNORE_INDEX
        )

        if np.any(valid):

            all_water += int(
                np.all(
                    label[valid] == 2
                )
            )


    valid_pixels = int(
        counts.sum()
    )


    print()
    print(
        "================================"
    )

    print(
        f"{split.upper()} DATASET"
    )

    print(
        "================================"
    )

    print(
        "Tiles:",
        len(files)
    )

    print(
        "All-water tiles:",
        all_water
    )

    print()
    print(
        "Class distribution:"
    )


    for class_id, name in enumerate(
        CLASS_NAMES
    ):

        percentage = (
            100.0 *
            counts[class_id]
            /
            valid_pixels
            if valid_pixels > 0
            else 0.0
        )

        print(
            f"{class_id} - "
            f"{name:<12}: "
            f"{counts[class_id]:,} "
            f"({percentage:.2f}%)"
        )


    print()
    print(
        "Ignore pixels:",
        f"{ignore:,}"
    )

    print(
        "Valid pixels:",
        f"{valid_pixels:,}"
    )