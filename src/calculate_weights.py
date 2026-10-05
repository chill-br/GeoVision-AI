from pathlib import Path
from collections import Counter

import tifffile
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "train.txt"


class_names = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Agriculture",
    4: "Bare land",
}


# Read training filenames
with open(SPLIT_FILE, "r") as f:
    files = [
        line.strip()
        for line in f
        if line.strip()
    ]


counts = Counter()


for filename in files:

    path = DATA_DIR / filename

    data = tifffile.imread(path)

    label = data[:, :, 6]

    # Ignore NaN pixels
    valid = label[np.isfinite(label)]

    values, pixel_counts = np.unique(
        valid,
        return_counts=True
    )

    for value, count in zip(values, pixel_counts):

        class_id = int(value)

        if class_id in class_names:
            counts[class_id] += int(count)


print("================================")
print("Training Class Distribution")
print("================================")

total = sum(counts.values())


for class_id in range(5):

    count = counts[class_id]

    percentage = (
        100 * count / total
    )

    print(
        f"{class_id}: "
        f"{class_names[class_id]:12s} "
        f"{count:12,d} "
        f"({percentage:6.2f}%)"
    )


# --------------------------------
# Calculate inverse-frequency weights
# --------------------------------

print("\n================================")
print("Class Weights")
print("================================")

frequencies = np.array(
    [
        counts[i] / total
        for i in range(5)
    ],
    dtype=np.float32
)


# Median-frequency balancing
median_frequency = np.median(
    frequencies
)


weights = (
    median_frequency / frequencies
)


# Normalize so average weight = 1
weights = weights / weights.mean()


for class_id, weight in enumerate(weights):

    print(
        f"{class_id}: "
        f"{class_names[class_id]:12s} "
        f"{weight:.4f}"
    )


print("\nWeights as PyTorch tensor:")

print(
    "torch.tensor(["
    + ", ".join(
        f"{w:.4f}"
        for w in weights
    )
    + "])"
)