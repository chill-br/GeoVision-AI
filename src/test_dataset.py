from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import GeoVisionDataset


print("================================")
print("GeoVision AI Dataset Test")
print("================================")


# Create training dataset
dataset = GeoVisionDataset(
    split="train"
)


print("\nDataset length:")
print(len(dataset))


# Load first sample
image, label = dataset[0]


print("\nFirst sample:")
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

print("Label shape:", label.shape)
print("Label dtype:", label.dtype)


print("\nImage statistics:")
print("Minimum:", image.min().item())
print("Maximum:", image.max().item())
print("Mean:", image.mean().item())


print("\nLabel values:")

unique_labels = torch.unique(label)

print(unique_labels)


# DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


images, labels = next(iter(loader))


print("\n================================")
print("DataLoader Test")
print("================================")

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

print("Batch image dtype:", images.dtype)
print("Batch label dtype:", labels.dtype)


print("\nDataset test complete!")