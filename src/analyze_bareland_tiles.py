from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GeoVisionDataset
from unet import UNet


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 5
BARE_LAND = 4
IGNORE_INDEX = 255

BATCH_SIZE = 4
THRESHOLD = 0.55

MODEL_FILENAME = "unet_next_bareland_best.pth"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# METRICS
# ============================================================

def bareland_metrics(prediction, target):

    valid = target != IGNORE_INDEX

    prediction = prediction[valid]
    target = target[valid]

    gt = target == BARE_LAND
    pred = prediction == BARE_LAND

    tp = (gt & pred).sum().item()
    fp = (~gt & pred).sum().item()
    fn = (gt & ~pred).sum().item()

    gt_pixels = gt.sum().item()
    pred_pixels = pred.sum().item()
    total_pixels = valid.sum().item()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    gt_percent = (
        100.0 * gt_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    pred_percent = (
        100.0 * pred_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "gt_percent": gt_percent,
        "pred_percent": pred_percent,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# HEADER
# ============================================================

print("================================")
print("GeoVision AI Bare-land Analysis")
print("================================")

print("Device:", device)
print("Model:", MODEL_PATH)
print("Threshold:", THRESHOLD)


# ============================================================
# MODEL
# ============================================================

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# DATASET
# ============================================================

dataset = GeoVisionDataset(
    split="test"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Test tiles:", len(dataset))


# ============================================================
# ANALYSIS
# ============================================================

results = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        )

        # Apply bare-land threshold
        bare_probability = probabilities[
            :,
            BARE_LAND
        ]

        prediction = torch.where(
            bare_probability >= THRESHOLD,
            torch.full_like(
                prediction,
                BARE_LAND
            ),
            prediction
        )

        for i in range(images.shape[0]):

            metrics = bareland_metrics(
                prediction[i],
                labels[i]
            )

            tile_index = (
                batch_index * BATCH_SIZE + i
            )

            metrics["tile"] = tile_index

            results.append(metrics)


# ============================================================
# SORT BY IoU
# ============================================================

results_sorted = sorted(
    results,
    key=lambda x: x["iou"]
)


# ============================================================
# TABLE
# ============================================================

print()
print("================================")
print("Per-Tile Bare-land Results")
print("================================")

print(
    f"{'Tile':>6} "
    f"{'GT %':>8} "
    f"{'Pred %':>8} "
    f"{'Prec':>8} "
    f"{'Recall':>8} "
    f"{'IoU':>8} "
    f"{'F1':>8}"
)

print("-" * 62)

for r in results_sorted:

    print(
        f"{r['tile']:>6} "
        f"{r['gt_percent']:>8.2f} "
        f"{r['pred_percent']:>8.2f} "
        f"{r['precision']:>8.4f} "
        f"{r['recall']:>8.4f} "
        f"{r['iou']:>8.4f} "
        f"{r['f1']:>8.4f}"
    )


# ============================================================
# SUMMARY
# ============================================================

ious = np.array([
    r["iou"]
    for r in results
])

precisions = np.array([
    r["precision"]
    for r in results
])

recalls = np.array([
    r["recall"]
    for r in results
])

f1s = np.array([
    r["f1"]
    for r in results
])


print()
print("================================")
print("Multi-Tile Summary")
print("================================")

print(
    f"Mean IoU:     {ious.mean():.4f}"
)

print(
    f"Median IoU:   {np.median(ious):.4f}"
)

print(
    f"Mean Precision: {precisions.mean():.4f}"
)

print(
    f"Mean Recall:    {recalls.mean():.4f}"
)

print(
    f"Mean F1:        {f1s.mean():.4f}"
)


# ============================================================
# WORST / BEST
# ============================================================

print()
print("================================")
print("Worst 10 Tiles")
print("================================")

for r in results_sorted[:10]:

    print(
        f"Tile {r['tile']:3d} | "
        f"GT {r['gt_percent']:6.2f}% | "
        f"Pred {r['pred_percent']:6.2f}% | "
        f"IoU {r['iou']:.4f} | "
        f"Recall {r['recall']:.4f}"
    )


print()
print("================================")
print("Best 10 Tiles")
print("================================")

for r in results_sorted[-10:][::-1]:

    print(
        f"Tile {r['tile']:3d} | "
        f"GT {r['gt_percent']:6.2f}% | "
        f"Pred {r['pred_percent']:6.2f}% | "
        f"IoU {r['iou']:.4f} | "
        f"Recall {r['recall']:.4f}"
    )


print()
print("================================")
print("Analysis complete!")
print("================================")