from pathlib import Path
import csv

import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader

from dataset import GeoVisionDataset
from unet import UNet


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 5
BARE_LAND_CLASS = 4
BARE_LAND_THRESHOLD = 0.55
BATCH_SIZE = 4

MODEL_FILENAME = "unet_next_bareland_best.pth"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("================================")
print("GeoVision AI Test-Tile Analysis")
print("================================")

print("Device:", device)
print("Model:", MODEL_PATH)
print("Bare-land threshold:", BARE_LAND_THRESHOLD)


# ============================================================
# DATASET
# ============================================================

test_dataset = GeoVisionDataset(
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Test tiles:", len(test_dataset))


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
# ANALYSIS
# ============================================================

results = []

total_tp = 0
total_fn = 0
total_fp = 0
total_tn = 0


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader
    ):

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

        # ----------------------------------------------------
        # Bare-land threshold
        # ----------------------------------------------------

        bare_probability = probabilities[
            :,
            BARE_LAND_CLASS
        ]

        prediction = torch.where(
            bare_probability >= BARE_LAND_THRESHOLD,
            torch.full_like(
                prediction,
                BARE_LAND_CLASS
            ),
            prediction
        )

        # ============================================================
        # THRESHOLD DIAGNOSTIC
        # ============================================================

        argmax_prediction = torch.argmax(
            probabilities,
        dim=1
        )

        threshold_prediction = torch.where(
            bare_probability >= BARE_LAND_THRESHOLD,
            torch.full_like(
            argmax_prediction,
        BARE_LAND_CLASS
        ),
        argmax_prediction
        )

        changed = (
            threshold_prediction != argmax_prediction
        )

        print()
        print("================================")
        print("Threshold Diagnostic")
        print("================================")

        print("Pixels changed by threshold:", changed.sum().item())

        print("Total pixels:", changed.numel())

        print(
            "Changed percentage:",
            f"{100.0 * changed.float().mean().item():.4f}%"
            )

        # ----------------------------------------------------
        # Process every tile separately
        # ----------------------------------------------------

        for i in range(images.shape[0]):

            pred = prediction[i]
            target = labels[i]

            # Ignore 255
            valid = target != 255

            pred = pred[valid]
            target = target[valid]

            actual_bare = (
                target == BARE_LAND_CLASS
            )

            predicted_bare = (
                pred == BARE_LAND_CLASS
            )

            tp = (
                actual_bare &
                predicted_bare
            ).sum().item()

            fn = (
                actual_bare &
                ~predicted_bare
            ).sum().item()

            fp = (
                ~actual_bare &
                predicted_bare
            ).sum().item()

            tn = (
                ~actual_bare &
                ~predicted_bare
            ).sum().item()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            precision = (
                tp /
                (tp + fp)
                if tp + fp > 0
                else 0.0
            )

            recall = (
                tp /
                (tp + fn)
                if tp + fn > 0
                else 0.0
            )

            iou = (
                tp /
                (tp + fp + fn)
                if tp + fp + fn > 0
                else 0.0
            )

            f1 = (
                2 * tp /
                (2 * tp + fp + fn)
                if 2 * tp + fp + fn > 0
                else 0.0
            )

            pixel_accuracy = (
                (pred == target).sum().item()
                /
                len(target)
                if len(target) > 0
                else 0.0
            )

            results.append({
                "tile_index": batch_index * BATCH_SIZE + i,
                "bare_gt_pixels": int(actual_bare.sum().item()),
                "predicted_bare_pixels": int(
                    predicted_bare.sum().item()
                ),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "f1": f1,
                "pixel_accuracy": pixel_accuracy
            })

            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_tn += tn


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR /
    "test_tile_bareland_analysis.csv"
)

fieldnames = list(
    results[0].keys()
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# SORT RESULTS
# ============================================================

# Only consider tiles containing meaningful bare-land area
tiles_with_bare_land = [
    r for r in results
    if r["bare_gt_pixels"] >= 100
]

worst = sorted(
    tiles_with_bare_land,
    key=lambda x: x["iou"]
)

best = sorted(
    tiles_with_bare_land,
    key=lambda x: x["iou"],
    reverse=True
)

best = sorted(
    results,
    key=lambda x: x["iou"],
    reverse=True
)


# ============================================================
# DATASET-WIDE METRICS
# ============================================================

total_precision = (
    total_tp /
    (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0.0
)

total_recall = (
    total_tp /
    (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0.0
)

total_iou = (
    total_tp /
    (total_tp + total_fp + total_fn)
    if total_tp + total_fp + total_fn > 0
    else 0.0
)

total_f1 = (
    2 * total_tp /
    (2 * total_tp + total_fp + total_fn)
    if 2 * total_tp + total_fp + total_fn > 0
    else 0.0
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("================================")
print("Dataset-wide Bare-land Results")
print("================================")

print(
    f"Precision: {total_precision:.4f}"
)

print(
    f"Recall:    {total_recall:.4f}"
)

print(
    f"IoU:       {total_iou:.4f}"
)

print(
    f"F1:        {total_f1:.4f}"
)

print()
print("Total TP:", total_tp)
print("Total FN:", total_fn)
print("Total FP:", total_fp)
print("Total TN:", total_tn)


# ============================================================
# WORST TILES
# ============================================================

print()
print("================================")
print("Worst 10 Tiles by Bare-land IoU")
print("================================")

print(
    f"{'Tile':>8}"
    f"{'GT':>10}"
    f"{'TP':>10}"
    f"{'FN':>10}"
    f"{'FP':>10}"
    f"{'IoU':>10}"
    f"{'F1':>10}"
)

for r in worst[:10]:

    print(
        f"{r['tile_index']:>8}"
        f"{r['bare_gt_pixels']:>10}"
        f"{r['tp']:>10}"
        f"{r['fn']:>10}"
        f"{r['fp']:>10}"
        f"{r['iou']:>10.4f}"
        f"{r['f1']:>10.4f}"
    )


# ============================================================
# BEST TILES
# ============================================================

print()
print("================================")
print("Best 10 Tiles by Bare-land IoU")
print("================================")

print(
    f"{'Tile':>8}"
    f"{'GT':>10}"
    f"{'TP':>10}"
    f"{'FN':>10}"
    f"{'FP':>10}"
    f"{'IoU':>10}"
    f"{'F1':>10}"
)

for r in best[:10]:

    print(
        f"{r['tile_index']:>8}"
        f"{r['bare_gt_pixels']:>10}"
        f"{r['tp']:>10}"
        f"{r['fn']:>10}"
        f"{r['fp']:>10}"
        f"{r['iou']:>10.4f}"
        f"{r['f1']:>10.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("================================")
print("Analysis complete!")
print("================================")

print()
print("CSV saved:")
print(csv_path)