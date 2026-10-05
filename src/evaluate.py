from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GeoVisionDataset
from unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
BARE_LAND_CLASS = 4
IGNORE_INDEX = 255

BATCH_SIZE = 4

MODEL_FILENAME = (
    "unet_next_bareland_best.pth"
)

# Bare-land thresholds to test.
# 0.00 = very aggressive Bare-land prediction
# 1.00 = very conservative Bare-land prediction
BARE_LAND_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    MODEL_FILENAME
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Agriculture",
    4: "Bare land",
}


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    predictions,
    targets,
    num_classes=5,
    ignore_index=255
):

    mask = (
        targets != ignore_index
    )

    predictions = predictions[mask]
    targets = targets[mask]

    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
        device=targets.device
    )

    if targets.numel() == 0:
        return confusion

    indices = (
        targets * num_classes
        + predictions
    )

    confusion += torch.bincount(
        indices,
        minlength=num_classes ** 2
    ).reshape(
        num_classes,
        num_classes
    )

    return confusion


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    confusion
):

    intersection = (
        torch.diag(confusion)
        .float()
    )

    actual = (
        confusion.sum(dim=1)
        .float()
    )

    predicted = (
        confusion.sum(dim=0)
        .float()
    )

    union = (
        actual
        +
        predicted
        -
        intersection
    )

    iou = (
        intersection /
        (union + 1e-7)
    )

    precision = (
        intersection /
        (predicted + 1e-7)
    )

    recall = (
        intersection /
        (actual + 1e-7)
    )

    f1 = (
        2.0 *
        precision *
        recall
        /
        (
            precision
            +
            recall
            +
            1e-7
        )
    )

    valid_classes = (
        union > 0
    )

    mean_iou = (
        iou[valid_classes]
        .mean()
        .item()
    )

    pixel_accuracy = (
        intersection.sum()
        /
        (actual.sum() + 1e-7)
    ).item()

    macro_precision = (
        precision[valid_classes]
        .mean()
        .item()
    )

    macro_recall = (
        recall[valid_classes]
        .mean()
        .item()
    )

    macro_f1 = (
        f1[valid_classes]
        .mean()
        .item()
    )

    return (
        iou,
        precision,
        recall,
        f1,
        mean_iou,
        pixel_accuracy,
        macro_precision,
        macro_recall,
        macro_f1
    )


# ============================================================
# BARE-LAND THRESHOLD PREDICTION
# ============================================================

def threshold_predictions(
    outputs,
    threshold
):
    """
    Convert model logits into predictions.

    Normal behavior:
        prediction = argmax(probabilities)

    Bare-land threshold behavior:
        If Bare-land probability >= threshold,
        predict Bare land.

        Otherwise choose the best non-Bare-land class.
    """

    probabilities = F.softmax(
        outputs,
        dim=1
    )

    bare_probability = (
        probabilities[:, BARE_LAND_CLASS]
    )

    non_bare_probabilities = (
        probabilities.clone()
    )

    non_bare_probabilities[
        :,
        BARE_LAND_CLASS
    ] = -1.0

    best_non_bare = (
        non_bare_probabilities.argmax(
            dim=1
        )
    )

    predictions = best_non_bare.clone()

    predictions[
        bare_probability >= threshold
    ] = BARE_LAND_CLASS

    return predictions


# ============================================================
# HEADER
# ============================================================

print("================================")
print("GeoVision AI Model Evaluation")
print("================================")

print(
    "Device:",
    device
)

print(
    "Model:",
    MODEL_PATH
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n"
        f"{MODEL_PATH}\n\n"
        "Run train.py first."
    )


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

print(
    f"test dataset: {len(test_dataset)} tiles"
)


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

model.load_state_dict(
    checkpoint
)

model = model.to(device)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

# Normal argmax evaluation
baseline_confusion = torch.zeros(
    NUM_CLASSES,
    NUM_CLASSES,
    dtype=torch.long,
    device=device
)

# One confusion matrix per threshold
threshold_confusions = {
    threshold: torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
        device=device
    )
    for threshold in BARE_LAND_THRESHOLDS
}


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Baseline: standard argmax
        # ----------------------------------------------------

        baseline_predictions = (
            outputs.argmax(
                dim=1
            )
        )

        baseline_confusion += (
            calculate_confusion_matrix(
                baseline_predictions,
                labels,
                NUM_CLASSES,
                IGNORE_INDEX
            )
        )

        # ----------------------------------------------------
        # Bare-land threshold sweep
        # ----------------------------------------------------

        for threshold in BARE_LAND_THRESHOLDS:

            predictions = threshold_predictions(
                outputs,
                threshold
            )

            threshold_confusions[
                threshold
            ] += calculate_confusion_matrix(
                predictions,
                labels,
                NUM_CLASSES,
                IGNORE_INDEX
            )


# ============================================================
# BASELINE METRICS
# ============================================================

(
    ious,
    precisions,
    recalls,
    f1,
    mean_iou,
    pixel_accuracy,
    macro_precision,
    macro_recall,
    macro_f1
) = calculate_metrics(
    baseline_confusion
)


# ============================================================
# BASELINE CONFUSION MATRIX
# ============================================================

print()

print("================================")
print("Baseline Confusion Matrix")
print("================================")

print(
    "Rows = Ground Truth"
)

print(
    "Columns = Prediction"
)

print()

print(
    baseline_confusion
    .detach()
    .cpu()
)


# ============================================================
# BASELINE RESULTS
# ============================================================

print()

print("================================")
print("Baseline Segmentation Results")
print("================================")

print(
    f"Pixel Accuracy:  "
    f"{pixel_accuracy:.4f}"
)

print(
    f"Mean IoU:         "
    f"{mean_iou:.4f}"
)

print(
    f"Macro Precision:  "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall:     "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1:         "
    f"{macro_f1:.4f}"
)


# ============================================================
# BASELINE PER-CLASS
# ============================================================

print()

print("================================")
print("Baseline Per-Class Metrics")
print("================================")

print(
    f"{'Class':<15}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'IoU':>12}"
    f"{'F1':>12}"
)

print("-" * 63)

for class_id in range(
    NUM_CLASSES
):

    print(
        f"{CLASS_NAMES[class_id]:<15}"
        f"{precisions[class_id].item():>12.4f}"
        f"{recalls[class_id].item():>12.4f}"
        f"{ious[class_id].item():>12.4f}"
        f"{f1[class_id].item():>12.4f}"
    )


# ============================================================
# BARE LAND BASELINE
# ============================================================

print()

print("================================")
print("Bare-land Baseline")
print("================================")

print(
    f"Bare-land Precision: "
    f"{precisions[4].item():.4f}"
)

print(
    f"Bare-land Recall:    "
    f"{recalls[4].item():.4f}"
)

print(
    f"Bare-land IoU:       "
    f"{ious[4].item():.4f}"
)

print(
    f"Bare-land F1:        "
    f"{f1[4].item():.4f}"
)


# ============================================================
# THRESHOLD SWEEP
# ============================================================

print()

print("================================")
print("Bare-land Threshold Sweep")
print("================================")

print(
    f"{'Threshold':>10}"
    f"{'mIoU':>10}"
    f"{'Accuracy':>12}"
    f"{'BL Precision':>15}"
    f"{'BL Recall':>12}"
    f"{'BL IoU':>12}"
    f"{'BL F1':>12}"
)

print("-" * 93)

threshold_results = []

for threshold in BARE_LAND_THRESHOLDS:

    (
        threshold_ious,
        threshold_precisions,
        threshold_recalls,
        threshold_f1,
        threshold_miou,
        threshold_accuracy,
        _,
        _,
        _
    ) = calculate_metrics(
        threshold_confusions[threshold]
    )

    bare_precision = (
        threshold_precisions[
            BARE_LAND_CLASS
        ].item()
    )

    bare_recall = (
        threshold_recalls[
            BARE_LAND_CLASS
        ].item()
    )

    bare_iou = (
        threshold_ious[
            BARE_LAND_CLASS
        ].item()
    )

    bare_f1 = (
        threshold_f1[
            BARE_LAND_CLASS
        ].item()
    )

    threshold_results.append({
        "threshold": threshold,
        "miou": threshold_miou,
        "accuracy": threshold_accuracy,
        "precision": bare_precision,
        "recall": bare_recall,
        "iou": bare_iou,
        "f1": bare_f1,
    })

    print(
        f"{threshold:>10.2f}"
        f"{threshold_miou:>10.4f}"
        f"{threshold_accuracy:>12.4f}"
        f"{bare_precision:>15.4f}"
        f"{bare_recall:>12.4f}"
        f"{bare_iou:>12.4f}"
        f"{bare_f1:>12.4f}"
    )


# ============================================================
# BEST THRESHOLDS BY DIFFERENT METRICS
# ============================================================

best_bare_iou = max(
    threshold_results,
    key=lambda x: x["iou"]
)

best_bare_f1 = max(
    threshold_results,
    key=lambda x: x["f1"]
)

best_miou = max(
    threshold_results,
    key=lambda x: x["miou"]
)


# ============================================================
# THRESHOLD SUMMARY
# ============================================================

print()

print("================================")
print("Threshold Summary")
print("================================")

print(
    "Best Bare-land IoU:"
)

print(
    f"  Threshold: "
    f"{best_bare_iou['threshold']:.2f}"
)

print(
    f"  Bare-land IoU: "
    f"{best_bare_iou['iou']:.4f}"
)

print(
    f"  Bare-land Precision: "
    f"{best_bare_iou['precision']:.4f}"
)

print(
    f"  Bare-land Recall: "
    f"{best_bare_iou['recall']:.4f}"
)

print(
    f"  Bare-land F1: "
    f"{best_bare_iou['f1']:.4f}"
)

print()

print(
    "Best Bare-land F1:"
)

print(
    f"  Threshold: "
    f"{best_bare_f1['threshold']:.2f}"
)

print(
    f"  Bare-land F1: "
    f"{best_bare_f1['f1']:.4f}"
)

print(
    f"  Bare-land IoU: "
    f"{best_bare_f1['iou']:.4f}"
)

print(
    f"  Bare-land Precision: "
    f"{best_bare_f1['precision']:.4f}"
)

print(
    f"  Bare-land Recall: "
    f"{best_bare_f1['recall']:.4f}"
)

print()

print(
    "Best Overall mIoU:"
)

print(
    f"  Threshold: "
    f"{best_miou['threshold']:.2f}"
)

print(
    f"  mIoU: "
    f"{best_miou['miou']:.4f}"
)

print(
    f"  Accuracy: "
    f"{best_miou['accuracy']:.4f}"
)


# ============================================================
# COMPLETE
# ============================================================

print()

print("================================")
print("Evaluation complete!")
print("================================")