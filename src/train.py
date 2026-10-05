from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import GeoVisionDataset
from unet import UNet
from losses import CombinedLoss


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 1e-4

NUM_CLASSES = 5

IGNORE_INDEX = 255

# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

CE_LOSS_WEIGHT = 0.5

DICE_LOSS_WEIGHT = 0.5


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# 0 = Vegetation
# 1 = Built-up
# 2 = Water
# 3 = Agriculture
# 4 = Bare land
#
# Keep these stable for this experiment.
# ============================================================

CLASS_WEIGHTS = torch.tensor(
    [
        0.1714,   # Vegetation
        0.1917,   # Built-up
        0.5398,   # Water
        0.2086,   # Agriculture
        2.0       # Bare land
    ],
    dtype=torch.float32
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_MIOU_MODEL_PATH = (
    MODEL_DIR /
    "unet_next_miou_best.pth"
)

BEST_BARE_MODEL_PATH = (
    MODEL_DIR /
    "unet_next_bareland_best.pth"
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

def calculate_metrics(confusion):

    intersection = torch.diag(
        confusion
    ).float()

    ground_truth = (
        confusion.sum(dim=1)
        .float()
    )

    prediction = (
        confusion.sum(dim=0)
        .float()
    )

    union = (
        ground_truth
        +
        prediction
        -
        intersection
    )

    iou = (
        intersection /
        (union + 1e-7)
    )

    recall = (
        intersection /
        (ground_truth + 1e-7)
    )

    precision = (
        intersection /
        (prediction + 1e-7)
    )

    valid_classes = (
        union > 0
    )

    if valid_classes.any():

        mean_iou = (
            iou[valid_classes]
            .mean()
            .item()
        )

    else:

        mean_iou = 0.0

    pixel_accuracy = (
        intersection.sum()
        /
        (ground_truth.sum() + 1e-7)
    ).item()

    return (
        iou,
        precision,
        recall,
        mean_iou,
        pixel_accuracy
    )


# ============================================================
# HEADER
# ============================================================

print("================================")
print("GeoVision AI U-Net Training")
print("Stable Bare-land Experiment")
print("================================")

print(
    "Device:",
    device
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print()

print("Loss:")
print(
    f"  Weighted CE = {CE_LOSS_WEIGHT}"
)

print(
    f"  Dice        = {DICE_LOSS_WEIGHT}"
)

print()

print(
    "Bare-land CE weight:",
    CLASS_WEIGHTS[4].item()
)

print()


# ============================================================
# DATASET
# ============================================================

train_dataset = GeoVisionDataset(
    split="train"
)

val_dataset = GeoVisionDataset(
    split="val"
)

print(
    "Train tiles:",
    len(train_dataset)
)

print(
    "Validation tiles:",
    len(val_dataset)
)

print()


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

class_weights = (
    CLASS_WEIGHTS.to(device)
)

criterion = CombinedLoss(
    class_weights=class_weights,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
    ce_weight=CE_LOSS_WEIGHT,
    dice_weight=DICE_LOSS_WEIGHT
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# ============================================================
# BEST METRICS
# ============================================================

best_val_miou = -1.0

best_bare_iou = -1.0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_train_loss = 0.0

    for batch_idx, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        running_train_loss += (
            loss.item()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 20 == 0
        ):

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0.0

    confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for images, labels in val_loader:

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

            loss = criterion(
                outputs,
                labels
            )

            running_val_loss += (
                loss.item()
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            confusion += (
                calculate_confusion_matrix(
                    predictions,
                    labels,
                    NUM_CLASSES,
                    IGNORE_INDEX
                )
            )

    val_loss = (
        running_val_loss /
        len(val_loader)
    )


    # ========================================================
    # METRICS
    # ========================================================

    (
        ious,
        precisions,
        recalls,
        mean_iou,
        pixel_accuracy
    ) = calculate_metrics(
        confusion
    )

    bare_iou = (
        ious[4].item()
    )


    # ========================================================
    # CURRENT LR
    # ========================================================

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()

    print("================================")
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print("================================")

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Pixel Acc  : "
        f"{pixel_accuracy * 100:.2f}%"
    )

    print(
        f"Mean IoU   : "
        f"{mean_iou:.4f}"
    )

    print(
        f"Bare IoU   : "
        f"{bare_iou:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )

    print()

    print("Per-Class IoU:")

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{ious[class_id].item():.4f}"
        )

    print()

    print("Per-Class Recall:")

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{recalls[class_id].item():.4f}"
        )

    print()

    print("Per-Class Precision:")

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{precisions[class_id].item():.4f}"
        )


    # ========================================================
    # SAVE BEST mIoU
    # ========================================================

    if mean_iou > best_val_miou:

        best_val_miou = (
            mean_iou
        )

        torch.save(
            model.state_dict(),
            BEST_MIOU_MODEL_PATH
        )

        print()

        print(
            "✓ Best mIoU model saved!"
        )

        print(
            f"  mIoU: {mean_iou:.4f}"
        )

        print(
            f"  Bare-land IoU: "
            f"{bare_iou:.4f}"
        )


    # ========================================================
    # SAVE BEST BARE-LAND
    # ========================================================

    if bare_iou > best_bare_iou:

        best_bare_iou = (
            bare_iou
        )

        torch.save(
            model.state_dict(),
            BEST_BARE_MODEL_PATH
        )

        print()

        print(
            "✓ Best Bare-land model saved!"
        )

        print(
            f"  Bare-land IoU: "
            f"{bare_iou:.4f}"
        )

        print(
            f"  mIoU: "
            f"{mean_iou:.4f}"
        )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        mean_iou
    )

    print()


# ============================================================
# COMPLETE
# ============================================================

print("================================")
print("Training complete!")
print("================================")

print(
    f"Best validation mIoU: "
    f"{best_val_miou:.4f}"
)

print(
    f"Best validation Bare-land IoU: "
    f"{best_bare_iou:.4f}"
)

print()

print(
    "Best mIoU model:"
)

print(
    BEST_MIOU_MODEL_PATH
)

print()

print(
    "Best Bare-land IoU model:"
)

print(
    BEST_BARE_MODEL_PATH
)

print()

print("Classes:")

for class_id in range(
    NUM_CLASSES
):

    print(
        f"{class_id}: "
        f"{CLASS_NAMES[class_id]}"
    )