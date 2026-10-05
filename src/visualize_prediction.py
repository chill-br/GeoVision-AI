from pathlib import Path

import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt

from unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

BARE_LAND_CLASS = 4

BARE_LAND_THRESHOLD = 0.55

TILE_FILENAME = "GeoVision_Mumbai_tile_114.tif"

MODEL_FILENAME = "unet_next_bareland_best.pth"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME

TILE_PATH = DATA_DIR / TILE_FILENAME

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Vegetation",
    "Built-up",
    "Water",
    "Agriculture",
    "Bare land",
]


# ============================================================
# HEADER
# ============================================================

print("================================")
print("GeoVision AI Error Analysis")
print("================================")

print("Tile:", TILE_PATH)
print("Model:", MODEL_PATH)
print("Bare-land threshold:", BARE_LAND_THRESHOLD)


# ============================================================
# CHECK FILES
# ============================================================

if not TILE_PATH.exists():
    raise FileNotFoundError(
        f"Tile not found:\n{TILE_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TILE
# ============================================================

print("\nLoading tile...")

data = tifffile.imread(TILE_PATH)

print("Tile shape:", data.shape)
print("Tile dtype:", data.dtype)

if data.shape != (256, 256, 7):
    raise RuntimeError(
        f"Unexpected tile shape: {data.shape}"
    )


# ============================================================
# IMAGE / LABEL
# ============================================================

image = data[:, :, :6].astype(np.float32)

label = data[:, :, 6]


# ============================================================
# PREPROCESS
# ============================================================

image_processed = image / 10000.0


# ============================================================
# RGB DISPLAY
# ============================================================

rgb = image[:, :, [2, 1, 0]]

valid_rgb = rgb[np.isfinite(rgb)]

if valid_rgb.size == 0:
    raise RuntimeError(
        "RGB contains no finite pixels."
    )

low = np.percentile(valid_rgb, 2)

high = np.percentile(valid_rgb, 98)

if high <= low:
    high = low + 1.0

rgb_display = (
    (rgb - low) /
    (high - low)
)

rgb_display = np.clip(
    rgb_display,
    0,
    1
)


# ============================================================
# INPUT TENSOR
# ============================================================

image_tensor = np.transpose(
    image_processed,
    (2, 0, 1)
)

image_tensor = torch.tensor(
    image_tensor,
    dtype=torch.float32
)

image_tensor = image_tensor.unsqueeze(0)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

image_tensor = image_tensor.to(device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

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
# INFERENCE
# ============================================================

print("\nRunning prediction...")

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    )

    bare_land_probability = (
        probabilities[
            :,
            BARE_LAND_CLASS
        ]
    )

    use_bare_land = (
        bare_land_probability
        >= BARE_LAND_THRESHOLD
    )

    prediction = torch.where(
        use_bare_land,
        torch.full_like(
            prediction,
            BARE_LAND_CLASS
        ),
        prediction
    )


prediction = (
    prediction[0]
    .cpu()
    .numpy()
)

bare_probability = (
    bare_land_probability[0]
    .cpu()
    .numpy()
)


# ============================================================
# GROUND TRUTH
# ============================================================

valid_mask = np.isfinite(label)

ground_truth = (
    label
    .astype(np.float32)
)


# ============================================================
# BARE-LAND ERROR MAP
# ============================================================

gt_bare = (
    ground_truth == BARE_LAND_CLASS
)

pred_bare = (
    prediction == BARE_LAND_CLASS
)

tp = (
    gt_bare &
    pred_bare &
    valid_mask
)

fn = (
    gt_bare &
    ~pred_bare &
    valid_mask
)

fp = (
    ~gt_bare &
    pred_bare &
    valid_mask
)

tn = (
    ~gt_bare &
    ~pred_bare &
    valid_mask
)


# ============================================================
# COUNTS
# ============================================================

tp_count = int(tp.sum())

fn_count = int(fn.sum())

fp_count = int(fp.sum())

tn_count = int(tn.sum())


print("\n================================")
print("Bare-land Error Analysis")
print("================================")

print(
    f"True Positive : {tp_count:,}"
)

print(
    f"False Negative: {fn_count:,}"
)

print(
    f"False Positive: {fp_count:,}"
)

print(
    f"True Negative : {tn_count:,}"
)


# ============================================================
# METRICS
# ============================================================

bare_precision = (
    tp_count /
    max(
        tp_count + fp_count,
        1
    )
)

bare_recall = (
    tp_count /
    max(
        tp_count + fn_count,
        1
    )
)

bare_iou = (
    tp_count /
    max(
        tp_count +
        fp_count +
        fn_count,
        1
    )
)

bare_f1 = (
    2.0 *
    bare_precision *
    bare_recall /
    max(
        bare_precision +
        bare_recall,
        1e-7
    )
)

print("\nBare-land metrics for this tile:")

print(
    f"Precision: {bare_precision:.4f}"
)

print(
    f"Recall:    {bare_recall:.4f}"
)

print(
    f"IoU:       {bare_iou:.4f}"
)

print(
    f"F1:        {bare_f1:.4f}"
)


# ============================================================
# TILE PIXEL ACCURACY
# ============================================================

if valid_mask.any():

    tile_accuracy = (
        prediction[valid_mask]
        ==
        ground_truth[valid_mask].astype(
            np.int64
        )
    ).mean()

    print(
        f"\nTile Pixel Accuracy: "
        f"{tile_accuracy:.4f}"
    )


# ============================================================
# PREPARE LABEL DISPLAY
# ============================================================

label_display = ground_truth.copy()

label_display[
    ~valid_mask
] = np.nan


# ============================================================
# CREATE ERROR MAP
# ============================================================

# 0 = background / TN
# 1 = TP
# 2 = FN
# 3 = FP

error_map = np.zeros(
    ground_truth.shape,
    dtype=np.uint8
)

error_map[tp] = 1

error_map[fn] = 2

error_map[fp] = 3


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(20, 10)
)


# ============================================================
# PANEL 1 — RGB
# ============================================================

axes[0, 0].imshow(
    rgb_display
)

axes[0, 0].set_title(
    "Sentinel-2 RGB"
)

axes[0, 0].axis("off")


# ============================================================
# PANEL 2 — GROUND TRUTH
# ============================================================

axes[0, 1].imshow(
    label_display,
    vmin=0,
    vmax=4
)

axes[0, 1].set_title(
    "Ground Truth"
)

axes[0, 1].axis("off")


# ============================================================
# PANEL 3 — PREDICTION
# ============================================================

axes[0, 2].imshow(
    prediction,
    vmin=0,
    vmax=4
)

axes[0, 2].set_title(
    "Thresholded Prediction"
)

axes[0, 2].axis("off")


# ============================================================
# PANEL 4 — BARE PROBABILITY
# ============================================================

im = axes[0, 3].imshow(
    bare_probability,
    vmin=0,
    vmax=1
)

axes[0, 3].set_title(
    "Bare-land Probability"
)

axes[0, 3].axis("off")

fig.colorbar(
    im,
    ax=axes[0, 3],
    fraction=0.046,
    pad=0.04
)


# ============================================================
# PANEL 5 — TRUE POSITIVE
# ============================================================

axes[1, 0].imshow(
    tp,
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Bare-land True Positive"
)

axes[1, 0].axis("off")


# ============================================================
# PANEL 6 — FALSE NEGATIVE
# ============================================================

axes[1, 1].imshow(
    fn,
    vmin=0,
    vmax=1
)

axes[1, 1].set_title(
    "Bare-land False Negative"
)

axes[1, 1].axis("off")


# ============================================================
# PANEL 7 — FALSE POSITIVE
# ============================================================

axes[1, 2].imshow(
    fp,
    vmin=0,
    vmax=1
)

axes[1, 2].set_title(
    "Bare-land False Positive"
)

axes[1, 2].axis("off")


# ============================================================
# PANEL 8 — ERROR MAP
# ============================================================

axes[1, 3].imshow(
    error_map,
    vmin=0,
    vmax=3
)

axes[1, 3].set_title(
    "Bare-land Error Map"
)

axes[1, 3].axis("off")


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

output_path = (
    OUTPUT_DIR /
    "GeoVision_Mumbai_tile_114_bareland_error_analysis.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# COMPLETE
# ============================================================

print("\n================================")
print("Error analysis complete!")
print("================================")

print(
    "Saved:",
    output_path
)