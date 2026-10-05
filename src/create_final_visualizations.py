"""
Create publication/resume-quality bare-land segmentation visualizations.

Run from the GeoVision-AI project root:
    python src/create_final_visualizations.py

Outputs:
    outputs/final_visualizations/final_bareland_examples.png
    outputs/final_visualizations/final_bareland_failure_cases.png
    outputs/final_visualizations/final_bareland_success_cases.png

Representative test tiles:
    213 - Chennai tile 19     - failed, GT bare=102, IoU=0.000
    172 - Ahmedabad tile 178  - failed, GT bare=301, IoU=0.003
    190 - Kolkata tile 39     - failed, GT bare=184, IoU=0.004
    143 - Chennai tile 157    - excellent, GT bare=1528, IoU=0.952
    160 - Chennai tile 124    - strong, GT bare=2073, IoU=0.798
    123 - test index 123       - large successful, GT bare=7279, IoU=0.711
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import torch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "unet_next_bareland_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import GeoVisionDataset
from unet import UNet


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = [
    "Vegetation",
    "Built-up",
    "Water",
    "Agriculture",
    "Bare land",
]

# Same colors are used for GT and prediction so comparisons are direct.
CLASS_COLORS = [
    "#2E7D32",  # vegetation
    "#D32F2F",  # built-up
    "#1976D2",  # water
    "#F9A825",  # agriculture
    "#BDBDBD",  # bare land
]
CMAP = ListedColormap(CLASS_COLORS)

# Test-set indices, not TIFF tile numbers.
CASES = [
    {
        "index": 213,
        "label": "Failure: small bare-land region",
        "short": "Small / failed",
        "gt_bare": 102,
        "iou": 0.000,
    },
    {
        "index": 172,
        "label": "Failure: spectrally difficult region",
        "short": "Small / failed",
        "gt_bare": 301,
        "iou": 0.003,
    },
    {
        "index": 190,
        "label": "Failure: bare land confused with built-up/water",
        "short": "Small / failed",
        "gt_bare": 184,
        "iou": 0.004,
    },
    {
        "index": 143,
        "label": "Successful bare-land segmentation",
        "short": "Successful",
        "gt_bare": 1528,
        "iou": 0.952,
    },
    {
        "index": 160,
        "label": "Successful bare-land segmentation",
        "short": "Successful",
        "gt_bare": 2073,
        "iou": 0.798,
    },
    {
        "index": 123,
        "label": "Large bare-land region",
        "short": "Large / successful",
        "gt_bare": 7279,
        "iou": 0.711,
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_model(device):
    model = UNet(in_channels=6, num_classes=NUM_CLASSES).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    # Support both a raw state_dict and common checkpoint dictionaries.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def robust_rgb(image_chw):
    """
    Convert B4/B3/B2 to a displayable RGB image.

    The dataset is already scaled by /10000, so use a robust 2-98%
    stretch independently for each RGB channel.
    """
    # B4=2, B3=1, B2=0 in the six-band input.
    rgb = image_chw[[2, 1, 0]].detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)

    out = np.zeros_like(rgb)

    for c in range(3):
        channel = rgb[:, :, c]
        lo, hi = np.percentile(channel, [2, 98])

        if hi <= lo:
            out[:, :, c] = np.clip(channel, 0, 1)
        else:
            out[:, :, c] = np.clip((channel - lo) / (hi - lo), 0, 1)

    return out


def predict(model, image, device):
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)[0]

    return prediction.cpu().numpy(), probabilities[0].cpu().numpy()


def get_filename(dataset, index):
    return dataset.files[index]


def add_mask(ax, mask, title):
    ax.imshow(mask, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1,
              interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def add_rgb(ax, rgb, title):
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def legend_handles():
    return [
        Patch(facecolor=CLASS_COLORS[i], edgecolor="none", label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{MODEL_PATH}")

    dataset = GeoVisionDataset(split="test")
    model = load_model(device)

    prepared = {}

    for case in CASES:
        idx = case["index"]

        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Test index {idx} is outside dataset.")

        image, label = dataset[idx]
        pred, probs = predict(model, image, device)

        prepared[idx] = {
            "case": case,
            "image": image,
            "label": label.numpy(),
            "pred": pred,
            "rgb": robust_rgb(image),
            "filename": get_filename(dataset, idx),
            "bare_prob": probs[4],
        }

        gt_bare = int((label.numpy() == 4).sum())
        pred_bare = int((pred == 4).sum())

        print(
            f"[{idx}] {get_filename(dataset, idx)} | "
            f"GT bare={gt_bare:,} | Pred bare={pred_bare:,} | "
            f"reported IoU={case['iou']:.3f}"
        )

    # ---------------------------------------------------------------
    # Figure 1: 3x3 overview — three failures + three successes.
    # Each row = RGB / GT / Prediction.
    # ---------------------------------------------------------------
    rows = [
        (213, "Failure case — 102 GT bare pixels"),
        (172, "Failure case — 301 GT bare pixels"),
        (143, "Successful case — 1,528 GT bare pixels"),
    ]

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=3,
        figsize=(13, 13),
        constrained_layout=True,
    )

    for r, (idx, row_title) in enumerate(rows):
        item = prepared[idx]
        case = item["case"]

        add_rgb(
            axes[r, 0],
            item["rgb"],
            f"{row_title}\nSentinel-2 RGB",
        )
        add_mask(
            axes[r, 1],
            item["label"],
            f"Ground Truth\nBare IoU = {case['iou']:.3f}",
        )
        add_mask(
            axes[r, 2],
            item["pred"],
            "Prediction",
        )

    fig.suptitle(
        "Bare-Land Semantic Segmentation: Representative Failure and Success Cases",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.savefig(
        OUTPUT_DIR / "final_bareland_examples.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------------------------------------------------------
    # Figure 2: failure cases.
    # ---------------------------------------------------------------
    failure_ids = [213, 172, 190]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(13, 13),
        constrained_layout=True,
    )

    for r, idx in enumerate(failure_ids):
        item = prepared[idx]
        case = item["case"]

        add_rgb(
            axes[r, 0],
            item["rgb"],
            f"{case['label']}\n{item['filename']}",
        )
        add_mask(
            axes[r, 1],
            item["label"],
            f"Ground Truth\nGT bare = {case['gt_bare']:,}",
        )
        add_mask(
            axes[r, 2],
            item["pred"],
            f"Prediction\nBare IoU = {case['iou']:.3f}",
        )

    fig.suptitle(
        "Bare-Land Failure Cases",
        fontsize=17,
        fontweight="bold",
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.savefig(
        OUTPUT_DIR / "final_bareland_failure_cases.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------------------------------------------------------
    # Figure 3: successful cases, including a large-region example.
    # ---------------------------------------------------------------
    success_ids = [143, 160, 123]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(13, 13),
        constrained_layout=True,
    )

    for r, idx in enumerate(success_ids):
        item = prepared[idx]
        case = item["case"]

        add_rgb(
            axes[r, 0],
            item["rgb"],
            f"{case['label']}\n{item['filename']}",
        )
        add_mask(
            axes[r, 1],
            item["label"],
            f"Ground Truth\nGT bare = {case['gt_bare']:,}",
        )
        add_mask(
            axes[r, 2],
            item["pred"],
            f"Prediction\nBare IoU = {case['iou']:.3f}",
        )

    fig.suptitle(
        "Bare-Land Successful Cases",
        fontsize=17,
        fontweight="bold",
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.savefig(
        OUTPUT_DIR / "final_bareland_success_cases.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------------------------------------------------------
    # Figure 4: direct area-effect comparison.
    # ---------------------------------------------------------------
    area_ids = [213, 160, 123]
    area_labels = [
        "Small / failed\n102 GT bare pixels",
        "Medium / successful\n2,073 GT bare pixels",
        "Large / successful\n7,279 GT bare pixels",
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(13, 13),
        constrained_layout=True,
    )

    for r, (idx, area_label) in enumerate(zip(area_ids, area_labels)):
        item = prepared[idx]
        case = item["case"]

        add_rgb(axes[r, 0], item["rgb"], f"{area_label}\nRGB")
        add_mask(axes[r, 1], item["label"], "Ground Truth")
        add_mask(
            axes[r, 2],
            item["pred"],
            f"Prediction\nBare IoU = {case['iou']:.3f}",
        )

    fig.suptitle(
        "Illustrative Relationship Between Bare-Land Area and Segmentation Quality",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.savefig(
        OUTPUT_DIR / "final_area_effect_examples.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nCreated:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
