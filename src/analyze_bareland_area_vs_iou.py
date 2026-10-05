from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "models" / "unet_next_bareland_best.pth"

# Make sure src can be imported
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import GeoVisionDataset
from unet import UNet


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5
BARE_CLASS = 4

CLASS_NAMES = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Agriculture",
    4: "Bare land",
}

# Bare-land pixel-count bins
AREA_BINS = [
    (1, 50, "1-50"),
    (51, 100, "51-100"),
    (101, 250, "101-250"),
    (251, 500, "251-500"),
    (501, 1000, "501-1,000"),
    (1001, 2500, "1,001-2,500"),
    (2501, 5000, "2,501-5,000"),
    (5001, float("inf"), "5,001+"),
]


# ============================================================
# Helper functions
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def get_area_bin(count):
    for low, high, label in AREA_BINS:
        if low <= count <= high:
            return label
    return "Unknown"


def rankdata(values):
    """
    Simple average-rank implementation.
    No scipy required.
    """
    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.zeros(len(values), dtype=np.float64)

    i = 0

    while i < len(values):
        j = i + 1

        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1

        # Average rank, 1-indexed
        avg_rank = (i + 1 + j) / 2.0

        ranks[order[i:j]] = avg_rank

        i = j

    return ranks


def spearman_correlation(x, y):
    """
    Spearman rank correlation without scipy.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return np.nan

    rx = rankdata(x)
    ry = rankdata(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denominator = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))

    if denominator == 0:
        return np.nan

    return float(np.sum(rx * ry) / denominator)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("BARE-LAND AREA vs IoU ANALYSIS")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nModel:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    dataset = GeoVisionDataset(split="test")

    print("Number of test tiles:", len(dataset))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading U-Net...")

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Handle either raw state_dict or checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    tile_results = []

    # --------------------------------------------------------
    # Process test tiles
    # --------------------------------------------------------

    print("\nProcessing test tiles...\n")

    with torch.no_grad():

        for idx in range(len(dataset)):

            image, label = dataset[idx]

            # ------------------------------------------------
            # Input
            # ------------------------------------------------

            image_batch = image.unsqueeze(0).to(device)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            logits = model(image_batch)

            probabilities = torch.softmax(logits, dim=1)

            prediction = torch.argmax(
                probabilities,
                dim=1
            )[0]

            bare_probability = probabilities[0, BARE_CLASS]

            # Move to CPU
            label_np = label.cpu().numpy()
            prediction_np = prediction.cpu().numpy()
            bare_probability_np = bare_probability.cpu().numpy()

            # ------------------------------------------------
            # Ground-truth bare land
            # ------------------------------------------------

            gt_bare = label_np == BARE_CLASS
            pred_bare = prediction_np == BARE_CLASS

            gt_bare_count = int(gt_bare.sum())
            pred_bare_count = int(pred_bare.sum())

            # ------------------------------------------------
            # Confusion counts
            # ------------------------------------------------

            tp = int(np.logical_and(gt_bare, pred_bare).sum())

            fn = int(np.logical_and(gt_bare, ~pred_bare).sum())

            fp = int(np.logical_and(~gt_bare, pred_bare).sum())

            tn = int(np.logical_and(~gt_bare, ~pred_bare).sum())

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            precision = safe_divide(
                tp,
                tp + fp
            )

            recall = safe_divide(
                tp,
                tp + fn
            )

            iou = safe_divide(
                tp,
                tp + fp + fn
            )

            f1 = safe_divide(
                2 * tp,
                2 * tp + fp + fn
            )

            # ------------------------------------------------
            # Bare-land fraction
            # ------------------------------------------------

            total_pixels = label_np.size

            bare_fraction = safe_divide(
                gt_bare_count,
                total_pixels
            )

            # ------------------------------------------------
            # Probability statistics
            # ------------------------------------------------

            if gt_bare_count > 0:

                gt_bare_probs = bare_probability_np[gt_bare]

                mean_bare_probability = float(
                    np.mean(gt_bare_probs)
                )

                median_bare_probability = float(
                    np.median(gt_bare_probs)
                )

                p95_bare_probability = float(
                    np.percentile(
                        gt_bare_probs,
                        95
                    )
                )

            else:

                mean_bare_probability = np.nan
                median_bare_probability = np.nan
                p95_bare_probability = np.nan

            # ------------------------------------------------
            # Area bin
            # ------------------------------------------------

            area_bin = get_area_bin(
                gt_bare_count
            ) if gt_bare_count > 0 else "No bare land"

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            tile_results.append({
                "test_index": idx,

                "gt_bare_pixels": gt_bare_count,

                "gt_bare_fraction": bare_fraction,

                "pred_bare_pixels": pred_bare_count,

                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,

                "precision": precision,
                "recall": recall,
                "iou": iou,
                "f1": f1,

                "mean_gt_bare_probability":
                    mean_bare_probability,

                "median_gt_bare_probability":
                    median_bare_probability,

                "p95_gt_bare_probability":
                    p95_bare_probability,

                "area_bin": area_bin,
            })

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (idx + 1) % 10 == 0 or idx == 0:

                print(
                    f"Processed {idx + 1:3d}/{len(dataset)} "
                    f"| GT bare: {gt_bare_count:6d} "
                    f"| IoU: {iou:.4f}"
                )

    # ========================================================
    # Convert to DataFrame
    # ========================================================

    df = pd.DataFrame(tile_results)

    # ========================================================
    # Save per-tile results
    # ========================================================

    per_tile_path = (
        OUTPUT_DIR /
        "bareland_area_vs_iou.csv"
    )

    df.to_csv(
        per_tile_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("PER-TILE RESULTS")
    print("=" * 70)

    print(
        f"\nSaved:\n{per_tile_path}"
    )

    # ========================================================
    # Separate tiles with and without bare land
    # ========================================================

    with_bare = df[
        df["gt_bare_pixels"] > 0
    ].copy()

    without_bare = df[
        df["gt_bare_pixels"] == 0
    ].copy()

    print("\nTiles with ground-truth bare land:")
    print(len(with_bare))

    print("\nTiles with NO ground-truth bare land:")
    print(len(without_bare))

    # ========================================================
    # Overall statistics for tiles containing bare land
    # ========================================================

    if len(with_bare) > 0:

        print("\n" + "=" * 70)
        print("OVERALL — TILES CONTAINING BARE LAND")
        print("=" * 70)

        print(
            f"\nMean GT bare pixels: "
            f"{with_bare['gt_bare_pixels'].mean():.2f}"
        )

        print(
            f"Median GT bare pixels: "
            f"{with_bare['gt_bare_pixels'].median():.2f}"
        )

        print(
            f"Mean bare fraction: "
            f"{with_bare['gt_bare_fraction'].mean():.4%}"
        )

        print(
            f"Mean IoU: "
            f"{with_bare['iou'].mean():.4f}"
        )

        print(
            f"Median IoU: "
            f"{with_bare['iou'].median():.4f}"
        )

        print(
            f"Mean recall: "
            f"{with_bare['recall'].mean():.4f}"
        )

        print(
            f"Mean precision: "
            f"{with_bare['precision'].mean():.4f}"
        )

        print(
            f"Mean F1: "
            f"{with_bare['f1'].mean():.4f}"
        )

    # ========================================================
    # Area-bin analysis
    # ========================================================

    summary_rows = []

    print("\n" + "=" * 70)
    print("BARE-LAND AREA BINS")
    print("=" * 70)

    for _, _, label in AREA_BINS:

        group = with_bare[
            with_bare["area_bin"] == label
        ]

        if len(group) == 0:
            continue

        row = {
            "area_bin": label,

            "tile_count": len(group),

            "mean_gt_bare_pixels":
                group["gt_bare_pixels"].mean(),

            "median_gt_bare_pixels":
                group["gt_bare_pixels"].median(),

            "mean_bare_fraction":
                group["gt_bare_fraction"].mean(),

            "mean_iou":
                group["iou"].mean(),

            "median_iou":
                group["iou"].median(),

            "mean_recall":
                group["recall"].mean(),

            "mean_precision":
                group["precision"].mean(),

            "mean_f1":
                group["f1"].mean(),

            "mean_bare_probability":
                group["mean_gt_bare_probability"].mean(),

            "median_bare_probability":
                group["median_gt_bare_probability"].mean(),

            "percent_iou_above_0_5":
                (group["iou"] > 0.5).mean() * 100,

            "percent_iou_above_0_25":
                (group["iou"] > 0.25).mean() * 100,
        }

        summary_rows.append(row)

        print(
            f"\n{label}"
        )

        print(
            f"  Tiles: "
            f"{len(group)}"
        )

        print(
            f"  Mean GT bare pixels: "
            f"{group['gt_bare_pixels'].mean():.1f}"
        )

        print(
            f"  Median GT bare pixels: "
            f"{group['gt_bare_pixels'].median():.1f}"
        )

        print(
            f"  Mean IoU: "
            f"{group['iou'].mean():.4f}"
        )

        print(
            f"  Median IoU: "
            f"{group['iou'].median():.4f}"
        )

        print(
            f"  Mean recall: "
            f"{group['recall'].mean():.4f}"
        )

        print(
            f"  Mean precision: "
            f"{group['precision'].mean():.4f}"
        )

        print(
            f"  Mean F1: "
            f"{group['f1'].mean():.4f}"
        )

        print(
            f"  IoU > 0.50: "
            f"{(group['iou'] > 0.5).mean() * 100:.1f}%"
        )

    # ========================================================
    # Save area-bin summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR /
        "bareland_area_bins_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("AREA-BIN SUMMARY SAVED")
    print("=" * 70)

    print(summary_path)

    # ========================================================
    # Correlation analysis
    # ========================================================

    if len(with_bare) >= 2:

        counts = with_bare[
            "gt_bare_pixels"
        ].to_numpy()

        ious = with_bare[
            "iou"
        ].to_numpy()

        fractions = with_bare[
            "gt_bare_fraction"
        ].to_numpy()

        # Pearson correlation
        pearson_count = np.corrcoef(
            counts,
            ious
        )[0, 1]

        # Spearman correlation
        spearman_count = spearman_correlation(
            counts,
            ious
        )

        pearson_fraction = np.corrcoef(
            fractions,
            ious
        )[0, 1]

        spearman_fraction = spearman_correlation(
            fractions,
            ious
        )

        # Log-area correlation
        log_counts = np.log1p(
            counts
        )

        pearson_log = np.corrcoef(
            log_counts,
            ious
        )[0, 1]

        spearman_log = spearman_correlation(
            log_counts,
            ious
        )

        print("\n" + "=" * 70)
        print("CORRELATION BETWEEN BARE-LAND AREA AND IoU")
        print("=" * 70)

        print(
            f"\nPearson correlation "
            f"(GT bare pixels vs IoU): "
            f"{pearson_count:.4f}"
        )

        print(
            f"Spearman correlation "
            f"(GT bare pixels vs IoU): "
            f"{spearman_count:.4f}"
        )

        print(
            f"\nPearson correlation "
            f"(GT bare fraction vs IoU): "
            f"{pearson_fraction:.4f}"
        )

        print(
            f"Spearman correlation "
            f"(GT bare fraction vs IoU): "
            f"{spearman_fraction:.4f}"
        )

        print(
            f"\nPearson correlation "
            f"(log1p(GT bare pixels) vs IoU): "
            f"{pearson_log:.4f}"
        )

        print(
            f"Spearman correlation "
            f"(log1p(GT bare pixels) vs IoU): "
            f"{spearman_log:.4f}"
        )

    # ========================================================
    # Best and worst tiles
    # ========================================================

    print("\n" + "=" * 70)
    print("BEST 10 BARE-LAND TILES")
    print("=" * 70)

    best = with_bare.sort_values(
        "iou",
        ascending=False
    ).head(10)

    print(
        best[
            [
                "test_index",
                "gt_bare_pixels",
                "pred_bare_pixels",
                "precision",
                "recall",
                "iou",
                "f1",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("WORST 10 BARE-LAND TILES")
    print("=" * 70)

    worst = with_bare.sort_values(
        "iou",
        ascending=True
    ).head(10)

    print(
        worst[
            [
                "test_index",
                "gt_bare_pixels",
                "pred_bare_pixels",
                "precision",
                "recall",
                "iou",
                "f1",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # Final interpretation hints
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)

    print(
        """
If Spearman correlation is strongly POSITIVE:
    Larger bare-land regions tend to have better IoU.
    This supports the hypothesis that small/disconnected regions
    are harder for the model.

If correlation is close to ZERO:
    Bare-land area alone probably does not explain the failures.
    Spectral characteristics, spatial context, or class ambiguity
    may be more important.

If correlation is NEGATIVE:
    Larger regions are not necessarily easier.
    Investigate spectral/domain differences and confusion patterns.

Look especially at the area-bin table:
    1-50
    51-100
    101-250
    251-500
    501-1,000
    1,001-2,500
    2,501-5,000
    5,001+

A clear upward trend in mean IoU across these bins would be
strong evidence for an area/context effect.
"""
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()