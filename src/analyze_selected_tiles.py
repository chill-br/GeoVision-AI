import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\GeoVision-AI")
SRC_DIR = PROJECT_ROOT / "src"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "unet_next_bareland_best.pth"
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import GeoVisionDataset
from unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
BARE_LAND_CLASS = 4
IGNORE_INDEX = 255

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# These are DATASET INDICES, not TIFF tile numbers.
FAILED_INDICES = [
    213,
    172,
    190,
]

SUCCESSFUL_INDICES = [
    143,
    160,
]

ALL_INDICES = (
    FAILED_INDICES
    + SUCCESSFUL_INDICES
)

CLASS_NAMES = [
    "Vegetation",
    "Built-up",
    "Water",
    "Agriculture",
    "Bare land",
]

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]


# ============================================================
# PRINT HELPERS
# ============================================================

def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def section(text):
    print()
    print("-" * 78)
    print(text)
    print("-" * 78)


# ============================================================
# MODEL CHECKPOINT
# ============================================================

def get_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def load_model():

    header("LOADING MODEL")

    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    state_dict = get_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# SAMPLE EXTRACTION
# ============================================================

def extract_image(sample):

    if isinstance(sample, dict):

        for key in [
            "image",
            "img",
            "input",
            "x",
        ]:

            if key in sample:
                return sample[key]

        raise KeyError(
            f"Could not find image. "
            f"Available keys: {list(sample.keys())}"
        )

    if isinstance(sample, (tuple, list)):

        return sample[0]

    raise TypeError(
        f"Unsupported sample type: {type(sample)}"
    )


def extract_target(sample):

    if isinstance(sample, dict):

        for key in [
            "mask",
            "label",
            "target",
            "y",
        ]:

            if key in sample:
                return sample[key]

        raise KeyError(
            f"Could not find target. "
            f"Available keys: {list(sample.keys())}"
        )

    if isinstance(sample, (tuple, list)):

        if len(sample) < 2:
            raise ValueError(
                "Dataset sample has no target."
            )

        return sample[1]

    raise TypeError(
        f"Unsupported sample type: {type(sample)}"
    )


# ============================================================
# FIND ACTUAL FILENAME
# ============================================================

def find_filename(dataset, index, sample):

    # Try dictionary sample.
    if isinstance(sample, dict):

        for key in [
            "filename",
            "file_name",
            "path",
            "image_path",
            "image_file",
            "name",
        ]:

            if key in sample:

                value = sample[key]

                if isinstance(
                    value,
                    (str, Path),
                ):

                    return Path(str(value)).name

    # Try extra tuple/list elements.
    if isinstance(sample, (tuple, list)):

        for item in sample[2:]:

            if isinstance(
                item,
                (str, Path),
            ):

                text = str(item)

                if text.lower().endswith(
                    (".tif", ".tiff")
                ):

                    return Path(text).name

    # Try common dataset attributes.
    for attr in [
        "files",
        "file_paths",
        "image_files",
        "image_paths",
        "paths",
        "samples",
        "images",
        "data",
    ]:

        if not hasattr(dataset, attr):
            continue

        values = getattr(dataset, attr)

        try:

            if index >= len(values):
                continue

            candidate = values[index]

            if isinstance(
                candidate,
                (str, Path),
            ):

                return Path(
                    str(candidate)
                ).name

            if isinstance(
                candidate,
                (tuple, list),
            ):

                for item in candidate:

                    if isinstance(
                        item,
                        (str, Path),
                    ):

                        text = str(item)

                        if text.lower().endswith(
                            (".tif", ".tiff")
                        ):

                            return Path(text).name

        except Exception:
            pass

    return f"test_index_{index}"


# ============================================================
# PREPARE TENSORS
# ============================================================

def prepare_image(image):

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.float()

    while image.ndim > 3:
        image = image[0]

    if image.ndim != 3:
        raise ValueError(
            f"Expected image [C,H,W], got {image.shape}"
        )

    if image.shape[0] != 6:
        raise ValueError(
            f"Expected 6 bands, got {image.shape[0]}"
        )

    return image


def prepare_target(target):

    if not torch.is_tensor(target):
        target = torch.as_tensor(target)

    while target.ndim > 2:
        target = target[0]

    if target.ndim != 2:
        raise ValueError(
            f"Expected target [H,W], got {target.shape}"
        )

    return target.long()


# ============================================================
# PERCENTILE HELPER
# ============================================================

def percentile(values, p):

    values = np.asarray(values)

    if values.size == 0:
        return 0.0

    return float(
        np.percentile(values, p)
    )


# ============================================================
# ANALYZE ONE TILE
# ============================================================

def analyze_tile(
    model,
    dataset,
    index,
):

    sample = dataset[index]

    image = prepare_image(
        extract_image(sample)
    )

    target = prepare_target(
        extract_target(sample)
    )

    filename = find_filename(
        dataset,
        index,
        sample,
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    image_batch = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = model(
            image_batch
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )[0]

    prediction = prediction.cpu()

    bare_probability = (
        probabilities[
            0,
            BARE_LAND_CLASS,
        ]
        .cpu()
    )

    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    image_np = image.numpy()

    target_np = target.numpy()

    prediction_np = prediction.numpy()

    bare_prob_np = (
        bare_probability.numpy()
    )

    valid = (
        target_np != IGNORE_INDEX
    )

    # --------------------------------------------------------
    # GROUND TRUTH CLASS COUNTS
    # --------------------------------------------------------

    gt_counts = {}

    pred_counts = {}

    for class_id in range(NUM_CLASSES):

        gt_counts[class_id] = int(
            np.logical_and(
                target_np == class_id,
                valid,
            ).sum()
        )

        pred_counts[class_id] = int(
            np.logical_and(
                prediction_np == class_id,
                valid,
            ).sum()
        )

    # --------------------------------------------------------
    # BARE-LAND METRICS
    # --------------------------------------------------------

    gt_bare = (
        target_np == BARE_LAND_CLASS
    )

    pred_bare = (
        prediction_np == BARE_LAND_CLASS
    )

    tp = int(
        np.logical_and(
            np.logical_and(
                gt_bare,
                pred_bare,
            ),
            valid,
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_and(
                gt_bare,
                ~pred_bare,
            ),
            valid,
        ).sum()
    )

    fp = int(
        np.logical_and(
            np.logical_and(
                ~gt_bare,
                pred_bare,
            ),
            valid,
        ).sum()
    )

    tn = int(
        np.logical_and(
            np.logical_and(
                ~gt_bare,
                ~pred_bare,
            ),
            valid,
        ).sum()
    )

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
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0
        else 0.0
    )

    # --------------------------------------------------------
    # WHERE BARE LAND IS MISCLASSIFIED
    # --------------------------------------------------------

    bare_gt_pixels = np.logical_and(
        gt_bare,
        valid,
    )

    confusion = {}

    for class_id in range(NUM_CLASSES):

        confusion[class_id] = int(
            np.logical_and(
                bare_gt_pixels,
                prediction_np == class_id,
            ).sum()
        )

    # --------------------------------------------------------
    # SPECTRAL STATISTICS
    # --------------------------------------------------------

    band_stats = {}

    for band_id, band_name in enumerate(
        BAND_NAMES
    ):

        values = image_np[
            band_id
        ][valid]

        band_stats[band_name] = {
            "mean": float(
                values.mean()
            ),
            "median": float(
                np.median(values)
            ),
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
            "p05": percentile(
                values,
                5,
            ),
            "p95": percentile(
                values,
                95,
            ),
        }

    # --------------------------------------------------------
    # SPECTRAL STATISTICS OF GT BARE LAND ONLY
    # --------------------------------------------------------

    bare_band_stats = {}

    for band_id, band_name in enumerate(
        BAND_NAMES
    ):

        values = image_np[
            band_id
        ][bare_gt_pixels]

        if values.size > 0:

            bare_band_stats[band_name] = {
                "mean": float(
                    values.mean()
                ),
                "median": float(
                    np.median(values)
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                ),
                "p05": percentile(
                    values,
                    5,
                ),
                "p95": percentile(
                    values,
                    95,
                ),
            }

        else:

            bare_band_stats[band_name] = {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p05": 0.0,
                "p95": 0.0,
            }

    # --------------------------------------------------------
    # BARE-LAND PROBABILITY ON GT BARE LAND
    # --------------------------------------------------------

    bare_gt_probabilities = (
        bare_prob_np[bare_gt_pixels]
    )

    # --------------------------------------------------------
    # BARE-LAND PROBABILITY ON NON-BARE LAND
    # --------------------------------------------------------

    nonbare_pixels = np.logical_and(
        ~gt_bare,
        valid,
    )

    nonbare_probabilities = (
        bare_prob_np[nonbare_pixels]
    )

    return {
        "index": index,
        "filename": filename,
        "image": image_np,
        "target": target_np,
        "prediction": prediction_np,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "confusion": confusion,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "band_stats": band_stats,
        "bare_band_stats": bare_band_stats,
        "bare_gt_probabilities": bare_gt_probabilities,
        "nonbare_probabilities": nonbare_probabilities,
    }


# ============================================================
# PRINT TILE ANALYSIS
# ============================================================

def print_tile_analysis(result):

    index = result["index"]
    filename = result["filename"]

    header(
        f"TEST INDEX {index} — {filename}"
    )

    gt_counts = result["gt_counts"]
    pred_counts = result["pred_counts"]

    print()
    print("CLASS DISTRIBUTION")
    print()

    print(
        f"{'Class':<15}"
        f"{'Ground Truth':>15}"
        f"{'Prediction':>15}"
    )

    print("-" * 45)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:<15}"
            f"{gt_counts[class_id]:>15,}"
            f"{pred_counts[class_id]:>15,}"
        )

    # --------------------------------------------------------
    # Bare-land metrics
    # --------------------------------------------------------

    section("BARE-LAND METRICS")

    print(
        f"TP        : {result['TP']:,}"
    )

    print(
        f"FN        : {result['FN']:,}"
    )

    print(
        f"FP        : {result['FP']:,}"
    )

    print(
        f"TN        : {result['TN']:,}"
    )

    print(
        f"Precision : {result['precision']:.4f}"
    )

    print(
        f"Recall    : {result['recall']:.4f}"
    )

    print(
        f"IoU       : {result['iou']:.4f}"
    )

    print(
        f"F1        : {result['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Bare-land confusion
    # --------------------------------------------------------

    section(
        "GROUND-TRUTH BARE LAND → PREDICTED CLASS"
    )

    total_bare = sum(
        result["confusion"].values()
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        count = result[
            "confusion"
        ][class_id]

        percentage = (
            100.0 * count / total_bare
            if total_bare > 0
            else 0.0
        )

        print(
            f"{class_name:<15}"
            f"{count:>10,}"
            f"  ({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    section(
        "BARE-LAND PROBABILITY"
    )

    gt_probs = result[
        "bare_gt_probabilities"
    ]

    nonbare_probs = result[
        "nonbare_probabilities"
    ]

    if gt_probs.size > 0:

        print(
            "GROUND-TRUTH BARE LAND:"
        )

        print(
            f"  Count  : {gt_probs.size:,}"
        )

        print(
            f"  Mean   : {gt_probs.mean():.4f}"
        )

        print(
            f"  Median : {np.median(gt_probs):.4f}"
        )

        print(
            f"  P05    : {percentile(gt_probs, 5):.4f}"
        )

        print(
            f"  P25    : {percentile(gt_probs, 25):.4f}"
        )

        print(
            f"  P75    : {percentile(gt_probs, 75):.4f}"
        )

        print(
            f"  P95    : {percentile(gt_probs, 95):.4f}"
        )

        print(
            f"  >= .25 : "
            f"{(gt_probs >= .25).sum():,}"
        )

        print(
            f"  >= .50 : "
            f"{(gt_probs >= .50).sum():,}"
        )

        print(
            f"  >= .75 : "
            f"{(gt_probs >= .75).sum():,}"
        )

    if nonbare_probs.size > 0:

        print()
        print(
            "GROUND-TRUTH NON-BARE LAND:"
        )

        print(
            f"  Count  : "
            f"{nonbare_probs.size:,}"
        )

        print(
            f"  Mean   : "
            f"{nonbare_probs.mean():.4f}"
        )

        print(
            f"  Median : "
            f"{np.median(nonbare_probs):.4f}"
        )

        print(
            f"  P95    : "
            f"{percentile(nonbare_probs, 95):.4f}"
        )

    # --------------------------------------------------------
    # Spectral statistics
    # --------------------------------------------------------

    section(
        "ALL-PIXEL SPECTRAL MEANS"
    )

    for band_name in BAND_NAMES:

        stats = result[
            "band_stats"
        ][band_name]

        print(
            f"{band_name:<5}"
            f" mean={stats['mean']:.5f}"
            f" median={stats['median']:.5f}"
            f" p05={stats['p05']:.5f}"
            f" p95={stats['p95']:.5f}"
        )

    # --------------------------------------------------------
    # Bare-land spectral statistics
    # --------------------------------------------------------

    section(
        "GROUND-TRUTH BARE-LAND SPECTRAL MEANS"
    )

    if total_bare == 0:

        print(
            "No ground-truth bare-land pixels."
        )

    else:

        for band_name in BAND_NAMES:

            stats = result[
                "bare_band_stats"
            ][band_name]

            print(
                f"{band_name:<5}"
                f" mean={stats['mean']:.5f}"
                f" median={stats['median']:.5f}"
                f" p05={stats['p05']:.5f}"
                f" p95={stats['p95']:.5f}"
            )


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_summary(results):

    header(
        "FAILED vs SUCCESSFUL TILE SUMMARY"
    )

    print()

    print(
        f"{'Index':>7}"
        f"{'Type':>12}"
        f"{'Filename':<38}"
        f"{'GT Bare':>10}"
        f"{'TP':>8}"
        f"{'FN':>8}"
        f"{'FP':>8}"
        f"{'IoU':>9}"
        f"{'F1':>9}"
    )

    print("-" * 120)

    for result in results:

        index = result["index"]

        tile_type = (
            "FAILED"
            if index in FAILED_INDICES
            else "SUCCESS"
        )

        filename = result[
            "filename"
        ]

        if len(filename) > 36:
            filename = filename[:36]

        gt_bare = result[
            "gt_counts"
        ][BARE_LAND_CLASS]

        print(
            f"{index:>7}"
            f"{tile_type:>12}"
            f"{filename:<38}"
            f"{gt_bare:>10,}"
            f"{result['TP']:>8,}"
            f"{result['FN']:>8,}"
            f"{result['FP']:>8,}"
            f"{result['iou']:>9.4f}"
            f"{result['f1']:>9.4f}"
        )


# ============================================================
# COMPARE SPECTRAL MEANS
# ============================================================

def print_group_spectral_comparison(results):

    section(
        "SPECTRAL COMPARISON: FAILED vs SUCCESSFUL"
    )

    failed = [
        r for r in results
        if r["index"] in FAILED_INDICES
    ]

    successful = [
        r for r in results
        if r["index"] in SUCCESSFUL_INDICES
    ]

    print()
    print(
        f"{'Band':<8}"
        f"{'Failed Mean':>18}"
        f"{'Success Mean':>18}"
        f"{'Difference':>18}"
    )

    print("-" * 65)

    for band_name in BAND_NAMES:

        failed_means = [
            r["band_stats"][band_name]["mean"]
            for r in failed
        ]

        success_means = [
            r["band_stats"][band_name]["mean"]
            for r in successful
        ]

        failed_mean = np.mean(
            failed_means
        )

        success_mean = np.mean(
            success_means
        )

        difference = (
            success_mean
            - failed_mean
        )

        print(
            f"{band_name:<8}"
            f"{failed_mean:>18.6f}"
            f"{success_mean:>18.6f}"
            f"{difference:>18.6f}"
        )


# ============================================================
# COMPARE BARE-LAND SPECTRAL MEANS
# ============================================================

def print_bare_spectral_comparison(results):

    section(
        "GROUND-TRUTH BARE-LAND SPECTRAL COMPARISON"
    )

    failed = [
        r for r in results
        if (
            r["index"] in FAILED_INDICES
            and r["gt_counts"][BARE_LAND_CLASS] > 0
        )
    ]

    successful = [
        r for r in results
        if (
            r["index"] in SUCCESSFUL_INDICES
            and r["gt_counts"][BARE_LAND_CLASS] > 0
        )
    ]

    print()
    print(
        f"{'Band':<8}"
        f"{'Failed Bare Mean':>20}"
        f"{'Success Bare Mean':>20}"
        f"{'Difference':>18}"
    )

    print("-" * 70)

    for band_name in BAND_NAMES:

        failed_means = [
            r["bare_band_stats"][band_name]["mean"]
            for r in failed
        ]

        success_means = [
            r["bare_band_stats"][band_name]["mean"]
            for r in successful
        ]

        failed_mean = (
            np.mean(failed_means)
            if failed_means
            else 0.0
        )

        success_mean = (
            np.mean(success_means)
            if success_means
            else 0.0
        )

        difference = (
            success_mean
            - failed_mean
        )

        print(
            f"{band_name:<8}"
            f"{failed_mean:>20.6f}"
            f"{success_mean:>20.6f}"
            f"{difference:>18.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "GeoVision-AI Selected Tile Analysis"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print()
    print(
        "Failed indices    : "
        f"{FAILED_INDICES}"
    )

    print(
        "Successful indices: "
        f"{SUCCESSFUL_INDICES}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    header(
        "LOADING TEST DATASET"
    )

    dataset = GeoVisionDataset(
        split="test"
    )

    print(
        f"Test dataset size: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    results = []

    for index in ALL_INDICES:

        print()
        print(
            f"Analyzing test index {index}..."
        )

        result = analyze_tile(
            model,
            dataset,
            index,
        )

        results.append(result)

        print(
            f"Done: {result['filename']}"
        )

    # --------------------------------------------------------
    # DETAILED RESULTS
    # --------------------------------------------------------

    for result in results:
        print_tile_analysis(
            result
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        results
    )

    # --------------------------------------------------------
    # GROUP COMPARISON
    # --------------------------------------------------------

    print_group_spectral_comparison(
        results
    )

    print_bare_spectral_comparison(
        results
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    header(
        "ANALYSIS COMPLETE"
    )

    print(
        "Five selected tiles were analyzed."
    )

    print()
    print(
        "Next step: paste the complete terminal "
        "output here."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()