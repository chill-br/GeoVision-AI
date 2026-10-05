import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\GeoVision-AI")
SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "unet_next_bareland_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "test_tile_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import GeoVisionDataset
from unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

# These are DATASET INDICES.
# They are NOT TIFF tile numbers.

WORST_INDICES = [
    213,
    172,
    190,
]

BEST_INDICES = [
    143,
    160,
]

SELECTED_INDICES = (
    WORST_INDICES
    + BEST_INDICES
)

NUM_CLASSES = 5
BARE_LAND_CLASS = 4
IGNORE_INDEX = 255

# Display/reference threshold only.
# It does NOT modify argmax prediction.
BARE_LAND_THRESHOLD = 0.55

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "Vegetation",
    "Built-up",
    "Water",
    "Agriculture",
    "Bare land",
]

CLASS_COLORS = [
    "#2ca02c",
    "#d62728",
    "#1f77b4",
    "#ffbf00",
    "#8c564b",
]

CLASS_CMAP = ListedColormap(
    CLASS_COLORS
)


# ============================================================
# HELPERS
# ============================================================

def print_header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# CHECKPOINT HANDLING
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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print_header("Loading model")

    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    state_dict = get_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    model.load_state_dict(
        state_dict
    )

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
            "Could not find image in dataset sample. "
            f"Available keys: {list(sample.keys())}"
        )

    if isinstance(sample, (tuple, list)):

        if len(sample) < 1:
            raise ValueError(
                "Dataset sample is empty."
            )

        return sample[0]

    raise TypeError(
        f"Unsupported dataset sample type: "
        f"{type(sample)}"
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
            "Could not find target in dataset sample. "
            f"Available keys: {list(sample.keys())}"
        )

    if isinstance(sample, (tuple, list)):

        if len(sample) < 2:
            raise ValueError(
                "Dataset sample contains fewer "
                "than two elements."
            )

        return sample[1]

    raise TypeError(
        f"Unsupported dataset sample type: "
        f"{type(sample)}"
    )


# ============================================================
# TENSOR PREPARATION
# ============================================================

def prepare_image(image):

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.float()

    while image.ndim > 3:
        image = image[0]

    if image.ndim != 3:
        raise ValueError(
            "Expected image shape [C,H,W], "
            f"got {tuple(image.shape)}"
        )

    if image.shape[0] != 6:
        raise ValueError(
            "Expected 6 input bands, "
            f"got {image.shape[0]}"
        )

    return image


def prepare_target(target):

    if not torch.is_tensor(target):
        target = torch.as_tensor(target)

    while target.ndim > 2:
        target = target[0]

    if target.ndim != 2:
        raise ValueError(
            "Expected target shape [H,W], "
            f"got {tuple(target.shape)}"
        )

    return target.long()


# ============================================================
# FIND FILENAME
# ============================================================

def find_filename(
    dataset,
    index,
    sample,
):

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
                    return Path(
                        str(value)
                    ).name

    if isinstance(sample, (tuple, list)):

        for item in sample[2:]:

            if isinstance(
                item,
                (str, Path),
            ):

                return Path(
                    str(item)
                ).name

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

        values = getattr(
            dataset,
            attr,
        )

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

                        item_string = str(item)

                        if item_string.lower().endswith(
                            (".tif", ".tiff")
                        ):

                            return Path(
                                item_string
                            ).name

        except Exception:
            continue

    return f"test_index_{index}"


# ============================================================
# RGB
# ============================================================

def normalize_rgb(image):

    image = (
        image
        .detach()
        .cpu()
        .numpy()
    )

    blue = image[0]
    green = image[1]
    red = image[2]

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    low = np.percentile(
        rgb,
        2,
    )

    high = np.percentile(
        rgb,
        98,
    )

    if high > low:

        rgb = (
            rgb - low
        ) / (
            high - low
        )

    else:

        rgb = rgb - low

    return np.clip(
        rgb,
        0,
        1,
    )


# ============================================================
# BARE-LAND METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
):

    prediction = (
        prediction
        .detach()
        .cpu()
        .numpy()
    )

    target = (
        target
        .detach()
        .cpu()
        .numpy()
    )

    valid = (
        target != IGNORE_INDEX
    )

    pred_bare = (
        prediction == BARE_LAND_CLASS
    )

    true_bare = (
        target == BARE_LAND_CLASS
    )

    tp = np.logical_and(
        pred_bare,
        true_bare,
    )

    tp = np.logical_and(
        tp,
        valid,
    ).sum()

    fn = np.logical_and(
        ~pred_bare,
        true_bare,
    )

    fn = np.logical_and(
        fn,
        valid,
    ).sum()

    fp = np.logical_and(
        pred_bare,
        ~true_bare,
    )

    fp = np.logical_and(
        fp,
        valid,
    ).sum()

    tn = np.logical_and(
        ~pred_bare,
        ~true_bare,
    )

    tn = np.logical_and(
        tn,
        valid,
    ).sum()

    precision_den = tp + fp
    recall_den = tp + fn
    iou_den = tp + fp + fn
    f1_den = (
        2 * tp
        + fp
        + fn
    )

    precision = (
        tp / precision_den
        if precision_den > 0
        else 0.0
    )

    recall = (
        tp / recall_den
        if recall_den > 0
        else 0.0
    )

    iou = (
        tp / iou_den
        if iou_den > 0
        else 0.0
    )

    f1 = (
        2 * tp / f1_den
        if f1_den > 0
        else 0.0
    )

    accuracy_den = (
        tp
        + tn
        + fp
        + fn
    )

    accuracy = (
        (tp + tn)
        / accuracy_den
        if accuracy_den > 0
        else 0.0
    )

    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


# ============================================================
# ERROR MAP
# ============================================================

def create_error_map(
    prediction,
    target,
):

    prediction = (
        prediction
        .cpu()
        .numpy()
    )

    target = (
        target
        .cpu()
        .numpy()
    )

    error_map = np.zeros_like(
        target,
        dtype=np.uint8,
    )

    pred_bare = (
        prediction == BARE_LAND_CLASS
    )

    true_bare = (
        target == BARE_LAND_CLASS
    )

    valid = (
        target != IGNORE_INDEX
    )

    # 0 = True Negative
    error_map[
        np.logical_and(
            np.logical_and(
                ~pred_bare,
                ~true_bare,
            ),
            valid,
        )
    ] = 0

    # 1 = True Positive
    error_map[
        np.logical_and(
            np.logical_and(
                pred_bare,
                true_bare,
            ),
            valid,
        )
    ] = 1

    # 2 = False Positive
    error_map[
        np.logical_and(
            np.logical_and(
                pred_bare,
                ~true_bare,
            ),
            valid,
        )
    ] = 2

    # 3 = False Negative
    error_map[
        np.logical_and(
            np.logical_and(
                ~pred_bare,
                true_bare,
            ),
            valid,
        )
    ] = 3

    # 4 = Ignore
    error_map[~valid] = 4

    return error_map


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    index,
    filename,
    metrics,
):

    print()
    print(
        f"Test index : {index}"
    )

    print(
        f"Filename   : {filename}"
    )

    print("-" * 50)

    print(
        f"TP         : {metrics['TP']:,}"
    )

    print(
        f"FN         : {metrics['FN']:,}"
    )

    print(
        f"FP         : {metrics['FP']:,}"
    )

    print(
        f"TN         : {metrics['TN']:,}"
    )

    print(
        f"Precision  : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"IoU        : "
        f"{metrics['iou']:.4f}"
    )

    print(
        f"F1         : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Accuracy   : "
        f"{metrics['accuracy']:.4f}"
    )


# ============================================================
# SAVE DIAGNOSTIC FIGURE
# ============================================================

def save_diagnostic_figure(
    index,
    filename,
    rgb,
    target,
    prediction,
    bare_probability,
    error_map,
    metrics,
):

    target_np = (
        target.cpu().numpy()
    )

    prediction_np = (
        prediction.cpu().numpy()
    )

    bare_probability_np = (
        bare_probability
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    fig.suptitle(
        f"GeoVision-AI Test Tile Diagnostic\n"
        f"Index {index} — {filename}",
        fontsize=16,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    axes[0, 0].imshow(rgb)

    axes[0, 0].set_title(
        "RGB Image"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_display = target_np.copy()

    gt_display[
        gt_display == IGNORE_INDEX
    ] = 0

    axes[0, 1].imshow(
        gt_display,
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=NUM_CLASSES - 1,
    )

    axes[0, 1].set_title(
        "Ground Truth"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[0, 2].imshow(
        prediction_np,
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=NUM_CLASSES - 1,
    )

    axes[0, 2].set_title(
        "Model Prediction"
    )

    axes[0, 2].axis("off")

    # --------------------------------------------------------
    # Bare probability
    # --------------------------------------------------------

    probability_plot = axes[1, 0].imshow(
        bare_probability_np,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    axes[1, 0].set_title(
        "Bare-land Probability\n"
        f"Reference = {BARE_LAND_THRESHOLD}"
    )

    axes[1, 0].axis("off")

    fig.colorbar(
        probability_plot,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Error map
    # --------------------------------------------------------

    error_colors = [
        "#dddddd",
        "#00aa00",
        "#ff0000",
        "#0000ff",
        "#000000",
    ]

    error_cmap = ListedColormap(
        error_colors
    )

    axes[1, 1].imshow(
        error_map,
        cmap=error_cmap,
        vmin=0,
        vmax=4,
    )

    axes[1, 1].set_title(
        "Bare-land Error Map"
    )

    axes[1, 1].axis("off")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    axes[1, 2].axis("off")

    metric_text = (
        "BARE-LAND METRICS\n\n"
        f"TP: {metrics['TP']:,}\n"
        f"FN: {metrics['FN']:,}\n"
        f"FP: {metrics['FP']:,}\n"
        f"TN: {metrics['TN']:,}\n\n"
        f"Precision: "
        f"{metrics['precision']:.4f}\n"
        f"Recall:    "
        f"{metrics['recall']:.4f}\n"
        f"IoU:       "
        f"{metrics['iou']:.4f}\n"
        f"F1:        "
        f"{metrics['f1']:.4f}\n"
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}\n\n"
        "ERROR MAP\n"
        "Gray  = True Negative\n"
        "Green = True Positive\n"
        "Red   = False Positive\n"
        "Blue  = False Negative\n"
        "Black = Ignore"
    )

    axes[1, 2].text(
        0.05,
        0.95,
        metric_text,
        transform=axes[1, 2].transAxes,
        fontsize=12,
        verticalalignment="top",
        family="monospace",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    safe_filename = (
        Path(filename)
        .stem
        .replace(" ", "_")
    )

    output_name = (
        f"test_index_{index:03d}_"
        f"{safe_filename}_diagnostic.png"
    )

    output_path = (
        OUTPUT_DIR / output_name
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PROCESS ONE TILE
# ============================================================

def process_tile(
    model,
    dataset,
    index,
):

    print_header(
        f"Processing test index {index}"
    )

    sample = dataset[index]

    image = extract_image(
        sample
    )

    target = extract_target(
        sample
    )

    image = prepare_image(
        image
    )

    target = prepare_target(
        target
    )

    filename = find_filename(
        dataset,
        index,
        sample,
    )

    print(
        f"Actual file : {filename}"
    )

    print(
        f"Image shape : {tuple(image.shape)}"
    )

    print(
        f"Target shape: {tuple(target.shape)}"
    )

    print(
        f"Image dtype : {image.dtype}"
    )

    print(
        f"Image min   : "
        f"{image.min().item():.6f}"
    )

    print(
        f"Image max   : "
        f"{image.max().item():.6f}"
    )

    image_tensor = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )[0]

        bare_probability = (
            probabilities[
                0,
                BARE_LAND_CLASS,
            ]
            .cpu()
        )

    if tuple(prediction.shape) != tuple(
        target.shape
    ):

        raise ValueError(
            "Prediction and target shapes "
            "do not match.\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Target:     {tuple(target.shape)}"
        )

    metrics = calculate_metrics(
        prediction,
        target,
    )

    print_metrics(
        index,
        filename,
        metrics,
    )

    bare_prob_np = (
        bare_probability.numpy()
    )

    print()
    print(
        "Bare-land probability:"
    )

    print(
        f"  Minimum : "
        f"{bare_prob_np.min():.4f}"
    )

    print(
        f"  Maximum : "
        f"{bare_prob_np.max():.4f}"
    )

    print(
        f"  Mean    : "
        f"{bare_prob_np.mean():.4f}"
    )

    print(
        f"  >= 0.10 : "
        f"{(bare_prob_np >= 0.10).sum():,}"
    )

    print(
        f"  >= 0.25 : "
        f"{(bare_prob_np >= 0.25).sum():,}"
    )

    print(
        f"  >= 0.50 : "
        f"{(bare_prob_np >= 0.50).sum():,}"
    )

    print(
        f"  >= 0.55 : "
        f"{(bare_prob_np >= 0.55).sum():,}"
    )

    error_map = create_error_map(
        prediction,
        target,
    )

    rgb = normalize_rgb(
        image
    )

    save_diagnostic_figure(
        index=index,
        filename=filename,
        rgb=rgb,
        target=target,
        prediction=prediction,
        bare_probability=bare_probability,
        error_map=error_map,
        metrics=metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "GeoVision-AI Test Tile Visual Diagnostics"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Raw data     : {RAW_DIR}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print_header(
        "Loading test dataset"
    )

    test_dataset = GeoVisionDataset(
        split="test"
    )

    dataset_length = len(
        test_dataset
    )

    print(
        f"Test dataset size: "
        f"{dataset_length}"
    )

    # --------------------------------------------------------
    # Validate indices
    # --------------------------------------------------------

    print()
    print(
        f"Selected test indices: "
        f"{SELECTED_INDICES}"
    )

    for index in SELECTED_INDICES:

        if (
            index < 0
            or index >= dataset_length
        ):

            raise IndexError(
                f"Test index {index} is outside "
                f"dataset range 0 to "
                f"{dataset_length - 1}"
            )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for index in SELECTED_INDICES:

        try:

            process_tile(
                model,
                test_dataset,
                index,
            )

            successful += 1

        except Exception as error:

            failed += 1

            print()

            print(
                f"ERROR while processing "
                f"test index {index}:"
            )

            print(
                repr(error)
            )

            continue

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header(
        "DIAGNOSTIC COMPLETE"
    )

    print(
        f"Successful tiles : "
        f"{successful}"
    )

    print(
        f"Failed tiles     : "
        f"{failed}"
    )

    print()

    print(
        "Diagnostic images saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()