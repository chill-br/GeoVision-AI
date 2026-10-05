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

MODEL_PATH = PROJECT_ROOT / "models" / "unet_next_bareland_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tile_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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

CLASS_CMAP = ListedColormap(CLASS_COLORS)

BARE_LAND_THRESHOLD = 0.55


# ============================================================
# PRINT HEADER
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
# LOAD TIFF
# ============================================================

def load_tiff(tiff_path):
    import rasterio

    print_header("Loading TIFF")

    print(f"File : {tiff_path}")

    if not tiff_path.exists():
        raise FileNotFoundError(
            f"TIFF file not found:\n{tiff_path}"
        )

    with rasterio.open(tiff_path) as src:
        data = src.read()

        print(f"Shape : {data.shape}")
        print(f"Dtype : {data.dtype}")
        print(f"CRS   : {src.crs}")
        print(f"Count : {src.count}")

        descriptions = src.descriptions

        print(f"Bands : {descriptions}")

    if data.shape[0] < 6:
        raise ValueError(
            f"Expected at least 6 bands, got {data.shape[0]}"
        )

    # First six bands are:
    # B2, B3, B4, B8, B11, B12
    image = data[:6].astype(np.float32)

    # Sentinel-2 preprocessing
    image /= 10000.0

    return image


# ============================================================
# RGB VISUALIZATION
# ============================================================

def normalize_rgb(image):
    """
    Input:
        [6, H, W]

    Uses:
        B4 = Red
        B3 = Green
        B2 = Blue
    """

    blue = image[0]
    green = image[1]
    red = image[2]

    rgb = np.stack(
        [red, green, blue],
        axis=-1,
    )

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high > low:
        rgb = (rgb - low) / (high - low)
    else:
        rgb = rgb - low

    return np.clip(rgb, 0, 1)


# ============================================================
# INFERENCE
# ============================================================

def predict(model, image):
    image_tensor = torch.from_numpy(image).float()

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )[0]

        bare_probability = probabilities[
            0,
            BARE_LAND_CLASS,
        ]

    return (
        prediction.cpu(),
        bare_probability.cpu(),
        probabilities.cpu(),
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    tiff_path,
    image,
    prediction,
    bare_probability,
):
    rgb = normalize_rgb(image)

    prediction_np = prediction.numpy()
    bare_probability_np = bare_probability.numpy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    fig.suptitle(
        f"GeoVision-AI Prediction\n{tiff_path.name}",
        fontsize=16,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    axes[0].imshow(rgb)

    axes[0].set_title("RGB")
    axes[0].axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[1].imshow(
        prediction_np,
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=NUM_CLASSES - 1,
    )

    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    # --------------------------------------------------------
    # Bare-land probability
    # --------------------------------------------------------

    probability_plot = axes[2].imshow(
        bare_probability_np,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"Bare-land Probability\n"
        f"Reference = {BARE_LAND_THRESHOLD}"
    )

    axes[2].axis("off")

    fig.colorbar(
        probability_plot,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{tiff_path.stem}_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved visualization:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "GeoVision-AI Single Tile Visualization"
    )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw data     : {RAW_DIR}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Device       : {DEVICE}")

    # --------------------------------------------------------
    # CHANGE THIS ONLY WHEN YOU WANT TO VISUALIZE ANOTHER TILE
    # --------------------------------------------------------

    TILE_NAME = "GeoVision_Mumbai_tile_114.tif"

    tiff_path = RAW_DIR / TILE_NAME

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_tiff(tiff_path)

    print()
    print(f"Input shape : {image.shape}")
    print(f"Input dtype : {image.dtype}")
    print(f"Input min   : {image.min():.6f}")
    print(f"Input max   : {image.max():.6f}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction, bare_probability, probabilities = predict(
        model,
        image,
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print_header("Prediction Distribution")

    total_pixels = prediction.numel()

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = (
            prediction == class_id
        ).sum().item()

        percentage = (
            100.0 * count / total_pixels
        )

        print(
            f"{class_id} - "
            f"{class_name:<12} : "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Bare-land probability
    # --------------------------------------------------------

    bare_np = bare_probability.numpy()

    print()
    print("Bare-land probability:")
    print(f"  Minimum : {bare_np.min():.4f}")
    print(f"  Maximum : {bare_np.max():.4f}")
    print(f"  Mean    : {bare_np.mean():.4f}")
    print(
        f"  >= 0.10 : "
        f"{(bare_np >= 0.10).sum():,}"
    )
    print(
        f"  >= 0.25 : "
        f"{(bare_np >= 0.25).sum():,}"
    )
    print(
        f"  >= 0.50 : "
        f"{(bare_np >= 0.50).sum():,}"
    )
    print(
        f"  >= 0.55 : "
        f"{(bare_np >= 0.55).sum():,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_visualization(
        tiff_path=tiff_path,
        image=image,
        prediction=prediction,
        bare_probability=bare_probability,
    )

    print_header("COMPLETE")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()