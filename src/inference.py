from pathlib import Path

import numpy as np
import torch
import rasterio

from unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

IN_CHANNELS = 6

TILE_SIZE = 256

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_FILENAME = (
    "unet_next_miou_best.pth"
)

# To use the Bare-land-specific model:
#
# MODEL_FILENAME = "unet_next_bareland_best.pth"
#


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

INPUT_FILENAME = (
    "GeoVision_Mumbai_tile_114.tif"
)


# ------------------------------------------------------------
# Bare-land threshold
# ------------------------------------------------------------
#
# 0.50 = normal argmax-like behavior
#
# Higher values make Bare-land harder to predict.
#
# Example:
#
# 0.55
# 0.60
# 0.65
#
# The goal is to see whether increasing this threshold
# improves Bare-land precision and IoU.
# ------------------------------------------------------------

BARE_LAND_THRESHOLD = 0.55


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

INPUT_IMAGE = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    INPUT_FILENAME
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

PREDICTION_MASK = (
    OUTPUT_DIR /
    "prediction_mask.tif"
)

PREDICTION_COLOR = (
    OUTPUT_DIR /
    "prediction_color.png"
)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Agriculture",
    4: "Bare land",
}


# ============================================================
# COLORS
# ============================================================

CLASS_COLORS = np.array(
    [
        [34, 139, 34],       # Vegetation
        [220, 60, 60],       # Built-up
        [30, 144, 255],      # Water
        [255, 215, 0],       # Agriculture
        [210, 180, 140],     # Bare land
    ],
    dtype=np.uint8
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
# HEADER
# ============================================================

print("================================")
print("GeoVision AI Inference")
print("================================")

print(
    "Device:",
    device
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Input:",
    INPUT_IMAGE
)

print(
    "Bare-land threshold:",
    BARE_LAND_THRESHOLD
)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n"
        f"{MODEL_PATH}"
    )

if not INPUT_IMAGE.exists():

    raise FileNotFoundError(
        f"\nInput image not found:\n"
        f"{INPUT_IMAGE}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()

print("Loading model...")

model = UNet(
    in_channels=IN_CHANNELS,
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
# PREPROCESS
# ============================================================

def preprocess_image(
    image
):

    image = image.astype(
        np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # EXACTLY MATCH DATASET.PY
    image = (
        image /
        10000.0
    )

    return image.astype(
        np.float32
    )


# ============================================================
# PAD
# ============================================================

def pad_image(
    image,
    tile_size
):

    bands, height, width = (
        image.shape
    )

    padded_height = (
        (
            height +
            tile_size -
            1
        )
        //
        tile_size
        *
        tile_size
    )

    padded_width = (
        (
            width +
            tile_size -
            1
        )
        //
        tile_size
        *
        tile_size
    )

    pad_height = (
        padded_height -
        height
    )

    pad_width = (
        padded_width -
        width
    )

    if (
        pad_height == 0
        and
        pad_width == 0
    ):

        return (
            image,
            height,
            width
        )

    padded = np.pad(
        image,
        (
            (0, 0),
            (0, pad_height),
            (0, pad_width)
        ),
        mode="reflect"
    )

    return (
        padded,
        height,
        width
    )


# ============================================================
# APPLY BARE-LAND THRESHOLD
# ============================================================

def apply_bare_land_threshold(
    probabilities,
    threshold
):

    # --------------------------------------------------------
    # Standard prediction
    # --------------------------------------------------------

    prediction = (
        probabilities.argmax(
            dim=1
        )
    )

    # --------------------------------------------------------
    # Bare-land probability
    # --------------------------------------------------------

    bare_probability = (
        probabilities[:, 4, :, :]
    )

    # --------------------------------------------------------
    # Find pixels where Bare-land
    # is currently predicted
    # --------------------------------------------------------

    bare_predicted = (
        prediction == 4
    )

    # --------------------------------------------------------
    # Remove Bare-land predictions
    # when confidence is below threshold
    # --------------------------------------------------------

    remove_bare = (
        bare_predicted
        &
        (
            bare_probability
            <
            threshold
        )
    )

    # --------------------------------------------------------
    # For removed Bare-land pixels,
    # select the strongest NON-bare class.
    # --------------------------------------------------------

    non_bare_probabilities = (
        probabilities.clone()
    )

    non_bare_probabilities[
        :,
        4,
        :,
        :
    ] = -1.0

    alternative_prediction = (
        non_bare_probabilities.argmax(
            dim=1
        )
    )

    prediction[
        remove_bare
    ] = alternative_prediction[
        remove_bare
    ]

    return prediction


# ============================================================
# PREDICT TILE
# ============================================================

def predict_tile(
    tile
):

    tensor = torch.from_numpy(
        tile
    ).float()

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                output,
                dim=1
            )
        )

        prediction = (
            apply_bare_land_threshold(
                probabilities,
                BARE_LAND_THRESHOLD
            )
        )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return prediction


# ============================================================
# FULL IMAGE PREDICTION
# ============================================================

def predict_image(
    image
):

    (
        padded_image,
        original_height,
        original_width
    ) = pad_image(
        image,
        TILE_SIZE
    )

    _, padded_height, padded_width = (
        padded_image.shape
    )

    prediction = np.zeros(
        (
            padded_height,
            padded_width
        ),
        dtype=np.uint8
    )

    total_tiles = (
        (
            padded_height //
            TILE_SIZE
        )
        *
        (
            padded_width //
            TILE_SIZE
        )
    )

    tile_number = 0

    print()

    print(
        "Running inference..."
    )

    print(
        f"Image size: "
        f"{original_width} x "
        f"{original_height}"
    )

    print(
        f"Tile size: "
        f"{TILE_SIZE} x "
        f"{TILE_SIZE}"
    )

    print(
        f"Total tiles: "
        f"{total_tiles}"
    )

    for y in range(
        0,
        padded_height,
        TILE_SIZE
    ):

        for x in range(
            0,
            padded_width,
            TILE_SIZE
        ):

            tile_number += 1

            tile = padded_image[
                :,
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            tile_prediction = (
                predict_tile(
                    tile
                )
            )

            prediction[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] = tile_prediction

            print(
                f"Tiles: "
                f"{tile_number}/"
                f"{total_tiles}"
            )

    prediction = prediction[
        :original_height,
        :original_width
    ]

    return prediction


# ============================================================
# COLOR MAP
# ============================================================

def create_color_map(
    prediction
):

    height, width = (
        prediction.shape
    )

    color_image = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    for class_id in range(
        NUM_CLASSES
    ):

        mask = (
            prediction ==
            class_id
        )

        color_image[
            mask
        ] = CLASS_COLORS[
            class_id
        ]

    return color_image


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_prediction_geotiff(
    prediction,
    source
):

    profile = (
        source.profile.copy()
    )

    profile.update(
        {
            "driver": "GTiff",
            "height": prediction.shape[0],
            "width": prediction.shape[1],
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "nodata": None,
        }
    )

    with rasterio.open(
        PREDICTION_MASK,
        "w",
        **profile
    ) as dst:

        dst.write(
            prediction,
            1
        )


# ============================================================
# MAIN
# ============================================================

with rasterio.open(
    INPUT_IMAGE
) as src:

    print()

    print("Opening satellite image...")

    print(
        "Image width:",
        src.width
    )

    print(
        "Image height:",
        src.height
    )

    print(
        "Number of bands:",
        src.count
    )

    print(
        "CRS:",
        src.crs
    )

    print(
        "Transform:",
        src.transform
    )

    print(
        "Band descriptions:",
        src.descriptions
    )

    # --------------------------------------------------------
    # Read first six bands
    # --------------------------------------------------------

    image = src.read(
        [
            1,
            2,
            3,
            4,
            5,
            6
        ]
    )

    print()

    print(
        "Reading input bands:"
    )

    print("1 = B2")
    print("2 = B3")
    print("3 = B4")
    print("4 = B8")
    print("5 = B11")
    print("6 = B12")

    print()

    print(
        "Raw image shape:",
        image.shape
    )

    print(
        "Raw image dtype:",
        image.dtype
    )

    print(
        "Raw minimum:",
        image.min()
    )

    print(
        "Raw maximum:",
        image.max()
    )

    print(
        "Raw means:",
        image.mean(
            axis=(1, 2)
        )
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print()

    print(
        "Preprocessing image..."
    )

    image = preprocess_image(
        image
    )

    print(
        "Processed shape:",
        image.shape
    )

    print(
        "Processed dtype:",
        image.dtype
    )

    print(
        "Processed minimum:",
        image.min()
    )

    print(
        "Processed maximum:",
        image.max()
    )

    print(
        "Processed means:",
        image.mean(
            axis=(1, 2)
        )
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = predict_image(
        image
    )

    print()

    print(
        "Prediction shape:",
        prediction.shape
    )

    # --------------------------------------------------------
    # Save GeoTIFF
    # --------------------------------------------------------

    save_prediction_geotiff(
        prediction,
        src
    )

    print()

    print(
        "Prediction GeoTIFF saved:"
    )

    print(
        PREDICTION_MASK
    )


# ============================================================
# DISTRIBUTION
# ============================================================

print()

print("================================")
print("Prediction Distribution")
print("================================")

total_pixels = (
    prediction.size
)

for class_id in range(
    NUM_CLASSES
):

    count = int(
        np.sum(
            prediction ==
            class_id
        )
    )

    percentage = (
        count /
        total_pixels *
        100.0
    )

    print(
        f"Class {class_id} "
        f"({CLASS_NAMES[class_id]}): "
        f"{count:,} pixels "
        f"({percentage:.2f}%)"
    )


# ============================================================
# COLOR OUTPUT
# ============================================================

color_image = (
    create_color_map(
        prediction
    )
)

from PIL import Image

Image.fromarray(
    color_image
).save(
    PREDICTION_COLOR
)

print()

print(
    "Colored prediction saved:"
)

print(
    PREDICTION_COLOR
)


# ============================================================
# FINAL
# ============================================================

print()

print("================================")
print("Inference complete!")
print("================================")

print()

print("Classes:")

for class_id in range(
    NUM_CLASSES
):

    print(
        f"{class_id}: "
        f"{CLASS_NAMES[class_id]}"
    )