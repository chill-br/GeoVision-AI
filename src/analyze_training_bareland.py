from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(r"D:\GeoVision-AI")
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import GeoVisionDataset


# ============================================================
# CONFIGURATION
# ============================================================

BARE_CLASS = 4

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

NUM_BANDS = 6

RANDOM_SEED = 42

# Maximum number of bare-land pixels kept in memory for
# percentile/distribution analysis.
#
# We don't need every pixel. A large random sample gives us
# a very good picture of the training distribution while
# keeping memory usage reasonable.
MAX_STORED_BARE_PIXELS = 2_000_000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_percentiles(values):
    """
    Calculate descriptive statistics for one spectral band.
    """

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def safe_city_from_filename(filename):
    """
    Extract city from filenames such as:

        GeoVision_Chennai_tile_19.tif
        GeoVision_Ahmedabad_tile_178.tif
    """

    name = Path(filename).stem

    if name.startswith("GeoVision_"):
        name = name[len("GeoVision_"):]

    if "_tile_" in name:
        return name.split("_tile_")[0]

    return "Unknown"


def update_running_stats(
    running_count,
    running_mean,
    running_M2,
    values,
):
    """
    Update running mean/std using batch statistics.

    values shape:
        [N, 6]
    """

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return (
            running_count,
            running_mean,
            running_M2,
        )

    batch_count = values.shape[0]

    batch_mean = np.mean(
        values,
        axis=0,
    )

    batch_M2 = np.sum(
        (values - batch_mean) ** 2,
        axis=0,
    )

    old_count = running_count.copy()
    old_mean = running_mean.copy()
    old_M2 = running_M2.copy()

    new_count = old_count + batch_count

    delta = batch_mean - old_mean

    new_mean = (
        old_mean
        + delta * batch_count / new_count
    )

    new_M2 = (
        old_M2
        + batch_M2
        + (
            delta ** 2
            * old_count
            * batch_count
            / new_count
        )
    )

    return (
        new_count,
        new_mean,
        new_M2,
    )


def bounded_random_sample(
    current_sample,
    new_values,
    max_size,
    rng,
):
    """
    Maintain a bounded random sample of spectral pixels.

    This avoids keeping millions of pixels in memory.
    """

    new_values = np.asarray(
        new_values,
        dtype=np.float32,
    )

    if new_values.size == 0:
        return current_sample

    if current_sample is None:
        current_sample = np.empty(
            (0, NUM_BANDS),
            dtype=np.float32,
        )

    combined = np.concatenate(
        [
            current_sample,
            new_values,
        ],
        axis=0,
    )

    if len(combined) <= max_size:
        return combined

    indices = rng.choice(
        len(combined),
        size=max_size,
        replace=False,
    )

    return combined[indices]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TRAINING DATASET BARE-LAND SPECTRAL ANALYSIS")
    print("=" * 80)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Bare class   : {BARE_CLASS}")
    print(f"Bands        : {BAND_NAMES}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # ========================================================
    # LOAD TRAINING DATASET
    # ========================================================

    print("Loading training dataset...")

    # IMPORTANT:
    # This matches the actual GeoVisionDataset API.
    dataset = GeoVisionDataset(
        split="train"
    )

    print(
        f"Training tiles: {len(dataset)}"
    )

    print()

    # ========================================================
    # GLOBAL COUNTERS
    # ========================================================

    total_pixels = 0
    total_bare_pixels = 0
    tiles_with_bare = 0

    # ========================================================
    # RUNNING SPECTRAL STATISTICS
    # ========================================================

    running_count = np.zeros(
        NUM_BANDS,
        dtype=np.int64,
    )

    running_mean = np.zeros(
        NUM_BANDS,
        dtype=np.float64,
    )

    running_M2 = np.zeros(
        NUM_BANDS,
        dtype=np.float64,
    )

    # ========================================================
    # RANDOM SAMPLE FOR PERCENTILES
    # ========================================================

    stored_bare_pixels = None

    # ========================================================
    # PER-TILE RESULTS
    # ========================================================

    per_tile_rows = []

    # ========================================================
    # PROCESS TRAINING DATASET
    # ========================================================

    print("Analyzing training tiles...")
    print()

    for idx in range(len(dataset)):

        try:

            image, label = dataset[idx]

        except Exception as e:

            print()
            print(
                f"[ERROR] Could not process "
                f"training tile index {idx}"
            )

            print(
                f"Reason: {e}"
            )

            print()

            continue

        # ----------------------------------------------------
        # Convert tensors to numpy
        # ----------------------------------------------------

        if torch.is_tensor(image):
            image_np = (
                image
                .detach()
                .cpu()
                .numpy()
            )
        else:
            image_np = np.asarray(image)

        if torch.is_tensor(label):
            label_np = (
                label
                .detach()
                .cpu()
                .numpy()
            )
        else:
            label_np = np.asarray(label)

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if image_np.shape != (
            NUM_BANDS,
            256,
            256,
        ):

            print(
                f"[WARNING] Tile {idx}: "
                f"unexpected image shape "
                f"{image_np.shape}"
            )

            continue

        # ----------------------------------------------------
        # Validate label
        # ----------------------------------------------------

        if label_np.shape != (
            256,
            256,
        ):

            print(
                f"[WARNING] Tile {idx}: "
                f"unexpected label shape "
                f"{label_np.shape}"
            )

            continue

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        filename = dataset.files[idx]

        city = safe_city_from_filename(
            filename
        )

        # ----------------------------------------------------
        # Total pixels
        # ----------------------------------------------------

        pixel_count = label_np.size

        # ----------------------------------------------------
        # Bare-land mask
        # ----------------------------------------------------

        bare_mask = (
            label_np == BARE_CLASS
        )

        bare_count = int(
            np.sum(bare_mask)
        )

        total_pixels += pixel_count

        total_bare_pixels += bare_count

        if bare_count > 0:
            tiles_with_bare += 1

        # ----------------------------------------------------
        # Basic per-tile information
        # ----------------------------------------------------

        row = {
            "dataset_index": idx,
            "filename": filename,
            "city": city,
            "total_pixels": pixel_count,
            "bare_pixels": bare_count,
            "bare_fraction": (
                bare_count / pixel_count
                if pixel_count > 0
                else 0.0
            ),
        }

        # ----------------------------------------------------
        # Bare-land spectral analysis
        # ----------------------------------------------------

        if bare_count > 0:

            # image_np:
            # [6, 256, 256]
            #
            # image_np[:, bare_mask]:
            # [6, N]
            #
            # transpose:
            # [N, 6]

            bare_spectra = (
                image_np[
                    :,
                    bare_mask
                ]
                .T
                .astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # Running global statistics
            # ------------------------------------------------

            (
                running_count,
                running_mean,
                running_M2,
            ) = update_running_stats(
                running_count,
                running_mean,
                running_M2,
                bare_spectra,
            )

            # ------------------------------------------------
            # Store bounded random sample
            # ------------------------------------------------

            stored_bare_pixels = (
                bounded_random_sample(
                    stored_bare_pixels,
                    bare_spectra,
                    MAX_STORED_BARE_PIXELS,
                    rng,
                )
            )

            # ------------------------------------------------
            # Per-tile statistics
            # ------------------------------------------------

            for band_idx, band_name in enumerate(
                BAND_NAMES
            ):

                values = (
                    bare_spectra[
                        :,
                        band_idx
                    ]
                )

                row[
                    f"{band_name}_mean"
                ] = float(
                    np.mean(values)
                )

                row[
                    f"{band_name}_std"
                ] = float(
                    np.std(values)
                )

                row[
                    f"{band_name}_min"
                ] = float(
                    np.min(values)
                )

                row[
                    f"{band_name}_median"
                ] = float(
                    np.median(values)
                )

                row[
                    f"{band_name}_max"
                ] = float(
                    np.max(values)
                )

        else:

            for band_name in BAND_NAMES:

                row[
                    f"{band_name}_mean"
                ] = np.nan

                row[
                    f"{band_name}_std"
                ] = np.nan

                row[
                    f"{band_name}_min"
                ] = np.nan

                row[
                    f"{band_name}_median"
                ] = np.nan

                row[
                    f"{band_name}_max"
                ] = np.nan

        per_tile_rows.append(row)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            idx == 0
            or (idx + 1) % 25 == 0
            or idx == len(dataset) - 1
        ):

            print(
                f"Processed "
                f"{idx + 1:4d}/"
                f"{len(dataset):4d} tiles | "
                f"Bare pixels: "
                f"{total_bare_pixels:,}"
            )

    # ========================================================
    # PER-TILE DATAFRAME
    # ========================================================

    tile_df = pd.DataFrame(
        per_tile_rows
    )

    tile_csv = (
        OUTPUT_DIR
        / "training_bareland_per_tile.csv"
    )

    tile_df.to_csv(
        tile_csv,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING DATASET SUMMARY")
    print("=" * 80)

    print()
    print(
        f"Training tiles                : "
        f"{len(dataset):,}"
    )

    print(
        f"Tiles with bare land          : "
        f"{tiles_with_bare:,}"
    )

    print(
        f"Total pixels                  : "
        f"{total_pixels:,}"
    )

    print(
        f"Total bare-land pixels        : "
        f"{total_bare_pixels:,}"
    )

    if total_pixels > 0:

        bare_percentage = (
            100.0
            * total_bare_pixels
            / total_pixels
        )

        print(
            f"Overall bare-land percentage : "
            f"{bare_percentage:.4f}%"
        )

    print(
        f"Stored random bare pixels    : "
        f"{0 if stored_bare_pixels is None else len(stored_bare_pixels):,}"
    )

    # ========================================================
    # GLOBAL MEAN / STD
    # ========================================================

    print()
    print("-" * 80)
    print("GLOBAL TRAINING BARE-LAND SPECTRAL STATISTICS")
    print("-" * 80)

    global_rows = []

    for band_idx, band_name in enumerate(
        BAND_NAMES
    ):

        count = int(
            running_count[band_idx]
        )

        if count > 1:

            variance = (
                running_M2[band_idx]
                / (count - 1)
            )

            std = float(
                np.sqrt(
                    max(
                        variance,
                        0.0,
                    )
                )
            )

        else:

            std = np.nan

        mean = float(
            running_mean[band_idx]
        )

        global_rows.append(
            {
                "band": band_name,
                "count": count,
                "mean": mean,
                "std": std,
            }
        )

        print(
            f"{band_name:4s} | "
            f"count={count:10,d} | "
            f"mean={mean:.6f} | "
            f"std={std:.6f}"
        )

    global_df = pd.DataFrame(
        global_rows
    )

    global_csv = (
        OUTPUT_DIR
        / "training_bareland_global_stats.csv"
    )

    global_df.to_csv(
        global_csv,
        index=False,
    )

    # ========================================================
    # GLOBAL PERCENTILES
    # ========================================================

    percentile_df = pd.DataFrame()

    if (
        stored_bare_pixels is not None
        and len(stored_bare_pixels) > 0
    ):

        print()
        print("-" * 80)
        print("GLOBAL TRAINING BARE-LAND PERCENTILES")
        print("-" * 80)

        percentile_rows = []

        for band_idx, band_name in enumerate(
            BAND_NAMES
        ):

            values = (
                stored_bare_pixels[
                    :,
                    band_idx
                ]
            )

            stats = calculate_percentiles(
                values
            )

            percentile_rows.append(
                {
                    "band": band_name,
                    **stats,
                }
            )

            print()
            print(
                f"{band_name}"
            )

            print(
                f"  Min    : "
                f"{stats['min']:.6f}"
            )

            print(
                f"  P01    : "
                f"{stats['p01']:.6f}"
            )

            print(
                f"  P05    : "
                f"{stats['p05']:.6f}"
            )

            print(
                f"  P10    : "
                f"{stats['p10']:.6f}"
            )

            print(
                f"  P25    : "
                f"{stats['p25']:.6f}"
            )

            print(
                f"  Median : "
                f"{stats['median']:.6f}"
            )

            print(
                f"  P75    : "
                f"{stats['p75']:.6f}"
            )

            print(
                f"  P90    : "
                f"{stats['p90']:.6f}"
            )

            print(
                f"  P95    : "
                f"{stats['p95']:.6f}"
            )

            print(
                f"  P99    : "
                f"{stats['p99']:.6f}"
            )

            print(
                f"  Max    : "
                f"{stats['max']:.6f}"
            )

            print(
                f"  Mean   : "
                f"{stats['mean']:.6f}"
            )

            print(
                f"  Std    : "
                f"{stats['std']:.6f}"
            )

        percentile_df = pd.DataFrame(
            percentile_rows
        )

    percentile_csv = (
        OUTPUT_DIR
        / "training_bareland_percentiles.csv"
    )

    percentile_df.to_csv(
        percentile_csv,
        index=False,
    )

    # ========================================================
    # TILE COVERAGE
    # ========================================================

    print()
    print("-" * 80)
    print("BARE-LAND COVERAGE ACROSS TRAINING TILES")
    print("-" * 80)

    bare_tiles = tile_df[
        tile_df["bare_pixels"] > 0
    ].copy()

    print(
        f"Tiles containing bare land: "
        f"{len(bare_tiles):,}"
    )

    if len(bare_tiles) > 0:

        print(
            f"Minimum bare pixels/tile : "
            f"{bare_tiles['bare_pixels'].min():,.0f}"
        )

        print(
            f"Median bare pixels/tile  : "
            f"{bare_tiles['bare_pixels'].median():,.0f}"
        )

        print(
            f"Mean bare pixels/tile    : "
            f"{bare_tiles['bare_pixels'].mean():,.1f}"
        )

        print(
            f"Maximum bare pixels/tile : "
            f"{bare_tiles['bare_pixels'].max():,.0f}"
        )

    # ========================================================
    # CITY ANALYSIS
    # ========================================================

    print()
    print("-" * 80)
    print("BARE-LAND COVERAGE BY CITY")
    print("-" * 80)

    city_df = (
        tile_df
        .groupby("city")
        .agg(
            tiles=(
                "dataset_index",
                "count",
            ),
            tiles_with_bare=(
                "bare_pixels",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            total_pixels=(
                "total_pixels",
                "sum",
            ),
            bare_pixels=(
                "bare_pixels",
                "sum",
            ),
        )
        .reset_index()
    )

    city_df["bare_fraction"] = (
        city_df["bare_pixels"]
        / city_df["total_pixels"]
    )

    city_df = city_df.sort_values(
        "bare_pixels",
        ascending=False,
    )

    for _, row in city_df.iterrows():

        print(
            f"{str(row['city']):15s} | "
            f"tiles={int(row['tiles']):4d} | "
            f"tiles_with_bare="
            f"{int(row['tiles_with_bare']):4d} | "
            f"bare_pixels="
            f"{int(row['bare_pixels']):10,d} | "
            f"bare_fraction="
            f"{100 * row['bare_fraction']:.3f}%"
        )

    city_csv = (
        OUTPUT_DIR
        / "training_bareland_by_city.csv"
    )

    city_df.to_csv(
        city_csv,
        index=False,
    )

    # ========================================================
    # TOP TRAINING TILES
    # ========================================================

    print()
    print("-" * 80)
    print("TRAINING TILES WITH MOST BARE LAND")
    print("-" * 80)

    top_tiles = (
        tile_df
        .sort_values(
            "bare_pixels",
            ascending=False,
        )
        .head(20)
    )

    for _, row in top_tiles.iterrows():

        print(
            f"Index {int(row['dataset_index']):4d} | "
            f"{row['filename']} | "
            f"Bare pixels="
            f"{int(row['bare_pixels']):,} | "
            f"Bare %="
            f"{100 * row['bare_fraction']:.2f}%"
        )

    # ========================================================
    # LOWEST BARE-LAND TRAINING TILES
    # ========================================================

    print()
    print("-" * 80)
    print("TRAINING TILES WITH LEAST BARE LAND")
    print("-" * 80)

    low_tiles = (
        tile_df[
            tile_df["bare_pixels"] > 0
        ]
        .sort_values(
            "bare_pixels",
            ascending=True,
        )
        .head(20)
    )

    for _, row in low_tiles.iterrows():

        print(
            f"Index {int(row['dataset_index']):4d} | "
            f"{row['filename']} | "
            f"Bare pixels="
            f"{int(row['bare_pixels']):,} | "
            f"Bare %="
            f"{100 * row['bare_fraction']:.4f}%"
        )

    # ========================================================
    # REFERENCE TEST TILES
    # ========================================================
    #
    # These are the GT bare-land spectral means from the
    # five tiles we previously analyzed.
    #
    # They are only used for comparison.
    # ========================================================

    test_reference = {

        "Failed_213_Chennai_19": {
            "B2": 0.15543,
            "B3": 0.18359,
            "B4": 0.20114,
            "B8": 0.26665,
            "B11": 0.29550,
            "B12": 0.24791,
        },

        "Failed_172_Ahmedabad_178": {
            "B2": 0.03767,
            "B3": 0.08732,
            "B4": 0.05198,
            "B8": 0.04919,
            "B11": 0.02137,
            "B12": 0.01674,
        },

        "Failed_190_Kolkata_39": {
            "B2": 0.09024,
            "B3": 0.11269,
            "B4": 0.12359,
            "B8": 0.16753,
            "B11": 0.18196,
            "B12": 0.15918,
        },

        "Success_143_Chennai_157": {
            "B2": 0.16700,
            "B3": 0.22770,
            "B4": 0.27769,
            "B8": 0.30636,
            "B11": 0.40997,
            "B12": 0.42102,
        },

        "Success_160_Chennai_124": {
            "B2": 0.15177,
            "B3": 0.19927,
            "B4": 0.23954,
            "B8": 0.27360,
            "B11": 0.33010,
            "B12": 0.32927,
        },
    }

    # ========================================================
    # TRAINING VS TEST DISTRIBUTION
    # ========================================================

    if (
        stored_bare_pixels is not None
        and len(stored_bare_pixels) > 0
    ):

        print()
        print("=" * 80)
        print("TRAINING VS TEST BARE-LAND DISTRIBUTION")
        print("=" * 80)

        train_mean = np.mean(
            stored_bare_pixels,
            axis=0,
        )

        train_std = np.std(
            stored_bare_pixels,
            axis=0,
        )

        train_std = np.where(
            train_std < 1e-8,
            1.0,
            train_std,
        )

        comparison_rows = []

        for name, reference in test_reference.items():

            test_vector = np.array(
                [
                    reference[band]
                    for band in BAND_NAMES
                ],
                dtype=np.float32,
            )

            # -----------------------------------------------
            # Standardized distance from training mean
            # -----------------------------------------------

            z = (
                test_vector
                - train_mean
            ) / train_std

            standardized_distance = float(
                np.sqrt(
                    np.sum(
                        z ** 2
                    )
                )
            )

            # -----------------------------------------------
            # Nearest training bare pixel
            # -----------------------------------------------

            nearest_distance = np.inf

            chunk_size = 100_000

            for start in range(
                0,
                len(stored_bare_pixels),
                chunk_size,
            ):

                end = min(
                    start + chunk_size,
                    len(stored_bare_pixels),
                )

                chunk = (
                    stored_bare_pixels[
                        start:end
                    ]
                )

                distances = np.sqrt(
                    np.sum(
                        (
                            chunk
                            - test_vector
                        ) ** 2,
                        axis=1,
                    )
                )

                local_min = float(
                    np.min(distances)
                )

                if local_min < nearest_distance:
                    nearest_distance = local_min

            row = {
                "test_tile": name,
                "nearest_training_euclidean_distance":
                    nearest_distance,
                "training_standardized_distance":
                    standardized_distance,
            }

            for band_idx, band_name in enumerate(
                BAND_NAMES
            ):

                row[
                    f"{band_name}_test_mean"
                ] = float(
                    test_vector[
                        band_idx
                    ]
                )

                row[
                    f"{band_name}_training_mean"
                ] = float(
                    train_mean[
                        band_idx
                    ]
                )

                row[
                    f"{band_name}_training_std"
                ] = float(
                    train_std[
                        band_idx
                    ]
                )

                row[
                    f"{band_name}_zscore"
                ] = float(
                    z[
                        band_idx
                    ]
                )

            comparison_rows.append(
                row
            )

            print()
            print(name)

            print(
                f"  Standardized distance: "
                f"{standardized_distance:.4f}"
            )

            print(
                f"  Nearest training pixel: "
                f"{nearest_distance:.6f}"
            )

            print(
                "  Band z-scores:"
            )

            for band_idx, band_name in enumerate(
                BAND_NAMES
            ):

                print(
                    f"    {band_name:4s}: "
                    f"{z[band_idx]:+.3f}"
                )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        comparison_csv = (
            OUTPUT_DIR
            / "training_vs_test_bareland_distribution.csv"
        )

        comparison_df.to_csv(
            comparison_csv,
            index=False,
        )

    else:

        comparison_csv = None

        print()
        print(
            "WARNING: No bare-land pixels were found "
            "in the training dataset."
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print()
    print(
        f"Per-tile analysis:"
    )

    print(
        f"  {tile_csv}"
    )

    print()
    print(
        f"Global statistics:"
    )

    print(
        f"  {global_csv}"
    )

    print()
    print(
        f"Percentiles:"
    )

    print(
        f"  {percentile_csv}"
    )

    print()
    print(
        f"City statistics:"
    )

    print(
        f"  {city_csv}"
    )

    if comparison_csv is not None:

        print()
        print(
            "Training vs test distribution:"
        )

        print(
            f"  {comparison_csv}"
        )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()