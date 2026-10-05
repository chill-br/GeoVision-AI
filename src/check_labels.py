from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"


CLASS_NAMES = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Agriculture",
    4: "Bare land",
    255: "Ignore / NaN",
}


def main():

    files = sorted(
        DATA_DIR.glob("*.tif")
    )

    print("================================")
    print("GeoVision AI Label Diagnostic")
    print("================================")

    print(
        "Data directory:",
        DATA_DIR
    )

    print(
        "Number of TIFF files:",
        len(files)
    )

    if not files:

        raise RuntimeError(
            "No TIFF files found."
        )

    total_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        255: 0,
    }

    for path in files:

        with rasterio.open(path) as src:

            label = src.read(7)

        label = np.nan_to_num(
            label,
            nan=255
        )

        unique, counts = np.unique(
            label,
            return_counts=True
        )

        print(
            "\n",
            path.name
        )

        for value, count in zip(
            unique,
            counts
        ):

            value = int(value)
            count = int(count)

            print(
                f"  {value:3d} "
                f"{CLASS_NAMES.get(value, 'UNKNOWN'):12s} "
                f"{count:,}"
            )

            if value in total_counts:
                total_counts[value] += count

    print("\n================================")
    print("TOTAL LABEL DISTRIBUTION")
    print("================================")

    total_pixels = sum(
        total_counts.values()
    )

    for class_id in [
        0,
        1,
        2,
        3,
        4,
        255,
    ]:

        count = total_counts[class_id]

        percentage = (
            count / total_pixels * 100
            if total_pixels > 0
            else 0
        )

        print(
            f"{class_id:3d} "
            f"{CLASS_NAMES[class_id]:12s}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()