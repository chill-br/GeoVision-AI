from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# Find all TIFF tiles
files = sorted(DATA_DIR.glob("*.tif"))

if len(files) == 0:
    raise RuntimeError("No TIFF files found.")


# Reproducible shuffle
random.seed(42)
random.shuffle(files)


# Split sizes
n = len(files)

train_end = int(0.70 * n)
val_end = train_end + int(0.15 * n)

train_files = files[:train_end]
val_files = files[train_end:val_end]
test_files = files[val_end:]


print("================================")
print("GeoVision AI Dataset Split")
print("================================")

print("Total tiles:", len(files))
print("Training:", len(train_files))
print("Validation:", len(val_files))
print("Testing:", len(test_files))


# Save filenames
def save_split(filename, split_files):

    output_path = SPLIT_DIR / filename

    with open(output_path, "w") as f:
        for path in split_files:
            f.write(path.name + "\n")

    print("Saved:", output_path)


save_split("train.txt", train_files)
save_split("val.txt", val_files)
save_split("test.txt", test_files)


print("\nSplit complete!")