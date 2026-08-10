"""
split_dataset.py

Splits the flat dataset/images + dataset/labels produced by generate_dataset.py
into train/val/test subfolders, in the layout YOLOv8 expects:

  dataset_split/
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/
    classes.txt

Usage:
  python split_dataset.py
(edit CONFIG below to tune the split ratios / paths)
"""

import random
import shutil
from pathlib import Path

# ---------------------------- CONFIG ----------------------------
INPUT_DIR = "dataset"            # output of generate_dataset.py
OUTPUT_DIR = "dataset_split"     # final split, ready for training

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1                 # should sum to 1.0 with the two above

SEED = 42
# ------------------------------------------------------------------

random.seed(SEED)


def split_dataset():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    images_dir = Path(INPUT_DIR) / "images"
    labels_dir = Path(INPUT_DIR) / "labels"

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    random.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # remainder goes to test, so all images are used even with rounding
    splits = {
        "train": image_paths[:n_train],
        "val": image_paths[n_train:n_train + n_val],
        "test": image_paths[n_train + n_val:],
    }

    for split_name, paths in splits.items():
        img_out = Path(OUTPUT_DIR) / split_name / "images"
        lbl_out = Path(OUTPUT_DIR) / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"WARNING: no label found for {img_path.name}, skipping")
                continue
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(label_path, lbl_out / label_path.name)

        print(f"{split_name}: {len(paths)} images")

    # copy classes.txt for reference
    classes_src = Path(INPUT_DIR) / "classes.txt"
    if classes_src.exists():
        shutil.copy(classes_src, Path(OUTPUT_DIR) / "classes.txt")

    print(f"Done. Split dataset saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    split_dataset()