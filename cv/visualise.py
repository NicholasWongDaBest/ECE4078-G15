"""
visualize_labels.py

Draws YOLO-format bounding boxes back onto a sample of generated images, so you
can visually spot-check that fruit scale/placement/labels look correct before
committing to training. Saves annotated copies to a separate folder (does not
modify your original dataset).

Usage:
  python visualize_labels.py
(edit CONFIG below to point at your dataset / tune sample size)
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------- CONFIG ----------------------------
DATASET_DIR = "dataset"          # folder containing images/, labels/, classes.txt
OUTPUT_DIR = "dataset_preview"   # where annotated preview images are saved
NUM_SAMPLES = 60                 # how many random images to visualize
BOX_COLOR = (0, 255, 0)          # bright green boxes
BOX_WIDTH = 3
TEXT_COLOR = (255, 255, 255)
SEED = 42
# ------------------------------------------------------------------

random.seed(SEED)


def load_classes(dataset_dir):
    classes_path = Path(dataset_dir) / "classes.txt"
    with open(classes_path) as f:
        return [line.strip() for line in f if line.strip()]


def draw_boxes_on_image(image_path, label_path, classes):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    if not label_path.exists():
        return img  # no labels for this image, return as-is

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_idx = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])

            # convert normalized YOLO coords back to pixel corner coordinates
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)

            label_text = classes[class_idx] if class_idx < len(classes) else str(class_idx)
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_h = text_bbox[3] - text_bbox[1]
            draw.rectangle(
                [x1, max(0, y1 - text_h - 4), text_bbox[2] + 4, y1],
                fill=BOX_COLOR,
            )
            draw.text((x1 + 2, max(0, y1 - text_h - 2)), label_text, fill=(0, 0, 0), font=font)

    return img


def visualize():
    classes = load_classes(DATASET_DIR)
    print(f"Classes: {classes}")

    images_dir = Path(DATASET_DIR) / "images"
    labels_dir = Path(DATASET_DIR) / "labels"
    image_paths = sorted(images_dir.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    sample = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample:
        label_path = labels_dir / (img_path.stem + ".txt")
        annotated = draw_boxes_on_image(img_path, label_path, classes)
        annotated.save(out_dir / img_path.name)

    print(f"Saved {len(sample)} annotated preview images to '{OUTPUT_DIR}/'.")
    print("Open a few of these and check:")
    print("  - boxes are tight around each fruit")
    print("  - fruit size looks plausible relative to the arena")
    print("  - class labels match the actual fruit shown")


if __name__ == "__main__":
    visualize()