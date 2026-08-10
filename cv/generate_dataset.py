"""
generate_dataset.py

Generates a large synthetic training dataset for YOLOv8 by superimposing
background-removed fruit cutouts onto arena background photos, at random
scale/position/rotation. Since we control the paste location, YOLO-format
bounding box labels are written out automatically -- no manual annotation
needed for the base dataset (Roboflow is still useful afterward for
spot-checking / cleaning up edge cases, or for augmentation).

Expected folder structure (run remove_background.py first to create images_fruits_cutout/):

  images_fruits_cutout/
    redapple/
      redapple_01.png    <- RGBA, transparent background
      redapple_02.png
      ...
    greenapple/
      ...
    orange/
    mango/
    capsicum/
    lemon/
    lime/
      ...

  images_arena/
    arena_01.jpg
    arena_02.jpg
    ...

Output:

  dataset/
    images/
      img_00000.jpg
      img_00001.jpg
      ...
    labels/
      img_00000.txt     <- YOLO format: "class_idx x_center y_center width height" (normalized 0-1)
      img_00001.txt
      ...
    classes.txt          <- one class name per line, in index order

Usage:
  python generate_dataset.py
(edit the CONFIG section below to point at your folders / tune parameters)
"""

import csv
import glob
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------- CONFIG ----------------------------
FRUIT_DIR = "images_fruits_cutout"  # input: folders of transparent-background fruit PNGs
ARENA_DIR = "images_arena"          # input: arena background photos
OUTPUT_DIR = "dataset"              # output: images/ + labels/ + classes.txt

NUM_IMAGES = 50                # total synthetic images to generate
MIN_FRUITS_PER_IMAGE = 1
MAX_FRUITS_PER_IMAGE = 3

# --- Physically-grounded scaling (recommended) ---
# Instead of picking an arbitrary fraction of image width, we use the pinhole
# camera model: apparent_pixel_width = (focal_length_px * real_width_m) / distance_m
# This makes a mango actually look bigger than a lime at the same distance,
# and makes fruit size in synthetic images match what the robot's camera
# would really see at realistic detection ranges.
USE_PHYSICAL_SCALING = True
OBJECT_LIST_CSV = "../object_list.csv"                  # has length/width/height per fruit (in meters)
INTRINSIC_PATH = "../calibration/param/intrinsic.txt"    # camera_matrix.txt from camera_calibration.py
DISTANCE_MIN_M = 0.15   # closest realistic distance the robot would detect a fruit from
DISTANCE_MAX_M = 1.00   # farthest realistic distance -- tune both based on your arena size / camera FOV
# NOTE: distance is sampled uniformly in 1/distance (not distance itself), so apparent
# SIZE is spread evenly between its min and max instead of being biased toward "small".

# --- Fallback scaling (used only if USE_PHYSICAL_SCALING is False, or CSV/intrinsics missing) ---
MIN_SCALE = 0.06                 # fruit width as a fraction of arena image width
MAX_SCALE = 0.25

# --- Floor-grounded placement ---
# Vertical position is tied to the same sampled distance used for scaling, so a
# close/big fruit lands low in the frame (near the bottom, like it's on the floor
# right in front of the camera) and a far/small fruit lands higher up (near the
# horizon). Only used when physical scaling is active; otherwise placement stays
# fully random (old behaviour).
FLOOR_TOP_FRAC = 0.35     # y-fraction where the FARTHEST fruits are anchored (near horizon)
FLOOR_BOTTOM_FRAC = 0.95  # y-fraction where the CLOSEST fruits are anchored (near bottom edge)
Y_JITTER_PX = 12          # small random vertical jitter so placement isn't perfectly rigid

# --- Drop shadow (helps fruit look grounded on the floor instead of floating) ---
SHADOW_ENABLED = True
SHADOW_OPACITY = 80        # 0-255 alpha of the shadow at its darkest point
SHADOW_WIDTH_FRAC = 0.85   # shadow ellipse width, as a fraction of the fruit's rendered width
SHADOW_HEIGHT_FRAC = 0.10  # shadow ellipse height, as a fraction of the fruit's rendered width
SHADOW_BLUR_RADIUS = 4

MAX_ROTATION_DEG = 25            # random in-plane rotation applied to each fruit
MAX_OVERLAP_IOU = 0.15           # reject placements that overlap existing fruit boxes more than this
MAX_PLACEMENT_TRIES = 20         # tries per fruit before giving up on that one

BRIGHTNESS_JITTER = 0.15         # +/- 15% random brightness scaling per fruit paste
JPEG_QUALITY = 95
SEED = 42
# ------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def load_classes(fruit_dir):
    classes = sorted([d.name for d in Path(fruit_dir).iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class subfolders found in {fruit_dir}")
    return classes


def load_fruit_images(fruit_dir, classes):
    """Returns dict: class_name -> list of PIL RGBA images (loaded once, reused many times)."""
    fruit_images = {}
    for c in classes:
        paths = glob.glob(os.path.join(fruit_dir, c, "*.png"))
        imgs = [Image.open(p).convert("RGBA") for p in paths]
        if not imgs:
            print(f"WARNING: no cutout images found for class '{c}' in {fruit_dir}/{c}")
        fruit_images[c] = imgs
    return fruit_images


def load_object_sizes(csv_path):
    """Reads object_list.csv -> dict: class_name -> average horizontal real-world
    size in meters (mean of length and width columns). Used with the pinhole
    camera model to size fruit cutouts realistically."""
    sizes = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["object"].strip()
            length = float(row["length(m)"])
            width = float(row["width(m)"])
            sizes[name] = (length + width) / 2.0
    return sizes


def load_camera_fx(intrinsic_path):
    """Reads the calibrated camera matrix and returns the focal length in pixels (fx).
    This value is only valid at the resolution the camera was calibrated at --
    make sure your arena images are captured at that same native resolution."""
    K = np.loadtxt(intrinsic_path, delimiter=",")
    return float(K[0, 0])


def load_arena_images(arena_dir):
    paths = glob.glob(os.path.join(arena_dir, "*.jpg")) + \
            glob.glob(os.path.join(arena_dir, "*.jpeg")) + \
            glob.glob(os.path.join(arena_dir, "*.png"))
    if not paths:
        raise RuntimeError(f"No arena images found in {arena_dir}")
    return paths


def random_brightness(img, jitter):
    """img: RGBA PIL image. Jitters brightness of RGB channels only (leaves alpha untouched)."""
    arr = np.array(img).astype(np.float32)
    factor = 1.0 + random.uniform(-jitter, jitter)
    arr[..., :3] = np.clip(arr[..., :3] * factor, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_shadow(arena_img, center_x, bottom_y, fruit_width):
    """Draws a soft, blurred dark ellipse under the fruit's bottom-center point,
    so it reads as sitting ON the floor rather than pasted on top of it.
    Composited directly onto arena_img (RGBA) before the fruit itself is pasted."""
    shadow_w = int(fruit_width * SHADOW_WIDTH_FRAC)
    shadow_h = max(4, int(fruit_width * SHADOW_HEIGHT_FRAC))

    shadow_layer = Image.new("RGBA", arena_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_layer)
    x0 = center_x - shadow_w // 2
    y0 = bottom_y - shadow_h // 2
    x1 = center_x + shadow_w // 2
    y1 = bottom_y + shadow_h // 2
    draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, SHADOW_OPACITY))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=SHADOW_BLUR_RADIUS))

    arena_img.alpha_composite(shadow_layer)


def paste_fruit(arena_img, fruit_img, placed_boxes, real_width_m=None, fx=None):
    """Try random scale/rotation/position for fruit_img on arena_img, avoiding heavy
    overlap with already-placed boxes. Pastes in-place using alpha as mask.
    Returns (success: bool, box or None).

    If real_width_m and fx are provided (physical mode):
      - Distance is sampled uniformly in 1/distance, so apparent SIZE is spread
        evenly between its min and max (rather than biased toward "small", which
        happens if distance itself is sampled uniformly).
      - Vertical position is grounded: close/big fruit is anchored low in the frame
        (near the floor right in front of the camera), far/small fruit is anchored
        near the horizon -- using the same sampled distance, so scale and position
        are consistent with each other instead of independent random choices.
      - A soft drop shadow is drawn under the fruit so it reads as sitting on the
        floor instead of floating.
    Otherwise falls back to the old flat MIN_SCALE/MAX_SCALE + fully random position.
    """
    aw, ah = arena_img.size
    fw, fh = fruit_img.size
    if fw == 0:
        return False, None

    physical_mode = real_width_m is not None and fx is not None

    for _ in range(MAX_PLACEMENT_TRIES):
        if physical_mode:
            inv_min = 1.0 / DISTANCE_MAX_M
            inv_max = 1.0 / DISTANCE_MIN_M
            inv_distance = random.uniform(inv_min, inv_max)
            distance = 1.0 / inv_distance
            target_w = int(fx * real_width_m / distance)
        else:
            scale = random.uniform(MIN_SCALE, MAX_SCALE)
            target_w = int(aw * scale)

        target_h = int(fh * (target_w / fw))
        if target_w < 5 or target_h < 5:
            continue

        resized = fruit_img.resize((target_w, target_h), Image.LANCZOS)

        angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        rotated = resized.rotate(angle, expand=True)

        # rotate(expand=True) pads the canvas with transparent pixels so the
        # rotated shape fits -- crop tightly to the actual visible (non-transparent)
        # pixels, otherwise the padding gets treated as part of the fruit, which
        # pushes the floor anchor/shadow/label below where the fruit actually is.
        alpha_bbox = rotated.split()[-1].getbbox()
        if alpha_bbox is None:
            continue  # fully transparent, nothing to paste
        rotated = rotated.crop(alpha_bbox)

        rw, rh = rotated.size
        if rw >= aw or rh >= ah:
            continue

        x = random.randint(0, aw - rw)

        if physical_mode:
            # t=0 -> closest (bottom of frame), t=1 -> farthest (near horizon)
            t = (distance - DISTANCE_MIN_M) / (DISTANCE_MAX_M - DISTANCE_MIN_M)
            t = min(max(t, 0.0), 1.0)
            floor_bottom_y = FLOOR_BOTTOM_FRAC * ah - t * (FLOOR_BOTTOM_FRAC - FLOOR_TOP_FRAC) * ah
            floor_bottom_y += random.uniform(-Y_JITTER_PX, Y_JITTER_PX)
            y = int(floor_bottom_y - rh)
            y = min(max(y, 0), ah - rh)
        else:
            y = random.randint(0, ah - rh)

        box = (x, y, x + rw, y + rh)

        if all(iou(box, pb) < MAX_OVERLAP_IOU for pb in placed_boxes):
            if physical_mode and SHADOW_ENABLED:
                draw_shadow(arena_img, center_x=x + rw // 2, bottom_y=y + rh, fruit_width=rw)
            arena_img.paste(rotated, (x, y), rotated)  # alpha channel used as paste mask
            return True, box

    return False, None


def generate_dataset():
    classes = load_classes(FRUIT_DIR)
    print(f"Found {len(classes)} classes: {classes}")

    fruit_images = load_fruit_images(FRUIT_DIR, classes)
    arena_paths = load_arena_images(ARENA_DIR)
    print(f"Found {len(arena_paths)} arena background images")

    object_sizes = None
    fx = None
    if USE_PHYSICAL_SCALING:
        try:
            object_sizes = load_object_sizes(OBJECT_LIST_CSV)
            fx = load_camera_fx(INTRINSIC_PATH)
            missing = [c for c in classes if c not in object_sizes]
            if missing:
                print(f"WARNING: {missing} not found in {OBJECT_LIST_CSV}, "
                      f"those classes will fall back to flat scaling")
            print(f"Physical scaling enabled (fx={fx:.1f}px, "
                  f"distance range {DISTANCE_MIN_M}-{DISTANCE_MAX_M}m)")
        except (FileNotFoundError, OSError) as e:
            print(f"WARNING: could not load {OBJECT_LIST_CSV} or {INTRINSIC_PATH} ({e}). "
                  f"Falling back to flat MIN_SCALE/MAX_SCALE for all fruit.")
            object_sizes = None
            fx = None

    images_out = Path(OUTPUT_DIR) / "images"
    labels_out = Path(OUTPUT_DIR) / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    with open(Path(OUTPUT_DIR) / "classes.txt", "w") as f:
        f.write("\n".join(classes))

    generated = 0
    for i in range(NUM_IMAGES):
        arena_path = random.choice(arena_paths)
        arena_img = Image.open(arena_path).convert("RGBA")
        aw, ah = arena_img.size

        num_fruits = random.randint(MIN_FRUITS_PER_IMAGE, MAX_FRUITS_PER_IMAGE)
        placed_boxes = []
        labels = []

        for _ in range(num_fruits):
            cls = random.choice(classes)
            if not fruit_images[cls]:
                continue
            fruit_img = random.choice(fruit_images[cls])
            fruit_img = random_brightness(fruit_img, BRIGHTNESS_JITTER)

            real_width_m = object_sizes.get(cls) if object_sizes else None
            success, box = paste_fruit(arena_img, fruit_img, placed_boxes, real_width_m, fx)
            if success:
                placed_boxes.append(box)
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2 / aw
                yc = (y1 + y2) / 2 / ah
                w = (x2 - x1) / aw
                h = (y2 - y1) / ah
                class_idx = classes.index(cls)
                labels.append(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not labels:
            continue  # nothing successfully placed on this background, skip

        out_name = f"img_{i:05d}"
        arena_img.convert("RGB").save(images_out / f"{out_name}.jpg", quality=JPEG_QUALITY)
        with open(labels_out / f"{out_name}.txt", "w") as f:
            f.write("\n".join(labels))
        generated += 1

        if generated % 200 == 0:
            print(f"Generated {generated} images so far...")

    print(f"Done. Generated {generated} labeled images in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    generate_dataset()