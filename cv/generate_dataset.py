"""
generate_dataset.py

Generates a large synthetic training dataset for YOLOv8 by superimposing
background-removed fruit cutouts onto arena background photos, at random
scale/position/rotation. Since we control the paste location, YOLO-format
bounding box labels are written out automatically -- no manual annotation
needed for the base dataset (Roboflow is still useful afterward for
spot-checking / cleaning up edge cases, or for augmentation).

Expected folder structure (run remove_background.py first to create the cutout folders):

  images/fruits_cutout/
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

  images/arena/
    arena_01.jpg
    arena_02.jpg
    ...

  images/landmark_cutout/          <- OPTIONAL, enables occlusion augmentation once populated
    aruco_cube_01.png              <- RGBA, transparent background
    ...

  images/distractors_cutout/       <- OPTIONAL, enables color-confusion mitigation once populated
    red_distractor_01.png          <- RGBA, transparent background, non-fruit objects
    ...

Output:

  dataset/
    images/
      img_00000.jpg
      img_00001.jpg
      ...
    labels/
      img_00000.txt     <- YOLO format: "class_idx x_center y_center width height" (normalized 0-1)
      img_00001.txt      <- can be EMPTY for hard-negative background-only images
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
FRUIT_DIR = "images/fruits_cutout"            # input: folders of transparent-background fruit PNGs
ARENA_DIR = "images/arena"                    # input: arena background photos
LANDMARK_DIR = "images/landmark_cutout"       # OPTIONAL: enables occlusion augmentation once populated
DISTRACTOR_DIR = "images/distractors_cutout"  # OPTIONAL: enables color-distractor clutter once populated
OUTPUT_DIR = "dataset"                        # output: images/ + labels/ + classes.txt

NUM_IMAGES = 3000                # total synthetic images to generate (fruit-containing)
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
DISTANCE_MAX_M = 0.75   # farthest realistic distance -- tune both based on your arena size / camera FOV
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
SHADOW_OPACITY = 80         # 0-255 alpha of the shadow at its darkest point
SHADOW_WIDTH_FRAC = 0.75    # shadow ellipse width, as a fraction of the fruit's rendered width
SHADOW_HEIGHT_FRAC = 0.10   # shadow ellipse height, as a fraction of the fruit's rendered width
SHADOW_BLUR_FRAC = 0.06     # blur radius as a fraction of fruit width -- scales down for small
                             # fruits so blur doesn't overwhelm tiny objects (e.g. lemon, lime)
SHADOW_BLUR_MIN = 1
SHADOW_BLUR_MAX = 5

# --- Floor reflection (glossy arena floor mirrors the fruit -- without this the
# model has never seen a reflection during training, so it fires a second,
# lower-confidence "fruit" detection on the mirrored blob at inference time) ---
REFLECTION_ENABLED = True
REFLECTION_OPACITY = 0.28     # 0-1, overall strength of the reflection vs the real fruit
REFLECTION_BLUR_FRAC = 0.05   # blur radius as a fraction of fruit width
REFLECTION_BLUR_MIN = 1
REFLECTION_BLUR_MAX = 6
REFLECTION_FADE_RATIO = 0.85  # how much the reflection fades out toward its bottom edge (0-1)
REFLECTION_HEIGHT_FRAC = 0.8  # reflection rendered height as a fraction of the fruit's own height
                               # (real reflections on a short floor gap fade out before a full mirror)
# NOTE: the reflection is composited but deliberately NEVER added to the label file --
# that's what teaches the model "blob below the fruit" is not a second object.

# --- Occlusion (landmark partially covering a fruit) ---
# Without this, the model has never seen a fruit split into two visible blobs by
# something in front of it, so at inference it treats each blob as a separate
# instance of the same fruit. Requires LANDMARK_DIR to contain sprites -- if that
# folder is missing/empty this feature is silently skipped (a warning is printed once).
OCCLUSION_ENABLED = True
OCCLUSION_CHANCE = 0.3        # probability, per successfully-placed fruit, that it gets occluded
OCCLUSION_OVERLAP_MIN = 0.2   # occluder covers between 20-50% of the fruit's rendered width
OCCLUSION_OVERLAP_MAX = 0.5
OCCLUSION_SCALE_MIN = 0.6     # occluder size relative to the fruit's rendered width
OCCLUSION_SCALE_MAX = 1.1

# --- Hard-negative / distractor backgrounds ---
# Pure background scenes (arena + landmarks + shadows + reflections, but NO fruit)
# with empty label files, so the model learns shadows/reflections/clutter alone
# are not a fruit class. Optionally sprinkle in same-colored non-fruit distractor
# objects (DISTRACTOR_DIR) as unlabeled clutter, which directly discourages the
# model from using color alone as a shortcut feature. Both landmark and distractor
# scattering here are optional -- if the folders are empty, negatives are still
# generated (just plain arena backgrounds only).
HARD_NEGATIVE_FRAC = 0.12     # fraction of NUM_IMAGES generated as additional pure-negative scenes
DISTRACTOR_CHANCE = 0.5       # probability a hard-negative scene also gets 1-2 distractor objects
MIN_DISTRACTORS = 1
MAX_DISTRACTORS = 2

MAX_ROTATION_DEG = 25            # random in-plane rotation applied to each fruit
MAX_OVERLAP_IOU = 0.15           # reject placements that overlap existing fruit boxes more than this
MAX_PLACEMENT_TRIES = 20         # tries per fruit before giving up on that one

# Some rembg cutouts retain a translucent "tail" of the object's own cast shadow from
# the original photo (rembg often treats a soft shadow gradient as semi-transparent
# foreground rather than fully removing it). A low threshold only catches faint noise;
# an actual retained shadow can have moderate alpha (50-150), so the threshold needs to
# be high enough that only solidly-opaque fruit pixels count. If boxes/shadows still
# look oversized for a specific fruit, raise this further (try 150-200) or inspect that
# fruit's cutout PNGs directly with check_cutouts.py.
ALPHA_THRESHOLD = 160

BRIGHTNESS_JITTER = 0.15         # +/- 15% random brightness scaling per fruit paste

# --- Exposure/color matching (closes sim-to-real domain gap) ---
# Nudges each fruit sprite's color statistics toward the local background patch
# it's about to be pasted onto, so it doesn't look "pasted on" under different
# lighting/color-temperature than the arena scene. Applied BEFORE random_brightness.
COLOR_MATCH_ENABLED = True
COLOR_MATCH_STRENGTH = 0.4       # 0 = no adjustment, 1 = fully match background color mean

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


def load_sprite_folder(dir_path):
    """Generic loader for a flat folder of RGBA cutout PNGs (landmarks, distractors).
    Returns [] if the folder doesn't exist or is empty -- callers should treat that
    as 'feature disabled' rather than an error."""
    if not os.path.isdir(dir_path):
        return []
    paths = glob.glob(os.path.join(dir_path, "*.png"))
    return [Image.open(p).convert("RGBA") for p in paths]


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


def match_color_to_background(fruit_img, arena_img, paste_x, paste_y, target_w, target_h,
                               strength=COLOR_MATCH_STRENGTH):
    """Nudges fruit_img's RGB color mean toward the color mean of the arena background
    patch it's about to be pasted onto, so it blends into that scene's lighting/color
    temperature instead of carrying over its original photo's lighting unchanged.

    fruit_img: RGBA fruit sprite (already resized, pre-rotation)
    arena_img: RGBA arena background
    paste_x, paste_y, target_w, target_h: where the fruit will land on arena_img
    strength: 0 = no change, 1 = fully match background mean (0.3-0.5 is usually plenty --
              too high starts washing out the fruit's own natural color identity)
    """
    aw, ah = arena_img.size
    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(aw, paste_x + target_w)
    y2 = min(ah, paste_y + target_h)
    if x2 <= x1 or y2 <= y1:
        return fruit_img  # region off-canvas, nothing to sample -- skip

    bg_patch = np.array(arena_img.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32)
    bg_mean = bg_patch.reshape(-1, 3).mean(axis=0)

    fruit_arr = np.array(fruit_img, dtype=np.float32)
    alpha_mask = fruit_arr[..., 3] > ALPHA_THRESHOLD
    if not alpha_mask.any():
        return fruit_img  # fully transparent, nothing to adjust

    fruit_rgb = fruit_arr[..., :3]
    fruit_mean = fruit_rgb[alpha_mask].mean(axis=0)

    shift = (bg_mean - fruit_mean) * strength
    fruit_arr[..., :3] = np.clip(fruit_rgb + shift, 0, 255)

    return Image.fromarray(fruit_arr.astype(np.uint8), mode="RGBA")


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
    shadow_h = max(2, int(fruit_width * SHADOW_HEIGHT_FRAC))
    blur_radius = int(min(max(fruit_width * SHADOW_BLUR_FRAC, SHADOW_BLUR_MIN), SHADOW_BLUR_MAX))

    shadow_layer = Image.new("RGBA", arena_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_layer)
    x0 = center_x - shadow_w // 2
    y0 = bottom_y - shadow_h // 2
    x1 = center_x + shadow_w // 2
    y1 = bottom_y + shadow_h // 2
    draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, SHADOW_OPACITY))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arena_img.alpha_composite(shadow_layer)


def draw_reflection(arena_img, fruit_sprite, x, bottom_y, fruit_width):
    """Composites a faded, blurred, vertically-flipped copy of the fruit sprite
    directly below its ground-contact point, mimicking the glossy arena floor.

    IMPORTANT: this is purely visual -- it must NEVER be added to the YOLO label
    file. Training the model on unlabeled reflections is what teaches it that a
    mirrored blob below a fruit isn't a second object.
    """
    reflection = fruit_sprite.transpose(Image.FLIP_TOP_BOTTOM)

    # Crop the reflection's height so it fades out before becoming a full mirror
    # (a real floor reflection off a small gloss gap doesn't fully replicate the object).
    rw, rh = reflection.size
    crop_h = max(1, int(rh * REFLECTION_HEIGHT_FRAC))
    reflection = reflection.crop((0, 0, rw, crop_h))
    rw, rh = reflection.size

    # Lower overall opacity
    alpha = reflection.split()[-1]
    alpha = alpha.point(lambda a: int(a * REFLECTION_OPACITY))

    # Vertical gradient: strongest right at the fruit's base, fading toward the bottom
    grad = Image.new("L", (rw, rh))
    grad_pixels = grad.load()
    for row in range(rh):
        fade = max(0, 255 - int(255 * REFLECTION_FADE_RATIO * (row / max(1, rh - 1))))
        for col in range(rw):
            grad_pixels[col, row] = fade
    alpha = Image.composite(alpha, Image.new("L", alpha.size, 0), grad)
    reflection.putalpha(alpha)

    blur_radius = int(min(max(fruit_width * REFLECTION_BLUR_FRAC, REFLECTION_BLUR_MIN),
                           REFLECTION_BLUR_MAX))
    reflection = reflection.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    aw, ah = arena_img.size
    paste_y = bottom_y
    if paste_y >= ah:
        return  # nothing visible would be pasted
    # Clip if the reflection would run past the bottom edge of the image
    visible_h = min(rh, ah - paste_y)
    if visible_h <= 0:
        return
    if visible_h < rh:
        reflection = reflection.crop((0, 0, rw, visible_h))

    arena_img.alpha_composite(reflection, (x, paste_y))


def maybe_occlude(arena_img, fruit_box, landmark_images):
    """With probability OCCLUSION_CHANCE, pastes a landmark sprite over part of the
    fruit's rendered region. The fruit's YOLO label bbox is deliberately left
    UNCHANGED (it's passed in already computed) -- this teaches the model that an
    object partly covered by a landmark is still one single instance, not two."""
    if not landmark_images or random.random() > OCCLUSION_CHANCE:
        return

    x1, y1, x2, y2 = fruit_box
    fruit_w = x2 - x1
    fruit_h = y2 - y1
    if fruit_w <= 0 or fruit_h <= 0:
        return

    landmark = random.choice(landmark_images)
    lw, lh = landmark.size
    if lw == 0 or lh == 0:
        return

    scale = random.uniform(OCCLUSION_SCALE_MIN, OCCLUSION_SCALE_MAX)
    target_w = max(4, int(fruit_w * scale))
    target_h = max(4, int(lh * (target_w / lw)))
    landmark_resized = landmark.resize((target_w, target_h), Image.LANCZOS)

    overlap_frac = random.uniform(OCCLUSION_OVERLAP_MIN, OCCLUSION_OVERLAP_MAX)
    # Place so it overlaps the fruit's left or right side by overlap_frac of fruit width
    if random.random() < 0.5:
        land_x = int(x1 - target_w * (1 - overlap_frac))
    else:
        land_x = int(x2 - target_w * overlap_frac)
    land_y = y1 + random.randint(0, max(1, fruit_h // 2)) - target_h // 2

    aw, ah = arena_img.size
    land_x = min(max(land_x, 0), max(0, aw - target_w))
    land_y = min(max(land_y, 0), max(0, ah - target_h))

    arena_img.paste(landmark_resized, (land_x, land_y), landmark_resized)
    # NOTE: fruit_box is intentionally not modified/re-tightened here -- the label
    # keeps the fruit's full original extent, occluded portion included.


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
      - A soft drop shadow and a soft floor reflection are rendered under the fruit
        so it reads as sitting on the glossy floor instead of floating.
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

        if COLOR_MATCH_ENABLED:
            # Estimate landing position before rotation just to sample the right
            # background patch -- doesn't need to be exact, only close enough to grab
            # locally-representative background color/lighting.
            if physical_mode:
                t_est = (distance - DISTANCE_MIN_M) / (DISTANCE_MAX_M - DISTANCE_MIN_M)
                t_est = min(max(t_est, 0.0), 1.0)
                est_bottom_y = FLOOR_BOTTOM_FRAC * ah - t_est * (FLOOR_BOTTOM_FRAC - FLOOR_TOP_FRAC) * ah
                est_y = int(min(max(est_bottom_y - target_h, 0), ah - target_h))
            else:
                est_y = max(0, (ah - target_h) // 2)
            est_x = max(0, (aw - target_w) // 2)
            resized = match_color_to_background(resized, arena_img, est_x, est_y, target_w, target_h)

        angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        rotated = resized.rotate(angle, expand=True)

        # rotate(expand=True) pads the canvas with transparent pixels so the
        # rotated shape fits -- crop tightly to the actual visible pixels.
        # Threshold the alpha first: faint near-zero-alpha artifacts from imperfect
        # background removal would otherwise get included in getbbox() and inflate
        # the box far beyond the actual fruit.
        alpha = rotated.split()[-1]
        mask = alpha.point(lambda a: 255 if a > ALPHA_THRESHOLD else 0)
        alpha_bbox = mask.getbbox()
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
            if physical_mode and REFLECTION_ENABLED:
                draw_reflection(arena_img, rotated, x=x, bottom_y=y + rh, fruit_width=rw)
            return True, box

    return False, None


def generate_negative_scene(out_name, images_out, labels_out, arena_paths,
                             landmark_images, distractor_images):
    """Generates one pure-background image (no fruit) with an EMPTY label file.
    Optionally scatters 1-2 landmark sprites and/or same-colored distractor
    objects as unlabeled clutter, so the model learns these are never a fruit
    class regardless of shape/color/shadow/reflection."""
    arena_path = random.choice(arena_paths)
    arena_img = Image.open(arena_path).convert("RGBA")
    aw, ah = arena_img.size

    # Scatter a landmark or two (unlabeled) so the arena isn't suspiciously empty
    if landmark_images and random.random() < 0.7:
        for _ in range(random.randint(1, 2)):
            landmark = random.choice(landmark_images)
            lw, lh = landmark.size
            if lw == 0:
                continue
            scale = random.uniform(0.08, 0.18)
            target_w = int(aw * scale)
            target_h = int(lh * (target_w / lw))
            if target_w < 5 or target_h < 5:
                continue
            resized = landmark.resize((target_w, target_h), Image.LANCZOS)
            x = random.randint(0, max(0, aw - target_w))
            y = random.randint(int(ah * 0.4), max(int(ah * 0.4), ah - target_h))
            arena_img.paste(resized, (x, y), resized)

    # Scatter same-colored non-fruit distractors (unlabeled) -- directly discourages
    # the model from using color alone as a shortcut feature for fruit classes.
    if distractor_images and random.random() < DISTRACTOR_CHANCE:
        for _ in range(random.randint(MIN_DISTRACTORS, MAX_DISTRACTORS)):
            distractor = random.choice(distractor_images)
            dw, dh = distractor.size
            if dw == 0:
                continue
            scale = random.uniform(0.06, 0.2)
            target_w = int(aw * scale)
            target_h = int(dh * (target_w / dw))
            if target_w < 5 or target_h < 5:
                continue
            resized = distractor.resize((target_w, target_h), Image.LANCZOS)
            x = random.randint(0, max(0, aw - target_w))
            y = random.randint(int(ah * 0.3), max(int(ah * 0.3), ah - target_h))
            arena_img.paste(resized, (x, y), resized)

    arena_img.convert("RGB").save(images_out / f"{out_name}.jpg", quality=JPEG_QUALITY)
    (labels_out / f"{out_name}.txt").write_text("")  # empty label file = background


def generate_dataset():
    classes = load_classes(FRUIT_DIR)
    print(f"Found {len(classes)} classes: {classes}")

    fruit_images = load_fruit_images(FRUIT_DIR, classes)
    arena_paths = load_arena_images(ARENA_DIR)
    print(f"Found {len(arena_paths)} arena background images")

    landmark_images = load_sprite_folder(LANDMARK_DIR) if OCCLUSION_ENABLED else []
    if OCCLUSION_ENABLED and not landmark_images:
        print(f"NOTE: no sprites found in '{LANDMARK_DIR}/' -- occlusion augmentation will be "
              f"skipped for now. Add RGBA landmark cutout PNGs there later to enable it.")

    distractor_images = load_sprite_folder(DISTRACTOR_DIR)
    if not distractor_images:
        print(f"NOTE: no sprites found in '{DISTRACTOR_DIR}/' -- color-distractor clutter "
              f"will be skipped for now. Add RGBA non-fruit cutout PNGs there later to enable it.")

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
                if OCCLUSION_ENABLED:
                    maybe_occlude(arena_img, box, landmark_images)
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

    print(f"Done. Generated {generated} labeled (fruit-containing) images in '{OUTPUT_DIR}/'.")

    # --- Hard-negative background scenes ---
    num_negatives = int(NUM_IMAGES * HARD_NEGATIVE_FRAC)
    print(f"Generating {num_negatives} hard-negative background scenes (no fruit, empty labels)...")
    for j in range(num_negatives):
        out_name = f"img_neg_{j:05d}"
        generate_negative_scene(out_name, images_out, labels_out, arena_paths,
                                 landmark_images, distractor_images)
        if (j + 1) % 100 == 0:
            print(f"Generated {j + 1} negative scenes so far...")

    print(f"Done. Generated {num_negatives} hard-negative images in '{OUTPUT_DIR}/'.")
    print(f"Total dataset size: {generated + num_negatives} images.")


if __name__ == "__main__":
    generate_dataset()