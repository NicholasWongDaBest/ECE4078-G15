"""
remove_background.py

Removes the background from your raw fruit photos, producing transparent-background
RGBA PNGs ready to be superimposed onto arena images (see generate_dataset.py).

Two methods available:
  1. rembg   -- AI-based background removal. Works even with imperfect/non-uniform
               backgrounds. Recommended. Requires: pip install rembg
  2. chroma  -- Simple color-threshold removal. Works well if you photographed the
               fruit on a plain, uniformly-colored background (e.g. white paper,
               green screen). No extra dependencies beyond Pillow/numpy.

Usage:
  python remove_background.py --method rembg  --input raw_fruits/apple --output fruit_cutouts/apple
  python remove_background.py --method chroma --input raw_fruits/apple --output fruit_cutouts/apple --bg_color white

Run this once per fruit folder (apple, orange, capsicum, etc.).
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image


def remove_bg_rembg(input_dir, output_dir):
    from rembg import remove  # pip install rembg

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    if not paths:
        print(f"No images found in {input_dir}")
        return

    for p in paths:
        img = Image.open(p)
        out = remove(img)  # returns RGBA with background removed
        out_name = Path(p).stem + ".png"
        out.save(Path(output_dir) / out_name)
        print(f"Processed {p} -> {output_dir}/{out_name}")


def remove_bg_chroma(input_dir, output_dir, bg_color="white", tolerance=30):
    """Simple distance-from-background-color threshold. Good for uniform backgrounds."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    if not paths:
        print(f"No images found in {input_dir}")
        return

    bg_rgb = {
        "white": (255, 255, 255),
        "green": (0, 255, 0),
        "black": (0, 0, 0),
    }.get(bg_color, (255, 255, 255))

    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img).astype(np.int16)
        diff = np.sqrt(((arr - np.array(bg_rgb)) ** 2).sum(axis=-1))
        mask = diff > tolerance  # True = keep as foreground

        rgba = np.dstack([arr.astype(np.uint8), (mask * 255).astype(np.uint8)])
        out_img = Image.fromarray(rgba, mode="RGBA")

        out_name = Path(p).stem + ".png"
        out_img.save(Path(output_dir) / out_name)
        print(f"Processed {p} -> {output_dir}/{out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from fruit photos.")
    parser.add_argument("--method", choices=["rembg", "chroma"], required=True)
    parser.add_argument("--input", required=True, help="Folder of raw fruit photos")
    parser.add_argument("--output", required=True, help="Folder to save transparent PNGs")
    parser.add_argument("--bg_color", default="white", choices=["white", "green", "black"],
                         help="Only used for --method chroma")
    parser.add_argument("--tolerance", type=int, default=30,
                         help="Only used for --method chroma. Higher = more aggressive removal")
    args = parser.parse_args()

    if args.method == "rembg":
        remove_bg_rembg(args.input, args.output)
    else:
        remove_bg_chroma(args.input, args.output, args.bg_color, args.tolerance)