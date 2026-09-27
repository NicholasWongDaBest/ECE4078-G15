"""
merge_labeled_batches.py - merge multiple label_tool.py output folders (each
with images/ + labels/, each independently starting img_0) into one combined
labeled pool, renaming to avoid collisions while keeping image/label pairs
matched.

Edit BATCHES below, then: python merge_labeled_batches.py
"""
import shutil
from pathlib import Path

# ============================ CONFIG ============================
# (prefix, source_dir) - source_dir must contain images/ and labels/
BATCHES = [
    ("batch1", "datasets/real/real_dataset"),   # your original ~102 images
    ("batch2", "datasets/half/half_dataset"),       # your new ~40 partial-occlusion images
]
OUT_DIR = "datasets/merged/merged_dataset"
# ================================================================

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    out_img = Path(OUT_DIR) / "images"
    out_lbl = Path(OUT_DIR) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    for prefix, src in BATCHES:
        src_img = Path(src) / "images"
        src_lbl = Path(src) / "labels"
        files = sorted(f for f in src_img.iterdir() if f.suffix.lower() in EXTS)
        for f in files:
            new_stem = f"{prefix}_{f.stem}"          # e.g. batch1_img_0
            shutil.copy(f, out_img / f"{new_stem}{f.suffix}")
            lp = src_lbl / (f.stem + ".txt")
            if lp.exists():
                shutil.copy(lp, out_lbl / f"{new_stem}.txt")
            else:
                print(f"WARNING: no label for {f.name} in {src}")
            total += 1
        print(f"{prefix}: {len(files)} images copied from {src}")

    print(f"Merged {total} images into {OUT_DIR}")


if __name__ == "__main__":
    main()