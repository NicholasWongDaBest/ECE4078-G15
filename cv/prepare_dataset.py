"""
prepare_dataset.py - split label_tool output into train/val and write data.yaml.

Edit the CONFIG block, then:  python prepare_dataset.py

Splits in blocks of consecutive frames (BLOCK) so near-duplicate neighbours don't
end up in both train and val. Class names are read from WEIGHTS so IDs match the model.
Output layout (upload the whole OUT_DIR folder to Colab):
    real_split/images/{train,val}
    real_split/labels/{train,val}
    real_split/synthetic/train/{images,labels}   (if SYNTH_SRC is set)
    real_split/data.yaml
"""
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# ============================ CONFIG ============================
LABELED_DIR = "real_dataset"           # OUT_DIR from label_tool.py
OUT_DIR = "real_split"
WEIGHTS = "model/best.pt"        # for class names
VAL_FRACTION = 0.2
BLOCK = 10                             # consecutive frames kept together
SEED = 0
SYNTH_SRC = "dataset_split/train"      # local synthetic train folder (must contain images/ and labels/);
                                       # copied into OUT_DIR/synthetic/train. "" = real only
# ================================================================

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    src_img = Path(LABELED_DIR) / "images"
    src_lbl = Path(LABELED_DIR) / "labels"
    files = sorted(f for f in src_img.iterdir() if f.suffix.lower() in EXTS)
    if not files:
        print("No labeled images found.")
        return

    blocks = [files[i:i + BLOCK] for i in range(0, len(files), BLOCK)]
    random.Random(SEED).shuffle(blocks)
    n_val = max(1, round(len(blocks) * VAL_FRACTION))
    split = {"val": blocks[:n_val], "train": blocks[n_val:]}

    out = Path(OUT_DIR)
    if out.exists():
        shutil.rmtree(out)
    for name, blks in split.items():
        (out / "images" / name).mkdir(parents=True)
        (out / "labels" / name).mkdir(parents=True)
        n = 0
        for blk in blks:
            for f in blk:
                shutil.copy(f, out / "images" / name / f.name)
                lp = src_lbl / (f.stem + ".txt")
                if lp.exists():
                    shutil.copy(lp, out / "labels" / name / lp.name)
                n += 1
        print(f"{name}: {n} images")

    train_entry = ["images/train"]
    if SYNTH_SRC:
        src = Path(SYNTH_SRC)
        if not (src / "images").is_dir() or not (src / "labels").is_dir():
            print(f"SYNTH_SRC '{src}' needs images/ and labels/ subfolders - skipping synthetic.")
        else:
            shutil.copytree(src, out / "synthetic" / "train")
            train_entry.append("synthetic/train/images")
            print(f"Copied synthetic data from {src}")

    names = YOLO(WEIGHTS).names
    lines = [f"train: {json.dumps(train_entry)}",
             "val: images/val",
             f"nc: {len(names)}",
             "names:"]
    lines += [f"  {i}: {json.dumps(n)}" for i, n in sorted(names.items())]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'data.yaml'}  (classes: {dict(names)})")


if __name__ == "__main__":
    main()