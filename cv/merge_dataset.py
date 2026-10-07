import shutil
from pathlib import Path

SPLIT = Path("datasets/merged_v2/merged_split")
SYN = SPLIT / "synthetic" / "train"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

n = 0
for img in sorted((SYN / "images").iterdir()):
    if img.suffix.lower() not in EXTS:
        continue
    lbl = SYN / "labels" / f"{img.stem}.txt"
    if not lbl.exists():
        print("no label, skipping:", img.name); continue
    new = f"synth_{img.stem}"
    dst_img = SPLIT / "images" / "train" / f"{new}{img.suffix}"
    if dst_img.exists():
        print("already exists, skipping:", dst_img.name); continue
    shutil.copy(img, dst_img)
    shutil.copy(lbl, SPLIT / "labels" / "train" / f"{new}.txt")
    n += 1
print("copied", n)

ids = set()
for f in (SPLIT / "labels" / "train").glob("synth_*.txt"):
    ids |= {int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()}
print("class ids in synthetic labels:", sorted(ids))   # expect only 0-6