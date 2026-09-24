"""
compare_models.py - run the old and new model on held-out frames and compare.

Edit the CONFIG block, then:  python compare_models.py

Prints per-class counts at your deployed detector settings:
    ok      = right class, box overlaps your label (IoU > 0.5)
    wrong   = box on your object but the wrong class (e.g. lime called capsicum)
    missed  = no box on your object
    extra   = a box that matches no label (includes a second-class duplicate on one fruit)
Saves old|new side-by-side images to OUT_DIR (white boxes = your labels).
If a frame has no label file it is still saved side-by-side, just not scored.
"""
import glob
import os

import cv2
from ultralytics import YOLO

# ============================ CONFIG ============================
OLD = "model/old_best.pt"           # model the robot currently runs
NEW = "model/new_best.pt"           # downloaded from Colab
IMG_DIR = "new_pics/images"            # held-out frames (NOT used in training)
LBL_DIR = "new_pics/labels"            # their YOLO labels from label_tool.py
OUT_DIR = "compare_out"
CONF = 0.75                            # match detector.py
IOU = 0.5                              # match detector.py
IMGSZ = 480                            # match detector.py
# ================================================================

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def load_gt(path, W, H):
    gt = []
    for line in open(path):
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        cx, cy, w, h = float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H
        gt.append((c, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return gt


def score(st, names, gt, preds):
    for c, box in gt:
        s = st[names[c]]
        s["gt"] += 1
        cands = [p for p in preds if iou(box, p["box"]) > 0.5]
        same = [p for p in cands if p["cls"] == c and not p["used"]]
        if same:
            max(same, key=lambda p: iou(box, p["box"]))["used"] = True
            s["ok"] += 1
        elif cands:
            max(cands, key=lambda p: iou(box, p["box"]))["used"] = True
            s["wrong"] += 1
        else:
            s["missed"] += 1
    for p in preds:
        if not p["used"]:
            st[names[p["cls"]]]["extra"] += 1


def main():
    models = {"old": YOLO(OLD), "new": YOLO(NEW)}
    names = models["new"].names
    stats = {tag: {n: dict(gt=0, ok=0, wrong=0, missed=0, extra=0) for n in names.values()}
             for tag in models}
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(IMG_DIR, "*")) if f.lower().endswith(EXTS))
    if not files:
        print(f"No images in {IMG_DIR}")
        return

    scored = 0
    for f in files:
        img = cv2.imread(f)  # BGR, as YOLO expects
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(f))[0]
        lp = os.path.join(LBL_DIR, stem + ".txt")
        gt = load_gt(lp, W, H) if os.path.exists(lp) else None
        scored += gt is not None

        panels = []
        for tag, m in models.items():
            r = m.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
            preds = [dict(cls=int(b.cls[0]), box=b.xyxy[0].tolist(), used=False) for b in r.boxes]
            if gt is not None:
                score(stats[tag], names, gt, preds)
            vis = r.plot()
            if gt is not None:
                for _, (x1, y1, x2, y2) in gt:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
            cv2.putText(vis, tag, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panels.append(vis)
        cv2.imwrite(os.path.join(OUT_DIR, stem + ".jpg"), cv2.hconcat(panels))

    print(f"\n{len(files)} frames ({scored} with labels) at conf={CONF} iou={IOU} imgsz={IMGSZ}")
    for tag in models:
        print(f"\n=== {tag}")
        print(f"{'class':<11}{'gt':>4}{'ok':>5}{'wrong':>7}{'missed':>8}{'extra':>7}")
        for n, s in stats[tag].items():
            if s["gt"] or s["extra"]:
                print(f"{n:<11}{s['gt']:>4}{s['ok']:>5}{s['wrong']:>7}{s['missed']:>8}{s['extra']:>7}")
    print(f"\nSide-by-side images: {OUT_DIR}/")


if __name__ == "__main__":
    main()