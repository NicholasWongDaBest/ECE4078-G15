"""
prelabel_markers.py - auto pre-label the marker_partial photos before hand-correcting
them in label_tool.py.

For each image it writes a YOLO label file containing:
  * fruit boxes predicted by the current YOLO weights (same as label_tool.py would), and
  * one box per marker cube that ArUco can decode (faces with the same ID are merged,
    and the box is padded out to cover the cube's white border).

Markers that are cut off / occluded can't be decoded by ArUco, so those are the only
boxes left to draw by hand in label_tool.py.

label_tool.py loads an existing label file instead of re-predicting, so run this first,
then open label_tool.py with the same OUT_DIR. Existing label files are never overwritten,
so it's safe to re-run after you've started correcting.

Usage (from the cv/ folder):   python prelabel_markers.py
"""
from pathlib import Path

import cv2
import numpy as np

# ============================ CONFIG ============================
IMAGES_DIR = "images/marker"     # same as label_tool.py
OUT_DIR = "marker/session1"      # same as label_tool.py
WEIGHTS = "model/new_new_best.pt"        # fruit pre-labels; set to "" to skip fruit
CONF = 0.5
IMGSZ = 480
MARKER_CLASS = 7                         # "marker_partial" in merged_v2 data.yaml
LABEL_FULL_MARKERS = True                # also label fully-visible (decodable) markers
ARUCO_DICT = cv2.aruco.DICT_4X4_100      # what the arena markers decode with
PAD = 0.12                               # grow the black-square box by this fraction per side
                                         # so it covers the cube's white border
OVERWRITE = False                        # never clobber labels you've already corrected
# ================================================================

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def aruco_boxes(img, detector):
    """One [x1, y1, x2, y2] box per marker ID (all visible faces of a cube merged)."""
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return []
    h, w = img.shape[:2]
    by_id = {}
    for mid, c in zip(ids.ravel(), corners):
        pts = c[0]
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        # pad each face individually (pad scales with that face's size)
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        box = [x1 - px, y1 - py, x2 + px, y2 + py]
        if mid in by_id:
            b = by_id[mid]
            box = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]
        by_id[mid] = box
    return [[float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
             float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h))]
            for x1, y1, x2, y2 in by_id.values()]


def fruit_boxes(model, img):
    if model is None:
        return []
    res = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    out = []
    for b in res.boxes:
        c = int(b.cls[0])
        if c == MARKER_CLASS:          # in case a later model already knows markers
            continue
        out.append([c] + b.xyxy[0].tolist())
    return out


def to_yolo(c, x1, y1, x2, y2, w, h):
    return (f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def main():
    model = None
    if WEIGHTS:
        from ultralytics import YOLO
        model = YOLO(WEIGHTS)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT))
    out_lbl = Path(OUT_DIR) / "labels"
    out_lbl.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in Path(IMAGES_DIR).iterdir() if f.suffix.lower() in EXTS)
    n_written = n_skipped = n_markers = n_fruit = 0
    no_marker = []
    for f in files:
        lbl = out_lbl / (f.stem + ".txt")
        if lbl.exists() and not OVERWRITE:
            n_skipped += 1
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = [to_yolo(*b, w, h) for b in fruit_boxes(model, img)]
        n_fruit += len(lines)
        markers = aruco_boxes(img, detector) if LABEL_FULL_MARKERS else []
        lines += [to_yolo(MARKER_CLASS, *b, w, h) for b in markers]
        n_markers += len(markers)
        if not markers:
            no_marker.append(f.name)

        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_written += 1

    print(f"{n_written} label files written, {n_skipped} skipped (already existed)")
    print(f"  {n_markers} decodable marker cubes, {n_fruit} fruit boxes pre-labelled")
    print(f"  {len(no_marker)} images had no decodable marker (all-partial -> draw by hand)")
    print(f"Now run label_tool.py (OUT_DIR = {OUT_DIR}) and draw the partial markers as class {MARKER_CLASS}.")


if __name__ == "__main__":
    main()
