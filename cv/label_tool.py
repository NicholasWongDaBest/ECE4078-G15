"""
label_tool.py - pre-label with current YOLO weights, correct by hand, save YOLO txt labels.

Usage:
    Edit the CONFIG block below, then:  python label_tool.py

Output:
    real_dataset/images/<name>.jpg
    real_dataset/labels/<name>.txt   (class cx cy w h, normalised)

Mouse:
    left-drag         draw new box (uses current class)
    left-click        select box under cursor
Keys:
    0-9               set current class; if a box is selected, change its class
    c                 cycle selected box's class (quick lime <-> capsicum swap)
    d / Delete        delete selected box
    r                 reset image to model predictions
    n / Space         save + next
    b                 save + back
    q / Esc           save + quit
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================ CONFIG ============================
IMAGES_DIR = "images/fruits_real"              # folder of raw (unannotated) frames to label
WEIGHTS = "model/new_best.pt"        # current model, used for pre-labels
OUT_DIR = "new_pics"               # writes OUT_DIR/images and OUT_DIR/labels
CONF = 0.25                            # pre-label confidence (low on purpose)
IMGSZ = 480                            # match detector.py inference size
SCALE = 1.5                            # display scale
# ================================================================

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PALETTE = [(0, 200, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
           (255, 255, 0), (128, 0, 255), (0, 0, 255), (128, 128, 0), (0, 128, 128)]
DELETE_KEYS = {ord("d"), 3014656, 127, 46}


def load_labels(path, w, h):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = map(float, p[1:])
            boxes.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def save_labels(path, boxes, w, h):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def predict(model, img_bgr, conf, imgsz):
    # img_bgr straight from cv2.imread -> BGR, which is what YOLO expects
    res = model.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append([int(b.cls[0]), x1, y1, x2, y2])
    return boxes


def main():
    model = YOLO(WEIGHTS)
    names = model.names  # {id: name}
    nc = len(names)
    print("Classes:", names)

    out_img = Path(OUT_DIR) / "images"
    out_lbl = Path(OUT_DIR) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in Path(IMAGES_DIR).iterdir() if f.suffix.lower() in EXTS)
    if not files:
        print("No images found.")
        return

    S = SCALE
    st = {"boxes": [], "sel": -1, "cls": 0, "start": None, "cur": None}

    def box_at(x, y):
        hits = [(i, (b[3] - b[1]) * (b[4] - b[2])) for i, b in enumerate(st["boxes"])
                if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
        return min(hits, key=lambda t: t[1])[0] if hits else -1

    def on_mouse(event, mx, my, flags, param):
        x, y = mx / S, my / S
        if event == cv2.EVENT_LBUTTONDOWN:
            st["start"] = (x, y)
            st["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["start"] is not None:
            st["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["start"] is not None:
            x0, y0 = st["start"]
            st["start"] = st["cur"] = None
            if abs(x - x0) * S < 5 and abs(y - y0) * S < 5:
                st["sel"] = box_at(x, y)
            else:
                st["boxes"].append([st["cls"], min(x0, x), min(y0, y), max(x0, x), max(y0, y)])
                st["sel"] = len(st["boxes"]) - 1

    win = "label_tool"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    idx = 0
    while 0 <= idx < len(files):
        f = files[idx]
        img = cv2.imread(str(f))
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        lbl_path = out_lbl / (f.stem + ".txt")
        st["boxes"] = load_labels(lbl_path, w, h) if lbl_path.exists() else predict(model, img, CONF, IMGSZ)
        st["sel"] = -1
        step = 0

        while True:
            disp = cv2.resize(img, None, fx=S, fy=S)
            for i, (c, x1, y1, x2, y2) in enumerate(st["boxes"]):
                col = PALETTE[c % len(PALETTE)]
                th = 3 if i == st["sel"] else 1
                p1, p2 = (int(x1 * S), int(y1 * S)), (int(x2 * S), int(y2 * S))
                cv2.rectangle(disp, p1, p2, col, th)
                cv2.putText(disp, f"{c}:{names.get(c, '?')}", (p1[0], max(12, p1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            if st["start"] is not None and st["cur"] is not None:
                cv2.rectangle(disp, (int(st["start"][0] * S), int(st["start"][1] * S)),
                              (int(st["cur"][0] * S), int(st["cur"][1] * S)),
                              PALETTE[st["cls"] % len(PALETTE)], 1)
            bar = f"[{idx + 1}/{len(files)}] {f.name} | draw class {st['cls']}:{names.get(st['cls'], '?')}"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(disp, bar, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)

            k = cv2.waitKeyEx(20)
            if k == -1:
                continue
            if ord("0") <= k <= ord("9") and (k - ord("0")) < nc:
                c = k - ord("0")
                st["cls"] = c
                if st["sel"] >= 0:
                    st["boxes"][st["sel"]][0] = c
            elif k == ord("c") and st["sel"] >= 0:
                b = st["boxes"][st["sel"]]
                b[0] = (b[0] + 1) % nc
            elif k in DELETE_KEYS and st["sel"] >= 0:
                st["boxes"].pop(st["sel"])
                st["sel"] = -1
            elif k == ord("r"):
                st["boxes"] = predict(model, img, CONF, IMGSZ)
                st["sel"] = -1
            elif k in (ord("n"), 32):
                step = 1
                break
            elif k == ord("b"):
                step = -1
                break
            elif k in (ord("q"), 27):
                step = 0
                idx = -1
                break

        # save current image's labels (also on quit)
        cur = files[idx] if idx >= 0 else f
        save_labels(out_lbl / (cur.stem + ".txt"), st["boxes"], w, h)
        if not (out_img / cur.name).exists():
            shutil.copy(cur, out_img / cur.name)
        if idx == -1:
            break
        idx += step

    cv2.destroyAllWindows()
    print(f"Done. Labels in {out_lbl}")


if __name__ == "__main__":
    main()