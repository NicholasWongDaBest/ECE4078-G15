import os
import glob
from collections import defaultdict

labels_dir = 'cv/dataset/labels'  # adjust to your actual path
ratios_by_class = defaultdict(list)

CLASS_NAMES = ['capsicum', 'greenapple', 'lemon', 'lime', 'mango', 'orange', 'redapple']

for label_file in glob.glob(os.path.join(labels_dir, '**/*.txt'), recursive=True):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = parts
            cls_id = int(cls_id)
            w, h = float(w), float(h)
            if h <= 0:
                continue
            ratios_by_class[CLASS_NAMES[cls_id]].append(w / h)

for cls, ratios in ratios_by_class.items():
    avg = sum(ratios) / len(ratios)
    print(f"{cls}: avg_ratio={avg:.3f}, n={len(ratios)}, min={min(ratios):.3f}, max={max(ratios):.3f}")