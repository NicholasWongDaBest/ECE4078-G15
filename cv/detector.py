import cv2
import os
import json
import csv
import ast
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops

# constants
MARKER_LABEL = "marker"
MARKER_COLOUR = (255, 0, 255)   # RGB, since the frame is RGB
EDGE_PX = 3                     # box within 3 px of the frame edge counts as cut off
MARKER_CONF = 0.5               # partial markers score lower than fruit
FRUIT_CONF = 0.75               # your existing threshold, now applied per class

def pick_device(device=None):
    """
    Where YOLO runs: 'gpu' = the first CUDA GPU, and an error if PyTorch has
    no CUDA (a '+cpu' build: torch.version.cuda is None) -- so a run never
    silently falls back to the ~4x slower CPU; 'cpu'; None/'auto' = the GPU
    when there is one, else the CPU. Returns what predict(device=...) takes.
    """
    import torch
    device = (device or 'auto').lower()
    if device == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 0
    if device == 'gpu':
        raise RuntimeError(
            "YOLO was asked to run on the GPU, but this PyTorch has no CUDA (torch {}, CUDA {}). Install the "
            "CUDA build: python -m pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url "
            "https://download.pytorch.org/whl/cu126 -- or run on the CPU with --yolo-device cpu".format(
                torch.__version__, torch.version.cuda))
    return 'cpu'


def use_all_cores():
    """Let PyTorch and OpenCV use every logical core for their own work
    (PyTorch defaults to the physical cores only). @return: the count"""
    import torch
    n = os.cpu_count() or 1
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    return n


class ObjectDetector:
    def __init__(self, yolo_path, device=None):
        self.model = YOLO(yolo_path)
        self.n_threads = use_all_cores()
        self.device = pick_device(device)
        # One prediction now: the model goes onto its device here, not
        # during the first frame of the run (a CUDA warm-up takes seconds).
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=480, verbose=False, device=self.device)
        self.device_name = self._device_name()
        print("YOLO: running on {} ({} CPU threads for PyTorch/OpenCV)".format(self.device_name, self.n_threads))

        self.colour_code = {}
        with open('object_list.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                obj = row['object']
                rgb = ast.literal_eval(row['rgb display'])
                self.colour_code[obj] = rgb
        
        self.pred_pose_fname = open(os.path.join('lab_output', 'pred.txt'), 'w')
        self.pred_count = 0
        self.last_markers = []

    def _device_name(self):
        """The device the model's weights actually sit on, readably."""
        import torch
        try:
            dev = next(self.model.model.parameters()).device
        except Exception:
            dev = torch.device('cuda:0' if self.device == 0 else 'cpu')
        if dev.type == 'cuda':
            return "the GPU, {} ({})".format(torch.cuda.get_device_name(dev), dev)
        return "the CPU"

    def detect_single_image(self, img):
        """
        Return:
            bboxes: fruit only, [label,[x,y,width,height]] (same format as before)
            img_out: image with fruit and markers drawn
        Markers are stored in self.last_markers as dicts:
            {"xyxy": (x1,y1,x2,y2), "partial": bool, "sides": ["left", ...]}
        """
        bboxes = self._get_bounding_boxes(img)
        img_out = deepcopy(img)
        H, W = img.shape[:2]

        fruit_bboxes = []
        self.last_markers = []
        for bbox in bboxes:
            label = bbox[0]
            xyxy = ops.xywh2xyxy(bbox[1])
            x1, y1, x2, y2 = (int(v) for v in xyxy)

            if label == MARKER_LABEL:
                hits = [("left", x1 <= EDGE_PX), ("right", x2 >= W - EDGE_PX),
                        ("top", y1 <= EDGE_PX), ("bottom", y2 >= H - EDGE_PX)]
                sides = [s for s, hit in hits if hit]
                self.last_markers.append({"xyxy": (x1, y1, x2, y2),
                                        "partial": bool(sides), "sides": sides})
                if sides:
                    print("marker cut off at:", sides)
                col = MARKER_COLOUR
                tag = "marker (partial)" if sides else "marker"
            else:
                fruit_bboxes.append(bbox)
                col = self.colour_code.get(label, (255, 255, 255))
                tag = label

            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), col, thickness=2)
            img_out = cv2.putText(img_out, tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        return fruit_bboxes, img_out

    def _get_bounding_boxes(self, img):
        # img arrives in RGB (per operate.py); Ultralytics predict() on a raw
        # numpy array assumes BGR (the cv2/training convention), so convert here
        # predict target type and bounding box with your trained YOLO
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        predictions = self.model.predict(img_bgr, imgsz=480, verbose=False, conf=MARKER_CONF, iou=0.5,
                                         device=self.device)

        bounding_boxes = []
        for prediction in predictions:
            for box in prediction.boxes:
                label = prediction.names[int(box.cls)]
                if label != MARKER_LABEL and float(box.conf) < FRUIT_CONF:
                    continue          # fruit keeps the old 0.75 threshold
                # .cpu(): on the GPU the box is a CUDA tensor, which numpy cannot read
                bounding_boxes.append([label, box.xywh[0].cpu().numpy()])
        return bounding_boxes
        
    def write_output(self, pred, state, bboxes, lab_output_dir):
        # save the object detection result to a file
        # 3 things are saved: the image with the bboxes overlaid, the pose of the robot when the image is taken, and the bboxes values.
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        pred_fname = os.path.join(lab_output_dir, 'pred_' + str(self.pred_count) + '.png')
        self.pred_count += 1
        cv2.imwrite(pred_fname, pred)#.astype(np.uint8))
        
        # for every prediction label image, save the state of the robot (its position when pressing "p")
        # this information is needed to estimate the pose of the object
        bboxes = [[label, coords.tolist()] for label, coords in bboxes]
        d = {"predfname": pred_fname, "robotpose": state, "bboxes": bboxes}
        self.pred_pose_fname.write(json.dumps(d) + '\n')
        self.pred_pose_fname.flush()
        
        return f'pred_{self.pred_count-1}.png'


# FOR TESTING ONLY
if __name__ == '__main__':
    # get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    yolo = Detector(f'{script_dir}/model/yolov8_model.pt')
    img = cv2.imread(f'{script_dir}/test/test_image_1.png')
    bboxes, img_out = yolo.detect_single_image(img)

    print(bboxes)
    print(len(bboxes))

    cv2.imshow('yolo detect', img_out)
    cv2.waitKey(0)