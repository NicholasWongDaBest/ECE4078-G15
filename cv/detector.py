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

class ObjectDetector:
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)
        
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
        predictions = self.model.predict(img_bgr, imgsz=480, verbose=False, conf=MARKER_CONF, iou=0.5)

        bounding_boxes = []
        for prediction in predictions:
            for box in prediction.boxes:
                label = prediction.names[int(box.cls)]
                if label != MARKER_LABEL and float(box.conf) < FRUIT_CONF:
                    continue          # fruit keeps the old 0.75 threshold
                bounding_boxes.append([label, np.asarray(box.xywh[0])])
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