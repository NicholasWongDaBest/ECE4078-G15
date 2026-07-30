import cv2
import os
import json
import csv
import ast
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops


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

    def detect_single_image(self, img):
        """
        Detect objects given a captured frame.
        Return:
            bboxes: list of lists, box info [label,[x,y,width,height]] for all detected objects in image
            img_out: image with bounding boxes and class labels drawn on
        """
        bboxes = self._get_bounding_boxes(img)
        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            #  translate bounding box info back to the format of [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box and class label
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.colour_code[bbox[0]], thickness=2)
            img_out = cv2.putText(img_out, bbox[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colour_code[bbox[0]], 2)

        return bboxes, img_out

    def _get_bounding_boxes(self, img):
        # predict target type and bounding box with your trained YOLO
        predictions = self.model.predict(img, imgsz=480, verbose=False)

        # get bounding box and class label for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                # bounding format in [x, y, width, height]
                box_cord = box.xywh[0]
                box_label = box.cls  # class label of the box
                bounding_boxes.append([prediction.names[int(box_label)], np.asarray(box_cord)])
        
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