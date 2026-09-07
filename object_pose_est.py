# estimate the pose of a detected object
import os
import ast
import json
import csv
import numpy as np

# Camera frame resolution used by the detector (matches operate.py's self.img/self.cv_vis
# shape, and the imgsz=480 passed to model.predict in detector.py).
IMG_WIDTH = 480
    
# estimate the pose (x,y) of a detected object given its bbox and the robot's pose
def estimate_pose(robot_pose, box, object_true_height, focal_length, cx):
    """
    robot_pose: [[x], [y], [theta]] -- world-frame robot pose from SLAM
    box: [x_center, y_center, width, height] in pixels (YOLO xywh)
    object_true_height: real-world height of the object (m)
    focal_length: fx from the camera intrinsic matrix
    """
    x_center, y_center, box_width, box_height = box

    ######### Replace with your codes #########
    # TODO: compute pose of the object based on bounding box [x,y,width,height] and robot's pose [[x],[y],[theta]]
    # You may want to use the true height of the object and the focal length also
    # This is the default code which estimates every pose to be (0,0)
    if box_height <= 0:
        return 0.0, 0.0
    # Depth via the pinhole model: apparent_height/f = true_height/depth
    depth = (focal_length * object_true_height) / box_height

    # Lateral offset in the camera frame from how far the box centre sits 
    # from the image's principal point (approximated as image centre)
    x_camera = (x_center - cx) * depth / focal_length

    # Camera frame (forward=depth, image-right=x_camera) -> robot frame
    # (forward, left). "Left" is positive by convention (matches lm.position
    # in ekf.py/aruco_sensor.py), so image-right flips sign to become lateral.
    forward = depth
    lateral = -x_camera

    # Rotate + translate from robot frame into world frame
    robot_x = robot_pose[0][0]
    robot_y = robot_pose[1][0]
    theta = robot_pose[2][0]

    pose_x = robot_x + forward * np.cos(theta) - lateral * np.sin(theta)
    pose_y = robot_y + forward * np.sin(theta) + lateral * np.cos(theta)
    ###########################################
    
    return pose_x, pose_y
    

# merge the estimations of the objects so that there is only 1 final estimate for each object type
def merge_estimations(object_pose_dict):
    object_pose_dict_final = {}
    for key in object_pose_dict:
        estimates = object_pose_dict[key]
        if not estimates:
            object_pose_dict_final[key + '_0'] = {'x': 0.0, 'y': 0.0}
            continue

        arr = np.array(estimates)  # shape (N, 3): x, y, dist
        pts = arr[:, :2]
        dists = arr[:, 2]

        if len(pts) == 1:
            merged = pts[0]
        else:
            median = np.median(pts, axis=0)
            outlier_dists = np.linalg.norm(pts - median, axis=1)
            outlier_thresh = 0.3
            inlier_mask = outlier_dists <= outlier_thresh
            if not inlier_mask.any():
                inlier_mask = np.ones(len(pts), dtype=bool)

            inlier_pts = pts[inlier_mask]
            inlier_dists = dists[inlier_mask]

            # inverse-distance weights: closer detections count more
            # small epsilon avoids divide-by-zero for near-zero distance
            weights = 1.0 / (inlier_dists + 1e-3)
            weights /= weights.sum()

            merged = np.average(inlier_pts, axis=0, weights=weights)

        object_pose_dict_final[key + '_0'] = {'x': float(merged[0]), 'y': float(merged[1])}
    return object_pose_dict_final

if __name__ == "__main__":
    
    '''
    # Every line is pred.txt is: one robot pose (when the image is taken) + one or more bounding boxes of the detected object(s) (visualized in pred_X.png)
    # For every box (every detected object), you need to calculate an estimate of the object pose (x,y)
    # After estimating obj poses for all images (all lines in pred.txt), each object will have more than one estimates (because each obj may appear in more than 1 image)
    # You need to merge the estimates so that every object has only one final estimate
    '''
    
    # load camera matrix parameter which is required for object pose estimation
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',') 
    focal_length = camera_matrix[0][0]
    cx = camera_matrix[0][2]
    
    # load the list of object names and dimensions
    with open('object_list.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    object_list = [row['object'] for row in data]
    object_dimensions = {}
    for row in data:  
        object_dimensions[str(row['object'])] = [float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])]
    
    # Initialize the object pose prediction as a dictionary (object name as key, list of estimate poses as value)
    object_pose_dict = {}
    for object_name in object_list:
        object_pose_dict[object_name] = []
    
    # Compute estimates
    with open('lab_output/pred.txt') as fp:
        
        # for every line in pred.txt (every image taken)
        for line in fp.readlines():
            entry = ast.literal_eval(line)
            robotpose, bboxes = entry['robotpose'], entry['bboxes']
            
            # for every bounding box detected
            # for bbox in bboxes:
            #     predicted_class = bbox[0]
            #     box = bbox[1]
            #     true_height = object_dimensions[predicted_class][2]
            #     object_pose_dict[predicted_class].append(estimate_pose(robotpose, box, true_height, focal_length, cx))
            for bbox in bboxes:
                predicted_class = bbox[0]
                box = bbox[1]
                true_height = object_dimensions[predicted_class][2]
                pose_x, pose_y = estimate_pose(robotpose, box, true_height, focal_length, cx)
                
                # distance from robot to this estimate, for weighting later
                robot_x, robot_y = robotpose[0][0], robotpose[1][0]
                dist = np.hypot(pose_x - robot_x, pose_y - robot_y)
                
                object_pose_dict[predicted_class].append((pose_x, pose_y, dist))

    # merge the estimations of the objects so that there are only one estimate for each object type
    object_pose_dict = merge_estimations(object_pose_dict)
                     
    # save object pose estimations
    with open('lab_output/objects.txt', 'w') as fo:
        json.dump(object_pose_dict, fo, indent=4)
    
    print('Estimations saved!')