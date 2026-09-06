# estimate the pose of a detected object
import os
import ast
import json
import csv
import numpy as np

# Camera frame resolution used by the detector (matches operate.py's self.img/self.cv_vis
# shape, and the imgsz=480 passed to model.predict in detector.py).
IMG_WIDTH = 640
IMG_HEIGHT = 480

# determine if fruit is fully in frame
def is_box_clipped(box, margin=25, img_width=None, img_height=None):
    """True if the detected bbox touches/exceeds the frame edge, meaning
    box_height doesn't reflect the object's true extent."""
    if img_width is None or img_height is None:
        raise ValueError("img_width and img_height must be passed explicitly")
    x_center, y_center, box_width, box_height = box
    x_min = x_center - box_width / 2
    x_max = x_center + box_width / 2
    y_min = y_center - box_height / 2
    y_max = y_center + box_height / 2
    return (x_min <= margin or y_min <= margin or
            x_max >= img_width - margin or y_max >= img_height - margin)

EXPECTED_ASPECT_RANGE = {
    'orange':     (0.461, 1.869),
    'capsicum':   (0.205, 1.067),
    'greenapple': (0.410, 1.112),
    'lemon':      (0.523, 1.033),
    'mango':      (0.341, 3.230),
    'lime':       (0.435, 2.016),
    'redapple':   (0.250, 0.905),
}

def is_box_malformed(box, predicted_class):
    """True if the box's aspect ratio falls outside the range seen in
    training data for this class."""
    _, _, box_width, box_height = box
    if box_height <= 0 or predicted_class not in EXPECTED_ASPECT_RANGE:
        return False
    observed_ratio = box_width / box_height
    lo, hi = EXPECTED_ASPECT_RANGE[predicted_class]
    return observed_ratio < lo or observed_ratio > hi
    
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
# Starting noise model for fruit depth estimation (pinhole/bbox-height based).
# Same quadratic-in-distance shape as the ArUco marker noise model in ekf.py --
# variance should grow with distance since a fixed pixel error in box_height
# translates to a growing depth error as objects get smaller/farther. The
# constants below are NOT calibrated for this sensor (they were fit for aruco
# corner detection) -- treat DEPTH_NOISE_SCALE as the first thing to tune
# against object_rmse_log.csv if merged results don't improve.
DEPTH_NOISE_SCALE = 1.0
LATERAL_NOISE_RATIO = 1.5   # lateral assumed noisier than depth, same ratio as ekf.py

def object_noise_covariance(dist, theta):
    """
    Returns the 2x2 world-frame covariance for a fruit observation at
    distance `dist`, taken while the robot was facing `theta`.
    Builds variance in the robot's local (forward, lateral) frame, then
    rotates into world (x, y) frame using the same R(theta) convention as
    estimate_pose(): world = robot + R(theta) @ [forward, lateral].
    """
    depth_var = DEPTH_NOISE_SCALE * (0.039 * dist**2 - 0.115 * dist + 0.1064)
    depth_var = max(depth_var, 1e-4)  # guard against negative/near-zero variance at short range
    lateral_var = depth_var * LATERAL_NOISE_RATIO

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Sigma_local = np.diag([depth_var, lateral_var])
    Sigma_world = R @ Sigma_local @ R.T
    return Sigma_world


def merge_estimations(object_pose_dict):
    object_pose_dict_final = {}
    for key in object_pose_dict:
        estimates = object_pose_dict[key]
        if not estimates:
            object_pose_dict_final[key + '_0'] = {'x': 0.0, 'y': 0.0}
            continue

        arr = np.array(estimates)  # shape (N, 4): x, y, dist, theta
        pts = arr[:, :2]
        dists = arr[:, 2]
        thetas = arr[:, 3]

        if len(pts) == 1:
            merged = pts[0]
        else:
            # Outlier rejection stays as a coarse first pass -- covariance
            # weighting alone doesn't protect against a gross misdetection
            # (wrong class, bad bbox), it just weighs GOOD points correctly.
            median = np.median(pts, axis=0)
            outlier_dists = np.linalg.norm(pts - median, axis=1)
            outlier_thresh = 0.3
            inlier_mask = outlier_dists <= outlier_thresh
            if not inlier_mask.any():
                inlier_mask = np.ones(len(pts), dtype=bool)

            inlier_pts = pts[inlier_mask]
            inlier_dists = dists[inlier_mask]
            inlier_thetas = thetas[inlier_mask]

            # Inverse-covariance weighted merge (2D weighted least squares):
            #   merged = (Σ Σᵢ⁻¹)⁻¹ (Σ Σᵢ⁻¹ pᵢ)
            # instead of the old scalar 1/distance weighting -- this lets
            # depth and lateral error shrink at DIFFERENT rates per
            # observation rather than discounting a far detection uniformly.
            Sigma_sum = np.zeros((2, 2))
            weighted_pt_sum = np.zeros(2)
            for i in range(len(inlier_pts)):
                Sigma_i = object_noise_covariance(inlier_dists[i], inlier_thetas[i])
                Sigma_i_inv = np.linalg.inv(Sigma_i)
                Sigma_sum += Sigma_i_inv
                weighted_pt_sum += Sigma_i_inv @ inlier_pts[i]

            merged = np.linalg.inv(Sigma_sum) @ weighted_pt_sum

        object_pose_dict_final[key + '_0'] = {'x': float(merged[0]), 'y': float(merged[1])}
    return object_pose_dict_final

def load_object_ground_truth(fname):
    """Load only the fruit/object entries from a truemap.txt-style file.
    Returns {object_type: np.array([[x],[y]])}, or None if unavailable.
    """
    if not os.path.exists(fname):
        return None
    try:
        with open(fname, 'r') as f:
            gt_dict = json.load(f)
    except Exception as e:
        print(f"Could not parse true map '{fname}': {e}")
        return None
    object_gt = {}
    for key in gt_dict:
        if not key.startswith('aruco'):
            object_type = key.split('_')[0]
            object_gt[object_type] = np.array([[gt_dict[key]['x']], [gt_dict[key]['y']]])
    return object_gt if object_gt else None


def compute_object_rmse(object_est_dict, object_gt_dict):
    """
    object_est_dict: output of merge_estimations, e.g. {'redapple_0': {'x':..,'y':..}}
    object_gt_dict: output of load_object_ground_truth
    Returns {'rmse':.., 'errors':{obj_type: err}, 'matched':[obj_types]} or None if no matches.
    """
    matched, errors, est_pts, gt_pts = [], {}, [], []
    for key_0 in object_est_dict:
        obj_type = key_0.rsplit('_', 1)[0]
        if obj_type not in object_gt_dict:
            continue
        est = object_est_dict[key_0]
        est_xy = np.array([[est['x']], [est['y']]])
        gt_xy = object_gt_dict[obj_type]
        errors[obj_type] = round(float(np.linalg.norm(est_xy - gt_xy)), 5)
        matched.append(obj_type)
        est_pts.append(est_xy)
        gt_pts.append(gt_xy)

    if not matched:
        return None

    residual = (np.hstack(est_pts) - np.hstack(gt_pts)).ravel()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return {'rmse': rmse, 'errors': errors, 'matched': matched}

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
            for bbox in bboxes:
                predicted_class = bbox[0]
                box = bbox[1]
                if is_box_clipped(box, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
                    print(f"[FILTERED] {predicted_class} clipped: box={box}") # debug
                    continue
                true_height = object_dimensions[predicted_class][2]
                pose_x, pose_y = estimate_pose(robotpose, box, true_height, focal_length, cx)
                
                # distance from robot to this estimate, for weighting later
                robot_x, robot_y = robotpose[0][0], robotpose[1][0]
                robot_theta = robotpose[2][0]
                dist = np.hypot(pose_x - robot_x, pose_y - robot_y)
                
                object_pose_dict[predicted_class].append((pose_x, pose_y, dist, robot_theta))

    # merge the estimations of the objects so that there are only one estimate for each object type
    object_pose_dict = merge_estimations(object_pose_dict)
                     
    # save object pose estimations
    with open('lab_output/objects.txt', 'w') as fo:
        json.dump(object_pose_dict, fo, indent=4)
    
    print('Estimations saved!')