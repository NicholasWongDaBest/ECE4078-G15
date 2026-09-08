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

# ---------------------------------------------------------------------------
# Per-box quality gates, ported from the ekf.py-based pipeline:
#   in_frame_fraction()/IN_FRAME_FRACTION_THRESHOLD -> is_box_clipped()
#   is_box_shape_plausible()/EXPECTED_ASPECT_RANGE   -> is_box_malformed()
# ---------------------------------------------------------------------------

IN_FRAME_FRACTION_THRESHOLD = 0.90  # box must be >=90% inside the frame to be trusted


def in_frame_fraction(box, img_width, img_height):
    """Fraction of this detection's box's AREA that lies within the camera
    frame, from 0.0 (fully outside/off-frame) to 1.0 (fully inside)."""
    x_center, y_center, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    x0, x1 = x_center - w / 2.0, x_center + w / 2.0
    y0, y1 = y_center - h / 2.0, y_center + h / 2.0

    vis_w = max(0.0, min(x1, img_width) - max(x0, 0.0))
    vis_h = max(0.0, min(y1, img_height) - max(y0, 0.0))
    return (vis_w * vis_h) / (w * h)


def is_box_clipped(box, img_width, img_height, threshold=IN_FRAME_FRACTION_THRESHOLD):
    """True if less than `threshold` fraction of the box's area lies inside
    the frame -- a clipped box under-reports the object's true apparent
    height/width, which throws off estimate_pose()'s pinhole depth calc."""
    return in_frame_fraction(box, img_width, img_height) < threshold


# Empirically measured (from training data) width/height aspect-ratio range
# per class -- catches occlusion, merged/overlapping detections, or bad
# reads that are fully inside the frame (so invisible to is_box_clipped).
EXPECTED_ASPECT_RANGE = {
    'orange':     (0.461, 1.869),
    'capsicum':   (0.205, 1.067),
    'greenapple': (0.410, 1.112),
    'lemon':      (0.523, 1.033),
    'mango':      (0.341, 3.230),
    'lime':       (0.435, 2.016),
    'redapple':   (0.250, 0.905),
}


def expected_aspect_ratio_range_from_dimensions(object_true_dims, slack=1.15):
    """
    Geometric aspect-ratio bound derived straight from a fruit's true
    (length, width, height): modelling the fruit as roughly convex, its
    projected aspect ratio from ANY viewing angle is bounded by
    [smallest/largest, largest/smallest] (narrowest end-on, widest
    broadside). `slack` widens this a bit to absorb non-ellipsoid shape
    and imperfect box fitting.
    """
    if len(object_true_dims) != 3:
        raise ValueError(f"expected (length, width, height), got {object_true_dims}")
    largest = max(object_true_dims)
    smallest = min(object_true_dims)
    if smallest <= 0:
        return (0.0, float('inf'))
    base_ratio = largest / smallest
    return (1.0 / (base_ratio * slack), base_ratio * slack)


def plausible_aspect_ratio_range(predicted_class, object_dimensions=None, slack=1.15):
    """Union (widest) of the empirical EXPECTED_ASPECT_RANGE and the
    analytical range derived from object_dimensions, when both are
    available -- narrower-than-geometric empirical data usually means an
    undersampled training angle, not a real physical constraint."""
    empirical = EXPECTED_ASPECT_RANGE.get(predicted_class)
    analytical = (expected_aspect_ratio_range_from_dimensions(object_dimensions[predicted_class], slack)
                  if object_dimensions is not None and predicted_class in object_dimensions else None)

    if empirical is not None and analytical is not None:
        return (min(empirical[0], analytical[0]), max(empirical[1], analytical[1]))
    if empirical is not None:
        return empirical
    if analytical is not None:
        return analytical
    return (0.0, float('inf'))


def is_box_malformed(box, predicted_class, object_dimensions=None, slack=1.15):
    """True if `box`'s width/height aspect ratio falls OUTSIDE the
    plausible range for `predicted_class` -- i.e. not a geometrically
    intact, unoccluded view of this fruit at any angle."""
    _, _, w, h = box
    if w <= 0 or h <= 0:
        return True
    aspect = w / h
    lo, hi = plausible_aspect_ratio_range(predicted_class, object_dimensions, slack)
    return not (lo <= aspect <= hi)


def load_object_ground_truth(fname):
    """Load only the fruit/object entries from a truemap.txt-style file.
    Returns {object_type: np.array([[x],[y]])}, or None if unavailable."""
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
    object_est_dict: merge_estimations() output, e.g. {'redapple_0': {'x':..,'y':..}}
    object_gt_dict: load_object_ground_truth() output
    Returns {'rmse':.., 'errors':{obj_type: err}, 'matched':[obj_types]},
    or None if nothing has matched yet.
    """
    matched, errors, est_pts, gt_pts = [], {}, [], []
    for key_0, est in object_est_dict.items():
        obj_type = key_0.rsplit('_', 1)[0]
        if obj_type not in object_gt_dict:
            continue
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

# estimate the pose (x,y) of a detected object given its bbox and the robot's pose
def estimate_pose(robot_pose, box, object_true_height, focal_length, cx):
    """
    robot_pose: [[x], [y], [theta]] -- world-frame robot pose from SLAM
    box: [x_center, y_center, width, height] in pixels (YOLO xywh)
    object_true_height: real-world height of the object (m)
    focal_length: fx from the camera intrinsic matrix
    """
    x_center, y_center, box_width, box_height = box

    if box_height <= 0:
        return 0.0, 0.0
    # Depth via the pinhole model: apparent_height/f = true_height/depth
    depth = (focal_length * object_true_height) / box_height

    # Lateral offset in the camera frame from how far the box centre sits
    # from the image's principal point (approximated as image centre)
    x_camera = (x_center - cx) * depth / focal_length

    # Camera frame (forward=depth, image-right=x_camera) -> robot frame
    # (forward, left). "Left" is positive by convention, so image-right
    # flips sign to become lateral.
    forward = depth
    lateral = -x_camera

    # Rotate + translate from robot frame into world frame
    robot_x = robot_pose[0][0]
    robot_y = robot_pose[1][0]
    theta = robot_pose[2][0]

    pose_x = robot_x + forward * np.cos(theta) - lateral * np.sin(theta)
    pose_y = robot_y + forward * np.sin(theta) + lateral * np.cos(theta)

    return pose_x, pose_y


# merge the estimations of the objects so that there is only 1 final estimate for each object type
def merge_estimations(object_pose_dict):
    object_pose_dict_final = {}
    for key in object_pose_dict:
        estimates = object_pose_dict[key]
        if not estimates:
            object_pose_dict_final[key + '_0'] = {'x': 0.0, 'y': 0.0}
            continue

        arr = np.array(estimates)  # shape (N, 3) or (N, 4): x, y, dist[, theta]
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