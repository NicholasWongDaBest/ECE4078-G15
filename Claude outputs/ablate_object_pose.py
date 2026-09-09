"""
Offline ablation for the fruit-position pipeline.

Re-derives object positions from ONE recorded run (lab_output/pred.txt plus the
saved pred_*.png frames) under every combination of refinement stages, and
scores each against truemap.txt. Nothing here drives the robot and nothing here
writes objects.txt -- it only measures, so you can find out which stages
actually earn their place instead of enabling them all and hoping.

The question it exists to answer first: is your object error POSE error or
DEPTH error? Those need opposite fixes, and the answer falls straight out of
the pose column below.

Usage:
    python ablate_object_pose.py                     # uses lab_output/
    python ablate_object_pose.py --pred lab_output_archive/run_20260909_1335/pred.txt
    python ablate_object_pose.py --csv results.csv   # also dump the full table

IMPORTANT -- what is and isn't legitimate to tune on the graded map:

  ArUco marker positions are GIVEN to you for this milestone, so anything
  derived from them (the aruco pose refinement, the focal-length check) is
  fair game on the real run -- it's input, like intrinsic.txt.

  Fruit positions are the ANSWER. This script scores against them, which makes
  it a diagnostic. Picking your flags by the fruit-truth column ON THE GRADED
  MAP is fitting to the answer and will not generalise. Do that selection on a
  practice map, then freeze the choice.
"""
import os
import sys
import ast
import json
import csv
import argparse
import numpy as np
import cv2

sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import (in_frame_fraction, is_box_shape_plausible,
                       resolve_confusable_class, IN_FRAME_FRACTION_THRESHOLD,
                       FRAME_WIDTH, FRAME_HEIGHT)
from eval import parse_map

MARKER_LENGTH = 0.06     # m, must match ArucoSensor(marker_length=...)


# ----------------------------------------------------------------------------
# core geometry -- same pinhole model object_pose_est.estimate_pose uses
# ----------------------------------------------------------------------------
def estimate_pose(robot_pose, box, true_height, focal_length, cx):
    x_center, _, _, box_height = box
    if box_height <= 0:
        return None
    depth = true_height * focal_length / box_height
    if not (0 < depth <= 5.0):
        return None
    lateral = -(x_center - cx) * depth / focal_length      # +y is left
    rx, ry, th = robot_pose[0][0], robot_pose[1][0], robot_pose[2][0]
    c, s = np.cos(th), np.sin(th)
    return (rx + c*depth - s*lateral, ry + s*depth + c*lateral)


def bearing_of(robot_pose, box, cx, focal_length):
    """Unit world-frame direction from the robot toward this detection.

    Depends only on where the box sits horizontally, NOT on box height -- so
    unlike depth it is untouched by YOLO's box-height bias. That's what makes
    the 'rays' merge below robust to a systematic depth error.
    """
    x_center = box[0]
    th = robot_pose[2][0]
    ang = th + np.arctan2(-(x_center - cx), focal_length)
    return np.array([np.cos(ang), np.sin(ang)])


# ----------------------------------------------------------------------------
# stage 2 -- robot pose from the true ArUco map (markers re-detected in frame)
# ----------------------------------------------------------------------------
_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
    cv2.aruco.DetectorParameters())


def aruco_pose_from_frame(img_path, aruco_gt, K, dist, slam_pose,
                           max_shift=0.5, max_resid=0.15):
    """Re-derive the robot pose by matching markers seen in this frame against
    their KNOWN true positions. Returns (pose, n_markers, residual) or None.

    Gated the same way the live recover_from_pause is: reject a fit that moves
    the robot more than max_shift, or whose residual exceeds max_resid, since
    a bad fit is worse than the SLAM pose it would replace.
    """
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    corners, ids, _ = _detector.detectMarkers(img)
    if ids is None or len(ids) < 2:
        return None

    half = MARKER_LENGTH / 2.0
    obj_pts = np.array([[-half, half, 0], [half, half, 0],
                        [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    seen, body = [], []
    for i in range(len(ids)):
        tag = int(ids[i, 0])
        if tag not in aruco_gt:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[i][0], K, dist,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        t = tvec.reshape(-1)
        body.append([t[2], -t[0]])          # camera -> robot frame (x fwd, y left)
        seen.append(tag)
    if len(seen) < 2:
        return None

    body = np.array(body).T                                    # 2 x n
    world = np.hstack([aruco_gt[t].reshape(2, 1) for t in seen])

    # rigid fit body -> world. With exactly 2 points use the exact 2-point
    # solution: umeyama's SVD is rank-deficient there and can return a MIRROR.
    if body.shape[1] == 2:
        v_f, v_t = body[:, 1]-body[:, 0], world[:, 1]-world[:, 0]
        a = np.arctan2(v_t[1], v_t[0]) - np.arctan2(v_f[1], v_f[0])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        t = world.mean(axis=1, keepdims=True) - R @ body.mean(axis=1, keepdims=True)
    else:
        mf, mt = body.mean(axis=1, keepdims=True), world.mean(axis=1, keepdims=True)
        H = (world-mt) @ (body-mf).T / body.shape[1]
        U, _, Vt = np.linalg.svd(H)
        S = np.eye(2)
        if np.linalg.det(H) < 0:
            S[1, 1] = -1
        R = U @ S @ Vt
        t = mt - R @ mf

    resid = float(np.sqrt(np.mean(np.sum((R @ body + t - world)**2, axis=0))))
    pose = [[float(t[0, 0])], [float(t[1, 0])], [float(np.arctan2(R[1, 0], R[0, 0]))]]
    shift = np.hypot(pose[0][0]-slam_pose[0][0], pose[1][0]-slam_pose[1][0])
    if resid > max_resid or shift > max_shift:
        return None
    return pose, len(seen), resid


# ----------------------------------------------------------------------------
# stage 5 -- merging repeated sightings
# ----------------------------------------------------------------------------
def merge(points, rays, mode):
    """points: list of (x, y). rays: list of (robot_xy, unit_dir)."""
    P = np.array(points)
    if len(P) == 0:
        return None
    if mode == 'median':
        return np.median(P, axis=0)
    if mode == 'mad':
        c = np.median(P, axis=0)
        d = np.linalg.norm(P - c, axis=1)
        mad = np.median(np.abs(d - np.median(d))) + 1e-9
        keep = P[np.abs(d - np.median(d)) / mad < 3.0]
        return np.median(keep if len(keep) else P, axis=0)
    if mode == 'rays':
        # Least-squares intersection of the BEARING rays. Each sighting says
        # "the fruit lies somewhere along this line", which is true even when
        # its depth is badly biased -- so a systematic depth error that shifts
        # every point outward along its own ray cancels here, while it survives
        # any median of the points themselves.
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for p, u in rays:
            M = np.eye(2) - np.outer(u, u)
            A += M
            b += M @ p
        if np.linalg.cond(A) > 1e8:      # all rays near-parallel: no intersection
            return np.median(P, axis=0)
        return np.linalg.solve(A, b)
    raise ValueError(mode)


def run_variant(entries, object_dimensions, aruco_gt, K, dist, fx, cx,
                pose_mode, merge_mode, depth_scale):
    per_class_pts, per_class_rays = {}, {}
    known = {}
    n_aruco_ok = 0
    for e in entries:
        pose = e['slam_pose']
        if pose_mode == 'aruco' and e['aruco_pose'] is not None:
            pose = e['aruco_pose']
            n_aruco_ok += 1
        for predicted_class, box in e['bboxes']:
            if predicted_class not in object_dimensions:
                continue
            if in_frame_fraction(box, FRAME_WIDTH, FRAME_HEIGHT) < IN_FRAME_FRACTION_THRESHOLD:
                continue
            if not is_box_shape_plausible(predicted_class, box, object_dimensions):
                continue
            h = object_dimensions[predicted_class][2] * depth_scale
            xy = estimate_pose(pose, box, h, fx, cx)
            if xy is None:
                continue
            cls, alt = resolve_confusable_class(predicted_class, box, pose,
                                                 object_dimensions, fx, cx,
                                                 lambda *a: estimate_pose(*a) or (0.0, 0.0),
                                                 known)
            if alt is not None:
                xy = alt
            per_class_pts.setdefault(cls, []).append(xy)
            per_class_rays.setdefault(cls, []).append(
                (np.array([pose[0][0], pose[1][0]]), bearing_of(pose, box, cx, fx)))
            known[cls] = xy

    out = {}
    for cls, pts in per_class_pts.items():
        m = merge(pts, per_class_rays[cls], merge_mode)
        if m is not None:
            out[cls] = np.array(m).reshape(2, 1)
    return out, n_aruco_ok


def score(est, object_gt):
    errs = {c: float(np.linalg.norm(est[c].ravel() - object_gt[c].ravel()))
            for c in object_gt if c in est}
    if not errs:
        return float('nan'), 0, errs
    rmse = float(np.sqrt(np.mean([v**2 for v in errs.values()])))
    return rmse, len(errs), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='lab_output/pred.txt')
    ap.add_argument('--truemap', default='truemap.txt')
    ap.add_argument('--calib', default='calibration/param/')
    ap.add_argument('--objects', default='object_list.csv')
    ap.add_argument('--csv', default=None, help='also write the full table here')
    args = ap.parse_args()

    if not os.path.exists(args.pred) or os.path.getsize(args.pred) == 0:
        print(f"'{args.pred}' is missing or empty -- nothing to ablate.\n"
              "cv/detector.py opens pred.txt with 'w' at startup, so the previous\n"
              "run's poses are gone the moment operate.py relaunches. Record a run,\n"
              "then point --pred at lab_output_archive/run_<stamp>/pred.txt.")
        return

    K = np.loadtxt(os.path.join(args.calib, 'intrinsic.txt'), delimiter=',')
    dist = np.loadtxt(os.path.join(args.calib, 'distCoeffs.txt'), delimiter=',')
    fx, cx = K[0][0], K[0][2]

    object_dimensions = {}
    with open(args.objects) as f:
        for row in csv.DictReader(f):
            object_dimensions[row['object']] = [float(row['length(m)']),
                                                 float(row['width(m)']),
                                                 float(row['height(m)'])]
    aruco_gt, object_gt = parse_map(args.truemap)
    pred_dir = os.path.dirname(os.path.abspath(args.pred))

    # load once; the ArUco re-detection is the slow part so cache it per frame
    entries = []
    with open(args.pred) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            d = ast.literal_eval(line)
            img = os.path.join(pred_dir, os.path.basename(d.get('predfname', '')))
            slam_pose = d['robotpose']
            ar = aruco_pose_from_frame(img, aruco_gt, K, dist, slam_pose)
            entries.append({'slam_pose': slam_pose,
                            'aruco_pose': ar[0] if ar else None,
                            'n_markers': ar[1] if ar else 0,
                            'bboxes': [(b[0], b[1]) for b in d['bboxes']]})
    n_ar = sum(1 for e in entries if e['aruco_pose'] is not None)
    print(f"{len(entries)} frames, ArUco pose recovered in {n_ar} "
          f"({100.0*n_ar/max(len(entries),1):.0f}%)\n")

    rows = []
    print(f"{'pose':<7} {'merge':<7} {'scale':<6} {'RMSE':>8} {'found':>6}   worst")
    for pose_mode in ('slam', 'aruco'):
        for merge_mode in ('median', 'mad', 'rays'):
            for scale in (1.00, 0.95, 1.05):
                est, _ = run_variant(entries, object_dimensions, aruco_gt, K, dist,
                                      fx, cx, pose_mode, merge_mode, scale)
                rmse, n, errs = score(est, object_gt)
                worst = max(errs.items(), key=lambda kv: kv[1]) if errs else ('-', 0)
                print(f"{pose_mode:<7} {merge_mode:<7} {scale:<6.2f} "
                      f"{rmse:>7.4f}m {n:>3}/{len(object_gt)}   "
                      f"{worst[0]} {worst[1]*1000:.0f}mm")
                rows.append(dict(pose=pose_mode, merge=merge_mode, scale=scale,
                                 rmse=rmse, found=n, worst=worst[0],
                                 worst_mm=worst[1]*1000))

    best = min((r for r in rows if r['rmse'] == r['rmse']), key=lambda r: r['rmse'])
    print(f"\nbest: pose={best['pose']} merge={best['merge']} "
          f"scale={best['scale']:.2f} -> {best['rmse']:.4f}m")
    base = next(r for r in rows if r['pose'] == 'slam' and r['merge'] == 'median'
                and r['scale'] == 1.00)
    print(f"baseline (slam/median/1.00): {base['rmse']:.4f}m")
    print("\nRead the pose column first: if 'aruco' beats 'slam' by a lot, your error\n"
          "is POSE error and the fix is upstream of anything depth-related.")
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"full table -> {args.csv}")


if __name__ == '__main__':
    main()
