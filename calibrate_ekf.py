# Offline EKF parameter calibration via log-and-replay.
#
# Workflow:
#   1. Drive ONE (or a few) careful, representative SLAM run(s) with logging
#      enabled (see the operate.py additions in the accompanying message) --
#      this captures the exact sequence of drive measurements and marker
#      observations that occurred.
#   2. This script REPLAYS that logged sequence through a fresh EKF instance,
#      for any given set of EKF parameters, and scores it against your
#      ground-truth map using the SAME alignment/RMSE math as eval.py.
#   3. An optimizer searches parameter space to minimize that RMSE.
#
# Because replay is deterministic and fast (no physical driving involved),
# this can test hundreds of parameter combinations from a single physical
# run -- far more thorough than manual trial-and-error, and perfectly
# reproducible (no run-to-run driving variance muddying the comparison).
#
# Usage:
#   python calibrate_ekf_params.py --logs lab_output/ekf_calib_log_....json --truemap truemap.txt

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pygame
from scipy.optimize import minimize

sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import DriveMeasurement, EKF
from slam.robot import Robot

# Reuse eval.py's own alignment/RMSE math directly, so calibration is scored
# EXACTLY the same way your real grading will score it.
from eval import parse_map, match_dict_key, solve_umeyama2d, apply_transform, compute_rmse


PARAM_NAMES = ['q_trans_scale', 'q_trans_floor', 'q_rot_scale', 'q_rot_floor',
               'r_base', 'r_dist_scale']


def find_latest_log(search_dir="lab_output"):
    candidates = glob.glob(os.path.join(search_dir, "ekf_calib_log_*.json"))
    if not candidates:
        return None
    # Filenames embed a unix timestamp (ekf_calib_log_<ts>.json), but sorting
    # by file modification time is more robust -- it's correct even if a file
    # were ever renamed or copied in.
    return max(candidates, key=os.path.getmtime)


class _LoggedMarker:
    """Minimal stand-in for a real Marker object -- only needs .tag and
    .position, since that's all predict()/add_landmarks()/update() read."""
    def __init__(self, tag, position):
        self.tag = tag
        self.position = np.array(position, dtype=float).reshape(2, 1)


def build_ekf(params, baseline, scale, ticks_per_meter, camera_matrix, dist_coeffs):
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=ticks_per_meter)
    return EKF(robot, params=params)


def replay(ekf, log_steps):
    for step in log_steps:
        sensor_measurement = [_LoggedMarker(m['tag'], m['position']) for m in step['sensor_measurement']]
        dm = DriveMeasurement(step['left_speed'], step['right_speed'], step['dt'],
                               delta_left_ticks=step['delta_left_ticks'],
                               delta_right_ticks=step['delta_right_ticks'])
        v_l, v_r = step['left_speed'], step['right_speed']
        if not (abs(v_l) < 1e-3 and abs(v_r) < 1e-3):
            ekf.predict(dm)
        ekf.add_landmarks(sensor_measurement)
        ekf.update(sensor_measurement)
    return ekf


def rmse_for_log(params, log_steps, aruco_gt, baseline, scale, ticks_per_meter, camera_matrix, dist_coeffs):
    ekf = build_ekf(params, baseline, scale, ticks_per_meter, camera_matrix, dist_coeffs)
    replay(ekf, log_steps)

    aruco_est = {tag: ekf.markers[:, i:i + 1] for i, tag in enumerate(ekf.taglist)}
    if len(aruco_est) < 2:
        return 5.0  # too few markers survived to align -- heavily penalize

    keys, est_vec, gt_vec = match_dict_key(aruco_est, aruco_gt)
    if est_vec.shape[1] < 2:
        return 5.0

    theta, t = solve_umeyama2d(est_vec, gt_vec)
    aligned = apply_transform(theta, t, est_vec)
    return compute_rmse(aligned, gt_vec)


def objective(x, all_log_steps, aruco_gt, baseline, scale, ticks_per_meter, camera_matrix, dist_coeffs):
    params = {name: 10 ** val for name, val in zip(PARAM_NAMES, x)}
    try:
        rmses = [rmse_for_log(params, log_steps, aruco_gt, baseline, scale,
                               ticks_per_meter, camera_matrix, dist_coeffs)
                 for log_steps in all_log_steps]
        return float(np.mean(rmses))
    except Exception as e:
        print(f"  (params caused an error, penalizing: {e})")
        return 5.0


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs='+', default=None,
                         help="One or more ekf_calib_log_*.json files. RMSE is averaged across all of them. "
                              "If omitted, auto-uses the most recently modified log in lab_output/.")
    parser.add_argument("--truemap", type=str, default=os.path.join(script_dir, "truemap.txt"))
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ticks_per_meter", type=float, default=173.5,
                         help="MUST match whatever operate.py currently uses")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--log_bound_range", type=float, default=2.0,
                         help="Keep each parameter within +/- this many orders of magnitude "
                              "of its default (log10-space). Prevents the optimizer from finding "
                              "numerically-pathological values (e.g. absurd process noise) that "
                              "happen to score well on the training log but blow up live.")
    parser.add_argument("--out", type=str, default="best_ekf_params.json")
    args, _ = parser.parse_known_args()

    if args.logs is None:
        latest = find_latest_log()
        if latest is None:
            print("No --logs given and no ekf_calib_log_*.json found in lab_output/. "
                  "Run a logged SLAM session first, or pass --logs explicitly.")
            sys.exit(1)
        args.logs = [latest]
        print(f"No --logs given -- auto-using most recent: {latest}")

    # EKF.__init__ loads UI marker images via pygame.image.load, which needs
    # a display mode set first -- we don't render anything, this just avoids a crash.
    pygame.init()
    pygame.display.set_mode((100, 100))

    # Read the EKF's ACTUAL current defaults directly from the class itself,
    # rather than keeping a second hardcoded copy in this script. This is
    # exactly what caused the baseline RMSE mismatch: this script's old
    # DEFAULTS dict was a frozen snapshot that had silently drifted out of
    # sync with whatever ekf.py's own default_params currently contains.
    _dummy_robot = Robot(1.0, 1.0, np.eye(3), np.zeros(5))
    DEFAULTS = EKF(_dummy_robot, params=None).params
    print(f"Using ekf.py's own current defaults: {DEFAULTS}\n")

    camera_matrix = np.loadtxt(os.path.join(args.calib_dir, 'intrinsic.txt'), delimiter=',')
    dist_coeffs = np.loadtxt(os.path.join(args.calib_dir, 'distCoeffs.txt'), delimiter=',')
    scale = np.loadtxt(os.path.join(args.calib_dir, 'scale.txt'), delimiter=',')
    baseline = np.loadtxt(os.path.join(args.calib_dir, 'baseline.txt'), delimiter=',')

    aruco_gt, _ = parse_map(args.truemap)

    all_log_steps = []
    for log_path in args.logs:
        with open(log_path) as f:
            all_log_steps.append(json.load(f))
        print(f"Loaded {len(all_log_steps[-1])} steps from {log_path}")

    x0 = [np.log10(DEFAULTS[name]) for name in PARAM_NAMES]
    t0 = time.time()
    baseline_rmse = objective(x0, all_log_steps, aruco_gt, baseline, scale,
                               args.ticks_per_meter, camera_matrix, dist_coeffs)
    eval_seconds = time.time() - t0
    total_steps = sum(len(s) for s in all_log_steps)
    print(f"\nBaseline (current default params) RMSE: {baseline_rmse:.5f}")
    print(f"One full replay ({total_steps} logged steps across {len(all_log_steps)} log file(s)) took {eval_seconds:.2f}s.")
    # Nelder-Mead does roughly 1-2 function evaluations per iteration --
    # this range brackets that, so it's an estimate, not exact.
    est_low = eval_seconds * args.maxiter / 60
    est_high = eval_seconds * args.maxiter * 2 / 60
    print(f"Estimated total optimization time: ~{est_low:.1f}-{est_high:.1f} minutes "
          f"for up to {args.maxiter} iterations.\n")

    print(f"Optimizing over {len(PARAM_NAMES)} parameters (log-scale), up to {args.maxiter} iterations...")
    bounds = [(np.log10(DEFAULTS[name]) - args.log_bound_range,
               np.log10(DEFAULTS[name]) + args.log_bound_range) for name in PARAM_NAMES]
    result = minimize(objective, x0,
                       args=(all_log_steps, aruco_gt, baseline, scale,
                             args.ticks_per_meter, camera_matrix, dist_coeffs),
                       method='Nelder-Mead',
                       bounds=bounds,
                       options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': args.maxiter, 'disp': True})

    best_params = {name: float(10 ** val) for name, val in zip(PARAM_NAMES, result.x)}
    print(f"\nBest RMSE found: {result.fun:.5f} (vs baseline {baseline_rmse:.5f})")
    print("Best parameters:")
    for name, val in best_params.items():
        print(f"  {name}: {val:.6g}")

    with open(args.out, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved to {args.out}")
    print("\nIMPORTANT: this was optimized against the log(s) you provided. If you only")
    print("logged one run, validate on a SEPARATE run before fully trusting these --")
    print("otherwise you risk overfitting to that specific run's path/noise.")