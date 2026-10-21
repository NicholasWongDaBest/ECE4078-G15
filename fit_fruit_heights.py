"""
Which height each fruit should have in object_list.csv, from final_demo_l3.py
run logs that have a truth map (run_logs/<date-time>/truemap.txt).

    python fit_fruit_heights.py                         # every run in run_logs/
    python fit_fruit_heights.py --since 20261021        # runs from that date on
    python fit_fruit_heights.py run_logs/20261021-151313 run_logs/20261021-121032
    python fit_fruit_heights.py --fruit lemon --per-run

A fruit's range comes from its box height: range = f * height / box_h. If the
height in object_list.csv is not how tall the fruit LOOKS to the detector
(a different fruit, one lying on its side, how YOLO draws its box), every
range is off by the same factor and the fruit is placed too near or too far.
For every sighting this compares the box's range with the TRUE camera range
(the fruit's truth position, put into the run's map frame by fitting the run's
final markers onto the truth map; the logged robot pose) and reports the
height that would have made them agree:

    implied height = csv height * true range / box range

per run (median over its sightings), and over all runs (median and IQR) as
the value to put in object_list.csv. Nothing is written: change the csv
yourself. The csv heights the run used do not matter -- the box heights are
in the log, so the answer is the same whatever the csv said then.

Runs are left out when their final map does not fit the truth map well
(median marker error over --max-map-error) or is scaled wrongly (a free-scale
fit off by more than --max-scale-error: usually the wrong --marker-size,
which scales every range in the run). A fruit needs --min-sightings in a run.
Sightings cut off by the frame edge are skipped, as are ones more than 35%
from the truth (another fruit under that label).

Checked against the logs to Oct 21 (51 runs): near (< 0.8 m) and far
sightings give the same height, so it is a height error and not an offset;
the alternatives were worse -- range from the box's bottom edge (camera 8 cm
above the floor: 7-14% spread, 24 cm off at 1.5-2.2 m, against 3-6% and
4 cm) and bearing-only triangulation (3.3 cm against 2.3 cm with views from
all round, much worse from one side).
"""
import os
import sys
import glob
import json
import math
import csv
import argparse

import numpy as np
from scipy.optimize import least_squares


def _rigid_fit(est, truth, scale=False):
    """Fit est markers onto truth markers (rotation + shift [+ scale]), robustly.
    @return: (params [tx, ty, rot, scale], median marker error after the fit)"""
    tags = sorted(t for t in est if t in truth)
    A = np.array([est[t] for t in tags], dtype=float)
    B = np.array([truth[t] for t in tags], dtype=float)

    def res(p):
        c, s = math.cos(p[2]), math.sin(p[2])
        k = p[3] if scale else 1.0
        return (k * (A @ np.array([[c, -s], [s, c]]).T) + p[:2] - B).ravel()

    p = least_squares(res, [0.0, 0.0, 0.0, 1.0], loss='soft_l1', f_scale=0.03).x
    err = float(np.median(np.hypot(*res(p).reshape(-1, 2).T)))
    return p, err


def _camera_offset(calib_dir):
    """How far the camera sits ahead of the robot's pose point (the wheel axle),
    m: aruco_faces.json's camera_offset. Physical, so the same for every run --
    NOT the run's --fruit-camera-offset, which only said how that run computed
    its own ranges (runs before Oct 20 used 0, and their poses are still the
    axle's)."""
    try:
        return float(json.load(open(os.path.join(calib_dir, 'aruco_faces.json'))).get('camera_offset', 0.05))
    except (OSError, ValueError):
        return 0.05


def fit_run(run, heights, focal, cx, off, args):
    """{fruit: {'implied': [...], 'near': [...], 'far': [...]}} for one run, or (None, reason)."""
    try:
        truth_map = json.load(open(os.path.join(run, 'truemap.txt')))
        final = json.load(open(os.path.join(run, 'final.json')))
    except (OSError, ValueError):
        return None, 'no truth map / final.json'
    truth = {int(k[5:-2]): (v['x'], v['y']) for k, v in truth_map.items() if k.startswith('aruco')}
    est = {int(k): v for k, v in final.get('markers', {}).items() if int(k) in truth}
    if len(est) < 5:
        return None, 'fewer than 5 markers mapped'
    p, err = _rigid_fit(est, truth)
    if err > args.max_map_error:
        return None, 'map fits the truth to %.1f cm only' % (100 * err)
    ps, _ = _rigid_fit(est, truth, scale=True)
    if abs(ps[3] - 1.0) > args.max_scale_error:
        return None, 'map scaled %+.1f%% (wrong --marker-size?)' % (100 * (ps[3] - 1.0))
    c, s = math.cos(p[2]), math.sin(p[2])
    R = np.array([[c, -s], [s, c]])
    out = {}
    for line in open(os.path.join(run, 'events.jsonl')):
        if '"fruit_obs"' not in line:
            continue
        e = json.loads(line)
        if e.get('kind') != 'fruit_obs':
            continue
        X, Y, TH = e['pose']
        for label, box in e['boxes']:
            u, v, w, h = box
            key = label + '_0'
            if label not in heights or key not in truth_map or h <= 0 or not (25 < u < 615):
                continue
            T = R.T @ (np.array([truth_map[key]['x'], truth_map[key]['y']]) - p[:2])   # truth in this run's map frame
            dx, dy = T[0] - X, T[1] - Y
            tf, tl = math.cos(TH) * dx + math.sin(TH) * dy, -math.sin(TH) * dx + math.cos(TH) * dy
            true_range = math.hypot(tf - off, tl)
            d = focal * heights[label] / h
            box_range = math.hypot(d, (u - cx) * d / focal)
            if true_range < 0.2 or abs(true_range / box_range - 1.0) > 0.35:
                continue
            implied = heights[label] * true_range / box_range
            r = out.setdefault(label, {'implied': [], 'near': [], 'far': []})
            r['implied'].append(implied)
            r['near' if true_range < 0.8 else 'far'].append(implied)
    return {k: v for k, v in out.items() if len(v['implied']) >= args.min_sightings}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('runs', nargs='*', help='run folders (default: every run in --log-dir)')
    ap.add_argument('--log-dir', default='run_logs')
    ap.add_argument('--since', default=None, help='only runs named from this date-time on, e.g. 20261021')
    ap.add_argument('--fruit', default=None, help='only this fruit')
    ap.add_argument('--per-run', action='store_true', help='print every run, not just the summary')
    ap.add_argument('--csv', default='object_list.csv')
    ap.add_argument('--calib-dir', default='calibration/param/')
    ap.add_argument('--camera-offset', type=float, default=None,
                    help='m the camera sits ahead of the pose point (default: aruco_faces.json camera_offset)')
    ap.add_argument('--min-sightings', type=int, default=8)
    ap.add_argument('--max-map-error', type=float, default=0.02, help='m (median marker error after the fit)')
    ap.add_argument('--max-scale-error', type=float, default=0.02, help='fraction')
    args = ap.parse_args()

    K = np.loadtxt(os.path.join(args.calib_dir, 'intrinsic.txt'), delimiter=',')
    focal, cx = float(K[0, 0]), float(K[0, 2])
    heights = {r['object']: float(r['height(m)']) for r in csv.DictReader(open(args.csv))}
    off = args.camera_offset if args.camera_offset is not None else _camera_offset(args.calib_dir)
    runs = args.runs or sorted(glob.glob(os.path.join(args.log_dir, '2*')))
    if args.since:
        runs = [r for r in runs if os.path.basename(os.path.normpath(r)) >= args.since]

    per_fruit = {}
    skipped = []
    for run in runs:
        res, why = fit_run(run, heights, focal, cx, off, args)
        name = os.path.basename(os.path.normpath(run))
        if res is None:
            skipped.append((name, why))
            continue
        for label, r in res.items():
            if args.fruit and label != args.fruit:
                continue
            med = float(np.median(r['implied']))
            per_fruit.setdefault(label, []).append((name, med, len(r['implied']),
                                                    float(np.median(r['near'])) if len(r['near']) >= 4 else None,
                                                    float(np.median(r['far'])) if len(r['far']) >= 4 else None))
    used = len({n for v in per_fruit.values() for n, *_ in v})
    print('Fruit heights from %d run(s) with a truth map (%d skipped); camera %.3f m ahead of the pose' % (
        used, len(skipped), off))
    if args.per_run:
        for label in sorted(per_fruit):
            print('\n%s (csv %.3f m)' % (label, heights[label]))
            for name, med, n, near, far in per_fruit[label]:
                print('  %s  %.3f m from %3d sightings   near %s  far %s' % (
                    name, med, n, '%.3f' % near if near else '  -  ', '%.3f' % far if far else '  -  '))
    print('\n%-11s %8s %10s %15s %6s %10s  %s' % ('fruit', 'csv', 'best fit', 'middle half', 'runs', 'change', ''))
    for label in sorted(per_fruit):
        v = np.array([m for _, m, *_ in per_fruit[label]])
        best = float(np.median(v))
        lo, hi = np.percentile(v, [25, 75])
        change = 100 * (best - heights[label]) / heights[label]
        note = 'ranges %.0f%% %s now -- worth changing' % (abs(change), 'long' if change < 0 else 'short') \
            if abs(change) >= 2.5 else 'fine as it is'
        print('%-11s %8.3f %10.3f %7.3f-%.3f %6d %+9.1f%%  %s' % (label, heights[label], best, lo, hi, len(v), change, note))
    if skipped and args.per_run:
        print('\nskipped:')
        for name, why in skipped:
            print('  %s: %s' % (name, why))


if __name__ == '__main__':
    main()
