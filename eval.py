"""
Score an estimated map against the true map: SLAM RMSE (raw and after the
best rigid alignment), the marker grade, per-object errors, and the object
RMSE and object grade.

Usage:
    python eval.py                              # lab_output/slam.txt + objects.txt (as before)
    python eval.py --run auto_1                 # submission/slam_auto_1.txt + objects_auto_1.txt
    python eval.py --run manual_2 --sub-dir X   # another folder
    python eval.py --run all                    # every attempt in --sub-dir, plus a summary table
    python eval.py --slam-est a.txt --object-est b.txt

Each evaluated run gets a plot (true vs estimated markers and objects, after
alignment, with error lines; the raw estimate is shown faintly so an offset
map is visible), saved next to the files as eval_<run>.png. --no-show only
saves it; --no-plot only prints the numbers.

The scoring maths (compute_rmse, solve_umeyama2d, apply_transform,
eval_object's error rule, compute_grade) is unchanged from the lab version.
The object RMSE uses the same choice as the per-object errors: the SLAM
alignment if it gives the smaller average error, the raw positions if not;
the object grade is compute_grade() on it with the --obj-* constants.
"""
import os
import re
import json
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

ARENA_HALF = 1.25      # m, inner edge of the boundary tape
NAV_RADIUS = 0.4       # m, a navigation run must stop this close to an object


def parse_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
    aruco_dict = {}
    object_dict = {}

    for key in gt_dict:
        if key.startswith('aruco'): # read SLAM map
            aruco_num = int(key.strip('aruco')[:-2])
            aruco_dict[aruco_num] = np.reshape([gt_dict[key]['x'], gt_dict[key]['y']], (2, 1))
        else: # read object map
            object_type = key.split('_')[0]
            object_dict[object_type] = np.reshape([gt_dict[key]['x'], gt_dict[key]['y']], (2, 1))
    return aruco_dict, object_dict

def match_dict_key(d1, d2):
    # pair up the values from the same keys
    points1 = []
    points2 = []
    keys = []
    for key in d1:
        if not key in d2:
            continue
        points1.append(d1[key])
        points2.append(d2[key])
        keys.append(key)
    if not keys:
        return keys, np.zeros((2, 0)), np.zeros((2, 0))
    return keys, np.hstack(points1), np.hstack(points2)

def compute_rmse(points1, points2):
    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1 - points2).ravel()
    MSE = 1.0 / num_points * np.sum(residual ** 2)
    return np.sqrt(MSE)

def eval_slam(aruco_est, aruco_gt):
    """Print the marker table. @return: (aligned RMSE, (theta, x), details)
    where details holds what plot_run() draws."""
    taglist, slam_est_vec, slam_gt_vec = match_dict_key(aruco_est, aruco_gt)
    if len(taglist) < 2:
        print("\nOnly {} marker(s) in common with the true map -- cannot align.".format(len(taglist)))
        return float('nan'), None, {'tags': taglist, 'est': slam_est_vec, 'aligned': slam_est_vec,
                                    'gt': slam_gt_vec, 'raw_rmse': float('nan')}
    theta, x = solve_umeyama2d(slam_est_vec, slam_gt_vec)
    slam_est_vec_aligned = apply_transform(theta, x, slam_est_vec)
    diff = slam_gt_vec - slam_est_vec_aligned
    slam_rmse_raw = compute_rmse(slam_est_vec, slam_gt_vec)
    slam_rmse_aligned = compute_rmse(slam_est_vec_aligned, slam_gt_vec)

    print()
    print("Number of found markers: {}".format(len(taglist)))
    print(f'SLAM RMSE before alignment = {np.round(slam_rmse_raw, 5)}')
    print(f'SLAM RMSE after alignment = {np.round(slam_rmse_aligned, 5)}')
    print("Alignment: rotate {:.1f} deg, shift ({:+.3f}, {:+.3f}) m".format(np.degrees(theta), x[0, 0], x[1, 0]))
    print()
    print('%s %7s %9s %7s %11s %9s %7s %8s' % ('Marker', 'Real x', 'Pred x', 'dx', 'Real y', 'Pred y', 'dy', 'err'))
    print('--------------------------------------------------------------------------')
    for i in range(len(taglist)):
        print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %8.3f' % (
            taglist[i], slam_gt_vec[0][i], slam_est_vec_aligned[0][i], diff[0][i], slam_gt_vec[1][i],
            slam_est_vec_aligned[1][i], diff[1][i], np.hypot(diff[0][i], diff[1][i])))
    missing = sorted(set(aruco_gt) - set(taglist))
    if missing:
        print("Markers missing from the estimate: {}".format(missing))
    return slam_rmse_aligned, (theta, x), {'tags': taglist, 'est': slam_est_vec, 'aligned': slam_est_vec_aligned,
                                           'gt': slam_gt_vec, 'raw_rmse': slam_rmse_raw}

def solve_umeyama2d(points1, points2):
    # Solve for optimal transform theta and x such that theta*p1 + x = p2

    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])

    num_points = points1.shape[1]
    mu1 = 1 / num_points * np.reshape(np.sum(points1, axis=1), (2, -1))
    mu2 = 1 / num_points * np.reshape(np.sum(points2, axis=1), (2, -1))
    Sig12 = 1 / num_points * (points2 - mu2) @ (points1 - mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1, -1] = -1

    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1, 0], R[0, 0])
    x = mu2 - R @ mu1
    return theta, x

def apply_transform(theta, x, points):
    assert (points.shape[0] == 2)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    points_transformed = R @ points + x
    return points_transformed


def eval_object(object_est, object_gt, transform=None):
    # returns the error (euclidean distance) for each individual object estimation against gt
    # calculate two error versions: before and after alignment, and obtain the smaller of the two

    MAX_ERROR = 1
    full_obj_list = list(object_gt.keys())
    errors = {}
    # initialize error dictionary with max error
    for obj_name in full_obj_list:
        errors[obj_name] = MAX_ERROR

    # detection result
    obj_list, obj_est_vec, obj_gt_vec = match_dict_key(object_est, object_gt)
    # raw (un-aligned) positions, used for the RMSE unless the alignment gives a smaller error
    rmse_est_vec = obj_est_vec
    for i, obj_name in enumerate(obj_list):
        err = np.linalg.norm(obj_est_vec[:,i] - obj_gt_vec[:,i])
        errors[obj_name] = np.round(err, 5)
    avg_error = sum(errors.values()) / len(errors)
    if transform is None: print('Note: When evaluating object pose only, transform is not applied.')

    # if need to apply transform, calculate error after transform
    aligned = False
    if transform is not None and obj_list:
        theta, x = transform
        object_est_vec_aligned = apply_transform(theta, x, obj_est_vec)
        errors_after = {}   # lab rule kept as is: objects never found drop out of the aligned average
        for i, obj_name in enumerate(obj_list):
            err = np.linalg.norm(object_est_vec_aligned[:,i] - obj_gt_vec[:,i])
            errors_after[obj_name] = min(np.round(err, 5), errors[obj_name])
        avg_error_after = sum(errors_after.values()) / len(errors_after)
        if avg_error_after < avg_error:
            avg_error = avg_error_after
            errors = errors_after
            aligned = True
            rmse_est_vec = object_est_vec_aligned

    # RMSE across the matched objects, with the same (better) alignment as above
    obj_rmse = compute_rmse(rmse_est_vec, obj_gt_vec) if obj_list else float('nan')

    print('Object pose estimation errors{}:'.format(' (after alignment)' if aligned else ''))
    print(json.dumps(errors, indent=4))
    print(f'Number of found objects: {len(obj_list)} / {len(full_obj_list)}')
    print(f'Average object pose estimation error: {np.round(avg_error, 5)}')
    print(f"Object RMSE ({'aligned' if aligned else 'raw'}) = {np.round(obj_rmse, 5)}")
    missing = sorted(set(full_obj_list) - set(obj_list))
    if missing:
        print("Objects missing from the estimate ({}): {}".format(
            "left out of the aligned average" if aligned else "counted as {} m".format(MAX_ERROR), missing))
    return errors, obj_rmse, len(obj_list)

def compute_grade(aligned_rmse, num_found, max_rmse, min_rmse, base, total_count=10):
    """Markers (num_found of total_count markers) or objects (of total_count objects)."""
    rating = (max_rmse - aligned_rmse) / (max_rmse - min_rmse)
    rating = np.clip(rating, 0.0, 1.0)
    grade = (base**rating - 1) / (base - 1) * num_found / total_count
    return rating, grade*100


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_run(title, aruco_gt, object_gt, slam=None, object_est=None, transform=None, object_errors=None,
             save_path=None, show=True):
    """True vs estimated map. Estimates are drawn after the SLAM alignment
    (what the score uses); the unaligned estimate is drawn faintly behind, so
    an offset or rotated map is easy to spot. Grey lines join each estimate
    to its true position."""
    fig, ax = plt.subplots(figsize=(8, 8))
    h = ARENA_HALF
    ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', lw=1, label='arena tape')

    # markers
    gt_tags = sorted(aruco_gt)
    gt_m = np.hstack([aruco_gt[t] for t in gt_tags]) if gt_tags else np.zeros((2, 0))
    ax.scatter(gt_m[0], gt_m[1], marker='s', s=90, facecolors='none', edgecolors='C0', lw=2, label='marker (true)')
    for t in gt_tags:
        ax.text(aruco_gt[t][0, 0] + 0.04, aruco_gt[t][1, 0] + 0.04, str(t), color='C0', size=11)
    if slam is not None and slam['tags']:
        ax.scatter(slam['est'][0], slam['est'][1], marker='x', s=40, color='C1', alpha=0.25,
                   label='marker (estimate, raw)')
        al = slam['aligned']
        ax.scatter(al[0], al[1], marker='x', s=90, color='C1', lw=2, label='marker (estimate, aligned)')
        for i, t in enumerate(slam['tags']):
            ax.plot([slam['gt'][0, i], al[0, i]], [slam['gt'][1, i], al[1, i]], color='grey', lw=1)

    # objects
    if object_gt:
        names = sorted(object_gt)
        pts = np.hstack([object_gt[n] for n in names])
        ax.scatter(pts[0], pts[1], marker='o', s=110, facecolors='none', edgecolors='C2', lw=2,
                   label='object (true)')
        for n in names:
            ax.text(object_gt[n][0, 0] + 0.04, object_gt[n][1, 0] - 0.08, n, color='C2', size=9)
            ax.add_patch(plt.Circle((object_gt[n][0, 0], object_gt[n][1, 0]), NAV_RADIUS, color='C2',
                                    fill=False, ls=':', lw=0.8, alpha=0.5))
    if object_est:
        names = [n for n in sorted(object_est)]
        raw = np.hstack([object_est[n] for n in names])
        shown = apply_transform(*transform, raw) if transform is not None else raw
        ax.scatter(shown[0], shown[1], marker='^', s=80, color='C3', label='object (estimate)')
        for i, n in enumerate(names):
            if n in object_gt:
                ax.plot([object_gt[n][0, 0], shown[0, i]], [object_gt[n][1, 0], shown[1, i]], color='grey', lw=1)
            err = (object_errors or {}).get(n)
            label = n if err is None else "{} {:.0f}cm".format(n, 100 * err)
            ax.text(shown[0, i] + 0.04, shown[1, i] + 0.04, label, color='C3', size=8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5])
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print("Plot saved to {}".format(save_path))
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# One run / every run
# ----------------------------------------------------------------------

def evaluate(slam_path, object_path, aruco_gt, object_gt, args, name=None, show=True):
    """Score one slam/objects pair (either may be missing) and plot it.
    @return: summary dict for the table in --run all."""
    name = name or os.path.splitext(os.path.basename(slam_path or object_path or 'run'))[0]
    aruco_est = parse_map(slam_path)[0] if slam_path and os.path.exists(slam_path) else {}
    object_est = parse_map(object_path)[1] if object_path and os.path.exists(object_path) else {}
    has_slam = bool(aruco_est) and bool(aruco_gt)
    has_obj = bool(object_est) and bool(object_gt)
    print("\n" + "=" * 74)
    print("{}   slam: {}   objects: {}".format(name, slam_path if has_slam else '-', object_path if has_obj else '-'))
    print("=" * 74)
    summary = {'run': name, 'markers': len(aruco_est), 'raw': float('nan'), 'aligned': float('nan'),
               'grade': float('nan'), 'objects': len(object_est), 'obj_mean': float('nan'),
               'obj_rmse': float('nan'), 'obj_grade': float('nan')}
    if not has_slam and not has_obj:
        print("Nothing to evaluate.")
        return summary

    slam, transform, object_errors = None, None, None
    if has_slam:
        print('Evaluating SLAM:')
        rmse, transform, slam = eval_slam(aruco_est, aruco_gt)
        summary.update(raw=slam['raw_rmse'], aligned=rmse)
        if np.isfinite(rmse):
            rating, grade = compute_grade(rmse, len(aruco_est), args.max_rmse, args.min_rmse, args.base,
                                          args.total_markers)
            summary['grade'] = grade
            print(f'\nSLAM Rating: {np.round(rating, 5)}')
            print(f'SLAM Grade: {np.round(grade, 5)}')
    if has_obj:
        print('\nEvaluating Object Detection:')
        object_errors, obj_rmse, n_obj = eval_object(object_est, object_gt, transform=transform)
        summary['obj_mean'] = sum(object_errors.values()) / len(object_errors)
        summary['obj_rmse'] = obj_rmse
        if np.isfinite(obj_rmse):
            obj_rating, obj_grade = compute_grade(obj_rmse, n_obj, args.obj_max_rmse, args.obj_min_rmse,
                                                  args.obj_base, args.total_objects)
            summary['obj_grade'] = obj_grade
            print(f'\nObject Rating: {np.round(obj_rating, 5)}')
            print(f'Object Grade: {np.round(obj_grade, 5)}')

    title = name
    if has_slam:
        title += "\nmarkers {}/{}  RMSE raw {:.3f} m, aligned {:.3f} m  grade {:.1f}".format(
            len(slam['tags']), len(aruco_gt), summary['raw'], summary['aligned'], summary['grade'])
    if has_obj:
        title += "\nobjects {}/{}  mean error {:.3f} m, RMSE {:.3f} m  grade {:.1f}".format(
            len(set(object_est) & set(object_gt)), len(object_gt), summary['obj_mean'], summary['obj_rmse'],
            summary['obj_grade'])
    if args.no_plot:
        return summary
    save_path = None
    if not args.no_save:
        folder = os.path.dirname(slam_path if has_slam else object_path) or '.'
        save_path = os.path.join(folder, 'eval_{}.png'.format(name))
    plot_run(title, aruco_gt, object_gt, slam=slam, object_est=object_est if has_obj else None,
             transform=transform, object_errors=object_errors, save_path=save_path, show=show)
    return summary


def find_runs(sub_dir):
    """['auto_1', 'manual_2', ...] for every slam_/objects_ file in sub_dir."""
    runs = set()
    for f in os.listdir(sub_dir) if os.path.isdir(sub_dir) else []:
        m = re.match(r'^(?:slam|objects)_((?:auto|manual)_\d+)\.txt$', f)
        if m:
            runs.add(m.group(1))
    return sorted(runs, key=lambda r: (r.split('_')[0], int(r.split('_')[1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Matching the estimated map and the true map')
    parser.add_argument('--truemap', type=str, default='truemap.txt')
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    parser.add_argument('--object-est', type=str, default='lab_output/objects.txt')
    parser.add_argument('--run', type=str, default=None,
                        help="final demo attempt to score from --sub-dir, e.g. auto_1 or manual_2 "
                             "(slam_<run>.txt + objects_<run>.txt); 'all' scores every attempt there")
    parser.add_argument('--sub-dir', type=str, default='submission',
                        help='folder holding the final demo files (final_demo_l3.py writes to submission/)')
    parser.add_argument('--no-show', action='store_true', help='save the plots without opening a window')
    parser.add_argument('--no-save', action='store_true', help='do not save the plots')
    parser.add_argument('--no-plot', action='store_true', help='print the numbers only: no plot window, no saved plot')
    parser.add_argument('--max-rmse', type=float, default=0.3, help='Max RMSE for SLAM grading scale')
    parser.add_argument('--min-rmse', type=float, default=0.0, help='Min RMSE for SLAM grading scale')
    parser.add_argument('--base', type=float, default=10.0, help='Base for the SLAM grading curve')
    parser.add_argument('--total-markers', type=int, default=10, help='Total possible markers')
    parser.add_argument('--obj-max-rmse', type=float, default=0.5, help='Max RMSE for object grading scale')
    parser.add_argument('--obj-min-rmse', type=float, default=0.0, help='Min RMSE for object grading scale')
    parser.add_argument('--obj-base', type=float, default=10.0, help='Base for the object grading curve')
    parser.add_argument('--total-objects', type=int, default=7, help='Total possible objects')
    args, _ = parser.parse_known_args()

    aruco_gt, object_gt = parse_map(args.truemap)

    if args.run is None:
        evaluate(args.slam_est, args.object_est, aruco_gt, object_gt, args, show=not args.no_show)
    else:
        runs = find_runs(args.sub_dir) if args.run == 'all' else [args.run]
        if not runs:
            raise SystemExit("No slam_*/objects_* files in {}".format(args.sub_dir))
        results = []
        for r in runs:
            results.append(evaluate(os.path.join(args.sub_dir, 'slam_{}.txt'.format(r)),
                                    os.path.join(args.sub_dir, 'objects_{}.txt'.format(r)),
                                    aruco_gt, object_gt, args, name=r,
                                    show=not args.no_show and len(runs) == 1))
        if len(results) > 1:
            print("\n%-10s %8s %9s %9s %7s %8s %9s %9s %9s" % ('run', 'markers', 'raw RMSE', 'aligned', 'grade',
                                                               'objects', 'obj mean', 'obj RMSE', 'obj grade'))
            print('-' * 86)
            for s in results:
                print("%-10s %8d %9.3f %9.3f %7.1f %8d %9.3f %9.3f %9.1f" % (
                    s['run'], s['markers'], s['raw'], s['aligned'], s['grade'], s['objects'], s['obj_mean'],
                    s['obj_rmse'], s['obj_grade']))
            graded = [s for s in results if np.isfinite(s['grade'])]
            if graded:
                best = max(graded, key=lambda s: s['grade'])
                print("\nBest marker map: {} (grade {:.1f}).{}".format(
                    best['run'], best['grade'],
                    "" if args.no_plot or args.no_save else " Plots saved as eval_<run>.png in {}".format(args.sub_dir)))
