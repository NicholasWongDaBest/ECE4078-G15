"""
aruco_orientation_test.py -- put a marker block in front of the robot and read
which way it is oriented. Nothing moves; the robot only streams its camera.

For every marker in view it prints, averaged over --print-every seconds:

  printed rotation  how the printed pattern is turned on its face: 0/90/180/270
                    deg counter-clockwise as you look at the face (k = 0..3).
                    Read from the marker's own axes; works because the camera
                    is level and the faces are vertical.
  face points       which way the face's normal points, in degrees, measured
                    like the robot's heading (0 = +x, 90 = +y). With the
                    default --pose 0 0 0 that is the ROBOT frame:
                    0 = straight ahead, 90 = robot's left, 180 = straight back
                    at the robot, -90 = robot's right.
  snapped           the face direction rounded to the nearest axis
                    (+x / +y / -x / -y), plus how many frames picked each
                    axis. Assumes the block is square to the robot / arena.
  view angle        how far off face-on the camera is looking (0 = square-on).
                    The face direction is least accurate near 0 and far away.
  face centre       the marker centre, forward / left of the camera, in m --
                    what slam/aruco_sensor.py reports today.
  block centre      the face centre pushed half a block into the block along
                    the snapped face direction -- what the true map stores.

With two faces of the same block in view it also prints the angle between
them (should be 90) and the block centre from both faces vs. the current
code's average of the two face centres.

A single marker's pose has two mirror-image solutions. Both are computed
(solvePnPGeneric, IPPE_SQUARE); the one whose face direction is nearer an
axis is kept, unless both are about as near (--tie-deg), then the one with
the lower reprojection error -- a coin toss for the axis if they snap to
different axes (it happens far away at ~45 deg). A solution whose reprojection error is clearly
lower (--clear-ratio / --clear-px) wins outright. --no-axis-prior keeps the
lower reprojection error always (use it to test a block placed at an angle).

Usage:
  python aruco_orientation_test.py --ip <robot_ip>
  python aruco_orientation_test.py --ip <robot_ip> --pose 0 0 90     robot facing +y in the arena
  python aruco_orientation_test.py --image raw_images/aruco_0.png    saved frame(s) instead of live
Options worth measuring: --marker-length (black square edge, m),
--block-size (block edge, m), --cam-offset (camera ahead of the wheel axle, m;
the M3 code assumes 0).

Keys (click the video window first):
  SPACE  full dump of the current frame, both pose solutions
  s      save the current frame to raw_images/aruco_<n>.png
  r      reset the running averages
  q/ESC  quit
"""

import os
import sys
import math
import time
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import cv2

REFINE = {
    'none': cv2.aruco.CORNER_REFINE_NONE,
    'subpix': cv2.aruco.CORNER_REFINE_SUBPIX,
    'contour': cv2.aruco.CORNER_REFINE_CONTOUR,
    'apriltag': cv2.aruco.CORNER_REFINE_APRILTAG,
}
AXIS_NAME = {0: '+x', 90: '+y', 180: '-x', 270: '-y'}
UP_CAM = np.array([0.0, -1.0, 0.0])      # camera frame: x right, y down, z forward


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def circ_mean_sd(deg):
    """Circular mean and spread (deg) of a list of angles."""
    r = np.radians(np.asarray(deg, dtype=float))
    m = math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))
    d = np.array([wrap180(a - m) for a in deg])
    return m, float(np.sqrt(np.mean(d ** 2)))


def relative_words(rel):
    rel = wrap180(rel)
    if abs(rel) >= 135:
        return 'facing the robot'
    if 45 <= rel < 135:
        return "facing the robot's left"
    if -135 < rel <= -45:
        return "facing the robot's right"
    return 'facing away (?)'


class OrientationReader:
    def __init__(self, args):
        self.args = args
        self.K = np.loadtxt(os.path.join(args.calib_dir, 'intrinsic.txt'), delimiter=',')
        self.dist = np.loadtxt(os.path.join(args.calib_dir, 'distCoeffs.txt'), delimiter=',')
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = REFINE[args.refine]
        # OpenCV 4.13 default is 0.125, which throws away one of two faces of the same
        # block seen at a corner (same id, sharing an edge). 0.05 was the old default.
        params.minMarkerDistanceRate = args.min_marker_distance
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), params)
        h = args.marker_length / 2.0
        # Same corner order / axes as slam/aruco_sensor.py: x right, y up, z out of the face
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
        self.px, self.py, self.heading = args.pose

    # ---------------------------------------------------------------- one solution
    def describe(self, rvec, tvec, err):
        R, _ = cv2.Rodrigues(rvec)
        n = R[:, 2]                                   # face normal, camera frame (points at the camera)
        t = tvec.reshape(3)
        face_robot = math.degrees(math.atan2(-n[0], n[2]))
        face_world = (self.heading + face_robot) % 360.0
        snapped = int(round(face_world / 90.0) * 90) % 360
        fwd, left = float(t[2]) + self.args.cam_offset, float(-t[0])
        bearing = math.degrees(math.atan2(left, fwd))
        return {
            'rvec': rvec, 'tvec': tvec, 'err': float(err),
            'face_robot': face_robot, 'face_world': face_world,
            'snapped': snapped, 'axis_dist': abs(wrap180(face_world - snapped)),
            'rot': math.degrees(math.atan2(R[:, 0] @ UP_CAM, R[:, 1] @ UP_CAM)),
            'tilt': math.degrees(math.asin(float(np.clip(n @ UP_CAM, -1.0, 1.0)))),
            'fwd': fwd, 'left': left, 'dist': math.hypot(fwd, left), 'bearing': bearing,
            'view': wrap180(face_robot - (bearing + 180.0)),
        }

    def pick(self, sols):
        if len(sols) == 1:
            return 0, 'only one solution'
        a, b = sols[0], sols[1]
        lo, hi = sorted((a['err'], b['err']))
        best_err = 0 if a['err'] <= b['err'] else 1
        if hi > self.args.clear_ratio * lo and hi - lo > self.args.clear_px:
            return best_err, 'clearly lower reprojection error'
        if self.args.no_axis_prior:
            return best_err, 'lower reprojection error'
        if abs(a['axis_dist'] - b['axis_dist']) > self.args.tie_deg:
            return (0 if a['axis_dist'] < b['axis_dist'] else 1), 'nearer an axis'
        if a['snapped'] == b['snapped']:
            return best_err, 'same axis either way'          # usual when seen nearly face-on
        return best_err, 'coin toss'                         # e.g. far away at ~45 deg: both land on an axis

    # ---------------------------------------------------------------- one frame
    def process(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        dets = []
        if ids is None:
            return dets, corners, ids
        half = self.args.block_size / 2.0
        for c, tag in zip(corners, ids.flatten()):
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self.obj, c.reshape(4, 1, 2).astype(np.float64), self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            errs = np.asarray(errs).flatten() if errs is not None else np.zeros(n_sol)
            sols = [self.describe(rvecs[i], tvecs[i], errs[i]) for i in range(n_sol)]
            if not sols:
                continue
            i, why = self.pick(sols)
            s = sols[i]
            k = int(round(s['rot'] / 90.0)) % 4
            if self.args.no_axis_prior:
                a = math.radians(s['face_robot'])
            else:
                a = math.radians(s['snapped'] - self.heading)   # snapped normal, robot frame
            cf, cl = s['fwd'] - half * math.cos(a), s['left'] - half * math.sin(a)
            h = math.radians(self.heading)
            dets.append({
                'tag': int(tag), 'u': float(c[0][:, 0].mean()), 'corners': c,
                'sols': sols, 'pick': i, 'why': why, 'k': k,
                'centre': (cf, cl),
                'centre_world': (self.px + math.cos(h) * cf - math.sin(h) * cl,
                                 self.py + math.sin(h) * cf + math.cos(h) * cl),
            })
        # face index: left-to-right among detections of the same tag
        by_tag = defaultdict(list)
        for d in dets:
            by_tag[d['tag']].append(d)
        for tag, lst in by_tag.items():
            lst.sort(key=lambda d: d['u'])
            for j, d in enumerate(lst):
                d['face'], d['nfaces'] = j, len(lst)
        return dets, corners, ids

    # ---------------------------------------------------------------- drawing
    def draw(self, bgr, dets, corners, ids):
        out = bgr.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for d in dets:
            s = d['sols'][d['pick']]
            try:
                cv2.drawFrameAxes(out, self.K, self.dist, s['rvec'], s['tvec'], self.args.marker_length * 0.5, 2)
            except cv2.error:
                pass
            x, y = int(d['corners'][0][:, 0].min()), int(d['corners'][0][:, 1].max()) + 14
            label = 'k=%d  %.0fdeg %s' % (d['k'], s['face_world'],
                                         '' if self.args.no_axis_prior else AXIS_NAME[s['snapped']])
            cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    # ---------------------------------------------------------------- printing
    def dump(self, dets):
        """Everything about one frame, both pose solutions."""
        if not dets:
            print('  no markers in this frame')
            return
        for d in sorted(dets, key=lambda d: (d['tag'], d['face'])):
            print('  tag %d  face %d/%d   printed rotation %d deg (k=%d)'
                  % (d['tag'], d['face'] + 1, d['nfaces'], d['k'] * 90, d['k']))
            for j, s in enumerate(d['sols']):
                print('    %s solution %s: reproj %.2f px | face points %6.1f deg (axis %s, %4.1f off)'
                      ' | view %+5.1f | rot %+6.1f | normal tilt %+4.1f'
                      % ('*' if j == d['pick'] else ' ', 'AB'[j], s['err'], s['face_world'],
                         AXIS_NAME[s['snapped']], s['axis_dist'], s['view'], s['rot'], s['tilt']))
            s = d['sols'][d['pick']]
            print('      kept %s (%s); dist %.3f m, bearing %+.1f deg'
                  % ('AB'[d['pick']], d['why'], s['dist'], s['bearing']))
            print('      face centre  fwd %.3f left %+.3f  |  block centre fwd %.3f left %+.3f'
                  ' (world x %.3f y %.3f)' % (s['fwd'], s['left'], d['centre'][0], d['centre'][1],
                                             d['centre_world'][0], d['centre_world'][1]))
        for tag, pair in self.pairs_in(dets).items():
            print('  tag %d: two faces %.1f deg apart (should be 90)' % (tag, pair['angle']))

    def pairs_in(self, dets):
        by_tag = defaultdict(list)
        for d in dets:
            by_tag[d['tag']].append(d)
        out = {}
        for tag, lst in by_tag.items():
            if len(lst) < 2:
                continue
            a, b = lst[0], lst[1]
            sa, sb = a['sols'][a['pick']], b['sols'][b['pick']]
            out[tag] = {
                'angle': abs(wrap180(sa['face_world'] - sb['face_world'])),
                'centre': ((a['centre'][0] + b['centre'][0]) / 2, (a['centre'][1] + b['centre'][1]) / 2),
                'faces': ((sa['fwd'] + sb['fwd']) / 2, (sa['left'] + sb['left']) / 2),
            }
        return out

    def summary(self, records, pair_records, frames, secs):
        print('\n---- %d frames over %.1f s ----' % (frames, secs))
        if not records:
            print('  no markers in view')
            return
        groups = OrderedDict()
        for d in sorted(records, key=lambda d: (d['tag'], d['face'])):
            groups.setdefault((d['tag'], d['face'], d['nfaces']), []).append(d)
        for (tag, face, nfaces), lst in groups.items():
            ss = [d['sols'][d['pick']] for d in lst]
            ks = [d['k'] for d in lst]
            k = max(set(ks), key=ks.count)
            rot_m, rot_sd = circ_mean_sd([s['rot'] for s in ss])
            dir_m, dir_sd = circ_mean_sd([s['face_world'] for s in ss])
            view_m = float(np.mean([s['view'] for s in ss]))
            votes = defaultdict(int)
            for s in ss:
                votes[s['snapped']] += 1
            best = max(votes, key=votes.get)
            vote_txt = '  '.join('%s %d/%d' % (AXIS_NAME[a], v, len(ss))
                                 for a, v in sorted(votes.items(), key=lambda kv: -kv[1]))
            ties = sum(1 for d in lst if d['why'] == 'coin toss')
            print('tag %d  face %d/%d   %d frames' % (tag, face + 1, nfaces, len(lst)))
            print('  printed rotation  %3d deg (k=%d)%s     raw %+.1f +/- %.1f'
                  % (k * 90, k, '' if ks.count(k) == len(ks) else ' [k varied: %s]'
                     % dict((x, ks.count(x)) for x in set(ks)), rot_m, rot_sd))
            if self.args.no_axis_prior:
                print('  face points       %6.1f +/- %.1f deg' % (dir_m % 360, dir_sd))
            else:
                print('  face points       %6.1f +/- %.1f deg   snapped %s, %s   [%s]'
                      % (dir_m % 360, dir_sd, AXIS_NAME[best], relative_words(best - self.heading), vote_txt))
            print('  view angle        %+5.1f deg off face-on   distance %.3f m   bearing %+.1f deg'
                  '   (mirror solution a coin-toss in %d/%d frames)'
                  % (view_m, np.mean([s['dist'] for s in ss]), np.mean([s['bearing'] for s in ss]),
                     ties, len(lst)))
            print('  face centre       fwd %.3f  left %+.3f     <- what aruco_sensor.py reports now'
                  % (np.mean([s['fwd'] for s in ss]), np.mean([s['left'] for s in ss])))
            print('  block centre      fwd %.3f  left %+.3f     world x %.3f  y %.3f'
                  % (np.mean([d['centre'][0] for d in lst]), np.mean([d['centre'][1] for d in lst]),
                     np.mean([d['centre_world'][0] for d in lst]), np.mean([d['centre_world'][1] for d in lst])))
        for tag, lst in sorted(pair_records.items()):
            ang = [p['angle'] for p in lst]
            print('tag %d, both faces (%d frames): %.1f +/- %.1f deg apart (should be 90)'
                  % (tag, len(lst), np.mean(ang), np.std(ang)))
            print('  block centre from both faces   fwd %.3f  left %+.3f'
                  % (np.mean([p['centre'][0] for p in lst]), np.mean([p['centre'][1] for p in lst])))
            print('  current code (face centres avg) fwd %.3f  left %+.3f'
                  % (np.mean([p['faces'][0] for p in lst]), np.mean([p['faces'][1] for p in lst])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='localhost', help='robot IP (live mode)')
    ap.add_argument('--image', nargs='+', default=None, help='saved image(s) to read instead of the live camera')
    ap.add_argument('--calib_dir', default='calibration/param/', help='folder with intrinsic.txt, distCoeffs.txt')
    ap.add_argument('--marker-length', type=float, default=0.06, help='black square edge, m (M3 code uses 0.06)')
    ap.add_argument('--block-size', type=float, default=0.06, help='block edge, m (truemap generator uses 0.06)')
    ap.add_argument('--cam-offset', type=float, default=0.0,
                    help='camera ahead of the robot reference point, m (M3 code assumes 0)')
    ap.add_argument('--pose', type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=('X', 'Y', 'DEG'),
                    help='robot pose in the arena; default 0 0 0 = report in the robot frame')
    ap.add_argument('--refine', choices=list(REFINE), default='subpix',
                    help='ArUco corner refinement (aruco_sensor.py uses none)')
    ap.add_argument('--tie-deg', type=float, default=15.0,
                    help='mirror solutions this close in axis distance are decided by reprojection error')
    ap.add_argument('--clear-ratio', type=float, default=3.0,
                    help='reprojection errors this many times apart (and --clear-px) override the axis rule')
    ap.add_argument('--clear-px', type=float, default=0.5, help='see --clear-ratio')
    ap.add_argument('--no-axis-prior', action='store_true', help='always keep the lower-reprojection solution')
    ap.add_argument('--min-marker-distance', type=float, default=0.05,
                    help='ArUco minMarkerDistanceRate (aruco_sensor.py uses the OpenCV default 0.125)')
    ap.add_argument('--print-every', type=float, default=1.0, help='seconds between summaries')
    args = ap.parse_args()

    reader = OrientationReader(args)
    print('ArUco orientation test | marker %.3f m, block %.3f m, cam offset %.3f m, refine %s, pose (%.2f, %.2f, %.0f deg)'
          % (args.marker_length, args.block_size, args.cam_offset, args.refine, *args.pose))
    print('Angles: 0 = +x, 90 = +y (robot frame with the default pose: 0 ahead, 90 left, 180 back at the robot).')

    if args.image:
        for path in args.image:
            bgr = cv2.imread(path)
            if bgr is None:
                print('could not read', path)
                continue
            dets, corners, ids = reader.process(bgr)
            print('\n== %s ==' % path)
            reader.dump(dets)
            cv2.imshow('aruco orientation (any key: next)', reader.draw(bgr, dets, corners, ids))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from botconnect import BotConnect
    bot = BotConnect(args.ip)
    save_dir = 'raw_images'
    n_saved = 0
    records, pair_records = [], defaultdict(list)
    frames, t_last = 0, time.time()
    prev = None
    print('Waiting for the camera... (SPACE dump, s save, r reset, q quit)')
    while True:
        raw = bot.get_image()
        if raw is None or raw.max() == 0:
            cv2.waitKey(50)
            continue
        if prev is not None and raw.shape == prev.shape and np.array_equal(raw, prev):
            key = cv2.waitKey(5) & 0xFF        # same frame as last time -- don't count it twice
        else:
            prev = raw
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            dets, corners, ids = reader.process(bgr)
            frames += 1
            records.extend(dets)
            for tag, p in reader.pairs_in(dets).items():
                pair_records[tag].append(p)
            cv2.imshow('aruco orientation  (SPACE dump, s save, r reset, q quit)',
                       reader.draw(bgr, dets, corners, ids))
            key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord(' '):
            print('\n== frame dump ==')
            reader.dump(dets)
        elif key == ord('s'):
            os.makedirs(save_dir, exist_ok=True)
            while os.path.exists(os.path.join(save_dir, 'aruco_%d.png' % n_saved)):
                n_saved += 1
            path = os.path.join(save_dir, 'aruco_%d.png' % n_saved)
            cv2.imwrite(path, bgr)
            print('saved', path)
        elif key == ord('r'):
            records, pair_records, frames, t_last = [], defaultdict(list), 0, time.time()
            print('averages reset')
        if time.time() - t_last >= args.print_every and frames:
            reader.summary(records, pair_records, frames, time.time() - t_last)
            records, pair_records, frames, t_last = [], defaultdict(list), 0, time.time()
    cv2.destroyAllWindows()
    bot.running = False


if __name__ == '__main__':
    main()
