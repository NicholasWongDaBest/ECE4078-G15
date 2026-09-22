# M3 - Autonomous object searching
import os
import sys
import cv2
import ast
import json
import time
import math
import argparse
import csv
import numpy as np
from botconnect import BotConnect # access the robot communication

# SLAM components -- same modules M1/M2 already use (operate.py), so M3 runs
# under identical SLAM machinery rather than a separate implementation.
from slam.ekf import (EKF, DriveMeasurement, in_frame_fraction, is_box_shape_plausible,
                      IN_FRAME_FRACTION_THRESHOLD)
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor, Marker

# Pseudo tag ids for fruit landmarks in the EKF's frozen map -- real ArUco
# markers are 1..10, fruits get FRUIT_TAG_BASE + index. See
# Navigator.load_fruit_landmarks().
FRUIT_TAG_BASE = 100

import path_planner
from m3_display import M3Display, NullDisplay, M3Abort


def read_true_map(fname):
    """
    Read the ground truth map and output the pose of the ArUco markers and objects to search
    @param fname: filename of the map
    @return:
        1) list of objects, e.g. ['redapple', 'greenapple', 'orange']
        2) positions of the objects, [[x1, y1], ..... [xn, yn]]
        3) positions of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())
        object_list = []
        object_true_pos = []
        # np.zeros, not np.empty: np.empty leaves whatever was in memory, so any
        # row the loop below doesn't fill became an obstacle at a random spot.
        aruco_true_pos = np.zeros([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            # Exact coordinates. The template rounded these to 0.1 m, which moved
            # obstacles by up to ~7 cm (more than the planner's 5 cm safety margin)
            # and disagreed with the exact values ekf.load_true_map() uses.
            x = float(gt_dict[key]['x'])
            y = float(gt_dict[key]['y'])

            if key.startswith('aruco'):
                # 'aruco7_0' -> 7, 'aruco10_0' -> 10. Marker N goes in row N-1, as
                # the docstring says. (The template used int(key[5]), which put
                # marker N in row N, so aruco10 overwrote aruco9 and row 0 was
                # never filled.)
                marker_id = int(key[len('aruco'):].split('_')[0])
                aruco_true_pos[marker_id - 1][0] = x
                aruco_true_pos[marker_id - 1][1] = y
            else:
                object_list.append(key[:-2])
                if len(object_true_pos) == 0:
                    object_true_pos = np.array([[x, y]])
                else:
                    object_true_pos = np.append(object_true_pos, [[x, y]], axis=0)

        return object_list, object_true_pos, aruco_true_pos


def read_search_list():
    """
    Read the search order of the objects
    @return: search order of the objects
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        objects = fd.readlines()

        for obj in objects:
            search_list.append(obj.strip())

    return search_list


def print_object_pos(search_list, object_list, object_true_pos):
    """
    Print out the objects pos in the search order
    @param search_list: search order of the objects
    @param object_list: list of objects
    @param object_true_pos: positions of the objects
    """
    print("Search order:")
    n_object = 1
    for obj in search_list:
        # all objects in the map, not just the first len(search_list) of them
        for i in range(len(object_list)):
            if obj == object_list[i]:
                print('{}) {} at [{}, {}]'.format(n_object, obj, np.round(object_true_pos[i][0], 2), np.round(object_true_pos[i][1], 2)))
        n_object += 1


def init_ekf(calib_dir):
    """Load calibration params and build a fresh EKF -- same construction as
    operate.py's init_ekf(), so M3 runs under identical SLAM parameters.
    @return: (ekf, baseline)"""
    camera_matrix = np.loadtxt(os.path.join(calib_dir, 'intrinsic.txt'), delimiter=',')
    dist_coeffs = np.loadtxt(os.path.join(calib_dir, 'distCoeffs.txt'), delimiter=',')
    scale = np.loadtxt(os.path.join(calib_dir, 'scale.txt'), delimiter=',')
    baseline = np.loadtxt(os.path.join(calib_dir, 'baseline.txt'), delimiter=',')
    # ticks_per_meter: same value operate.py's init_ekf() currently uses (marked
    # there as "##change this value" -- if you recalibrate it, update BOTH places).
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=190)
    return EKF(robot), float(baseline)


def load_turn_scale(fname=os.path.join('calibration', 'param', 'turn_scale.txt')):
    """
    Rotation correction factor, as measured by calibrate_turn.py.

    Navigator.turn() works out its tick count from arc-length geometry, which
    assumes the wheels roll cleanly about the robot's centre. Turning on the
    spot scrubs both tyres sideways instead, so a tick of wheel rotation
    sweeps less body angle than the geometry predicts and the robot lands
    short. turn_scale is the measured ratio between the two.

    Falls back to 1.0 (the raw geometry, i.e. what turn() did before any
    calibration) if the file isn't there, so the script still runs on a fresh
    clone -- it just under-turns.
    """
    if not os.path.exists(fname):
        print(f"No {fname} -- turns will use the raw geometry (turn_scale = 1.0), "
              f"which under-turns. Run calibrate_turn.py to measure it.")
        return 1.0
    try:
        scale = float(np.loadtxt(fname, delimiter=','))
    except Exception as e:
        print(f"Could not read {fname} ({e}) -- falling back to turn_scale = 1.0")
        return 1.0
    if not (0.5 <= scale <= 2.0):
        print(f"turn_scale = {scale:.4f} from {fname} is outside the plausible 0.5-2.0 "
              f"range -- ignoring it and using 1.0. Re-run calibrate_turn.py.")
        return 1.0
    print(f"Turn calibration: turn_scale = {scale:.4f} (from {fname})")
    return scale


def load_object_dimensions(fname='object_list.csv'):
    """{name: [length, width, height]} in metres, as operate.py loads it. The
    height is what turns a YOLO box into a range (pinhole model)."""
    dims = {}
    with open(fname, newline='') as f:
        for row in csv.DictReader(f):
            dims[row['object']] = [float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])]
    return dims


class Navigator:
    """
    Wraps the robot connection and the frozen-map SLAM (slam/ekf.py's
    load_true_map()) that M3 navigation needs: pose estimation
    (get_robot_pose) and calibrated turn-then-drive execution (turn /
    drive_forward), built on the same EKF + ArUco pipeline M1/M2 already
    use in operate.py rather than a separate one.

    How the pose is tracked:
      - Each turn()/drive_forward() feeds the EKF a predict() step for the move
        it just COMMANDED (the tick counts sent to the robot), straight after
        the move finishes. It doesn't read the wheel counters across the move:
        the Pi zeroes them whenever the robot stops (see calibrate_encoder.py),
        so a before/after reading around a whole move sees ~0 ticks.
      - Commanded motion is less certain than measured motion (wheels slip,
        especially turning on the spot), so each predict also widens the pose
        uncertainty in proportion to the move. That's what lets the marker
        update() in get_robot_pose() pull the estimate back when a move comes
        up short or long.
    """

    def __init__(self, botconnect, ekf, aruco_sensor, baseline,
                 drive_speed=0.4, turn_speed=0.35, move_timeout=15.0,
                 settle_time=0.4, turn_noise_frac=0.10, drive_noise_frac=0.05,
                 marker_noise=(0.03, 0.03), turn_scale=1.0):
        self.botconnect = botconnect
        self.ekf = ekf
        self.aruco_sensor = aruco_sensor
        self.ticks_per_meter = ekf.robot.ticks_per_meter
        # Physical distance between the wheels. baseline.txt stores it as a
        # NEGATIVE number (-0.1386: the calibration formula used the -0.5 wheel
        # speed). The EKF's motion model has used it that way consistently since
        # M1, so ekf.robot.baseline is left alone -- only the tick maths for
        # turning needs the plain physical distance.
        self.wheel_separation = abs(float(baseline))
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.move_timeout = move_timeout
        self.settle_time = settle_time            # s to wait after a move before the next camera frame
        self.turn_noise_frac = turn_noise_frac    # heading std dev added per turn, as a fraction of the turn
        self.drive_noise_frac = drive_noise_frac  # x/y std dev added per drive, as a fraction of the distance
        # Wheels slip when turning on the spot, so a turn can come up short even
        # though the wheels turned exactly the ticks asked for. If the robot
        # consistently turns e.g. 80 deg when asked for 90, set turn_scale to
        # 90/80 = 1.125 and it will send proportionally more ticks per turn.
        self.turn_scale = turn_scale

        self.min_move = 0.02                      # m; closer than this counts as "already there"
        self.heading_tolerance = np.deg2rad(5.0)  # re-turn before driving if still off by more than this
        self.max_heading_corrections = 2

        # Active re-localisation after each waypoint (see relocalise()).
        self.relocalise_enabled = True
        self.relocalise_max_residual = 0.05       # m; reject a rigid fit worse than this
        # Fast-turn-then-slow-pan geometry. The best direction is scored by
        # best_view_heading(); the pan sweeps +/- pan_half about it in pan_step
        # increments at pan_turn_speed, sensing at every step.
        self.pan_half = np.deg2rad(20.0)
        self.pan_step = np.deg2rad(5.0)     # ~1 encoder tick; finer than this does nothing
        self.pan_turn_speed = 0.25
        self.min_pan_hits = 2                     # steps that saw a landmark before a pan counts as done
        # Direction scoring (see best_view_heading()).
        self.view_kernel_sd = np.deg2rad(12.0)    # how sharply a landmark's pull falls off with bearing
        # Range quality: (too_close, good_from, good_to, too_far) in metres. A
        # landmark is worth 1.0 between good_from and good_to, ramping to 0 at
        # the ends. Too close: it fills or drops out of the frame, and a few
        # cm of lateral error becomes degrees of heading error. Too far: too
        # few pixels for a reliable range.
        self.marker_range_profile = (0.35, 0.60, 1.40, 2.00)
        self.fruit_range_profile = (0.35, 0.45, 0.90, 1.20)
        self.view_turn_penalty = 0.02             # score per radian of turning: a 180 deg turn costs
                                                  # ~2 s, a poor view costs a whole extra pan
        # Convergence: keep scanning new directions until the EKF's own pose
        # uncertainty is under these, or max_scan_rounds is used up.
        self.converge_pos_sd = 0.06               # m, on each axis
        self.converge_ang_sd = np.deg2rad(6.0)
        self.max_scan_rounds = 3
        self.last_relocalise_converged = False
        # Bonus per EXTRA landmark inside the field of view at once. One landmark
        # gives range+bearing, which cannot separate a sideways position error
        # from a heading error; two in the same frame can. So a view with a
        # pair in it is worth far more than two views with one each.
        self.view_pair_bonus = 1.0
        # Rigid re-anchor only from this many markers in ONE frame. With two,
        # the fit is exact whatever the noise, so its residual says nothing
        # and recover_from_pause() then trusts a pose that can be 10 cm off.
        # Two markers still correct the pose -- through the weighted update().
        self.anchor_min_markers = 3
        # Fruit landmarks (see load_fruit_landmarks() / _detect_fruits()).
        self.fruit_view_weight = 0.5              # a fruit is worth this much of a marker when scoring
        self.fruit_max_range = self.fruit_range_profile[3]   # m; a 7 cm fruit's box-height range is unusable beyond this
        self.fruit_noise_scale = 3.0              # fruit range/bearing sigma = this x the ArUco model
        self.detector = None
        self.object_dimensions = {}
        self.fruit_tags = {}                      # fruit name -> pseudo tag in ekf.taglist
        # Honest pose uncertainty to assume on arriving at a waypoint, before
        # the scan. See _inflate_pose_covariance() for why this is needed.
        # 20 cm / 15 deg: measured post-leg errors on this robot are 10-15 cm,
        # and the prior has to be LARGER than the marker noise model (3 cm +
        # 3 cm/m, i.e. ~7-9 cm at typical ranges) or a single good view only
        # closes half the error. Relocalise's job is to replace the
        # dead-reckoned pose, not nudge it.
        self.relocalise_prior_sd = (0.20, np.deg2rad(15.0))   # (m, rad)

        self._localised = False       # True once the initial marker fix succeeded
        self._has_moved = False
        self._warned_no_initial_fix = False

        # Live window (m3_display.py). NullDisplay does nothing, so the
        # navigation code can call it unconditionally; main() swaps in an
        # M3Display unless --no-display is given.
        self.display = NullDisplay()

        # Marker noise model for M3. slam/ekf.py's measurement_variance() was
        # tuned for M1 mapping and puts a 15-30 cm std dev on every marker
        # reading, which is so wide that update() barely moves the robot's
        # position at all (it only ever corrects ~2% of a position error). With
        # the map frozen there are no landmark estimates to protect, so M3 uses
        # std dev = base + per_m * distance instead (3 cm + 3 cm per metre by
        # default). Only this EKF instance is affected; ekf.py and operate.py are
        # unchanged. Pass marker_noise=None to keep M1's model.
        if marker_noise is not None:
            self.marker_sd_base, self.marker_sd_per_m = marker_noise
            self.ekf.measurement_variance = self._marker_variance
        # Same reasoning for the repeat-view penalty: update() multiplies a
        # sighting's noise by up to (1 + redundancy_penalty) = 21x when the
        # robot hasn't moved 15 cm or turned 15 deg since it last saw that
        # tag. That protects a landmark ESTIMATE from being over-trusted
        # while mapping. With the map frozen there is nothing to protect,
        # and relocalise()'s slow pan re-observes the same markers 10 deg
        # apart on purpose -- the penalty would throw most of that away.
        self.ekf.redundancy_penalty = 0.0

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    def _marker_variance(self, position):
        """(depth_var, lateral_var) for one marker reading -- same signature as
        EKF.measurement_variance(), and the same 1.5x lateral:depth ratio."""
        distance = float(np.linalg.norm(np.asarray(position, dtype=float).reshape(-1)[:2]))
        depth_var = (self.marker_sd_base + self.marker_sd_per_m * distance) ** 2
        return depth_var, depth_var * 1.5

    def get_robot_pose(self):
        """
        Returns the SLAM-corrected pose as np.array([x, y, theta]).

        Before the robot has moved: load_true_map() left the robot's
        uncertainty P at exactly zero, and a Kalman update() with zero P
        computes zero gain and changes nothing. So until the first move, solve
        the pose directly from the visible markers with recover_from_pause()
        (a best-fit alignment against the map; needs >= 2 known markers). If
        that never succeeds, the pose stays at the default (0, 0, 0) -- the
        centre of the arena, facing the map's +x.

        After that: the move itself was already predicted by turn() /
        drive_forward(), so this just folds in the markers visible now with
        update(). If there are >= 2 known markers in view and the update
        rejects every one of them as inconsistent with the predicted pose, the
        prediction has gone badly wrong, so re-anchor directly from the markers
        with recover_from_pause() instead.
        """
        img = self._latest_image()
        sensor_measurement, fruit_measurement = self._sense(img)
        n_known = sum(1 for lm in sensor_measurement if lm.tag in self.ekf.taglist)

        if not self._localised and not self._has_moved:
            if self.ekf.recover_from_pause(sensor_measurement):
                self._localised = True
                s = self.ekf.robot.state
                print("Initial pose from {} markers: [{:.3f}, {:.3f}, {:.1f}deg] (fit residual {:.3f} m)".format(
                    n_known, s[0, 0], s[1, 0], math.degrees(_normalize_angle(s[2, 0])), self.ekf.last_recover_residual))
            elif not self._warned_no_initial_fix:
                self._warned_no_initial_fix = True
                print("get_robot_pose: fewer than 2 known markers in view for the initial fix -- "
                      "using the start pose (0, 0) facing the map's +x. If the robot isn't at the "
                      "centre facing +x, press Ctrl+C and reposition it.")
        else:
            self._apply_measurements(sensor_measurement, fruit_measurement, allow_anchor=False)
            diagnostics = self.ekf.last_diagnostics
            if n_known >= 2 and diagnostics and all(d['gated'] for d in diagnostics):
                if self.ekf.recover_from_pause(sensor_measurement):
                    s = self.ekf.robot.state
                    print("get_robot_pose: markers disagreed with the predicted pose -- re-anchored "
                          "from {} markers to [{:.3f}, {:.3f}, {:.1f}deg]".format(
                              n_known, s[0, 0], s[1, 0], math.degrees(_normalize_angle(s[2, 0]))))

        state = self.ekf.robot.state
        return np.array([state[0, 0], state[1, 0], state[2, 0]])

    # ------------------------------------------------------------------
    # Active re-localisation
    # ------------------------------------------------------------------

    def camera_fov(self):
        """Horizontal field of view in radians, from the calibrated intrinsics.
        Works out at only ~31 deg on this camera, which is the whole reason
        relocalise() has to pan: standing still, the robot usually has one
        marker in frame or none, and a rigid pose fit needs two."""
        K = np.asarray(self.ekf.robot.camera_matrix, dtype=float)
        return 2.0 * math.atan(float(K[0, 2]) / float(K[0, 0]))

    # ------------------------------------------------------------------
    # Fruit landmarks: the true map's fruits as extra things to localise on
    # ------------------------------------------------------------------

    def load_fruit_landmarks(self, object_positions, object_dimensions):
        """
        Append the true map's fruit positions to the EKF's frozen landmark set.

        operate.py's FruitEKF runs the other way round (robot pose -> fruit
        positions). Here the fruit positions are GIVEN, so a YOLO detection of
        a fruit is a range+bearing observation of a known landmark -- exactly
        what an ArUco sighting is -- and update() can use it to correct the
        robot's pose through the very same maths. Each fruit gets a pseudo tag
        (FRUIT_TAG_BASE + i) so it lives in ekf.taglist alongside markers 1-10,
        and its covariance block is zero like theirs, which is what keeps it
        frozen (see load_true_map()).
        """
        for i, name in enumerate(sorted(object_positions)):
            x, y = object_positions[name]
            tag = FRUIT_TAG_BASE + i
            self.fruit_tags[name] = tag
            self.ekf.taglist.append(tag)
            self.ekf.markers = np.concatenate(
                [self.ekf.markers, np.array([[float(x)], [float(y)]])], axis=1)
        n = self.ekf.number_landmarks()
        P = np.zeros((3 + 2 * n, 3 + 2 * n))
        k = min(self.ekf.P.shape[0], P.shape[0])
        P[:k, :k] = self.ekf.P[:k, :k]
        self.ekf.P = P
        self.object_dimensions = dict(object_dimensions)

    def enable_fruit_detection(self, yolo_path):
        """Load the YOLO detector. Imported here, not at module level, so the
        script still runs without ultralytics installed (--yolo-path '')."""
        from cv.detector import ObjectDetector
        self.detector = ObjectDetector(yolo_path)

    def _detect_fruits(self, img):
        """
        YOLO boxes -> body-frame landmark measurements for fruits on the map.

        Same per-box gates as operate.py's process_object_estimates() (mostly
        inside the frame, plausible aspect ratio), then the same pinhole
        geometry as object_pose_est.estimate_pose() but stopped at the body
        frame: forward = f * true_height / box_height, left = -(x - cx) *
        forward / f. Tagged with noise_scale so update() trusts them less than
        a marker -- the range comes from a box height that jitters by pixels.
        Mis-classifications are left to update()'s chi-square gate: a "lemon"
        that is really a lime lands far from the lemon's mapped position and
        gets rejected there.
        """
        if self.detector is None or not self.fruit_tags:
            return []
        bboxes, _ = self.detector.detect_single_image(img)
        K = np.asarray(self.ekf.robot.camera_matrix, dtype=float)
        f, cx = float(K[0, 0]), float(K[0, 2])
        frame_h, frame_w = img.shape[0], img.shape[1]
        out = []
        for label, box in bboxes:
            if label not in self.fruit_tags or label not in self.object_dimensions:
                continue
            if in_frame_fraction(box, frame_w, frame_h) < IN_FRAME_FRACTION_THRESHOLD:
                continue
            if not is_box_shape_plausible(label, box, self.object_dimensions):
                continue
            x_c, _, _, box_h = [float(v) for v in box]
            if box_h <= 0:
                continue
            forward = f * self.object_dimensions[label][2] / box_h
            if forward > self.fruit_max_range:
                continue
            lateral = -(x_c - cx) * forward / f
            m = Marker(np.array([[forward], [lateral]]), self.fruit_tags[label])
            m.noise_scale = self.fruit_noise_scale
            out.append(m)
        return out

    def _sense(self, img):
        """@return: (aruco_measurements, fruit_measurements) for one frame."""
        aruco, _ = self.aruco_sensor.detect_marker_positions(img)
        return aruco, self._detect_fruits(img)

    def _apply_measurements(self, aruco, fruits, allow_anchor=True):
        """
        Fold one frame's sightings into the pose.

        anchor_min_markers ArUco markers in the same frame with a good rigid fit ->
        recover_from_pause() REPLACES the pose outright (the strongest fix we
        have, and the one that discards a bad dead-reckoned pose entirely).
        Otherwise everything -- markers and fruits -- goes through the Kalman
        update(), which blends. Fruits never enter the rigid fit: their range
        error would swamp the ArUco geometry in an unweighted least squares.

        @return: ('anchor' | 'update' | 'none', n_aruco_known, n_fruit)
        """
        n_ar = sum(1 for lm in aruco if lm.tag in self.ekf.taglist)
        if allow_anchor and n_ar >= self.anchor_min_markers:
            _, residual = self.ekf.preview_recovery(aruco)
            if residual is not None and residual <= self.relocalise_max_residual:
                self.ekf.recover_from_pause(aruco)
                if fruits:
                    self.ekf.update(fruits)
                return 'anchor', n_ar, len(fruits)
        if n_ar or fruits:
            self.ekf.update(list(aruco) + list(fruits))
            return 'update', n_ar, len(fruits)
        return 'none', 0, 0

    # ------------------------------------------------------------------
    # Active re-localisation: best direction, fast turn, slow pan
    # ------------------------------------------------------------------

    @staticmethod
    def _range_quality(d, profile):
        """0..1 worth of a landmark at range d, see marker_range_profile."""
        too_close, good_from, good_to, too_far = profile
        if d <= too_close or d >= too_far:
            return 0.0
        if d < good_from:
            return (d - too_close) / (good_from - too_close)
        if d > good_to:
            return (too_far - d) / (too_far - good_to)
        return 1.0

    def pose_uncertainty(self):
        """
        (sd_x, sd_y, sd_theta, weak_direction, anisotropy) from the EKF's own P.
        weak_direction is the world-frame angle along which position is least
        certain (major axis of the x-y covariance ellipse); anisotropy is the
        major/minor sd ratio, 1.0 meaning equally uncertain in every direction.
        """
        P = np.asarray(self.ekf.P, dtype=float)
        Pxy = P[0:2, 0:2]
        vals, vecs = np.linalg.eigh(0.5 * (Pxy + Pxy.T))
        major = int(np.argmax(vals))
        weak_dir = math.atan2(vecs[1, major], vecs[0, major])
        anis = math.sqrt(max(vals[major], 1e-12) / max(min(vals), 1e-12))
        return (math.sqrt(max(P[0, 0], 0.0)), math.sqrt(max(P[1, 1], 0.0)),
                math.sqrt(max(P[2, 2], 0.0)), weak_dir, anis)

    def best_view_heading(self, pose, n=2, min_separation=np.deg2rad(60.0), exclude=(),
                          weak_direction=None):
        """
        Score every heading by how much localisation it would buy, from the
        true map and the robot's rough position, and return the top `n` as
        [(heading_rad, score)] best first, each at least `min_separation`
        from the ones before it and from any heading in `exclude` (directions
        already scanned this round -- looking there again buys nothing).

        Score(h) = sum over landmarks of  w * q(range) * a * exp(-0.5 (delta/sd)^2)
                   + view_pair_bonus * (landmarks in frame at once - 1)
                   - view_turn_penalty * |turn needed|
        where delta is the bearing offset from h, w = 1 for a marker and
        fruit_view_weight for a fruit, q the range quality (a sweet spot, NOT
        1/range: the nearest landmark is usually the worst one to stare at --
        see marker_range_profile), and a is an axis-alignment weight, 1.0
        unless `weak_direction` is given. Then landmarks whose bearing runs
        along that direction are favoured, because a landmark's RANGE pins the
        robot along the line to it -- so a view down the uncertain axis is the
        one that fixes it, and a view across it mostly repeats what is known.

        No hard field-of-view cut on the kernel term: the +/- pan_half sweep
        that follows is what gets things at the edges into frame. Fruits are
        ignored unless a detector is loaded.
        """
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        headings = np.deg2rad(np.arange(-180.0, 180.0, 5.0))
        scores = np.zeros_like(headings)
        in_view = np.zeros_like(headings)
        half_fov = self.camera_fov() / 2.0
        markers = np.asarray(self.ekf.markers, dtype=float)
        for i, tag in enumerate(self.ekf.taglist):
            is_fruit = int(tag) >= FRUIT_TAG_BASE
            if is_fruit and self.detector is None:
                continue
            dx, dy = markers[0, i] - x, markers[1, i] - y
            d = math.hypot(dx, dy)
            q = self._range_quality(d, self.fruit_range_profile if is_fruit else self.marker_range_profile)
            if q <= 0.0:
                continue
            bearing = math.atan2(dy, dx)
            w = self.fruit_view_weight if is_fruit else 1.0
            align = 1.0
            if weak_direction is not None:
                align = 0.3 + 0.7 * abs(math.cos(bearing - weak_direction))
            delta = (headings - bearing + math.pi) % (2 * math.pi) - math.pi
            scores += w * q * align * np.exp(-0.5 * (delta / self.view_kernel_sd) ** 2)
            in_view += q * (np.abs(delta) <= half_fov)
        scores += self.view_pair_bonus * np.maximum(0.0, in_view - 1.0)
        turn_cost = np.abs((headings - theta + math.pi) % (2 * math.pi) - math.pi)
        scores -= self.view_turn_penalty * turn_cost

        ranked = []
        for idx in np.argsort(-scores):
            h = float(headings[idx])
            if any(abs(_normalize_angle(h - e)) < min_separation for e in exclude):
                continue
            if all(abs(_normalize_angle(h - other)) >= min_separation for other, _ in ranked):
                ranked.append((h, float(scores[idx])))
            if len(ranked) >= n:
                break
        return ranked

    def _pan_about(self, centre_heading):
        """
        Fast turn to one edge of the window about `centre_heading`, then step
        slowly across it, sensing markers and fruits at every step. Starts
        from whichever edge is nearer the current heading.
        @return: dict(steps, hits, anchors) -- hits = steps that saw >= 1 landmark
        """
        theta = float(self.ekf.robot.state[2, 0])
        lo, hi = centre_heading - self.pan_half, centre_heading + self.pan_half
        if abs(_normalize_angle(lo - theta)) <= abs(_normalize_angle(hi - theta)):
            first, step = lo, abs(self.pan_step)
        else:
            first, step = hi, -abs(self.pan_step)
        n_steps = max(1, int(round(2 * self.pan_half / abs(self.pan_step))))

        self.turn(_normalize_angle(first - theta))               # quick, at normal speed
        hits = anchors = 0
        for k in range(n_steps + 1):
            if k > 0:
                self.turn(step, speed=self.pan_turn_speed)        # slow, small
            img = self._latest_image()
            aruco, fruits = self._sense(img)
            kind, n_ar, n_fr = self._apply_measurements(aruco, fruits)
            if kind != 'none':
                hits += 1
            if kind == 'anchor':
                anchors += 1
            s = self.ekf.robot.state
            strong = kind == 'anchor' or (n_ar + n_fr) >= 2
            sd_x, sd_y, sd_t, _, _ = self.pose_uncertainty()
            converged = (sd_x <= self.converge_pos_sd and sd_y <= self.converge_pos_sd
                         and sd_t <= self.converge_ang_sd)
            print("  pan {:+6.1f}deg: {} marker(s) {} fruit(s) -> {:<6s} [{:.3f}, {:.3f}, {:.1f}deg]".format(
                math.degrees(_normalize_angle(float(s[2, 0]))), n_ar, n_fr, kind,
                s[0, 0], s[1, 0], math.degrees(_normalize_angle(float(s[2, 0])))))
            if hits >= 2 and (strong or converged):
                # A frame with two landmarks pins x, y AND heading, and a
                # covariance already under the convergence thresholds means
                # the filter agrees. Either way the rest of the sweep is only
                # cost -- and every extra turn is its own source of error.
                return {'steps': k + 1, 'hits': hits, 'anchors': anchors}
        return {'steps': n_steps + 1, 'hits': hits, 'anchors': anchors}

    def _inflate_pose_covariance(self):
        """
        Raise the pose covariance to an honest post-move level before scanning.

        Without this the scan mostly achieves nothing. A Kalman update weights
        the markers by P / (P + R): _predict_commanded_motion() adds only
        drive_noise_frac (5%) of the distance driven, leaving P around 1.5 cm,
        while the marker noise model gives R around 5 cm. That puts the weight
        near 0.08, so each update closes under a tenth of the error and the
        filter keeps believing its own dead reckoning.

        A 5% drive error is also just not what the robot does -- measured
        overshoot on straight legs is far larger than that. So on arrival,
        floor P at what the pose is really worth after an open-loop move.
        Only a floor: if P is already larger, it is left alone.

        This matters most where two markers never share a frame, which is the
        normal case at this FOV -- the correction then has to come from
        single-marker updates taken from different headings, and those only
        work if the filter is willing to move.
        """
        pos_sd, theta_sd = self.relocalise_prior_sd
        self.ekf.P[0, 0] = max(self.ekf.P[0, 0], pos_sd ** 2)
        self.ekf.P[1, 1] = max(self.ekf.P[1, 1], pos_sd ** 2)
        self.ekf.P[2, 2] = max(self.ekf.P[2, 2], theta_sd ** 2)

    def relocalise(self):
        """
        Re-anchor the pose against the true map after a move, and keep going
        until the pose has actually converged.

        Each round:
          1. Read the EKF's own uncertainty (pose_uncertainty()). If every
             axis is under converge_pos_sd / converge_ang_sd, stop: converged.
          2. best_view_heading(): pick the most informative direction not yet
             scanned. When the uncertainty is lopsided, favour landmarks that
             lie along the uncertain axis -- their range is what pins it.
          3. Turn there quickly, pan slowly across it (_pan_about), folding
             every frame in: 3+ markers -> rigid re-anchor, else Kalman update.

        Gives up after max_scan_rounds and reports which axis is still loose;
        run_level1() then drives a shorter leg. last_relocalise_converged
        records the outcome for callers.

        @return: the pose afterwards, np.array([x, y, theta])
        """
        state = self.ekf.robot.state
        pose = np.array([state[0, 0], state[1, 0], state[2, 0]])
        self.last_relocalise_converged = False
        if self.ekf.number_landmarks() == 0:
            return pose

        self._inflate_pose_covariance()
        scanned = []
        total_hits = 0
        for round_no in range(1, self.max_scan_rounds + 1):
            sd_x, sd_y, sd_t, weak_dir, anis = self.pose_uncertainty()
            if round_no > 1 and sd_x <= self.converge_pos_sd and sd_y <= self.converge_pos_sd \
                    and sd_t <= self.converge_ang_sd:
                self.last_relocalise_converged = True
                break

            state = self.ekf.robot.state
            pose = np.array([state[0, 0], state[1, 0], state[2, 0]])
            use_weak = weak_dir if (round_no > 1 and anis > 1.4) else None
            candidates = self.best_view_heading(pose, n=1, exclude=scanned, weak_direction=use_weak)
            if not candidates or candidates[0][1] <= 0:
                print("  relocalise: no useful direction left to scan")
                break
            heading, score = candidates[0]
            why = ""
            if use_weak is not None:
                why = " -- uncertainty is {:.1f}x larger along {:+.0f}deg, looking that way".format(
                    anis, math.degrees(_normalize_angle(weak_dir)))
            print("  relocalise round {}: sd x {:.0f} cm, y {:.0f} cm, heading {:.0f} deg -> scan {:+.0f}deg "
                  "(score {:.2f}){}".format(round_no, sd_x * 100, sd_y * 100, math.degrees(sd_t),
                                            math.degrees(_normalize_angle(heading)), score, why))
            result = self._pan_about(heading)
            scanned.append(heading)
            total_hits += result['hits']
        else:
            sd_x, sd_y, sd_t, _, _ = self.pose_uncertainty()
            self.last_relocalise_converged = (sd_x <= self.converge_pos_sd and sd_y <= self.converge_pos_sd
                                              and sd_t <= self.converge_ang_sd)

        sd_x, sd_y, sd_t, _, _ = self.pose_uncertainty()
        if total_hits == 0:
            print("  relocalise: no markers or fruits seen in any direction -- pose is still dead-reckoned")
        elif self.last_relocalise_converged:
            print("  relocalise: converged (sd x {:.0f} cm, y {:.0f} cm, heading {:.0f} deg) after {} scan(s)".format(
                sd_x * 100, sd_y * 100, math.degrees(sd_t), len(scanned)))
        else:
            loose = [n for n, v, t in (("x", sd_x, self.converge_pos_sd), ("y", sd_y, self.converge_pos_sd),
                                       ("heading", sd_t, self.converge_ang_sd)) if v > t]
            print("  relocalise: NOT converged after {} scan(s) -- still loose on {} "
                  "(sd x {:.0f} cm, y {:.0f} cm, heading {:.0f} deg); next leg will be short".format(
                      len(scanned), ", ".join(loose), sd_x * 100, sd_y * 100, math.degrees(sd_t)))

        state = self.ekf.robot.state
        return np.array([state[0, 0], state[1, 0], state[2, 0]])

    def _latest_image(self):
        # Right after connecting, the camera thread may not have delivered a
        # frame yet, and get_image() would return a blank image with no markers.
        deadline = time.time() + 5.0
        while getattr(self.botconnect, 'frame', True) is None and time.time() < deadline:
            time.sleep(0.05)
        return self.botconnect.get_image()

    def _predict_commanded_motion(self, wheel_speeds, dt, dtheta=0.0, distance=0.0, completed=True):
        """Feed the EKF one predict() step for a move the robot was just told to
        make: an in-place turn of `dtheta` rad and/or a straight drive of
        `distance` m."""
        tpm = self.ticks_per_meter
        b_model = self.ekf.robot.baseline  # signed, exactly as Robot.drive() uses it
        # Wheel displacements (in ticks) in the motion model's own convention.
        # Robot.drive() turns (d_right - d_left) / baseline into rotation, so
        # this reproduces exactly `dtheta` and `distance` whatever sign
        # baseline.txt carries.
        d_right = distance * tpm + dtheta * b_model * tpm / 2.0
        d_left = distance * tpm - dtheta * b_model * tpm / 2.0
        drive_measurement = DriveMeasurement(wheel_speeds[0], wheel_speeds[1], max(dt, 0.05),
                                             delta_left_ticks=d_left, delta_right_ticks=d_right)
        self.ekf.predict(drive_measurement)

        # Widen the pose uncertainty to match how far off a commanded move can be.
        xy_sd = self.drive_noise_frac * abs(distance)
        th_sd = self.turn_noise_frac * abs(dtheta)
        if not completed:
            # Timed out and force-stopped: no idea how much of the move happened.
            xy_sd += 0.10
            th_sd += np.deg2rad(20.0)
        self.ekf.P[0, 0] += xy_sd ** 2
        self.ekf.P[1, 1] += xy_sd ** 2
        self.ekf.P[2, 2] += th_sd ** 2
        self._has_moved = True

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------

    def turn(self, dtheta, speed=None):
        """In-place turn by `dtheta` radians (positive = anticlockwise/left,
        matching robot.py's state[2] convention and operate.py's K_LEFT
        binding of [-0.35, 0.35]). One tick is the smallest possible turn
        (about 4.8 deg), so anything under half a tick is skipped. `speed`
        overrides turn_speed (relocalise() pans slowly with it).
        @return: the rotation the EKF was told happened (0.0 if skipped)"""
        arc_length = (self.wheel_separation / 2.0) * abs(dtheta)
        ticks = int(round(arc_length * self.ticks_per_meter * self.turn_scale))
        if ticks < 1:
            return 0.0
        magnitude = self.turn_speed if speed is None else abs(float(speed))
        signed = magnitude if dtheta > 0 else -magnitude
        wheel_speeds = [-signed, signed]
        start = time.time()
        self.botconnect.move_auto_encoder(wheel_speeds, ticks, ticks)
        completed = self._wait_for_move()
        # the rotation those whole ticks are expected to produce
        commanded = math.copysign(
            2.0 * ticks / (self.ticks_per_meter * self.wheel_separation * self.turn_scale), dtheta)
        self._predict_commanded_motion(wheel_speeds, time.time() - start,
                                       dtheta=commanded, completed=completed)
        self._settle()
        return commanded

    def drive_forward(self, distance):
        """Drive straight forward by `distance` metres (>= 0)."""
        ticks = int(round(distance * self.ticks_per_meter))
        if ticks < 1:
            return
        wheel_speeds = [self.drive_speed, self.drive_speed]
        start = time.time()
        self.botconnect.move_auto_encoder(wheel_speeds, ticks, ticks)
        completed = self._wait_for_move()
        self._predict_commanded_motion(wheel_speeds, time.time() - start,
                                       distance=ticks / self.ticks_per_meter, completed=completed)
        self._settle()

    def _wait_for_move(self):
        """Wait for the robot to finish a move_auto_encoder() command.
        @return: True if it finished, False if it timed out and was force-stopped"""
        start = time.time()
        completed = True
        while not self.botconnect.autonomous_done:
            if time.time() - start > self.move_timeout:
                print("WARNING: move exceeded {:.0f}s timeout -- stopping".format(self.move_timeout))
                self.botconnect.stop()
                completed = False
                break
            self.display.idle(0.02)   # keeps the window updating while the robot moves
        return completed

    def _settle(self):
        """Give the camera time to deliver a frame taken AFTER the robot
        stopped, so the next marker update isn't comparing the new pose with a
        blurred frame from mid-move. (Called after the move has been fed to
        the EKF, so the map already shows where the robot should now be.)"""
        self.display.idle(self.settle_time)


def _normalize_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
def drive_to_point(waypoint, nav, max_distance=None):
    """
    Turn-then-drive to `waypoint` = (x, y), using nav's SLAM-corrected pose to
    aim and nav's calibrated encoder-based driving to execute. After turning,
    re-checks the pose and corrects the heading (up to
    nav.max_heading_corrections times) before committing to the straight
    line, since a 10 deg heading error puts the robot 17 cm off course after
    1 m. Checks the pose once more after arriving, per the manual's own
    recommendation to correct pose after every waypoint step.
    @param max_distance: cap on the straight drive (m). run_level1() caps legs
        so overshoot stays bounded, but the distance is recomputed here AFTER
        the heading corrections, and the pose update in between can lengthen
        it -- so the cap has to be applied here too.
    @return: robot pose (np.array([x, y, theta])) after arriving
    """
    nav.display.set_waypoint(waypoint)
    pose = nav.get_robot_pose()
    if math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1]) < nav.min_move:
        return pose  # already there -- and atan2 of a ~zero vector is a random heading

    for attempt in range(1 + nav.max_heading_corrections):
        target_heading = math.atan2(waypoint[1] - pose[1], waypoint[0] - pose[0])
        dtheta = _normalize_angle(target_heading - pose[2])
        if attempt > 0 and abs(dtheta) <= nav.heading_tolerance:
            break
        nav.turn(dtheta)
        pose = nav.get_robot_pose()

    distance = math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1])
    if max_distance is not None:
        distance = min(distance, max_distance)
    nav.drive_forward(distance)

    pose = nav.get_robot_pose()
    print("Arrived near [{:.3f}, {:.3f}] -- pose now [{:.3f}, {:.3f}, {:.1f}deg]".format(
        waypoint[0], waypoint[1], pose[0], pose[1], math.degrees(_normalize_angle(pose[2]))))

    # Pan across the nearest markers and re-anchor before the next leg is
    # planned, so the heading and distance for that leg are computed from a
    # measured pose rather than an accumulated one.
    if nav.relocalise_enabled:
        pose = nav.relocalise()
        print("  after re-localising: [{:.3f}, {:.3f}, {:.1f}deg]".format(
            pose[0], pose[1], math.degrees(_normalize_angle(pose[2]))))
    return pose


def get_robot_pose(nav):
    """Thin wrapper matching the manual's function name -- see
    Navigator.get_robot_pose() in this file for the actual SLAM logic."""
    return nav.get_robot_pose()


def _escape_obstacles(point, obstacles, bounds=path_planner.ARENA_BOUNDS, clearance=0.02):
    """
    rrt_star() refuses to start from inside an inflated obstacle circle, and
    pose noise can put the SLAM estimate a centimetre or two inside a
    neighbour's circle after parking. Return the nearest collision-free point
    to plan from instead: `point` itself if it's already free, otherwise the
    closest of (a) the point pushed straight out of each circle it's in and
    (b) a ring search out to 0.4 m. Returns None if nothing free was found.
    """
    p = np.asarray(point, dtype=float)
    if not path_planner.point_in_collision(p, obstacles):
        return tuple(p)

    (xmin, xmax), (ymin, ymax) = bounds
    candidates = []
    for ox, oy, r in obstacles:
        centre = np.array([ox, oy])
        v = p - centre
        d = float(np.hypot(v[0], v[1]))
        if d <= r:
            u = v / d if d > 1e-9 else np.array([1.0, 0.0])
            candidates.append(centre + u * (r + clearance))
    for radius in np.arange(0.05, 0.401, 0.05):
        for k in range(16):
            a = 2 * math.pi * k / 16
            candidates.append(p + radius * np.array([math.cos(a), math.sin(a)]))

    free = [c for c in candidates
            if xmin <= c[0] <= xmax and ymin <= c[1] <= ymax
            and not path_planner.point_in_collision(c, obstacles)]
    if not free:
        return None
    return tuple(min(free, key=lambda c: path_planner.dist_between(c, p)))


def run_level1(nav, search_list, object_list, object_true_pos, aruco_true_pos,
               planner='astar', grid_resolution=0.05, max_legs=20, goal_tolerance=0.05,
               max_leg_length=0.25, safety_margin=0.10, found_radius=0.35,
               clearance_pref=0.15, stall_legs=4):
    """
    M3 Level 1: full known map given, navigate to every search_list.txt target
    IN ORDER, then wait for the demonstrator's verification before moving on
    (the manual's 4.4.1 checklist).

    The route is re-planned from scratch after EVERY waypoint: drive_to_point()
    ends with relocalise(), which measures where the robot actually is, and
    the next A* plan starts from that measured pose rather than from where
    the previous plan assumed the robot would be. The occupancy grid is built
    once per target (obstacles don't change within one), so each re-plan is
    a few milliseconds. The standoff goal is chosen once per target and kept,
    so successive plans converge on one point instead of chasing a moving one.

    Each leg is capped at max_leg_length. Re-localising only tells the robot
    where it ended up; it cannot un-drive an overshoot. Straight-line drives
    on this robot have measured up to ~40% long, and the final approach runs
    straight at the fruit and stops 0.3 m short, so a long last leg carries
    the robot into the fruit before anything can correct it. With 0.25 m legs
    the worst overshoot is ~0.1 m, inside the physical clearance, and the
    pose gets re-measured that much more often.

    safety_margin inflates every marker and fruit circle beyond the geometric
    robot+object sum, to absorb pose error. found_radius ends a target as
    soon as the measured pose is that close to the fruit -- whether or not
    the standoff point was reached. An overshoot that lands inside the
    scoring radius is a success; driving back out to the standoff point
    would only be another chance to overshoot.

    goal_tolerance stacks with the 0.3 m standoff: a leg may end that far
    short of the standoff point, so 0.05 caps the believed finish at 0.35 m
    from the fruit -- the same as found_radius -- leaving 5 cm of the 0.4 m
    scoring radius for pose error. (0.10 allowed a believed 0.40 m finish,
    which a 1 cm pose error turned into a miss.)

    A pose inside a safety circle is planned FROM, not escaped from: the
    planner shrinks that circle to the robot's current depth so the route
    leads out along the way to the goal (see OccupancyGrid.relaxed_blocked).
    The old escape hop -- drive to the nearest free point first -- is what
    ping-ponged the robot between two close obstacles. clearance_pref is
    the berth A* pays to keep; stall_legs ends a target that has made no
    progress for that many legs rather than let it oscillate to max_legs.
    """
    object_radii = path_planner.load_object_radii()
    object_positions = {name: tuple(pos) for name, pos in zip(object_list, object_true_pos)}
    display = nav.display

    def skip(name, reason):
        print(f"[{name}] {reason} -- skipping")
        display.notify(f"{name}: {reason} -- skipped")
        display.target_done(name, None, False)

    for i, target_name in enumerate(search_list, start=1):
        if target_name not in object_positions:
            print(f"WARNING: '{target_name}' from search_list.txt isn't in the true map -- skipping")
            display.target_done(target_name, None, False)
            continue
        target = object_positions[target_name]

        obstacles = path_planner.build_obstacles(
            aruco_true_pos, object_positions, path_planner.ROBOT_RADIUS,
            object_radii=object_radii, safety_margin=safety_margin, exclude=target_name)

        grid = (path_planner.OccupancyGrid(obstacles, resolution=grid_resolution, clearance_pref=clearance_pref)
                if planner == 'astar' else None)
        display.begin_target(i, target_name, target, obstacles)
        display.notify(f"Planning route to {target_name} ({i}/{len(search_list)})")
        display.refresh(force=True)

        pose = nav.get_robot_pose()
        if nav.relocalise_enabled and not nav.last_relocalise_converged:
            pose = nav.relocalise()   # start every target from a converged pose
        goal = None
        failure = None
        best_remaining = math.inf
        stalled = 0
        for leg in range(1, max_legs + 1):
            d_target = math.hypot(pose[0] - target[0], pose[1] - target[1])
            if d_target <= found_radius:
                print(f"[{target_name}] already {d_target:.2f} m from the fruit (<= {found_radius:.2f}) -- stopping here")
                break
            start = (float(pose[0]), float(pose[1]))
            inside = path_planner.circles_containing(start, obstacles)
            if inside:
                deepest = max(depth for _, depth in inside)
                print(f"[{target_name}] pose is inside {len(inside)} safety circle(s), deepest by {deepest * 100:.0f} cm "
                      f"-- planning a route that leaves it, not an escape hop")

            if goal is None:
                goal = path_planner.standoff_point(target, obstacles, reference=start)
                if goal is None:
                    failure = "no collision-free standoff point"
                    break

            remaining = path_planner.dist_between(start, goal)
            if remaining <= goal_tolerance:
                break

            if leg > 1 and remaining > best_remaining - 0.02:
                stalled += 1
                if stalled >= stall_legs:
                    failure = f"no progress towards the standoff point for {stall_legs} legs (oscillating?)"
                    break
            else:
                stalled = 0
            best_remaining = min(best_remaining, remaining)

            path = path_planner.plan(start, goal, obstacles, planner=planner, grid=grid)
            if path is None and planner != 'astar':
                # RRT* refuses a start inside a circle; only then fall back to an escape point.
                alt = _escape_obstacles(start, obstacles)
                if alt is not None:
                    path = path_planner.plan(alt, goal, obstacles, planner=planner, grid=grid)
                    if path is not None:
                        path = [start] + list(path)
            if path is None:
                failure = f"{planner} found no route from [{start[0]:.2f}, {start[1]:.2f}]"
                break
            path = path_planner.smooth_path(path, obstacles, min_clearance=clearance_pref)
            # path[0] is the planning start; only drive to it if an escape moved it.
            waypoints = path[1:] if path_planner.dist_between(path[0], pose[:2]) < nav.min_move else path
            if not waypoints:
                break
            next_wp = waypoints[0]
            leg_len = path_planner.dist_between(start, next_wp)
            # Half-length legs while the pose hasn't converged: less distance
            # for a wrong heading to turn into a wrong position.
            leg_cap = max_leg_length if nav.last_relocalise_converged or not nav.relocalise_enabled else max_leg_length / 2
            if leg_len > leg_cap:
                f = leg_cap / leg_len
                next_wp = (start[0] + f * (next_wp[0] - start[0]), start[1] + f * (next_wp[1] - start[1]))
                waypoints = [next_wp] + list(waypoints)

            display.set_route(goal, waypoints)
            display.notify(f"{target_name} ({i}/{len(search_list)}): leg {leg}, {remaining:.2f} m to go")
            print(f"\n[{target_name}] leg {leg}: {remaining:.2f} m to the standoff point, "
                  f"{len(waypoints)}-waypoint plan -> driving to [{waypoints[0][0]:.2f}, {waypoints[0][1]:.2f}]")
            pose = drive_to_point(next_wp, nav, max_distance=leg_cap)   # ends with relocalise(); re-plan from there
        else:
            print(f"[{target_name}] still {path_planner.dist_between(pose[:2], goal):.2f} m from the "
                  f"standoff point after {max_legs} legs -- reporting where it got to")

        if failure is not None:
            skip(target_name, failure)
            continue

        final_pose = nav.get_robot_pose()
        dist_to_target = math.hypot(final_pose[0] - target[0], final_pose[1] - target[1])
        status = "OK" if dist_to_target <= 0.4 else "OUT OF TOLERANCE"
        print(f"=== Found {target_name} at [{target[0]:.2f}, {target[1]:.2f}] "
              f"(robot is {dist_to_target:.3f}m away -- {status}) ===")
        display.target_done(target_name, dist_to_target, dist_to_target <= 0.4)
        display.wait_for_key(f"Found {target_name} ({dist_to_target:.2f} m). ENTER when verified")


def run_manual(nav):
    """Manual waypoint entry -- for demonstrating drive_to_point()/get_robot_pose()
    alone, per the manual's 4.1 'command line waypoint input' suggestion."""
    while True:
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        pose = drive_to_point([x, y], nav)
        print("Finished driving to waypoint: {}; New robot pose: {}".format([x, y], pose))

        uInput = input("Add a new waypoint? [Y/N]")
        if uInput.lower() == 'n':
            break


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Object searching")
    parser.add_argument("--map", type=str, default='truemap.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--calib-dir", type=str, default='calibration/param/')
    parser.add_argument("--manual", action="store_true",
                         help="manual waypoint entry instead of the full Level 1 search_list run")
    parser.add_argument("--no-display", action="store_true",
                         help="terminal only: no camera/map window and no setup phase")
    parser.add_argument("--turn-scale", type=float, default=None,
                         help="override the measured turn_scale from "
                              "calibration/param/turn_scale.txt (see calibrate_turn.py)")
    parser.add_argument("--no-relocalise", action="store_true",
                         help="don't turn-and-pan to re-anchor the pose after each waypoint "
                              "(faster, but errors accumulate)")
    parser.add_argument("--pan-half", type=float, default=20.0,
                         help="half-width of the slow pan about the best view direction, deg")
    parser.add_argument("--pan-step", type=float, default=5.0,
                         help="slow-pan step size, deg (one encoder tick is ~5 deg)")
    parser.add_argument("--yolo-path", type=str, default='cv/model/best.pt',
                         help="YOLO weights for fruit-based localisation; '' disables it")
    parser.add_argument("--no-fruit-localise", action="store_true",
                         help="don't use fruit detections to correct the robot pose")
    parser.add_argument("--planner", choices=['astar', 'rrt'], default='astar',
                         help="path planner (default A*; rrt keeps the old RRT* for comparison)")
    parser.add_argument("--grid-res", type=float, default=0.05, help="A* cell size, m")
    parser.add_argument("--max-leg", type=float, default=0.25,
                         help="longest single drive between re-localisations, m")
    parser.add_argument("--safety-margin", type=float, default=0.10,
                         help="extra clearance around every marker and fruit beyond robot+object radius, m")
    parser.add_argument("--found-radius", type=float, default=0.35,
                         help="stop a target as soon as the pose is this close to the fruit, m")
    parser.add_argument("--clearance", type=float, default=0.15,
                         help="berth A* prefers to keep from every safety circle, m (priced, not forbidden)")
    args, _ = parser.parse_known_args()

    botconnect = BotConnect(args.ip)
    time.sleep(1)  # give connection threads a moment to establish
    botconnect.set_pid(use_pid=1, kp=2, ki=0.04, kd=0.29)  # same gains as operate.py

    ekf, baseline = init_ekf(args.calib_dir)
    ekf.load_true_map(args.map)  # freezes markers, M3 Level 1/2/3 all localise-only
    aruco_sensor = ArucoSensor(ekf.robot, marker_length=0.06)

    turn_scale = args.turn_scale if args.turn_scale is not None else load_turn_scale()
    nav = Navigator(botconnect, ekf, aruco_sensor, baseline, turn_scale=turn_scale)
    nav.relocalise_enabled = not args.no_relocalise
    nav.pan_half = np.deg2rad(args.pan_half)
    nav.pan_step = np.deg2rad(args.pan_step)

    # read in the true map
    object_list, object_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_object_pos(search_list, object_list, object_true_pos)
    object_positions = {name: tuple(pos) for name, pos in zip(object_list, object_true_pos)}

    # Fruits from the true map become landmarks too; whether they're SEEN
    # depends on a detector being loaded.
    nav.load_fruit_landmarks(object_positions, load_object_dimensions())
    if args.yolo_path and not args.no_fruit_localise:
        try:
            nav.enable_fruit_detection(args.yolo_path)
        except Exception as e:
            print(f"Fruit detector not loaded ({e}) -- localising on ArUco markers only.")
    print("Re-localisation after each waypoint: {} (camera FOV {:.0f}deg); landmarks: {} markers + {} fruits{}".format(
        "on" if nav.relocalise_enabled else "off", math.degrees(nav.camera_fov()),
        sum(1 for t in ekf.taglist if t < FRUIT_TAG_BASE), len(nav.fruit_tags),
        "" if nav.detector is not None else " (fruits not detected: no YOLO model)"))
    print(f"Planner: {args.planner}" + (f" (grid {args.grid_res} m)" if args.planner == 'astar' else ""))

    display = None
    try:
        if not args.no_display:
            object_positions = {name: tuple(pos) for name, pos in zip(object_list, object_true_pos)}
            display = M3Display(nav, aruco_true_pos, object_positions, search_list,
                                object_radii=path_planner.load_object_radii())
            # (begin_target() hands the display the planner's inflated circles per target)
            nav.display = display
            # Turn with the arrow keys until 2+ markers are in view, then ENTER.
            display.run_setup()

        if args.manual:
            if display is not None:
                display.run_manual(lambda wp: drive_to_point(wp, nav))
            else:
                run_manual(nav)
        else:
            run_level1(nav, search_list, object_list, object_true_pos, aruco_true_pos,
                       planner=args.planner, grid_resolution=args.grid_res,
                       max_leg_length=args.max_leg, safety_margin=args.safety_margin,
                       found_radius=args.found_radius, clearance_pref=args.clearance)
            print("Level 1 run finished.")
            if display is not None:
                display.finish("Run finished. ESC to exit")
    except (M3Abort, KeyboardInterrupt):
        print("Stopped by user.")
    finally:
        botconnect.stop()
        if display is not None:
            import pygame
            pygame.quit()
