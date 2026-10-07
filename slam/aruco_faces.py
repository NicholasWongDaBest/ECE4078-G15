"""
slam/aruco_faces.py -- what an ArUco block's FACE tells us beyond its position.

Shared by mapping (operate.py) and navigation (auto_fruit_search.py), so the
two always measure markers the same way. Settings: calibration/param/aruco_faces.json
(missing file or key -> the defaults in DEFAULTS below).

Background
----------
Every marker is a cube with the same tag printed on its sides, and every block
sits square to the arena: its faces point along the arena's +x / +y / -x / -y.
For each detected face, slam/aruco_sensor.py keeps BOTH pose solutions of the
square-marker solve (cv2.solvePnPGeneric, IPPE_SQUARE). A single small marker
has two mirror-image poses that fit the 4 corners almost equally well, so the
one with the lower reprojection error is not always the right one. From those,
this module works out:

  * which way the face points. The axis rule picks the solution whose face
    direction is nearer an arena axis, unless the reprojection errors clearly
    disagree. Two faces of the same block in one frame must be perpendicular,
    which settles the case the axis rule can't: far away at ~45 deg, where
    the mirror solution lands exactly on the other axis.
  * the printed rotation k (0..3, counter-clockwise as you look at the face).
    On blocks whose faces are printed at different rotations, (tag, k) names
    the face: the run remembers which arena axis each one points along, and a
    remembered face can put a heading that has gone badly wrong (> 45 deg, past
    what snapping to the nearest axis can fix) back right.

and uses it twice:

  1. BLOCK CENTRES. The pose solve measures the centre of the face it sees,
     but maps (truemap.txt, the slam.txt the marking evaluates) hold block
     centres. Each face reading is pushed half a block into the block, along
     the face's normal, before it reaches the EKF. Two faces of one block in
     the same frame are averaged as block centres (the old code averaged the
     two FACE centres, which lands off towards the corner).

  2. HEADING. The face's arena axis minus the direction the robot sees it
     pointing is the robot's heading, from one marker, without knowing the
     position first. Fused as a scalar Kalman update on theta (over the full
     covariance, so x, y and any landmarks follow their correlations).

The arena axes in the map frame
-------------------------------
"Faces point along the arena axes" only helps if we know where the arena axes
are in the map frame. The map frame is the robot's START pose in the mapping
run, which is square to the arena only if the robot was placed square. So
the axes' rotation (alpha0, between -45 and 45 deg) is LEARNT: from frames
whose heading is known independently of the faces (the start of a mapping
run, where the heading is exactly 0 by definition; or, in navigation, a frame
with two or more known markers, whose heading the marker geometry fixes).
Heading fusion waits until alpha0 is known well enough (axes_ready_sd_deg,
from at least axes_min_frames spots). One sample per spot the robot stops at:
frames at one spot share the same heading error, so the best frame there (the
surest heading) stands for the spot. Set "axes_deg" in the JSON, or
auto_fruit_search.py --arena-axes 0, to fix it instead -- right for runs on
truemap.txt, whose frame IS the arena's. Only faces whose mirror solution was
ruled out teach it, i.e. faces seen at least ~10 deg off face-on; a mapping run
whose first view shows its blocks only face-on can't learn it at the start.

Distortion polynomial
---------------------
operate.py corrects marker positions with distortion_correction.json, a
polynomial fitted to face-on readings (x_meas, y_meas -> tape-measured x, y).
With faces on, that same polynomial is applied HERE, in both programs, so
navigation measures markers exactly as the map was built:
  - the polynomial first, to the raw face-centre reading (what it was fitted on);
  - then the face shift. "polynomial_includes_face_offset": true means the
    tape-measured points were block centres, so a face-on reading already
    comes out at the block centre and only the part of the shift that depends
    on the viewing angle is added. false (points measured to the face):
    the whole half-block shift is added.
With no polynomial file, "camera_offset" (camera ahead of the wheel axle, m)
is added instead; the polynomial's constant term already contains it.
"""

import os
import json
import math

import numpy as np
import cv2


DEFAULTS = {
    # Master switch. false = the old behaviour everywhere: face-centre readings,
    # OpenCV's default detector settings, no heading from faces.
    "faces": True,
    # Readings become block centres (mapping AND navigation -- keep them the same,
    # a map built one way and navigated the other is off by half a block).
    "block_centres": True,
    "block_size": 0.06,              # block edge, m (truemap_generator uses 0.06) -- MEASURE
    "marker_length": None,           # black square edge, m; null = what the program passes (0.06) -- MEASURE
    "camera_offset": 0.0,            # m camera ahead of the wheel axle; only used with no polynomial -- MEASURE
    "polynomial": "distortion_correction.json",   # "" = none
    "polynomial_includes_face_offset": True,
    # Detector (faces on only). subpix: sub-pixel corners. 0.05 keeps both faces
    # of a block at a corner; OpenCV 4.13's default 0.125 throws one away.
    "refine": "subpix",
    "min_marker_distance": 0.05,
    # Mirror-solution choice (same rules as aruco_orientation_test.py).
    "tie_deg": 15.0,
    "clear_ratio": 3.0,
    "clear_px": 0.5,
    # Heading from faces.
    "heading_navigation": True,      # auto_fruit_search.py
    "heading_mapping": False,        # operate.py -- off until tested with the EKF changes
    "heading_sd_floor_deg": 2.0,     # per block: how square it really sits (+ anything unmodelled)
    "heading_sd_per_m_deg": 1.0,     # sd = floor (+) per_m * range / sin(view angle)
    "heading_min_view_deg": 10.0,    # nearer face-on than this: skip (mirror poses ~equal)
    "heading_max_view_deg": 75.0,    # more edge-on than this: skip (a sliver of a face)
    "heading_max_range": 1.5,        # m, farther: skip
    "heading_max_theta_sd_deg": 15.0,  # don't fuse while the heading itself is this unsure
    "heading_gate_sigma": 3.0,
    # Arena axes in the map frame.
    "axes_deg": None,                # null = learn; a number = fixed (0 for runs on truemap.txt)
    "axes_ready_sd_deg": 2.0,        # heading from faces starts once the axes are known this well...
    "axes_min_frames": 4,            # ...from at least this many spots
    "axes_max_frames": 20,           # spots to learn from (the estimate keeps improving until then)
    "axes_learn_max_theta_sd_deg": 5.0,  # a frame teaches the axes only if the heading is this sure
    "axes_spot_m": 0.10,             # frames closer together than this are one spot (one sample)
    # Step 3: ONE marker seen at an angle fixes the whole pose -- heading from its
    # face, position from its range and bearing -- with an honest uncertainty; if
    # that is within the "converged" limits, a re-localisation pan can stop there.
    # Opt-in (auto_fruit_search.py --one-marker-fix): in simulation it neither saved
    # time nor hurt (most pans found a second marker anyway); worth trying on the
    # robot, whose pans see one marker far more often.
    "one_marker_fix": False,
    "one_marker_max_range": 1.0,     # m
    "one_marker_view_deg": [15.0, 65.0],   # face seen this far off face-on (angle well measured)
    "one_marker_meas_sd": [0.015, 0.015],  # marker position noise: base m + per metre of range
    # Step 4: remember which printed face (printed rotation k) of each block points
    # along which arena axis, and use it to reset a heading that has gone badly wrong.
    "face_ids": True,
    "face_id_min_votes": 3,          # sightings before a face's direction counts as known
    "face_id_agree": 0.8,            # ...and this share of them must agree
    "face_id_recover_deg": 45.0,     # a remembered face disagreeing with the heading by more than this resets it
    "face_id_max_range": 1.2,        # m, only near faces are trusted for a reset
}

CONFIG_NAME = 'aruco_faces.json'

REFINE = {
    'none': cv2.aruco.CORNER_REFINE_NONE,
    'subpix': cv2.aruco.CORNER_REFINE_SUBPIX,
    'contour': cv2.aruco.CORNER_REFINE_CONTOUR,
    'apriltag': cv2.aruco.CORNER_REFINE_APRILTAG,
}

UP_CAM = np.array([0.0, -1.0, 0.0])      # camera frame: x right, y down, z forward


def load_config(calib_dir='calibration/param/', path=None, quiet=False):
    """DEFAULTS overlaid with calibration/param/aruco_faces.json (if it exists)."""
    cfg = dict(DEFAULTS)
    path = path or os.path.join(calib_dir, CONFIG_NAME)
    if os.path.exists(path):
        try:
            with open(path) as f:
                user = json.load(f)
            unknown = [k for k in user if k not in DEFAULTS and not k.startswith('_')]
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
            if unknown and not quiet:
                print("aruco_faces: ignoring unknown setting(s) in {}: {}".format(path, ', '.join(unknown)))
        except Exception as e:
            if not quiet:
                print("aruco_faces: could not read {} ({}) -- using the defaults".format(path, e))
    cfg['_path'] = path
    return cfg


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def wrap45(a):
    return (a + 45.0) % 90.0 - 45.0


def circ_mean(values, mod=360.0):
    """
    Inverse-variance weighted circular mean of [(angle_deg, var_deg2)], for
    angles that repeat every `mod` degrees (360 for directions, 90 for the
    arena axes). Returns (mean_deg, var_deg2) -- var is the combined variance.
    """
    k = 360.0 / mod
    w = np.array([1.0 / max(v, 1e-6) for _, v in values])
    a = np.radians([k * ang for ang, _ in values])
    m = math.degrees(math.atan2(float(w @ np.sin(a)), float(w @ np.cos(a)))) / k
    return m, 1.0 / float(w.sum())


# ----------------------------------------------------------------------
# One detected face
# ----------------------------------------------------------------------

class Solution:
    """One of the two pose solutions for a face (angles in degrees)."""
    __slots__ = ('rvec', 'tvec', 'err', 'phi', 'rot')

    def __init__(self, rvec, tvec, err):
        self.rvec, self.tvec, self.err = rvec, tvec, float(err)
        R, _ = cv2.Rodrigues(rvec)
        n = R[:, 2]                                         # face normal, camera frame
        # Direction the face points, robot frame (0 ahead, 90 left, 180 back at the robot)
        self.phi = math.degrees(math.atan2(-n[0], n[2]))
        # In-plane rotation of the printed pattern, CCW as seen facing it
        self.rot = math.degrees(math.atan2(float(R[:, 0] @ UP_CAM), float(R[:, 1] @ UP_CAM)))


class Face:
    """
    One detected face. Built by analyse_corners(); MarkerCorrector.correct()
    fills in the resolved fields (pick, why, phi, axis, view, centre, ...).
    """

    def __init__(self, tag, corners, sols):
        self.tag = int(tag)
        self.corners = corners
        self.sols = sols
        best = min(range(len(sols)), key=lambda i: sols[i].err)
        self.best = best
        t = np.asarray(sols[best].tvec, dtype=float).reshape(3)
        self.raw = np.array([t[2], -t[0]])                  # face centre: forward, left (m), camera frame
        self.u = float(np.asarray(corners).reshape(-1, 2)[:, 0].mean())
        self.k = int(round(sols[best].rot / 90.0)) % 4       # printed rotation, 0..3
        self.dist = float(math.hypot(self.raw[0], self.raw[1]))
        self.bearing = math.degrees(math.atan2(self.raw[1], self.raw[0]))
        # resolved later
        self.pick = best
        self.why = 'lower reprojection error'
        self.coin = False
        self.phi = sols[best].phi
        self.axis = None          # arena axis the face points along (map frame, deg), None if unsnapped
        self.view = wrap180(self.phi - (self.bearing + 180.0))
        self.centre = None        # block centre (forward, left), m

    def use(self, i, why):
        self.pick, self.why = i, why
        self.phi = self.sols[i].phi
        self.view = wrap180(self.phi - (self.bearing + 180.0))


def analyse_corners(tag, corners, camera_matrix, dist_coeffs, object_points):
    """Both IPPE_SQUARE solutions for one detected marker -> Face (or None)."""
    img_pts = corners[0] if np.asarray(corners).ndim == 3 else corners
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(object_points, img_pts, camera_matrix, dist_coeffs,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    if not n:
        return None
    errs = np.asarray(errs, dtype=float).reshape(-1) if errs is not None and len(errs) else np.zeros(n)
    return Face(tag, corners, [Solution(rvecs[i], tvecs[i], errs[i]) for i in range(n)])


# ----------------------------------------------------------------------
# Arena axes
# ----------------------------------------------------------------------

class ArenaAxes:
    """
    Rotation alpha0 (deg, -45..45) of the arena axes in the map frame.

    One sample per spot the robot stops at: the frame's faces averaged, with
    the frame's own heading variance added on top (it is common to every face
    in that frame, so averaging faces doesn't remove it). A better frame at the
    same spot (surer heading) replaces that spot's sample. The samples are
    combined as a weighted circular mean in the 4x angle domain, so -44 and +44
    deg average to +/-45, not 0.
    """

    def __init__(self, fixed_deg=None, ready_sd=2.0, min_frames=4, max_frames=20, spot_m=0.10):
        self.fixed = None if fixed_deg is None else wrap45(float(fixed_deg))
        self.ready_sd, self.min_frames, self.max_frames = ready_sd, min_frames, max_frames
        self.spot_m = spot_m
        self.samples = []            # [alpha_deg, var_deg2, x, y], one per spot

    @property
    def frames(self):
        return len(self.samples)

    def _sums(self):
        c = s = w = 0.0
        for a, v, _, _ in self.samples:
            wi = 1.0 / max(v, 1e-6)
            c += wi * math.cos(math.radians(4.0 * a))
            s += wi * math.sin(math.radians(4.0 * a))
            w += wi
        return c, s, w

    @property
    def deg(self):
        if self.fixed is not None:
            return self.fixed
        c, s, w = self._sums()
        return math.degrees(math.atan2(s, c)) / 4.0 if w > 0 else 0.0

    @property
    def sd(self):
        if self.fixed is not None:
            return 0.0
        w = self._sums()[2]
        return math.sqrt(1.0 / w) if w > 0 else float('inf')

    @property
    def ready(self):
        return self.fixed is not None or (self.frames >= self.min_frames and self.sd <= self.ready_sd)

    @property
    def learning(self):
        return self.fixed is None and self.frames < self.max_frames

    def offer(self, alpha, var, x, y, same_spot_ok=False):
        """
        One frame's sample taken at (x, y). Returns 'new' (a new spot), 'better'
        (replaced this spot's sample with a surer one) or None (not used).
        same_spot_ok: the heading is known exactly (start of a mapping run), so
        frames there are independent -- but only until the axes could be ready.
        """
        if not self.learning:
            return None
        if self.samples:
            last = self.samples[-1]
            if math.hypot(x - last[2], y - last[3]) < self.spot_m and \
                    not (same_spot_ok and self.frames < self.min_frames):
                if var < last[1]:
                    self.samples[-1] = [alpha, var, x, y]
                    return 'better'
                return None
        self.samples.append([alpha, var, x, y])
        return 'new'

    def snap(self, direction_deg):
        """Nearest arena axis to a map-frame direction: (axis_deg, distance_deg)."""
        a0 = self.deg
        m = round((direction_deg - a0) / 90.0)
        axis = (a0 + 90.0 * m) % 360.0
        return axis, abs(wrap180(direction_deg - axis))


# ----------------------------------------------------------------------
# Polynomial (same maths as operate.apply_distortion_correction)
# ----------------------------------------------------------------------

class Polynomial:
    def __init__(self, path):
        with open(path) as f:
            dc = json.load(f)
        self.path = path
        self.degree = int(dc['degree'])
        self.cx = np.array(dc['coeffs_x'], dtype=float)
        self.cy = np.array(dc['coeffs_y'], dtype=float)

    def __call__(self, x, y):
        feats = [1.0, x, y]
        if self.degree >= 2:
            feats += [x * x, x * y, y * y]
        if self.degree >= 3:
            feats += [x ** 3, x * x * y, x * y * y, y ** 3]
        feats = np.array(feats)
        return float(feats @ self.cx), float(feats @ self.cy)


# ----------------------------------------------------------------------
# The corrector
# ----------------------------------------------------------------------

class FoldResult:
    def __init__(self):
        self.learned = 0          # faces used to learn the arena axes this frame
        self.fused = 0            # faces fused as heading measurements
        self.gated = 0            # faces rejected by the innovation gate
        self.moved_deg = 0.0      # total heading change from the fused faces
        self.axes_ready_now = False
        self.recovered = False    # a remembered face reset a badly wrong heading (step 4)


class MarkerCorrector:
    """
    correct(markers, theta, theta_sd) -> markers with block-centre positions.
    fold(ekf, markers, n_known, fuse) -> learn the arena axes / fuse headings.

    Markers without face information (fruit landmarks, simulator fakes) pass
    through untouched.
    """

    def __init__(self, cfg, program='navigation', verbose=True):
        self.cfg = cfg
        self.enabled = bool(cfg.get('faces', True))
        self.block_centres = self.enabled and bool(cfg['block_centres'])
        self.half = 0.5 * float(cfg['block_size'])
        self.camera_offset = float(cfg['camera_offset'] or 0.0)
        self.poly = None
        if self.enabled and cfg.get('polynomial'):
            if os.path.exists(cfg['polynomial']):
                try:
                    self.poly = Polynomial(cfg['polynomial'])
                except Exception as e:
                    print("aruco_faces: could not load {} ({}) -- no polynomial".format(cfg['polynomial'], e))
        self.poly_has_face = bool(cfg['polynomial_includes_face_offset'])
        key = 'heading_mapping' if program == 'mapping' else 'heading_navigation'
        self.heading = self.enabled and bool(cfg[key])
        self.axes = ArenaAxes(cfg['axes_deg'], cfg['axes_ready_sd_deg'], int(cfg['axes_min_frames']),
                              int(cfg['axes_max_frames']), float(cfg['axes_spot_m']))
        self.program = program
        self.stats = {'faces': 0, 'fused': 0, 'gated': 0, 'pair_fixes': 0, 'coin': 0,
                      # arena-axes learning: frames used, and why the others weren't
                      'learn_used': 0, 'learn_unsure': 0, 'learn_one_marker': 0, 'learn_same_spot': 0,
                      'no_usable_face': 0, 'recoveries': 0}
        # Step 4: tag -> {printed rotation k: [votes for arena axis 0, 1, 2, 3]}
        self.face_votes = {}
        self._recover_pending = None    # (implied heading deg, tag) from the previous frame
        if verbose:
            print(self.describe())

    @property
    def applies_polynomial(self):
        return self.poly is not None

    def describe(self):
        if not self.enabled:
            return "ArUco faces: off (face-centre readings, as before)"
        parts = []
        if self.block_centres:
            if self.poly is not None:
                parts.append("block centres (half block {:.3f} m; polynomial {} first{})".format(
                    self.half, os.path.basename(self.poly.path),
                    ", fitted to block centres" if self.poly_has_face else ", fitted to faces"))
            else:
                parts.append("block centres (half block {:.3f} m, camera offset {:.3f} m, no polynomial)".format(
                    self.half, self.camera_offset))
        else:
            parts.append("face centres" + (" (polynomial {})".format(os.path.basename(self.poly.path))
                                           if self.poly is not None else ""))
        parts.append("heading from faces {}".format("ON" if self.heading else "off"))
        a = self.axes
        parts.append("arena axes {}".format("fixed at {:+.1f} deg".format(a.fixed) if a.fixed is not None
                                             else "learnt"))
        return "ArUco faces ({}): {}".format(self.program, "; ".join(parts))

    # ---------------------------------------------------------------- picking
    def _pick(self, face, theta_deg):
        """Choose one of the two mirror solutions. theta_deg None = heading unknown."""
        cfg, sols = self.cfg, face.sols
        face.coin = False
        if len(sols) == 1:
            face.use(0, 'only one solution')
            return
        a, b = sols[0], sols[1]
        lo, hi = sorted((a.err, b.err))
        best_err = 0 if a.err <= b.err else 1
        if hi > cfg['clear_ratio'] * lo and hi - lo > cfg['clear_px']:
            face.use(best_err, 'clearly lower reprojection error')
            return
        if theta_deg is None:
            face.use(best_err, 'lower reprojection error')
            face.coin = True                 # no way to check it
            return
        ax_a, d_a = self.axes.snap(theta_deg + a.phi)
        ax_b, d_b = self.axes.snap(theta_deg + b.phi)
        if abs(d_a - d_b) > cfg['tie_deg']:
            face.use(0 if d_a < d_b else 1, 'nearer an axis')
        elif ax_a == ax_b:
            face.use(best_err, 'same axis either way')
        else:
            face.use(best_err, 'coin toss')
            face.coin = True

    def _centre(self, face, normal_deg):
        """Block centre (forward, left) from one face whose normal points normal_deg (robot frame)."""
        x, y = float(face.raw[0]), float(face.raw[1])
        n = np.array([math.cos(math.radians(normal_deg)), math.sin(math.radians(normal_deg))])
        if self.poly is not None:
            x, y = self.poly(x, y)
            p = np.array([x, y])
            if not self.block_centres:
                return p
            if self.poly_has_face:
                # Face-on, the polynomial already added half a block along the line of
                # sight; add only the difference between that and the true direction.
                los = face.raw / max(np.linalg.norm(face.raw), 1e-6)
                return p - self.half * n - self.half * los
            return p - self.half * n
        p = np.array([x + self.camera_offset, y])
        return p - self.half * n if self.block_centres else p

    def _resolve_pair(self, f1, f2, theta_deg):
        """Two faces of one block: their normals must be 90 deg apart. Try all four
        solution pairs and keep the perpendicular one whose block centres agree best."""
        best = None
        for i, s1 in enumerate(f1.sols):
            for j, s2 in enumerate(f2.sols):
                ax1, _ = self.axes.snap(theta_deg + s1.phi)
                ax2, _ = self.axes.snap(theta_deg + s2.phi)
                if abs(abs(wrap180(ax1 - ax2)) - 90.0) > 1e-3:
                    continue
                c1 = self._centre(f1, ax1 - theta_deg)
                c2 = self._centre(f2, ax2 - theta_deg)
                score = (float(np.linalg.norm(c1 - c2)), s1.err + s2.err)
                if best is None or score < best[0]:
                    best = (score, i, j)
        if best is None:
            return False
        _, i, j = best
        changed = (i != f1.pick) or (j != f2.pick)
        f1.use(i, 'two faces agree'); f2.use(j, 'two faces agree')
        f1.coin = f2.coin = False
        if changed:
            self.stats['pair_fixes'] += 1
        return True

    # ---------------------------------------------------------------- correct
    def correct(self, markers, theta=None, theta_sd=None):
        """
        markers: Marker list from ArucoSensor.detect_marker_positions().
        theta: current heading estimate (rad) in the map frame, or None if unknown
        (navigation before the first fix). theta_sd: its sd (rad); above
        heading_max_theta_sd_deg the axes aren't used to pick or snap.
        Returns a NEW list; markers with faces get block-centre positions.
        """
        if not self.enabled or not markers:
            return markers
        th = None
        if theta is not None and (theta_sd is None or
                                  math.degrees(theta_sd) <= self.cfg['heading_max_theta_sd_deg']):
            th = math.degrees(float(theta))
        out = []
        for m in markers:
            faces = getattr(m, 'faces', None)
            if not faces:
                out.append(m)
                continue
            for f in faces:
                self._pick(f, th)
            if len(faces) == 2 and th is not None:
                ax = [self.axes.snap(th + f.phi)[0] for f in faces]
                if any(f.coin for f in faces) or abs(abs(wrap180(ax[0] - ax[1])) - 90.0) > 1e-3:
                    if not self._resolve_pair(faces[0], faces[1], th):
                        for f in faces:          # can't both be right: use neither for the heading
                            f.coin = True
            centres = []
            for f in faces:
                if th is not None:
                    f.axis = self.axes.snap(th + f.phi)[0]
                    normal = f.axis - th            # snapped normal, robot frame
                else:
                    # Heading unknown: the measured normal -- or, when the mirror pose
                    # wasn't ruled out, the line of sight, which lies exactly half-way
                    # between the two mirror solutions (right for any face-on view).
                    f.axis = None
                    normal = f.phi if not f.coin else f.bearing + 180.0
                f.centre = self._centre(f, normal)
                centres.append(f.centre)
                self.stats['faces'] += 1
                self.stats['coin'] += int(f.coin)
            pos = np.mean(centres, axis=0).reshape(2, 1)
            new = type(m)(pos, m.tag)
            for attr in ('noise_scale',):
                if hasattr(m, attr):
                    setattr(new, attr, getattr(m, attr))
            new.faces = faces
            new.legacy_position = m.position
            out.append(new)
        return out

    # ---------------------------------------------------------------- heading
    # Picks whose mirror solution was ruled out by something other than a guess.
    # 'same axis either way' (nearly face-on: both solutions point along the same
    # axis, a few degrees either side of the line of sight) is fine for the block
    # centre but not for the heading -- the wrong one is off by twice the view angle.
    DECISIVE = ('only one solution', 'clearly lower reprojection error', 'nearer an axis', 'two faces agree')

    def face_heading_sd(self, f):
        """
        Measurement noise (deg) of the face direction read from face f, or None
        if the face isn't usable for the heading. Grows with range and as the
        view gets nearer face-on. The floor for how square the block itself sits
        (heading_sd_floor_deg) is added once per BLOCK in _per_block(), not per
        face: two faces of one block, or ten frames of it, share that error.
        """
        cfg = self.cfg
        if f.coin or f.why not in self.DECISIVE:
            return None
        if not (cfg['heading_min_view_deg'] <= abs(f.view) <= cfg['heading_max_view_deg']) \
                or f.dist > cfg['heading_max_range']:
            return None
        s = max(math.sin(math.radians(abs(f.view))), 1e-3)
        return max(0.5, cfg['heading_sd_per_m_deg'] * f.dist / s)

    def _per_block(self, items, value, mod=360.0):
        """
        [(face, sd)] -> [(value_deg, var_deg2)], one per block (tag): the
        block's faces in this frame averaged (circular mean modulo `mod`,
        weighted by 1/sd^2), plus the block-placement floor.
        """
        groups = {}
        for f, sd in items:
            groups.setdefault(f.tag, []).append((value(f), sd ** 2))
        floor2 = self.cfg['heading_sd_floor_deg'] ** 2
        out = []
        for vals in groups.values():
            m, v = circ_mean(vals, mod)
            out.append((m, v + floor2))
        return out

    # ---------------------------------------------------------------- face identity (step 4)
    # Picks that don't depend on the heading being right (the axis rule does).
    HEADING_FREE = ('only one solution', 'clearly lower reprojection error', 'two faces agree')

    def _axis_index(self, axis_deg):
        return int(round((axis_deg - self.axes.deg) / 90.0)) % 4

    def known_face_axis(self, tag, k):
        """Arena axis (deg) the face (tag, printed rotation k) points along, or None if
        it isn't known yet. Known = face_id_min_votes sightings, face_id_agree of them on
        one axis, AND some other side of the same block has been seen printed at a
        different rotation pointing along a different axis -- proof the block's sides
        are told apart by their printing (on a block printed the same on every side,
        k names no face, and its votes split across axes)."""
        votes = self.face_votes.get(tag)
        if not votes or k not in votes:
            return None
        v = votes[k]
        total = sum(v)
        m = int(np.argmax(v))
        if total < self.cfg['face_id_min_votes'] or v[m] < self.cfg['face_id_agree'] * total:
            return None
        for kk, w in votes.items():
            if kk == k or not sum(w):
                continue
            mm = int(np.argmax(w))
            if mm != m and w[mm] >= self.cfg['face_id_agree'] * sum(w):
                return (self.axes.deg + 90.0 * m) % 360.0
        return None

    def _vote_faces(self, usable, th):
        for f, _ in usable:
            m = self._axis_index(self.axes.snap(th + f.phi)[0])
            self.face_votes.setdefault(f.tag, {}).setdefault(f.k, [0, 0, 0, 0])[m] += 1

    def _check_recovery(self, ekf, faces, th, res):
        """
        A remembered face, near and with a mirror pick that didn't rely on the heading,
        implies a heading more than face_id_recover_deg away from the current one:
        the heading has gone badly wrong. Two agreeing readings (this frame, or this
        and the previous frame) reset it; position is then flagged for re-checking.
        """
        cands = []
        for f in faces:
            if f.why not in self.HEADING_FREE or f.dist > self.cfg['face_id_max_range']:
                continue
            axis = self.known_face_axis(f.tag, f.k)
            if axis is None:
                continue
            implied = axis - f.phi
            if abs(wrap180(implied - th)) > self.cfg['face_id_recover_deg']:
                cands.append((implied, f.tag))
        if not cands:
            self._recover_pending = None
            return False
        pending = self._recover_pending
        if len(cands) >= 2 and abs(wrap180(cands[0][0] - cands[1][0])) <= 10.0:
            agreed = cands[:2]
        elif pending is not None and abs(wrap180(cands[0][0] - pending[0])) <= 10.0:
            agreed = [cands[0], pending]
        else:
            self._recover_pending = cands[0]
            return False
        new = circ_mean([(a, 1.0) for a, _ in agreed])[0]
        x = ekf.get_state_vector()
        x[2, 0] = math.radians(wrap180(new))
        ekf.set_state_vector(x)
        P = ekf.P
        P[2, :] = 0.0
        P[:, 2] = 0.0
        P[2, 2] = math.radians(5.0) ** 2
        P[0, 0] = max(P[0, 0], 0.20 ** 2)
        P[1, 1] = max(P[1, 1], 0.20 ** 2)
        self._recover_pending = None
        self.stats['recoveries'] += 1
        res.recovered = True
        print("ArUco faces: heading was {:+.0f} deg but marker {}'s remembered face says {:+.0f} -- "
              "heading reset (position to be re-checked)".format(wrap180(th), agreed[0][1], wrap180(new)))
        return True

    # ---------------------------------------------------------------- one-marker fix (step 3)
    def one_marker_pose(self, ekf, marker, map_xy):
        """
        Full pose from ONE known marker: heading from its face(s), position from
        the marker's map position minus its measured offset rotated by that heading.
        Returns (state [x, y, theta], 3x3 covariance, note) or None when the faces
        aren't good enough (range, view angle, mirror pick) or a remembered face
        (step 4) disagrees with the axis the face snaps to.
        """
        cfg, a = self.cfg, self.axes
        if not a.ready:
            return None
        lo, hi = cfg['one_marker_view_deg']
        th = math.degrees(float(ekf.robot.state[2, 0]))
        good = []
        for f in getattr(marker, 'faces', None) or []:
            sd = self.face_heading_sd(f)
            if sd is None or f.dist > cfg['one_marker_max_range'] or not (lo <= abs(f.view) <= hi):
                continue
            axis = a.snap(th + f.phi)[0]
            known = self.known_face_axis(f.tag, f.k) if cfg['face_ids'] else None
            if known is not None and abs(wrap180(known - axis)) > 1.0:
                return None                      # remembered face says another axis: don't trust it
            good.append((f, sd))
        if not good:
            return None
        meas, var = circ_mean(self._per_block(good, lambda f: a.snap(th + f.phi)[0] - f.phi))
        var += a.sd ** 2
        theta = math.radians(meas)
        var_t = math.radians(math.sqrt(var)) ** 2
        z = np.asarray(marker.position, dtype=float).reshape(2)
        c, s_ = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        dR = np.array([[-s_, -c], [c, -s_]])
        pos = np.asarray(map_xy, dtype=float) - R @ z
        base, per_m = cfg['one_marker_meas_sd']
        sd_z = base + per_m * float(np.linalg.norm(z))
        J = -(dR @ z)                          # d(position)/d(theta)
        cov = np.zeros((3, 3))
        cov[0:2, 0:2] = sd_z ** 2 * np.eye(2) + np.outer(J, J) * var_t
        cov[0:2, 2] = cov[2, 0:2] = J * var_t
        cov[2, 2] = var_t
        return (np.array([pos[0], pos[1], math.atan2(s_, c)]), cov,
                "one-marker fix from marker {} ({} face(s))".format(int(marker.tag), len(good)))

    def fold(self, ekf, markers, n_known, fuse=None):
        """
        After the frame's marker update: learn the arena axes from this frame if
        its heading is known independently, otherwise fuse its faces as heading
        measurements (if heading fusion is on and the axes are known).

        n_known: known ArUco markers in this frame (2+ means the update just fixed
        the heading from marker geometry). fuse: override the config switch.
        """
        res = FoldResult()
        if not self.enabled or not markers:
            return res
        fuse = self.heading if fuse is None else fuse
        faces = [f for m in markers for f in (getattr(m, 'faces', None) or [])]
        if not faces:
            return res
        theta = float(ekf.robot.state[2, 0])
        th = math.degrees(theta)
        sd_t = math.degrees(math.sqrt(max(float(ekf.P[2, 2]), 0.0)))
        if self.cfg['face_ids'] and self.axes.ready and self._check_recovery(ekf, faces, th, res):
            return res
        usable = []
        for f in faces:
            sd = self.face_heading_sd(f)
            if sd is not None:
                usable.append((f, sd))
        if not usable:
            self.stats['no_usable_face'] += 1
            return res

        a = self.axes
        if a.learning:
            start_like = sd_t <= 0.6     # heading known exactly (start of a mapping run; the EKF floors it at 0.5 deg)
            if sd_t > self.cfg['axes_learn_max_theta_sd_deg']:
                self.stats['learn_unsure'] += 1
            elif n_known < 2 and not start_like:
                self.stats['learn_one_marker'] += 1      # one marker: its heading is the filter's, not measured
            else:
                x, y = float(ekf.robot.state[0, 0]), float(ekf.robot.state[1, 0])
                # one value per block, then per frame; the frame's heading error is
                # common to all its faces, so it's added after averaging them
                alpha, var = circ_mean(self._per_block(usable, lambda f: wrap45(th + f.phi), 90.0), 90.0)
                var += sd_t ** 2
                was_ready = a.ready
                got = a.offer(alpha, var, x, y, same_spot_ok=start_like)
                if got:
                    res.learned = len(usable)
                    self.stats['learn_used'] += 1
                    if a.ready and not was_ready:
                        res.axes_ready_now = True
                        print("ArUco faces: arena axes at {:+.1f} deg in the map frame (+/-{:.1f}, from {} spots){}".format(
                            a.deg, a.sd, a.frames, " -- heading from faces now ON" if fuse else ""))
                    elif got == 'new' and not a.ready:
                        print("ArUco faces: learning the arena axes -- spot {} at [{:.2f}, {:.2f}]: {:+.1f} deg so far "
                              "(+/-{:.1f}; heading from faces starts at +/-{:.1f} from {} spots)".format(
                                  a.frames, x, y, a.deg, a.sd, a.ready_sd, a.min_frames))
                    return res                   # a frame that taught the axes isn't also fused
                self.stats['learn_same_spot'] += 1

        if self.cfg['face_ids'] and a.ready and sd_t <= self.cfg['axes_learn_max_theta_sd_deg']:
            self._vote_faces(usable, th)       # remember which face points where (step 4)
        if not fuse or not a.ready or sd_t > self.cfg['heading_max_theta_sd_deg']:
            return res
        # ONE heading measurement per frame: each block's faces give the heading
        # (its arena axis minus the direction the face is seen pointing), blocks
        # are averaged, and the arena axes' own uncertainty is added on top --
        # it is common to every face, so more faces can't average it away.
        start = float(ekf.robot.state[2, 0])
        blocks = self._per_block(usable, lambda f: a.snap(th + f.phi)[0] - f.phi)
        meas, var = circ_mean(blocks)
        var += a.sd ** 2
        if fuse_heading(ekf, math.radians(meas), math.radians(math.sqrt(var)) ** 2, self.cfg['heading_gate_sigma']):
            res.fused = len(usable)
        else:
            res.gated = len(usable)
        res.moved_deg = math.degrees(math.atan2(math.sin(float(ekf.robot.state[2, 0]) - start),
                                                math.cos(float(ekf.robot.state[2, 0]) - start)))
        self.stats['fused'] += res.fused
        self.stats['gated'] += res.gated
        return res

    def summary(self):
        s = self.stats
        a = self.axes
        if a.fixed is not None:
            axes = "fixed at {:+.1f} deg".format(a.fixed)
        elif a.frames:
            axes = "{:+.1f} +/- {:.1f} deg from {} spot(s){}".format(
                a.deg, a.sd, a.frames, "" if a.ready else " -- NOT ready, so no heading from faces")
        else:
            axes = "not learnt (no spot taught it) -- so no heading from faces"
        known = sum(1 for t, v in self.face_votes.items() for k in v if self.known_face_axis(t, k) is not None)
        lines = ["ArUco faces: {} face readings ({} mirror coin-tosses, {} fixed by the other face); "
                 "heading from faces: {} fused, {} rejected; arena axes {}".format(
                     s['faces'], s['coin'], s['pair_fixes'], s['fused'], s['gated'], axes)]
        if self.cfg['face_ids']:
            lines.append("  remembered faces: {} (on {} block(s) printed differently on their sides); "
                         "heading resets from them: {}".format(
                             known, len({t for t, v in self.face_votes.items()
                                         if any(self.known_face_axis(t, k) is not None for k in v)}),
                             s['recoveries']))
        if a.fixed is None:
            lines.append("  arena-axes learning: {} frame(s) used; frames skipped: {} heading too unsure (> {:.0f} deg), "
                         "{} only one marker, {} same spot, {} no usable face".format(
                             s['learn_used'], s['learn_unsure'], self.cfg['axes_learn_max_theta_sd_deg'],
                             s['learn_one_marker'], s['learn_same_spot'], s['no_usable_face']))
        return "\n".join(lines)


def fuse_heading(ekf, theta_meas, var, gate_sigma=3.0):
    """
    Scalar Kalman update of the heading: z = theta, H = [0, 0, 1, 0, ...].

    Over the FULL state and covariance (Joseph form), so x, y and any landmark
    move with whatever the filter knows about their correlation with theta --
    shrinking P[2, 2] alone would leave P inconsistent. Returns False (and
    changes nothing) when the innovation is beyond gate_sigma sigmas.
    """
    P = ekf.P
    x = ekf.get_state_vector()
    innov = math.atan2(math.sin(theta_meas - x[2, 0]), math.cos(theta_meas - x[2, 0]))
    S = float(P[2, 2]) + var
    if S <= 0 or innov * innov > gate_sigma * gate_sigma * S:
        return False
    K = P[:, 2:3] / S                                    # n x 1
    x = x + K * innov
    x[2, 0] = math.atan2(math.sin(x[2, 0]), math.cos(x[2, 0]))
    n = P.shape[0]
    I_KH = np.eye(n)
    I_KH[:, 2:3] -= K
    P_new = I_KH @ P @ I_KH.T + var * (K @ K.T)
    ekf.set_state_vector(x)
    ekf.P = 0.5 * (P_new + P_new.T)
    return True
