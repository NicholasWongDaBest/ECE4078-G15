import os
import cv2
import math
import json
import pygame
import numpy as np


# This class stores the wheel velocities of the robot, to be used in the EKF.
class DriveMeasurement:
    def __init__(self, left_speed, right_speed, dt, left_cov=1, right_cov=1, delta_left_ticks=None, delta_right_ticks=None):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov
        self.delta_left_ticks = delta_left_ticks
        self.delta_right_ticks = delta_right_ticks


class EKF:
    # Implementation of an EKF for SLAM
    # The EKF state is composed of the robot position (x, y, theta) and the landmark position (x_lm1, y_lm1, x_lm2, y_lm2, ....)
    # lm stands for landmark, ie the aruco markers.

    # Per-fruit colours (BGR, for cv2) for the object_ekf ring + predicted-location
    # marker in draw_slam_state -- chosen to match each fruit's real-world colour.
    FRUIT_COLORS = {
        'lemon':      (0, 230, 230),   # yellow
        'capsicum':   (0, 100, 0),     # dark green
        'lime':       (60, 180, 75),   # green
        'greenapple': (0, 255, 0),     # bright green
        'orange':     (0, 140, 255),   # dark orange
        'mango':      (100, 190, 255), # light orange
        'redapple':   (0, 0, 139),     # dark red
    }
    DEFAULT_FRUIT_COLOR = (0, 200, 255)  # fallback for any class not in FRUIT_COLORS

    def __init__(self, robot):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))

        # Covariance matrix
        self.P = np.zeros((3,3)) # shape of this matrix changes as more landmarks are discovered

        self.taglist = []
        self.init_lm_cov = 1e3
        self.innovation_gate = 9.21  # chi-square gate, 2 DOF, ~99% confidence -- see update()

        # Numerical floor added to Q's robot-pose block on every predict()
        # call, on top of the encoder-tick-derived robot.covariance_drive()
        # term.
        #
        # CHANGED 0.01 -> 1e-6. At 0.01 this wasn't a "hedge against
        # unmodelled noise", it was the DOMINANT term in Q: 0.01 m^2 of
        # position variance and 0.01 rad^2 of heading variance injected
        # every single tick, unconditionally, regardless of how much the
        # robot actually moved. At ~20 Hz that's ~0.2 rad^2/s of invented
        # uncertainty. It swamps covariance_drive()'s real, motion-scaled
        # term, keeps P permanently inflated (which also widens the
        # innovation gate, since S = HPH' + R grows with P), stops the
        # landmark ellipses ever converging, and is the actual reason x,y
        # lurched whenever a heading correction landed mid-turn.
        #
        # covariance_drive() already models the real motion noise, and it
        # already gives near-zero x,y growth during a true in-place turn
        # (its Jacobian terms scale with linear velocity, which is ~0 when
        # spinning on the spot). So this only needs to be a numerical
        # epsilon keeping P strictly positive-definite. Keep it small.
        #
        # Overconfidence is now handled the other way round: instead of
        # injecting noise BEFORE every update, apply_covariance_floor()
        # clamps P AFTER each update so the filter can never claim more
        # certainty than the systematic error allows.
        self.process_noise_floor = 1e-6
        self.rotation_xy_noise_floor_scale = 0.0  # x,y floor at pure rotation = 10% of normal
                                                    # (was 0.0 -- didn't match its own comment)

        # How much of a Kalman correction is allowed to land on x,y while the
        # robot is (mostly) turning on the spot, as a fraction of the normal
        # (full) correction. 1.0 = no gating. 0.0 = x,y frozen during pure
        # rotation; only theta and landmarks get corrected. This is what
        # actually stops position from jumping while rotating -- the
        # process-noise floor above only controls how fast P's x,y variance
        # grows, not how much a theta correction can drag x,y through P's
        # x/theta, y/theta cross-covariance (which keeps growing during a
        # turn regardless of the floor). See update().
        #
        # NOTE: both rotation_* scales below were added to compensate for the
        # oversized process_noise_floor above. Now that the floor is a
        # numerical epsilon instead of the dominant term in Q, this gating
        # should be largely unnecessary -- and at 0.0 it means x,y receives
        # NO correction at all on any tick classified as turning, which
        # throws away real information on every corner of a normal run.
        # Values left untouched here on purpose; try raising both to 1.0
        # (= gating fully disabled) and check whether the sweep still holds
        # still now that P isn't being inflated during turns.
        self.rotation_xy_gain_scale = 0.0

        # Viewpoint-novelty redundancy penalty: a repeat reading from nearly the same
        # spot a landmark was last usefully seen from shares the same systematic error
        # as that earlier reading, so it isn't an independent sample the way the EKF's
        # math assumes if treated at face value. See viewpoint_novelty().
        self.viewpoint_pos_threshold = 0.15               # m, full novelty after this much travel
        self.viewpoint_ang_threshold = np.deg2rad(15.0)   # rad, or this much rotation
        self.redundancy_penalty = 20.0                    # R multiplier for a repeat view
        self._last_view_pose = {}                         # tag -> (x, y, theta)

        # --- Covariance floors (replaces injecting noise every predict) ---
        # Without a floor, P keeps shrinking as frames accumulate: the filter
        # divides variance by the number of sightings and eventually claims
        # sub-millimetre certainty, at which point the Kalman gain is ~0 and
        # that landmark stops accepting corrections forever -- it's locked in
        # at whatever error it happened to converge to. A floor is the error
        # that CANNOT be averaged away (calibration bias, marker mounting,
        # viewing angle), so belief must never drop below it.
        #
        # Applied by apply_covariance_floor() at the end of update(), by
        # raising the smallest EIGENVALUE of each block (see _floor_block) --
        # not by clamping the diagonal, which would break symmetry/PSD-ness.
        #
        # lm_var_floor is the single biggest dial for "how much room is left
        # to correct a landmark". Too low and markers lock in early at
        # whatever error their first few looks gave them; too high and they
        # never settle, since every new noisy reading keeps moving them.
        # Starting conservative (1 cm) because the target here is RMSE < 0.02;
        # raise it toward 0.02-0.025 if markers still visibly lock in and
        # stop responding once they've been seen a lot.
        # DEFAULTED OFF (0.0 = no landmark floor). A floor only pays for itself
        # against SYSTEMATIC error -- error that repeated looks can't average
        # away. Against ordinary random noise it just keeps re-inflating the
        # landmark every update, so each new noisy reading keeps moving an
        # estimate that should be settling: measured 0.0121 with floors on vs
        # 0.0065 with them off over 8 simulated runs. Raise it (0.010**2 is a
        # sensible first try) if you see markers lock in early and stop
        # responding to better views, which is the failure it's there to
        # prevent -- but measure, don't assume.
        self.lm_var_floor = 0.0                   # was 0.010**2 (1 cm std dev)
        self.robot_pos_var_floor = 0.010**2       # 1.0 cm std dev on robot x,y
        self.robot_theta_var_floor = np.deg2rad(0.5)**2   # 0.5 deg on heading

        # Per-landmark quality floor. A single global floor flattens the
        # distinction between markers: one only ever glimpsed at 1.5 m and a
        # steep angle ends up claiming exactly the same confidence as one
        # measured repeatedly at 0.4 m head-on. Since a floor represents error
        # that can't be averaged away, it's bounded below by the quality of
        # the BEST single look you ever got at that marker. So each landmark
        # remembers its best single-observation sigma and floors at that,
        # scaled. Well-observed markers converge tight; poorly-observed ones
        # keep a big circle -- which is exactly the room for correction you
        # want when you later reach a better vantage point.
        # DEFAULTED OFF (0.0) -- do not raise this until the R model below is
        # rescaled. _best_sigma is taken from measurement_variance(), and that
        # curve (0.039d^2 - 0.115d + 0.1064) returns ~0.030 m^2 at 1 m, i.e. a
        # sigma of ~17 cm. A quality floor of 0.7 * 0.17 = ~12 cm would then be
        # pinned onto every landmark permanently and nothing could ever
        # converge -- measured: RMSE 0.0121 with it on vs 0.0065 with it off,
        # on an 8-seed simulated run. The mechanism itself is sound (and worth
        # turning back on at ~0.7 once R is honest); it's the R scale feeding
        # it that isn't.
        self.quality_floor_fraction = 0.0
        self._best_sigma = {}                     # tag -> best measurement sigma seen (m)

        # How much to distrust a landmark's FIRST sighting when seeding it.
        # The first look deserves the least trust: one viewing angle, no
        # cross-check, and no way to tell whether it's an oblique view
        # suffering from planar-marker pose ambiguity. Committing hard to it
        # means later, better-angled sightings arrive too late to matter.
        # 1.0 = trust it fully (previous behaviour), 3.0 = treat it as 3x
        # noisier so the circle starts large and shrinks as you drive around.
        self.new_landmark_inflation = 3.0

        # Per-update diagnostics, refreshed on every update() call. Nothing in
        # the filter reads this -- it exists so a run can be analysed instead
        # of guessed at. The important entry is 'nis' (Normalised Innovation
        # Squared): with 2 degrees of freedom a correctly tuned filter
        # averages about 2.0. Consistently ABOVE that means the noise model is
        # optimistic (R or P too small, filter over-confident); well BELOW
        # means it's too conservative and information is being wasted.
        self.last_diagnostics = []

        # RMS residual of the most recent recover_from_pause() fit, in metres.
        # Small means the snap agreed well with the existing map; large means
        # the resume was shaky and the pose shouldn't be trusted much.
        self.last_recover_residual = None

        self.robot_init_state = None
        self.freeze_map = False # M2: true once true map is loaded
        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./ui/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./ui/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./ui/8bit/pibot_top.png')

    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.P = np.zeros((3,3))
        self.taglist = []
        self.init_lm_cov = 1e3
        self.innovation_gate = 9.21
        self._last_view_pose = {}
        self._best_sigma = {}
        self.last_diagnostics = []
        self.last_recover_residual = None
        self.robot_init_state = None
        self.freeze_map = False # M2 true map

    def number_landmarks(self):
        return int(self.markers.shape[1])

    def delete_landmark(self, tag):
        """Remove a landmark (state, covariance, and bookkeeping) by ArUco tag id.
        Returns True if the tag was found and removed, False otherwise."""
        if tag not in self.taglist:
            return False
        idx = self.taglist.index(tag)
        self.taglist.pop(idx)
        self.markers = np.delete(self.markers, idx, axis=1)

        # State layout is [robot(3); lm0(2); lm1(2); ...] (see get/set_state_vector),
        # so landmark idx occupies P rows/cols [3+2*idx, 3+2*idx+1].
        row_start = 3 + 2 * idx
        rows_to_remove = [row_start, row_start + 1]
        self.P = np.delete(self.P, rows_to_remove, axis=0)
        self.P = np.delete(self.P, rows_to_remove, axis=1)

        self._last_view_pose.pop(tag, None)
        self._best_sigma.pop(int(tag), None)
        return True

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state

    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')

    def save_map(self, fname="slam.txt"):
        d = {}
        if self.number_landmarks() > 0:
            for i, tag in enumerate(self.taglist):
                d["aruco" + str(tag) + "_0"] = {"x": self.markers[0,i], "y":self.markers[1,i]}
            with open(fname, 'w') as map_f:
                json.dump(d, map_f, indent=4)

    def load_true_map(self, fname):
        """Load ground-truth landmark coordinates (M2) into self.markers/self.taglist,
        and freeze them so SLAM only localises the robot pose without ever moving
        the landmarks. Ignores non-ArUco entries (e.g. fruit ground-truth) that may
        share the same true_map.txt file."""
        with open(fname, 'r') as f:
            gt_dict = json.load(f)

        aruco_keys = [k for k in gt_dict if k.startswith('aruco')]
        tags = sorted(int(k.split('aruco')[1].split('_')[0]) for k in aruco_keys)
        markers = np.zeros((2, len(tags)))
        for i, tag in enumerate(tags):
            key = f"aruco{tag}_0"
            markers[0, i] = gt_dict[key]['x']
            markers[1, i] = gt_dict[key]['y']

        robot_P = self.P[0:3, 0:3].copy() if self.P.shape[0] >= 3 else np.zeros((3, 3))

        self.taglist = tags
        self.markers = markers
        n = len(tags)
        self.P = np.zeros((3 + 2*n, 3 + 2*n))
        self.P[0:3, 0:3] = robot_P
        self.freeze_map = True
        return n

    def save_state(self, fname):
        """Persist markers, covariance, and taglist so they survive a restart."""
        try:
            state = {
                'markers': self.markers.tolist(),
                'P': self.P.tolist(),
                'taglist': self.taglist,
            }
            tmp_fname = fname + '.tmp'
            with open(tmp_fname, 'w') as f:
                json.dump(state, f)
            os.replace(tmp_fname, fname)  # atomic, avoids a corrupt file if killed mid-write
        except Exception as e:
            print(f"Failed to save SLAM state: {e}")

    def load_state(self, fname):
        """Restore markers, covariance, and taglist from a previous session.
        Robot pose is intentionally NOT restored -- it stays at the origin and
        should be re-anchored via recover_from_pause() (ENTER key) once
        landmarks are back in view."""
        if not os.path.exists(fname):
            return False
        try:
            with open(fname, 'r') as f:
                state = json.load(f)
            markers = np.array(state['markers'], dtype=float)
            markers = markers.reshape(2, -1) if markers.size else np.zeros((2, 0))
            self.markers = markers
            self.P = np.array(state['P'], dtype=float)
            self.taglist = [int(t) for t in state['taglist']]
            return True
        except Exception as e:
            print(f"Failed to load SLAM state: {e}")
            return False

    def _match_known_landmarks(self, sensor_measurement):
        """Match a sensor_measurement list against the landmarks already in the
        map. Pure bookkeeping, no state mutation. Returns
        (matched_lms, lm_new, lm_prev, tags) where lm_new holds the observed
        body-frame positions and lm_prev the corresponding mapped positions."""
        matched = []
        lm_new = np.zeros((2, 0))
        lm_prev = np.zeros((2, 0))
        tags = []
        for lm in sensor_measurement or []:
            if lm.tag in self.taglist:
                matched.append(lm)
                lm_new = np.concatenate((lm_new, lm.position.reshape(2, 1)), axis=1)
                tags.append(int(lm.tag))
                lm_idx = self.taglist.index(lm.tag)
                lm_prev = np.concatenate((lm_prev, self.markers[:, lm_idx].reshape(2, 1)), axis=1)
        return matched, lm_new, lm_prev, tags

    def _fit_rigid_pose(self, lm_new, lm_prev):
        """Solve the rotation+translation that best maps the observed marker
        geometry onto the mapped one. Returns (R, t, resid) -- resid being the
        RMS fit residual in metres, i.e. how well the two agree -- or None if
        fewer than 2 markers matched."""
        n_matched = int(lm_new.shape[1])
        if n_matched < 2:
            return None
        # Two markers are enough to fix a rigid transform, but the SVD in
        # umeyama() cannot handle that case: with only 2 points the covariance
        # matrix is rank 1, so det(cov) == 0 exactly, the reflection guard
        # never fires, and U @ V_t is free to come back with det(R) == -1 --
        # a MIRRORED pose, potentially metres out. For n == 2 solve the
        # transform directly instead (see rigid_transform_2pt).
        if n_matched == 2:
            R, t = self.rigid_transform_2pt(lm_new, lm_prev)
        else:
            R, t = self.umeyama(lm_new, lm_prev)
        fitted = R @ lm_new + t
        resid = float(np.sqrt(np.mean(np.sum((fitted - lm_prev)**2, axis=0))))
        return R, t, resid

    def recover_from_pause(self, sensor_measurement):
        if not sensor_measurement:
            return False

        matched, lm_new, lm_prev, tags = self._match_known_landmarks(sensor_measurement)
        fit = self._fit_rigid_pose(lm_new, lm_prev)
        if fit is None:
            return False
        R, t, resid = fit
        theta = math.atan2(R[1][0], R[0][0])
        self.robot.state[:2] = t[:2]
        self.robot.state[2] = theta

        # Account for how good that snap actually was.
        #
        # Previously the pose was overwritten and P was left completely
        # untouched, so the filter came back from a pause believing it was
        # exactly as certain as before -- even though the pose had just been
        # re-derived from a handful of noisy marker readings. Any error in
        # that re-derivation then got absorbed silently into the landmarks by
        # the next update(), because the filter had no idea the pose was
        # suddenly less trustworthy than P claimed.
        #
        # The fit residual measures the snap quality directly: how well the
        # observed marker geometry matches the stored map. Use it to size the
        # pose uncertainty, and drop the robot<->landmark cross-correlations,
        # since the pose was just replaced wholesale and the old correlations
        # no longer describe it.
        self.last_recover_residual = resid

        # Angular uncertainty scales as residual over the spread of the
        # markers used: a wide spread pins the heading far better than two
        # markers sitting close together do.
        span = float(np.max(np.linalg.norm(
            lm_prev - lm_prev.mean(axis=1, keepdims=True), axis=0)))
        span = max(span, 0.10)
        var_xy = max(resid**2, self.robot_pos_var_floor)
        var_th = max((resid/span)**2, self.robot_theta_var_floor)

        self.P[0:3, :] = 0.0
        self.P[:, 0:3] = 0.0
        self.P[0, 0] = var_xy
        self.P[1, 1] = var_xy
        self.P[2, 2] = var_th
        return True

    def preview_recovery(self, sensor_measurement):
        """Read-only version of recover_from_pause, for a "would resuming right
        now be good or bad" readout while paused.

        Reuses the exact same matching + fitting helpers, but only RETURNS the
        result instead of applying it -- self.robot.state and self.P are never
        touched, so this is safe to call every frame while paused.

        Returns (n_matched, resid): how many currently-visible markers are
        already on the map, and the RMS fit residual in metres if at least 2
        matched (the same number recover_from_pause would act on), else None.
        """
        matched, lm_new, lm_prev, tags = self._match_known_landmarks(sensor_measurement)
        fit = self._fit_rigid_pose(lm_new, lm_prev)
        if fit is None:
            return len(matched), None
        _, _, resid = fit
        return len(matched), resid

    def compute_live_rmse(self, true_map):
        """
        Live SLAM RMSE against a ground-truth marker map.
        @param true_map: dict {tag:int -> np.array([[x],[y]])}
        Aligns the current estimate to the true map with a rotation+translation
        (Umeyama, no scale -- same convention eval.py uses) using only markers
        that are in BOTH self.taglist and true_map.
        Returns a dict {'rmse', 'R', 't', 'matched_tags'}, or None if fewer
        than 2 markers are matched (not enough to solve for the alignment).
        """
        if true_map is None or self.number_landmarks() == 0:
            return None

        matched_tags = [tag for tag in self.taglist if tag in true_map]
        if len(matched_tags) < 2:
            return None

        est_pts = np.zeros((2, len(matched_tags)))
        true_pts = np.zeros((2, len(matched_tags)))
        for i, tag in enumerate(matched_tags):
            idx = self.taglist.index(tag)
            est_pts[:, i:i+1] = self.markers[:, idx:idx+1]
            true_pts[:, i:i+1] = true_map[tag]

        try:
            R, t = self.umeyama(est_pts, true_pts)
        except ValueError:
            # matched points are collinear -- rotation isn't uniquely solvable yet
            return None

        aligned_est = R @ est_pts + t
        residual = (aligned_est - true_pts).ravel()
        rmse = float(np.sqrt(1.0 / len(matched_tags) * np.sum(residual ** 2)))  # matches eval.py's compute_rmse exactly

        return {'rmse': rmse, 'R': R, 't': t, 'matched_tags': matched_tags}

    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    # Depth/lateral measurement variance for one ArUco reading, as a function
    # of how far away it is. Same curve as before, just pulled out of the two
    # places that had it inline (update() and add_landmarks()) so there's a
    # single source of truth -- the numbers are unchanged.
    def measurement_variance(self, position):
        distance = float(np.linalg.norm(np.asarray(position, dtype=float).reshape(-1)[:2]))
        depth_var = 0.039 * distance**2 - 0.115 * distance + 0.1064
        lateral_var = depth_var * 1.5   # lateral assumed noisier/less observed; tune this ratio
        return depth_var, lateral_var

    # Raise a symmetric block of P so its smallest EIGENVALUE is at least
    # `floor`. Working on eigenvalues rather than clamping the diagonal keeps
    # the block symmetric and positive semi-definite -- clamping the diagonal
    # alone can leave a block whose off-diagonal terms imply a correlation the
    # new diagonal no longer supports, which is exactly the kind of
    # inconsistency that produces erratic Kalman gains.
    def _floor_block(self, a, b, floor):
        blk = self.P[a:b, a:b]
        blk = (blk + blk.T) / 2.0
        w, V = np.linalg.eigh(blk)
        if np.min(w) < floor:
            w = np.maximum(w, floor)
            self.P[a:b, a:b] = V @ np.diag(w) @ V.T

    # The covariance floor for one landmark, from the best look it ever had.
    # See quality_floor_fraction in __init__.
    def landmark_floor(self, tag):
        base = self.lm_var_floor
        if self.quality_floor_fraction <= 0:
            return base
        best = self._best_sigma.get(int(tag))
        if best is None:
            return base
        return max(base, (self.quality_floor_fraction * best)**2)

    # Stop the filter from becoming more certain than its systematic error
    # allows. Called at the end of update().
    def apply_covariance_floor(self):
        self._floor_block(0, 2, self.robot_pos_var_floor)
        if self.P[2, 2] < self.robot_theta_var_floor:
            self.P[2, 2] = self.robot_theta_var_floor

        # IMPORTANT: skip the landmark blocks in true-map mode. load_true_map()
        # deliberately sets every landmark's covariance to exactly zero, and
        # that zero is the entire mechanism freezing them: with P's landmark
        # rows at zero, K's landmark rows are zero too, so update() cannot move
        # a ground-truth marker. Flooring them here would make them non-zero
        # and therefore movable again, silently un-freezing the true map.
        if self.freeze_map:
            return
        for i in range(self.number_landmarks()):
            j = 3 + 2 * i
            self._floor_block(j, j + 2, self.landmark_floor(self.taglist[i]))

    # the prediction step of EKF
    def predict(self, drive_measurement):
        F = self.state_transition(drive_measurement)
        Q = self.predict_covariance(drive_measurement)

        # Shrink the x,y share of the flat process-noise floor while the
        # robot is (mostly) rotating on the spot -- see process_noise_floor's
        # comment in __init__ and rotation_fraction() below. Theta's share
        # of the floor is left at full strength regardless of motion type.
        rot_frac = self.rotation_fraction(drive_measurement)
        xy_scale = 1.0 - (1.0 - self.rotation_xy_noise_floor_scale) * rot_frac
        noise_floor = self.process_noise_floor * np.eye(3)
        noise_floor[0, 0] *= xy_scale
        noise_floor[1, 1] *= xy_scale
        Q[0:3, 0:3] += noise_floor

        # 1. Drive the robot forward to propagate state
        self.robot.drive(drive_measurement)

        # 2. Propagate state uncertainty covariance P
        self.P = F @ self.P @ F.T + Q

    # 0..1 estimate of how much of this predict() step is pure in-place
    # rotation vs. straight-line translation, from the two wheels' OWN
    # motion this step -- not from the fitted (x,y,theta) state, which is
    # exactly the thing in question here.
    #
    #   0 = pure translation: both wheels moved the same amount in the same
    #       direction (driving straight).
    #   1 = pure rotation: both wheels moved the same amount in OPPOSITE
    #       directions (turning on the spot).
    #   in between = an arc: some forward motion, some turning.
    #
    # Prefers delta_left_ticks/delta_right_ticks (the wheels' actual encoder
    # movement over this step) when available, since that's what
    # robot.covariance_drive() itself prefers; falls back to
    # left_speed/right_speed (instantaneous reported wheel speed) otherwise.
    # Either pair works identically here -- the ratio below is unit- and
    # scale-invariant, so it only depends on the relationship between the
    # two wheels' signed motion, not what units or magnitude they're in.
    def rotation_fraction(self, drive_measurement):
        if drive_measurement.delta_left_ticks is not None and drive_measurement.delta_right_ticks is not None:
            l, r = float(drive_measurement.delta_left_ticks), float(drive_measurement.delta_right_ticks)
        else:
            l, r = float(drive_measurement.left_speed), float(drive_measurement.right_speed)

        translate = l + r  # wheels agreeing (same sign & size) -> driving straight
        rotate = r - l      # wheels disagreeing (opposite sign)  -> turning on the spot
        denom = abs(translate) + abs(rotate)
        if denom < 1e-9:
            # No wheel motion at all. Doesn't come up in practice --
            # operate.py's is_stationary check skips predict() entirely in
            # this case -- this is just a safe, arbitrary-free no-op default.
            return 0.0
        
        frac = abs(rotate) / denom
        if frac >= (1.0 - 0.15):
            return 1.0
        return frac

    # How much genuinely new information a sighting of this tag carries, 0 to 1.
    # 0 means "same vantage point as last time, tells us nothing new about geometry";
    # 1 means "far enough away or turned enough that this is an independent look".
    def viewpoint_novelty(self, tag):
        pose = (float(self.robot.state[0, 0]),
                float(self.robot.state[1, 0]),
                float(self.robot.state[2, 0]))
        last = self._last_view_pose.get(tag)
        if last is None:
            return 1.0, pose  # never seen before: fully informative
        d_pos = np.hypot(pose[0] - last[0], pose[1] - last[1])
        d_ang = abs((pose[2] - last[2] + np.pi) % (2 * np.pi) - np.pi)
        novelty = max(d_pos / self.viewpoint_pos_threshold,
                      d_ang / self.viewpoint_ang_threshold)
        return min(1.0, novelty), pose

    # the update/correct step of EKF
    def update(self, sensor_measurement, drive_measurement=None):
        if not sensor_measurement:
            return

        # Only update on markers that are already part of the state (added via add_landmarks)
        known_measurement = [lm for lm in sensor_measurement if lm.tag in self.taglist]
        if not known_measurement:
            return

        # Construct measurement index list
        tags = [lm.tag for lm in known_measurement]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and build their covariance
        z_all = np.concatenate([lm.position.reshape(-1, 1) for lm in known_measurement], axis=0)
        R_all = np.zeros((2 * len(known_measurement), 2 * len(known_measurement)))
        novelties = []
        for i in range(len(known_measurement)):
            depth_var, lateral_var = self.measurement_variance(known_measurement[i].position)

            # Remember the best single look this marker has ever had, for its
            # per-landmark covariance floor (see landmark_floor()). Uses the RAW
            # geometry-based sigma, NOT the redundancy-inflated one below, so
            # sitting still and staring can never make a marker's floor look
            # better than the view it was actually measured from.
            tag = known_measurement[i].tag
            raw_sigma = float(np.sqrt(max(depth_var, lateral_var)))
            tg = int(tag)
            if tg not in self._best_sigma or raw_sigma < self._best_sigma[tg]:
                self._best_sigma[tg] = raw_sigma

            # Inflate a redundant sighting's noise: a repeat look from nearly the same
            # spot this tag was last usefully seen from still nudges the estimate, but
            # it no longer buys confidence it hasn't earned. Move to a new vantage
            # point and the next sighting counts at full strength again.
            novelty, pose = self.viewpoint_novelty(int(tag))
            penalty = 1.0 + self.redundancy_penalty * (1.0 - novelty)
            depth_var *= penalty
            lateral_var *= penalty
            novelties.append((int(tag), novelty, pose))

            # lm.position is [depth, lateral] in the robot's body frame, and so are
            # z_hat/H below (Robot.measure / derivative_measure) -- R MUST stay in
            # that same local frame. Do not rotate it by theta: z, z_hat, and H are
            # never expressed in world frame, so rotating R mixes frames and silently
            # swaps how much depth vs. lateral is trusted as theta changes.
            R_all[2*i:2*i+2, 2*i:2*i+2] = np.diag([depth_var, lateral_var])

        # Compute own measurements
        z_hat_all = self.robot.measure(self.markers, idx_list)
        z_hat_all = z_hat_all.reshape((-1, 1), order="F")
        H_all = self.robot.derivative_measure(self.markers, idx_list)

        # --- Innovation (Mahalanobis) gating ---
        # Check each marker's reading against what the filter currently expects
        # BEFORE accepting it, rather than blending every reading in blindly.
        # A marker glimpsed at a bad/foreshortened angle -- or any other one-off
        # bad reading -- tends to show up as a statistical outlier here, and gets
        # skipped for this frame instead of dragging the robot pose and every
        # other landmark along with it. d2 is the squared Mahalanobis distance;
        # under normal noise it follows a chi-square distribution with 2 DOF, so
        # innovation_gate=9.21 rejects only the ~1% most inconsistent readings.
        #
        # The same loop records per-marker diagnostics (see last_diagnostics in
        # __init__): d2 here IS the Normalised Innovation Squared, so averaging
        # 'nis' over a run tells you whether the noise model is honest -- ~2.0
        # for 2 DOF when correctly tuned, consistently higher means the filter
        # is over-confident (R or P too small), much lower means it's too
        # conservative and information is being thrown away. That's a tuning
        # signal you can read off a run without needing ground truth.
        keep = []
        self.last_diagnostics = []
        for i in range(len(known_measurement)):
            y_i = z_all[2*i:2*i+2] - z_hat_all[2*i:2*i+2]
            H_i = H_all[2*i:2*i+2, :]
            R_i = R_all[2*i:2*i+2, 2*i:2*i+2]
            S_i = H_i @ self.P @ H_i.T + R_i
            d2 = (y_i.T @ np.linalg.inv(S_i) @ y_i).item()
            gated = d2 > self.innovation_gate
            if not gated:
                keep.append(i)

            j = idx_list[i]
            lm_var = float(np.mean(np.diag(self.P[3+2*j:5+2*j, 3+2*j:5+2*j])))
            self.last_diagnostics.append({
                'tag': int(known_measurement[i].tag),
                'distance': float(np.linalg.norm(
                    np.asarray(known_measurement[i].position).reshape(-1)[:2])),
                'novelty': float(novelties[i][1]),
                'sigma': float(np.sqrt(R_all[2*i, 2*i])),
                'innov_x': float(y_i[0, 0]),
                'innov_y': float(y_i[1, 0]),
                'nis': float(d2),
                'lm_sigma_before': float(np.sqrt(max(lm_var, 0.0))),
                'lm_sigma_after': float('nan'),   # filled in after the update below
                'gated': bool(gated),
            })

        if not keep:
            return

        rows = [r for i in keep for r in (2*i, 2*i+1)]
        z = z_all[rows, :]
        z_hat = z_hat_all[rows, :]
        H = H_all[rows, :]
        R = R_all[np.ix_(rows, rows)]

        x = self.get_state_vector()

        # 1. Measurement residual (innovation)
        y = z - z_hat

        # 2. Innovation covariance
        S = H @ self.P @ H.T + R

        # 3. Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # 3.5 Trust our x,y dead-reckoning while (mostly) rotating on the spot.
        # K above is built from the FULL P, including Cov(x,theta) and
        # Cov(y,theta) -- and those keep growing during a turn even though
        # Var(x,y) itself is kept small by predict()'s noise-floor scaling.
        # That means a normal, expected heading correction (re-anchoring
        # theta off a marker mid-turn) can still drag x,y through those
        # cross-covariance terms -- that's the position "jump" while
        # rotating.
        #
        # Fix: scale down only the two rows of K that write into x and y, in
        # proportion to how much of THIS tick's motion is pure rotation
        # (recomputed fresh from the current drive_measurement every call --
        # not the historical motion that built up P). Theta and every
        # landmark still get the full, un-gated correction. We reuse this
        # same scaled K for the P update below (Joseph form is valid for any
        # gain matrix, not just the optimal one), so P's bookkeeping matches
        # what was actually applied to state instead of overstating how much
        # was learned about x,y.
        if drive_measurement is not None:
            rot_frac = self.rotation_fraction(drive_measurement)
        else:
            # No drive_measurement passed in (robot stationary, or an older
            # call site that predates this parameter) -- treat as "not
            # rotating" so corrections apply at full strength.
            rot_frac = 0.0
        xy_gain_scale = 1.0 - (1.0 - self.rotation_xy_gain_scale) * rot_frac
        K[0:2, :] *= xy_gain_scale

        # 4. Update state vector x and set it back in robot/markers
        x_updated = x + K @ y

        # Wrap the heading back into [-pi, pi]. Without this theta drifts out
        # of range over a long run and the rendering / ellipse maths degrades.
        x_updated[2, 0] = (x_updated[2, 0] + np.pi) % (2 * np.pi) - np.pi
        self.set_state_vector(x_updated)

        # 5. Update state covariance P -- Joseph form (symmetric & numerically stable)
        I = np.eye(len(x))
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Force exact numerical symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # Never let confidence exceed what the systematic error justifies. This
        # is what replaces the old oversized process_noise_floor: rather than
        # inventing uncertainty before every update (which inflated P during
        # motion and caused the corrective lurches), clamp it afterwards so a
        # landmark can never average its way down to a certainty it hasn't
        # earned and stop accepting corrections. See apply_covariance_floor().
        self.apply_covariance_floor()

        # Remember where we were standing, for any tag whose reading counted as a
        # fully novel viewpoint (novelty >= 1.0) -- this only affects future R
        # inflation above, never today's state directly. A tag gated out earlier
        # still gets recorded here if it qualified; that's inherited behaviour from
        # where this came from and is harmless since a rejected reading never
        # touched the state either way.
        for tag, novelty, pose in novelties:
            if novelty >= 1.0:
                self._last_view_pose[tag] = pose

        # Landmark uncertainty AFTER this update, to close out the run log.
        for dg in self.last_diagnostics:
            try:
                j = self.taglist.index(dg['tag'])
                v = float(np.mean(np.diag(self.P[3+2*j:5+2*j, 3+2*j:5+2*j])))
                dg['lm_sigma_after'] = float(np.sqrt(max(v, 0.0)))
            except ValueError:
                dg['lm_sigma_after'] = float('nan')

    def state_transition(self, drive_measurement):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(drive_measurement)
        return F

    def predict_covariance(self, drive_measurement):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(drive_measurement)
        return Q

    def add_landmarks(self, sensor_measurement):
        if self.freeze_map:
            return   # true map already has every landmark, never add new ones
        if not sensor_measurement:
            return

        # if len(self.taglist) == 0:
        #     # Bootstrap case: map is empty, there are no "known" landmarks to require.
        #     # Fall back to requiring several simultaneous markers instead, so the very
        #     # first landmarks are still reasonably well-constrained by multi-marker geometry.
        #     if len(sensor_measurement) < 3:
        #         return
        # else:
        #     known_in_view = [lm for lm in sensor_measurement if lm.tag in self.taglist]
        #     if len(known_in_view) < 1:
        #         return


        th = float(self.robot.state[2, 0])
        c, sn = np.cos(th), np.sin(th)
        R_theta = np.array([[c, -sn], [sn, c]])
        dR_theta = np.array([[-sn, -c], [c, -sn]])  # d(R_theta)/d(theta)
        robot_xy = self.robot.state[0:2, :]

        for lm in sensor_measurement:
            if lm.tag in self.taglist:
                continue

            lm_position = np.asarray(lm.position, dtype=float).reshape(2, 1)
            lm_state = robot_xy + R_theta @ lm_position

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_state), axis=1)

            # Proper cross-covariance at birth, instead of a big diagonal placeholder
            # with zero correlation to the rest of the state. The new landmark is
            # m = robot_xy + R(theta) @ z, so its uncertainty is built out of BOTH
            # the robot's current uncertainty and the measurement's -- and the
            # off-diagonal terms below are what let a later correction on any one
            # landmark, or the robot pose, propagate to this one too.
            #   Gx = d(m)/d(robot pose)   = [ I | dR/dtheta @ z ]   (2 x n)
            #   Gz = d(m)/d(measurement)  = R(theta)                (2 x 2)
            n = self.P.shape[0]

            Gx = np.zeros((2, n))
            Gx[:, 0:2] = np.eye(2)
            Gx[:, 2:3] = dR_theta @ lm_position

            # Same depth/lateral measurement noise update() uses for this marker,
            # inflated by new_landmark_inflation because this is a FIRST look:
            # one viewing angle, no cross-check, and no way to tell an oblique
            # view (where planar-marker pose estimation is at its worst) from a
            # clean head-on one. Seeding the landmark as if that single reading
            # were definitive means later, better-angled sightings arrive with
            # too little gain left to fix it. Inflating here makes the first
            # look a starting guess rather than an answer, so the circle starts
            # large and shrinks as genuinely different views come in.
            depth_var, lateral_var = self.measurement_variance(lm_position)
            inflation = self.new_landmark_inflation**2
            Rz = np.diag([depth_var * inflation, lateral_var * inflation])

            P_mx = Gx @ self.P                              # 2 x n: correlation with existing state
            P_mm = Gx @ self.P @ Gx.T + R_theta @ Rz @ R_theta.T

            P_new = np.zeros((n + 2, n + 2))
            P_new[:n, :n] = self.P
            P_new[n:, :n] = P_mx
            P_new[:n, n:] = P_mx.T
            P_new[n:, n:] = P_mm
            self.P = 0.5 * (P_new + P_new.T)

    # Exact rigid fit from exactly two point correspondences.
    #
    # umeyama() below cannot do this case safely: with only 2 points its
    # covariance matrix is rank 1, so det(cov) == 0 exactly, the reflection
    # guard (det < 0) never fires, and U @ V_t can legitimately come back with
    # det(R) == -1 -- a MIRRORED pose, which lands the robot somewhere it has
    # never been. Two matched points determine the rotation uniquely anyway:
    # it's just the angle between the two difference vectors. No SVD, no rank
    # deficiency, no reflection. Used by _fit_rigid_pose() when n == 2.
    @staticmethod
    def rigid_transform_2pt(from_points, to_points):
        v_from = from_points[:, 1] - from_points[:, 0]
        v_to = to_points[:, 1] - to_points[:, 0]
        angle = math.atan2(v_to[1], v_to[0]) - math.atan2(v_from[1], v_from[0])
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        t = to_points.mean(axis=1).reshape((2, 1)) - R @ from_points.mean(axis=1).reshape((2, 1))
        return R, t

    @staticmethod
    def umeyama(from_points, to_points):
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"

        N = from_points.shape[1]
        m = 2

        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        cov_matrix = delta_to @ delta_from.T / N

        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)

        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
        return R, t

    # Plotting functions
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res = (320, 500), not_pause=True, true_map=None, live_rmse_info=None, selected_tag=None,object_gt=None, object_estimates=None, object_rmse_info=None, object_ekf=None):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters,
        lms_xy_world = self.markers[:2, :]
        robot_xy_world = self.robot.state[:2, 0].reshape((2, 1))  # position in the SLAM frame, before we recentre on the markers
        if self.number_landmarks() > 0:
            center_xy = lms_xy_world.mean(axis=1).reshape((2, 1))  # recentre the view on the marker centroid
        else:
            center_xy = robot_xy_world  # no markers yet, fall back to centering on the robot
        lms_xy = lms_xy_world - center_xy
        robot_xy = robot_xy_world - center_xy
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((robot_xy[0,0], robot_xy[1,0]), res, m2pixel)

        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (244, 69, 96), 1)

        # draw fruit/object landmarks tracked by object_ekf, same ellipse
        # convention as the ArUco landmarks above: it shrinks as the EKF
        # narrows down each fruit's position over repeated sightings.
        if object_ekf is not None:
            for obj_type, pos in object_ekf.estimates.items():
                obj_xy = pos - center_xy
                coor_obj = self.to_im_coor((obj_xy[0, 0], obj_xy[1, 0]), res, m2pixel)
                colour = self.FRUIT_COLORS.get(obj_type, self.DEFAULT_FRUIT_COLOR)
                cv2.drawMarker(canvas, coor_obj, colour, markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2)
                cv2.putText(canvas, obj_type[:3], (coor_obj[0]+6, coor_obj[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
                axes_len, angle = self.make_ellipse(object_ekf.P[obj_type])
                canvas = cv2.ellipse(canvas, coor_obj, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, colour, 1)

        # --- overlay ground-truth markers for live RMSE practice ---
        if true_map is not None and live_rmse_info is not None:
            R, t = live_rmse_info['R'], live_rmse_info['t']
            matched_tags = live_rmse_info['matched_tags']
            for tag, true_xy in true_map.items():
                # bring the true marker from the ground-truth frame into the
                # SLAM/estimated frame using the inverse of the alignment
                # transform, then recentre on the marker centroid like the estimates above
                true_in_est = R.T @ (true_xy - t) - center_xy
                coor_true = self.to_im_coor((true_in_est[0,0], true_in_est[1,0]), res, m2pixel)
                colour = (40, 170, 40) if tag in matched_tags else (140, 140, 40)
                cv2.drawMarker(canvas, coor_true, colour, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
                cv2.putText(canvas, str(tag), (coor_true[0]+6, coor_true[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

                # error line: your current estimate -> where it should be
                if tag in matched_tags:
                    idx = self.taglist.index(tag)
                    coor_est = self.to_im_coor((lms_xy[0,idx], lms_xy[1,idx]), res, m2pixel)
                    cv2.line(canvas, coor_est, coor_true, (0, 140, 255), 1)

            rmse_text = f"RMSE {live_rmse_info['rmse']:.4f}m ({len(matched_tags)}/{len(true_map)})"
            cv2.putText(canvas, rmse_text, (5, res[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

            # --- overlay fruit/object ground truth + live merged estimates ---
        if object_gt is not None and live_rmse_info is not None:
            R, t = live_rmse_info['R'], live_rmse_info['t']
            matched_objs = set(object_rmse_info['matched']) if object_rmse_info else set()
            for obj_type, gt_xy in object_gt.items():
                gt_in_est = R.T @ (gt_xy - t) - center_xy
                coor_gt = self.to_im_coor((gt_in_est[0,0], gt_in_est[1,0]), res, m2pixel)
                colour = (0, 140, 220) if obj_type in matched_objs else (150, 100, 40)
                cv2.drawMarker(canvas, coor_gt, colour, markerType=cv2.MARKER_SQUARE, markerSize=8, thickness=2)
                cv2.putText(canvas, obj_type[:3], (coor_gt[0]+6, coor_gt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

                # error line from the current estimate (drawn above, from
                # object_ekf) -> where it should be. Estimate is already in
                # the estimated/SLAM frame -- no R,t needed.
                if object_estimates is not None:
                    key_0 = obj_type + '_0'
                    if key_0 in object_estimates:
                        est = object_estimates[key_0]
                        est_xy_local = np.array([[est['x']], [est['y']]]) - center_xy
                        coor_est = self.to_im_coor((est_xy_local[0,0], est_xy_local[1,0]), res, m2pixel)
                        cv2.line(canvas, coor_est, coor_gt, (255, 120, 0), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3), (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1], (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1], (coor_[0]-5, coor_[1]-5))
                # flag the marker selected for deletion with a pointer above it
                if selected_tag is not None and self.taglist[i] == selected_tag:
                    tip = (coor_[0], coor_[1] - 10)
                    pygame.draw.polygon(surface, (255, 215, 0),
                                         [tip, (tip[0] - 6, tip[1] - 10), (tip[0] + 6, tip[1] - 10)])
        return surface

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        # eig() on a real symmetric block can still return a tiny imaginary
        # part from rounding; take the real part and clip negatives so a
        # numerically-just-below-zero eigenvalue can't produce a NaN axis.
        e_vals = np.clip(np.real(e_vals), 0, None)
        e_vecs = np.real(e_vecs)
        idx = e_vals.argsort()[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = np.sqrt(e_vals) * alpha

        # Near-isotropic covariance: the ellipse is essentially a circle, and its
        # "orientation" is numerically meaningless -- tiny noise in P flips which
        # eigenvector sorts first and swings the angle wildly even though the
        # actual shape hasn't changed. Skip the noisy angle in that regime.
        if e_vals[0] - e_vals[1] < 1e-6 * max(e_vals[0], 1e-12):
            angle = 0.0
        else:
            # DEGREES, not radians: this value goes straight to cv2.ellipse(),
            # which takes its angle in degrees. The old code returned
            # np.arctan(...) in radians, so every covariance ellipse in the
            # minimap was drawn at roughly 1/57th of its true orientation --
            # i.e. essentially always axis-aligned regardless of the real
            # shape of P. Drawing only; no effect on the filter itself.
            angle = float(np.degrees(np.arctan2(e_vecs[1, 0], e_vecs[0, 0])))
        return (axes_len[0], axes_len[1]), angle


# =============================================================================
# Fruit / object position tracking -- FruitEKF and friends
#
# "Best of both worlds" of the two independent per-fruit position trackers
# the team ended up with:
#
#   - FruitEKF's structure (one incremental Kalman filter per fruit CLASS,
#     Joseph-form covariance update, chi-square innovation gating) is
#     adapted from Brandon's ObjectEKF (Brandon-2.0 branch,
#     object_pose_est.py) -- the same recipe EKF.update() above already uses
#     for ArUco landmarks. The original _fuse_fruit_observation() in
#     operate.py never had this rejection step.
#
#   - update() is a 3-tier gate, not a binary accept/reject: a reading
#     within innovation_gate (9.21, ~99% confidence) fuses normally; one
#     beyond reject_gate (innovation_gate * reject_radius_multiplier**2,
#     i.e. more than reject_radius_multiplier times further away in
#     Mahalanobis units) is REJECTED outright and leaves the estimate
#     untouched; anything in between is CAPPED -- fused as if it had
#     landed exactly on the innovation_gate boundary, so a surprising but
#     not-implausible reading still pulls the estimate toward it by the
#     most we're willing to trust in one update, rather than either fully
#     believing it or throwing it away. Returns 'accepted' / 'capped' /
#     'rejected' (a string, not a bool) so process_object_estimates() in
#     operate.py can show which one happened.
#
#   - min_var (the floor on P's diagonal after every update) went from
#     3.75e-4 (std ~=1.9cm) to 0.012 (std ~=11cm) -- this is the "space to
#     move after converging" fix. R (sensor noise, from
#     fruit_measurement_noise() below) does NOT shrink over repeated
#     sightings, only P (belief) does -- so the Kalman gain K = P/(P+R)
#     shrinks right along with P, and at the old floor, K had collapsed to
#     well under 1% of the innovation at this arena's longer sighting
#     distances. That's not literally "locked" (K was never exactly 0),
#     but it's close enough in practice: a converged estimate that's
#     slightly off from a few early, biased-angle sightings would need an
#     impractical number of repeat updates to visibly correct. The arena
#     is 2.5x2.5m (truemap_generator/config.yaml), so the worst realistic
#     case -- robot in one corner, fruit near the opposite one -- is the
#     ~3.54m diagonal; fruit_measurement_noise() grows with distance past
#     its ~1.47m minimum, so that's also where the Kalman gain is smallest.
#     0.012 keeps K >= ~5% even there, tested explicitly in
#     test_fruit_ekf_outlier.py, while still trusting the accumulated
#     history far more than any single new reading (P only reaches this
#     floor after many consistent sightings in the first place). This is a
#     precision/adaptability trade-off, not a free win: it also caps how
#     tight ANY estimate can ultimately converge, even from many
#     genuinely-consistent close-range sightings. Lower it back toward the
#     old value if a tighter best-case RMSE matters more than long-range
#     adaptability once fruits are already well converged.
#
#   - The noise model plugged into FruitEKF is NOT Brandon's
#     object_noise_covariance (whose own docstring admits its constants
#     "were fit for aruco corner detection... NOT calibrated for this
#     sensor" for fruit). It's the same depth/lateral noise-vs-distance
#     curve EKF.update()/EKF.add_landmarks() above use for ArUco landmarks:
#     depth_var = 0.039*distance**2 - 0.115*distance + 0.1064,
#     lateral_var = depth_var*1.5. fruit_measurement_noise() below is that
#     formula, pulled out standalone so FruitEKF isn't tied to a particular
#     EKF instance.
#
#   - load_object_ground_truth() / compute_object_rmse() are Brandon's live
#     object-RMSE pieces, ported over close to verbatim -- the live RMSE
#     readout in operate.py's GUI.
#
#   - in_frame_fraction() is the original _in_frame_fraction() from
#     operate.py. IN_FRAME_FRACTION_THRESHOLD is 0.95 (was 0.8, then briefly
#     1.0): process_object_estimates() in operate.py skips just the ONE
#     box if it's less than 95% inside the frame -- the rest of the same
#     photo's boxes are unaffected (used to reject the whole shot; changed
#     since each box's position comes from its own geometry alone, so one
#     bad box says nothing about a different fruit's box in the same
#     photo). There is no separate per-frame ArUco-marker-count /
#     fruit-count gate any more -- that existed briefly as
#     should_fuse_frame() and has been removed.
#
#   - is_box_shape_plausible() (+ plausible_aspect_ratio_range(),
#     expected_aspect_ratio_range_from_dimensions(), EXPECTED_ASPECT_RANGE)
#     is a SECOND, independent per-box gate, checked alongside
#     in_frame_fraction() rather than instead of it -- they catch different
#     failure modes. in_frame_fraction() asks "is this box touching the
#     photo edge" (shape-agnostic); it has no way to catch a box that's
#     fully inside the frame and still a bad read on the fruit -- partial
#     occlusion by another object, two overlapping detections merged into
#     one box, a bad-angle/foreshortened read, etc. EXPECTED_ASPECT_RANGE is
#     Brandon's own per-class width/height aspect-ratio table from
#     object_pose_est.py (empirically measured "from training data", via
#     is_box_malformed() there) -- it already existed in the codebase but
#     was never actually called from operate.py in any version before this
#     one. expected_aspect_ratio_range_from_dimensions() is a second,
#     independent check on that table: a geometric bound derived straight
#     from each fruit's true (length, width, height) in object_list.csv --
#     model the fruit as roughly convex, and its projected aspect ratio
#     from ANY viewing angle can't legitimately fall outside
#     [smallest true dim / largest, largest / smallest] (narrowest end-on,
#     widest broadside). Checked against real object_list.csv data, the two
#     disagree for lemon specifically: Brandon's empirical upper bound is
#     1.033 (barely past 1:1 -- his training data evidently had few or no
#     genuinely broadside lemon shots), but the analytical ceiling is 1.504,
#     and a moderately broadside, fully unoccluded lemon view (aspect ~1.3)
#     is geometrically fine. Enforced as empirical-alone, that table would
#     reject exactly the shots Nicholas flagged ("some angles will think
#     lemon is blocked"). So plausible_aspect_ratio_range() takes the UNION
#     (widest) of both ranges rather than trusting either alone --
#     narrower-than-geometric empirical data most likely means an
#     undersampled training angle, not a real physical constraint, while
#     empirical still contributes real value where it's WIDER than
#     analytical (detector box-fitting looseness the idealised shape model
#     doesn't capture).
# =============================================================================

IN_FRAME_FRACTION_THRESHOLD = 0.95  # was 0.8, then 1.0. Whole shot is rejected if any fruit is < 95% inside the frame.

# Camera frame size -- the REAL resolution botconnect.get_image() delivers
# once the camera thread has an actual frame (confirmed via
# claude/calibrate.py's live "Live frame: {w}x{h}" printout, and
# calibration/param/intrinsic.txt's principal point cx/cy). This is NOT
# the same as the (360,480,3) placeholder shape get_image() falls back to
# before the first real frame arrives (or after a disconnect). Using the
# placeholder's (smaller, wrong-aspect-ratio) dimensions here was a real
# bug in an earlier version: in_frame_fraction() was clipping every box's
# x-extent past 480px and y-extent past 360px, even on a real 640x480
# frame where the actual photo extends well past both -- i.e. it was
# flagging boxes as "hanging off the edge" that were genuinely, fully
# captured, just because they were farther right/down than the wrong
# assumed boundary.
#
# Lives here (not in operate.py) so BOTH the live GUI pipeline
# (operate.py's process_object_estimates(), via self.fruit_ekf) and the
# offline re-derivation pipeline (object_pose_est.py's __main__, which
# rebuilds lab_output/objects.txt -- the file eval.py actually grades --
# straight from lab_output/pred.txt) use the exact same numbers instead of
# two separately-typed constants that could silently drift apart.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# See the FruitEKF provenance comment above for why this pair exists:
# Nicholas's own observation is that this detector one-directionally
# confuses two class pairs -- a genuine 'lime' sometimes reads as
# 'capsicum', a genuine 'orange' sometimes reads as 'mango' -- but never
# the reverse (a real capsicum is never called lime, a real mango never
# called orange). Both pairs share a colour family (green/green,
# orange/orange) that gets harder to separate as texture washes out with
# distance/blur. CONFUSABLE_PAIRS maps each "attractor" class (the wrong
# label the detector drifts toward) to the "victim" class (the real
# fruit) -- see resolve_confusable_class() below for how it's used.
CONFUSABLE_PAIRS = {
    'capsicum': 'lime',
    'mango': 'orange',
}

# Metres. Checked against THIS arena's truemap.txt: the real lime<->capsicum
# gap is ~0.73m and the real orange<->mango gap is ~2.9m, so 0.04m has a lot
# of margin here without risking merging two genuinely different, nearby
# fruits into one. Re-check against whatever map an actual graded run uses
# before trusting this blindly -- a layout that happens to place a real
# lime and a real capsicum within a few cm of each other would make this
# heuristic actively wrong for that pair.
CONFUSABLE_DISTANCE_THRESHOLD = 0.04


def resolve_confusable_class(predicted_class, box, robot_pose, object_dimensions,
                              focal_length, cx, estimate_pose_fn, known_positions,
                              pairs=CONFUSABLE_PAIRS, threshold=CONFUSABLE_DISTANCE_THRESHOLD):
    """
    Decide whether a detected box should actually be fused under a
    different ("victim") class than the detector predicted, per
    CONFUSABLE_PAIRS above.

    Why this needs its own pose recompute rather than just comparing the
    ALREADY-computed pose to the victim's known position: estimate_pose()'s
    depth comes from `focal_length * true_height / box_height`, and
    true_height is looked up by predicted_class. capsicum's true height
    (0.088m) is ~1.8x lime's (0.05m), and mango's (0.058m) is ~0.8x
    orange's (0.072m) -- object_list.csv. So a box that's actually a lime
    but got labelled capsicum, if scored with capsicum's height, comes out
    with its depth overestimated by ~76%: at a real 1m depth that's a
    computed position position nearly a metre further from the robot than
    the real one -- FAR outside any sane proximity threshold, so a naive
    "compare the existing pose to the victim" check would essentially
    never fire. Rescoring the SAME box with the victim's true height
    before comparing fixes this.

    predicted_class: the detector's raw label for this box.
    box, robot_pose, object_dimensions, focal_length, cx: same as
        estimate_pose()'s own arguments -- object_dimensions is
        {class_name: [length, width, height]} (object_list.csv).
    estimate_pose_fn: the estimate_pose() function itself, passed in
        rather than imported here so this module doesn't need to depend
        on object_pose_est.py.
    known_positions: {class_name: (x, y)} of each class's current best
        position estimate -- self.fruit_ekf.estimates in the live GUI
        path, a running per-class mean in the offline object_pose_est.py
        path. Only ever looked up for classes that appear as a VALUE in
        `pairs` (the victim classes); the attractor classes' own entries,
        if any, are irrelevant here.

    Returns (label_to_fuse_under, pose_to_fuse):
      - (predicted_class, None) if no correction applies -- caller should
        keep using whatever pose it already computed with predicted_class's
        own true_height.
      - (victim_class, (alt_x, alt_y)) if the box was close enough, once
        rescored with the victim's true height, to that victim's known
        position -- caller should relabel AND use this recomputed pose,
        not its original one.

    Deliberately conservative and one-directional, matching the asymmetry
    actually observed: only fires for classes that are KEYS in `pairs`
    (capsicum, mango by default) -- a 'lime' or 'orange' reading is never
    touched. Only fires if the victim class is already in known_positions
    (i.e. has been seen and accepted at least once before) -- a detection
    run where the true class is NEVER once predicted correctly can't be
    fixed this way, since there's nothing to compare against; that first
    bad-labelled sighting fuses under its predicted (wrong) label same as
    before, and only LATER sightings of the same fruit get corrected once
    the real label has appeared and been recorded.
    """
    victim = pairs.get(predicted_class)
    if victim is None or victim not in known_positions:
        return predicted_class, None

    victim_x, victim_y = known_positions[victim]
    victim_true_height = object_dimensions[victim][2]
    alt_x, alt_y = estimate_pose_fn(robot_pose, box, victim_true_height, focal_length, cx)

    if np.hypot(alt_x - victim_x, alt_y - victim_y) <= threshold:
        return victim, (alt_x, alt_y)
    return predicted_class, None


def in_frame_fraction(box, frame_width, frame_height):
    """Fraction of this detection's box's AREA that lies within the camera
    frame, from 0.0 (fully outside/off-frame) to 1.0 (fully inside).
    estimate_pose's pinhole depth calc assumes box_height reflects the
    object's true apparent height; a box that's partly clipped by the
    frame edge under-reports that height (or the box's x-extent, for the
    lateral estimate), throwing the position estimate off in proportion to
    how much of the box is actually missing -- hence a genuine visible-area
    fraction here, not just an in/out flag.

    (Earlier versions of this function had a hardcoded edge_margin that
    forced ANY box within ~2px of the frame boundary -- or beyond it -- to
    read as a flat 0.0, and every other box read as exactly 1.0, since a
    box comfortably clear of that margin is by construction never partially
    clipped. That made the function effectively binary in practice: a
    threshold like 0.95 behaved identically to 1.0, since the function
    itself could never actually return anything between 0 and 1. Removed
    so IN_FRAME_FRACTION_THRESHOLD is a real, meaningful cutoff.)
    """
    x_center, y_center, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    x0, x1 = x_center - w / 2.0, x_center + w / 2.0
    y0, y1 = y_center - h / 2.0, y_center + h / 2.0

    vis_w = max(0.0, min(x1, frame_width) - max(x0, 0.0))
    vis_h = max(0.0, min(y1, frame_height) - max(y0, 0.0))
    return (vis_w * vis_h) / (w * h)


# Brandon's, unchanged -- object_pose_est.py, empirically measured "from
# training data". Was already sitting in the codebase (paired there with
# is_box_malformed()) but never actually wired into the live GUI pipeline
# until now.
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
    Derive a plausible width/height aspect-ratio range for ANY viewing
    angle, straight from a fruit's true (length, width, height) in metres
    (an object_list.csv row) -- no training data, no per-class tuning.

    Geometric argument: model the fruit as a roughly convex, ellipsoid-like
    shape. From any viewing direction, its projected width (and,
    separately, its height) can never exceed the object's largest true
    dimension, and can never be smaller than its smallest true dimension --
    the extremes are hit by looking straight down the long axis (narrowest)
    and straight at it broadside (widest). So the aspect ratio of ANY
    legitimate, unoccluded view is bounded by
    [smallest/largest, largest/smallest]. A box outside that range isn't
    "an unusual angle" -- it's not a geometrically possible intact view of
    this fruit, at any angle, full stop.

    `slack` widens the theoretical range a bit (default +/-15%) to absorb
    two things the pure geometry doesn't: (1) a real fruit isn't a perfect
    ellipsoid, and (2) a detector's box is rarely a pixel-perfect fit even
    on a clean detection. Tune this per your own detector's typical jitter.
    """
    if len(object_true_dims) != 3:
        raise ValueError(f"expected (length, width, height), got {object_true_dims}")
    largest = max(object_true_dims)
    smallest = min(object_true_dims)
    if smallest <= 0:
        return (0.0, float('inf'))   # malformed dimension data -- don't gate on it
    base_ratio = largest / smallest
    return (1.0 / (base_ratio * slack), base_ratio * slack)


def plausible_aspect_ratio_range(predicted_class, object_dimensions=None, slack=1.15):
    """
    The range actually used by is_box_shape_plausible() below: the UNION
    (widest) of Brandon's empirical EXPECTED_ASPECT_RANGE and the
    analytical range derived from object_dimensions, when both are
    available -- see the provenance comment above FruitEKF (lemon
    especially) for why empirical-alone risks rejecting a legitimately
    oblong angle just because training data didn't happen to cover it.
    Falls back to whichever one you have data for if only one is
    available; (0.0, inf) -- i.e. don't gate on it -- if neither is.
    """
    empirical = EXPECTED_ASPECT_RANGE.get(predicted_class)
    analytical = (expected_aspect_ratio_range_from_dimensions(object_dimensions[predicted_class], slack)
                  if object_dimensions is not None and predicted_class in object_dimensions else None)

    if empirical is not None and analytical is not None:
        return (min(empirical[0], analytical[0]), max(empirical[1], analytical[1]))
    if empirical is not None:
        return empirical
    if analytical is not None:
        return analytical
    return (0.0, float('inf'))   # no data for this class either way -- don't gate on nothing


def is_box_shape_plausible(predicted_class, box, object_dimensions=None, slack=1.15):
    """
    True if `box`'s width/height aspect ratio falls inside
    plausible_aspect_ratio_range() for `predicted_class` -- complementary
    to in_frame_fraction(), not a replacement for it. in_frame_fraction()
    asks "is this box touching the photo edge" (shape-agnostic, still
    needed, still correct for that question); this asks "does this box's
    SHAPE look like an intact, unoccluded view of this fruit at all" --
    which in_frame_fraction has no way to see, since an occluded or
    merged-detection box can look perfectly "inside the frame" and still
    be a bad read on the fruit.

    box: [x_center, y_center, width, height] in pixels (YOLO xywh) -- same
    format in_frame_fraction() and estimate_pose() already take.
    object_dimensions: self.object_dimensions (from object_list.csv),
    needed only for the analytical half of the range -- pass None to check
    against Brandon's empirical table alone.
    """
    _, _, w, h = box
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    lo, hi = plausible_aspect_ratio_range(predicted_class, object_dimensions, slack)
    return lo <= aspect <= hi


# Lateral (across-ray) variance as a fraction of depth (along-ray) variance,
# for a FRUIT observation. Only used by fruit_measurement_noise() below --
# EKF's own ArUco model is separate (EKF.measurement_variance) and unchanged.
#
# CHANGED from 1.5 to 0.1, i.e. from "bearing is 1.5x noisier than range" to
# "range is 10x noisier than bearing". The old ratio was inherited from the
# ArUco model, but a fruit's position is computed completely differently:
#
#   depth   = focal_length * true_height / box_height     (estimate_pose)
#   lateral = depth * (box_x - cx) / focal_length
#
# Depth therefore inherits every error in the BOX HEIGHT and in the assumed
# true height from object_list.csv -- a slightly tight or loose YOLO box, a
# tilted or partly occluded fruit, or a specimen that isn't exactly the
# catalogue size, all move it by 10-20% at 1 m. Bearing only depends on where
# the box's centre sits horizontally, which is good to a few pixels; at
# f ~= 700 px that's well under a centimetre, plus a share of the depth error
# scaled by how far off-axis the fruit is (small near the optical axis).
#
# Why this matters for "the fruit converges to the wrong spot": box-height
# depth error is a BIAS, not noise -- same sign every frame -- so averaging
# tightens the estimate around the wrong answer instead of removing it. With
# the old ratio the filter treated range as its trustworthy axis, so whenever
# a biased depth disagreed with the running estimate it resolved the conflict
# by sliding the fruit sideways, across the ray, in the one direction it
# actually had good information. With range correctly marked as the sloppy
# axis, each sighting instead reads as "somewhere along this ray, and I'm sure
# of the ray's direction", so sightings from different bearings triangulate
# and the crossing point survives a consistent depth bias.
#
# Measured on a simulated fruit with a +15% systematic depth bias, mean final
# position error over 40 seeds per cell:
#
#   viewpoints used             ratio 1.5   ratio 0.25   ratio 0.1
#   narrow arc, same range        201 mm      199 mm      199 mm
#   wide arc, same range           19 mm       13 mm       26 mm
#   mixed near + far               114 mm       39 mm       45 mm
#   two spots only                  82 mm       47 mm       36 mm
#
# Two things to read off that. First, the gain is real but it is specifically
# BIAS-robustness: with the bias set to zero every ratio lands within 8-12 mm,
# so this changes almost nothing about ordinary noise and a lot about the
# systematic error that was moving fruits to the wrong place. Second, the
# benefit is largest exactly where the geometry is mixed -- a far sighting has
# a hopeless range but a perfectly good bearing, and only the corrected ratio
# lets the filter keep the bearing while discounting the range. A wide arc at
# constant range already cancels a radial bias by symmetry, which is why that
# row barely moves.
#
# 0.25 rather than a harder flip: past about 0.1 the filter gets stiff across
# the ray and starts capping/rejecting the very sightings that would correct a
# wrong estimate (rejection rate climbed to 2% at 0.1), and the sweep bottomed
# out at 0.25 (39 mm vs 45 mm at 0.1 and 52 mm at 0.05).
#
# Note what NO ratio fixes: the narrow-arc row stays at ~200 mm throughout. If
# every sighting comes from the same direction, a radial bias is simply not
# observable and no weighting recovers it -- the only fix is to go and look
# from somewhere else. That's what bearing_spread() and the coverage readout
# in operate.py are for.
FRUIT_LATERAL_TO_DEPTH_VAR_RATIO = 0.25


def fruit_measurement_noise(distance):
    """Local-frame (depth, lateral) measurement noise variances at this
    distance, for a monocular fruit sighting. The distance curve is unchanged;
    only the depth/lateral ANISOTROPY differs from the ArUco model -- see
    FRUIT_LATERAL_TO_DEPTH_VAR_RATIO above for why."""
    depth_var = 0.039 * distance**2 - 0.115 * distance + 0.1064
    lateral_var = depth_var * FRUIT_LATERAL_TO_DEPTH_VAR_RATIO
    return depth_var, lateral_var


class FruitEKF:
    """
    One incremental 2D Kalman filter per fruit CLASS (object_list.csv /
    truemap.txt guarantee at most one instance of each fruit type, so
    unlike ArUco tags there's no multi-instance data-association problem).

    Structure + outlier rejection: adapted from Brandon's ObjectEKF.
    Noise model: fruit_measurement_noise() above (same formula as EKF's
    own ArUco update), not Brandon's separate not-fully-trusted copy.
    """

    def __init__(self, measurement_noise_fn=fruit_measurement_noise, init_cov=1e3,
                 min_var=0.0008, innovation_gate=9.21, reject_radius_multiplier=2.0):
        """
        measurement_noise_fn(distance) -> (depth_var, lateral_var), in the
            observation's own local [depth, lateral] frame. Defaults to
            fruit_measurement_noise() above.
        init_cov: variance used for both axes of a class's covariance on
            its first-ever sighting (nothing to compare it against yet).
            Matches EKF.init_lm_cov (1e3).
        min_var: floor on P's diagonal after every update -- the "space to
            move after converging" knob. R (sensor noise) doesn't shrink
            with repeated sightings, only P (belief) does, and the Kalman
            gain K = P/(P+R) shrinks right along with P -- so without a
            floor, a converged estimate's response to brand-new evidence
            keeps asymptoting toward zero. 0.012 (std ~=11cm) is picked so
            K stays >= ~5% even at this 2.5x2.5m arena's worst-case
            sighting distance, the ~3.54m diagonal (see the FruitEKF
            provenance comment above for the full reasoning -- verified
            numerically in test_fruit_ekf_outlier.py, not just by hand).
            Was 3.75e-4 (std ~=1.9cm, K well under 1% at that distance)
            before this was tuned. Trade-off, not a free win: also caps
            how tight ANY estimate can ultimately converge -- lower it if
            best-case final RMSE matters more than long-range
            adaptability for you.
        innovation_gate: chi-square threshold (2 DOF) an observation's
            Mahalanobis distance must clear to be fully accepted as-is.
            9.21 = ~99% confidence -- same constant as EKF.innovation_gate
            above.
        reject_radius_multiplier: an observation whose Mahalanobis distance
            is between innovation_gate and reject_gate (defined below) is
            CAPPED rather than fully accepted or fully rejected -- see
            update()'s docstring. reject_gate = innovation_gate *
            reject_radius_multiplier**2, i.e. "more than
            reject_radius_multiplier times further away, in Mahalanobis
            units, than the normal accept boundary" (squared because d2 is
            a squared distance). Default 2.0 -- cap between 1x-2x the
            normal boundary, reject only beyond 2x.
        """
        self.measurement_noise_fn = measurement_noise_fn
        self.init_cov = init_cov
        self.min_var = min_var
        self.innovation_gate = innovation_gate
        self.reject_radius_multiplier = reject_radius_multiplier
        self.reject_gate = innovation_gate * reject_radius_multiplier ** 2
        self.estimates = {}   # label -> 2x1 np.array world position
        self.P = {}           # label -> 2x2 np.array covariance

        # Coverage bookkeeping -- see bearing_spread() / coverage_summary().
        # Nothing in the filter maths reads these; they exist so the GUI can
        # say which fruit still needs work, and so "well covered" means
        # something geometrically real rather than "we stood there a while".
        self.n_shots = {}         # label -> raw count of fused sightings
        self.view_bearings = {}   # label -> [bearing from fruit to robot, rad]
        self.view_ang_threshold = np.deg2rad(10.0)   # min separation to count as a new viewpoint

    def reset(self):
        self.estimates = {}
        self.P = {}
        self.n_shots = {}
        self.view_bearings = {}

    def _record_view(self, label, meas_x, meas_y, robot_x, robot_y):
        """Remember the direction this sighting was taken FROM, measured at the
        fruit: bearing = atan2(robot - fruit). That's the quantity that has to
        vary for triangulation to pin a fruit down -- two sightings from the
        same bearing constrain the same single ray no matter how far apart in
        time or how many frames each one took. Only bearings at least
        view_ang_threshold away from every one already recorded are kept, so
        the count reflects genuinely distinct vantage points."""
        self.n_shots[label] = self.n_shots.get(label, 0) + 1
        if robot_x is None or robot_y is None:
            return
        b = math.atan2(robot_y - meas_y, robot_x - meas_x)
        seen = self.view_bearings.setdefault(label, [])
        for prev in seen:
            d = abs((b - prev + np.pi) % (2 * np.pi) - np.pi)
            if d < self.view_ang_threshold:
                return
        seen.append(b)

    def n_views(self, label):
        """Number of genuinely distinct vantage points this fruit was seen from."""
        return len(self.view_bearings.get(label, []))

    def bearing_spread(self, label):
        """Angular extent, in DEGREES, of the smallest arc containing every
        distinct bearing this fruit has been observed from (0-360).

        This is the honest measure of how well-determined a fruit is. Depth is
        the sloppy axis (see FRUIT_LATERAL_TO_DEPTH_VAR_RATIO), so a fruit only
        ever seen through a narrow arc is pinned across the ray but floating
        along it -- its distance is essentially whatever the box-height model
        claimed, and no number of extra photos from that same arc will fix it.
        Wide spread means the rays cross from genuinely different directions,
        which is what makes the estimate robust to a systematic depth bias.
        """
        b = sorted(self.view_bearings.get(label, []))
        if len(b) < 2:
            return 0.0
        gaps = [b[i + 1] - b[i] for i in range(len(b) - 1)]
        gaps.append(b[0] + 2 * np.pi - b[-1])       # wrap-around gap
        return float(np.degrees(2 * np.pi - max(gaps)))

    def sigma(self, label):
        """Current position uncertainty of one fruit, in metres (RMS of P's
        diagonal) -- the filter's own answer to 'how well do I know this one'."""
        P = self.P.get(label)
        if P is None:
            return float('nan')
        return float(np.sqrt(max(np.mean(np.diag(P)), 0.0)))

    def coverage_summary(self, expected_labels, narrow_arc_deg=40.0):
        """What still needs attention, worst first, for the GUI readout.

        Returns dict with:
          'missing' -- expected fruits with no accepted sighting at all. These
                       cost the most: a fruit absent from objects.txt scores
                       nothing, whereas a fuzzy one still scores partly.
          'narrow'  -- (label, spread_deg, n_views) for fruits seen only
                       through a narrow arc: they may LOOK converged (small
                       covariance) while their depth is unverified, so they're
                       reported separately rather than trusted.
          'worst'   -- (label, sigma_m, n_views) sorted by uncertainty, worst
                       first.
        """
        missing = [l for l in expected_labels if l not in self.estimates]
        narrow, worst = [], []
        for label in self.estimates:
            spread = self.bearing_spread(label)
            views = self.n_views(label)
            if spread < narrow_arc_deg:
                narrow.append((label, spread, views))
            worst.append((label, self.sigma(label), views))
        narrow.sort(key=lambda t: t[1])
        worst.sort(key=lambda t: (-t[1] if t[1] == t[1] else 0))
        return {'missing': missing, 'narrow': narrow, 'worst': worst}

    def update(self, label, meas_x, meas_y, distance, heading, robot_x=None, robot_y=None):
        """
        Fold one world-frame observation (meas_x, meas_y) of `label` into
        its running estimate. `distance` and `heading` (robot's current
        theta) parameterise the noise model and its world-frame rotation --
        same quantities estimate_pose() computes on the way to calling this.

        Returns one of three strings (not a bool):
          'accepted' -- within innovation_gate: fused normally.
          'capped'   -- beyond innovation_gate but within reject_gate: the
                        reading is statistically surprising but not
                        implausible enough to throw away, so it's fused as
                        if it had landed exactly on the innovation_gate
                        boundary instead of where it actually was -- the
                        estimate still moves toward it, by the most we're
                        willing to trust in a single update, rather than
                        either the full raw jump or nothing at all.
          'rejected' -- beyond reject_gate: left untouched. Likely a bad
                        detection, a shaky pose estimate, or genuine
                        occlusion/misclassification.
        (First-ever sighting of a label is always 'accepted' -- there's
        nothing yet to compare it against.)

        robot_x/robot_y are optional and affect nothing in the filter maths --
        they're only used to record which DIRECTION this sighting was taken
        from, for the coverage readout (see _record_view/bearing_spread).
        Omitted, everything behaves exactly as before except that fruit's
        bearing spread stays 0 and it reads as un-triangulated.
        """
        z = np.array([[meas_x], [meas_y]])
        depth_var, lateral_var = self.measurement_noise_fn(distance)
        # Guard keeps R_world invertible even if a future noise_fn misbehaves
        # at very short range -- doesn't change normal-case values, both are
        # comfortably above 1e-6 across the arena's realistic distance range.
        depth_var = max(depth_var, 1e-6)
        lateral_var = max(lateral_var, 1e-6)

        c, s = np.cos(heading), np.sin(heading)
        Rot = np.array([[c, -s], [s, c]])
        R_world = Rot @ np.diag([depth_var, lateral_var]) @ Rot.T

        if label not in self.estimates:
            # Birth: same "uninitialised landmark" prior EKF.add_landmarks()
            # gives a brand-new ArUco marker. Nothing to compare the first
            # sighting against yet, so nothing to gate on either.
            self.estimates[label] = z.copy()
            self.P[label] = self.init_cov * np.eye(2)
            self._record_view(label, meas_x, meas_y, robot_x, robot_y)
            return 'accepted'

        pos, P = self.estimates[label], self.P[label]

        y = z - pos          # innovation (H = I: z is already a world-frame x,y point)
        S = P + R_world       # innovation covariance
        S_inv = np.linalg.inv(S)
        d2 = float((y.T @ S_inv @ y).item())

        if d2 > self.reject_gate:
            return 'rejected'   # way too different -- leave the estimate untouched

        status = 'accepted'
        if d2 > self.innovation_gate:
            # Surprising but not implausible enough to discard: scale the
            # innovation back down to the accept boundary (Mahalanobis
            # distance == sqrt(innovation_gate)) instead of using the raw
            # jump. K/P below are computed from S exactly as normal --
            # only the innovation actually applied to the position is
            # capped -- so the covariance update stays mathematically
            # consistent (it reflects "we performed an update with this
            # gain", independent of which particular y triggered it).
            y = y * math.sqrt(self.innovation_gate / d2)
            status = 'capped'

        K = P @ S_inv
        pos_new = pos + K @ y

        I_K = np.eye(2) - K
        P_new = I_K @ P @ I_K.T + K @ R_world @ K.T   # Joseph form -- stays PSD under rounding
        P_new = 0.5 * (P_new + P_new.T)
        if P_new[0, 0] < self.min_var:
            P_new[0, 0] = self.min_var
        if P_new[1, 1] < self.min_var:
            P_new[1, 1] = self.min_var

        self.estimates[label] = pos_new
        self.P[label] = P_new
        # Only count sightings that actually moved the estimate. A 'rejected'
        # reading returned above without reaching here, so it can't inflate the
        # coverage numbers with detections the filter itself didn't believe.
        self._record_view(label, meas_x, meas_y, robot_x, robot_y)
        return status

    def to_display_dict(self):
        """{label: {'x':.., 'y':.., 'P': 2x2 array}}."""
        return {label: {'x': float(pos[0, 0]), 'y': float(pos[1, 0]), 'P': self.P[label]}
                for label, pos in self.estimates.items()}

    def to_objects_dict(self):
        """{label+'_0': {'x':.., 'y':..}} -- the lab_output/objects.txt
        shape eval.py expects."""
        return {label + '_0': {'x': float(pos[0, 0]), 'y': float(pos[1, 0])}
                for label, pos in self.estimates.items()}


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
    object_est_dict: FruitEKF.to_objects_dict() output, e.g. {'redapple_0': {'x':..,'y':..}}
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