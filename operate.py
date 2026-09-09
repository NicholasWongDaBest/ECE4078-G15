import cv2
import json
import time
import shutil
import argparse
import os, sys
import numpy as np
import pygame # python package for GUI
from botconnect import BotConnect # access the robot communication

# import SLAM components (M1)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import (DriveMeasurement, EKF, FruitEKF, in_frame_fraction,
                       is_box_shape_plausible, resolve_confusable_class,
                       load_object_ground_truth, compute_object_rmse,
                       IN_FRAME_FRACTION_THRESHOLD, FRAME_WIDTH, FRAME_HEIGHT,
                       CONFUSABLE_DISTANCE_THRESHOLD)
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M2)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import ObjectDetector

import csv
from object_pose_est import estimate_pose

# FRAME_WIDTH/FRAME_HEIGHT now live in slam/ekf.py (see the comment there)
# so the live GUI pipeline here and the offline object_pose_est.py
# re-derivation pipeline can't drift apart on what the real camera frame
# size is -- imported above, not redefined here.


def load_true_map(fname):
    """
    Load ground-truth ArUco marker positions from a truemap.txt-style JSON
    file (same format eval.py's --truemap expects).
    Returns a dict {tag:int -> np.array([[x],[y]])}, or None if the file is
    missing/unreadable.
    """
    if not os.path.exists(fname):
        return None
    try:
        with open(fname, 'r') as f:
            gt_dict = json.load(f)
    except Exception as e:
        print(f"Could not parse true map '{fname}': {e}")
        return None

    true_map = {}
    for key in gt_dict:
        if key.startswith('aruco'):
            tag = int(key.split('_')[0].replace('aruco', ''))
            true_map[tag] = np.array([[gt_dict[key]['x']], [gt_dict[key]['y']]])
    return true_map if true_map else None


class Operate:
    # Number row -> ArUco tag id, for selecting/deleting a marker from the map.
    MARKER_DELETE_KEYS = {
        pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4, pygame.K_5: 5,
        pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9, pygame.K_0: 10,
    }

    # Arrow key -> [left, right] wheel speed for continuous driving. A tap
    # pulse (see tap_pulse_duration) uses these same magnitudes so a hold
    # that outlasts the pulse hands off to continuous driving at the same speed.
    ARROW_KEY_SPEEDS = {
        pygame.K_UP: [0.4, 0.4],
        pygame.K_DOWN: [-0.4, -0.4],
        pygame.K_LEFT: [-0.35, 0.35],
        pygame.K_RIGHT: [0.35, -0.35],
    }

    def __init__(self, args):

        # Initialise robot controller object
        self.botconnect = BotConnect(args.ip)
        self.command = {'wheel_speed':[0, 0], # left wheel speed, right wheel speed
                        'save_slam': False,
                        'run_obj_detector': False,
                        'save_obj_detector': False,
                        'save_image': False,
                        'load_true_map': False} # M2

        # PID gains -- fixed at startup (no more live keyboard tuning; see
        # botconnect.set_pid()'s own docstring/BotConnect for what these do
        # on the robot side).
        self.botconnect.set_pid(use_pid=1, kp=2, ki=0.04, kd=0.29)

        # Create a folder "lab_output" that stores the results of the lab
        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)

        # Initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06) # size of the ARUCO markers (6cm)

        # Persisted SLAM map: survives program restarts unless 'r','r' is pressed
        self.slam_state_fname = os.path.join(self.lab_output_dir, 'slam_state.json')
        self.ekf.load_state(self.slam_state_fname)  # no-op if the file doesn't exist

        # Ground-truth map for live RMSE tracking (optional -- practice tool only,
        # eval.py against the real truemap.txt is still what's graded)
        self.true_map = load_true_map(args.truemap)
        if self.true_map is not None:
            print(f"Loaded true map with {len(self.true_map)} markers for live RMSE tracking.")
        else:
            print(f"No true map found at '{args.truemap}' -- live RMSE tracking disabled.")
        self.true_map_fname = "truemap.txt" # M2

        # Object ground truth + accumulators for live object RMSE tracking
        self.true_map_objects = load_object_ground_truth(args.truemap)
        if self.true_map_objects is not None:
            print(f"Loaded {len(self.true_map_objects)} ground-truth object positions for live RMSE tracking.")

        self.object_dimensions = {}
        self.object_list = []
        obj_csv = 'object_list.csv'   # <-- CONFIRM THIS PATH, see note below
        if os.path.exists(obj_csv):
            with open(obj_csv, 'r') as f:
                for row in csv.DictReader(f):
                    self.object_list.append(row['object'])
                    self.object_dimensions[row['object']] = [
                        float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])
                    ]
        else:
            print(f"Object list not found at '{obj_csv}' -- live object RMSE disabled.")

        # M2: FruitEKF -- per-class incremental Kalman filter with chi-square
        # outlier rejection (adapted from Brandon's ObjectEKF) using our own
        # noise model (see the FruitEKF comment block in slam/ekf.py for why).
        # init_cov matches self.ekf's own landmark-birth prior; min_var and
        # measurement_noise_fn are left at FruitEKF's own defaults since EKF
        # itself (the ArUco filter, same file) has no equivalent to read them from.
        self.fruit_ekf = FruitEKF(init_cov=self.ekf.init_lm_cov)
        self.live_object_rmse_info = None
        self.live_object_estimates = None

        self.obj_rmse_log_fname = os.path.join(self.lab_output_dir, 'object_rmse_log.csv')
        with open(self.obj_rmse_log_fname, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['shot_id', 'object', 'robot_x', 'robot_y', 'robot_theta_deg', 'est_x', 'est_y', 'raw_error_m']
            )
        self.obj_shot_id = 0

        # M2: auto-capture fruit pictures repeatedly while the robot is
        # stationary, instead of requiring manual 'p' then 'n' -- see
        # auto_capture_fruit().
        self.is_stationary = True     # updated every tick in perform_slam(); starts at rest
        self.stationary_since = time.time()   # time.time() of the most recent False->True
                                                # transition of is_stationary -- see perform_slam()
        self.capture_settle_time = 0.3   # s the robot must have been (commanded) stationary before
                                           # a frame is trusted not to be blurred by residual motion
                                           # or camera pipeline latency -- see auto_capture_fruit(); tune this
        self.auto_capture_enabled = True   # toggle with 'a'
        self.auto_capture_interval = 1.0 / 3.0   # s between auto-captures while stationary (~3/sec) -- tune this
        self.last_auto_capture_time = 0.0        # time.time() of the last auto-capture

        # Per-pose capture cap. Auto-capture fires ~3/sec for as long as you
        # sit still, but FruitEKF has no viewpoint-novelty penalty (unlike the
        # ArUco filter), so 30 frames from one parking spot shrink a fruit's
        # covariance as if they were 30 independent measurements when they all
        # share the same viewing angle and therefore the same systematic error.
        # That's how a fruit ends up looking confidently converged in the wrong
        # place. Cap how many shots one pose can contribute; moving far enough
        # (either threshold below) starts a fresh budget.
        self.max_captures_per_pose = 5
        self.capture_pose_pos_threshold = 0.15              # m of travel to re-arm
        self.capture_pose_ang_threshold = np.deg2rad(15.0)  # or this much rotation
        self.captures_at_pose = 0
        self.capture_burst_pose = None    # (x, y, theta) the current burst started from
        self.last_marker_count = 0   # number of known (tag 1-10) ArUco markers seen in the most
                                       # recent frame -- set in perform_slam(), read by
                                       # auto_capture_fruit() so it can skip a marker-free frame

        # Initialise CV detector
        if args.yolo_path == "":
            self.obj_detector = None
            self.cv_vis = cv2.imread('ui/8bit/detector_splash.png')
        else:
            self.obj_detector = ObjectDetector(args.yolo_path)
            self.cv_vis = np.ones((480,640,3))* 100

        # Create a folder to save raw camera images after pressing "i"
        self.raw_img_dir = 'raw_images/'
        if not os.path.exists(self.raw_img_dir):
            os.makedirs(self.raw_img_dir)
        else:
            # Delete the folder and create an empty one, i.e. every time operate.py is run, this folder will be empty.
            shutil.rmtree(self.raw_img_dir)
            os.makedirs(self.raw_img_dir)

        #Straight line sync correction
        self.sync_kp = 0.0005   # tune this — start small and increase
        self.sync_ki = 0.0
        self.sync_error_integral = 0.0
        self.prev_left_count = 0
        self.prev_right_count = 0
        self.base_wheel_speed = [0.0, 0.0]  # intended speed, before correction

        # Other auxiliary objects/variables
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.pending_delete_tag = None  # marker tag awaiting a second keypress to confirm deletion
        self.image_id = 0
        self.show_live_rmse = True   # toggle with 'L'
        if self.ekf.number_landmarks() > 0:
            self.notification = f'Restored {self.ekf.number_landmarks()} landmark(s) - view markers & press ENTER to relocalise'
        else:
            self.notification = 'Press ENTER to start SLAM'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.bg = pygame.image.load('ui/gui_mask.jpg')

        self.prev_left_count_ekf = 0
        self.prev_right_count_ekf = 0

        # Startup ramp (avoids wheel slip from an instant full-power command)
        self.ramp_active = False
        self.ramp_start_time = 0.0
        self.ramp_duration = 0.15      # seconds to reach full commanded speed, tune this
        self.prev_base_wheel_speed = [0.0, 0.0]

        self.prev_correct_time = time.time()
        self.sync_kp_rate = 0.00005   # NEW tunable -- replaces sync_kp, needs retuning (see below)

        # Tap-vs-hold arrow key driving. A quick tap fires ONE bounded pulse
        # (move_auto_time, fixed duration) so a tap always covers the same
        # small, consistent distance regardless of how long the key was
        # actually held down -- real hold-duration is too jittery (pygame
        # event timing, network latency) to size a "small increment" on.
        # Holding past the pulse's duration hands off to the existing
        # continuous ramp-drive below, at the same speed, so it feels seamless.
        self.tap_pulse_duration = 0.01   # s -- tune on hardware: shortest reliable single nudge
        self.pulse_active_until = 0.0    # monotonic deadline; correct_straight_drive() stays
                                          # quiet until this passes so it doesn't stomp the pulse
        self.key_press_time = {}         # arrow key -> time.time() at KEYDOWN
        self.continuous_keys = set()     # arrow keys promoted to continuous driving

        # Distortion correction (optional). Rename/move distortion_correction.json
        # away to disable it and test whether it's the source of a problem --
        # if the file is missing, marker positions pass through unchanged.
        self.dist_correction_enabled = False
        dist_correction_path = 'distortion_correction.json'
        if os.path.exists(dist_correction_path):
            try:
                with open(dist_correction_path) as f:
                    dc = json.load(f)
                self.dist_degree = dc['degree']
                self.dist_coeffs_x = np.array(dc['coeffs_x'])
                self.dist_coeffs_y = np.array(dc['coeffs_y'])
                self.dist_correction_enabled = True
                print(f"Loaded distortion correction (degree {self.dist_degree}) from {dist_correction_path}")
            except Exception as e:
                print(f"Failed to load {dist_correction_path}: {e} -- distortion correction disabled")
        else:
            print(f"No {dist_correction_path} found -- distortion correction disabled (raw positions used)")

    # update control parameters for ekf
    def control(self):
        dt = time.time() - self.control_clock
        left, right = self.botconnect.get_encoder_counts()
        delta_left = left - self.prev_left_count_ekf
        delta_right = right - self.prev_right_count_ekf
        # Guard against encoder counter reset on reconnect (large negative jump)
        if delta_left < -10 or delta_right < -10:
            delta_left, delta_right = 0, 0

        self.prev_left_count_ekf, self.prev_right_count_ekf = left, right

        drive_measurement = DriveMeasurement(
            self.botconnect.left_speed, self.botconnect.right_speed, dt,
            delta_left_ticks=delta_left, delta_right_ticks=delta_right
        )
        self.control_clock = time.time()
        return drive_measurement

    # camera control
    def take_pic(self):
        self.img = self.botconnect.get_image() # self.img will be RGB

    # wheel and camera calibration for SLAM
    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        self.obj_focal_length = camera_matrix[0][0]
        self.obj_cx = camera_matrix[0][2]
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=172.5) ##change this value
        return EKF(robot)

    def apply_distortion_correction(self, x, y):
            if not self.dist_correction_enabled:
                return x, y
            feats = [1, x, y]
            if self.dist_degree >= 2:
                feats += [x**2, x*y, y**2]
            if self.dist_degree >= 3:
                feats += [x**3, x**2*y, x*y**2, y**3]
            feats = np.array(feats)
            return float(feats @ self.dist_coeffs_x), float(feats @ self.dist_coeffs_y)


    # SLAM with ARUCO markers
    def perform_slam(self, drive_measurement):
        sensor_measurement, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)

        for lm in sensor_measurement:
            x_c, y_c = self.apply_distortion_correction(lm.position[0,0], lm.position[1,0])
            lm.position[0,0], lm.position[1,0] = x_c, y_c

        # Discard any detected tag outside our known marker set (1-10).
        # DICT_4X4_100 can detect tags 0-99, so a stray/misread marker would
        # otherwise get added as a landmark and show up as "?" on the map.
        sensor_measurement = [lm for lm in sensor_measurement if 1 <= lm.tag <= 10]

        # M2: remember how many known markers were actually visible in THIS
        # frame (self.img, the same frame auto_capture_fruit() would use) --
        # see auto_capture_fruit()'s marker guard.
        self.last_marker_count = len(sensor_measurement)

        # M2: computed unconditionally every tick (not just while ekf_on) so
        # auto_capture_fruit() can also read it. Same test as before, just
        # pulled out of the elif branch below so it's available either way.
        v_l = drive_measurement.left_speed
        v_r = drive_measurement.right_speed
        was_stationary = self.is_stationary
        self.is_stationary = abs(v_l) < 1e-3 and abs(v_r) < 1e-3
        if self.is_stationary and not was_stationary:
            # Just transitioned from moving to commanded-stopped. "Stationary"
            # here only means the last commanded wheel speed is ~0 -- the
            # physical robot (motor/wheel inertia) and the camera feed
            # (self.img comes from BotConnect's background camera thread over
            # a socket -- see take_pic()/BotConnect.get_image()) both lag
            # behind that command by some real time. auto_capture_fruit()
            # waits out capture_settle_time from this timestamp before
            # trusting a frame, so a picture isn't taken the instant this
            # flips True while the robot/frame may still reflect motion.
            self.stationary_since = time.time()

        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(sensor_measurement)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:
            if not self.is_stationary: #prevent predict to run when bot is not moving (prevent uncertainty to be added)
                self.ekf.predict(drive_measurement)
            self.ekf.add_landmarks(sensor_measurement)
            self.ekf.update(sensor_measurement, drive_measurement)  # pass drive_measurement so update() knows how much rotation is happening right now

    def save_result(self):
        # save slam map after pressing "s"
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False

        # load the true/ground-truth map and freeze it "l" (M2)
        if self.command['load_true_map']:
            if os.path.exists(self.true_map_fname):
                n_lm = self.ekf.load_true_map(self.true_map_fname)
                self.notification = f'Loaded true map ({n_lm} landmarks), frozen - press ENTER to relocalise'
            else:
                self.notification = f'True map file not found: {self.true_map_fname}'
            self.command['load_true_map'] = False

        # save obj_detector result with the matching robot pose and detector labels
        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:
                self.pred_fname = self.obj_detector.write_output(*self.obj_detector_output, self.lab_output_dir)
                self.notification = f'Prediction is saved to {self.pred_fname}'
                self.process_object_estimates()
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_obj_detector'] = False

        # save raw images taken by the camera after pressing "i"
        if self.command['save_image']:
            image = self.botconnect.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            f_ = os.path.join(self.raw_img_dir, f'img_{self.image_id}.png')
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # M2: shared by the manual ('p') and automatic (auto_capture_fruit) paths
    # so both stash obj_detector_output identically.
    def _run_object_detector(self):
        bboxes, self.cv_vis = self.obj_detector.detect_single_image(self.img)
        self.obj_detector_output = (self.cv_vis, self.ekf.robot.state.tolist(), bboxes) # three things to be saved
        return len(set([box[0] for box in bboxes]))

    # using computer vision to detect objects
    def detect_object(self):
        if self.command['run_obj_detector'] and self.obj_detector is not None:
            self.command['run_obj_detector'] = False
            unique_detected = self._run_object_detector()
            self.notification = f'{unique_detected} object type(s) detected'

    # M2: automatically capture + fuse fruit shots, repeatedly, for as long
    # as the robot stays stationary -- instead of requiring manual 'p' then
    # 'n' every time.
    def auto_capture_fruit(self):
        """
        Runs up to once every auto_capture_interval seconds (~3/sec by
        default) for as long as is_stationary stays True (computed each
        tick in perform_slam() from the same test that gates ekf.predict()),
        not edge-triggered once per stop -- so it keeps sampling the whole
        time the robot is parked at a vantage point, not just the instant
        it arrives. The first capture_settle_time seconds after arriving are
        skipped (see the guard below) so the first frame(s) aren't still
        blurred by residual motion or camera-pipeline lag from just before
        the stop.

        Note: unlike ArUco landmarks, FruitEKF has no viewpoint-novelty
        penalty for a repeat look from the same spot (see slam/ekf.py's
        viewpoint_novelty(), which only applies to the ArUco EKF) -- so
        sampling this fast will shrink a fruit's covariance faster than
        that many genuinely independent sightings would justify. That's an
        accepted trade-off of sampling this often, not a bug; the
        chi-square gate in FruitEKF.update() still rejects any individual
        reading that's inconsistent with the running estimate.

        Runs the full pipeline a manual 'p' + 'n' would: detect, write to
        pred.txt (so the offline object_pose_est.py/eval.py path still sees
        it), then fuse into fruit_ekf via process_object_estimates() -- same
        per-box in_frame_fraction / is_box_shape_plausible gates apply
        either way.
        Only runs while SLAM is on and a model is loaded; toggle off with
        'a' to fall back to fully manual control.
        """
        if not (self.is_stationary and self.auto_capture_enabled and self.ekf_on
                and self.obj_detector is not None):
            return

        # Settle guard: is_stationary going True only means the last commanded
        # wheel speed hit ~0 this tick -- it says nothing about whether the
        # physical robot has actually finished decelerating, or whether the
        # camera frame we'd capture has caught up past that motion (camera
        # pipeline latency -- see perform_slam()'s comment on
        # stationary_since). Skip capturing until we've been stationary for
        # at least capture_settle_time, so the first frame(s) right after a
        # stop aren't still motion-blurred.
        if time.time() - self.stationary_since < self.capture_settle_time:
            return

        # Marker guard: skip this frame if no known ArUco marker (tag 1-10)
        # is currently visible (last_marker_count set in perform_slam() from
        # the same detection run on this same self.img). Requested so a fruit
        # shot only gets taken/fused while there's a landmark in view.
        if self.last_marker_count <= 1:
            return

        # Per-pose cap: extra frames from a pose we've already sampled buy no
        # new geometry, they just make the filter over-confident (see
        # max_captures_per_pose in __init__). Reset the budget once the robot
        # has actually moved to a new vantage point.
        pose = (float(self.ekf.robot.state[0, 0]),
                float(self.ekf.robot.state[1, 0]),
                float(self.ekf.robot.state[2, 0]))
        if self.capture_burst_pose is None:
            self.capture_burst_pose = pose
        else:
            bx, by, bth = self.capture_burst_pose
            moved = np.hypot(pose[0] - bx, pose[1] - by)
            turned = abs((pose[2] - bth + np.pi) % (2 * np.pi) - np.pi)
            if (moved >= self.capture_pose_pos_threshold
                    or turned >= self.capture_pose_ang_threshold):
                self.capture_burst_pose = pose
                self.captures_at_pose = 0
        if self.captures_at_pose >= self.max_captures_per_pose:
            return

        now = time.time()
        if now - self.last_auto_capture_time < self.auto_capture_interval:
            return
        self.last_auto_capture_time = now
        self.captures_at_pose += 1

        unique_detected = self._run_object_detector()
        self.pred_fname = self.obj_detector.write_output(*self.obj_detector_output, self.lab_output_dir)
        self.notification = f'[Auto] {unique_detected} object type(s) saved to {self.pred_fname}'
        self.process_object_estimates()

    def fruit_coverage_line(self):
        """One line of "where should I drive next", plus a colour for it.

        Priority order is by what each case actually costs you. A fruit with no
        sighting at all contributes nothing to objects.txt, so it's worth more
        than any amount of polish on one already found. Next is a fruit seen
        only through a narrow arc of bearings: depth is the sloppy axis of a
        monocular estimate, so those look converged (small covariance) while
        their distance is really just whatever the box-height model said --
        the fix is a sighting from a different DIRECTION, not more photos from
        here. Only after those does plain uncertainty rank.

        Counts shown are distinct viewpoints, not frames, for the same reason
        the per-pose cap exists: frames from one spot aren't independent looks.
        """
        if not self.object_list:
            return "", (220, 220, 220)

        cov = self.fruit_ekf.coverage_summary(self.object_list)

        if cov['missing']:
            return ("NOT SEEN: " + ", ".join(cov['missing'][:4]), (240, 90, 90))

        if cov['narrow']:
            label, spread, views = cov['narrow'][0]
            more = f" +{len(cov['narrow'])-1}" if len(cov['narrow']) > 1 else ""
            return (f"NARROW ARC: {label} {spread:.0f}deg ({views} views){more}"
                    f" - view from another side", (240, 180, 60))

        if cov['worst']:
            label, sigma, views = cov['worst'][0]
            return (f"worst: {label} +/-{sigma*1000:.0f}mm ({views} views)", (120, 220, 120))

        return "No fruit fused yet", (220, 220, 220)

    # paint the GUI
    def draw(self, canvas):
        canvas.fill((0, 0, 0))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        # compute live RMSE only if a true map is loaded AND tracking is enabled
        active_true_map = self.true_map if (self.true_map is not None and self.show_live_rmse) else None
        live_rmse_info = self.ekf.compute_live_rmse(active_true_map) if active_true_map is not None else None

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause=self.ekf_on,
                                    true_map=active_true_map, live_rmse_info=live_rmse_info,
                                    selected_tag=self.pending_delete_tag,
                                    object_gt=self.true_map_objects if self.show_live_rmse else None,
                                    object_estimates=self.live_object_estimates,
                                    object_rmse_info=self.live_object_rmse_info,
                                    object_ekf=self.fruit_ekf)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for object detector
        detector_view = cv2.resize(self.cv_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240+2*v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='Robot Cam', position=(h_pad, v_pad))
        notification = TEXT_FONT.render(self.notification[:55], False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        # live RMSE readout in the main window
        if self.true_map_objects is None:
            obj_rmse_line = "No object ground truth loaded"
        elif not self.show_live_rmse:
            obj_rmse_line = ""
        elif self.live_object_rmse_info is None:
            obj_rmse_line = "Object RMSE: no matched estimates yet"
        else:
            info = self.live_object_rmse_info
            obj_rmse_line = (f"Object RMSE: {info['rmse']:.4f} m "
                            f"({len(info['matched'])}/{len(self.true_map_objects)} objects)")
        obj_rmse_surface = TEXT_FONT.render(obj_rmse_line, False, text_colour)
        canvas.blit(obj_rmse_surface, (h_pad+10, 652))

        # Fruit coverage: what still needs driving to, worst first. Ordered by
        # how much it costs to leave alone -- a fruit never seen scores nothing
        # at all, a fruit seen only through a narrow arc has an unverified
        # depth (see FruitEKF.bearing_spread), and only then plain uncertainty.
        cov_line, cov_colour = self.fruit_coverage_line()
        cov_surface = TEXT_FONT.render(cov_line[:55], False, cov_colour)
        canvas.blit(cov_surface, (h_pad+10, 624))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (h_pad+10, 680))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption, False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # Keyboard teleoperation
    # For pibot motion, set two numbers for the self.command['wheel_speed']. Eg self.command['wheel_speed'] = [0.6, 0.6]
    # These numbers specify how fast to power the left and right wheels
    # The numbers must be between -1 (full speed backward) and 1 (full speed forward). 0 means stop.
    # Study the code in botconnect.py to see the function to call after setting wheel speed
    def update_keyboard(self):
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN and event.key in self.ARROW_KEY_SPEEDS:
                if event.key not in self.key_press_time:  # ignore OS key-repeat re-fires
                    self.key_press_time[event.key] = time.time()
                    self.pulse_active_until = time.time() + self.tap_pulse_duration
                    self.botconnect.move_auto_time(self.ARROW_KEY_SPEEDS[event.key], self.tap_pulse_duration)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.base_wheel_speed = [0.0, 0.0]
            if event.type == pygame.KEYUP and event.key in self.ARROW_KEY_SPEEDS:
                self.key_press_time.pop(event.key, None)
                self.continuous_keys.discard(event.key)
                self.base_wheel_speed = [0.0, 0.0]
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '>2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '>2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['save_slam'] = True
            # load true map M2 (freezes landmark positions, SLAM only localises)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                self.command['load_true_map'] = True
            # show live rmse
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                self.show_live_rmse = not self.show_live_rmse
                state = 'ON' if self.show_live_rmse else 'OFF'
                self.notification = f'Live RMSE tracking {state}'
            # M2: toggle automatic fruit capture-on-stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.auto_capture_enabled = not self.auto_capture_enabled
                state = 'ON' if self.auto_capture_enabled else 'OFF'
                self.notification = f'Auto fruit capture {state}'
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()

                    # clear in-memory fruit/object accumulators
                    self.fruit_ekf.reset()
                    self.live_object_rmse_info = None
                    self.live_object_estimates = None
                    self.obj_shot_id = 0
                    self.captures_at_pose = 0
                    self.capture_burst_pose = None

                    # pred.txt is held open by ObjectDetector for the whole session, so it
                    # must be closed/reopened by the class itself rather than deleted here
                    # -- os.remove() would fail with PermissionError while it's still open.
                    # Brandon-2.0's cv/detector.py has a working ObjectDetector.reset()
                    # that does exactly this; guarded with hasattr() rather than called
                    # unconditionally because a cv/detector.py missing it (as happened
                    # here -- AttributeError on this line took down the whole program,
                    # losing the SLAM/fruit reset that had already succeeded above) should
                    # degrade to "pred.txt wasn't cleared" instead of a hard crash.
                    if self.obj_detector is not None:
                        if hasattr(self.obj_detector, 'reset'):
                            self.obj_detector.reset()
                        else:
                            print("WARNING: ObjectDetector has no reset() method -- "
                                  "pred.txt was NOT cleared, so it may still hold "
                                  "detections from before this reset. Add reset() to "
                                  "ObjectDetector in cv/detector.py (see chat).")

                    # remove persisted SLAM state + remaining output files so a relaunch
                    # or reload doesn't pick up stale data from before this reset
                    for fname in [
                        self.slam_state_fname,
                        os.path.join(self.lab_output_dir, 'slam.txt'),
                        os.path.join(self.lab_output_dir, 'objects.txt'),
                        self.obj_rmse_log_fname,
                    ]:
                        if os.path.exists(fname):
                            os.remove(fname)

                    # re-create a fresh (empty, headered) rmse log so process_object_estimates
                    # can keep appending to it right away
                    with open(self.obj_rmse_log_fname, 'w', newline='') as f:
                        csv.writer(f).writerow(
                            ['shot_id', 'object', 'robot_x', 'robot_y', 'robot_theta_deg', 'est_x', 'est_y', 'raw_error_m']
                        )
            # run object/fruit detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_obj_detector'] = True
            # capture and save raw image
            elif event.type == pygame.KEYDOWN and event.key  == pygame.K_i:
                self.command['save_image'] = True
            # select/delete a marker by its tag number (press once to select, again to delete)
            elif event.type == pygame.KEYDOWN and event.key in self.MARKER_DELETE_KEYS:
                tag = self.MARKER_DELETE_KEYS[event.key]
                if self.pending_delete_tag == tag:
                    if self.ekf.delete_landmark(tag):
                        self.notification = f'Marker {tag} deleted'
                    else:
                        self.notification = f'Marker {tag} not in map'
                    self.pending_delete_tag = None
                else:
                    self.pending_delete_tag = tag
                    if tag in self.ekf.taglist:
                        self.notification = f'Marker {tag} selected - press {tag} again to delete'
                    else:
                        self.notification = f'Marker {tag} not in map - press {tag} again to clear selection'
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

        # A key still held after its tap pulse has run its course gets
        # promoted to continuous driving, at the same speed the pulse used.
        pressed = pygame.key.get_pressed()
        for key, press_time in list(self.key_press_time.items()):
            if (pressed[key] and key not in self.continuous_keys
                    and time.time() - press_time >= self.tap_pulse_duration):
                self.continuous_keys.add(key)
                self.base_wheel_speed = self.ARROW_KEY_SPEEDS[key]

        if self.quit:
            if self.ekf.number_landmarks() > 0:
                self.ekf.save_state(self.slam_state_fname)
            pygame.quit()
            sys.exit()

    def correct_straight_drive(self):
        base_l, base_r = self.base_wheel_speed

        # A tap pulse is a one-shot move_auto_time() call (mode 1) that the Pi
        # runs to completion on its own. base_wheel_speed stays [0,0] the
        # whole time (no continuous drive requested), so without this guard
        # the move_manual([0,0]) call below would immediately flip botconnect
        # back to mode 0 and cancel the pulse before it's even sent.
        if base_l == 0.0 and base_r == 0.0 and time.time() < self.pulse_active_until:
            self.prev_base_wheel_speed = [base_l, base_r]
            return

        # Detect stop -> move transition, start a ramp
        was_stopped = (self.prev_base_wheel_speed == [0.0, 0.0])
        now_moving = (base_l != 0.0 or base_r != 0.0)
        if was_stopped and now_moving and not self.ramp_active:
            self.ramp_active = True
            self.ramp_start_time = time.time()

        if self.ramp_active:
            elapsed = time.time() - self.ramp_start_time
            if elapsed >= self.ramp_duration:
                self.ramp_active = False
                ramp_scale = 1.0
            else:
                ramp_scale = elapsed / self.ramp_duration
        else:
            ramp_scale = 1.0

        # Left/right sync correction is intentionally NOT done here anymore --
        # it's already handled by the Pi's own PID loop (pid_control() in the
        # server script), which runs at a fixed rate on a separate machine and
        # is immune to this PC's variable SLAM/ArUco processing cost. Trying to
        # replicate that correction here, coupled to this loop's timing, was the
        # source of the intermittent swerving.
        adjusted = [base_l * ramp_scale, base_r * ramp_scale]

        self.command['wheel_speed'] = adjusted
        self.botconnect.move_manual(adjusted)
        self.prev_base_wheel_speed = [base_l, base_r]

    def process_object_estimates(self):
        """Runs after every 'n' (save prediction), and automatically from
        auto_capture_fruit() while the robot is stationary. Each
        recognised fruit box in the shot is gated INDEPENDENTLY -- a box
        that fails either check below is just skipped; every other box in
        the same photo still gets processed and fused normally. (This
        used to reject the WHOLE shot if any one box failed either check,
        on the reasoning that every box shares the same robot-pose
        snapshot. That pose is only ever SLAM's -- self.ekf, from
        aruco_sensor -- and neither check below has anything to do with
        it: both are purely about one box's own geometry. A clipped or
        oddly-shaped box for fruit A says nothing about whether fruit B's
        completely separate box, a few pixels away in the same photo, is
        trustworthy -- estimate_pose() computes each fruit's position
        from its own box alone. So there was never a real reason for one
        bad box to also throw out an otherwise-good sighting of a
        different fruit; changed so a photo with one bad box and two good
        ones now keeps the two good ones instead of the whole photo.)

          1. in_frame_fraction() < IN_FRAME_FRACTION_THRESHOLD (95%): a box
             clipped by the photo edge under-reports that fruit's true
             apparent height, which throws off the pinhole depth estimate.
          2. is_box_shape_plausible(): a box whose width/height aspect
             ratio falls outside that class's plausible range (the union
             of Brandon's empirical EXPECTED_ASPECT_RANGE and a geometric
             bound derived from object_list.csv -- see slam/ekf.py) --
             catches occlusion, merged/overlapping detections, or a bad
             read that's fully inside the frame and so invisible to check
             1. Both checks are shape-agnostic vs. shape-aware; neither can
             see what the other is checking for.

        A box that clears both gates above still goes through ONE more
        step before fusion: resolve_confusable_class() (slam/ekf.py) may
        relabel it. This is a correction, not a gate -- it never drops a
        box, only possibly changes which class's EKF track it's fused
        into, for the two one-directional confusions Nicholas identified
        (a real lime sometimes read as 'capsicum', a real orange
        sometimes read as 'mango', never the reverse). See that
        function's own docstring for the full mechanics (why it needs to
        RECOMPUTE the pose with the victim class's true height rather
        than just comparing the box's already-computed position).

        There is no separate ArUco-marker-count / fruit-count gate any
        more. Boxes that clear both checks get fused into self.fruit_ekf
        -- a per-class Kalman filter (see FruitEKF in slam/ekf.py) that
        narrows a fruit's position down over repeated sightings the way
        self.ekf narrows down ArUco landmarks, via its OWN 3-tier
        per-reading gate (FruitEKF.update()'s return value): readings
        close to the running estimate are fused normally ('accepted');
        readings a fair bit off are still fused, but with their pull on
        the estimate capped rather than applied in full ('capped');
        readings way off are left out entirely so they can't corrupt the
        estimate ('rejected'). FruitEKF's min_var also floors how far P
        (its own uncertainty) can shrink, so even a well-converged
        fruit's estimate keeps some real ability to move in response to
        new evidence rather than effectively freezing. self.notification
        ends up a ';'-separated note per box that needed one (skipped /
        capped / rejected) -- a box that just fused cleanly doesn't add
        anything, so a fully clean shot leaves whatever notification was
        already showing (e.g. auto_capture_fruit()'s "[Auto] ... saved
        to ..." message) alone instead of stomping on it. Also logs the
        raw per-shot pose estimate for every box that fused, capped, or
        was chi-square-rejected (but NOT one skipped by checks 1/2 above
        -- its geometry was never trusted enough to compute a pose from
        in the first place), for angle analysis later, and updates the
        running RMSE against truemap.txt, if loaded."""
        if not self.object_dimensions:
            return

        _, robot_pose, bboxes = self.obj_detector_output
        relevant_boxes = [(predicted_class, box) for predicted_class, box in bboxes
                           if predicted_class in self.object_dimensions]

        robot_theta_deg = np.degrees(robot_pose[2][0])
        robot_x, robot_y = robot_pose[0][0], robot_pose[1][0]
        heading = robot_pose[2][0]

        shot_rows = []
        notes = []   # one note per box that needed attention (skipped / capped / rejected)
        for predicted_class, box in relevant_boxes:
            # M2 per-box gates -- skip just THIS box, keep processing the
            # rest of the shot.
            if in_frame_fraction(box, FRAME_WIDTH, FRAME_HEIGHT) < IN_FRAME_FRACTION_THRESHOLD:
                notes.append(f'{predicted_class} skipped (<{IN_FRAME_FRACTION_THRESHOLD * 100:.0f}% in frame)')
                continue
            if not is_box_shape_plausible(predicted_class, box, self.object_dimensions):
                notes.append(f'{predicted_class} skipped (implausible box shape)')
                continue

            true_height = self.object_dimensions[predicted_class][2]
            pose_x, pose_y = estimate_pose(robot_pose, box, true_height, self.obj_focal_length, self.obj_cx)

            # Class-confusion correction -- may relabel predicted_class to
            # its "victim" class and swap in a pose recomputed with the
            # victim's true height. See resolve_confusable_class()'s
            # docstring in slam/ekf.py. known_positions is built fresh each
            # shot from self.fruit_ekf's CURRENT estimates (cheap -- at
            # most 7 fruit classes), so it reflects every earlier shot's
            # fusions, including relabels from earlier in this same shot.
            known_positions = {label: (pos[0, 0], pos[1, 0])
                                for label, pos in self.fruit_ekf.estimates.items()}
            fuse_class, alt_pose = resolve_confusable_class(
                predicted_class, box, robot_pose, self.object_dimensions,
                self.obj_focal_length, self.obj_cx, estimate_pose, known_positions)
            if alt_pose is not None:
                notes.append(f'{predicted_class} relabelled as {fuse_class} '
                             f'(within {CONFUSABLE_DISTANCE_THRESHOLD * 100:.0f}cm of existing {fuse_class})')
                pose_x, pose_y = alt_pose

            dist = float(np.hypot(pose_x - robot_x, pose_y - robot_y))

            # robot_x/robot_y are for coverage bookkeeping only (which
            # direction this sighting came FROM -- see FruitEKF._record_view);
            # they don't enter the filter maths.
            fusion_status = self.fruit_ekf.update(fuse_class, pose_x, pose_y, dist, heading,
                                                   robot_x=robot_x, robot_y=robot_y)
            if fusion_status == 'rejected':
                notes.append(f'{fuse_class} rejected (inconsistent with existing estimate)')
            elif fusion_status == 'capped':
                notes.append(f'{fuse_class} capped (large jump from current estimate)')

            raw_error = ''
            if self.true_map_objects is not None and fuse_class in self.true_map_objects:
                gt = self.true_map_objects[fuse_class]
                raw_error = float(np.hypot(pose_x - gt[0][0], pose_y - gt[1][0]))

            self.obj_shot_id += 1
            shot_rows.append([self.obj_shot_id, fuse_class, robot_x, robot_y,
                            robot_theta_deg, pose_x, pose_y, raw_error])

        if notes:
            self.notification = '; '.join(notes)

        if shot_rows:
            with open(self.obj_rmse_log_fname, 'a', newline='') as f:
                csv.writer(f).writerows(shot_rows)

        # fruit_ekf.estimates only ever holds classes actually fused, so
        # to_objects_dict() is already the "detected so far" set -- no
        # separate bookkeeping set needed any more.
        if self.true_map_objects is not None:
            self.live_object_estimates = self.fruit_ekf.to_objects_dict()
            self.live_object_rmse_info = compute_object_rmse(self.live_object_estimates, self.true_map_objects)
            if self.live_object_rmse_info:
                info = self.live_object_rmse_info
                print(f"[Live] Object RMSE: {info['rmse']:.4f} m "
                    f"({len(info['matched'])}/{len(self.true_map_objects)} objects)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost') # you can hardcode ip here, but it may change from time to time.
    parser.add_argument("--calib_dir", type=str, default="calibration/param/") # calibration directory
    parser.add_argument("--yolo_path", default='cv/model/yolo26n.pt') # directory for your trained AI model
    parser.add_argument("--truemap", type=str, default='truemap.txt', help="ground-truth map for live RMSE practice tracking (optional)")
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)

    width, height = 900, 760
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab')
    pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('ui/loading.png')
    pibot_animate = [pygame.image.load('ui/8bit/pibot1.png'),
                     pygame.image.load('ui/8bit/pibot2.png'),
                     pygame.image.load('ui/8bit/pibot3.png'),
                    pygame.image.load('ui/8bit/pibot4.png'),
                     pygame.image.load('ui/8bit/pibot5.png')]
    pygame.display.update()

    start = False
    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    width, height = 900, 760
    canvas = pygame.display.set_mode((width, height))
    operate = Operate(args)
    while start:
        operate.update_keyboard()
        operate.correct_straight_drive()
        operate.take_pic()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.auto_capture_fruit()
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()