import cv2
import csv
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
from slam.ekf import DriveMeasurement
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M2)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import ObjectDetector
from object_pose_est import estimate_pose

# Camera frame size, matches self.img's shape (see Operate.__init__) and the
# imgsz=480 passed to model.predict in cv/detector.py.
FRAME_WIDTH = 480
FRAME_HEIGHT = 360


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
                        'save_image': False}

        # TODO: Tune PID parameters here. If you don't want to use PID, set use_pid=0
        # self.botconnect.set_pid(use_pid=1, k  p=0, ki=0, kd=0)

        # PID gains — now adjustable live via keyboard, not fixed at startup
        self.pid_gains = {'kp': 2.3, 'ki': 0.04, 'kd': 0.29}
        self.pid_step = 0.005
        self.botconnect.set_pid(use_pid=1, **self.pid_gains)

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
        # eval.py against the real truemap.txt is still what's graded), and for
        # M2's "load the true map into self.markers and freeze it" mode (press
        # 't', see update_keyboard()). Same file, two different consumers.
        self.truemap_path = args.truemap
        self.true_map = load_true_map(args.truemap)
        if self.true_map is not None:
            print(f"Loaded true map with {len(self.true_map)} markers for live RMSE tracking.")
        else:
            print(f"No true map found at '{args.truemap}' -- live RMSE tracking disabled.")

        # M2: for now, auto-load + freeze the true ArUco map into self.ekf.markers
        # at startup instead of waiting for a manual 't' press. M2 only grades
        # fruit position (lab_output/objects.txt) -- self.ekf's own marker
        # estimates aren't graded this milestone -- so there's no reason to make
        # the robot rediscover marker positions it's already allowed to know.
        # SLAM still runs and localises the robot pose against these markers
        # (recover_from_pause does a landmark resection, then update() keeps
        # refining the pose every frame -- see ekf.py) -- it just never edits
        # the markers themselves. This intentionally overrides whatever
        # load_state() above just restored from a previous session, since the
        # true map is what M2 wants the robot anchored to. 't' toggles this off
        # (unload) and back on (reload) any time -- see K_t below. While the
        # true map is loaded, 'r' no longer touches it at all (self.ekf is left
        # completely alone) and only clears fruit tracking -- see K_r below.
        if os.path.exists(self.truemap_path):
            n = self.ekf.load_true_map(self.truemap_path)
            print(f"Auto-loaded and froze {n} true markers into self.ekf.markers (M2 mode).")
        else:
            print(f"No true map at '{self.truemap_path}' to auto-load -- SLAM will build its own map (press 't' once one exists).")

        # Initialise CV detector
        if args.yolo_path == "":
            self.obj_detector = None
            self.cv_vis = cv2.imread('ui/8bit/detector_splash.png')
        else:
            self.obj_detector = ObjectDetector(args.yolo_path)
            self.cv_vis = np.ones((360,480,3))* 100

        # Live fruit-position estimates, for the SLAM GUI only (practice visual --
        # lab_output/objects.txt from object_pose_est.py, saved separately with
        # "n", is still what's graded). Each detection is fused into a running
        # per-label Kalman estimate (self.fruit_state: label -> {'pos','P'}) via
        # _fuse_fruit_observation(), using EKF.measurement_noise() and the same
        # [depth,lateral]->world rotation, initial uncertainty prior, Joseph-form
        # update and covariance floor self.ekf uses for ArUco landmark births --
        # so a fruit's position AND uncertainty are computed the way a marker's
        # are. This is intentionally independent of self.ekf.markers/taglist/P:
        # fruits are never added to the actual SLAM state, so M1's ArUco-only
        # RMSE grading is unaffected either way. self.fruit_map_display is the
        # drawable/saveable snapshot of self.fruit_state, rebuilt after each
        # detection pass.
        self.object_dimensions = {}
        self.fruit_state = {}        # label -> {'pos': 2x1 np.array, 'P': 2x2 np.array}
        self.fruit_map_display = {}  # label -> {'x': float, 'y': float, 'P': 2x2 np.array}
        if self.obj_detector is not None:
            with open('object_list.csv', 'r') as f:
                for row in csv.DictReader(f):
                    self.object_dimensions[row['object']] = float(row['height(m)'])

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
        # Auto-run the fruit detector in the background, like ArUco runs every
        # frame -- instead of only on "p". YOLO inference is far more expensive
        # than ArUco corner detection, so this is throttled to once every
        # auto_detect_interval seconds rather than every loop iteration; raise
        # the interval if the control loop feels sluggish, lower it if your
        # machine keeps up fine. Manual "p" still always works too, and "n"
        # always saves whichever result (auto or manual) is most recent.
        self.auto_detect_enabled = True   # toggle with 'A'
        self.auto_detect_interval = 0.75  # seconds between automatic YOLO runs -- lower
                                           # further if your machine keeps up, raise it
                                           # (or press 'A') if driving starts feeling laggy
        self.last_auto_detect_time = time.time()
        if self.ekf.freeze_map:
            self.notification = f'True map loaded ({self.ekf.number_landmarks()} markers) - view markers & press ENTER to localise'
        elif self.ekf.number_landmarks() > 0:
            self.notification = f'Restored {self.ekf.number_landmarks()} landmark(s) - view markers & press ENTER to relocalise'
        else:
            self.notification = 'Press ENTER to start SLAM'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([360,480,3], dtype=np.uint8)
        self.aruco_img = np.zeros([360,480,3], dtype=np.uint8)
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

        # Only run the fruit detector once the robot's been still for a bit,
        # so a captured frame isn't blurred by motion. Checking the commanded
        # wheel speed alone isn't enough -- the chassis/camera still has
        # physical momentum and vibration for a moment right after a command
        # of [0,0] is sent, so last_moving_time is refreshed every tick the
        # robot is actually being driven (see correct_straight_drive()) and
        # detect_object() waits until it's been stationary_settle_duration
        # since the last such tick. Tune the duration up if frames still come
        # out blurry, down if the wait feels sluggish.
        self.stationary_settle_duration = 0.3   # s
        self.last_moving_time = time.time()

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

    #real time update pid
    def adjust_pid(self, param, delta):
        self.pid_gains[param] = max(0.0, self.pid_gains[param] + delta)
        success = self.botconnect.set_pid(use_pid=1, **self.pid_gains)
        if success:
            self.notification = (f"PID: kp={self.pid_gains['kp']:.3f} "
                                f"ki={self.pid_gains['ki']:.3f} "
                                f"kd={self.pid_gains['kd']:.3f}")
        else:
            self.notification = 'Failed to update PID on robot'

    # camera control
    def take_pic(self):
        self.img = self.botconnect.get_image() # self.img will be RGB

    # wheel and camera calibration for SLAM
    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=174.5) ##change this value
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
            v_l = drive_measurement.left_speed
            v_r = drive_measurement.right_speed
            if not (abs(v_l) < 1e-3 and abs(v_r) < 1e-3): #prevent predict to run when bot is not moving (prevent uncertainty to be added)
                self.ekf.predict(drive_measurement)
            self.ekf.add_landmarks(sensor_measurement)
            self.ekf.update(sensor_measurement)

    def save_full_map(self, fname):
        """Combine the current SLAM ArUco estimates with the live fruit estimates
        into one truemap.txt-style file (aruco*_0 + fruitname_0 keys, same shape
        as the ground-truth map). Convenience only -- a combined snapshot of
        everything mapped so far, e.g. to try your own M3 navigation code
        against. Not what eval.py reads for grading -- see save_objects_txt()
        for that."""
        d = {}
        for i, tag in enumerate(self.ekf.taglist):
            d[f"aruco{tag}_0"] = {"x": float(self.ekf.markers[0, i]), "y": float(self.ekf.markers[1, i])}
        for name, pos in self.fruit_map_display.items():
            d[f"{name}_0"] = {"x": pos['x'], "y": pos['y']}
        with open(fname, 'w') as f:
            json.dump(d, f, indent=4)

    def save_objects_txt(self, fname):
        """M2 graded output: write self.fruit_map_display (the live per-fruit
        Kalman estimate, continuously updated by _fuse_fruit_observation as
        detections stream in -- see detect_object()) as lab_output/objects.txt,
        in the exact {"<label>_0": {"x":.., "y":..}} shape eval.py's
        parse_map()/eval_object() read. This is what closes the M2 loop: press
        't' (load + freeze the true map into self.ekf.markers), ENTER (SLAM
        localises the robot pose against those known markers via
        recover_from_pause's landmark resection, then keeps refining it),
        drive around with auto-detect on, then 's' -- objects.txt is ready for
        `python eval.py`, no separate step needed.

        The spec's own p/n -> lab_output/pred.txt -> `python object_pose_est.py`
        pipeline (estimate_pose + merge_estimations) still works unchanged and
        remains the more deliberate multi-viewpoint route; if you run it after
        pressing 's' its output will overwrite this one, and vice versa --
        they write the same file, so whichever you run last wins."""
        d = {f"{name}_0": {"x": pos['x'], "y": pos['y']} for name, pos in self.fruit_map_display.items()}
        with open(fname, 'w') as f:
            json.dump(d, f, indent=4)

    def save_result(self):
        # save slam map after pressing "s"
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.save_full_map(os.path.join(self.lab_output_dir, 'full_map.txt'))
            if self.fruit_map_display:
                # Only (re)write objects.txt when there's live fruit data to report --
                # never silently blank out a previously-good objects.txt (e.g. one
                # produced earlier via object_pose_est.py) just because this session
                # hasn't detected any fruit yet.
                self.save_objects_txt(os.path.join(self.lab_output_dir, 'objects.txt'))
                self.notification = 'Map is saved (slam.txt + full_map.txt + objects.txt)'
            else:
                self.notification = 'Map is saved (slam.txt + full_map.txt) - no fruits tracked yet'
            self.command['save_slam'] = False

        # save obj_detector result with the matching robot pose and detector labels
        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:
                self.pred_fname = self.obj_detector.write_output(*self.obj_detector_output, self.lab_output_dir)
                self.notification = f'Prediction is saved to {operate.pred_fname}'
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

    # using computer vision to detect objects
    @staticmethod
    def _in_frame_fraction(box):
        """Fraction of this detection's box that lies within the camera frame.
        estimate_pose's pinhole depth calc assumes box_height reflects the
        object's true apparent height; if the object is cut off by the frame
        edge, the reported height (or the box's x-extent, for the lateral
        estimate) comes out too small and the position estimate is wrong.
        Computed from the box's own reported centre/width/height (not
        pre-clipped), so a box that genuinely extends past the frame scores
        low. Some detector/library versions clip predicted boxes to the frame
        internally, in which case a truncated object's box already sits fully
        within bounds -- as a second signal, a box whose edge sits right at
        the frame boundary is also treated as truncated either way.
        """
        x_center, y_center, w, h = box
        if w <= 0 or h <= 0:
            return 0.0
        x0, x1 = x_center - w / 2.0, x_center + w / 2.0
        y0, y1 = y_center - h / 2.0, y_center + h / 2.0

        edge_margin = 2.0  # pixels; a box flush against the boundary either way
        if (x0 <= edge_margin or x1 >= FRAME_WIDTH - edge_margin
                or y0 <= edge_margin or y1 >= FRAME_HEIGHT - edge_margin):
            return 0.0

        vis_w = max(0.0, min(x1, FRAME_WIDTH) - max(x0, 0.0))
        vis_h = max(0.0, min(y1, FRAME_HEIGHT) - max(y0, 0.0))
        return (vis_w * vis_h) / (w * h)

    def _fuse_fruit_observation(self, label, meas_x, meas_y, distance, heading):
        """Incorporate one fruit position observation into that label's running
        position/uncertainty estimate (self.fruit_state). Deliberately kept
        independent of self.ekf.markers/taglist/P -- fruits are never added to
        the actual SLAM state, so this can't affect M1's ArUco-only RMSE
        grading -- but reuses self.ekf.measurement_noise() (the same fitted
        noise-vs-distance model ArUco landmarks use), the same [depth,lateral]
        -> world rotation by robot heading that estimate_pose itself uses to
        build meas_x/meas_y (see object_pose_est.py), the same initial
        uninitialised-landmark prior (self.ekf.init_lm_cov), the same
        Joseph-form update, and the same covariance floor
        (self.ekf.min_lm_var) -- so a fruit's uncertainty ellipse is computed
        exactly the way an ArUco marker's is. Since estimate_pose already
        outputs a direct world-frame (x,y) position, the observation model
        here is linear (H = identity) -- unlike ArUco's EKF.update(), which
        also has to account for robot-pose uncertainty via a Jacobian, this
        takes the robot pose as given (whatever self.ekf currently estimates
        it to be) rather than re-estimating it.
        """
        z = np.array([[meas_x], [meas_y]])
        depth_var, lateral_var = self.ekf.measurement_noise(distance)
        Rot = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        R_world = Rot @ np.diag([depth_var, lateral_var]) @ Rot.T

        if label not in self.fruit_state:
            # First sighting -- same "uninitialised landmark" prior self.ekf starts a
            # newly-added ArUco marker at (see EKF.init_lm_cov)
            self.fruit_state[label] = {'pos': z, 'P': (self.ekf.init_lm_cov ** 2) * np.eye(2)}
            return

        st = self.fruit_state[label]
        pos, P = st['pos'], st['P']

        y = z - pos       # innovation; H = I since z is already a direct world-frame position
        S = P + R_world
        K = P @ np.linalg.inv(S)
        pos = pos + K @ y

        I2 = np.eye(2)
        I_K = I2 - K
        P = I_K @ P @ I_K.T + K @ R_world @ K.T
        P = 0.5 * (P + P.T)

        if P[0, 0] < self.ekf.min_lm_var:
            P[0, 0] = self.ekf.min_lm_var
        if P[1, 1] < self.ekf.min_lm_var:
            P[1, 1] = self.ekf.min_lm_var

        self.fruit_state[label] = {'pos': pos, 'P': P}

    def detect_object(self):
        # Only run the detector once the robot's been physically still for a
        # bit -- see stationary_settle_duration in __init__ -- so the captured
        # frame isn't motion-blurred.
        is_stationary = (time.time() - self.last_moving_time) >= self.stationary_settle_duration

        # Auto-trigger: same flag "p" sets, just fired on a timer instead of a
        # keypress, so the two paths share every line of detection logic below.
        # Gated on SLAM being on -- same condition ArUco landmarks are added/
        # updated under (see perform_slam()) -- so this doesn't spend YOLO
        # cycles automatically when there's no trustworthy pose to anchor a
        # position estimate to anyway.
        if (self.auto_detect_enabled and self.ekf_on and self.obj_detector is not None
                and is_stationary
                and time.time() - self.last_auto_detect_time >= self.auto_detect_interval):
            self.command['run_obj_detector'] = True
            self.last_auto_detect_time = time.time()

        if self.command['run_obj_detector'] and self.obj_detector is not None:
            if not is_stationary:
                # Manual "p" press while still moving/settling -- leave the flag
                # set rather than dropping it, so this fires on its own on the
                # very next still tick without needing to press "p" again.
                self.notification = 'Waiting for robot to stop before running detector (avoids blur)'
                return
            bboxes, self.cv_vis = self.obj_detector.detect_single_image(self.img)
            self.command['run_obj_detector'] = False
            self.obj_detector_output = (self.cv_vis, self.ekf.robot.state.tolist(), bboxes) # three things to be saved
            unique_detected = len(set([box[0] for box in bboxes]))
            self.notification = f'{unique_detected} object type(s) detected'

            # Live fruit-position estimate for the SLAM GUI/map (see __init__
            # note). Same condition ArUco landmarks are added/updated under
            # (self.ekf_on) -- with SLAM paused, self.ekf.robot.state is
            # frozen wherever it was left, so a position computed from it now
            # would be anchored to a stale pose. "p" still runs the detector
            # (so the Detector view updates) even with SLAM off; it just
            # won't feed fruit positions into the map while paused.
            if self.ekf_on:
                robot_pose = self.ekf.robot.state.tolist()
                focal_length = self.ekf.robot.camera_matrix[0][0]
                cx = self.ekf.robot.camera_matrix[0][2]
                rx, ry = robot_pose[0][0], robot_pose[1][0]
                heading = self.ekf.robot.state[2, 0]
                for label, box in bboxes:
                    true_height = self.object_dimensions.get(label)
                    if true_height is None:
                        continue  # unknown class, e.g. not in object_list.csv
                    if self._in_frame_fraction(box) < 0.8:
                        continue  # too close to/past the frame edge -- unreliable box size
                    pose_x, pose_y = estimate_pose(robot_pose, box, true_height, focal_length, cx)
                    dist = float(np.hypot(pose_x - rx, pose_y - ry))
                    self._fuse_fruit_observation(label, pose_x, pose_y, dist, heading)

                self.fruit_map_display = {
                    label: {'x': float(st['pos'][0, 0]), 'y': float(st['pos'][1, 0]), 'P': st['P']}
                    for label, st in self.fruit_state.items()
                }

    # paint the GUI
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        # compute live RMSE only if a true map is loaded AND tracking is enabled
        active_true_map = self.true_map if (self.true_map is not None and self.show_live_rmse) else None
        live_rmse_info = self.ekf.compute_live_rmse(active_true_map) if active_true_map is not None else None

        # paint SLAM outputs
        fruit_colours = self.obj_detector.colour_code if self.obj_detector is not None else None
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause=self.ekf_on,
                                            true_map=active_true_map, live_rmse_info=live_rmse_info,
                                            selected_tag=self.pending_delete_tag,
                                            fruit_map=self.fruit_map_display, fruit_colours=fruit_colours)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for object detector
        detector_view = cv2.resize(self.cv_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240+2*v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='Robot Cam', position=(h_pad, v_pad))
        notification = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        # live RMSE readout in the main window
        if self.true_map is None:
            rmse_line = "No true map loaded"
        elif not self.show_live_rmse:
            rmse_line = "Live RMSE tracking OFF (press L to toggle)"
        elif live_rmse_info is None:
            rmse_line = f"Live RMSE: need >=2 matched markers (have {len(self.ekf.taglist)})"
        else:
            rmse_line = (f"Live RMSE: {live_rmse_info['rmse']:.4f} m "
                        f"({len(live_rmse_info['matched_tags'])}/{len(self.true_map)} markers)")
        rmse_surface = TEXT_FONT.render(rmse_line, False, text_colour)
        canvas.blit(rmse_surface, (h_pad+10, 624))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
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

            if event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                self.adjust_pid('kp', -self.pid_step)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                self.adjust_pid('kp', self.pid_step)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                self.adjust_pid('ki', -self.pid_step)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                self.adjust_pid('ki', self.pid_step)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                self.adjust_pid('kd', -self.pid_step)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.adjust_pid('kd', self.pid_step)

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
            # reset SLAM map (or, while the true map is loaded, just the fruits --
            # see below)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    confirm_msg = ('Press again to confirm CLEAR FRUITS' if self.ekf.freeze_map
                                    else 'Press again to confirm CLEAR MAP')
                    self.notification = confirm_msg
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.double_reset_comfirm = 0
                    self.fruit_state = {}
                    self.fruit_map_display = {}
                    if self.ekf.freeze_map:
                        # True map is loaded -- 'r' only clears fruit tracking.
                        # self.ekf (markers/taglist/P/freeze_map, robot pose) is
                        # left completely untouched; 't' is the only thing that
                        # unloads the true map now (see below).
                        self.notification = 'Fruits cleared - true map still loaded'
                    else:
                        self.ekf.reset()
                        if os.path.exists(self.slam_state_fname):
                            os.remove(self.slam_state_fname)
                        self.notification = 'SLAM map and fruits are cleared'
            # M2: toggle-load the ground-truth ArUco map into self.ekf.markers and
            # freeze it, so SLAM keeps localising the robot pose without ever
            # adding or moving landmarks -- this is what the M2 spec means by
            # loading truemap.txt's coordinates into self.markers, and what it
            # explicitly allows ("you are allowed to use the true map during
            # demonstration"). A frozen, correctly-anchored robot pose is what
            # makes estimate_pose's fruit positions (see detect_object())
            # accurate. Press again to unload (drop the true markers, un-freeze,
            # robot pose belief left as-is) and go back to SLAM building its own
            # map from scratch -- 'r' no longer does this while a true map is
            # loaded (see above).
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                if self.ekf.freeze_map:
                    n = self.ekf.unload_true_map()
                    self.notification = f'True map unloaded ({n} markers dropped) - SLAM will build its own map'
                elif os.path.exists(self.truemap_path):
                    n = self.ekf.load_true_map(self.truemap_path)
                    self.notification = f'True map loaded ({n} markers) - map is now frozen'
                else:
                    self.notification = f'No true map found at {self.truemap_path}'
            # run object/fruit detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_obj_detector'] = True
            # capture and save raw image
            elif event.type == pygame.KEYDOWN and event.key  == pygame.K_i:
                self.command['save_image'] = True
            # toggle live RMSE tracking on/off
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                self.show_live_rmse = not self.show_live_rmse
                state = 'ON' if self.show_live_rmse else 'OFF'
                self.notification = f'Live RMSE tracking {state}'
            # toggle automatic background fruit detection on/off
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.auto_detect_enabled = not self.auto_detect_enabled
                state = 'ON' if self.auto_detect_enabled else 'OFF'
                self.notification = f'Auto fruit detection {state}'
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
            self.last_moving_time = time.time()  # the pulse is still physically driving the robot
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
        if adjusted[0] != 0.0 or adjusted[1] != 0.0:
            self.last_moving_time = time.time()




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

    width, height = 700, 660
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

    width, height = 900, 660
    canvas = pygame.display.set_mode((width, height))
    operate = Operate(args)
    while start:
        operate.update_keyboard()
        operate.correct_straight_drive()
        operate.take_pic()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()
