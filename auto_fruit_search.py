# M3 - Autonomous object searching
import os
import sys
import cv2
import ast
import json
import time
import math
import argparse
import numpy as np
from botconnect import BotConnect # access the robot communication

# SLAM components -- same modules M1/M2 already use (operate.py), so M3 runs
# under identical SLAM machinery rather than a separate implementation.
from slam.ekf import EKF, DriveMeasurement
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

import path_planner


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
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
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
        for i in range(len(search_list)):
            if obj == object_list[i]:
                print('{}) {} at [{}, {}]'.format(n_object, obj, np.round(object_true_pos[i][0], 1), np.round(object_true_pos[i][1], 1)))
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
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=172.5)
    return EKF(robot), float(baseline)


class Navigator:
    """
    Wraps the robot connection and the frozen-map SLAM (slam/ekf.py's
    load_true_map()) that M3 navigation needs: pose estimation
    (get_robot_pose) and calibrated turn-then-drive execution (turn /
    drive_forward), built on the same encoder + ArUco pipeline M1/M2 already
    use in operate.py rather than a separate one.
    """

    def __init__(self, botconnect, ekf, aruco_sensor, baseline,
                 drive_speed=0.4, turn_speed=0.35, move_timeout=15.0):
        self.botconnect = botconnect
        self.ekf = ekf
        self.aruco_sensor = aruco_sensor
        self.ticks_per_meter = ekf.robot.ticks_per_meter
        self.baseline = baseline
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.move_timeout = move_timeout

        self._prev_left = None
        self._prev_right = None
        self._control_clock = None

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    def get_robot_pose(self):
        """
        Returns the SLAM-corrected pose as np.array([x, y, theta]).

        First call (no drive history yet): ekf.predict() hasn't run, so P is
        still exactly zero (set that way by load_true_map()) and a Kalman
        update() would compute a zero gain and change nothing. Use
        recover_from_pause()'s direct rigid alignment (Umeyama) against
        whatever markers are currently visible instead -- that's exactly
        what it's built for (see its docstring in slam/ekf.py).

        Every later call: predict() with the encoder delta accumulated since
        the previous get_robot_pose() call (this is also what makes P
        non-zero, so update() has something to correct with), then update()
        to fold in whatever markers are visible now.
        """
        drive_measurement = self._drive_measurement_since_last_call()

        if drive_measurement is not None:
            self.ekf.predict(drive_measurement)

        img = self.botconnect.get_image()
        sensor_measurement, _ = self.aruco_sensor.detect_marker_positions(img)

        if drive_measurement is None:
            if not self.ekf.recover_from_pause(sensor_measurement):
                print("get_robot_pose: fewer than 2 markers visible for the initial "
                      "fix -- pose is still the default (0, 0, 0). Rotate in place "
                      "until at least 2 markers are visible before driving.")
        else:
            self.ekf.update(sensor_measurement)

        state = self.ekf.robot.state
        return np.array([state[0, 0], state[1, 0], state[2, 0]])

    def _drive_measurement_since_last_call(self):
        left, right = self.botconnect.get_encoder_counts()
        now = time.time()

        if self._prev_left is None:
            self._prev_left, self._prev_right, self._control_clock = left, right, now
            return None

        dt = now - self._control_clock
        delta_left = left - self._prev_left
        delta_right = right - self._prev_right
        # Same guard as operate.py's get_drive_signal(): the Pi resets its
        # encoder counters on every stop, which would otherwise show up here
        # as a large spurious negative jump.
        if delta_left < -10 or delta_right < -10:
            delta_left, delta_right = 0, 0

        self._prev_left, self._prev_right, self._control_clock = left, right, now
        return DriveMeasurement(self.botconnect.left_speed, self.botconnect.right_speed, dt,
                                 delta_left_ticks=delta_left, delta_right_ticks=delta_right)

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------

    def turn(self, dtheta):
        """In-place turn by `dtheta` radians (positive = anticlockwise/left,
        matching robot.py's state[2] convention and operate.py's K_LEFT
        binding of [-0.35, 0.35])."""
        if abs(dtheta) < 1e-3:
            return
        arc_length = (self.baseline / 2.0) * abs(dtheta)
        ticks = max(1, int(round(arc_length * self.ticks_per_meter)))
        speed = self.turn_speed if dtheta > 0 else -self.turn_speed
        self.botconnect.move_auto_encoder([-speed, speed], ticks, ticks)
        self._wait_for_move()

    def drive_forward(self, distance):
        """Drive straight forward by `distance` metres (>= 0)."""
        if distance < 1e-3:
            return
        ticks = max(1, int(round(distance * self.ticks_per_meter)))
        self.botconnect.move_auto_encoder([self.drive_speed, self.drive_speed], ticks, ticks)
        self._wait_for_move()

    def _wait_for_move(self):
        start = time.time()
        while not self.botconnect.autonomous_done:
            if time.time() - start > self.move_timeout:
                print("WARNING: move exceeded {:.0f}s timeout -- stopping".format(self.move_timeout))
                self.botconnect.stop()
                return
            time.sleep(0.02)


def _normalize_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
def drive_to_point(waypoint, nav):
    """
    Turn-then-drive to `waypoint` = (x, y), using nav's SLAM-corrected pose to
    aim and nav's calibrated encoder-based driving to execute. Re-checks pose
    after turning -- a turn also accumulates drive ticks -- before committing
    to a straight-line distance, and once more after arriving, per the
    manual's own recommendation to correct pose after every waypoint step.
    @return: robot pose (np.array([x, y, theta])) after arriving
    """
    pose = nav.get_robot_pose()
    target_heading = math.atan2(waypoint[1] - pose[1], waypoint[0] - pose[0])
    dtheta = _normalize_angle(target_heading - pose[2])
    nav.turn(dtheta)

    pose = nav.get_robot_pose()
    distance = math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1])
    nav.drive_forward(distance)

    pose = nav.get_robot_pose()
    print("Arrived near [{:.3f}, {:.3f}] -- pose now [{:.3f}, {:.3f}, {:.1f}deg]".format(
        waypoint[0], waypoint[1], pose[0], pose[1], math.degrees(pose[2])))
    return pose


def get_robot_pose(nav):
    """Thin wrapper matching the manual's function name -- see
    Navigator.get_robot_pose() in this file for the actual SLAM logic."""
    return nav.get_robot_pose()


def run_level1(nav, search_list, object_list, object_true_pos, aruco_true_pos):
    """
    M3 Level 1: full known map given, navigate to every search_list.txt target
    IN ORDER. Builds a fresh RRT* route per target (see path_planner.py),
    drives it waypoint-by-waypoint, then waits for the demonstrator's
    verification before moving on -- matching the manual's 4.4.1 checklist.
    """
    object_radii = path_planner.load_object_radii()
    object_positions = {name: tuple(pos) for name, pos in zip(object_list, object_true_pos)}

    for i, target_name in enumerate(search_list, start=1):
        if target_name not in object_positions:
            print(f"WARNING: '{target_name}' from search_list.txt isn't in the true map -- skipping")
            continue
        target = object_positions[target_name]

        pose = nav.get_robot_pose()
        obstacles = path_planner.build_obstacles(
            aruco_true_pos, object_positions, path_planner.ROBOT_RADIUS,
            object_radii=object_radii, exclude=target_name)

        goal = path_planner.standoff_point(target, obstacles, reference=pose[:2])
        if goal is None:
            print(f"[{target_name}] no collision-free standoff point found -- skipping")
            continue

        path = path_planner.rrt_star(pose[:2], goal, obstacles)
        if path is None:
            print(f"[{target_name}] RRT* found no path within the iteration budget -- skipping")
            continue
        path = path_planner.smooth_path(path, obstacles)

        print(f"\n[{target_name}] driving {len(path)}-waypoint route towards [{target[0]:.2f}, {target[1]:.2f}]")
        for wp in path:
            drive_to_point(wp, nav)

        final_pose = nav.get_robot_pose()
        dist_to_target = math.hypot(final_pose[0] - target[0], final_pose[1] - target[1])
        status = "OK" if dist_to_target <= 0.4 else "OUT OF TOLERANCE"
        print(f"=== Found {target_name} at [{target[0]:.2f}, {target[1]:.2f}] "
              f"(robot is {dist_to_target:.3f}m away -- {status}) ===")
        input(f"Press ENTER once the demonstrator has verified this ({i}/{len(search_list)})...")


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
    args, _ = parser.parse_known_args()

    botconnect = BotConnect(args.ip)
    time.sleep(1)  # give connection threads a moment to establish
    botconnect.set_pid(use_pid=1, kp=2, ki=0.04, kd=0.29)  # same gains as operate.py

    ekf, baseline = init_ekf(args.calib_dir)
    ekf.load_true_map(args.map)  # freezes markers, M3 Level 1/2/3 all localise-only
    aruco_sensor = ArucoSensor(ekf.robot, marker_length=0.06)

    nav = Navigator(botconnect, ekf, aruco_sensor, baseline)

    # read in the true map
    object_list, object_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_object_pos(search_list, object_list, object_true_pos)

    if args.manual:
        run_manual(nav)
    else:
        run_level1(nav, search_list, object_list, object_true_pos, aruco_true_pos)
