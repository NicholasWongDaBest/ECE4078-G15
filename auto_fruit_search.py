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
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=172.5)
    return EKF(robot), float(baseline)


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
        sensor_measurement, _ = self.aruco_sensor.detect_marker_positions(img)
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
            self.ekf.update(sensor_measurement)
            diagnostics = self.ekf.last_diagnostics
            if n_known >= 2 and diagnostics and all(d['gated'] for d in diagnostics):
                if self.ekf.recover_from_pause(sensor_measurement):
                    s = self.ekf.robot.state
                    print("get_robot_pose: markers disagreed with the predicted pose -- re-anchored "
                          "from {} markers to [{:.3f}, {:.3f}, {:.1f}deg]".format(
                              n_known, s[0, 0], s[1, 0], math.degrees(_normalize_angle(s[2, 0]))))

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

    def turn(self, dtheta):
        """In-place turn by `dtheta` radians (positive = anticlockwise/left,
        matching robot.py's state[2] convention and operate.py's K_LEFT
        binding of [-0.35, 0.35]). One tick is the smallest possible turn
        (about 4.8 deg), so anything under half a tick is skipped."""
        arc_length = (self.wheel_separation / 2.0) * abs(dtheta)
        ticks = int(round(arc_length * self.ticks_per_meter * self.turn_scale))
        if ticks < 1:
            return
        speed = self.turn_speed if dtheta > 0 else -self.turn_speed
        wheel_speeds = [-speed, speed]
        start = time.time()
        self.botconnect.move_auto_encoder(wheel_speeds, ticks, ticks)
        completed = self._wait_for_move()
        # the rotation those whole ticks are expected to produce
        commanded = math.copysign(
            2.0 * ticks / (self.ticks_per_meter * self.wheel_separation * self.turn_scale), dtheta)
        self._predict_commanded_motion(wheel_speeds, time.time() - start,
                                       dtheta=commanded, completed=completed)
        self._settle()

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
def drive_to_point(waypoint, nav):
    """
    Turn-then-drive to `waypoint` = (x, y), using nav's SLAM-corrected pose to
    aim and nav's calibrated encoder-based driving to execute. After turning,
    re-checks the pose and corrects the heading (up to
    nav.max_heading_corrections times) before committing to the straight
    line, since a 10 deg heading error puts the robot 17 cm off course after
    1 m. Checks the pose once more after arriving, per the manual's own
    recommendation to correct pose after every waypoint step.
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
    nav.drive_forward(distance)

    pose = nav.get_robot_pose()
    print("Arrived near [{:.3f}, {:.3f}] -- pose now [{:.3f}, {:.3f}, {:.1f}deg]".format(
        waypoint[0], waypoint[1], pose[0], pose[1], math.degrees(_normalize_angle(pose[2]))))
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


def run_level1(nav, search_list, object_list, object_true_pos, aruco_true_pos):
    """
    M3 Level 1: full known map given, navigate to every search_list.txt target
    IN ORDER. Builds a fresh RRT* route per target (see path_planner.py),
    drives it waypoint-by-waypoint, then waits for the demonstrator's
    verification before moving on -- matching the manual's 4.4.1 checklist.
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
            object_radii=object_radii, exclude=target_name)
        display.begin_target(i, target_name, target, obstacles)
        display.notify(f"Planning route to {target_name} ({i}/{len(search_list)})")
        display.refresh(force=True)

        pose = nav.get_robot_pose()

        start = _escape_obstacles(pose[:2], obstacles)
        if start is None:
            skip(target_name, f"robot at [{pose[0]:.2f}, {pose[1]:.2f}] is boxed in by safety circles")
            continue
        if path_planner.dist_between(start, pose[:2]) > 1e-9:
            print(f"[{target_name}] pose estimate is inside an obstacle's safety circle -- "
                  f"planning from the nearest free point [{start[0]:.2f}, {start[1]:.2f}]")

        goal = path_planner.standoff_point(target, obstacles, reference=start)
        if goal is None:
            skip(target_name, "no collision-free standoff point")
            continue

        path = None
        try:
            path = path_planner.rrt_star(start, goal, obstacles)
            if path is None:
                print(f"[{target_name}] RRT* found no path in 2000 iterations -- retrying with 4000")
                path = path_planner.rrt_star(start, goal, obstacles, max_iter=4000)
        except ValueError as e:
            print(f"[{target_name}] planner refused: {e}")
        if path is None:
            skip(target_name, "no route found")
            continue
        path = path_planner.smooth_path(path, obstacles)

        # path[0] is the planning start. If that's where the robot already is,
        # there's nothing to drive to (it's only different after an escape).
        waypoints = path[1:] if path_planner.dist_between(path[0], pose[:2]) < nav.min_move else path

        print(f"\n[{target_name}] driving {len(waypoints)}-waypoint route towards [{target[0]:.2f}, {target[1]:.2f}]")
        display.set_route(goal, waypoints)
        display.notify(f"Driving to {target_name} ({i}/{len(search_list)})")
        for wp in waypoints:
            drive_to_point(wp, nav)

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

    display = None
    try:
        if not args.no_display:
            object_positions = {name: tuple(pos) for name, pos in zip(object_list, object_true_pos)}
            display = M3Display(nav, aruco_true_pos, object_positions, search_list,
                                object_radii=path_planner.load_object_radii())
            nav.display = display
            # Turn with the arrow keys until 2+ markers are in view, then ENTER.
            display.run_setup()

        if args.manual:
            if display is not None:
                display.run_manual(lambda wp: drive_to_point(wp, nav))
            else:
                run_manual(nav)
        else:
            run_level1(nav, search_list, object_list, object_true_pos, aruco_true_pos)
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
