# Rotation calibration for M3's encoder-based turning.
#
# calibrate_encoder.py runs the forward direction: "drive this command for this
# long, then tell me how many ticks came back". This script runs the INVERSE,
# which is the direction auto_fruit_search.py actually needs: "I want the robot
# to turn 90 degrees -- how many encoder ticks must I send, and what correction
# factor makes the commanded angle match the real one?"
#
# Why a separate number is needed at all
# --------------------------------------
# Navigator.turn() in auto_fruit_search.py converts an angle into ticks purely
# from geometry:
#
#     arc_length = (wheel_separation / 2) * |dtheta|
#     ticks      = round(arc_length * ticks_per_meter * turn_scale)
#
# That geometry assumes the wheels roll cleanly about the robot's centre. They
# don't: turning on the spot scrubs both tyres sideways across the floor, so a
# tick of wheel rotation produces LESS body rotation than the arc-length maths
# predicts, and the robot lands short. ticks_per_meter was measured from
# straight-line driving, where no scrub happens, so it cannot capture this --
# which is exactly why calibrate_encoder.py's own header warns that a
# forward-only ticks_per_meter "may not hold once the robot starts turning".
#
# turn_scale is the one number that absorbs the difference. This script
# measures it.
#
# What it does
# ------------
#   1. Commands a series of turns through the SAME tick maths Navigator.turn()
#      uses, at the same wheel speed and PID gains, so the result transfers.
#   2. Measures how far the robot ACTUALLY turned, either automatically from
#      the ArUco markers (--measure aruco, needs the true map) or from your own
#      protractor/floor-mark reading (--measure manual).
#   3. Least-squares fits ticks-per-radian against those measurements, reports
#      turn_scale, the implied effective wheel separation, per-direction
#      asymmetry and the residuals, then writes calibration/param/turn_scale.txt.
#
# Usage
# -----
#   python calibrate_turn.py --ip <robot_ip> --map truemap.txt      # auto-measured
#   python calibrate_turn.py --ip <robot_ip> --measure manual       # protractor
#   python calibrate_turn.py --dry-run                              # maths only, no robot
#
# Place the robot near the centre of the arena with at least two markers in
# view, and keep them in view throughout -- in aruco mode the heading is solved
# from whichever known markers are visible, so it's fine if they're different
# markers before and after the turn, but there must be two of them each time.

import os
import sys
import json
import math
import time
import argparse

import numpy as np

from botconnect import BotConnect


# --- Everything below is mirrored from auto_fruit_search.py on purpose. -------
# A calibration run has to experience the same control conditions as a real M3
# run, or the number it produces describes a regime the robot never drives in.
# If you change any of these there, change them here too.

TURN_SPEED = 0.35                                  # Navigator.__init__'s turn_speed
PID_GAINS = {'kp': 2, 'ki': 0.04, 'kd': 0.29}      # auto_fruit_search.py's __main__
TICKS_PER_METER = 172.5                            # init_ekf()'s Robot(...) argument
SETTLE_TIME = 0.4                                  # Navigator.__init__'s settle_time
MOVE_TIMEOUT = 15.0                                # Navigator.__init__'s move_timeout
MARKER_LENGTH = 0.06                               # ArucoSensor(...) in __main__

PARAM_DIR = os.path.join('calibration', 'param')
TURN_SCALE_FILE = os.path.join(PARAM_DIR, 'turn_scale.txt')
LOG_DIR = 'lab_output'

# Angles to test, in degrees, signed (+ = anticlockwise/left, matching
# robot.py's state[2] convention). Both directions and two magnitudes, twice
# over: two directions expose motor/mechanical asymmetry, two magnitudes
# separate a genuine scale error (grows with the angle) from a fixed
# per-move offset (doesn't), and the repeat gives a repeatability figure.
#
# Deliberately kept under 150 deg so that in aruco mode the wrapped
# before/after heading difference is unambiguous -- see measure_trial().
DEFAULT_TRIALS = [90, -90, 135, -135, 90, -90]


def wrap_pi(angle):
    """Wrap an angle to (-pi, pi] -- same convention as auto_fruit_search.py's
    _normalize_angle()."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# The tick maths under test
# ---------------------------------------------------------------------------

class TurnRig:
    """
    Navigator.turn()'s angle-to-ticks conversion and move execution, extracted
    so it can be driven with an arbitrary turn_scale. Intentionally a copy of
    the real thing rather than an import of it: Navigator also runs the EKF
    predict step, which would fight the direct marker-based heading readings
    this script takes.
    """

    def __init__(self, botconnect, wheel_separation, ticks_per_meter=TICKS_PER_METER,
                 turn_speed=TURN_SPEED, move_timeout=MOVE_TIMEOUT, settle_time=SETTLE_TIME):
        self.botconnect = botconnect
        # baseline.txt stores the wheel separation NEGATIVE (-0.1386: the
        # calibration formula used the -0.5 wheel speed). Navigator takes
        # abs() of it for the tick maths and leaves the EKF's signed copy
        # alone; do the same here.
        self.wheel_separation = abs(float(wheel_separation))
        self.ticks_per_meter = float(ticks_per_meter)
        self.turn_speed = turn_speed
        self.move_timeout = move_timeout
        self.settle_time = settle_time

    def nominal_ticks_per_radian(self):
        """Ticks per radian of body rotation that the pure geometry predicts,
        i.e. what Navigator.turn() sends at turn_scale = 1.0."""
        return (self.wheel_separation / 2.0) * self.ticks_per_meter

    def ticks_for(self, dtheta, turn_scale):
        """Exactly Navigator.turn()'s tick count for a requested turn (>= 0)."""
        arc_length = (self.wheel_separation / 2.0) * abs(dtheta)
        return int(round(arc_length * self.ticks_per_meter * turn_scale))

    def angle_for(self, ticks, turn_scale):
        """Inverse of ticks_for(): the body rotation `ticks` is believed to
        produce. This is the number Navigator.turn() hands to the EKF as the
        commanded motion, so an error here is an error the filter never sees."""
        return 2.0 * ticks / (self.ticks_per_meter * self.wheel_separation * turn_scale)

    def execute(self, dtheta, turn_scale):
        """
        Send one in-place turn and wait for it to finish.
        @return: dict with the ticks sent, the angle those ticks were believed
            to produce, whether the move completed, and the encoder counts the
            robot reported either side of it.
        """
        ticks = self.ticks_for(dtheta, turn_scale)
        if ticks < 1:
            # One tick is the smallest possible turn; below half a tick
            # Navigator.turn() rounds to zero and does nothing at all.
            return {'ticks': 0, 'believed_rad': 0.0, 'completed': True,
                    'enc_before': None, 'enc_after': None, 'skipped': True}

        speed = self.turn_speed if dtheta > 0 else -self.turn_speed
        enc_before = self.botconnect.get_encoder_counts()
        self.botconnect.move_auto_encoder([-speed, speed], ticks, ticks)
        completed = self._wait_for_move()
        enc_after = self.botconnect.get_encoder_counts()

        return {'ticks': ticks,
                'believed_rad': math.copysign(self.angle_for(ticks, turn_scale), dtheta),
                'completed': completed,
                'enc_before': enc_before,
                'enc_after': enc_after,
                'skipped': False}

    def _wait_for_move(self):
        """Navigator._wait_for_move(): block until the robot reports done, then
        let the camera deliver a frame taken after it stopped moving."""
        start = time.time()
        completed = True
        while not self.botconnect.autonomous_done:
            if time.time() - start > self.move_timeout:
                print("  WARNING: move exceeded {:.0f}s timeout -- stopping".format(self.move_timeout))
                self.botconnect.stop()
                completed = False
                break
            time.sleep(0.02)
        time.sleep(self.settle_time)
        return completed


# ---------------------------------------------------------------------------
# Measuring what actually happened
# ---------------------------------------------------------------------------

class ArucoHeading:
    """
    Absolute heading from the ArUco markers, via the frozen true map.

    Uses ekf.recover_from_pause(), which best-fits the observed marker geometry
    onto the mapped geometry and writes the resulting pose straight into
    ekf.robot.state -- an absolute fix, not an increment. That matters here:
    the robot may be looking at a completely different set of markers after a
    90 deg turn than before it, and this still works as long as at least two
    KNOWN markers are in view each time.
    """

    def __init__(self, botconnect, map_file, calib_dir):
        from slam.ekf import EKF
        from slam.robot import Robot
        from slam.aruco_sensor import ArucoSensor

        camera_matrix = np.loadtxt(os.path.join(calib_dir, 'intrinsic.txt'), delimiter=',')
        dist_coeffs = np.loadtxt(os.path.join(calib_dir, 'distCoeffs.txt'), delimiter=',')
        scale = np.loadtxt(os.path.join(calib_dir, 'scale.txt'), delimiter=',')
        baseline = np.loadtxt(os.path.join(calib_dir, 'baseline.txt'), delimiter=',')

        robot = Robot(baseline, scale, camera_matrix, dist_coeffs, ticks_per_meter=TICKS_PER_METER)
        self.ekf = EKF(robot)
        n = self.ekf.load_true_map(map_file)
        self.aruco_sensor = ArucoSensor(robot, marker_length=MARKER_LENGTH)
        self.botconnect = botconnect
        print(f"Loaded {n} ground-truth markers from {map_file} for heading measurement.")

    def read(self):
        """
        @return: (heading_rad, n_known_markers, fit_residual_m), or
            (None, n_known_markers, None) if fewer than 2 known markers were
            visible -- in which case the heading could not be solved at all.
        """
        img = self._latest_image()
        sensor_measurement, _ = self.aruco_sensor.detect_marker_positions(img)
        n_known = sum(1 for lm in sensor_measurement if lm.tag in self.ekf.taglist)
        if not self.ekf.recover_from_pause(sensor_measurement):
            return None, n_known, None
        return float(self.ekf.robot.state[2, 0]), n_known, float(self.ekf.last_recover_residual)

    def _latest_image(self):
        deadline = time.time() + 5.0
        while getattr(self.botconnect, 'frame', True) is None and time.time() < deadline:
            time.sleep(0.05)
        return self.botconnect.get_image()

    def noise_check(self, n=5):
        """
        Take several headings without moving, to establish the measurement
        floor. If the spread here is comparable to the turn errors being
        measured, the trials are measuring marker noise rather than turn
        accuracy, and the fit will be meaningless -- move the robot somewhere
        with more markers in view before continuing.
        """
        readings = []
        for _ in range(n):
            heading, n_known, _ = self.read()
            if heading is not None:
                readings.append(heading)
            time.sleep(0.15)
        if len(readings) < 2:
            return None
        # Spread about the circular mean, so a reading near +/-pi doesn't
        # blow the standard deviation up.
        mean = math.atan2(np.mean(np.sin(readings)), np.mean(np.cos(readings)))
        deviations = [wrap_pi(r - mean) for r in readings]
        return float(np.std(deviations))


def measure_trial(rig, requested_deg, turn_scale, heading_source):
    """
    Run one trial and return what was sent and what actually happened.

    @param heading_source: an ArucoHeading, or None for manual measurement
    @return: trial dict, or None if the trial had to be discarded
    """
    requested_rad = math.radians(requested_deg)
    ticks = rig.ticks_for(requested_rad, turn_scale)
    believed_deg = math.degrees(rig.angle_for(ticks, turn_scale)) if ticks else 0.0

    print(f"\n  requested {requested_deg:+.1f} deg -> {ticks} ticks "
          f"(which the robot is told is {math.copysign(believed_deg, requested_deg):+.1f} deg)")
    if ticks < 1:
        print("  skipped: rounds to zero ticks")
        return None

    before = None
    if heading_source is not None:
        before, n_known, residual = heading_source.read()
        if before is None:
            print(f"  skipped: only {n_known} known markers in view, need 2 to solve a heading")
            return None
        print(f"  heading before: {math.degrees(before):+7.2f} deg "
              f"({n_known} markers, fit residual {residual:.3f} m)")

    move = rig.execute(requested_rad, turn_scale)
    if not move['completed']:
        print("  discarded: the move timed out, so the tick target wasn't reached")
        return None

    if heading_source is not None:
        after, n_known, residual = heading_source.read()
        if after is None:
            print(f"  discarded: only {n_known} known markers in view after the turn")
            return None
        # The wrapped difference is only unambiguous for |turn| < 180 deg --
        # DEFAULT_TRIALS stays under 150 for exactly this reason.
        actual_rad = wrap_pi(after - before)
        print(f"  heading after:  {math.degrees(after):+7.2f} deg "
              f"({n_known} markers, fit residual {residual:.3f} m)")
        if abs(requested_deg) > 150:
            print("  WARNING: turns beyond 150 deg are ambiguous once the heading "
                  "difference is wrapped -- treat this trial with suspicion")
    else:
        measured_deg = prompt_measured_degrees(requested_deg)
        if math.isnan(measured_deg):
            print("  discarded at your request")
            return None
        actual_rad = math.radians(measured_deg)

    actual_deg = math.degrees(actual_rad)
    print(f"  ACTUAL: {actual_deg:+.2f} deg   (asked for {requested_deg:+.1f}, "
          f"short by {abs(requested_deg) - abs(actual_deg):+.2f} deg)")

    if move['enc_before'] is not None:
        el, er = move['enc_before']
        al, ar = move['enc_after']
        print(f"  encoder counts reported: before ({el}, {er}) after ({al}, {ar})")

    # Signed ticks: positive for a left turn, negative for a right turn, so a
    # single least-squares fit covers both directions at once.
    return {'requested_deg': float(requested_deg),
            'ticks_signed': float(math.copysign(move['ticks'], requested_deg)),
            'actual_rad': float(actual_rad),
            'actual_deg': float(actual_deg),
            'believed_deg': float(math.copysign(believed_deg, requested_deg)),
            'turn_scale_used': float(turn_scale)}


def prompt_measured_degrees(requested_deg):
    """Ask for a protractor/floor-mark reading. Signed, + = anticlockwise."""
    hint = ""
    if abs(requested_deg) >= 360:
        revs = int(abs(requested_deg) // 360)
        hint = (f"\n    (that was about {revs} full revolutions -- if it stopped 12 deg short "
                f"of the mark, enter {math.copysign(revs * 360 - 12, requested_deg):+.0f})")
    while True:
        raw = input(f"  How many degrees did it ACTUALLY turn? "
                    f"(+ = anticlockwise/left, blank to discard){hint}\n  > ").strip()
        if raw == "":
            return float('nan')
        try:
            value = float(raw)
        except ValueError:
            print("    Please enter a number, or blank to discard this trial.")
            continue
        if value * requested_deg < 0:
            confirm = input("    That's the opposite direction to the request. Is that right? [y/N] ")
            if confirm.lower() != 'y':
                continue
        return value


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_ticks_per_radian(trials):
    """
    Least-squares fit of k in `angle = ticks / k`, through the origin.

    Fitting the ANGLE residual (not the tick residual) is the right way round
    here: the ticks are exact integers the robot was told to hit, and all the
    uncertainty lives in the measured angle. Minimising
    sum((ticks_i/k - angle_i)^2) gives k = sum(ticks^2) / sum(ticks*angle).

    Forcing it through the origin is deliberate. An intercept would model a
    fixed per-move offset, but such an offset would have to come from
    start/stop transients, which the PID controller already absorbs; letting
    two magnitudes of turn fight over an intercept just makes the scale noisy.
    If the residuals below come out systematically signed by magnitude, that
    assumption is the thing to question.
    """
    trials = [t for t in trials if not math.isnan(t['actual_rad'])]
    if not trials:
        return None
    numerator = sum(t['ticks_signed'] ** 2 for t in trials)
    denominator = sum(t['ticks_signed'] * t['actual_rad'] for t in trials)
    if abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def residuals_deg(trials, k):
    """
    How much further the robot turned than the fit predicts, per trial, in
    degrees. Signed relative to the turn's own direction (positive = turned
    further than predicted, whichever way it was going) rather than in world
    terms, so that a robot which over-turns left and under-turns right shows
    up as +x and -x rather than as two same-signed numbers.
    """
    out = []
    for t in trials:
        predicted = t['ticks_signed'] / k
        signed = math.copysign(1.0, t['ticks_signed'])
        out.append(signed * math.degrees(t['actual_rad'] - predicted))
    return out


def report(trials, rig, k_all):
    """Print the fit, the per-trial table, the direction split and the limits."""
    k_nominal = rig.nominal_ticks_per_radian()
    turn_scale = k_all / k_nominal
    effective_separation = 2.0 * k_all / rig.ticks_per_meter

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  ticks per radian, geometric (turn_scale = 1):  {k_nominal:.3f}")
    print(f"  ticks per radian, measured:                    {k_all:.3f}")
    print(f"\n  >>> turn_scale = {turn_scale:.4f} <<<")
    if turn_scale > 1:
        print(f"      The robot under-turns by {100 * (turn_scale - 1):.1f}% -- it needs that "
              f"much\n      extra wheel rotation to sweep the angle asked for. Expected: "
              f"tyre scrub\n      during an on-the-spot turn eats some of the wheel motion.")
    else:
        print(f"      The robot over-turns by {100 * (1 - turn_scale):.1f}%.")

    print(f"\n  Implied effective wheel separation: {effective_separation:.4f} m")
    print(f"  (baseline.txt holds {rig.wheel_separation:.4f} m. The gap between them IS the")
    print(f"   scrub -- the robot turns as if its wheels were that far apart. Leave")
    print(f"   baseline.txt alone: the EKF's motion model has used it since M1.)")

    print("\n  Per-trial ('residual' = turned further than the fit predicts, relative to")
    print("  the turn's own direction, so + is always 'went too far'):")
    print("    requested   ticks   believed   actual    residual")
    for t, r in zip(trials, residuals_deg(trials, k_all)):
        print(f"    {t['requested_deg']:+8.1f}  {t['ticks_signed']:+6.0f}  "
              f"{t['believed_deg']:+9.2f}  {t['actual_deg']:+8.2f}  {r:+9.2f}")

    res = residuals_deg(trials, k_all)
    rms = math.sqrt(sum(r ** 2 for r in res) / len(res))
    print(f"\n  RMS residual about the fit: {rms:.2f} deg")
    print("  (This is the repeatability, NOT the accuracy turn_scale fixes. It's what")
    print("   is left over once the scale is right -- slip that varies run to run.")
    print("   drive_to_point() re-checks the heading after turning and corrects up to")
    print("   twice, so a couple of degrees here is fine; ten is not.)")

    _report_direction_split(trials, k_nominal)
    _report_resolution(k_all)


def _report_direction_split(trials, k_nominal):
    """Fit left and right turns separately. A single turn_scale cannot correct
    a robot that turns differently each way, so it's worth knowing."""
    left = [t for t in trials if t['ticks_signed'] > 0]
    right = [t for t in trials if t['ticks_signed'] < 0]
    if not left or not right:
        return
    k_left = fit_ticks_per_radian(left)
    k_right = fit_ticks_per_radian(right)
    if k_left is None or k_right is None:
        return

    print(f"\n  By direction:")
    print(f"    left  (anticlockwise): turn_scale = {k_left / k_nominal:.4f}  ({len(left)} trials)")
    print(f"    right (clockwise):     turn_scale = {k_right / k_nominal:.4f}  ({len(right)} trials)")
    asymmetry = abs(k_left - k_right) / ((k_left + k_right) / 2)
    if asymmetry > 0.05:
        print(f"    WARNING: the two directions differ by {100 * asymmetry:.1f}%.")
        print("    One turn_scale splits that difference, leaving roughly half the error")
        print("    in each direction. If that's too much, give Navigator.turn() a")
        print("    direction-dependent scale:")
        print("        scale = self.turn_scale_left if dtheta > 0 else self.turn_scale_right")
        print("    Before doing that, check the mechanical causes first -- unequal tyre")
        print("    grip, a dragging castor, or a battery low enough that one motor")
        print("    reaches its PID limit and the other doesn't.")
    else:
        print(f"    Difference is {100 * asymmetry:.1f}% -- small enough for one shared scale.")


def _report_resolution(k):
    """The floor no amount of calibration gets under: turns are quantised to
    whole encoder ticks."""
    deg_per_tick = math.degrees(1.0 / k)
    print(f"\n  Quantisation floor: 1 tick = {deg_per_tick:.2f} deg, so every commanded")
    print(f"  turn rounds to a multiple of that -- worst case {deg_per_tick / 2:.2f} deg of error")
    print("  that calibration cannot remove. Expected error after calibration:")
    print("    requested   ticks   achievable   error")
    for requested in (15, 30, 45, 60, 90, 120, 180):
        ticks = int(round(math.radians(requested) * k))
        achievable = math.degrees(ticks / k) if ticks else 0.0
        print(f"    {requested:8.0f}   {ticks:5d}   {achievable:10.2f}   {achievable - requested:+6.2f}")
    if deg_per_tick > 4.0:
        print(f"\n  NOTE: {deg_per_tick:.1f} deg per tick is coarse next to drive_to_point()'s")
        print("  5 deg heading_tolerance -- a single tick can overshoot the tolerance band,")
        print("  so the heading-correction loop may sit there flipping between +1 and -1")
        print("  ticks. Consider raising heading_tolerance to just above one tick.")


def save_results(turn_scale, trials, rig, k_all, log_name):
    """Write turn_scale.txt next to the other calibration parameters, in the
    same one-number np.savetxt format wheel_calibration.py uses, plus a full
    JSON log of the run for the report."""
    os.makedirs(PARAM_DIR, exist_ok=True)
    np.savetxt(TURN_SCALE_FILE, np.array([turn_scale]), delimiter=',')
    print(f"\nSaved turn_scale = {turn_scale:.4f} to {TURN_SCALE_FILE}")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_name)
    with open(log_path, 'w') as f:
        json.dump({'turn_scale': turn_scale,
                   'ticks_per_radian_measured': k_all,
                   'ticks_per_radian_geometric': rig.nominal_ticks_per_radian(),
                   'ticks_per_meter': rig.ticks_per_meter,
                   'wheel_separation': rig.wheel_separation,
                   'effective_wheel_separation': 2.0 * k_all / rig.ticks_per_meter,
                   'turn_speed': rig.turn_speed,
                   'pid_gains': PID_GAINS,
                   'trials': trials}, f, indent=2)
    print(f"Saved the full run log to {log_path}")

    print("\nTo use it, pass it to Navigator in auto_fruit_search.py's __main__:")
    print("    turn_scale = float(np.loadtxt('calibration/param/turn_scale.txt', delimiter=','))")
    print("    nav = Navigator(botconnect, ekf, aruco_sensor, baseline, turn_scale=turn_scale)")


def print_nominal_table(rig):
    """What the current, uncalibrated geometry sends for each angle. No robot
    needed -- run with --dry-run to see whether the resolution is even usable
    before booking bench time."""
    k = rig.nominal_ticks_per_radian()
    print(f"\nCurrent geometry: wheel_separation = {rig.wheel_separation:.4f} m, "
          f"ticks_per_meter = {rig.ticks_per_meter}")
    print(f"  -> {k:.3f} ticks per radian ({math.radians(1) * k:.4f} ticks per degree)")
    print(f"  -> 1 tick = {math.degrees(1.0 / k):.2f} deg, the smallest turn possible")
    print("\nWhat Navigator.turn() sends today (turn_scale = 1.0):")
    print("    requested   ticks   believed")
    for requested in (5, 15, 30, 45, 60, 90, 120, 180, 360):
        ticks = rig.ticks_for(math.radians(requested), 1.0)
        print(f"    {requested:8.0f}   {ticks:5d}   {math.degrees(rig.angle_for(ticks, 1.0)) if ticks else 0.0:8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate turn_scale so commanded turns match real ones")
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--calib-dir", type=str, default='calibration/param/')
    parser.add_argument("--map", type=str, default='truemap.txt',
                        help="true map, for automatic heading measurement from the markers")
    parser.add_argument("--measure", choices=['aruco', 'manual'], default='aruco',
                        help="aruco: heading solved from the markers (needs --map). "
                             "manual: you measure the angle and type it in.")
    parser.add_argument("--trials", type=str, default=None,
                        help="comma-separated signed angles in degrees, e.g. '90,-90,135,-135'. "
                             "Default: " + ",".join(str(t) for t in DEFAULT_TRIALS))
    parser.add_argument("--turn-scale", type=float, default=1.0,
                        help="scale to drive the trials at. Leave at 1.0 for a fresh "
                             "calibration; set it to a previous result to VERIFY that result "
                             "(a good one gives a new scale near 1.0).")
    parser.add_argument("--no-pid", action="store_true",
                        help="don't enable the robot's PID speed control (M3 runs with it on)")
    parser.add_argument("--no-save", action="store_true",
                        help="print the result but don't write turn_scale.txt")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the tick maths and exit, without connecting to a robot")
    args, _ = parser.parse_known_args()

    trial_angles = DEFAULT_TRIALS
    if args.trials:
        try:
            trial_angles = [float(a) for a in args.trials.split(',') if a.strip()]
        except ValueError:
            sys.exit("--trials must be comma-separated numbers, e.g. '90,-90,135,-135'")
        if not trial_angles:
            sys.exit("--trials was empty")

    baseline = np.loadtxt(os.path.join(args.calib_dir, 'baseline.txt'), delimiter=',')

    if args.dry_run:
        print_nominal_table(TurnRig(None, baseline))
        print("\n(dry run -- no robot contacted, nothing measured, nothing saved)")
        sys.exit(0)

    botconnect = BotConnect(args.ip)
    time.sleep(1)  # give connection threads a moment to establish
    if not args.no_pid:
        botconnect.set_pid(use_pid=1, **PID_GAINS)

    rig = TurnRig(botconnect, baseline)
    print_nominal_table(rig)

    heading_source = None
    if args.measure == 'aruco':
        heading_source = ArucoHeading(botconnect, args.map, args.calib_dir)
        noise = heading_source.noise_check()
        if noise is None:
            sys.exit("Could not solve a heading at all -- fewer than 2 known markers are in "
                     "view. Reposition the robot and try again, or use --measure manual.")
        print(f"Heading measurement noise while stationary: {math.degrees(noise):.2f} deg "
              f"(1 sigma)")
        if math.degrees(noise) > 2.0:
            print("  WARNING: that's large. Trials will be measuring marker noise as much as")
            print("  turn error. Move the robot somewhere with more markers in clear view,")
            print("  check the lighting, and make sure the camera is in focus.")
    else:
        print("\nManual measurement. Put a protractor or a marked circle under the robot and")
        print("line up a reference point on the chassis before each trial.")
        print("Tip: for the most precise result, use a few whole revolutions per trial")
        print("(--trials 1080,-1080) -- the tick quantisation error is then a tiny fraction")
        print("of the total angle, and you only need to read off how far past or short of")
        print("the mark it finished.")

    print(f"\nRunning {len(trial_angles)} trials at turn_scale = {args.turn_scale}, "
          f"turn speed {TURN_SPEED}.")
    print("Keep hands clear; the robot will turn on the spot.")
    input("Press ENTER to start...")

    trials = []
    try:
        for i, angle in enumerate(trial_angles, start=1):
            print(f"\n--- trial {i}/{len(trial_angles)} ---", end="")
            trial = measure_trial(rig, angle, args.turn_scale, heading_source)
            if trial is not None and not math.isnan(trial['actual_rad']):
                trials.append(trial)
    except KeyboardInterrupt:
        print("\nInterrupted -- fitting whatever was collected so far.")
        botconnect.stop()

    if len(trials) < 2:
        sys.exit(f"\nOnly {len(trials)} usable trial(s) -- need at least 2 to fit a scale.")

    k_all = fit_ticks_per_radian(trials)
    if k_all is None or k_all <= 0:
        sys.exit("\nThe fit failed (the measured angles don't agree with the ticks sent at all). "
                 "Check that the measured angles are signed the same way as the requests: "
                 "positive is anticlockwise.")

    report(trials, rig, k_all)

    # Trials driven at a non-unity scale measure the residual error on top of
    # that scale, so the scale to save is the product, not the new fit alone.
    final_scale = (k_all / rig.nominal_ticks_per_radian()) * args.turn_scale
    if args.turn_scale != 1.0:
        print(f"\n  Trials ran at turn_scale = {args.turn_scale}, so the scale to use from "
              f"here is\n  {k_all / rig.nominal_ticks_per_radian():.4f} x {args.turn_scale} "
              f"= {final_scale:.4f}.")

    if args.no_save:
        print("\n(--no-save: nothing written)")
    else:
        save_results(final_scale, trials, rig, k_all,
                     f"turn_calibration_{int(time.time())}.json")

    print("\nNext: re-run with --turn-scale {:.4f} to verify. A good calibration comes back "
          "with a\nnew scale within a couple of percent of 1.0.".format(final_scale))
