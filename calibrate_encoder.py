# Encoder / turn calibration for the PiBot.
#
# WHY THIS VERSION
# The old script drove with a stream of move_manual() commands for a fixed
# time. auto_fruit_search.py (M3) drives with ONE move_auto_encoder() command
# that the Pi runs until it has counted N ticks. Those are different code
# paths on the Pi, so a number calibrated one way isn't guaranteed to hold the
# other way. The default here is therefore 'auto' mode: exactly the call M3
# makes, with the same speeds and PID gains.
#
# WHAT IT MEASURES
# Every move overshoots its tick target by a roughly FIXED amount (the robot
# coasts after the Pi cuts the motors). So for a commanded N ticks:
#
#     straight:  d     = N / k + c          k = ticks per metre, c = coast (m)
#     turning:   theta = N * s + theta0     s = rad per tick,   theta0 = coast (rad)
#
# A single trial can't separate k from c (N/d mixes them). Several trials at
# different N and a straight-line fit can: the slope gives k (or s), the
# intercept gives the coast. Run at least 3 different N per mode, e.g.
# 'f 50', 'f 100', 'f 200', then type 'fit'.
#
# For turns, the fit also gives turn_scale in the form auto_fruit_search.py
# uses it. Navigator.turn() sends
#     ticks = (b/2) * |theta| * k * turn_scale
# i.e. it assumes each tick turns 2 / (b * k * turn_scale) rad. Setting that
# equal to the measured s gives
#     turn_scale = 2 / (b * k * s)
#
# 'timed' mode (--timed) keeps the old move_manual() behaviour for comparison,
# but its ramp now starts at RAMP_FLOOR instead of 0: the old ramp's first
# commands were [0, 0] and then speeds below the PWM threshold, where the weak
# left wheel could stay stalled while the right one moved.
#
# Usage:
#   python drive_timed.py --ip <robot_ip>              # auto mode (like M3)
#   python drive_timed.py --ip <robot_ip> --timed      # old timed mode
#   then enter e.g. 'f 100' (auto: 100 ticks) or 'f 2' (timed: 2 s),
#   type in what you measured, repeat, and enter 'fit'.
#   Bad trial? Press Enter at the measurement prompt to discard it, or type
#   'undo' afterwards to remove the last saved one.

import os
import time
import argparse
import numpy as np
from botconnect import BotConnect

# name, [left_speed, right_speed] -- same as operate.py's keys and M3's
# Navigator (drive_speed=0.4, turn_speed=0.35; turn left = [-0.35, 0.35]).
MODES = {
    'f': ('Forward',    [0.4, 0.4]),
    'b': ('Backward',   [-0.4, -0.4]),
    'l': ('Turn left',  [-0.35, 0.35]),
    'r': ('Turn right', [0.35, -0.35]),
}
TURN_MODES = ('l', 'r')

PID_GAINS = {'kp': 2, 'ki': 0.04, 'kd': 0.29}   # same as operate.py and M3
MOVE_TIMEOUT = 15.0                              # s, same as M3's Navigator

# Timed mode only
RAMP_DURATION = 0.15   # s, as operate.py
RAMP_FLOOR = 0.5       # ramp starts at 50% of commanded speed, not 0


# ----------------------------------------------------------------------
# Driving
# ----------------------------------------------------------------------

def drive_auto(botconnect, wheel_speed, ticks):
    """Exactly M3's move: one move_auto_encoder() call, then wait for the Pi
    to finish.

    How botconnect reports counts in this mode: its wheel thread sends the
    command and then BLOCKS in recv() until the Pi replies at the END of the
    move, so get_encoder_counts() doesn't change while the robot is moving.
    The reply carries the Pi's final counts; the thread stores them and sets
    autonomous_done. About 10 ms later it drops back to manual mode, whose
    next reply overwrites them (and the Pi zeroes its counters once stopped).
    So: poll the flag every 1 ms, read the counts the instant it flips (= what
    the Pi counted when it declared the move done), then keep watching for
    0.3 s and record the largest count seen, which also catches ticks from
    the coast after the motors were cut, if any arrive before the reset.
    @return: (completed, (done_left, done_right), (max_left, max_right))"""
    botconnect.move_auto_encoder(wheel_speed, ticks, ticks)
    start = time.time()
    completed = True
    while not botconnect.autonomous_done:
        if time.time() - start > MOVE_TIMEOUT:
            print("  WARNING: move exceeded {:.0f}s -- stopping".format(MOVE_TIMEOUT))
            botconnect.stop()
            completed = False
            break
        time.sleep(0.001)
    done_l, done_r = botconnect.get_encoder_counts()
    max_l, max_r = abs(done_l), abs(done_r)
    t_end = time.time() + 0.3
    while time.time() < t_end:
        l, r = botconnect.get_encoder_counts()
        max_l = max(max_l, abs(l))
        max_r = max(max_r, abs(r))
        time.sleep(0.001)
    return completed, (done_l, done_r), (max_l, max_r)


def drive_timed(botconnect, wheel_speed, duration):
    """Old behaviour: stream move_manual() for `duration` s, ramping from
    RAMP_FLOOR to full speed. Encoders are read BEFORE stop() because the Pi
    zeroes them when it stops. Note the coast after stop() is in your tape
    measurement but not in these ticks -- the line fit's intercept absorbs it.
    @return: (delta_left, delta_right)"""
    left_start, right_start = botconnect.get_encoder_counts()
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        scale = RAMP_FLOOR + (1.0 - RAMP_FLOOR) * min(1.0, elapsed / RAMP_DURATION)
        botconnect.move_manual([wheel_speed[0] * scale, wheel_speed[1] * scale])
        time.sleep(0.02)
    left_end, right_end = botconnect.get_encoder_counts()
    botconnect.stop()
    return left_end - left_start, right_end - right_start


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

def line_fit(x, y):
    """Least-squares y = slope * x + intercept.
    @return: (slope, intercept, rms residual)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((A @ [slope, intercept] - y) ** 2)))
    return float(slope), float(intercept), rms


def report_fits(samples, baseline, tpm_ref):
    """samples: {mode_key: [(ticks, measured), ...]}, measured in m or rad."""
    k_fit = {}
    for key in ('f', 'b'):
        pts = samples[key]
        name = MODES[key][0]
        if len({n for n, _ in pts}) < 2:
            print(f"  {name}: need trials at >= 2 different tick counts ({len(pts)} so far)")
            continue
        n, d = zip(*pts)
        slope, c, rms = line_fit(n, d)
        k = 1.0 / slope
        k_fit[key] = k
        print(f"  {name}: ticks_per_meter k = {k:.1f}, coast c = {c * 100:+.1f} cm "
              f"(rms residual {rms * 100:.1f} cm, {len(pts)} trials)")
        print(f"      naive N/d per trial: " +
              ", ".join(f"{ni / di:.1f}" for ni, di in pts if di > 0))

    if k_fit:
        k_use = float(np.mean(list(k_fit.values())))
        print(f"  -> ticks_per_meter to use: {k_use:.1f} "
              f"(currently {tpm_ref:.1f} in operate.py / M3's init_ekf)")
    else:
        k_use = tpm_ref
        print(f"  (no straight-line fit yet -- turn_scale below uses k = {k_use:.1f})")

    for key in TURN_MODES:
        pts = samples[key]
        name = MODES[key][0]
        if len({n for n, _ in pts}) < 2:
            print(f"  {name}: need trials at >= 2 different tick counts ({len(pts)} so far)")
            continue
        n, th = zip(*pts)
        s, th0, rms = line_fit(n, th)
        turn_scale = 2.0 / (baseline * k_use * s)
        print(f"  {name}: {np.degrees(s):.2f} deg/tick, coast {np.degrees(th0):+.1f} deg "
              f"(rms {np.degrees(rms):.1f} deg) -> turn_scale = {turn_scale:.3f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def load_baseline(calib_dir):
    path = os.path.join(calib_dir, 'baseline.txt')
    try:
        # stored negative (-0.1386); the tick geometry needs the physical distance
        return abs(float(np.loadtxt(path, delimiter=',')))
    except Exception as e:
        print(f"Could not read {path} ({e}) -- turn_scale won't be computed")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='localhost')
    parser.add_argument("--timed", action="store_true",
                        help="old move_manual() timed trials instead of M3-style tick trials")
    parser.add_argument("--no-pid", action="store_true",
                        help="don't enable the robot's PID speed control")
    parser.add_argument("--calib-dir", type=str, default='calibration/param/')
    parser.add_argument("--tpm", type=float, default=172.5,
                        help="current ticks_per_meter, used for turn_scale until you've "
                             "fitted your own")
    args, _ = parser.parse_known_args()

    baseline = load_baseline(args.calib_dir)

    botconnect = BotConnect(args.ip)
    time.sleep(1)
    if not args.no_pid:
        botconnect.set_pid(use_pid=1, **PID_GAINS)

    unit = "s" if args.timed else "ticks"
    print("Modes:")
    for key, (name, speed) in MODES.items():
        print(f"  {key} = {name:<10s} wheel_speed={speed}")
    print(f"\nDrive: {'TIMED move_manual (ramp from ' + str(RAMP_FLOOR) + ')' if args.timed else 'move_auto_encoder, same as M3'}")
    print(f"PID:   {'off' if args.no_pid else f'on {PID_GAINS}'}")
    print(f"\nEnter '<mode> <{unit}>', e.g. 'f {'2' if args.timed else '100'}'.")
    print("After each trial, type the distance (m) or angle (deg) you measured; blank = discard.")
    print("'fit' = fit the trials so far, 'list' = show them, 'undo' = remove the last saved trial,")
    print("'q' = quit (fits first).\n")

    samples = {key: [] for key in MODES}
    last_key = None   # mode of the most recently saved trial, for 'undo'

    while True:
        user_input = input(f"Mode {unit}: ").strip().lower()
        if user_input == 'q':
            break
        if user_input == 'fit':
            if baseline is None:
                print("  (no baseline -- turn fits will fail)")
            report_fits(samples, baseline or float('nan'), args.tpm)
            print()
            continue
        if user_input == 'undo':
            if last_key is None or not samples[last_key]:
                print("  nothing to undo\n")
            else:
                n, m = samples[last_key].pop()
                shown = f"{np.degrees(m):.1f} deg" if last_key in TURN_MODES else f"{m:.3f} m"
                print(f"  removed {MODES[last_key][0]}: {n} ticks -> {shown}\n")
                last_key = None   # one undo per trial, so a second 'undo' can't eat an older one
            continue
        if user_input == 'list':
            for key, pts in samples.items():
                if pts:
                    conv = np.degrees if key in TURN_MODES else (lambda v: v)
                    print(f"  {MODES[key][0]}: " +
                          ", ".join(f"{n} -> {conv(m):.3f}" for n, m in pts))
            print()
            continue

        parts = user_input.split()
        if len(parts) != 2 or parts[0] not in MODES:
            print(f"  Enter one of {list(MODES)} and a number, or 'fit' / 'list' / 'undo' / 'q'.")
            continue
        key = parts[0]
        try:
            amount = float(parts[1])
        except ValueError:
            print("  Please enter a number.")
            continue
        if amount <= 0:
            print("  Must be positive.")
            continue

        name, wheel_speed = MODES[key]
        if args.timed:
            dl, dr = drive_timed(botconnect, wheel_speed, amount)
            n_ticks = (abs(dl) + abs(dr)) / 2.0
            print(f"  [{name}] left {dl}, right {dr} -> using {n_ticks:.1f} ticks")
        else:
            n_ticks = int(round(amount))
            completed, (dl, dr), (ml, mr) = drive_auto(botconnect, wheel_speed, n_ticks)
            print(f"  [{name}] commanded {n_ticks} ticks; Pi reported at finish: left {dl}, right {dr};"
                  f" max within 0.3 s after: left {ml}, right {mr}"
                  + ("" if completed else "  (TIMED OUT -- discard this trial)"))
            if abs(ml - mr) > max(3, 0.1 * n_ticks):
                print("  note: wheels counted quite differently -- check the robot drove straight")

        prompt = "  measured angle (deg): " if key in TURN_MODES else "  measured distance (m): "
        meas = input(prompt).strip()
        if not meas:
            print("  discarded\n")
            continue
        try:
            value = abs(float(meas))
        except ValueError:
            print("  not a number -- discarded\n")
            continue
        if key in TURN_MODES:
            value = np.radians(value)
        elif value > 3.0:
            print(f"  {value} m is more than any trial here drives -- did you type cm? "
                  f"Discarded; re-run the trial and enter metres (e.g. 0.503).\n")
            continue
        samples[key].append((n_ticks, value))
        last_key = key
        print()

    if any(samples.values()):
        print("\nFinal fits:")
        report_fits(samples, baseline or float('nan'), args.tpm)
    print("Done.")