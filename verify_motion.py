# Check the calibrated motion values against a tape measure / protractor.
#
# calibrate_encoder.py goes "send N ticks, tell me what happened" and FITS the
# numbers. This script goes the other way, the way auto_fruit_search.py uses
# them: "drive 0.5 m" / "turn 90 deg" -> the SAME tick maths and wheel speeds
# as Navigator.drive_forward() / Navigator.turn() -> you measure what the robot
# actually did -> the error, and what the calibrated value should be.
#
# Values used (printed at start):
#   ticks_per_meter  read from auto_fruit_search.py's init_ekf() (or --tpm)
#   turn_scale       calibration/param/turn_scale.txt       turns <= 20 deg, at 0.25  (or --turn-scale)
#   turn_scale_fast  calibration/param/turn_scale_fast.txt  bigger turns, at 0.35    (or --turn-scale-fast)
#   wheel separation |calibration/param/baseline.txt|
#   speeds           drive 0.4; turns as above (same as M3)
#
# Usage:
#   python verify_motion.py --ip <robot_ip>
# then at the prompt:
#   d 0.5      drive forward 0.5 m        d -0.3   backward 0.3 m
#   t 90       turn left 90 deg           t -45    turn right 45 deg
#   (blank)    repeat the last command
#   sum        summary so far: mean error, and the value that would fix it
#   undo       drop the last saved trial
#   q          quit (prints the summary)
# After each move type what you measured (m or deg; + = forward / left),
# or press Enter to skip it.
#
# Measuring tips: distance -- mark the floor under one fixed point on the
# chassis before and after. Angle -- tape a ruler to the robot pointing
# forward and read its angle on a protractor or floor marks; small turns are
# much easier to read at the end of a long pointer. Do several of each size:
# 1-3 tick turns (5-20 deg) are what the robot makes before most drives.

import os
import re
import sys
import math
import time
import argparse

import numpy as np

from botconnect import BotConnect

PID_GAINS = {'kp': 2, 'ki': 0.04, 'kd': 0.29}   # auto_fruit_search.py __main__
DRIVE_SPEED = 0.4                               # Navigator drive_speed
TURN_SPEED = 0.35                               # Navigator turn_speed
SMALL_TURN_SPEED = 0.25                         # Navigator small_turn_speed
SMALL_TURN_MAX_DEG = 25.0                       # Navigator small_turn_max_deg
MOVE_TIMEOUT = 15.0
ROLL_WATCH = 1.0                                # s of watching the counts after the Pi says done


def tpm_from_auto_fruit_search(path='auto_fruit_search.py'):
    """ticks_per_meter as auto_fruit_search.py's init_ekf() builds the Robot."""
    try:
        src = open(path, encoding='utf-8').read()
        m = re.search(r"def init_ekf.*?ticks_per_meter\s*=\s*([0-9.]+)", src, re.S)
        return float(m.group(1)) if m else None
    except OSError:
        return None


def wait_and_count(bot, before):
    """Wait for the move to finish; ticks counted when the Pi stopped and after the roll."""
    start = time.time()
    completed = True
    while not bot.autonomous_done:
        if time.time() - start > MOVE_TIMEOUT:
            print("  WARNING: move timed out -- stopping")
            bot.stop()
            completed = False
            break
        time.sleep(0.002)
    stop_l, stop_r = bot.get_encoder_counts()
    last = (stop_l, stop_r)
    t_end = time.time() + ROLL_WATCH
    went_down = False
    while time.time() < t_end:
        c = bot.get_encoder_counts()
        if c[0] < last[0] or c[1] < last[1]:
            went_down = True
        last = c
        time.sleep(0.002)
    return (completed, (stop_l - before[0], stop_r - before[1]),
            (last[0] - before[0], last[1] - before[1]), went_down)


def ask(prompt):
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("  not a number -- skipped")
        return None


def summary(trials, tpm, sep, ts, ts_fast):
    if not trials:
        print("  no trials saved yet")
        return
    groups = [('d', None, "DRIVES"), ('t', False, "SLOW TURNS (<= {:.0f} deg, {}, turn_scale.txt)".format(
        SMALL_TURN_MAX_DEG, SMALL_TURN_SPEED)), ('t', True, "FAST TURNS (> {:.0f} deg, {}, turn_scale_fast.txt)".format(
        SMALL_TURN_MAX_DEG, TURN_SPEED))]
    for kind, fast, title in groups:
        rows = [t for t in trials if t['kind'] == kind and (fast is None or t['fast'] == fast)]
        if not rows:
            continue
        unit, conv = ("m", 1.0) if kind == 'd' else ("deg", 180 / math.pi)
        cur_ts = ts_fast if fast else ts
        print("\n  {}: {} trial(s)".format(title, len(rows)))
        print("    asked     ticks   believed   measured   error")
        for t in rows:
            print("    {:+8.3f}  {:5d}   {:+8.3f}   {:+8.3f}   {:+6.1f}%".format(
                t['asked'] * conv, t['ticks'], t['believed'] * conv, t['measured'] * conv,
                100 * (t['measured'] / t['believed'] - 1)))
        ratio = np.array([t['measured'] / t['believed'] for t in rows])
        print("    measured / believed: mean {:.3f}, spread +/-{:.3f}".format(ratio.mean(), ratio.std()))
        n = np.array([t['ticks'] for t in rows], float)
        m = np.array([abs(t['measured']) for t in rows], float)
        if kind == 'd':
            # least squares through the origin of ticks = k * distance
            k = float(np.sum(n * m) / np.sum(m * m))
            print("    -> ticks_per_meter that fits these: {:.1f} (now {:.1f})".format(k, tpm))
        else:
            s0 = float(np.sum(n * m) / np.sum(n * n))   # rad per tick, coast included
            new_ts = 2.0 / (sep * tpm * s0)
            print("    -> {:.2f} deg per tick incl. coast; {} that fits these: {:.4f} (now {:.4f})".format(
                math.degrees(s0), "turn_scale_fast" if fast else "turn_scale", new_ts, cur_ts))
            by = {}
            for t in rows:
                by.setdefault(t['ticks'], []).append(abs(t['measured']))
            print("    per tick count: " + ", ".join("{} tick(s) {:.1f} deg (+/-{:.1f}, n={})".format(
                k_, math.degrees(np.mean(v)), math.degrees(np.std(v)), len(v)) for k_, v in sorted(by.items())))
        print("    within +/-2%: good.  A mean off by more than that, the same way every time: "
              "the value needs changing.  A big spread: slip/coast varies -- more trials, or a slower speed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default='localhost')
    ap.add_argument("--calib-dir", default='calibration/param/')
    ap.add_argument("--tpm", type=float, default=None,
                    help="ticks_per_meter (default: read from auto_fruit_search.py)")
    ap.add_argument("--turn-scale", type=float, default=None,
                    help="slow-turn scale (default: calibration/param/turn_scale.txt)")
    ap.add_argument("--turn-scale-fast", type=float, default=None,
                    help="fast-turn scale (default: calibration/param/turn_scale_fast.txt, else the slow one)")
    ap.add_argument("--no-pid", action="store_true")
    args = ap.parse_args()

    tpm = args.tpm or tpm_from_auto_fruit_search() or 193.3
    sep = abs(float(np.loadtxt(os.path.join(args.calib_dir, 'baseline.txt'), delimiter=',')))
    if args.turn_scale is not None:
        ts = args.turn_scale
    else:
        try:
            ts = float(np.loadtxt(os.path.join(args.calib_dir, 'turn_scale.txt'), delimiter=','))
        except Exception:
            ts = 1.0
    if args.turn_scale_fast is not None:
        ts_fast = args.turn_scale_fast
    else:
        try:
            ts_fast = float(np.loadtxt(os.path.join(args.calib_dir, 'turn_scale_fast.txt'), delimiter=','))
        except Exception:
            ts_fast = ts
            print("(no turn_scale_fast.txt -- fast turns use the slow scale too)")
    tick_deg = math.degrees(2.0 / (tpm * sep * ts))
    tick_deg_fast = math.degrees(2.0 / (tpm * sep * ts_fast))
    print("ticks_per_meter {:.1f}  ->  1 tick = {:.1f} mm".format(tpm, 1000.0 / tpm))
    print("wheel separation {:.4f} m".format(sep))
    print("turns <= {:.0f} deg: speed {}, turn_scale {:.4f}  ->  1 tick = {:.2f} deg".format(
        SMALL_TURN_MAX_DEG, SMALL_TURN_SPEED, ts, tick_deg))
    print("bigger turns:     speed {}, turn_scale_fast {:.4f}  ->  1 tick = {:.2f} deg".format(
        TURN_SPEED, ts_fast, tick_deg_fast))

    bot = BotConnect(args.ip)
    time.sleep(1)
    if not args.no_pid:
        bot.set_pid(use_pid=1, **PID_GAINS)
    print("\n'd 0.5' drive, 't 90' turn, blank = repeat, 'sum', 'undo', 'q'.\n")

    trials, last_cmd = [], None
    while True:
        raw = input("> ").strip().lower()
        if raw == '' and last_cmd:
            raw = last_cmd
            print("  (repeating: {})".format(raw))
        if raw in ('q', 'quit', 'exit'):
            break
        if raw == 'sum':
            summary(trials, tpm, sep, ts, ts_fast)
            continue
        if raw == 'undo':
            print("  removed" if trials and trials.pop() else "  nothing to undo")
            continue
        parts = raw.split()
        if len(parts) != 2 or parts[0] not in ('d', 't'):
            print("  'd <metres>' or 't <degrees>', or sum / undo / q")
            continue
        try:
            val = float(parts[1])
        except ValueError:
            print("  not a number")
            continue
        last_cmd = raw
        kind = parts[0]

        fast = None
        if kind == 'd':
            # Navigator.drive_forward / drive_backward
            ticks = int(round(abs(val) * tpm))
            if ticks < 1:
                print("  under half a tick -- the robot would not move")
                continue
            believed = math.copysign(ticks / tpm, val)
            speeds = [DRIVE_SPEED, DRIVE_SPEED] if val > 0 else [-DRIVE_SPEED, -DRIVE_SPEED]
            print("  {} ticks -> the EKF would be told {:+.3f} m".format(ticks, believed))
            asked = val
        else:
            # Navigator.turn
            dtheta = math.radians(val)
            fast = abs(val) > SMALL_TURN_MAX_DEG + 1e-6
            scale = ts_fast if fast else ts
            ticks = int(round((sep / 2.0) * abs(dtheta) * tpm * scale))
            if ticks < 1:
                print("  under half a tick ({:.1f} deg) -- turn() would skip it".format(tick_deg / 2))
                continue
            believed = math.copysign(2.0 * ticks / (tpm * sep * scale), dtheta)
            mag = TURN_SPEED if fast else SMALL_TURN_SPEED
            speeds = [-mag, mag] if val > 0 else [mag, -mag]
            print("  {} ticks at speed {} -> the EKF would be told {:+.1f} deg".format(ticks, mag, math.degrees(believed)))
            asked = dtheta

        before = bot.get_encoder_counts()
        bot.move_auto_encoder(speeds, ticks, ticks)
        completed, (sl, sr), (al, ar), went_down = wait_and_count(bot, before)
        print("  counted when the Pi stopped: L {} R {}; after {:.0f} s: L {} R {} (rolled on +{} / +{})".format(
            sl, sr, ROLL_WATCH, al, ar, al - sl, ar - sr))
        if went_down:
            print("  WARNING: counts went DOWN -- the old resetting listen.py is running; tick numbers invalid")
        if not completed:
            print("  timed out -- not saved")
            continue

        meas = ask("  measured {} (Enter = skip): ".format("distance, m" if kind == 'd' else "angle, deg"))
        if meas is None:
            continue
        if kind == 't':
            meas = math.radians(meas)
        if meas * believed < 0:
            print("  opposite sign to the command -- check the sign (+ = forward / left); not saved")
            continue
        err = 100 * (meas / believed - 1)
        print("  error {:+.1f}% ({:+.1f} {})".format(
            err, (meas - believed) * (1 if kind == 'd' else 180 / math.pi) * (100 if kind == 'd' else 1),
            "cm" if kind == 'd' else "deg"))
        trials.append({'kind': kind, 'asked': asked, 'ticks': ticks, 'believed': believed, 'measured': meas,
                       'fast': fast})

    bot.stop()
    print("\nSummary:")
    summary(trials, tpm, sep, ts, ts_fast)


if __name__ == "__main__":
    main()
