# Check -- and calibrate -- the robot's motion against a tape measure / protractor,
# using exactly what final_demo_l3.py does.
#
# Drives (final_demo_l3.Navigator._drive): the wheels roll on for ~14 ticks after
# the Pi stops them, whatever the drive length. So a drive is commanded that many
# ticks SHORT (the robot lands where it was asked to), and the EKF is told the
# ticks the encoders counted once the wheels stopped, divided by ticks_per_meter.
# Two numbers describe it, both in calibration/param/drive_motion.json:
#   ticks_per_meter   ticks per metre of travel, roll-on included
#   coast_ticks       roll-on after a full-speed drive (coast_ramp_ticks: shorter
#                     drives roll on less: coast * (1 - exp(-commanded / ramp)))
# Turns (Navigator.turn / _turn_plan): the wheels roll on after a turn too, a
# fixed few degrees whatever the turn's size. A turn of n ticks comes out
#   n x (one tick's angle at that speed's turn_scale) + that speed's roll-on
# so the demo picks the n whose result is nearest the request, and tells the EKF
# that result. Per speed, in calibration/param/turn_motion.json:
#   turn_scale / turn_scale_fast     ticks per degree (as turn_scale*.txt)
#   offset_deg / offset_fast_deg     the fixed roll-on, deg
# (no turn_motion.json: turn_scale*.txt and no roll-on -- the old turns)
#
# Usage:
#   python verify_motion.py --ip <robot_ip>
# then at the prompt:
#   d 30       drive 30 cm the way the demo does (early stop + counted ticks)
#   r 30       RAW drive: 30 cm x ticks_per_meter ticks, no early stop -- for calibrating
#   d -20      backward (also r -20)
#   t 90       turn left 90 deg the way the demo does (roll-on allowed for)
#   t -45      turn right 45 deg
#   tr 90      RAW turn: angle / one tick's angle, rounded -- no roll-on cut
#   (blank)    repeat the last command
#   sum        summary: errors, and the calibration that fits
#   save       write the fitted drive calibration to drive_motion.json, and the
#              fitted turn calibration to turn_motion.json (+ turn_scale*.txt)
#   undo       drop the last saved trial
#   q          quit (prints the summary)
# After each move type what you measured (cm or deg; + = forward / left), or
# Enter to skip it.
#
# To calibrate drives: 6-10 'r' drives from 10 to 70 cm, then 'sum' and 'save'.
# To calibrate turns: 2-3 of each of several sizes per speed -- slow: 5, 10, 15,
# 20 deg; fast: 30, 45, 60, 90, 120 deg -- then 'sum' and 'save'. The fit needs
# two or more different tick counts per speed. To check: 't' turns -- 'EKF told'
# should be within ~2 deg of your protractor for every size.
# To check: 'd' drives of the lengths the demo uses (5-25 cm) -- 'landed' should
# be within ~1 cm of what you asked, and 'EKF told' within ~1 cm of your tape.
#
# Measuring tips: distance -- mark the floor under one fixed point on the
# chassis before and after. Angle -- tape a ruler to the robot pointing forward
# and read its angle on floor marks; small turns are easier at the end of a long
# pointer.

import os
import json
import math
import time
import argparse

import numpy as np

from botconnect import BotConnect

PID_GAINS = {'kp': 2, 'ki': 0.04, 'kd': 0.29}   # as the demo sets them
DRIVE_SPEED = 0.4                               # Navigator drive_speed
TURN_SPEED = 0.35                               # Navigator turn_speed (--turn-speed)
SMALL_TURN_SPEED = 0.25                         # Navigator small_turn_speed (--small-turn-speed)
SMALL_TURN_MAX_DEG = 20.0                       # Navigator small_turn_max_deg (--small-turn-deg)
MOVE_TIMEOUT = 15.0
STILL_S, MAX_WAIT_S = 0.15, 1.0                 # Navigator._counted_ticks: wheels stopped = no tick for 0.15 s

DRIVE_MOTION_FILE = 'drive_motion.json'
TURN_MOTION_FILE = 'turn_motion.json'
DEFAULTS = {'ticks_per_meter': 193.3, 'coast_ticks': 14.0, 'coast_ramp_ticks': 9.0}


def load_drive_motion(calib_dir):
    out, src = dict(DEFAULTS), 'defaults'
    path = os.path.join(calib_dir, DRIVE_MOTION_FILE)
    try:
        with open(path) as f:
            out.update({k: float(v) for k, v in json.load(f).items() if k in DEFAULTS})
        src = path
    except (OSError, ValueError):
        pass
    return out, src


def load_turn_motion(calib_dir):
    path = os.path.join(calib_dir, TURN_MOTION_FILE)
    try:
        with open(path) as f:
            raw = json.load(f)
        return {k: float(raw[k]) for k in ('turn_scale', 'turn_scale_fast', 'offset_deg', 'offset_fast_deg')
                if k in raw}, path
    except (OSError, ValueError, TypeError):
        return {}, None


def tick_angle(tpm, sep, scale):
    """One tick's turn angle, rad, at that scale (Navigator.turn's geometry)."""
    return 2.0 / (tpm * sep * scale)


def turn_plan(angle, tick, offset):
    """Navigator._turn_plan: the tick count whose result (n x tick + offset, or 0
    for n = 0) is nearest |angle|. @return: (ticks, rotation the EKF is told, rad)"""
    want = abs(angle)
    n = max(1, int(round((want - offset) / tick)))
    best = (0, 0.0)
    for k in (n - 1, n, n + 1):
        if k >= 1 and abs(k * tick + offset - want) < abs(best[1] - want):
            best = (k, k * tick + offset)
    return best


def coast(n, c_full, ramp):
    return c_full * (1.0 - math.exp(-max(n, 0.0) / max(ramp, 1e-6)))


def drive_ticks(distance, tpm, c_full, ramp):
    """Navigator._drive_ticks: ticks to command so that commanded + roll-on = distance."""
    want = abs(distance) * tpm
    if want < 1.0:
        return 0
    lo, hi = 0.0, want
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mid + coast(mid, c_full, ramp) < want:
            lo = mid
        else:
            hi = mid
    return max(1, int(round(0.5 * (lo + hi))))


def wait_and_count(bot, before):
    """Wait for the move; ticks when the Pi stopped, and once the wheels have stopped."""
    start = time.time()
    completed = True
    while not bot.autonomous_done:
        if time.time() - start > MOVE_TIMEOUT:
            print("  WARNING: move timed out -- stopping")
            bot.stop()
            completed = False
            break
        time.sleep(0.002)
    at_stop = bot.get_encoder_counts()
    last, waited, quiet, went_down = tuple(at_stop), 0.0, 0.0, False
    while waited < MAX_WAIT_S:
        time.sleep(0.02)
        waited += 0.02
        c = tuple(bot.get_encoder_counts())
        if c[0] < last[0] or c[1] < last[1]:
            went_down = True
        if c != last:
            last, quiet = c, 0.0
        else:
            quiet += 0.02
            if quiet >= STILL_S:
                break
    return (completed, (at_stop[0] - before[0], at_stop[1] - before[1]),
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


def fit_drives(rows):
    """ticks_per_meter (counted ticks vs tape, through the origin) and the roll-on
    model (c_full, ramp) from (commanded, counted) pairs."""
    n_cnt = np.array([t['counted'] for t in rows], float)
    m = np.array([abs(t['measured']) for t in rows], float)
    tpm = float(np.sum(n_cnt * m) / np.sum(m * m))
    cmd = np.array([t['ticks'] for t in rows], float)
    roll = n_cnt - cmd
    best = None
    for ramp in np.arange(1.0, 30.01, 0.5):
        g = 1.0 - np.exp(-cmd / ramp)
        c = float(np.sum(g * roll) / max(np.sum(g * g), 1e-9))
        sse = float(np.sum((roll - c * g) ** 2))
        if best is None or sse < best[0]:
            best = (sse, c, float(ramp))
    resid = np.sqrt(best[0] / len(rows))
    return tpm, best[1], best[2], resid


def summary(trials, cal, sep, turns):
    if not trials:
        print("  no trials saved yet")
        return None, {}
    tpm = cal['ticks_per_meter']
    fitted = None
    turn_fit = {}
    drives = [t for t in trials if t['kind'] in ('d', 'r')]
    if drives:
        print("\n  DRIVES: {} trial(s)   (ticks_per_meter {:.1f}, roll-on {:.1f} ticks, ramp {:.1f})".format(
            len(drives), tpm, cal['coast_ticks'], cal['coast_ramp_ticks']))
        print("    type  asked(cm)  cmd  counted  rolled  EKF told(cm)  tape(cm)  told-tape  landed-asked")
        for t in drives:
            told = 100 * t['counted'] / tpm
            print("     {}   {:+7.1f}   {:4d}   {:5d}   {:+4d}     {:+7.1f}     {:+7.1f}    {:+5.1f}      {:+5.1f}".format(
                t['kind'], 100 * t['asked'], t['ticks'], t['counted'], t['counted'] - t['ticks'],
                math.copysign(told, t['asked']), 100 * t['measured'], told - 100 * abs(t['measured']),
                100 * (abs(t['measured']) - abs(t['asked']))))
        told_err = np.array([100 * (t['counted'] / tpm - abs(t['measured'])) for t in drives])
        print("    EKF told vs tape: mean {:+.1f} cm, spread +/-{:.1f} cm".format(told_err.mean(), told_err.std()))
        dd = [t for t in drives if t['kind'] == 'd']
        if dd:
            land = np.array([100 * (abs(t['measured']) - abs(t['asked'])) for t in dd])
            print("    'd' drives landed vs asked: mean {:+.1f} cm, spread +/-{:.1f} cm".format(land.mean(), land.std()))
        if len(drives) >= 4:
            f_tpm, f_c, f_ramp, resid = fit_drives(drives)
            fitted = {'ticks_per_meter': round(f_tpm, 1), 'coast_ticks': round(f_c, 1),
                      'coast_ramp_ticks': round(f_ramp, 1)}
            print("    -> fits these: ticks_per_meter {:.1f} (now {:.1f}); roll-on {:.1f} ticks (now {:.1f}), "
                  "ramp {:.1f} (now {:.1f}), roll-on model off by +/-{:.1f} ticks".format(
                      f_tpm, tpm, f_c, cal['coast_ticks'], f_ramp, cal['coast_ramp_ticks'], resid))
            print("    'save' writes these to {}".format(DRIVE_MOTION_FILE))
        else:
            print("    (4+ drives with a tape reading to fit the calibration)")
        print("    good: EKF told vs tape within ~1 cm with a small spread; 'd' drives landing within ~1 cm.")
    for fast in (False, True):
        rows = [t for t in trials if t['kind'] in ('t', 'tr') and t['fast'] == fast]
        if not rows:
            continue
        key_s, key_o = ('turn_scale_fast', 'offset_fast_deg') if fast else ('turn_scale', 'offset_deg')
        scale, off = turns[key_s], math.radians(turns[key_o])
        tick = tick_angle(tpm, sep, scale)
        print("\n  {} TURNS ({} {:.0f} deg, speed {}): {} trial(s)   (now: {:.2f} deg per tick + {:.1f} deg roll-on)".format(
            "FAST" if fast else "SLOW", ">" if fast else "<=", SMALL_TURN_MAX_DEG,
            TURN_SPEED if fast else SMALL_TURN_SPEED, len(rows), math.degrees(tick), math.degrees(off)))
        print("    type  asked  ticks  rolled  EKF told  measured  told-meas  landed-asked")
        for t in rows:
            print("     {:<2s} {:+6.1f}  {:4d}   {:+4d}    {:+7.1f}   {:+7.1f}    {:+5.1f}      {:+5.1f}".format(
                t['kind'], math.degrees(t['asked']), t['ticks'], t['counted'] - t['ticks'], math.degrees(t['believed']),
                math.degrees(t['measured']), math.degrees(abs(t['believed']) - abs(t['measured'])),
                math.degrees(abs(t['measured']) - abs(t['asked']))))
        n = np.array([t['ticks'] for t in rows], float)
        m = np.array([abs(t['measured']) for t in rows], float)
        now_err = np.array([abs(t['believed']) for t in rows]) - m
        print("    EKF told vs measured, as the trials ran: mean {:+.1f} deg, spread +/-{:.1f} deg".format(
            math.degrees(now_err.mean()), math.degrees(now_err.std())))
        by = {}
        for t in rows:
            by.setdefault(t['ticks'], []).append(abs(t['measured']))
        print("    repeatability, same tick count: " + ", ".join("{} tick(s) {:.1f} deg (+/-{:.1f}, n={})".format(
            k_, math.degrees(np.mean(v)), math.degrees(np.std(v)), len(v)) for k_, v in sorted(by.items())))
        if len(rows) >= 4 and len(by) >= 2:
            a_, b_ = np.linalg.lstsq(np.vstack([n, np.ones_like(n)]).T, m, rcond=None)[0]
            b_ = max(0.0, float(b_))
            if b_ == 0.0:
                a_ = float(np.sum(n * m) / np.sum(n * n))
            resid = m - (a_ * n + b_)
            new_scale = 2.0 / (tpm * sep * a_)
            turn_fit[key_s] = round(float(new_scale), 4)
            turn_fit[key_o] = round(float(math.degrees(b_)), 2)
            print("    -> fits these: {:.2f} deg per tick + {:.1f} deg roll-on = {} {:.4f} (now {:.4f}), {} {:.1f} deg "
                  "(now {:.1f}); left over +/-{:.1f} deg".format(
                      math.degrees(a_), math.degrees(b_), key_s, new_scale, scale, key_o, math.degrees(b_),
                      math.degrees(off), math.degrees(resid.std())))
            print("    'save' writes these to {} (and {}.txt)".format(TURN_MOTION_FILE, key_s))
        else:
            print("    (4+ trials over 2+ different tick counts to fit this speed)")
        print("    good: EKF told within ~2 deg of measured for every size, with a small spread.")
    return fitted, turn_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default='localhost')
    ap.add_argument("--calib-dir", default='calibration/param/')
    ap.add_argument("--tpm", type=float, default=None, help="override ticks_per_meter")
    ap.add_argument("--coast-ticks", type=float, default=None, help="override the roll-on, ticks")
    ap.add_argument("--turn-scale", type=float, default=None)
    ap.add_argument("--turn-scale-fast", type=float, default=None)
    ap.add_argument("--turn-offset", type=float, default=None, metavar="DEG", help="override the slow roll-on, deg")
    ap.add_argument("--turn-offset-fast", type=float, default=None, metavar="DEG", help="override the fast roll-on, deg")
    ap.add_argument("--small-turn-deg", type=float, default=SMALL_TURN_MAX_DEG,
                    help="turns up to this go at the slow speed -- use the same value as the demo (default 20)")
    ap.add_argument("--no-pid", action="store_true")
    args = ap.parse_args()

    cal, src = load_drive_motion(args.calib_dir)
    if args.tpm is not None:
        cal['ticks_per_meter'] = args.tpm
    if args.coast_ticks is not None:
        cal['coast_ticks'] = args.coast_ticks
    sep = abs(float(np.loadtxt(os.path.join(args.calib_dir, 'baseline.txt'), delimiter=',')))

    def load_scale(name, fallback):
        try:
            return float(np.loadtxt(os.path.join(args.calib_dir, name), delimiter=','))
        except Exception:
            return fallback
    globals()['SMALL_TURN_MAX_DEG'] = args.small_turn_deg   # summary() and the turn speed choice read it
    tm, tm_src = load_turn_motion(args.calib_dir)
    turns = {
        'turn_scale': (args.turn_scale if args.turn_scale is not None else
                       tm.get('turn_scale', load_scale('turn_scale.txt', 1.0))),
        'offset_deg': args.turn_offset if args.turn_offset is not None else tm.get('offset_deg', 0.0)}
    turns['turn_scale_fast'] = (args.turn_scale_fast if args.turn_scale_fast is not None else
                                tm.get('turn_scale_fast', load_scale('turn_scale_fast.txt', turns['turn_scale'])))
    turns['offset_fast_deg'] = args.turn_offset_fast if args.turn_offset_fast is not None else tm.get('offset_fast_deg', 0.0)

    tpm = cal['ticks_per_meter']
    print("drives: ticks_per_meter {:.1f} (1 tick = {:.1f} mm), roll-on {:.1f} ticks, ramp {:.1f}  [{}]".format(
        tpm, 1000.0 / tpm, cal['coast_ticks'], cal['coast_ramp_ticks'], src))
    print("wheel separation {:.4f} m".format(sep))
    print("turns <= {:.0f} deg: speed {}, turn_scale {:.4f} -> 1 tick = {:.2f} deg, roll-on {:.1f} deg".format(
        args.small_turn_deg, SMALL_TURN_SPEED, turns['turn_scale'],
        math.degrees(tick_angle(tpm, sep, turns['turn_scale'])), turns['offset_deg']))
    print("bigger turns:     speed {}, turn_scale_fast {:.4f} -> 1 tick = {:.2f} deg, roll-on {:.1f} deg".format(
        TURN_SPEED, turns['turn_scale_fast'], math.degrees(tick_angle(tpm, sep, turns['turn_scale_fast'])),
        turns['offset_fast_deg']))
    print("  [turns from {}]".format(tm_src or "turn_scale.txt / turn_scale_fast.txt -- no turn_motion.json, no roll-on"))

    bot = BotConnect(args.ip)
    time.sleep(1)
    if not args.no_pid:
        bot.set_pid(use_pid=1, **PID_GAINS)
    print("\n'd 20' demo drive, 'r 20' raw drive, 't 90' demo turn, 'tr 90' raw turn, blank = repeat, 'sum', 'save', "
          "'undo', 'q'.\n")

    trials, last_cmd, fitted, turn_fit = [], None, None, {}
    while True:
        raw = input("> ").strip().lower()
        if raw == '' and last_cmd:
            raw = last_cmd
            print("  (repeating: {})".format(raw))
        if raw in ('q', 'quit', 'exit'):
            break
        if raw in ('sum', 'fit'):
            fitted, turn_fit = summary(trials, cal, sep, turns)
            continue
        if raw == 'save':
            fitted, turn_fit = summary(trials, cal, sep, turns)
            if not fitted and not turn_fit:
                print("  nothing to save: needs 4+ drives with a tape reading, or 4+ turns over 2+ tick counts "
                      "at one speed")
                continue
            if fitted:
                path = os.path.join(args.calib_dir, DRIVE_MOTION_FILE)
                if input("  write {} to {}? [y/N] ".format(fitted, path)).strip().lower() == 'y':
                    with open(path, 'w') as f:
                        json.dump(fitted, f, indent=2)
                    cal.update(fitted)
                    tpm = cal['ticks_per_meter']
                    print("  saved -- final_demo_l3.py and the next 'd' drives here use it")
            if turn_fit:
                path = os.path.join(args.calib_dir, TURN_MOTION_FILE)
                out = {k: round(float(turns[k]), 4) for k in ('turn_scale', 'offset_deg', 'turn_scale_fast',
                                                               'offset_fast_deg')}
                out.update(turn_fit)
                if input("  write {} to {} (the fitted speed(s) changed, the other kept)? [y/N] ".format(
                        out, path)).strip().lower() == 'y':
                    with open(path, 'w') as f:
                        json.dump(out, f, indent=2)
                    for key, fname in (('turn_scale', 'turn_scale.txt'), ('turn_scale_fast', 'turn_scale_fast.txt')):
                        if key in turn_fit:
                            with open(os.path.join(args.calib_dir, fname), 'w') as f:
                                f.write("{:.4f}\n".format(turn_fit[key]))
                    turns.update(out)
                    print("  saved (and turn_scale*.txt for the fitted speed(s)) -- final_demo_l3.py and the next "
                          "'t' turns here use it")
            continue
        if raw == 'undo':
            print("  removed" if trials and trials.pop() else "  nothing to undo")
            continue
        parts = raw.split()
        if len(parts) != 2 or parts[0] not in ('d', 'r', 't', 'tr'):
            print("  'd <cm>', 'r <cm>', 't <degrees>' or 'tr <degrees>', or sum / save / undo / q")
            continue
        try:
            val = float(parts[1])
        except ValueError:
            print("  not a number")
            continue
        last_cmd = raw
        kind = parts[0]

        fast = None
        if kind in ('d', 'r'):
            val = val / 100.0
            if kind == 'd':
                ticks = drive_ticks(val, tpm, cal['coast_ticks'], cal['coast_ramp_ticks'])
            else:
                ticks = int(round(abs(val) * tpm))
            if ticks < 1:
                print("  under a tick -- the robot would not move")
                continue
            speeds = [DRIVE_SPEED, DRIVE_SPEED] if val > 0 else [-DRIVE_SPEED, -DRIVE_SPEED]
            expect = ticks + coast(ticks, cal['coast_ticks'], cal['coast_ramp_ticks'])
            print("  {} ticks commanded{} -> expect ~{:.0f} counted = {:+.1f} cm".format(
                ticks, " ({:.0f} short for the roll-on)".format(abs(val) * tpm - ticks) if kind == 'd' else " (raw)",
                expect, math.copysign(100 * expect / tpm, val)))
            asked = val
        else:
            dtheta = math.radians(val)
            fast = abs(val) > args.small_turn_deg + 1e-6
            scale = turns['turn_scale_fast'] if fast else turns['turn_scale']
            off = math.radians(turns['offset_fast_deg'] if fast else turns['offset_deg'])
            tick = tick_angle(tpm, sep, scale)
            if kind == 't':
                ticks, b_abs = turn_plan(dtheta, tick, off)
            else:
                ticks = int(round(abs(dtheta) / tick))
                b_abs = ticks * tick + off
            if ticks < 1:
                print("  nearer zero than one tick + roll-on -- turn() would skip it")
                continue
            believed = math.copysign(b_abs, dtheta)
            mag = TURN_SPEED if fast else SMALL_TURN_SPEED
            speeds = [-mag, mag] if val > 0 else [mag, -mag]
            print("  {} ticks at speed {}{} -> {} x {:.2f} + {:.1f} roll-on = the EKF would be told {:+.1f} deg".format(
                ticks, mag, "" if kind == 't' else " (raw)", ticks, math.degrees(tick), math.degrees(off),
                math.degrees(believed)))
            asked = dtheta

        before = bot.get_encoder_counts()
        bot.move_auto_encoder(speeds, ticks, ticks)
        completed, (sl, sr), (al, ar), went_down = wait_and_count(bot, before)
        print("  counted when the Pi stopped: L {} R {}; once the wheels stopped: L {} R {} (rolled on +{} / +{})".format(
            sl, sr, al, ar, al - sl, ar - sr))
        if went_down:
            print("  WARNING: counts went DOWN -- an old resetting listen.py is running; tick numbers invalid")
        if not completed:
            print("  timed out -- not saved")
            continue
        counted = int(round(0.5 * (al + ar)))
        if kind in ('d', 'r'):
            told = math.copysign(counted / tpm, asked)
            print("  the demo would tell the EKF {:+.1f} cm".format(100 * told))

        is_turn = kind in ('t', 'tr')
        meas = ask("  measured {} (Enter = skip): ".format("angle, deg" if is_turn else "distance, cm"))
        if meas is None:
            continue
        meas = math.radians(meas) if is_turn else meas / 100.0
        if meas * asked < 0:
            print("  opposite sign to the command -- check the sign (+ = forward / left); not saved")
            continue
        if kind in ('d', 'r'):
            print("  EKF told {:+.1f} cm vs tape: {:+.1f} cm off;  landed {:+.1f} cm from what was asked".format(
                100 * told, 100 * (abs(told) - abs(meas)), 100 * (abs(meas) - abs(asked))))
            trials.append({'kind': kind, 'asked': asked, 'ticks': ticks, 'counted': counted, 'measured': meas})
        else:
            print("  EKF told {:+.1f} deg vs measured: {:+.1f} deg off;  landed {:+.1f} deg from what was asked".format(
                math.degrees(believed), math.degrees(abs(believed) - abs(meas)), math.degrees(abs(meas) - abs(asked))))
            trials.append({'kind': kind, 'asked': asked, 'ticks': ticks, 'counted': counted, 'believed': believed,
                           'measured': meas, 'fast': fast})

    bot.stop()
    print("\nSummary:")
    summary(trials, cal, sep, turns)


if __name__ == "__main__":
    main()
