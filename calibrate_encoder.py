# Drive the robot using the SAME wheel-speed commands operate.py's keyboard
# controls actually send, for a user-specified duration, then print the
# resulting encoder tick counts.
#
# Useful for encoder/wheel calibration: measure the physical distance (for
# forward/backward) or rotation angle (for turning) the robot actually
# completed during the given time, then compute ticks_per_meter yourself.
#
# Extended to cover every driving mode operate.py uses, not just forward,
# because ticks/distance isn't guaranteed to match across them in practice --
# turning in particular involves wheel scrub against the ground that
# straight-line driving doesn't, so a ticks_per_meter measured from
# forward-only trials may not hold once the robot starts turning during real
# SLAM driving. Run trials in each mode and compare before trusting one
# number for everything.
#
# Also mirrors two things about HOW operate.py drives that can themselves
# affect the ticks a trial produces, so a trial here experiences the same
# control conditions real driving does:
#   - PID speed control is enabled with the same gains operate.py uses
#     (Operate.__init__'s self.pid_gains). Calibrating open-loop (PID off)
#     measures a different speed-control regime than real driving runs under.
#   - Motion ramps up from 0 over RAMP_DURATION seconds instead of jumping
#     straight to full commanded speed, matching operate.py's
#     correct_straight_drive(). Real driving never starts at an instant full
#     speed, so a trial that does is testing an acceleration profile that
#     doesn't occur during actual use.
# Both can be turned off (--no-pid, --no-ramp) if you specifically want to
# isolate their effect or reproduce the older simple-forward-only behaviour.
#
# Usage:
#   python drive_timed.py --ip <robot_ip>
#   (you will be prompted for a mode letter and a duration for each trial)

import time
import argparse
from botconnect import BotConnect

# Wheel speeds copied from operate.py's update_keyboard() key bindings, so a
# trial here drives at exactly the speed real SLAM driving would command.
# name, [left_speed, right_speed]
MODES = {
    'f': ('Forward',    [0.4, 0.4]),      # operate.py K_UP
    'b': ('Backward',   [-0.4, -0.4]),    # operate.py K_DOWN
    'l': ('Turn left',  [-0.35, 0.35]),   # operate.py K_LEFT
    'r': ('Turn right', [0.35, -0.35]),   # operate.py K_RIGHT
}

# Matches Operate.__init__'s self.pid_gains in operate.py.
PID_GAINS = {'kp': 0.8, 'ki': 0.02, 'kd': 0.25}

# Matches Operate.__init__'s self.ramp_duration in operate.py.
RAMP_DURATION = 0.15


def drive_for_duration(botconnect, wheel_speed, duration, ramp=True):
    left_start, right_start = botconnect.get_encoder_counts()

    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        scale = min(1.0, elapsed / RAMP_DURATION) if ramp else 1.0
        botconnect.move_manual([wheel_speed[0] * scale, wheel_speed[1] * scale])
        time.sleep(0.02)  # resend periodically, same idea as operate.py's main loop

    # Read encoder counts BEFORE stopping. The Pi resets its encoder counters
    # to zero the instant it detects the robot has stopped, so reading after
    # stop() would return ~0 ticks regardless of how far the robot travelled.
    left_end, right_end = botconnect.get_encoder_counts()

    botconnect.stop()

    delta_left = left_end - left_start
    delta_right = right_end - right_start
    return delta_left, delta_right


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='localhost')
    parser.add_argument("--no-ramp", action="store_true",
                         help="jump straight to full commanded speed instead of ramping over "
                              f"{RAMP_DURATION}s like operate.py does")
    parser.add_argument("--no-pid", action="store_true",
                         help="don't enable the robot's PID speed control before driving")
    args, _ = parser.parse_known_args()

    botconnect = BotConnect(args.ip)
    time.sleep(1)  # give connection threads a moment to establish

    if not args.no_pid:
        botconnect.set_pid(use_pid=1, **PID_GAINS)

    print("Modes:")
    for key, (name, speed) in MODES.items():
        print(f"  {key} = {name:<10s} wheel_speed={speed}")
    print(f"\nRamp: {'off (instant full speed)' if args.no_ramp else f'on ({RAMP_DURATION}s, matches operate.py)'}")
    print(f"PID:  {'off' if args.no_pid else f'on {PID_GAINS} (matches operate.py)'}")
    print("\nEnter a mode letter and a duration, e.g. 'f 2' for 2s forward.")
    print("For turning trials, left/right ticks cancel out in the signed average -- use the")
    print("magnitude average instead, and measure the actual angle turned to relate it to baseline.")
    print("Enter 'q' to quit.\n")

    while True:
        user_input = input("Mode duration (e.g. 'f 2'): ").strip()
        if user_input.lower() == 'q':
            break

        parts = user_input.split()
        if len(parts) != 2 or parts[0].lower() not in MODES:
            print(f"  Please enter one of {list(MODES)} followed by a duration, or 'q' to quit.")
            continue

        mode_key = parts[0].lower()
        try:
            duration = float(parts[1])
        except ValueError:
            print("  Please enter a number for duration.")
            continue

        if duration <= 0:
            print("  Duration must be positive.")
            continue

        name, wheel_speed = MODES[mode_key]
        delta_left, delta_right = drive_for_duration(
            botconnect, wheel_speed, duration, ramp=not args.no_ramp)

        avg_ticks = (delta_left + delta_right) / 2.0
        abs_avg_ticks = (abs(delta_left) + abs(delta_right)) / 2.0

        print(f"  [{name}] Left ticks: {delta_left}, Right ticks: {delta_right}")
        print(f"    signed avg: {avg_ticks:.1f}   magnitude avg: {abs_avg_ticks:.1f}")
        if mode_key in ('f', 'b'):
            print("    -> for ticks_per_meter, use signed avg / measured straight-line distance\n")
        else:
            print("    -> for turning, use magnitude avg / measured rotation (wheels move opposite ways)\n")

    print("Done.")