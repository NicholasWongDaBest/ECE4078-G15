# Drive forward at a fixed speed for a user-specified duration, then print
# the resulting encoder tick counts.
#
# Useful for encoder/wheel calibration: measure the physical distance the
# robot actually travelled during the given time, then compute
# ticks_per_meter = avg_ticks / measured_distance yourself.
#
# Usage:
#   python drive_timed.py --ip <robot_ip>
#   (you will be prompted to enter a duration in seconds for each run)

import time
import argparse
from botconnect import BotConnect

SPEED = 0.6


def drive_for_duration(botconnect, duration):
    left_start, right_start = botconnect.get_encoder_counts()

    botconnect.move_manual([SPEED, SPEED])
    time.sleep(duration)

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
    args, _ = parser.parse_known_args()

    botconnect = BotConnect(args.ip)
    time.sleep(1)  # give connection threads a moment to establish

    print(f"Driving at fixed speed {SPEED}. Enter a duration (seconds) to run each trial.")
    print("Enter 'q' to quit.\n")

    while True:
        user_input = input("Duration (s): ").strip()
        if user_input.lower() == 'q':
            break

        try:
            duration = float(user_input)
        except ValueError:
            print("  Please enter a number, or 'q' to quit.")
            continue

        if duration <= 0:
            print("  Duration must be positive.")
            continue

        delta_left, delta_right = drive_for_duration(botconnect, duration)
        avg_ticks = (delta_left + delta_right) / 2.0

        print(f"  Left ticks: {delta_left}, Right ticks: {delta_right}, avg: {avg_ticks:.1f}\n")

    print("Done.")