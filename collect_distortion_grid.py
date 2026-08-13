# Collect (measured, true) landmark position pairs for empirical distortion
# correction. The robot must stay COMPLETELY STATIONARY during this whole
# session -- self.robot.state is never advanced (no drive()/predict() calls),
# so ArucoSensor.detect_marker_positions() always returns each marker's
# position in the robot's fixed body frame. That's exactly the "measured"
# quantity we're correcting; you supply the physically-measured "true" x,y
# for comparison.
#
# Usage:
#   python collect_distortion_grid.py --ip <robot_ip>
#
# Procedure:
#   1. Fix the robot in one position and orientation. Do not move it for the
#      entire session -- every recorded pair assumes the robot hasn't moved.
#   2. For each sample: place a SINGLE ArUco marker at a spot within the
#      camera's view, measure its true (x, y) relative to the robot's own
#      origin/heading (x = forward, y = left -- same convention as
#      Robot.state), enter those values when prompted, then press ENTER to
#      capture. Enter the ACTUAL measured position, not the suggested target
#      -- exact placement doesn't need to match the suggestion.
#   3. Repeat across the suggested grid printed at startup (or your own
#      spread of distances/angles).
#   4. Type 'q' instead of a distance to finish and save.

import os
import sys
import csv
import time
import argparse
import numpy as np
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor
from botconnect import BotConnect


def suggest_grid(camera_matrix, image_width, min_r, max_r, n_rings, n_angles):
    fx = camera_matrix[0, 0]
    half_fov_rad = np.arctan((image_width / 2) / fx)
    half_fov_deg = np.degrees(half_fov_rad)
    # Stay a bit inside the true edge -- detection quality degrades right at
    # the boundary, and we want the fit's domain to cover where you'll
    # actually operate, not the absolute limit of visibility.
    usable_half_fov_deg = half_fov_deg * 0.85

    radii = np.linspace(min_r, max_r, n_rings)
    angles_deg = np.linspace(-usable_half_fov_deg, usable_half_fov_deg, n_angles)

    print(f"Estimated horizontal FOV: {2*half_fov_deg:.1f} deg (from camera_matrix fx={fx:.1f}, width={image_width})")
    print(f"Suggested angle range: +/-{usable_half_fov_deg:.1f} deg\n")
    print("Suggested sample grid (approximate targets -- place markers near these,")
    print("then enter the ACTUAL measured x,y, not these targets):")
    print(f"{'radius(m)':>10} {'angle(deg)':>12} {'approx x':>10} {'approx y':>10}")
    for r in radii:
        for a_deg in angles_deg:
            a_rad = np.radians(a_deg)
            x = r * np.cos(a_rad)
            y = r * np.sin(a_rad)
            print(f"{r:10.2f} {a_deg:12.1f} {x:10.2f} {y:10.2f}")
    print(f"\nTotal suggested points: {n_rings * n_angles}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='localhost')
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--marker_length", type=float, default=0.06)
    parser.add_argument("--min_r", type=float, default=0.3, help="Nearest ring distance (m)")
    parser.add_argument("--max_r", type=float, default=1.3, help="Farthest ring distance (m)")
    parser.add_argument("--n_rings", type=int, default=4)
    parser.add_argument("--n_angles", type=int, default=7)
    parser.add_argument("--out", type=str, default="distortion_grid_data.csv")
    args, _ = parser.parse_known_args()

    camera_matrix = np.loadtxt(os.path.join(args.calib_dir, 'intrinsic.txt'), delimiter=',')
    dist_coeffs = np.loadtxt(os.path.join(args.calib_dir, 'distCoeffs.txt'), delimiter=',')
    # baseline/scale are irrelevant here (never used, since we never call
    # robot.drive()) -- placeholders only so Robot() can be constructed.
    robot = Robot(baseline=0.1, scale=1.0, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    aruco_sensor = ArucoSensor(robot, marker_length=args.marker_length)

    botconnect = BotConnect(args.ip)
    time.sleep(1)

    suggest_grid(camera_matrix, image_width=480,
                 min_r=args.min_r, max_r=args.max_r,
                 n_rings=args.n_rings, n_angles=args.n_angles)

    print("IMPORTANT: keep the robot completely stationary for this entire session.")
    print("Only ONE marker should be visible in frame per shot.\n")

    rows = []
    if os.path.exists(args.out):
        print(f"Appending to existing {args.out}")
        with open(args.out, 'r') as f:
            rows = list(csv.DictReader(f))

    try:
        while True:
            raw = input("\nTrue x (forward, m) [or 'q' to finish]: ").strip()
            if raw.lower() == 'q':
                break
            try:
                x_true = float(raw)
                y_true = float(input("True y (left, m): ").strip())
            except ValueError:
                print("  Please enter numbers.")
                continue

            input("  Place the marker at that measured position, then press ENTER to capture...")
            img = botconnect.get_image()
            sensor_measurement, _ = aruco_sensor.detect_marker_positions(img)

            if len(sensor_measurement) == 0:
                print("  No marker detected -- not saved. Try again.")
                continue
            if len(sensor_measurement) > 1:
                print(f"  {len(sensor_measurement)} markers detected -- keep only ONE in frame. Not saved.")
                continue

            lm = sensor_measurement[0]
            x_meas, y_meas = float(lm.position[0, 0]), float(lm.position[1, 0])
            err = np.hypot(x_meas - x_true, y_meas - y_true)
            print(f"  Measured: x={x_meas:.3f}, y={y_meas:.3f}  "
                  f"(true: x={x_true:.3f}, y={y_true:.3f}, error: {err:.3f} m)")

            rows.append({'tag': lm.tag, 'x_true': x_true, 'y_true': y_true,
                         'x_meas': x_meas, 'y_meas': y_meas})

    except KeyboardInterrupt:
        pass

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'x_true', 'y_true', 'x_meas', 'y_meas'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} sample points to {args.out}")
    if len(rows) < 20:
        print("Consider collecting more points (20-35+) for a robust polynomial fit.")