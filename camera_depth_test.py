"""
camera_depth_test.py

Quick sanity test for a completed camera calibration: place the printed
checkerboard at a known, physically measured distance from the robot's
camera, and check how closely this script's estimated distance matches
that measurement.

Two independent depth estimates are computed from the same detected
corners, as a cross-check against each other:

1. solvePnP (primary) - fits the full 3D pose of the checkerboard using
   all 54 detected corners plus the known camera_matrix/dist_coeffs.
   Distance = the Z component of the resulting translation vector.
   Robust to the board being tilted relative to the camera, since pose
   is solved for explicitly (this is the same method aruco_sensor.py
   uses for ArUco markers).

2. Pinhole cross-check (secondary) - measures how many pixels one
   checkerboard square spans (averaged across the top/bottom rows for
   the horizontal estimate, and the left/right columns for the vertical
   estimate), then applies distance = (real_size * focal_length_px) /
   apparent_size_px directly. This assumes the board is held roughly
   fronto-parallel to the camera; if it disagrees a lot with the
   solvePnP estimate, that usually means the board was tilted more than
   expected, not that the calibration itself is wrong.

Usage:
    python camera_depth_test.py --square_size 25 --true_distance 500
    python camera_depth_test.py --image saved_shot.png --square_size 25
"""

import os
import sys
import argparse
import cv2
import numpy as np

CHECKERBOARD = (6, 9)  # inner corners, same as camera_calibration.py
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def get_frame(args):
    """Return one BGR frame: either loaded from disk or captured live."""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image from {args.image}")
            sys.exit(1)
        return frame

    # Live capture from the robot, same connection pattern as take_pic.py
    sys.path.insert(0, "..")
    from botconnect import BotConnect
    bot = BotConnect(args.ip)

    print("Live preview - press SPACE to capture, ESC to cancel.")
    frame = None
    while True:
        raw = bot.get_image()
        raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live preview - SPACE to capture, ESC to cancel", raw_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            frame = raw_bgr
            break
        elif key == 27:  # ESC
            print("Cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)
    cv2.destroyAllWindows()
    return frame


def detect_corners(frame):
    """Find and sub-pixel-refine checkerboard corners, with a visual check."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        print("Checkerboard not detected. Check framing, lighting, and focus, then try again.")
        sys.exit(1)

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

    preview = frame.copy()
    cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, found)
    cv2.imshow("Detected corners - press any key to continue", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners


def estimate_depth_solvepnp(corners, square_size, camera_matrix, dist_coeffs):
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size  # grid units -> real millimetres

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        print("solvePnP failed to converge.")
        sys.exit(1)
    return float(tvec[2, 0])


def estimate_depth_pinhole(corners, square_size, camera_matrix, dist_coeffs):
    # Undistort the corner pixel coordinates first, so this simple formula
    # (which assumes an ideal, distortion-free pinhole camera) is valid.
    undistorted = cv2.undistortPoints(corners, camera_matrix, dist_coeffs, P=camera_matrix)
    grid = undistorted.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)  # 9 rows x 6 cols

    top_left, top_right = grid[0, 0], grid[0, -1]
    bot_left, bot_right = grid[-1, 0], grid[-1, -1]

    top_span = np.linalg.norm(top_right - top_left)
    bot_span = np.linalg.norm(bot_right - bot_left)
    left_span = np.linalg.norm(bot_left - top_left)
    right_span = np.linalg.norm(bot_right - top_right)

    px_per_square_h = (top_span + bot_span) / 2 / (CHECKERBOARD[0] - 1)
    px_per_square_v = (left_span + right_span) / 2 / (CHECKERBOARD[1] - 1)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    depth_h = (square_size * fx) / px_per_square_h
    depth_v = (square_size * fy) / px_per_square_v
    return depth_h, depth_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost",
                         help="Robot IP address (ignored if --image is given)")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a saved image to test instead of a live capture")
    parser.add_argument("--calib_dir", type=str, default="param/",
                         help="Folder containing intrinsic.txt and distCoeffs.txt")
    parser.add_argument("--square_size", type=float, required=True,
                         help="Real-world side length of one checkerboard square, in millimetres")
    parser.add_argument("--true_distance", type=float, default=None,
                         help="Optional: measured true distance to the board, in millimetres")
    args, _ = parser.parse_known_args()

    camera_matrix = np.loadtxt(os.path.join(args.calib_dir, "intrinsic.txt"), delimiter=',')
    dist_coeffs = np.loadtxt(os.path.join(args.calib_dir, "distCoeffs.txt"), delimiter=',')

    frame = get_frame(args)
    corners = detect_corners(frame)

    depth_solvepnp = estimate_depth_solvepnp(corners, args.square_size, camera_matrix, dist_coeffs)
    depth_h, depth_v = estimate_depth_pinhole(corners, args.square_size, camera_matrix, dist_coeffs)
    depth_pinhole_avg = (depth_h + depth_v) / 2

    print("\n--- Depth test results (mm) ---")
    print(f"solvePnP estimate:            {depth_solvepnp:.1f}")
    print(f"Pinhole cross-check (horiz.): {depth_h:.1f}")
    print(f"Pinhole cross-check (vert.):  {depth_v:.1f}")
    print(f"Pinhole cross-check (avg):    {depth_pinhole_avg:.1f}")

    if args.true_distance is not None:
        error = depth_solvepnp - args.true_distance
        pct_error = 100 * error / args.true_distance
        print(f"\nMeasured true distance:       {args.true_distance:.1f}")
        print(f"solvePnP error:               {error:+.1f} ({pct_error:+.2f}%)")
