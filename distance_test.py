# Test camera intrinsic accuracy by printing the live distance to a detected ArUco marker.
#
# Usage:
#   python test_intrinsic_distance.py --ip <robot_ip> --marker_length 0.06
#
# Hold an ArUco marker at a known, physically-measured distance from the camera
# (e.g. use a tape measure) and compare it against the printed value. Press 'q'
# in the image window, or Ctrl+C in the terminal, to quit.

import os
import sys
import time
import argparse
import cv2
import numpy as np
from botconnect import BotConnect


def load_calibration(calib_dir):
    camera_matrix = np.loadtxt(os.path.join(calib_dir, 'intrinsic.txt'), delimiter=',')
    dist_coeffs = np.loadtxt(os.path.join(calib_dir, 'distCoeffs.txt'), delimiter=',')
    return camera_matrix, dist_coeffs


def estimate_marker_distances(img, camera_matrix, dist_coeffs, marker_length):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(img)
    results = []
    if ids is None:
        return results, img

    half = marker_length / 2.0
    marker_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]], dtype=np.float32)

    for i in range(len(ids)):
        tag = int(ids[i, 0])
        success, rvec, tvec = cv2.solvePnP(
            marker_points, corners[i][0], camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if success:
            distance = float(np.linalg.norm(tvec))
            results.append((tag, distance))

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    return results, img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='localhost')
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--marker_length", type=float, default=0.06,
                         help="Physical side length of the ArUco marker in metres")
    args, _ = parser.parse_known_args()

    camera_matrix, dist_coeffs = load_calibration(args.calib_dir)
    print("Loaded intrinsic matrix:\n", camera_matrix)
    print("Loaded distortion coefficients:\n", dist_coeffs)
    print(f"\nUsing marker_length = {args.marker_length} m")
    print("Hold a marker at a known distance and compare against the printed value.")
    print("Press Ctrl+C or 'q' in the image window to quit.\n")

    botconnect = BotConnect(args.ip)
    time.sleep(1)  # give the camera thread a moment to connect

    try:
        while True:
            img = botconnect.get_image()  # RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            results, annotated = estimate_marker_distances(
                img_bgr, camera_matrix, dist_coeffs, args.marker_length)

            if results:
                for tag, distance in results:
                    print(f"Marker {tag}: distance = {distance:.3f} m")
            else:
                print("No marker detected.")

            cv2.imshow("Intrinsic distance test", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.3)  # slow down printing so it's readable

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nDone.")