# detect ARUCO markers and estimate their positions
import cv2
import os, sys
import numpy as np


class ArucoSensor:
    def __init__(self, robot, marker_length=0.06):
        self.camera_matrix = robot.camera_matrix
        self.dist_coeffs = robot.dist_coeffs

        self.marker_length = marker_length  
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    # Perform detection of aruco markers
    # Obtain the position of the markers and their respective tag/id (number), then store it in the measurements array
    # These positions are position seen from the camera, not the actual position in the arena
    def detect_marker_positions(self, img):
        corners, ids, rejected = self.detector.detectMarkers(img)
        if ids is None or len(corners) == 0:
            return [], img
        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
        
        half = self.marker_length / 2.0
        marker_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]], dtype=np.float32)
        
        rvecs, tvecs = [], []
        for c in corners:
            success, rvec, tvec = cv2.solvePnP(marker_points, c[0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(rvec)
            tvecs.append(tvec)

        rvecs, tvecs = np.array(rvecs), np.array(tvecs)
        rvecs, tvecs = rvecs.reshape(-1, 1, 3), tvecs.reshape(-1, 1, 3)

        # Apparent height of each marker in PIXELS, for the focal-length check
        # in operate.py (record_focal_check). Measured as the mean length of
        # the two VERTICAL edges rather than the bounding-box height: ArUco
        # corners come back ordered TL, TR, BR, BL, so those edges are 0->3 and
        # 1->2. Edge lengths beat a bounding box for two reasons -- a marker
        # rotated in the image plane inflates its bbox height without its true
        # extent changing, and a marker viewed obliquely (rotated about the
        # vertical axis) foreshortens in WIDTH while its vertical edges stay
        # close to true, which is exactly what depth-from-height depends on.
        heights_px = {}
        for i in range(len(ids)):
            c = np.asarray(corners[i][0], dtype=float)
            h = 0.5 * (np.linalg.norm(c[0] - c[3]) + np.linalg.norm(c[1] - c[2]))
            heights_px.setdefault(int(ids[i, 0]), []).append(float(h))

        # Compute the marker positions. lm stands for landmark, ie the aruco markers
        sensor_measurement, seen_tag = [], []
        for i in range(len(ids)):
            tag = ids[i,0]
            if tag in seen_tag:
                continue # Some markers appear multiple times but should only be handled once.
            else:
                seen_tag.append(tag)

            lm_tvecs = tvecs[ids==tag].T
            lm_position = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_position = np.mean(lm_position, axis=1).reshape(-1,1)

            hs = heights_px.get(int(tag), [])
            lm_measurement = Marker(lm_position, tag,
                                     height_px=(float(np.mean(hs)) if hs else None))
            sensor_measurement.append(lm_measurement)
        
        # Draw markers on image copy
        aruco_img = img.copy()
        cv2.aruco.drawDetectedMarkers(aruco_img, corners, ids)

        return sensor_measurement, aruco_img
        

class Marker:
    # Measurements are landmarks in 2D and have a position as well as tag id.
    # height_px is optional and defaults to None so anything that builds a
    # Marker by hand (calibrate_ekf.py's _LoggedMarker stub, replayed logs)
    # keeps working -- read it with getattr(lm, 'height_px', None).
    def __init__(self, position, tag, height_px=None):
        self.height_px = height_px
        self.position = position
        self.tag = tag