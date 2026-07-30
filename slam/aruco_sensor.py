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

            lm_measurement = Marker(lm_position, tag)
            sensor_measurement.append(lm_measurement)
        
        # Draw markers on image copy
        aruco_img = img.copy()
        cv2.aruco.drawDetectedMarkers(aruco_img, corners, ids)

        return sensor_measurement, aruco_img
        

class Marker:
    # Measurements are landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag):
        self.position = position
        self.tag = tag