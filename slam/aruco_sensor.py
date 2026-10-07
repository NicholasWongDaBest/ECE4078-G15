# detect ARUCO markers and estimate their positions
import cv2
import os, sys
import numpy as np

try:
    from slam import aruco_faces
except ImportError:          # slam/ itself on sys.path
    import aruco_faces


class ArucoSensor:
    def __init__(self, robot, marker_length=0.06, faces_config=None):
        self.camera_matrix = robot.camera_matrix
        self.dist_coeffs = robot.dist_coeffs

        # ArUco faces (slam/aruco_faces.py, calibration/param/aruco_faces.json).
        # Only a program that passes faces_config (final_demo_l3.py) gets them;
        # without it -- operate.py, auto_fruit_search.py on this branch -- the
        # sensor is exactly the old one: default detector settings, no .faces.
        self.faces_cfg = faces_config
        self.faces_on = faces_config is not None and bool(faces_config.get('faces', True))
        if self.faces_on and faces_config.get('marker_length'):
            marker_length = float(faces_config['marker_length'])

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters()
        if self.faces_on:
            # Sub-pixel corners: roughly halves the face-direction noise.
            self.aruco_params.cornerRefinementMethod = aruco_faces.REFINE[faces_config['refine']]
            # OpenCV 4.13's default (0.125) treats the two faces of one block seen
            # at a corner (same id, sharing an edge) as duplicates and drops one.
            self.aruco_params.minMarkerDistanceRate = float(faces_config['min_marker_distance'])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        half = self.marker_length / 2.0
        self.marker_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]], dtype=np.float32)

    # Perform detection of aruco markers
    # Obtain the position of the markers and their respective tag/id (number), then store it in the measurements array
    # These positions are position seen from the camera, not the actual position in the arena
    #
    # Each Marker's .position is the same as it always was (the face centre, averaged
    # over the faces of that tag in view). With faces on, each Marker also carries
    # .faces: one aruco_faces.Face per detected face (both pose solutions, the printed
    # rotation, ...), which aruco_faces.MarkerCorrector.correct() turns into
    # block-centre positions once the robot's heading is known.
    def detect_marker_positions(self, img):
        corners, ids, rejected = self.detector.detectMarkers(img)
        if ids is None or len(corners) == 0:
            return [], img
        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

        marker_points = self.marker_points

        rvecs, tvecs = [], []
        for c in corners:
            success, rvec, tvec = cv2.solvePnP(marker_points, c[0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(rvec)
            tvecs.append(tvec)

        rvecs, tvecs = np.array(rvecs), np.array(tvecs)
        rvecs, tvecs = rvecs.reshape(-1, 1, 3), tvecs.reshape(-1, 1, 3)

        faces = [None] * len(corners)
        if self.faces_on:
            for i, c in enumerate(corners):
                faces[i] = aruco_faces.analyse_corners(int(ids[i, 0]), c, self.camera_matrix,
                                                       self.dist_coeffs, marker_points)

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
            if self.faces_on:
                tag_faces = [f for j, f in enumerate(faces) if ids[j, 0] == tag and f is not None]
                lm_measurement.faces = sorted(tag_faces, key=lambda f: f.u)   # left to right in the image
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
