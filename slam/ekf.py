import os
import cv2
import math
import json
import pygame
import numpy as np


# This class stores the wheel velocities of the robot, to be used in the EKF.
class DriveMeasurement:
    def __init__(self, left_speed, right_speed, dt, left_cov=1, right_cov=1, delta_left_ticks=None, delta_right_ticks=None):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov
        self.delta_left_ticks = delta_left_ticks
        self.delta_right_ticks = delta_right_ticks


class EKF:
    # Implementation of an EKF for SLAM
    # The EKF state is composed of the robot position (x, y, theta) and the landmark position (x_lm1, y_lm1, x_lm2, y_lm2, ....)
    # lm stands for landmark, ie the aruco markers.

    def __init__(self, robot):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))

        # Covariance matrix
        self.P = np.zeros((3,3)) # shape of this matrix changes as more landmarks are discovered

        self.taglist = []
        self.init_lm_cov = 1e3
        self.innovation_gate = 9.21  # chi-square gate, 2 DOF, ~99% confidence -- see update()
        self.robot_init_state = None
        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./ui/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./ui/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./ui/8bit/pibot_top.png')

    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.P = np.zeros((3,3))
        self.taglist = []
        self.init_lm_cov = 1e3
        self.innovation_gate = 9.21
        self.robot_init_state = None

    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state

    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')

    def save_map(self, fname="slam.txt"):
        d = {}
        if self.number_landmarks() > 0:
            for i, tag in enumerate(self.taglist):
                d["aruco" + str(tag) + "_0"] = {"x": self.markers[0,i], "y":self.markers[1,i]}
        with open(fname, 'w') as map_f:
            json.dump(d, map_f, indent=4)

    def save_state(self, fname):
        """Persist markers, covariance, and taglist so they survive a restart."""
        try:
            state = {
                'markers': self.markers.tolist(),
                'P': self.P.tolist(),
                'taglist': self.taglist,
            }
            tmp_fname = fname + '.tmp'
            with open(tmp_fname, 'w') as f:
                json.dump(state, f)
            os.replace(tmp_fname, fname)  # atomic, avoids a corrupt file if killed mid-write
        except Exception as e:
            print(f"Failed to save SLAM state: {e}")

    def load_state(self, fname):
        """Restore markers, covariance, and taglist from a previous session.
        Robot pose is intentionally NOT restored -- it stays at the origin and
        should be re-anchored via recover_from_pause() (ENTER key) once
        landmarks are back in view."""
        if not os.path.exists(fname):
            return False
        try:
            with open(fname, 'r') as f:
                state = json.load(f)
            markers = np.array(state['markers'], dtype=float)
            markers = markers.reshape(2, -1) if markers.size else np.zeros((2, 0))
            self.markers = markers
            self.P = np.array(state['P'], dtype=float)
            self.taglist = [int(t) for t in state['taglist']]
            return True
        except Exception as e:
            print(f"Failed to load SLAM state: {e}")
            return False

    def recover_from_pause(self, sensor_measurement):
        if not sensor_measurement:
            return False
        else:
            lm_new = np.zeros((2,0))
            lm_prev = np.zeros((2,0))
            tag = []
            for lm in sensor_measurement:
                if lm.tag in self.taglist:
                    lm_new = np.concatenate((lm_new, lm.position), axis=1)
                    tag.append(int(lm.tag))
                    lm_idx = self.taglist.index(lm.tag)
                    lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
            if int(lm_new.shape[1]) >= 2:
                R,t = self.umeyama(lm_new, lm_prev)
                theta = math.atan2(R[1][0], R[0][0])
                self.robot.state[:2]=t[:2]
                self.robot.state[2]=theta
                return True
            else:
                return False

    def compute_live_rmse(self, true_map):
        """
        Live SLAM RMSE against a ground-truth marker map.
        @param true_map: dict {tag:int -> np.array([[x],[y]])}
        Aligns the current estimate to the true map with a rotation+translation
        (Umeyama, no scale -- same convention eval.py uses) using only markers
        that are in BOTH self.taglist and true_map.
        Returns a dict {'rmse', 'R', 't', 'matched_tags'}, or None if fewer
        than 2 markers are matched (not enough to solve for the alignment).
        """
        if true_map is None or self.number_landmarks() == 0:
            return None

        matched_tags = [tag for tag in self.taglist if tag in true_map]
        if len(matched_tags) < 2:
            return None

        est_pts = np.zeros((2, len(matched_tags)))
        true_pts = np.zeros((2, len(matched_tags)))
        for i, tag in enumerate(matched_tags):
            idx = self.taglist.index(tag)
            est_pts[:, i:i+1] = self.markers[:, idx:idx+1]
            true_pts[:, i:i+1] = true_map[tag]

        try:
            R, t = self.umeyama(est_pts, true_pts)
        except ValueError:
            # matched points are collinear -- rotation isn't uniquely solvable yet
            return None

        aligned_est = R @ est_pts + t
        residual = (aligned_est - true_pts).ravel()
        rmse = float(np.sqrt(1.0 / len(matched_tags) * np.sum(residual ** 2)))  # matches eval.py's compute_rmse exactly

        return {'rmse': rmse, 'R': R, 't': t, 'matched_tags': matched_tags}

    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    # the prediction step of EKF
    def predict(self, drive_measurement):
        F = self.state_transition(drive_measurement)
        Q = self.predict_covariance(drive_measurement)
        Q[0:3, 0:3] += 0.01 * np.eye(3)

        # 1. Drive the robot forward to propagate state
        self.robot.drive(drive_measurement)

        # 2. Propagate state uncertainty covariance P
        self.P = F @ self.P @ F.T + Q

    # the update/correct step of EKF
    def update(self, sensor_measurement):
        if not sensor_measurement:
            return

        # Only update on markers that are already part of the state (added via add_landmarks)
        known_measurement = [lm for lm in sensor_measurement if lm.tag in self.taglist]
        if not known_measurement:
            return

        # Construct measurement index list
        tags = [lm.tag for lm in known_measurement]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and build their covariance
        z_all = np.concatenate([lm.position.reshape(-1, 1) for lm in known_measurement], axis=0)
        R_all = np.zeros((2 * len(known_measurement), 2 * len(known_measurement)))
        for i in range(len(known_measurement)):
            distance = np.linalg.norm(known_measurement[i].position)
            depth_var = 0.005 + 0.2 * distance**2
            lateral_var = depth_var * 1.5   # lateral assumed noisier/less observed; tune this ratio

            # lm.position is [depth, lateral] in the robot's body frame, and so are
            # z_hat/H below (Robot.measure / derivative_measure) -- R MUST stay in
            # that same local frame. Do not rotate it by theta: z, z_hat, and H are
            # never expressed in world frame, so rotating R mixes frames and silently
            # swaps how much depth vs. lateral is trusted as theta changes.
            R_all[2*i:2*i+2, 2*i:2*i+2] = np.diag([depth_var, lateral_var])

        # Compute own measurements
        z_hat_all = self.robot.measure(self.markers, idx_list)
        z_hat_all = z_hat_all.reshape((-1, 1), order="F")
        H_all = self.robot.derivative_measure(self.markers, idx_list)

        # --- Innovation (Mahalanobis) gating ---
        # Check each marker's reading against what the filter currently expects
        # BEFORE accepting it, rather than blending every reading in blindly.
        # A marker glimpsed at a bad/foreshortened angle -- or any other one-off
        # bad reading -- tends to show up as a statistical outlier here, and gets
        # skipped for this frame instead of dragging the robot pose and every
        # other landmark along with it. d2 is the squared Mahalanobis distance;
        # under normal noise it follows a chi-square distribution with 2 DOF, so
        # innovation_gate=9.21 rejects only the ~1% most inconsistent readings.
        keep = []
        for i in range(len(known_measurement)):
            y_i = z_all[2*i:2*i+2] - z_hat_all[2*i:2*i+2]
            H_i = H_all[2*i:2*i+2, :]
            R_i = R_all[2*i:2*i+2, 2*i:2*i+2]
            S_i = H_i @ self.P @ H_i.T + R_i
            d2 = (y_i.T @ np.linalg.inv(S_i) @ y_i).item()
            if d2 <= self.innovation_gate:
                keep.append(i)

        if not keep:
            return

        rows = [r for i in keep for r in (2*i, 2*i+1)]
        z = z_all[rows, :]
        z_hat = z_hat_all[rows, :]
        H = H_all[rows, :]
        R = R_all[np.ix_(rows, rows)]

        x = self.get_state_vector()

        # 1. Measurement residual (innovation)
        y = z - z_hat

        # 2. Innovation covariance
        S = H @ self.P @ H.T + R

        # 3. Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. Update state vector x and set it back in robot/markers
        x_updated = x + K @ y
        self.set_state_vector(x_updated)

        # 5. Update state covariance P -- Joseph form (symmetric & numerically stable)
        I = np.eye(len(x))
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Force exact numerical symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def state_transition(self, drive_measurement):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(drive_measurement)
        return F

    def predict_covariance(self, drive_measurement):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(drive_measurement)
        return Q

    def add_landmarks(self, sensor_measurement):
        if not sensor_measurement:
            return

        if len(self.taglist) == 0:
            # Bootstrap case: map is empty, there are no "known" landmarks to require.
            # Fall back to requiring several simultaneous markers instead, so the very
            # first landmarks are still reasonably well-constrained by multi-marker geometry.
            if len(sensor_measurement) < 3:
                return
        else:
            known_in_view = [lm for lm in sensor_measurement if lm.tag in self.taglist]
            if len(known_in_view) < 1:
                return

        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

        for lm in sensor_measurement:
            if lm.tag in self.taglist:
                continue

            lm_position = lm.position
            lm_state = robot_xy + R_theta @ lm_position

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_state), axis=1)

            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)

            # Scale initial covariance with distance at first sighting -- a landmark
            # first seen far away starts with more uncertainty than one seen close up.
            distance = np.linalg.norm(lm_position)
            lm_cov = self.init_lm_cov * (1.0 + 0.3 * distance)  # tune the 0.3 factor
            self.P[-2,-2] = lm_cov**2
            self.P[-1,-1] = lm_cov**2

    @staticmethod
    def umeyama(from_points, to_points):
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"

        N = from_points.shape[1]
        m = 2

        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        cov_matrix = delta_to @ delta_from.T / N

        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)

        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
        return R, t

    # Plotting functions
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res = (320, 500), not_pause=True, true_map=None, live_rmse_info=None):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters,
        lms_xy = self.markers[:2, :]
        robot_xy_world = self.robot.state[:2, 0].reshape((2, 1))  # position in the SLAM frame, before we recentre on the robot
        lms_xy = lms_xy - robot_xy_world
        robot_xy = robot_xy_world*0
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)

        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (244, 69, 96), 1)

        # --- overlay ground-truth markers for live RMSE practice ---
        if true_map is not None and live_rmse_info is not None:
            R, t = live_rmse_info['R'], live_rmse_info['t']
            matched_tags = live_rmse_info['matched_tags']
            for tag, true_xy in true_map.items():
                # bring the true marker from the ground-truth frame into the
                # SLAM/estimated frame using the inverse of the alignment
                # transform, then recentre on the robot like the estimates above
                true_in_est = R.T @ (true_xy - t) - robot_xy_world
                coor_true = self.to_im_coor((true_in_est[0,0], true_in_est[1,0]), res, m2pixel)
                colour = (40, 170, 40) if tag in matched_tags else (140, 140, 40)
                cv2.drawMarker(canvas, coor_true, colour, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
                cv2.putText(canvas, str(tag), (coor_true[0]+6, coor_true[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

                # error line: your current estimate -> where it should be
                if tag in matched_tags:
                    idx = self.taglist.index(tag)
                    coor_est = self.to_im_coor((lms_xy[0,idx], lms_xy[1,idx]), res, m2pixel)
                    cv2.line(canvas, coor_est, coor_true, (0, 140, 255), 1)

            rmse_text = f"RMSE {live_rmse_info['rmse']:.4f}m ({len(matched_tags)}/{len(true_map)})"
            cv2.putText(canvas, rmse_text, (5, res[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3), (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1], (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1], (coor_[0]-5, coor_[1]-5))
        return surface

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = np.sqrt(np.maximum(0, e_vals)) * alpha

        # Near-isotropic covariance: the ellipse is essentially a circle, and its
        # "orientation" is numerically meaningless -- tiny noise in P flips which
        # eigenvector sorts first and swings the angle wildly even though the
        # actual shape hasn't changed. Skip the noisy angle in that regime.
        if e_vals[0] - e_vals[1] < 1e-6 * max(e_vals[0], 1e-12):
            angle = 0.0
        elif abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle