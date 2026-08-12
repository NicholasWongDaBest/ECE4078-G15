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
        self.robot_init_state = None
        self.freeze_map = False # M2: true once true map is loaded
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
        self.robot_init_state = None
        self.freeze_map = False # M2 true map

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
        if self.number_landmarks() > 0:
            d = {}
            for i, tag in enumerate(self.taglist):
                d["aruco" + str(tag) + "_0"] = {"x": self.markers[0,i], "y":self.markers[1,i]}
            with open(fname, 'w') as map_f:
                json.dump(d, map_f, indent=4)

    def load_true_map(self, fname):
        """Load ground-truth landmark coordinates (M2) into self.markers/self.taglist,
        and freeze them so SLAM only localises the robot pose without ever moving
        the landmarks. Ignores non-ArUco entries (e.g. fruit ground-truth) that may
        share the same true_map.txt file."""
        with open(fname, 'r') as f:
            gt_dict = json.load(f)

        aruco_keys = [k for k in gt_dict if k.startswith('aruco')]
        tags = sorted(int(k.split('aruco')[1].split('_')[0]) for k in aruco_keys)
        markers = np.zeros((2, len(tags)))
        for i, tag in enumerate(tags):
            key = f"aruco{tag}_0"
            markers[0, i] = gt_dict[key]['x']
            markers[1, i] = gt_dict[key]['y']

        robot_P = self.P[0:3, 0:3].copy() if self.P.shape[0] >= 3 else np.zeros((3, 3))

        self.taglist = tags
        self.markers = markers
        n = len(tags)
        self.P = np.zeros((3 + 2*n, 3 + 2*n))
        self.P[0:3, 0:3] = robot_P
        self.freeze_map = True
        return n

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
        
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    # the prediction step of EKF
    def predict(self, drive_measurement):
        # # OLD CODE, ADD FLAT UNCERTAINTY (0.01) TO PURE ROTATION AND TRANSLATION
        # F = self.state_transition(drive_measurement)
        # x = self.get_state_vector()
        # Q = self.predict_covariance(drive_measurement)
        # Q[0:3,0:3] += 0.01*np.eye(3)

        # # TODO: add your codes here to compute the predicted x
        # # 1. Drive the robot forward to propagate state
        # self.robot.drive(drive_measurement)
        
        # # 2. Propagate state uncertainty covariance P
        # self.P = F @ self.P @ F.T + Q
        # # TODO end

        ## NOW DO DIFFERENT NOISE DEPENDING ON TRANSLATION OR ROTATION + SCALE Q WITH TIME
        F = self.state_transition(drive_measurement)
        x = self.get_state_vector()
        Q = self.predict_covariance(drive_measurement)

        dt = drive_measurement.dt

        # Calculate linear and angular components
        v_left = abs(drive_measurement.left_speed)
        v_right = abs(drive_measurement.right_speed)
        v_avg = abs(v_left + v_right) / 2.0  # Forward speed


        if v_avg < 0.05:  # Pure rotation / Turning on the spot
            # Only add process noise to theta (heading), lock x and y!
            Q_boost = np.diag([0.0, 0.0, 0.005 * v_avg * dt + 1e-4,])
        else:  # Moving forward
            # Add normal process noise to x, y, and theta
            Q_boost = np.diag(
                        [
                            0.001 * v_avg * dt + 1e-5,  # x noise
                            0.001 * v_avg * dt + 1e-5,  # y noise
                            0.005 * v_avg * dt + 1e-4,  # theta noise
                        ]
                )

        Q[0:3, 0:3] += Q_boost

        # Propagate state and covariance
        self.robot.drive(drive_measurement)

        # 2. Propagate the covariance forward: P = F P F^T + Q
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

        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in known_measurement], axis=0)
        R = np.zeros((2*len(known_measurement),2*len(known_measurement)))
        for i in range(len(known_measurement)):
            distance = np.linalg.norm(known_measurement[i].position)
            # Build R in the marker's LOCAL frame (robot-relative), then rotate into world
            depth_var = 0.03 + 0.015 * distance**2
            lateral_var = depth_var * 1.5   # lateral assumed noisier/less observed; tune this ratio

            # lm.position is [depth, lateral] roughly, in robot frame
            R_local = np.diag([depth_var, lateral_var])

            # Rotate into world frame to match how z/z_hat are expressed
            th = self.robot.state[2, 0]
            Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            R_world = Rot @ R_local @ Rot.T
            R[2*i:2*i+2, 2*i:2*i+2] = R_world


        x_prior = self.get_state_vector()      # x⁽⁰⁾ before any iteration
        P_prior = self.P.copy()                # P before this update, held fixed across iterations


    # --- IEKF ITERATION LOOP ---
        x_iter = x_prior.copy()
        max_iters = 3

        for iteration in range(max_iters):
            # 1. Update robot/markers state temporarily to re-evaluate z_hat and H
            self.set_state_vector(x_iter)

            z_hat = self.robot.measure(self.markers, idx_list).reshape(
                (-1, 1), order="F"
            )
            H = self.robot.derivative_measure(self.markers, idx_list)

            # 2. Recompute Innovation with prior offset constraint
            # y = (z - z_hat) - H @ (x_prior - x_iter)
            y = (z - z_hat) + H @ (x_iter - x_prior)

            # 3. Kalman Gain
            S = H @ P_prior @ H.T + R
            K = P_prior @ H.T @ np.linalg.inv(S)

            # 4. Next state iterate
            x_next = x_prior + K @ y

            # Convergence check (stop early if change is negligible)
            if np.linalg.norm(x_next - x_iter) < 1e-4:
                x_iter = x_next
                break
            x_iter = x_next

        # Apply final converged state and Joseph-form covariance
        self.set_state_vector(x_iter)
        I = np.eye(len(x_prior))
        I_KH = I - K @ H
        self.P = I_KH @ P_prior @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Floor landmark covariance so the filter never becomes fully "locked in" --
        # keeps landmarks correctable even after many observations, which matters
        # while camera distortion calibration is still being refined.
        if not self.freeze_map:
            min_lm_var = 3.75e-4   # tune: larger = more correctable, but noisier steady-state
            for i in range(self.number_landmarks()):
                idx = 3 + 2*i
                if self.P[idx, idx] < min_lm_var:
                    self.P[idx, idx] = min_lm_var
                if self.P[idx+1, idx+1] < min_lm_var:
                    self.P[idx+1, idx+1] = min_lm_var

        # # Compute own measurements
        # z_hat = self.robot.measure(self.markers, idx_list)
        # z_hat = z_hat.reshape((-1,1), order="F")
        # H = self.robot.derivative_measure(self.markers, idx_list)

        # x = self.get_state_vector()
        
        # # TODO: add your codes here to compute the updated x
        # # 1. Measurement residual (innovation)
        # y = z - z_hat
        
        # # 2. Innovation covariance
        # S = H @ self.P @ H.T + R
        
        # # 3. Kalman Gain
        # K = self.P @ H.T @ np.linalg.inv(S)
        
        # # 4. Update state vector x and set it back in robot/markers
        # x_updated = x + K @ y
        # self.set_state_vector(x_updated)
        
        # # 5. Update state covariance P
        # # I = np.eye(len(x))
        # # self.P = (I - K @ H) @ self.P

        # # Joseph Form (Symmetric & Numerically Stable):
        # I = np.eye(len(x))
        # I_KH = I - K @ H
        # self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # # Force exact numerical symmetry
        # self.P = 0.5 * (self.P + self.P.T)
        # # TODO ends


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
        if self.freeze_map:
            return   # true map already has every landmark, never add new ones
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
            if len(known_in_view) < 2:
                return  # need at least 2 already-known landmarks to trust robot pose for new ones


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
            self.P[-2,-2] = self.init_lm_cov**2
            self.P[-1,-1] = self.init_lm_cov**2


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

    def draw_slam_state(self, res = (320, 500), not_pause=True):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy*0
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
        axes_len = np.sqrt(np.maximum(0, e_vals)) * 2 * alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle