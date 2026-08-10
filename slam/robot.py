import numpy as np


class Robot:
    def __init__(self, baseline, scale, camera_matrix, dist_coeffs,ticks_per_meter=None):
        # State is a 3 x 1 vector containing information on x-pos, y-pos, and orientation, ie [x; y; theta]
        # Positive x-axis is the direction the robot is facing, positive y-axis 90 degree anticlockwise of positive x-axis
        # For orientation, it is in radian. Positive when turning anticlockwise (left)
        self.state = np.zeros((3,1))
        
        # Wheel parameters
        self.baseline = baseline  # The distance between the left and right wheels
        self.scale = scale  # The scaling factor converting M/s to m/s

        # Camera parameters
        self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
        self.dist_coeffs = dist_coeffs  # Distortion coefficients
        self.ticks_per_meter = ticks_per_meter  # set via calibration; None disables tick-based drive
    
    def drive(self, drive_measurement):
        # This is the "f" function in EKF
        # left_speed and right_speed are the speeds in M/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        dt = drive_measurement.dt

        if drive_measurement.delta_left_ticks is not None and drive_measurement.delta_right_ticks is not None:
            # Use measured encoder displacement instead of commanded_speed * dt
            dist_left = drive_measurement.delta_left_ticks / self.ticks_per_meter
            dist_right = drive_measurement.delta_right_ticks / self.ticks_per_meter
            linear_velocity = (dist_left + dist_right) / (2.0 * dt) if dt > 0 else 0.0
            angular_velocity = (dist_right - dist_left) / (self.baseline * dt) if dt > 0 else 0.0
        else:
            linear_velocity, angular_velocity = self.convert_wheel_speeds(
                drive_measurement.left_speed, drive_measurement.right_speed)

        # Apply the velocities
       
        if angular_velocity == 0:
            self.state[0] += np.cos(self.state[2]) * linear_velocity * dt
            self.state[1] += np.sin(self.state[2]) * linear_velocity * dt
        else:
            th = self.state[2]
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity

    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.scale
        right_speed_m = right_speed * self.scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.baseline
        
        return linear_velocity, angular_velocity
    
    def measure(self, markers, idx_list):
        # This is the "h" function in EKF
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]

        measurements_hat = []
        for idx in idx_list:
            lm_state = markers[:,idx:idx+1]
            lm_position = Rot_theta.T @ (lm_state - robot_xy) # transpose of rotation matrix is the inverse
            measurements_hat.append(lm_position)

        # Stack the measurements in a 2xn structure.
        measurements_hat = np.concatenate(measurements_hat, axis=1)
        return measurements_hat

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_measurement):
        # Compute the differential of drive w.r.t. the robot state
        DFx = np.zeros((3,3))
        DFx[0,0] = 1
        DFx[1,1] = 1
        DFx[2,2] = 1

        lin_vel, ang_vel = self.convert_wheel_speeds(drive_measurement.left_speed, drive_measurement.right_speed)

        dt = drive_measurement.dt
        th = self.state[2,0]

        if ang_vel == 0:
            # x += cos(th)*v*dt, y += sin(th)*v*dt
            DFx[0, 2] = -np.sin(th) * lin_vel * dt   # dx/dtheta
            DFx[1, 2] =  np.cos(th) * lin_vel * dt   # dy/dtheta
        else:
            th2 = th + dt * ang_vel
            # x += (v/w)*(sin(th2)-sin(th)), y += -(v/w)*(cos(th2)-cos(th))
            DFx[0, 2] = (lin_vel / ang_vel) * (np.cos(th2) - np.cos(th))   # dx/dtheta
            DFx[1, 2] = (lin_vel / ang_vel) * (np.sin(th2) - np.sin(th))   # dy/dtheta

        return DFx

    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3 + 2*markers.shape[1]

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.

            lmj_state = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_state - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_state - robot_xy)
            # lm xy DH
            DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T

            # print(DH[i:i+2,:])

        return DH
    
    def covariance_drive(self, drive_measurement):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
        # Jac1 = np.array([[self.scale/2, self.scale/2],
        #         [-self.scale/self.baseline, self.scale/self.baseline]])
        # distance = ticks / ticks_per_meter ; v = (d_left+d_right)/(2dt) ; ω = (d_right-d_left)/(baseline*dt)

        dt = drive_measurement.dt
        lin_vel, ang_vel = self.convert_wheel_speeds(drive_measurement.left_speed, drive_measurement.right_speed)
        th = self.state[2,0]
        th2 = th + dt*ang_vel

        use_ticks = (drive_measurement.delta_left_ticks is not None
                and drive_measurement.delta_right_ticks is not None
                and self.ticks_per_meter is not None
                and dt > 0)
        
        if use_ticks:
            # Derivative of lin_vel, ang_vel w.r.t. delta_left_ticks, delta_right_ticks
            Jac1 = np.array([
                [1.0/(2*dt*self.ticks_per_meter),  1.0/(2*dt*self.ticks_per_meter)],
                [-1.0/(dt*self.baseline*self.ticks_per_meter), 1.0/(dt*self.baseline*self.ticks_per_meter)]
            ])
            # Noise on encoder tick counts (tune empirically, e.g. ~1 tick^2 of quantization noise)
            tick_count_variance = 1.0
            cov_input = np.diag((tick_count_variance, tick_count_variance))
        else:
            # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed (original behaviour)
            Jac1 = np.array([[self.scale/2, self.scale/2],
                    [-self.scale/self.baseline, self.scale/self.baseline]])
            cov_input = np.diag((drive_measurement.left_cov, drive_measurement.right_cov))

            

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        Jac2 = np.zeros((3,2))

        if ang_vel == 0:
            # Straight-line case: x += cos(th)*v*dt, y += sin(th)*v*dt, theta += 0
            Jac2[0, 0] = np.cos(th) * dt      # dx/dv
            Jac2[1, 0] = np.sin(th) * dt      # dy/dv
            Jac2[0, 1] = 0.0                  # dx/dw
            Jac2[1, 1] = 0.0                  # dy/dw
            Jac2[2, 0] = 0.0                  # dtheta/dv
            Jac2[2, 1] = dt                   # dtheta/dw
        else:
            # Derivatives with respect to linear velocity v
            Jac2[0, 0] = (np.sin(th2) - np.sin(th)) / ang_vel
            Jac2[1, 0] = -(np.cos(th2) - np.cos(th)) / ang_vel
            Jac2[2, 0] = 0
            
            # Derivatives with respect to angular velocity w (using quotient rule)
            Jac2[0, 1] = (lin_vel * dt * np.cos(th2)) / ang_vel - (lin_vel * (np.sin(th2) - np.sin(th))) / (ang_vel**2)
            Jac2[1, 1] = (lin_vel * dt * np.sin(th2)) / ang_vel + (lin_vel * (np.cos(th2) - np.cos(th))) / (ang_vel**2)
            Jac2[2, 1] = dt


        # Derivative of x,y,theta w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1

        # Compute covariance
        # cov = np.diag((drive_measurement.left_cov, drive_measurement.right_cov))
        cov = Jac @ cov_input  @ Jac.T
        
        return cov