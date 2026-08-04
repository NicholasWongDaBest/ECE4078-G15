import cv2 
import time
import shutil
import argparse
import os, sys
import numpy as np
import pygame # python package for GUI
from botconnect import BotConnect # access the robot communication

# import SLAM components (M1)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import DriveMeasurement
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M2)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import ObjectDetector


class Operate:
    def __init__(self, args):
        
        # Initialise robot controller object
        self.botconnect = BotConnect(args.ip)
        self.command = {'wheel_speed':[0, 0], # left wheel speed, right wheel speed
                        'save_slam': False,
                        'run_obj_detector': False,                       
                        'save_obj_detector': False,
                        'save_image': False}
                        
        # TODO: Tune PID parameters here. If you don't want to use PID, set use_pid=0
        # self.botconnect.set_pid(use_pid=1, kp=0, ki=0, kd=0)

        # PID gains — now adjustable live via keyboard, not fixed at startup
        self.pid_gains = {'kp': 2.3, 'ki': 1.5, 'kd': 0.7}
        self.pid_step = 0.01
        self.botconnect.set_pid(use_pid=1, **self.pid_gains)

        # Create a folder "lab_output" that stores the results of the lab
        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)
    
        # Initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06) # size of the ARUCO markers (6cm)
        
        # Initialise CV detector
        if args.yolo_path == "":
            self.obj_detector = None
            self.cv_vis = cv2.imread('ui/8bit/detector_splash.png')
        else:
            self.obj_detector = ObjectDetector(args.yolo_path)
            self.cv_vis = np.ones((360,480,3))* 100
        
        # Create a folder to save raw camera images after pressing "i"
        self.raw_img_dir = 'raw_images/'
        if not os.path.exists(self.raw_img_dir):
            os.makedirs(self.raw_img_dir)
        else:
            # Delete the folder and create an empty one, i.e. every time operate.py is run, this folder will be empty.
            shutil.rmtree(self.raw_img_dir)
            os.makedirs(self.raw_img_dir)

        #Straight line sync correction 
        self.sync_kp = 0.0005   # tune this — start small and increase
        self.sync_ki = 0.0
        self.sync_error_integral = 0.0
        self.prev_left_count = 0
        self.prev_right_count = 0
        self.base_wheel_speed = [0.0, 0.0]  # intended speed, before correction

        # Other auxiliary objects/variables      
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([360,480,3], dtype=np.uint8)
        self.aruco_img = np.zeros([360,480,3], dtype=np.uint8)
        self.bg = pygame.image.load('ui/gui_mask.jpg')

    # update control parameters for ekf
    def control(self):
        dt = time.time() - self.control_clock
        drive_measurement = DriveMeasurement(self.botconnect.left_speed, self.botconnect.right_speed, dt)
        self.control_clock = time.time()
        return drive_measurement

    #real time update pid 
    def adjust_pid(self, param, delta):
        self.pid_gains[param] = max(0.0, self.pid_gains[param] + delta)
        success = self.botconnect.set_pid(use_pid=1, **self.pid_gains)
        if success:
            self.notification = (f"PID: kp={self.pid_gains['kp']:.3f} "
                                f"ki={self.pid_gains['ki']:.3f} "
                                f"kd={self.pid_gains['kd']:.3f}")
        else:
            self.notification = 'Failed to update PID on robot'

    # camera control
    def take_pic(self):
        self.img = self.botconnect.get_image() # self.img will be RGB
        
    # wheel and camera calibration for SLAM
    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # SLAM with ARUCO markers       
    def perform_slam(self, drive_measurement):
        sensor_measurement, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(sensor_measurement)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:
            self.ekf.predict(drive_measurement)
            self.ekf.add_landmarks(sensor_measurement)
            self.ekf.update(sensor_measurement)
            
    def save_result(self):
        # save slam map after pressing "s"
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False
        
        # save obj_detector result with the matching robot pose and detector labels
        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:   
                self.pred_fname = self.obj_detector.write_output(*self.obj_detector_output, self.lab_output_dir)
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_obj_detector'] = False
        
        # save raw images taken by the camera after pressing "i"
        if self.command['save_image']:
            image = self.botconnect.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            f_ = os.path.join(self.raw_img_dir, f'img_{self.image_id}.png')
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # using computer vision to detect objects
    def detect_object(self):
        if self.command['run_obj_detector'] and self.obj_detector is not None:
            bboxes, self.cv_vis = self.obj_detector.detect_single_image(self.img)
            self.command['run_obj_detector'] = False
            self.obj_detector_output = (self.cv_vis, self.ekf.robot.state.tolist(), bboxes) # three things to be saved
            unique_detected = len(set([box[0] for box in bboxes]))
            self.notification = f'{unique_detected} object type(s) detected'

    # paint the GUI            
    def draw(self, canvas):    
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for object detector
        detector_view = cv2.resize(self.cv_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240+2*v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='Robot Cam', position=(h_pad, v_pad))
        notification = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption, False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # Keyboard teleoperation
    # For pibot motion, set two numbers for the self.command['wheel_speed']. Eg self.command['wheel_speed'] = [0.6, 0.6]
    # These numbers specify how fast to power the left and right wheels
    # The numbers must be between -1 (full speed backward) and 1 (full speed forward). 0 means stop.
    # Study the code in botconnect.py to see the function to call after setting wheel speed
    def update_keyboard(self):
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                self.adjust_pid('kp', -self.pid_step)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                self.adjust_pid('kp', self.pid_step)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                self.adjust_pid('ki', -self.pid_step)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                self.adjust_pid('ki', self.pid_step)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                self.adjust_pid('kd', -self.pid_step)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.adjust_pid('kd', self.pid_step)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.base_wheel_speed = [0.35, 0.35]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.base_wheel_speed = [-0.35, -0.35]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.base_wheel_speed = [-0.35, 0.35]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.base_wheel_speed = [0.35, -0.35]
            elif event.type == pygame.KEYDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.base_wheel_speed = [0.0, 0.0]
            elif event.type == pygame.KEYUP and event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                self.base_wheel_speed = [0.0, 0.0]
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '>2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '>2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused' 
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['save_slam'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()          
            # run object/fruit detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_obj_detector'] = True
            # capture and save raw image
            elif event.type == pygame.KEYDOWN and event.key  == pygame.K_i:
                self.command['save_image'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

            self.botconnect.move_manual(self.command['wheel_speed'])

        if self.quit:
            pygame.quit()
            sys.exit()

    def correct_straight_drive(self):
        left, right = self.botconnect.get_encoder_counts()
        delta_left = left - self.prev_left_count
        delta_right = right - self.prev_right_count
        self.prev_left_count, self.prev_right_count = left, right

        base_l, base_r = self.base_wheel_speed
        if base_l == base_r and base_l != 0:
            # Only correct when driving straight (turns should curve on purpose)
            error = delta_left - delta_right  # >0 means left is outrunning right
            self.sync_error_integral += error
            correction = self.sync_kp * error + self.sync_ki * self.sync_error_integral
            adjusted = [base_l - correction, base_r + correction]
        else:
            self.sync_error_integral = 0.0
            adjusted = [base_l, base_r]

        self.command['wheel_speed'] = adjusted
        self.botconnect.move_manual(adjusted)
        print(f"L:{left} R:{right}  dL:{delta_left} dR:{delta_right}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost') # you can hardcode ip here, but it may change from time to time.
    parser.add_argument("--calib_dir", type=str, default="calibration/param/") # calibration directory
    parser.add_argument("--yolo_path", default='cv/model/yolov8_model.pt') # directory for your trained AI model
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab')
    pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('ui/loading.png')
    pibot_animate = [pygame.image.load('ui/8bit/pibot1.png'),
                     pygame.image.load('ui/8bit/pibot2.png'),
                     pygame.image.load('ui/8bit/pibot3.png'),
                    pygame.image.load('ui/8bit/pibot4.png'),
                     pygame.image.load('ui/8bit/pibot5.png')]
    pygame.display.update()

    start = False
    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2
    
    width, height = 900, 660
    canvas = pygame.display.set_mode((width, height))
    operate = Operate(args)
    while start:
        operate.update_keyboard()
        operate.correct_straight_drive()
        operate.take_pic()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()